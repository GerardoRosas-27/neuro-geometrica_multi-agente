#!/usr/bin/env python3
"""Runner for exp/wave-collision-autocal (DEV 0xF100..0xF10F / CONFIRM 0xF200..0xF20F)."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "wave_collision"
BIN = CRATE / "target" / "release" / "wave_collision"
DOCS = ROOT / "docs" / "wave_collision_v1"

DEV_SEEDS = [0xF100 + i for i in range(16)]
CONFIRM_SEEDS = [0xF200 + i for i in range(16)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen() -> None:
    frozen_path = DOCS / "FROZEN.json"
    if not frozen_path.exists():
        raise RuntimeError(f"Missing {frozen_path}; run freeze before CONFIRM!")
    data = json.loads(frozen_path.read_text())
    for rel, expected in data["files"].items():
        p = ROOT / rel
        if not p.exists():
            raise RuntimeError(f"Frozen file missing: {rel}")
        actual = sha256_file(p)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch on {rel}: expected {expected}, got {actual}")
    print(f"[FROZEN OK] Verified {len(data['files'])} SHA-256 hashes before CONFIRM.")


def run_one_seed(split: str, seed: int, trace: bool, resume: bool) -> tuple[int, float, bool]:
    out_dir = DOCS / "raw" / split / f"seed_0x{seed:X}"
    req = ["metrics_t1.csv", "metrics_t2.csv", "metrics_t3.csv", "metrics_t4.csv"]
    if trace:
        req.append("trace_wave_collision.csv")
    if resume and all((out_dir / f).exists() and (out_dir / f).stat().st_size > 0 for f in req):
        return (seed, 0.0, True)

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "run-seed", "--seed", f"0x{seed:X}", "--out", str(out_dir)]
    if trace:
        cmd.append("--trace")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    dt = time.perf_counter() - t0
    return (seed, dt, False)


def write_environment() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    rustc_v = subprocess.check_output(["rustc", "--version"], text=True).strip()
    cargo_v = subprocess.check_output(["cargo", "--version"], text=True).strip()
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    git_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=str(ROOT), text=True
    ).strip()
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "rustc": rustc_v,
        "cargo": cargo_v,
        "git_branch": git_branch,
        "git_head": git_head,
    }
    (DOCS / "environment.json").write_text(json.dumps(env, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "confirm"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=16)
    args = ap.parse_args()

    if not BIN.exists():
        raise RuntimeError(f"Release binary missing at {BIN}")
    if args.split == "confirm":
        verify_frozen()

    seeds = (DEV_SEEDS if args.split == "dev" else CONFIRM_SEEDS)[: args.limit]
    trace_seed = seeds[0]
    t_all = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(run_one_seed, args.split, s, s == trace_seed, args.resume) for s in seeds
        ]
        for fut in concurrent.futures.as_completed(futs):
            s, dt, skipped = fut.result()
            tag = "SKIP" if skipped else f"{dt:.2f}s"
            print(f"[{args.split.upper()}] seed 0x{s:X}: {tag}", flush=True)

    total_s = time.perf_counter() - t_all
    write_environment()
    run_path = DOCS / "run.json"
    run_meta = json.loads(run_path.read_text()) if run_path.exists() else {}
    run_meta[args.split] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [f"0x{s:X}" for s in seeds],
        "n_seeds": len(seeds),
        "wall_seconds": round(total_s, 3),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip(),
    }
    run_path.write_text(json.dumps(run_meta, indent=2) + "\n")
    print(f"Done {args.split.upper()} ({len(seeds)} seeds) in {total_s:.2f}s.")


if __name__ == "__main__":
    main()
