#!/usr/bin/env python3
"""Parallel runner for `exp/orbita-adaptive-spectral` (DEV 0xA400..0xA40F, CONFIRM 0xB400..0xB40F)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
ORBITA_DIR = REPO_ROOT / "orbita"
BIN_PATH = ORBITA_DIR / "target" / "release" / "orbita"
DOCS_DIR = REPO_ROOT / "docs" / "orbita_adaptive_spectral_v1"
RAW_DIR = DOCS_DIR / "raw"

DEV_SEEDS = [0xA400 + i for i in range(16)]
CONFIRM_SEEDS = [0xB400 + i for i in range(16)]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(65536), b""):
            h.update(ch)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()


def verify_frozen() -> None:
    fp = DOCS_DIR / "FROZEN.json"
    if not fp.exists():
        raise RuntimeError(f"CONFIRM requires {fp} created prior to execution!")
    frozen = json.loads(fp.read_text(encoding="utf-8"))
    bad = []
    for rel, expected in frozen.get("files", {}).items():
        p = REPO_ROOT / rel
        if not p.exists() or sha256_file(p) != expected:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"FROZEN.json SHA-256 mismatch: {bad}")


def run_one(phase: str, seed: int) -> tuple[int, float, str]:
    sub = "dev" if phase == "DEV" else "confirm"
    wpath = REPO_ROOT / "artifacts" / "orbita" / sub / f"weights_0x{seed:X}.json"
    mpath = REPO_ROOT / "artifacts" / "orbita" / sub / f"meta_0x{seed:X}.json"
    out_dir = RAW_DIR / sub / f"seed_0x{seed:X}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN_PATH),
        "asc-run",
        "--seed",
        f"0x{seed:X}",
        "--phase",
        phase,
        "--weights",
        str(wpath),
        "--meta",
        str(mpath),
        "--out",
        str(out_dir),
    ]
    if seed in (0xA400, 0xB400):
        cmd.append("--export-traces")
    t0 = time.monotonic()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)
    return seed, time.monotonic() - t0, res.stdout.strip()


def seed_complete(phase_arg: str, seed: int) -> bool:
    sub = "dev" if phase_arg == "DEV" else "confirm"
    sdir = RAW_DIR / sub / f"seed_0x{seed:X}"
    return all((sdir / f"metrics_o{k}.csv").exists() for k in (1, 2, 3, 4))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["DEV", "CONFIRM", "SMOKE"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="Skip seeds whose metrics_o1..o4.csv already exist")
    ap.add_argument("--limit", type=int, default=0, help="Run at most N pending seeds (0 = all)")
    args = ap.parse_args()

    if not BIN_PATH.exists():
        raise RuntimeError(f"Missing release binary {BIN_PATH}")

    frozen_verified = False
    if args.phase == "CONFIRM":
        verify_frozen()
        frozen_verified = True
        seeds = CONFIRM_SEEDS
    elif args.phase == "DEV":
        seeds = DEV_SEEDS
    else:
        seeds = [0xA400]

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    env_info = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "git_head": git_head(),
        "binary_sha256": sha256_file(BIN_PATH),
    }
    (DOCS_DIR / "environment.json").write_text(json.dumps(env_info, indent=2) + "\n", encoding="utf-8")

    t_all = time.monotonic()
    run_path = DOCS_DIR / "run.json"
    meta = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    per_seed: dict[str, float] = {}
    if args.resume and args.phase in meta:
        per_seed.update(meta[args.phase].get("per_seed_seconds", {}))

    phase_arg = "DEV" if args.phase == "SMOKE" else args.phase
    pending = [s for s in seeds if not (args.resume and seed_complete(phase_arg, s))]
    if args.limit > 0:
        pending = pending[: args.limit]
    if args.resume and len(pending) < len(seeds):
        print(f"[{args.phase}] resuming: {len(seeds) - len(pending)}/{len(seeds)} already done, running {len(pending)} pending", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, phase_arg, s): s for s in pending}
        for fut in as_completed(futs):
            s, dt_s, line = fut.result()
            per_seed[f"0x{s:X}"] = round(dt_s, 4)
            print(f"[{args.phase}] {line} ({dt_s:.2f}s)", flush=True)

    total_s = time.monotonic() - t_all
    if args.phase in ("DEV", "CONFIRM"):
        meta[args.phase] = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_head": git_head(),
            "binary_sha256": sha256_file(BIN_PATH),
            "n_seeds": len(seeds),
            "completed_seeds": sum(1 for s in seeds if seed_complete(phase_arg, s)),
            "seeds": [f"0x{s:X}" for s in seeds],
            "workers": args.workers,
            "total_seconds_last_batch": round(total_s, 4),
            "per_seed_seconds": dict(sorted(per_seed.items())),
            "frozen_verified": frozen_verified,
        }
        run_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[{args.phase}] finished batch of {len(pending)} seeds in {total_s:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
