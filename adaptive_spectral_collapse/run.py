#!/usr/bin/env python3
"""Runner for isolated experiment `exp/adaptive-spectral-collapse`.

Executes 16-seed DEV (0xC100..=0xC10F) or frozen 16-seed CONFIRM (0xD100..=0xD10F),
enforcing SHA-256 verification against FROZEN.json before any CONFIRM execution.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DOCS_DIR = REPO_ROOT / "docs" / "adaptive_spectral_collapse_v1"
RAW_DIR = DOCS_DIR / "raw"
BIN_PATH = ROOT / "target" / "release" / "adaptive-spectral-collapse-v1"

DEV_SEEDS = [0xC100 + i for i in range(16)]
CONFIRM_SEEDS = [0xD100 + i for i in range(16)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_output(args: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        out = subprocess.check_output(args, cwd=str(cwd), stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception as exc:
        return f"unavailable ({exc})"


def verify_frozen() -> dict:
    frozen_path = DOCS_DIR / "FROZEN.json"
    if not frozen_path.exists():
        raise RuntimeError(
            f"Missing {frozen_path}. Run `python3 freeze_and_verify.py freeze` and commit before CONFIRM."
        )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    mismatches = []
    for rel, expected_hash in frozen.get("files", {}).items():
        p = REPO_ROOT / rel
        if not p.exists():
            mismatches.append(f"MISSING {rel}")
            continue
        actual = sha256_file(p)
        if actual != expected_hash:
            mismatches.append(f"MISMATCH {rel}: expected {expected_hash[:12]} got {actual[:12]}")
    if mismatches:
        raise RuntimeError(
            "FROZEN.json SHA-256 verification FAILED before CONFIRM:\n  "
            + "\n  ".join(mismatches)
        )
    return frozen


def write_environment() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    env_info = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "rustc_version": cmd_output(["rustc", "--version"], cwd=ROOT),
        "cargo_version": cmd_output(["cargo", "--version"], cwd=ROOT),
        "git_branch": cmd_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_head": cmd_output(["git", "rev-parse", "HEAD"]),
    }
    (DOCS_DIR / "environment.json").write_text(
        json.dumps(env_info, indent=2) + "\n", encoding="utf-8"
    )


def run_one_seed(seed: int, phase: str, out_dir: Path, export_traces: bool) -> tuple[int, float]:
    t0 = time.monotonic()
    cmd = [str(BIN_PATH), "run", f"0x{seed:X}", phase, str(out_dir)]
    if export_traces:
        cmd.append("--export-traces")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return seed, time.monotonic() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DEV or CONFIRM seeds for adaptive_spectral_collapse")
    parser.add_argument("phase", choices=["DEV", "CONFIRM"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    if not args.skip_build:
        if args.phase == "CONFIRM":
            raise RuntimeError("Do not rebuild during CONFIRM; pass --skip-build to use frozen binary.")
        subprocess.run(["cargo", "build", "--release"], check=True, cwd=str(ROOT))

    if not BIN_PATH.exists():
        raise RuntimeError(f"Release binary not found at {BIN_PATH}")

    frozen_meta = None
    if args.phase == "CONFIRM":
        frozen_meta = verify_frozen()

    write_environment()

    seeds = DEV_SEEDS if args.phase == "DEV" else CONFIRM_SEEDS
    phase_dir = RAW_DIR / args.phase.lower()
    phase_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    timings: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {}
        for idx, seed in enumerate(seeds):
            out_dir = phase_dir / f"seed_0x{seed:X}"
            fut = pool.submit(run_one_seed, seed, args.phase, out_dir, idx == 0)
            futures[fut] = seed
        for fut in as_completed(futures):
            seed, elapsed = fut.result()
            timings[f"0x{seed:X}"] = round(elapsed, 4)
            print(f"[{args.phase}] seed 0x{seed:X} done in {elapsed:.2f}s", flush=True)

    total_s = round(time.monotonic() - t_start, 4)
    print(f"[{args.phase}] completed {len(seeds)} seeds in {total_s:.2f}s", flush=True)

    run_json_path = DOCS_DIR / "run.json"
    record = {}
    if run_json_path.exists():
        record = json.loads(run_json_path.read_text(encoding="utf-8"))

    record[args.phase] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": cmd_output(["git", "rev-parse", "HEAD"]),
        "binary_sha256": sha256_file(BIN_PATH),
        "n_seeds": len(seeds),
        "seeds": [f"0x{s:X}" for s in seeds],
        "workers": args.workers,
        "total_seconds": total_s,
        "per_seed_seconds": dict(sorted(timings.items())),
        "frozen_verified": frozen_meta is not None,
    }
    run_json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
