#!/usr/bin/env python3
"""Parallel runner for Prueba B (nervous_wave_b) with strict FROZEN.json SHA-256 check on CONFIRM."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "nervous_wave_b" / "target" / "release" / "nervous_wave_b"
OUT_ROOT = ROOT / "docs" / "nervous_wave_b_v1" / "runs"
FROZEN_PATH = ROOT / "docs" / "nervous_wave_b_v1" / "FROZEN.json"

DEV_SEEDS = [0xB100 + i for i in range(16)]
CONFIRM_SEEDS = [0xB200 + i for i in range(16)]
EXPECTED_FILES = ["b1_physics.csv", "b2_inference.csv", "b3_resilience.csv", "b4_transfer.csv", "seed_meta.json"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen() -> None:
    if not FROZEN_PATH.exists():
        print(f"ERROR: {FROZEN_PATH} not found. Run nwb_freeze_audit.py freeze before CONFIRM.", file=sys.stderr)
        sys.exit(1)
    frozen = json.loads(FROZEN_PATH.read_text())
    mismatches = []
    for rel, expected in frozen["files"].items():
        p = ROOT / rel
        if not p.exists():
            mismatches.append(f"MISSING {rel}")
        else:
            actual = sha256_file(p)
            if actual != expected:
                mismatches.append(f"MISMATCH {rel}: expected {expected[:12]} got {actual[:12]}")
    if mismatches:
        print("ERROR: FROZEN.json SHA-256 verification FAILED:", file=sys.stderr)
        for m in mismatches:
            print("  ", m, file=sys.stderr)
        sys.exit(1)
    print(f"[FROZEN OK] Verified {len(frozen['files'])} files against FROZEN.json")


def is_complete(seed_dir: Path) -> bool:
    return all((seed_dir / fn).exists() and (seed_dir / fn).stat().st_size > 0 for fn in EXPECTED_FILES)


def run_one(seed: int, split: str) -> tuple[int, int, str]:
    seed_dir = OUT_ROOT / split / f"seed_0x{seed:X}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "--seed", f"0x{seed:X}", "--split", split, "--out-dir", str(seed_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return seed, res.returncode, res.stderr
    meta = json.loads((seed_dir / "seed_meta.json").read_text())
    return seed, 0, f"{meta['elapsed_ms']} ms (frozen_ok={meta['frozen_all_ok']})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "confirm"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=16)
    args = ap.parse_args()

    if not BIN.exists():
        print(f"ERROR: Release binary not found at {BIN}. Build with cargo build --release first.", file=sys.stderr)
        sys.exit(1)

    if args.split == "confirm":
        verify_frozen()

    seeds = (DEV_SEEDS if args.split == "dev" else CONFIRM_SEEDS)[: args.limit]
    todo = []
    for s in seeds:
        sdir = OUT_ROOT / args.split / f"seed_0x{s:X}"
        if args.resume and is_complete(sdir):
            print(f"[SKIP] 0x{s:X} already complete")
        else:
            todo.append(s)

    if not todo:
        print(f"All {len(seeds)} {args.split} seeds already complete.")
        return

    print(f"Running {len(todo)}/{len(seeds)} {args.split} seeds with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, s, args.split): s for s in todo}
        for fut in as_completed(futs):
            seed, rc, msg = fut.result()
            if rc != 0:
                print(f"[FAIL] 0x{seed:X} rc={rc}: {msg}", file=sys.stderr)
                sys.exit(rc)
            print(f"[DONE] 0x{seed:X}: {msg}", flush=True)


if __name__ == "__main__":
    main()
