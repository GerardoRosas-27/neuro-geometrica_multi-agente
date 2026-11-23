#!/usr/bin/env python3
"""Deterministic orchestrator for AURA v2 DEV/CONFIRM sweeps with SHA-256 freeze verification."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = Path(__file__).resolve().parent
DOCS_DIR = REPO_ROOT / "docs" / "aura_v2"

PHASE_SEEDS = {
    "DEV": [0xA500 + i for i in range(16)],
    "CONFIRM": [0xB500 + i for i in range(16)],
}


def compute_sources_sha256() -> tuple[str, dict[str, str]]:
    files = sorted((ENGINE_DIR / "crates").rglob("*.rs"))
    per_file = {}
    master = hashlib.sha256()
    for f in files:
        rel = f.relative_to(ENGINE_DIR).as_posix()
        data = f.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        per_file[rel] = h
        master.update(rel.encode("utf-8"))
        master.update(b"\0")
        master.update(data)
        master.update(b"\0")
    return master.hexdigest(), per_file


def gather_environment() -> dict:
    rustc = subprocess.check_output(["rustc", "--version"], text=True).strip()
    cargo = subprocess.check_output(["cargo", "--version"], text=True).strip()
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    git_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    cpu_model = ""
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "rustc": rustc,
        "cargo": cargo,
        "git_branch": git_branch,
        "git_head": git_head,
    }


def run_seed(bin_path: Path, seed: int, out_dir: Path, emit_samples: bool) -> tuple[int, float]:
    seed_dir = out_dir / f"0x{seed:04X}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nice", "-n", "10",
        str(bin_path),
        "all",
        f"0x{seed:04X}",
        str(seed_dir),
        "--repo",
        str(REPO_ROOT),
    ]
    if emit_samples:
        cmd.append("--samples")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Seed 0x{seed:04X} failed:\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}")
    return seed, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["DEV", "CONFIRM"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    out_root = args.output if args.output else (DOCS_DIR / "raw")
    phase_dir = out_root / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)

    sha, per_file = compute_sources_sha256()
    if args.phase == "CONFIRM":
        frozen_path = DOCS_DIR / "FROZEN.json"
        if not frozen_path.exists():
            sys.exit("ERROR: docs/aura_v2/FROZEN.json missing! Freeze DEV before running CONFIRM.")
        frozen = json.loads(frozen_path.read_text())
        if frozen.get("sources_sha256") != sha:
            sys.exit(
                f"ERROR: Source SHA-256 mismatch!\n"
                f"  FROZEN.json: {frozen.get('sources_sha256')}\n"
                f"  Current    : {sha}"
            )
        print(f"[CONFIRM] Verified FROZEN.json SHA-256 match: {sha}")

    if not args.skip_build:
        print("[BUILD] Building aura-v2 release binary...")
        subprocess.run(
            ["nice", "-n", "10", "cargo", "build", "--release", "--manifest-path", str(ENGINE_DIR / "Cargo.toml"), "-j", str(args.workers)],
            cwd=REPO_ROOT,
            check=True,
        )

    bin_path = ENGINE_DIR / "target" / "release" / "aura-v2"
    if not bin_path.exists():
        sys.exit(f"Binary not found: {bin_path}")

    seeds = PHASE_SEEDS[args.phase]
    print(f"[{args.phase}] Running {len(seeds)} seeds with {args.workers} workers...")
    t_all = time.perf_counter()
    per_seed_sec = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(run_seed, bin_path, s, phase_dir, idx == 0)
            for idx, s in enumerate(seeds)
        ]
        for fut in concurrent.futures.as_completed(futs):
            s, dt = fut.result()
            per_seed_sec[f"0x{s:04X}"] = round(dt, 3)
            print(f"  done 0x{s:04X} ({dt:.2f}s)")

    elapsed = time.perf_counter() - t_all
    meta = {
        "phase": args.phase,
        "seeds": [f"0x{s:04X}" for s in seeds],
        "sources_sha256": sha,
        "file_count": len(per_file),
        "elapsed_sec": round(elapsed, 3),
        "per_seed_sec": dict(sorted(per_seed_sec.items())),
        "environment": gather_environment(),
    }
    (phase_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.output is None:
        (DOCS_DIR / "environment.json").write_text(json.dumps(meta["environment"], indent=2) + "\n")
    print(f"[{args.phase}] Completed {len(seeds)} seeds in {elapsed:.2f}s -> {phase_dir}")


if __name__ == "__main__":
    main()
