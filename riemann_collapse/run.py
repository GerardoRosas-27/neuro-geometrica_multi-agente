#!/usr/bin/env python3
"""Orchestrate isolated Rust jobs for riemann-collapse-v1. Python performs NO optimization/training."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

CRATE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CRATE_DIR.parent
DOCS_DIR = REPO_ROOT / "docs" / "riemann_collapse_v1"


def compute_sources_sha256() -> tuple[str, dict[str, str]]:
    files = sorted((CRATE_DIR / "src").glob("*.rs")) + [CRATE_DIR / "Cargo.toml"]
    per_file: dict[str, str] = {}
    h = hashlib.sha256()
    for p in files:
        rel = str(p.relative_to(REPO_ROOT))
        data = p.read_bytes()
        per_file[rel] = hashlib.sha256(data).hexdigest()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
    return h.hexdigest(), per_file


def write_environment_json():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    env = {
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    (DOCS_DIR / "environment.json").write_text(json.dumps(env, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["DEV", "CONFIRM"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--skip-build", action="store_true")
    a = p.parse_args()
    assert 1 <= a.workers <= 4

    out = DOCS_DIR / "raw" / a.phase
    out.mkdir(parents=True, exist_ok=True)
    assert not list(out.glob("*/metrics_c1.csv")), f"Refusing to overwrite completed {a.phase} run"

    sources_sha, per_file = compute_sources_sha256()
    freeze_path = DOCS_DIR / "FROZEN.json"
    if a.phase == "CONFIRM":
        assert freeze_path.exists(), "Must freeze FROZEN.json before CONFIRM"
        frozen = json.loads(freeze_path.read_text())
        assert frozen["sources_sha256"] == sources_sha, (
            f"SHA-256 mismatch vs FROZEN.json: {sources_sha} != {frozen['sources_sha256']}"
        )

    if not a.skip_build:
        subprocess.run(["cargo", "build", "--release"], cwd=CRATE_DIR, check=True)

    bin_path = CRATE_DIR / "target" / "release" / "riemann-collapse-v1"
    assert bin_path.exists(), f"Missing binary {bin_path}"
    write_environment_json()

    base = 0xA800 if a.phase == "DEV" else 0xB800
    start = time.time()
    meta = {
        "phase": a.phase,
        "workers": a.workers,
        "seed_start": f"0x{base:04X}",
        "seed_end": f"0x{base + 15:04X}",
        "sources_sha256": sources_sha,
        "file_count": len(per_file),
        "start_epoch": start,
        "command": sys.argv,
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "system": f"{os.uname().sysname} {os.uname().machine}",
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    def job(seed: int) -> int:
        d = out / f"0x{seed:04X}"
        d.mkdir(parents=True, exist_ok=True)
        cmd = [str(bin_path), "run", f"0x{seed:04X}", a.phase, str(d)]
        if seed == base:
            cmd.append("--export-traces")
        with (d / "stdout.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        return seed

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for seed in ex.map(job, range(base, base + 16)):
            print(f"[{a.phase}] 0x{seed:04X} done elapsed={time.time() - start:.2f}s", flush=True)
            assert time.time() - start < 1200, "Exceeded 20-minute wall-time bound"

    meta["elapsed_seconds"] = round(time.time() - start, 3)
    meta["completed"] = 16
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[{a.phase}] ALL 16 SEEDS COMPLETE in {meta['elapsed_seconds']}s (SHA={sources_sha[:12]})")


if __name__ == "__main__":
    main()
