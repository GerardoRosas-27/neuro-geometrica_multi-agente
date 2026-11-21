#!/usr/bin/env python3
"""Orquestador determinista de procesos Rust para AURA + CorticalFabric v1.
Python NO realiza entrenamiento ni inferencia; solo lanza binarios Rust y verifica SHA-256."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "aura-engine"
DOC = ROOT / "docs/aura_v1"


def compute_sources_fingerprint():
    files = sorted((ENGINE / "crates").glob("*/src/*.rs"))
    per_file = {}
    h_all = hashlib.sha256()
    for f in files:
        rel = str(f.relative_to(ROOT))
        data = f.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        per_file[rel] = digest
        h_all.update(rel.encode("utf-8"))
        h_all.update(b"\0")
        h_all.update(data)
        h_all.update(b"\0")
    return h_all.hexdigest(), per_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["DEV", "CONFIRM"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output", default="docs/aura_v1/raw")
    p.add_argument("--force", action="store_true", help="Permitido solo en DEV durante calibración")
    a = p.parse_args()

    assert 1 <= a.workers <= 4, "workers must be in 1..=4"
    out = ROOT / a.output / a.phase
    out.mkdir(parents=True, exist_ok=True)

    existing = list(out.glob("*/metrics_e1.csv"))
    if existing:
        if a.phase == "CONFIRM" or not a.force:
            raise SystemExit(f"Refusing to overwrite completed {a.phase} experiment in {out}")

    combined_sha, per_file_sha = compute_sources_fingerprint()
    freeze = DOC / "FROZEN.json"
    if a.phase == "CONFIRM":
        assert freeze.exists(), "Freeze implementation (FROZEN.json) before CONFIRM"
        frozen_data = json.loads(freeze.read_text())
        assert (
            frozen_data["sources_sha256"] == combined_sha
        ), f"Source SHA mismatch vs FROZEN.json! frozen={frozen_data['sources_sha256']} current={combined_sha}"

    bin_path = ENGINE / "target/release/aura-exp"
    assert bin_path.exists(), f"Build release binary first: {bin_path}"

    start = time.time()
    base = 0xA400 if a.phase == "DEV" else 0xB400
    meta = {
        "phase": a.phase,
        "workers": a.workers,
        "seed_start": f"0x{base:04X}",
        "seed_end": f"0x{base + 15:04X}",
        "sources_sha256": combined_sha,
        "per_file_sha256": per_file_sha,
        "start_epoch": start,
        "command": sys.argv,
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "system": os.uname().sysname + " " + os.uname().machine,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    def job(seed):
        d = out / f"{seed:04X}"
        d.mkdir(exist_ok=True)
        with (d / "stdout.log").open("w") as log:
            subprocess.run(
                [str(bin_path), "run", f"0x{seed:04X}", a.phase, str(d)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        return seed

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for seed in ex.map(job, range(base, base + 16)):
            print(f"{a.phase} 0x{seed:04X} completed elapsed={time.time() - start:.2f}s", flush=True)
            assert time.time() - start < 2700, "Global 45-minute bound exceeded"

    meta["elapsed_seconds"] = round(time.time() - start, 4)
    meta["completed"] = 16
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"phase": a.phase, "completed": 16, "elapsed_seconds": meta["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
