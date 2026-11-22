#!/usr/bin/env python3
"""
Runs AURA-v3-Genesis across all 16 DEV seeds (0xA400..0xA40F) and 16 CONFIRM seeds (0xB400..0xB40F)
in the exact canonical `orbita metafe3s-run` harness under 100% equal training conditions
(n_tr=128, n_va=96, HPF=64, syn_ood=48, RS-A=48, RS-B=24, E33) and merges the `AURA-v3-Genesis`
arm into `artifacts/metafe3s/{dev,confirm}/metrics_0x*.json` while preserving all existing arms bit-for-bit.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "orbita" / "target" / "release" / "orbita"
TMP_OUT = Path("/tmp/aura_v3_orbita_run")


def run_seed(phase: str, seed_hex: str) -> tuple[str, float, float, float]:
    w_path = ROOT / "artifacts" / "orbita" / phase / f"weights_{seed_hex}.json"
    cmd = [
        "nice", "-n", "10",
        str(BIN),
        "metafe3s-run",
        "--phase", phase,
        "--seed", seed_hex,
        "--weights", str(w_path),
        "--out", str(TMP_OUT),
        "--arms", "AURA-v3-Genesis",
        "--force",
    ]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed on {phase} {seed_hex}:\n{res.stderr}\n{res.stdout}")

    new_json_path = TMP_OUT / phase / f"metrics_{seed_hex}.json"
    new_data = json.loads(new_json_path.read_text())
    arm_data = new_data["arms"]["AURA-v3-Genesis"]

    target_json_path = ROOT / "artifacts" / "metafe3s" / phase / f"metrics_{seed_hex}.json"
    target_data = json.loads(target_json_path.read_text())
    target_data["arms"]["AURA-v3-Genesis"] = arm_data
    target_json_path.write_text(json.dumps(target_data, indent=2) + "\n")

    med = arm_data["held"]["new_med_log"]
    acc = arm_data["held"]["new_acc16"] * 100.0
    wall = arm_data["wall_s"]
    return (seed_hex, med, acc, wall)


def run_phase(phase: str, base_seed: int):
    seeds = [f"0x{base_seed + i:X}" for i in range(16)]
    print(f"=== Running AURA-v3-Genesis on {phase.upper()} ({seeds[0]}..{seeds[-1]}) ===")
    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(run_seed, phase, s): s for s in seeds}
        for fut in concurrent.futures.as_completed(futs):
            s, med, acc, wall = fut.result()
            results.append((s, med, acc, wall))
            print(f"  {phase:<7} {s}  new_med={med:+.4f}  new_acc={acc:.2f}%  wall={wall:.2f}s", flush=True)
    results.sort(key=lambda x: int(x[0], 16))
    mean_med = sum(x[1] for x in results) / len(results)
    mean_acc = sum(x[2] for x in results) / len(results)
    mean_wall = sum(x[3] for x in results) / len(results)
    print(f"==> {phase.upper()} DONE in {time.time() - t0:.1f}s | mean new_med={mean_med:+.4f} | mean new_acc={mean_acc:.2f}% | mean wall_s={mean_wall:.2f}s\n", flush=True)


if __name__ == "__main__":
    phases = sys.argv[1:] if len(sys.argv) > 1 else ["dev", "confirm"]
    for ph in phases:
        if ph == "dev":
            run_phase("dev", 0xA400)
        elif ph == "confirm":
            run_phase("confirm", 0xB400)
