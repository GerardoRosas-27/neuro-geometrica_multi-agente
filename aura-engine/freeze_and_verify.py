#!/usr/bin/env python3
"""Congelación de configuración pre-CONFIRM (FROZEN.json) y verificación de replay bit a bit."""

import argparse
import csv
import hashlib
import json
import pathlib
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "aura-engine"
DOC = ROOT / "docs/aura_v1"

TIME_COLS = {"eval_ms", "ticks_per_sec"}


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


def strip_time_csv(path: pathlib.Path):
    with path.open() as f:
        r = csv.DictReader(f)
        cols = [c for c in r.fieldnames if c not in TIME_COLS]
        return [{c: row[c] for c in cols} for row in r]


def do_freeze(prereg_commit: str):
    combined, per_file = compute_sources_fingerprint()
    frozen = {
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg_commit": prereg_commit,
        "sources_sha256": combined,
        "per_file_sha256": per_file,
        "hyperparameters": {
            "latent_dim": 256,
            "ensemble_heads": 4,
            "key_dim": 64,
            "ema_shift": 8,
            "hnsw_m": 16,
            "hnsw_ef": 32,
            "e1_patcher": {"min_len": 1, "max_len": 6, "entropy_thresh_q8": 420},
            "e2_metacog": {"doubt_threshold": 1900, "abstain_threshold": 2150, "ne_deliberate_thresh": 5, "epi_weight_q2": 2},
            "e3_sleep": {"cycles": 2400, "replay_ratio_pct": 50, "hysteresis_hi": 96, "hysteresis_lo": 24, "shadow_cap": 512, "margin": 192, "step_delta": 12},
            "e4_fabric": {"n_neurons": 256, "bands": [17, 9, 4], "minicolumn_size": 8, "max_delay": 16, "ne_fast_shift": 3, "ne_slow_shift": 6},
        },
        "seeds": {
            "DEV": "0xA400..=0xA40F (16 seeds)",
            "CONFIRM": "0xB400..=0xB40F (16 fresh seeds)",
        },
    }
    (DOC / "FROZEN.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"Wrote {DOC / 'FROZEN.json'} with sources_sha256={combined}")


def do_replay():
    orig_dir = DOC / "raw/CONFIRM/B400"
    assert orig_dir.exists(), f"Missing CONFIRM B400 at {orig_dir}"
    bin_path = ENGINE / "target/release/aura-exp"
    with tempfile.TemporaryDirectory(prefix="aura_replay_") as tmp:
        replay_dir = pathlib.Path(tmp)
        subprocess.run(
            [str(bin_path), "run", "0xB400", "CONFIRM", str(replay_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        checks = {}
        all_ok = True
        for fname in ["metrics_e1.csv", "metrics_e2.csv", "metrics_e3.csv", "metrics_e4.csv"]:
            a = strip_time_csv(orig_dir / fname)
            b = strip_time_csv(replay_dir / fname)
            ok = a == b
            checks[fname] = {"identical_excluding_timing": ok, "rows": len(a)}
            all_ok = all_ok and ok

        b5_a = json.loads((orig_dir / "bench_e5.json").read_text())
        b5_b = json.loads((replay_dir / "bench_e5.json").read_text())
        det_keys = ["seed", "phase", "max_abs_diff_zero_mul", "memory_ratio_f32_over_packed", "dim1024_packed_bytes", "dim1024_f32_bytes"]
        ok_b5 = all(b5_a[k] == b5_b[k] for k in det_keys)
        checks["bench_e5.json"] = {"identical_deterministic_fields": ok_b5}
        all_ok = all_ok and ok_b5

        report = {
            "seed": "0xB400",
            "phase": "CONFIRM",
            "all_identical_excluding_timing": all_ok,
            "excluded_timing_columns": sorted(TIME_COLS),
            "checks": checks,
        }
        (DOC / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        assert all_ok, "Bit-exact replay FAILED!"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--prereg-commit", default="fa31fd1e44777822704fa9e11ac22c9b05b2078b")
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    if a.freeze:
        do_freeze(a.prereg_commit)
    if a.replay:
        do_replay()


if __name__ == "__main__":
    main()
