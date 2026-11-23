#!/usr/bin/env python3
"""Freeze SHA-256 before CONFIRM, verify bit-exact replay on 0xB600, and audit AURA v3 Genesis."""
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from run import compute_sources_sha256, ENGINE_DIR, REPO_ROOT
from run_v3 import DOCS_V3_DIR

TIMING_COLS_V3 = {
    "throughput_tps_2048",
    "train_bytes_per_sec",
}


def cmd_freeze():
    sha, per_file = compute_sources_sha256()
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    frozen = {
        "sources_sha256": sha,
        "file_count": len(per_file),
        "files": per_file,
        "dev_git_parent": git_head,
    }
    out = DOCS_V3_DIR / "FROZEN.json"
    out.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"[FREEZE-v3] Wrote {out} (SHA-256 = {sha}, {len(per_file)} .rs files)")


def compare_csv_excluding_timing(p1: Path, p2: Path) -> tuple[bool, int, list[str]]:
    with p1.open() as f1, p2.open() as f2:
        r1 = list(csv.DictReader(f1))
        r2 = list(csv.DictReader(f2))
    if len(r1) != len(r2):
        return False, 0, [f"row count mismatch {len(r1)} vs {len(r2)}"]
    diffs = []
    checked = 0
    for idx, (a, b) in enumerate(zip(r1, r2)):
        for k in a.keys():
            if k in TIMING_COLS_V3:
                continue
            checked += 1
            if a[k] != b[k]:
                diffs.append(f"row {idx} col {k}: {a[k]} != {b[k]}")
    return len(diffs) == 0, checked, diffs[:10]


def cmd_replay(seed_hex: str = "0xB600"):
    bin_path = ENGINE_DIR / "target" / "release" / "aura-v2"
    orig_dir = DOCS_V3_DIR / "raw" / "CONFIRM" / seed_hex
    if not orig_dir.exists():
        sys.exit(f"Original CONFIRM dir not found: {orig_dir}")

    with tempfile.TemporaryDirectory(prefix="aura_v3_replay_") as tmp:
        rep_dir = Path(tmp) / seed_hex
        subprocess.run(
            ["nice", "-n", "10", str(bin_path), "genesis", seed_hex, str(rep_dir), "--repo", str(REPO_ROOT), "--samples"],
            cwd=REPO_ROOT,
            check=True,
        )
        results = {}
        all_ok = True
        for fn in ["g1.csv", "g2.csv", "g3.csv", "g4.csv", "g5_fabric.csv", "g5_bytes.csv"]:
            ok, n_chk, diffs = compare_csv_excluding_timing(orig_dir / fn, rep_dir / fn)
            results[fn] = {"identical": ok, "fields_checked": n_chk, "diffs": diffs}
            all_ok = all_ok and ok
        s1 = (orig_dir / "g5_samples.txt").read_text()
        s2 = (rep_dir / "g5_samples.txt").read_text()
        samples_ok = (s1 == s2)
        results["g5_samples.txt"] = {"identical": samples_ok}
        all_ok = all_ok and samples_ok

    report = {
        "seed": seed_hex,
        "all_identical_excluding_timing": all_ok,
        "files": results,
    }
    out = DOCS_V3_DIR / "replay.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[REPLAY-v3] {seed_hex}: all_identical_excluding_timing = {all_ok}")
    if not all_ok:
        print(json.dumps(report, indent=2))
        sys.exit(1)


def cmd_audit():
    frozen = json.loads((DOCS_V3_DIR / "FROZEN.json").read_text())
    dev_meta = json.loads((DOCS_V3_DIR / "raw" / "DEV" / "run.json").read_text())
    conf_meta = json.loads((DOCS_V3_DIR / "raw" / "CONFIRM" / "run.json").read_text())
    sha_now, _ = compute_sources_sha256()

    assert frozen["sources_sha256"] == dev_meta["sources_sha256"] == conf_meta["sources_sha256"] == sha_now, \
        "SHA-256 mismatch across DEV / FROZEN / CONFIRM / current source!"

    expected_rows = {
        "g1": 64,         # 32 seeds * 2 arms
        "g2": 160,        # 32 seeds * 5 arms
        "g3": 96,         # 32 seeds * 3 arms
        "g4": 96,         # 32 seeds * 3 arms
        "g5_fabric": 64,  # 32 seeds * 2 arms
        "g5_bytes": 128,  # 32 seeds * 4 arms
    }
    nan_count = 0
    row_counts = {}
    for name, exp in expected_rows.items():
        p = DOCS_V3_DIR / f"metrics_{name}.csv"
        with p.open() as f:
            rows = list(csv.DictReader(f))
        row_counts[name] = len(rows)
        assert len(rows) == exp, f"{p.name}: expected {exp} rows, got {len(rows)}"
        for r in rows:
            for k, v in r.items():
                if k in ("phase", "arm", "modality"):
                    continue
                x = float(v)
                if math.isnan(x) or math.isinf(x):
                    nan_count += 1
    assert nan_count == 0, f"Found {nan_count} NaN/Inf values!"

    rep = json.loads((DOCS_V3_DIR / "replay.json").read_text())
    assert rep["all_identical_excluding_timing"] is True, "Replay verification failed!"

    print(f"[AUDIT-v3 PASS] SHA-256={sha_now} | rows={row_counts} | NaN/Inf=0 | replay=EXACT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["freeze", "replay", "audit"])
    args = ap.parse_args()
    if args.action == "freeze":
        cmd_freeze()
    elif args.action == "replay":
        cmd_replay()
    elif args.action == "audit":
        cmd_audit()


if __name__ == "__main__":
    main()
