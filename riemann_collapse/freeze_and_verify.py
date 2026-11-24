#!/usr/bin/env python3
"""Freeze SHA-256 pre-CONFIRM, verify bit-exact replay on 0xB800, and audit all evidence."""
import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys
import tempfile

from run import CRATE_DIR, DOCS_DIR, REPO_ROOT, compute_sources_sha256

TIMING_COLS = {"eval_ms"}
METRIC_FILES = [f"metrics_c{i}.csv" for i in range(1, 5)]


def cmd_freeze():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    sha, per_file = compute_sources_sha256()
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    frozen = {
        "sources_sha256": sha,
        "file_count": len(per_file),
        "files": per_file,
        "dev_git_parent": git_head,
    }
    out = DOCS_DIR / "FROZEN.json"
    out.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"[FREEZE] Wrote {out} (SHA-256 = {sha}, {len(per_file)} files)")


def compare_csv_excluding_timing(p1: pathlib.Path, p2: pathlib.Path) -> tuple[bool, int, list[str]]:
    with p1.open() as f1, p2.open() as f2:
        r1 = list(csv.DictReader(f1))
        r2 = list(csv.DictReader(f2))
    if len(r1) != len(r2):
        return False, 0, [f"row count mismatch {len(r1)} vs {len(r2)}"]
    diffs = []
    checked = 0
    for idx, (a, b) in enumerate(zip(r1, r2)):
        for k in a.keys():
            if k in TIMING_COLS:
                continue
            checked += 1
            if a[k] != b[k]:
                diffs.append(f"row {idx} col {k}: {a[k]} != {b[k]}")
    return len(diffs) == 0, checked, diffs[:10]


def cmd_replay(seed_hex: str = "0xB800"):
    bin_path = CRATE_DIR / "target" / "release" / "riemann-collapse-v1"
    orig_dir = DOCS_DIR / "raw" / "CONFIRM" / seed_hex
    assert orig_dir.exists(), f"Missing CONFIRM dir: {orig_dir}"

    with tempfile.TemporaryDirectory(prefix="riemann_collapse_replay_") as tmp:
        rep_dir = pathlib.Path(tmp) / seed_hex
        subprocess.run(
            [str(bin_path), "run", seed_hex, "CONFIRM", str(rep_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        per_file = {}
        total_checked = 0
        all_ok = True
        for fn in METRIC_FILES:
            ok, n, diffs = compare_csv_excluding_timing(orig_dir / fn, rep_dir / fn)
            per_file[fn] = {"exact_match": ok, "cells_checked": n, "diffs": diffs}
            total_checked += n
            all_ok = all_ok and ok

    report = {
        "seed": seed_hex,
        "bit_exact_match": all_ok,
        "total_non_timing_cells_checked": total_checked,
        "files": per_file,
    }
    (DOCS_DIR / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[REPLAY] {seed_hex}: bit_exact_match={all_ok}, cells={total_checked}")
    if not all_ok:
        sys.exit(1)


def cmd_audit():
    checks: list[tuple[str, bool, str]] = []

    sha_now, _ = compute_sources_sha256()
    frozen = json.loads((DOCS_DIR / "FROZEN.json").read_text())
    confirm_run = json.loads((DOCS_DIR / "raw" / "CONFIRM" / "run.json").read_text())
    sha_ok = (sha_now == frozen["sources_sha256"] == confirm_run["sources_sha256"])
    checks.append(("SHA-256 código fuente == FROZEN.json == CONFIRM/run.json", sha_ok, sha_now[:16]))

    expected_rows = {"metrics_c1.csv": 5, "metrics_c2.csv": 4, "metrics_c3.csv": 17, "metrics_c4.csv": 4}
    total_rows = 0
    nan_count = 0
    missing = []
    for phase, base in [("DEV", 0xA800), ("CONFIRM", 0xB800)]:
        for s in range(base, base + 16):
            sd = DOCS_DIR / "raw" / phase / f"0x{s:04X}"
            for fn, exp_n in expected_rows.items():
                p = sd / fn
                if not p.exists():
                    missing.append(str(p))
                    continue
                rows = list(csv.DictReader(p.open()))
                if len(rows) != exp_n:
                    missing.append(f"{p}: rows={len(rows)} != {exp_n}")
                total_rows += len(rows)
                for r in rows:
                    for k, v in r.items():
                        if k in {"seed", "phase", "operator", "arm", "probe"}:
                            continue
                        if not math.isfinite(float(v)):
                            nan_count += 1
    checks.append(("32 semillas completas (16 DEV + 16 CONFIRM, 960 filas totales)", len(missing) == 0 and total_rows == 32 * 30, f"rows={total_rows}, missing={len(missing)}"))
    checks.append(("Cero NaN / Inf en todas las métricas", nan_count == 0, f"non_finite={nan_count}"))

    rep = json.loads((DOCS_DIR / "replay.json").read_text())
    checks.append(("Replay determinista bit-a-bit en 0xB800", bool(rep["bit_exact_match"]), f"{rep['total_non_timing_cells_checked']} celdas exactas"))

    t_dbg = (DOCS_DIR / "tests.log").read_text()
    t_rel = (DOCS_DIR / "tests_release.log").read_text()
    tests_ok = ("test result: ok." in t_dbg and "0 failed" in t_dbg and "test result: ok." in t_rel and "0 failed" in t_rel)
    checks.append(("Tests unitarios debug y release 100% PASS", tests_ok, "tests.log & tests_release.log"))

    diff_files = subprocess.check_output(["git", "diff", "--name-only", "main"], cwd=REPO_ROOT, text=True).splitlines()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=REPO_ROOT, text=True).splitlines()
    all_touched = sorted(set(diff_files + untracked))
    isolated = all(f.startswith("riemann_collapse/") or f.startswith("docs/riemann_collapse_v1/") for f in all_touched)
    checks.append(("Aislamiento estricto (solo riemann_collapse/ y docs/riemann_collapse_v1/)", isolated, f"{len(all_touched)} archivos en rama"))

    overall = all(ok for _, ok, _ in checks)
    lines = [
        "# Auditoría de Evidencia — Colapso de Ondas Pasado–Futuro v1 (`exp/riemann-wave-collapse`)",
        "",
        f"**Estado global:** `{'AUDIT PASS' if overall else 'AUDIT FAIL'}`",
        "",
        "| Comprobación | Estado | Detalle |",
        "|---|---|---|",
    ]
    for title, ok, detail in checks:
        lines.append(f"| {title} | `{'PASS' if ok else 'FAIL'}` | `{detail}` |")
    lines += [
        "",
        "## Integración con aplicación principal (`main`)",
        "- Checks del crate raíz (`Paso0`, `known_acc16`, `E33`, `aura-engine`, `orbita`): **NO EVALUADOS** (rama aislada; ningún archivo de `main` fue modificado).",
        "",
    ]
    (DOCS_DIR / "AUDIT.md").write_text("\n".join(lines))
    print(f"[AUDIT] overall={'PASS' if overall else 'FAIL'}")
    if not overall:
        for title, ok, detail in checks:
            if not ok:
                print(f"  FAIL: {title} ({detail})")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["freeze", "replay", "audit"])
    a = p.parse_args()
    if a.cmd == "freeze":
        cmd_freeze()
    elif a.cmd == "replay":
        cmd_replay()
    elif a.cmd == "audit":
        cmd_audit()


if __name__ == "__main__":
    main()
