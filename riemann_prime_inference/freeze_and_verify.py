#!/usr/bin/env python3
"""Freeze, bit-exact replay verification, and final scientific audit for
`exp/riemann-prime-inference` (Python standard library only).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DOCS_DIR = REPO_ROOT / "docs" / "riemann_prime_inference_v1"
RAW_DIR = DOCS_DIR / "raw"
FIG_DIR = DOCS_DIR / "figures"
BIN_PATH = ROOT / "target" / "release" / "riemann-prime-inference-v1"

DEV_SEEDS = [0xA900 + i for i in range(16)]
CONFIRM_SEEDS = [0xB900 + i for i in range(16)]

TRACKED_FROZEN_REL = [
    "docs/riemann_prime_inference_v1/PLAN.md",
    "riemann_prime_inference/Cargo.toml",
    "riemann_prime_inference/src/lib.rs",
    "riemann_prime_inference/src/main.rs",
    "riemann_prime_inference/src/rng.rs",
    "riemann_prime_inference/src/riemann.rs",
    "riemann_prime_inference/src/inference.rs",
    "riemann_prime_inference/src/suites.rs",
    "riemann_prime_inference/run.py",
    "riemann_prime_inference/summarize.py",
    "riemann_prime_inference/freeze_and_verify.py",
    "riemann_prime_inference/render_plots.py",
]

EXPECTED_CSVS = {
    "metrics_i1.csv": 12,
    "metrics_i2.csv": 4,
    "metrics_i3.csv": 4,
    "metrics_i4.csv": 13,
}

EXPECTED_FIGS = [
    "fig1_retrodictive_tracing.svg",
    "fig2_downstream_and_counterfactual.svg",
    "fig3_bayesian_disambiguation.svg",
    "fig4_analogy_and_limits.svg",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(args: list[str]) -> str:
    out = subprocess.check_output(["git"] + args, cwd=str(REPO_ROOT), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace").strip()


def cmd_freeze() -> int:
    if not BIN_PATH.exists():
        raise RuntimeError(f"Release binary missing at {BIN_PATH}; build before freezing.")
    files = {}
    for rel in TRACKED_FROZEN_REL:
        p = REPO_ROOT / rel
        if not p.exists():
            raise RuntimeError(f"Cannot freeze; missing {p}")
        files[rel] = sha256_file(p)
    files["riemann_prime_inference/target/release/riemann-prime-inference-v1"] = sha256_file(BIN_PATH)

    payload = {
        "experiment": "riemann_prime_inference_v1",
        "branch": git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head_at_freeze": git_output(["rev-parse", "HEAD"]),
        "dev_seeds": [f"0x{s:X}" for s in DEV_SEEDS],
        "confirm_seeds": [f"0x{s:X}" for s in CONFIRM_SEEDS],
        "files": files,
    }
    out_path = DOCS_DIR / "FROZEN.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(files)} frozen hashes).")
    return 0


def cmd_replay() -> int:
    ref_dir = RAW_DIR / "confirm" / "seed_0xB900"
    if not ref_dir.exists():
        raise RuntimeError(f"Reference CONFIRM seed 0xB900 not found at {ref_dir}")

    check_files = [
        "metrics_i1.csv",
        "metrics_i2.csv",
        "metrics_i3.csv",
        "metrics_i4.csv",
        "trace_retro_example.csv",
        "trace_bayesian_example.csv",
    ]
    results = {}
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="rpi_replay_") as tmp:
        tmp_dir = Path(tmp)
        subprocess.run(
            [str(BIN_PATH), "run", "0xB900", "CONFIRM", str(tmp_dir), "--export-traces"],
            check=True,
            cwd=str(ROOT),
        )
        for fname in check_files:
            ref_p = ref_dir / fname
            rep_p = tmp_dir / fname
            ref_h = sha256_file(ref_p)
            rep_h = sha256_file(rep_p)
            matched = ref_h == rep_h
            if not matched:
                all_ok = False
            results[fname] = {
                "ref_sha256": ref_h,
                "replay_sha256": rep_h,
                "match": matched,
            }

    payload = {
        "seed": "0xB900",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bit_exact_match": all_ok,
        "files": results,
    }
    out_path = DOCS_DIR / "replay_check.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Replay check on 0xB900: bit_exact_match={all_ok}")
    return 0 if all_ok else 1


def check_csv_integrity(phase_name: str, seeds: list[int]) -> tuple[bool, str]:
    missing = []
    bad_rows = []
    non_finite = []
    total_files = 0
    for seed in seeds:
        sdir = RAW_DIR / phase_name / f"seed_0x{seed:X}"
        for fname, expected_rows in EXPECTED_CSVS.items():
            p = sdir / fname
            if not p.exists():
                missing.append(str(p.relative_to(REPO_ROOT)))
                continue
            total_files += 1
            with p.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if len(rows) != expected_rows:
                bad_rows.append(f"{p.name}@0x{seed:X}: got {len(rows)} rows expected {expected_rows}")
            for r_idx, row in enumerate(rows):
                for col, val in row.items():
                    if col in ("seed", "phase", "arm", "subtask", "param_name"):
                        continue
                    try:
                        fv = float(val)
                        if not math.isfinite(fv):
                            non_finite.append(f"{p.name}@0x{seed:X} r{r_idx} {col}={val}")
                    except ValueError:
                        non_finite.append(f"{p.name}@0x{seed:X} r{r_idx} {col}={val}")
    ok = not missing and not bad_rows and not non_finite
    detail = f"{total_files}/{len(seeds)*len(EXPECTED_CSVS)} CSVs valid, 0 NaN/Inf"
    if not ok:
        detail = f"missing={len(missing)}, bad_rows={len(bad_rows)}, non_finite={len(non_finite)}"
    return ok, detail


def check_test_log(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Missing {path.name}"
    text = path.read_text(encoding="utf-8", errors="replace")
    ok = ("test result: ok." in text) and ("0 failed" in text)
    return ok, f"{path.name}: {'ok (0 failed)' if ok else 'FAILED or incomplete'}"


def cmd_audit() -> int:
    checks: list[tuple[str, bool, str]] = []

    # 1. FROZEN.json SHA-256 verification
    frozen_path = DOCS_DIR / "FROZEN.json"
    if not frozen_path.exists():
        checks.append(("FROZEN.json SHA-256", False, "Missing FROZEN.json"))
    else:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        bad = []
        for rel, expected in frozen.get("files", {}).items():
            p = REPO_ROOT / rel
            if not p.exists() or sha256_file(p) != expected:
                bad.append(rel)
        checks.append(
            (
                "FROZEN.json SHA-256",
                len(bad) == 0,
                f"{len(frozen.get('files', {})) - len(bad)}/{len(frozen.get('files', {}))} hashes exact"
                if not bad
                else f"Mismatches: {', '.join(bad)}",
            )
        )

    # 2. DEV and CONFIRM CSV completeness & 0 NaN/Inf
    dev_ok, dev_msg = check_csv_integrity("dev", DEV_SEEDS)
    checks.append(("DEV CSVs (16 seeds)", dev_ok, dev_msg))
    conf_ok, conf_msg = check_csv_integrity("confirm", CONFIRM_SEEDS)
    checks.append(("CONFIRM CSVs (16 seeds)", conf_ok, conf_msg))

    # 3. Bit-exact replay on 0xB900
    rep_path = DOCS_DIR / "replay_check.json"
    if not rep_path.exists():
        checks.append(("Bit-exact replay 0xB900", False, "Missing replay_check.json"))
    else:
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        ok = bool(rep.get("bit_exact_match"))
        checks.append(("Bit-exact replay 0xB900", ok, f"{len(rep.get('files', {}))} files matched={ok}"))

    # 4. Debug & Release unit tests
    ok_d, msg_d = check_test_log(DOCS_DIR / "tests.log")
    checks.append(("Rust unit tests (debug)", ok_d, msg_d))
    ok_r, msg_r = check_test_log(DOCS_DIR / "tests_release.log")
    checks.append(("Rust unit tests (release)", ok_r, msg_r))

    # 5. Figures present
    missing_figs = [f for f in EXPECTED_FIGS if not (FIG_DIR / f).exists()]
    checks.append(
        (
            "SVG Figures (4)",
            len(missing_figs) == 0,
            f"{len(EXPECTED_FIGS) - len(missing_figs)}/4 present"
            if not missing_figs
            else f"Missing: {', '.join(missing_figs)}",
        )
    )

    # 6. Branch isolation strictly within riemann_prime_inference/ and docs/riemann_prime_inference_v1/
    branch = git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    diff_files = [
        line for line in git_output(["diff", "--name-only", "main...HEAD"]).splitlines() if line.strip()
    ]
    status_lines = [
        line[3:].strip()
        for line in git_output(["status", "--porcelain"]).splitlines()
        if line.strip()
    ]
    all_touched = sorted(set(diff_files + status_lines))
    allowed_prefixes = ("riemann_prime_inference/", "docs/riemann_prime_inference_v1/")
    violations = [p for p in all_touched if not p.startswith(allowed_prefixes)]
    iso_ok = (branch == "exp/riemann-prime-inference") and (len(violations) == 0)
    checks.append(
        (
            "Strict branch isolation",
            iso_ok,
            f"branch={branch}, {len(all_touched)} touched files all inside allowed directories"
            if iso_ok
            else f"branch={branch}, violations={violations}",
        )
    )

    overall_pass = all(c[1] for c in checks)
    verdict = "AUDIT PASS" if overall_pass else "AUDIT FAIL"

    lines = [
        "# Auditoría Científica y Técnica — `exp/riemann-prime-inference`",
        "",
        f"**Veredicto final:** **`{verdict}`**",
        f"**Fecha UTC:** {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"**Rama:** `{branch}` · **HEAD:** `{git_output(['rev-parse', 'HEAD'])}`",
        "",
        "## Verificaciones Ejecutadas",
        "",
        "| Comprobación | Estado | Detalle |",
        "|---|---|---|",
    ]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    lines.extend(
        [
            "",
            "## Alcance y Checks de la App Raíz",
            "",
            "- Experimento estrictamente aislado en `riemann_prime_inference/` y `docs/riemann_prime_inference_v1/`.",
            "- Checks de la aplicación raíz en `main` (`Paso0`, `known_acc16`, `E33`, `aura-engine`, `orbita`): **NO EVALUADOS**.",
            "",
        ]
    )

    audit_path = DOCS_DIR / "AUDIT.md"
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {audit_path}: {verdict}")
    return 0 if overall_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "replay", "audit"])
    args = parser.parse_args()
    if args.action == "freeze":
        return cmd_freeze()
    if args.action == "replay":
        return cmd_replay()
    return cmd_audit()


if __name__ == "__main__":
    sys.exit(main())
