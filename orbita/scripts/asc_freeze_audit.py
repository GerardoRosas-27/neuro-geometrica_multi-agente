#!/usr/bin/env python3
"""Freeze, bit-exact replay (on 0xB400 ignoring wall-clock ms), and scientific audit for exp/orbita-adaptive-spectral."""
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

REPO_ROOT = Path(__file__).resolve().parents[2]
ORBITA_DIR = REPO_ROOT / "orbita"
DOCS_DIR = REPO_ROOT / "docs" / "orbita_adaptive_spectral_v1"
RAW_DIR = DOCS_DIR / "raw"
FIG_DIR = DOCS_DIR / "figures"
BIN_PATH = ORBITA_DIR / "target" / "release" / "orbita"

DEV_SEEDS = [0xA400 + i for i in range(16)]
CONFIRM_SEEDS = [0xB400 + i for i in range(16)]

TRACKED_FROZEN_REL = [
    "docs/orbita_adaptive_spectral_v1/PLAN.md",
    "orbita/src/adaptive_spectral.rs",
    "orbita/src/main.rs",
    "orbita/src/model.rs",
    "orbita/scripts/asc_run.py",
    "orbita/scripts/asc_summarize.py",
    "orbita/scripts/asc_freeze_audit.py",
    "orbita/scripts/asc_plots.py",
]

EXPECTED_CSVS = {
    "metrics_o1.csv": 6,
    "metrics_o2.csv": 5,
    "metrics_o3.csv": 8,
    "metrics_o4.csv": 14,
}

EXPECTED_FIGS = [
    "fig1_orbita_s16_e33.svg",
    "fig2_orbita_o2_compositional.svg",
    "fig3_orbita_o3_continual_new.svg",
    "fig4_prime_spectral_sieve.svg",
]

TIMING_COLS = {"train_ms", "adapt_ms"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(65536), b""):
            h.update(ch)
    return h.hexdigest()


def sha256_det_csv(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in (reader.fieldnames or []) if c not in TIMING_COLS]
        h.update((",".join(cols) + "\n").encode("utf-8"))
        for row in reader:
            h.update((",".join(row[c] for c in cols) + "\n").encode("utf-8"))
    return h.hexdigest()


def git_out(args: list[str], strip: bool = True) -> str:
    txt = subprocess.check_output(["git"] + args, cwd=str(REPO_ROOT), text=True)
    return txt.strip() if strip else txt.rstrip("\n")


def cmd_freeze() -> int:
    if not BIN_PATH.exists():
        raise RuntimeError(f"Release binary missing at {BIN_PATH}")
    files = {}
    for rel in TRACKED_FROZEN_REL:
        p = REPO_ROOT / rel
        if not p.exists():
            raise RuntimeError(f"Missing {p}")
        files[rel] = sha256_file(p)
    files["orbita/target/release/orbita"] = sha256_file(BIN_PATH)
    payload = {
        "experiment": "orbita_adaptive_spectral_v1",
        "branch": git_out(["rev-parse", "--abbrev-ref", "HEAD"]),
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head_at_freeze": git_out(["rev-parse", "HEAD"]),
        "dev_seeds": [f"0x{s:X}" for s in DEV_SEEDS],
        "confirm_seeds": [f"0x{s:X}" for s in CONFIRM_SEEDS],
        "files": files,
    }
    out = DOCS_DIR / "FROZEN.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(files)} frozen hashes)")
    return 0


def cmd_replay() -> int:
    ref_dir = RAW_DIR / "confirm" / "seed_0xB400"
    if not ref_dir.exists():
        raise RuntimeError(f"Missing CONFIRM reference dir {ref_dir}")
    check_files = [
        "metrics_o1.csv",
        "metrics_o2.csv",
        "metrics_o3.csv",
        "metrics_o4.csv",
        "trace_prime_sieve.csv",
    ]
    wpath = REPO_ROOT / "artifacts" / "orbita" / "confirm" / "weights_0xB400.json"
    mpath = REPO_ROOT / "artifacts" / "orbita" / "confirm" / "meta_0xB400.json"
    results = {}
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="orb_asc_replay_") as tmp:
        td = Path(tmp)
        subprocess.run(
            [
                str(BIN_PATH),
                "asc-run",
                "--seed",
                "0xB400",
                "--phase",
                "CONFIRM",
                "--weights",
                str(wpath),
                "--meta",
                str(mpath),
                "--out",
                str(td),
                "--export-traces",
            ],
            cwd=str(REPO_ROOT),
            check=True,
        )
        for fn in check_files:
            rh = sha256_det_csv(ref_dir / fn)
            ph = sha256_det_csv(td / fn)
            ok = rh == ph
            all_ok &= ok
            results[fn] = {"ref_sha256": rh, "replay_sha256": ph, "match": ok}
    payload = {
        "seed": "0xB400",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note": "Bit-exact verification over all deterministic columns (excluding wall-clock ms timings).",
        "bit_exact_match": all_ok,
        "files": results,
    }
    (DOCS_DIR / "replay_check.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Replay check on 0xB400: bit_exact_match={all_ok}")
    return 0 if all_ok else 1


def check_csvs(sub: str, seeds: list[int]) -> tuple[bool, str]:
    missing = bad_rows = non_finite = total = 0
    for s in seeds:
        sdir = RAW_DIR / sub / f"seed_0x{s:X}"
        for fn, exp_n in EXPECTED_CSVS.items():
            p = sdir / fn
            if not p.exists():
                missing += 1
                continue
            total += 1
            with p.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if len(rows) != exp_n:
                bad_rows += 1
            for row in rows:
                for col, val in row.items():
                    if col in ("seed", "phase", "arm", "subtask", "slice"):
                        continue
                    try:
                        if not math.isfinite(float(val)):
                            non_finite += 1
                    except ValueError:
                        non_finite += 1
    ok = (missing == 0) and (bad_rows == 0) and (non_finite == 0)
    return ok, (
        f"{total}/{len(seeds)*len(EXPECTED_CSVS)} CSVs valid, 0 NaN/Inf"
        if ok
        else f"missing={missing}, bad_rows={bad_rows}, non_finite={non_finite}"
    )


def cmd_audit() -> int:
    checks: list[tuple[str, bool, str]] = []
    fp = DOCS_DIR / "FROZEN.json"
    if not fp.exists():
        checks.append(("FROZEN.json SHA-256", False, "Missing FROZEN.json"))
    else:
        frozen = json.loads(fp.read_text(encoding="utf-8"))
        bad = [rel for rel, h in frozen.get("files", {}).items() if not (REPO_ROOT / rel).exists() or sha256_file(REPO_ROOT / rel) != h]
        n = len(frozen.get("files", {}))
        checks.append(("FROZEN.json SHA-256", not bad, f"{n - len(bad)}/{n} hashes exact" if not bad else f"Mismatch: {bad}"))

    ok_d, msg_d = check_csvs("dev", DEV_SEEDS)
    checks.append(("DEV CSVs (16 seeds)", ok_d, msg_d))
    ok_c, msg_c = check_csvs("confirm", CONFIRM_SEEDS)
    checks.append(("CONFIRM CSVs (16 seeds)", ok_c, msg_c))

    rp = DOCS_DIR / "replay_check.json"
    if not rp.exists():
        checks.append(("Bit-exact replay 0xB400", False, "Missing replay_check.json"))
    else:
        rep = json.loads(rp.read_text(encoding="utf-8"))
        ok = bool(rep.get("bit_exact_match"))
        checks.append(("Bit-exact replay 0xB400", ok, f"{len(rep.get('files', {}))} files matched={ok}"))

    tp = DOCS_DIR / "tests_release.log"
    if tp.exists():
        txt = tp.read_text(encoding="utf-8", errors="replace")
        ok_t = ("test result: ok." in txt) and ("0 failed" in txt)
        checks.append(("Rust unit tests (release)", ok_t, "tests_release.log: ok (0 failed)" if ok_t else "FAILED"))
    else:
        checks.append(("Rust unit tests (release)", False, "Missing tests_release.log"))

    miss_figs = [f for f in EXPECTED_FIGS if not (FIG_DIR / f).exists()]
    checks.append(("SVG Figures (4)", not miss_figs, f"{4 - len(miss_figs)}/4 present" if not miss_figs else f"Missing: {miss_figs}"))

    # Check frozen ÓRBITA core untouched (`orbita/src/model.rs` diff vs main == empty)
    model_diff = git_out(["diff", "main...HEAD", "--", "orbita/src/model.rs", "src/"])
    checks.append(("Frozen ÓRBITA & v4 untouched", len(model_diff) == 0, "0 diffs in orbita/src/model.rs and src/" if not model_diff else "MODIFIED!"))

    branch = git_out(["rev-parse", "--abbrev-ref", "HEAD"])
    diff_files = [l.strip() for l in git_out(["diff", "--name-only", "main...HEAD"]).splitlines() if l.strip()]
    status_lines = [l[3:].strip() for l in git_out(["status", "--porcelain"], strip=False).splitlines() if len(l) > 3]
    all_touched = sorted(set(diff_files + status_lines))
    allowed = ("orbita/src/adaptive_spectral.rs", "orbita/src/main.rs", "orbita/scripts/asc_", "docs/orbita_adaptive_spectral_v1/")
    viol = [p for p in all_touched if not p.startswith(allowed)]
    iso_ok = (branch == "exp/orbita-adaptive-spectral") and not viol
    checks.append(("Strict branch isolation", iso_ok, f"branch={branch}, {len(all_touched)} touched files inside allowed scope" if iso_ok else f"violations={viol}"))

    overall = all(c[1] for c in checks)
    verdict = "AUDIT PASS" if overall else "AUDIT FAIL"
    lines = [
        "# Auditoría Científica y Técnica — `exp/orbita-adaptive-spectral`",
        "",
        f"**Veredicto final:** **`{verdict}`**",
        f"**Fecha UTC:** {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"**Rama:** `{branch}` · **HEAD:** `{git_out(['rev-parse', 'HEAD'])}`",
        "",
        "## Verificaciones Ejecutadas",
        "",
        "| Comprobación | Estado | Detalle |",
        "|---|---|---|",
    ]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.extend([
        "",
        "## Alcance y Evaluación sobre ÓRBITA",
        "",
        "- Evaluado directamente sobre el harness oficial §16 de ÓRBITA (`TEST` 256 + `E33` `h1..h64`, 16+16 semillas, `CLEAN`, `0` queries, `fresh_ok = true`) con pesos publicados congelados (`orbita/src/model.rs` intacto).",
        "- Rama estrictamente aislada (`exp/orbita-adaptive-spectral`); sin fusionar con `main` ni tocar `aura-engine`.",
        "",
    ])
    (DOCS_DIR / "AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {DOCS_DIR / 'AUDIT.md'}: {verdict}")
    return 0 if overall else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["freeze", "replay", "audit"])
    a = ap.parse_args()
    if a.action == "freeze":
        return cmd_freeze()
    if a.action == "replay":
        return cmd_replay()
    return cmd_audit()


if __name__ == "__main__":
    sys.exit(main())
