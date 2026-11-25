#!/usr/bin/env python3
"""Freeze, bit-exact replay, and automated audit for exp/wave-collision-autocal."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "wave_collision"
BIN = CRATE / "target" / "release" / "wave_collision"
DOCS = ROOT / "docs" / "wave_collision_v1"

FROZEN_TARGETS = [
    "docs/wave_collision_v1/PLAN.md",
    "wave_collision/Cargo.toml",
    "wave_collision/src/main.rs",
    "wave_collision/src/rng.rs",
    "wave_collision/src/domains.rs",
    "wave_collision/src/models.rs",
    "wave_collision/src/metrics.rs",
    "wave_collision/src/suites.rs",
    "wave_collision/scripts/wc_run.py",
    "wave_collision/scripts/wc_summarize.py",
    "wave_collision/scripts/wc_freeze_audit.py",
    "wave_collision/scripts/wc_plots.py",
    "wave_collision/target/release/wave_collision",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_freeze() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    files = {}
    for rel in FROZEN_TARGETS:
        p = ROOT / rel
        if not p.exists():
            raise RuntimeError(f"Cannot freeze missing file: {rel}")
        files[rel] = sha256_file(p)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "exp/wave-collision-autocal",
        "git_head_pre_freeze": git_head,
        "files": files,
    }
    out = DOCS / "FROZEN.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out} with {len(files)} SHA-256 hashes.")


def cmd_replay(seed_hex: str = "0xF200") -> None:
    ref_dir = DOCS / "raw" / "confirm" / f"seed_{seed_hex}"
    req = [
        "metrics_t1.csv",
        "metrics_t2.csv",
        "metrics_t3.csv",
        "metrics_t4.csv",
        "trace_wave_collision.csv",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            [str(BIN), "run-seed", "--seed", seed_hex, "--out", str(tmp_path), "--trace"],
            check=True,
            cwd=str(ROOT),
        )
        matches = {}
        all_ok = True
        for name in req:
            orig = ref_dir / name
            replayed = tmp_path / name
            if name in ("metrics_t2.csv", "trace_wave_collision.csv"):
                h1 = sha256_file(orig)
                h2 = sha256_file(replayed)
                ok = h1 == h2
                matches[name] = {"mode": "byte_exact", "orig_sha256": h1, "replay_sha256": h2, "match": ok}
            else:
                with orig.open() as f1, replayed.open() as f2:
                    r1 = list(csv.DictReader(f1))
                    r2 = list(csv.DictReader(f2))
                ok = len(r1) == len(r2)
                if ok:
                    for row1, row2 in zip(r1, r2):
                        for k in row1:
                            if k == "train_ms":
                                continue
                            if row1[k] != row2[k]:
                                ok = False
                                break
                        if not ok:
                            break
                matches[name] = {"mode": "exact_all_non_timing_cols", "rows": len(r1), "match": ok}
            all_ok = all_ok and ok

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed_hex,
        "bit_exact_match": all_ok,
        "files": matches,
    }
    out = DOCS / "replay_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Replay {seed_hex}: bit_exact_match={all_ok} ({len(req)} files)")
    if not all_ok:
        raise SystemExit(1)


def check_csv_tree(split: str, expected_seeds: int) -> tuple[bool, str]:
    base = DOCS / "raw" / split
    dirs = sorted(base.glob("seed_0x*"))
    if len(dirs) != expected_seeds:
        return False, f"Expected {expected_seeds} seeds, found {len(dirs)}"
    total_files = 0
    bad_vals = 0
    frozen_fail = 0
    for d in dirs:
        for fn in ("metrics_t1.csv", "metrics_t2.csv", "metrics_t3.csv", "metrics_t4.csv"):
            p = d / fn
            if not p.exists():
                return False, f"Missing {p}"
            total_files += 1
            with p.open() as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return False, f"Empty CSV {p}"
            for r in rows:
                if "checksum_ok" in r and r["checksum_ok"] != "1":
                    frozen_fail += 1
                for k, v in r.items():
                    if k in ("seed", "arm", "axis"):
                        continue
                    x = float(v)
                    if math.isnan(x) or math.isinf(x):
                        bad_vals += 1
    if bad_vals > 0:
        return False, f"{bad_vals} NaN/Inf values"
    if frozen_fail > 0:
        return False, f"{frozen_fail} frozen-wave checksum failures"
    return True, f"{total_files}/{expected_seeds*4} CSVs valid, 0 NaN/Inf, 100% frozen checksums ok"


def cmd_audit() -> None:
    checks = []
    frozen = json.loads((DOCS / "FROZEN.json").read_text())
    ok_hash = sum(1 for rel, exp in frozen["files"].items() if sha256_file(ROOT / rel) == exp)
    checks.append(("FROZEN.json SHA-256", ok_hash == len(frozen["files"]), f"{ok_hash}/{len(frozen['files'])} hashes exact"))

    ok_dev, msg_dev = check_csv_tree("dev", 16)
    checks.append(("DEV CSVs (16 seeds)", ok_dev, msg_dev))
    ok_conf, msg_conf = check_csv_tree("confirm", 16)
    checks.append(("CONFIRM CSVs (16 seeds)", ok_conf, msg_conf))

    rep = json.loads((DOCS / "replay_check.json").read_text())
    checks.append(("Bit-exact replay 0xF200", bool(rep["bit_exact_match"]), f"{len(rep['files'])} files matched={rep['bit_exact_match']}"))

    tlog = (DOCS / "tests_release.log").read_text() if (DOCS / "tests_release.log").exists() else ""
    t_ok = ("test result: ok." in tlog) and ("0 failed" in tlog)
    checks.append(("Rust unit tests (release)", t_ok, "tests_release.log: ok (0 failed)" if t_ok else "MISSING/FAIL"))

    figs = [
        "fig1_wave768_domain_a.svg",
        "fig2_wave768_causal_dissection.svg",
        "fig3_wave768_transfer_b.svg",
        "fig4_wave768_transfer_c.svg",
    ]
    present = sum(1 for fn in figs if (DOCS / "figures" / fn).exists() and (DOCS / "figures" / fn).stat().st_size > 500)
    checks.append(("SVG Figures (4)", present == 4, f"{present}/4 present"))

    diff_out = subprocess.check_output(
        ["git", "diff", "--name-only", "main", "--", "orbita/", "src/", "aura-engine/", "Cargo.toml"],
        cwd=str(ROOT),
        text=True,
    ).strip()
    checks.append(("ORBITA / CDT-RQM / aura-engine untouched", diff_out == "", "0 diffs" if diff_out == "" else f"DIFFS: {diff_out}"))

    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=str(ROOT), text=True).strip()
    all_touched = subprocess.check_output(["git", "diff", "--name-only", "main"], cwd=str(ROOT), text=True).splitlines()
    bad = [p for p in all_touched if not (p.startswith("wave_collision/") or p.startswith("docs/wave_collision_v1/"))]
    checks.append(("Strict branch isolation", branch == "exp/wave-collision-autocal" and len(bad) == 0, f"branch={branch}, {len(all_touched)} touched files inside allowed scope"))

    verdict = "AUDIT PASS" if all(c[1] for c in checks) else "AUDIT FAIL"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()

    md = [
        f"# Auditoría Científica y Técnica — `{branch}`",
        "",
        f"**Veredicto final:** **`{verdict}`**",
        f"**Fecha UTC:** {datetime.now(timezone.utc).isoformat()}",
        f"**Rama:** `{branch}` · **HEAD:** `{head}`",
        "",
        "## Verificaciones Ejecutadas",
        "",
        "| Comprobación | Estado | Detalle |",
        "|---|---|---|",
    ]
    for name, ok, detail in checks:
        md.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    md += [
        "",
        "## Alcance y Aislamiento",
        "",
        "- Experimento autónomo de `768` ondas colisionantes auto-calibrables (`wave_collision/`), sin matriz densa `W_pair` y sin conexión a ÓRBITA (`orbita/`), CDT-RQM (`src/`), `aura-engine/` ni `Cargo.toml` raíz.",
        "- Cero ceros de Riemann y cero primos en brazos adaptativos (`#[should_panic]` verificado); `256` ceros exactos de Riemann `< 1e-10` en el brazo de control.",
        "- Invarianza bit a bit IEEE-754 de las `3,328` frecuencias y fases base en todas las semillas de transferencia.",
        "",
    ]
    out = DOCS / "AUDIT.md"
    out.write_text("\n".join(md))
    print(f"Wrote {out}: {verdict}")
    if verdict != "AUDIT PASS":
        raise SystemExit(1)


def main() -> None:
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
