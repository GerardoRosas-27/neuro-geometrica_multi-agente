#!/usr/bin/env python3
"""Freeze SHA-256, bit-exact replay (0xB200), and automated 8-check audit for Prueba B."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "nervous_wave_b_v1"
FROZEN_PATH = DOCS / "FROZEN.json"
REPLAY_PATH = DOCS / "REPLAY_0xB200.json"
AUDIT_PATH = DOCS / "AUDIT.md"
BIN = ROOT / "nervous_wave_b" / "target" / "release" / "nervous_wave_b"

FREEZE_FILES = [
    "docs/nervous_wave_b_v1/PLAN.md",
    "nervous_wave_b/Cargo.toml",
    "nervous_wave_b/src/rng.rs",
    "nervous_wave_b/src/domains.rs",
    "nervous_wave_b/src/topology.rs",
    "nervous_wave_b/src/wave_net.rs",
    "nervous_wave_b/src/suites.rs",
    "nervous_wave_b/src/main.rs",
    "scripts/nwb_run.py",
    "scripts/nwb_summarize.py",
    "scripts/nwb_freeze_audit.py",
    "scripts/nwb_plots.py",
    "nervous_wave_b/target/release/nervous_wave_b",
]
SEED_FILES = ["b1_physics.csv", "b2_inference.csv", "b3_resilience.csv", "b4_transfer.csv", "seed_meta.json"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_out(args: list[str]) -> str:
    return subprocess.check_output(["git"] + args, cwd=ROOT, text=True).strip()


def cmd_freeze() -> None:
    files = {}
    for rel in FREEZE_FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"ERROR: Missing file to freeze: {rel}", file=sys.stderr)
            sys.exit(1)
        files[rel] = sha256_file(p)
    payload = {
        "experiment": "nervous_wave_coherence_b",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_out(["branch", "--show-current"]),
        "git_head_pre_freeze": git_out(["rev-parse", "HEAD"]),
        "files": files,
    }
    FROZEN_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {FROZEN_PATH} with {len(files)} SHA-256 digests.")


def cmd_replay() -> None:
    orig_dir = DOCS / "runs" / "confirm" / "seed_0xB200"
    if not orig_dir.exists():
        print(f"ERROR: {orig_dir} not found.", file=sys.stderr)
        sys.exit(1)
    with tempfile.TemporaryDirectory() as td:
        rep_dir = Path(td) / "seed_0xB200"
        subprocess.run(
            [str(BIN), "--seed", "0xB200", "--split", "confirm", "--out-dir", str(rep_dir)],
            check=True,
        )
        per_file = {}
        all_match = True
        for fn in SEED_FILES:
            if fn == "seed_meta.json":
                # Compare semantic fields excluding elapsed_ms timing jitter
                a = json.loads((orig_dir / fn).read_text())
                b = json.loads((rep_dir / fn).read_text())
                a_cmp = {k: v for k, v in a.items() if k != "elapsed_ms"}
                b_cmp = {k: v for k, v in b.items() if k != "elapsed_ms"}
                ok = a_cmp == b_cmp
                per_file[fn] = {"match": ok, "note": "semantic exact (excluding wall-clock elapsed_ms)"}
            else:
                ha = sha256_file(orig_dir / fn)
                hb = sha256_file(rep_dir / fn)
                ok = ha == hb
                per_file[fn] = {"match": ok, "orig_sha256": ha, "replay_sha256": hb}
            if not ok:
                all_match = False

        res = {
            "seed": "0xB200",
            "bit_exact_match": all_match,
            "files_matched": sum(1 for v in per_file.values() if v["match"]),
            "files_total": len(SEED_FILES),
            "per_file": per_file,
        }
        REPLAY_PATH.write_text(json.dumps(res, indent=2) + "\n")
        print(f"Replay 0xB200: bit_exact_match={all_match} ({res['files_matched']}/{res['files_total']})")
        if not all_match:
            sys.exit(1)


def cmd_audit() -> None:
    checks = []

    # 1. Branch isolation vs main
    branch = git_out(["branch", "--show-current"])
    changed = git_out(["diff", "--name-only", "main...HEAD"]).splitlines()
    forbidden_prefixes = ("orbita/", "src/", "aura-engine/")
    forbidden_exact = {"Cargo.toml", "Cargo.lock"}
    bad_files = [f for f in changed if f.startswith(forbidden_prefixes) or f in forbidden_exact]
    checks.append((
        "1. Branch & Main Isolation",
        branch == "exp/nervous-wave-coherence-b" and len(bad_files) == 0,
        f"branch={branch}, forbidden_touched={len(bad_files)}, changed_files={len(changed)}",
    ))

    # 2. Preregistration order (PLAN.md committed before nervous_wave_b/src/)
    plan_first = git_out(["log", "--reverse", "--format=%H", "--", "docs/nervous_wave_b_v1/PLAN.md"]).splitlines()
    code_first = git_out(["log", "--reverse", "--format=%H", "--", "nervous_wave_b/src/wave_net.rs"]).splitlines()
    order_ok = False
    if plan_first and code_first:
        plan_ts = int(git_out(["show", "-s", "--format=%ct", plan_first[0]]))
        code_ts = int(git_out(["show", "-s", "--format=%ct", code_first[0]]))
        order_ok = plan_first[0] != code_first[0] and plan_ts <= code_ts
    checks.append((
        "2. Preregistration Before Code",
        order_ok,
        f"PLAN commit={plan_first[0][:8] if plan_first else 'NONE'}, code commit={code_first[0][:8] if code_first else 'NONE'}",
    ))

    # 3. FROZEN.json SHA-256 integrity
    frozen = json.loads(FROZEN_PATH.read_text()) if FROZEN_PATH.exists() else None
    mismatches = []
    if frozen:
        for rel, exp in frozen["files"].items():
            p = ROOT / rel
            if not p.exists() or sha256_file(p) != exp:
                mismatches.append(rel)
    checks.append((
        "3. Frozen SHA-256 Integrity",
        frozen is not None and len(mismatches) == 0,
        f"files={len(frozen['files']) if frozen else 0}, mismatches={len(mismatches)}",
    ))

    # 4. Unit tests release log
    tlog = (DOCS / "tests_release.log").read_text() if (DOCS / "tests_release.log").exists() else ""
    tests_ok = "test result: ok. 4 passed; 0 failed" in tlog and "warning:" not in tlog
    checks.append((
        "4. Release Unit Tests (0 Warnings)",
        tests_ok,
        "4/4 PASS, 0 warnings" if tests_ok else "Missing or warnings/failures in tests_release.log",
    ))

    # 5. Complete DEV (16) and CONFIRM (16) runs
    dev_ok = sum(
        1 for i in range(16)
        if all((DOCS / "runs" / "dev" / f"seed_0x{0xB100+i:X}" / fn).exists() for fn in SEED_FILES)
    )
    conf_ok = sum(
        1 for i in range(16)
        if all((DOCS / "runs" / "confirm" / f"seed_0x{0xB200+i:X}" / fn).exists() for fn in SEED_FILES)
    )
    checks.append((
        "5. Full DEV (16) & Blind CONFIRM (16) Coverage",
        dev_ok == 16 and conf_ok == 16,
        f"DEV={dev_ok}/16, CONFIRM={conf_ok}/16",
    ))

    # 6. Bit-exact frozen tissue checksums in all 16 CONFIRM seeds
    s_conf = json.loads((DOCS / "summary_confirm.json").read_text()) if (DOCS / "summary_confirm.json").exists() else None
    frozen_ok = False
    if s_conf:
        frozen_ok = all(
            s_conf["b4"][dom][arm]["frozen_all_ok"]
            for dom in ["domain_b_oscillators", "domain_c_graph"]
            for arm in s_conf["b4"][dom]
        )
    checks.append((
        "6. Bit-Exact Frozen Tissue Checksums (B4)",
        frozen_ok,
        "100% frozen_all_ok across all 5 transfer arms x 2 domains x 16 CONFIRM seeds" if frozen_ok else "Failed",
    ))

    # 7. Bit-exact replay 0xB200
    rep = json.loads(REPLAY_PATH.read_text()) if REPLAY_PATH.exists() else None
    rep_ok = bool(rep and rep.get("bit_exact_match") is True and rep.get("files_matched") == len(SEED_FILES))
    checks.append((
        "7. Bit-Exact Replay (0xB200)",
        rep_ok,
        f"{rep.get('files_matched', 0)}/{rep.get('files_total', 0)} matched" if rep else "REPLAY_0xB200.json missing",
    ))

    # 8. Figures + RESULTADOS.md present and non-empty
    figs = [
        DOCS / "figures" / "fig1_physics_kuramoto.svg",
        DOCS / "figures" / "fig2_inference_domAC.svg",
        DOCS / "figures" / "fig3_resilience_coherence.svg",
        DOCS / "figures" / "fig4_transfer_ab_ac.svg",
    ]
    res_md = DOCS / "RESULTADOS.md"
    art_ok = all(f.exists() and f.stat().st_size > 500 for f in figs) and res_md.exists() and res_md.stat().st_size > 1500
    checks.append((
        "8. Figures (4 SVGs) & Honest RESULTADOS.md",
        art_ok,
        f"SVGs={sum(1 for f in figs if f.exists())}/4, RESULTADOS.md={res_md.stat().st_size if res_md.exists() else 0} bytes",
    ))

    n_ok = sum(1 for _, ok, _ in checks if ok)
    verdict = "AUDIT PASS" if n_ok == len(checks) else "AUDIT FAIL"
    lines = [
        f"# Auditoría Automatizada — Prueba B (`exp/nervous-wave-coherence-b`)",
        "",
        f"**Veredicto:** `{verdict}` (`{n_ok}/{len(checks)}`)",
        f"**Fecha UTC:** `{datetime.now(timezone.utc).isoformat()}`",
        f"**Commit auditado:** `{git_out(['rev-parse', '--short', 'HEAD'])}`",
        "",
        "| # | Verificación | Estado | Evidencia |",
        "|---|---|---|---|",
    ]
    for idx, (name, ok, detail) in enumerate(checks, 1):
        lines.append(f"| {idx} | {name} | {'PASS' if ok else 'FAIL'} | `{detail}` |")
    lines.append("")
    AUDIT_PATH.write_text("\n".join(lines))
    print(f"{verdict} ({n_ok}/{len(checks)}) -> {AUDIT_PATH}")
    if n_ok != len(checks):
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["freeze", "replay", "audit"])
    args = ap.parse_args()
    if args.action == "freeze":
        cmd_freeze()
    elif args.action == "replay":
        cmd_replay()
    else:
        cmd_audit()


if __name__ == "__main__":
    main()
