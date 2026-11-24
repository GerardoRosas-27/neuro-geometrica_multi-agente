#!/usr/bin/env python3
"""Paired bootstrap (10,000 resamples, seed=12345, IC 95%) summary for exp/orbita-adaptive-spectral."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import statistics as st
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs" / "orbita_adaptive_spectral_v1"
RAW_DIR = DOCS_DIR / "raw"

DEV_SEEDS = [0xA400 + i for i in range(16)]
CONFIRM_SEEDS = [0xB400 + i for i in range(16)]


def boot(vals: list[float], n: int = 10000, seed: int = 12345) -> dict[str, float]:
    rng = random.Random(seed)
    k = len(vals)
    means = []
    for _ in range(n):
        means.append(sum(vals[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return {
        "mean": st.mean(vals),
        "ci95_lo": means[int(0.025 * n)],
        "ci95_hi": means[int(0.975 * n) - 1],
    }


def boot_diff(a: list[float], b: list[float]) -> dict[str, float]:
    return boot([x - y for x, y in zip(a, b)])


def load_phase(phase: str, seeds: list[int]) -> dict:
    sub = "dev" if phase == "DEV" else "confirm"
    o1, o2, o3, o4 = {}, {}, {}, {}
    for s in seeds:
        sdir = RAW_DIR / sub / f"seed_0x{s:X}"
        if not sdir.exists():
            continue
        for fn, target in [("metrics_o1.csv", o1), ("metrics_o2.csv", o2), ("metrics_o3.csv", o3)]:
            with (sdir / fn).open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    target.setdefault(row["arm"], []).append(row)
        with (sdir / "metrics_o4.csv").open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                o4.setdefault((row["subtask"], row["slice"]), []).append(row)
    return {"o1": o1, "o2": o2, "o3": o3, "o4": o4}


def col(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


def summarize_phase(phase: str, seeds: list[int]) -> dict | None:
    d = load_phase(phase, seeds)
    if not d["o1"]:
        return None
    n_seeds = len(d["o1"]["ORBITA_BASE"])
    print(f"\n================ {phase} (N={n_seeds} seeds, 10,000 paired bootstrap) ================")

    out: dict = {"phase": phase, "n_seeds": n_seeds, "suites": {"O1": {}, "O2": {}, "O3": {}, "O4": {}}, "criteria": {}}

    print("\n--- Suite O1: Oficial ÓRBITA §16 (TEST 256 + E33 Free-Run h1..h64) ---")
    for arm in ["ORBITA_BASE", "ORBITA_ASC", "ORBITA_RIEMANN_FIXED", "ORBITA_FOURIER_FIXED", "ORBITA_POISSON_FIXED", "ORBITA_MLP_RES"]:
        rs = d["o1"][arm]
        stats = {k: boot(col(rs, k)) for k in ["train_ms", "acc1", "rel1", "acc_indist", "acc_hole", "rel_refl", "rel_rad", "h1", "h4", "h8", "h16", "h32", "h64", "region", "init_r", "learned_r"]}
        stats["params"] = int(rs[0]["params"])
        stats["clean_cnt"] = sum(int(r["clean"]) for r in rs)
        stats["gate_cnt"] = sum(int(r["gate_pass"]) for r in rs)
        stats["fresh_cnt"] = sum(int(r["fresh_ok"]) for r in rs)
        out["suites"]["O1"][arm] = stats
        print(
            f"  {arm:21s} p={stats['params']:4d} ms={stats['train_ms']['mean']:5.1f}  "
            f"acc1={100*stats['acc1']['mean']:5.2f}%  rel1={stats['rel1']['mean']:.4f}  hole={100*stats['acc_hole']['mean']:5.2f}%  "
            f"h8={100*stats['h8']['mean']:5.2f}%  h16={100*stats['h16']['mean']:5.2f}% [{100*stats['h16']['ci95_lo']:5.2f}%, {100*stats['h16']['ci95_hi']:5.2f}%]  "
            f"h32={100*stats['h32']['mean']:5.2f}%  gate={stats['gate_cnt']}/{n_seeds}  <r>={stats['init_r']['mean']:.2f}->{stats['learned_r']['mean']:.2f}"
        )

    print("\n--- Suite O2: Abducción de Contexto Desconocido & 40 Pares Compuestos Inéditos (ÓRBITA) ---")
    for arm in ["ORBITA_ASC", "ORBITA_RIEMANN_FIXED", "ORBITA_FOURIER_FIXED", "ORBITA_POISSON_FIXED", "ORBITA_MLP"]:
        rs = d["o2"][arm]
        stats = {k: boot(col(rs, k)) for k in ["train_ms", "unseen_pair_acc", "train_pair_acc", "cfr", "init_r", "learned_r", "psr"]}
        stats["params"] = int(rs[0]["params"])
        out["suites"]["O2"][arm] = stats
        print(
            f"  {arm:21s} p={stats['params']:4d}  unseen={100*stats['unseen_pair_acc']['mean']:5.2f}% "
            f"[{100*stats['unseen_pair_acc']['ci95_lo']:5.2f}%, {100*stats['unseen_pair_acc']['ci95_hi']:5.2f}%]  "
            f"train={100*stats['train_pair_acc']['mean']:5.1f}%  cfr={100*stats['cfr']['mean']:5.1f}%  "
            f"<r>={stats['init_r']['mean']:.3f}->{stats['learned_r']['mean']:.3f}  psr={stats['psr']['mean']:.2f}"
        )

    print("\n--- Suite O3: Aprendizaje Continuo Few-Shot de 4 Familias Nuevas No Lineales (8..11) ---")
    for arm in ["ORBITA_BASE", "ORBITA_OBS", "ORBITA_LAT", "ORBITA_ASC", "ORBITA_RIEMANN_FIXED", "ORBITA_FOURIER_FIXED", "ORBITA_POISSON_FIXED", "ORBITA_MLP"]:
        rs = d["o3"][arm]
        stats = {k: boot(col(rs, k)) for k in ["adapt_ms", "known_h8", "known_h16", "new_h1", "new_h8", "new_h16", "new_rel16", "route_purity"]}
        stats["expert_params"] = int(rs[0]["expert_params"])
        out["suites"]["O3"][arm] = stats
        print(
            f"  {arm:21s} ep={stats['expert_params']:4d} ms={stats['adapt_ms']['mean']:6.1f}  "
            f"known16={100*stats['known_h16']['mean']:5.2f}%  new_h1={100*stats['new_h1']['mean']:5.2f}%  "
            f"new_h8={100*stats['new_h8']['mean']:5.2f}%  new_h16={100*stats['new_h16']['mean']:5.2f}% "
            f"[{100*stats['new_h16']['ci95_lo']:5.2f}%, {100*stats['new_h16']['ci95_hi']:5.2f}%]  "
            f"rel16={stats['new_rel16']['mean']:.3f}  pur={100*stats['route_purity']['mean']:5.1f}%"
        )

    print("\n--- Suite O4: Tamiz Espectral de Números Primos (¿Qué Inferencia Se Esconde?) ---")
    slices = ["full_adaptive", "prime_harmonic_exact", "riemann_prime_dual", "log_prime_exact", "composite_only", "even_only", "random_matched"]
    out["suites"]["O4"]["O4a_dynamics"] = {}
    out["suites"]["O4"]["O4b_collapse"] = {}
    print("  [O4a Dinámica ÓRBITA 12 familias]")
    for sl in slices:
        rs = d["o4"][("O4a_dynamics", sl)]
        stats = {k: boot(col(rs, k)) for k in ["acc1", "acc16_or_pair", "rel1_or_ghost", "disp_cos_or_psr", "manifold_dist", "odd_parity_frac", "irred_rel1", "comp_rel1", "energy_retained"]}
        out["suites"]["O4"]["O4a_dynamics"][sl] = stats
        print(
            f"    {sl:21s} acc1={100*stats['acc1']['mean']:5.2f}%  acc16={100*stats['acc16_or_pair']['mean']:5.2f}%  "
            f"rel1={stats['rel1_or_ghost']['mean']:.4f}  cos={stats['disp_cos_or_psr']['mean']:.4f}  "
            f"odd={100*stats['odd_parity_frac']['mean']:5.2f}%  irred_rel={stats['irred_rel1']['mean']:.4f}  "
            f"comp_rel={stats['comp_rel1']['mean']:.4f}  E={100*stats['energy_retained']['mean']:5.1f}%"
        )
    print("  [O4b Colapso Pasado x Futuro (40 pares inéditos)]")
    for sl in slices:
        rs = d["o4"][("O4b_collapse", sl)]
        stats = {k: boot(col(rs, k)) for k in ["acc16_or_pair", "rel1_or_ghost", "disp_cos_or_psr", "energy_retained"]}
        out["suites"]["O4"]["O4b_collapse"][sl] = stats
        print(
            f"    {sl:21s} pair_acc={100*stats['acc16_or_pair']['mean']:5.2f}% "
            f"[{100*stats['acc16_or_pair']['ci95_lo']:5.2f}%, {100*stats['acc16_or_pair']['ci95_hi']:5.2f}%]  "
            f"PSR={stats['disp_cos_or_psr']['mean']:5.2f} [{stats['disp_cos_or_psr']['ci95_lo']:4.2f}, {stats['disp_cos_or_psr']['ci95_hi']:4.2f}]  "
            f"ghost={stats['rel1_or_ghost']['mean']:.4f}  E={100*stats['energy_retained']['mean']:5.1f}%"
        )

    # ── Evaluate the 8 exact preregistered criteria H_O1a .. H_O4b ──
    c_rel1_diff = boot_diff(col(d["o1"]["ORBITA_ASC"], "rel1"), col(d["o1"]["ORBITA_BASE"], "rel1"))
    c_acc1_diff = boot_diff(col(d["o1"]["ORBITA_ASC"], "acc1"), col(d["o1"]["ORBITA_BASE"], "acc1"))
    h_o1a = (c_rel1_diff["ci95_hi"] < -0.0005) and (c_acc1_diff["ci95_lo"] > -0.005) and (out["suites"]["O1"]["ORBITA_ASC"]["clean_cnt"] == n_seeds)

    c_h16_vs_base = boot_diff(col(d["o1"]["ORBITA_ASC"], "h16"), col(d["o1"]["ORBITA_BASE"], "h16"))
    h_o1b = (c_h16_vs_base["ci95_lo"] > 0.03) and (out["suites"]["O1"]["ORBITA_ASC"]["gate_cnt"] == n_seeds)

    c_h16_vs_four = boot_diff(col(d["o1"]["ORBITA_ASC"], "h16"), col(d["o1"]["ORBITA_FOURIER_FIXED"], "h16"))
    c_h16_vs_pois = boot_diff(col(d["o1"]["ORBITA_ASC"], "h16"), col(d["o1"]["ORBITA_POISSON_FIXED"], "h16"))
    c_h16_vs_mlp = boot_diff(col(d["o1"]["ORBITA_ASC"], "h16"), col(d["o1"]["ORBITA_MLP_RES"], "h16"))
    h_o1c = (c_h16_vs_four["ci95_lo"] > 0.02) and (c_h16_vs_pois["ci95_lo"] > 0.02)

    o2_vs_p = boot_diff(col(d["o2"]["ORBITA_ASC"], "unseen_pair_acc"), col(d["o2"]["ORBITA_POISSON_FIXED"], "unseen_pair_acc"))
    o2_vs_f = boot_diff(col(d["o2"]["ORBITA_ASC"], "unseen_pair_acc"), col(d["o2"]["ORBITA_FOURIER_FIXED"], "unseen_pair_acc"))
    o2_vs_m = boot_diff(col(d["o2"]["ORBITA_ASC"], "unseen_pair_acc"), col(d["o2"]["ORBITA_MLP"], "unseen_pair_acc"))
    h_o2a = (o2_vs_p["ci95_lo"] > 0.05) and (o2_vs_f["ci95_lo"] > 0.05) and (o2_vs_m["ci95_lo"] > 0.05)

    o3_vs_lat = boot_diff(col(d["o3"]["ORBITA_ASC"], "new_h16"), col(d["o3"]["ORBITA_LAT"], "new_h16"))
    o3_known_vs_base = boot_diff(col(d["o3"]["ORBITA_ASC"], "known_h16"), col(d["o3"]["ORBITA_BASE"], "known_h16"))
    h_o3a = (o3_vs_lat["ci95_lo"] > 0.10) and (o3_known_vs_base["ci95_lo"] > -0.02)

    o3_vs_four = boot_diff(col(d["o3"]["ORBITA_ASC"], "new_h16"), col(d["o3"]["ORBITA_FOURIER_FIXED"], "new_h16"))
    o3_vs_pois = boot_diff(col(d["o3"]["ORBITA_ASC"], "new_h16"), col(d["o3"]["ORBITA_POISSON_FIXED"], "new_h16"))
    o3_vs_mlp = boot_diff(col(d["o3"]["ORBITA_ASC"], "new_h16"), col(d["o3"]["ORBITA_MLP"], "new_h16"))
    h_o3b = (o3_vs_four["ci95_lo"] > 0.03) and (o3_vs_pois["ci95_lo"] > 0.03)

    o4b_psr_diff = boot_diff(col(d["o4"][("O4b_collapse", "prime_harmonic_exact")], "disp_cos_or_psr"), col(d["o4"][("O4b_collapse", "composite_only")], "disp_cos_or_psr"))
    o4b_acc_diff = boot_diff(col(d["o4"][("O4b_collapse", "prime_harmonic_exact")], "acc16_or_pair"), col(d["o4"][("O4b_collapse", "composite_only")], "acc16_or_pair"))
    h_o4a = (o4b_psr_diff["ci95_lo"] > 0.05) and (o4b_acc_diff["ci95_lo"] > 0.03)

    o4a_odd_diff = boot_diff(col(d["o4"][("O4a_dynamics", "prime_harmonic_exact")], "odd_parity_frac"), col(d["o4"][("O4a_dynamics", "composite_only")], "odd_parity_frac"))
    o4a_acc_vs_full = boot_diff(col(d["o4"][("O4a_dynamics", "prime_harmonic_exact")], "acc1"), col(d["o4"][("O4a_dynamics", "full_adaptive")], "acc1"))
    h_o4b = (o4a_odd_diff["ci95_lo"] > 0.15) and (o4a_acc_vs_full["ci95_hi"] < 0.0)

    out["criteria"] = {
        "H_O1a": {"pass": h_o1a, "rel1_diff": c_rel1_diff, "acc1_diff": c_acc1_diff},
        "H_O1b": {"pass": h_o1b, "h16_vs_base": c_h16_vs_base},
        "H_O1c": {"pass": h_o1c, "h16_vs_fourier": c_h16_vs_four, "h16_vs_poisson": c_h16_vs_pois, "h16_vs_mlp_info": c_h16_vs_mlp},
        "H_O2a": {"pass": h_o2a, "vs_poisson": o2_vs_p, "vs_fourier": o2_vs_f, "vs_mlp": o2_vs_m},
        "H_O3a": {"pass": h_o3a, "new_h16_vs_lat": o3_vs_lat, "known_h16_vs_base": o3_known_vs_base},
        "H_O3b": {"pass": h_o3b, "new_h16_vs_fourier": o3_vs_four, "new_h16_vs_poisson": o3_vs_pois, "new_h16_vs_mlp_info": o3_vs_mlp},
        "H_O4a": {"pass": h_o4a, "psr_prime_vs_comp": o4b_psr_diff, "acc_prime_vs_comp": o4b_acc_diff},
        "H_O4b": {"pass": h_o4b, "odd_prime_vs_comp": o4a_odd_diff, "acc1_prime_vs_full": o4a_acc_vs_full},
    }

    print("\n--- Veredicto de los 8 Criterios Preregistrados ---")
    n_pass = 0
    for k, v in out["criteria"].items():
        if v["pass"]:
            n_pass += 1
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}")
    print(f"TOTAL {phase}: {n_pass}/8 PASS")
    return out


def main() -> int:
    res = {}
    for ph, seeds in [("DEV", DEV_SEEDS), ("CONFIRM", CONFIRM_SEEDS)]:
        s = summarize_phase(ph, seeds)
        if s is not None:
            res[ph] = s
    out_p = DOCS_DIR / "summary_confirm.json"
    out_p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved structured summary to {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
