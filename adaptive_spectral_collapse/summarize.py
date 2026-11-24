#!/usr/bin/env python3
"""Paired bootstrap (10,000 resamples, seed=12345) and evaluation of all 8
preregistered criteria (`H_A1a`–`H_A4a`) from `PLAN.md` (stdlib only).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DOCS_DIR = REPO_ROOT / "docs" / "adaptive_spectral_collapse_v1"
RAW_DIR = DOCS_DIR / "raw"

DEV_SEEDS = [f"0x{0xC100 + i:X}" for i in range(16)]
CONFIRM_SEEDS = [f"0x{0xD100 + i:X}" for i in range(16)]

ARMS_7 = [
    "adaptive_spectral",
    "riemann_fixed",
    "dyson_gue_fixed",
    "qmc_halton_fixed",
    "fourier_fixed",
    "poisson_fixed",
    "mlp_conventional",
]

N_BOOT = 10_000
BOOT_SEED = 12345


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def boot_ci(xs: list[float], seed: int = BOOT_SEED) -> dict[str, float]:
    n = len(xs)
    rng = random.Random(seed)
    means = []
    for _ in range(N_BOOT):
        s = 0.0
        for _ in range(n):
            s += xs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return {
        "mean": mean(xs),
        "ci95_lo": means[int(0.025 * N_BOOT)],
        "ci95_hi": means[int(0.975 * N_BOOT)],
    }


def paired_diff_ci(a: list[float], b: list[float], seed: int = BOOT_SEED) -> dict[str, float]:
    assert len(a) == len(b)
    return boot_ci([x - y for x, y in zip(a, b)], seed=seed)


def load_phase(phase: str, seeds: list[str]) -> dict[str, list[dict]]:
    phase_dir = RAW_DIR / phase.lower()
    tables = {"a1": [], "a2": [], "a3": [], "a4": []}
    for seed in seeds:
        sdir = phase_dir / f"seed_{seed}"
        for key in tables:
            with (sdir / f"metrics_{key}.csv").open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    tables[key].append(row)
    return tables


def series_by_seed(rows: list[dict], seeds: list[str], pred, col: str) -> list[float]:
    m = {}
    for r in rows:
        if pred(r):
            m[r["seed"]] = float(r[col])
    return [m[s] for s in seeds]


def fmt(st: dict[str, float], pct: bool = False) -> str:
    if pct:
        return f"{100*st['mean']:.2f}% [{100*st['ci95_lo']:.2f}%, {100*st['ci95_hi']:.2f}%]"
    return f"{st['mean']:.4f} [{st['ci95_lo']:.4f}, {st['ci95_hi']:.4f}]"


def evaluate_phase(phase: str, seeds: list[str]) -> dict:
    data = load_phase(phase, seeds)
    summary: dict = {"phase": phase, "n_seeds": len(seeds), "suites": {}, "criteria": {}}

    # --- Suite A1 ---
    a1 = {}
    for arm in ARMS_7:
        acc_u = series_by_seed(data["a1"], seeds, lambda r, a=arm: r["arm"] == a, "unseen_abduction_acc")
        f1_u = series_by_seed(data["a1"], seeds, lambda r, a=arm: r["arm"] == a, "unseen_abduction_f1")
        acc_t = series_by_seed(data["a1"], seeds, lambda r, a=arm: r["arm"] == a, "train_abduction_acc")
        cfr = series_by_seed(data["a1"], seeds, lambda r, a=arm: r["arm"] == a, "collision_free_rate")
        r0 = series_by_seed(data["a1"], seeds, lambda r, a=arm: r["arm"] == a, "init_spacing_r")
        r1 = series_by_seed(data["a1"], seeds, lambda r, a=arm: r["arm"] == a, "learned_spacing_r")
        snr = series_by_seed(data["a1"], seeds, lambda r, a=arm: r["arm"] == a, "field_contrast_snr")
        tms = series_by_seed(data["a1"], seeds, lambda r, a=arm: r["arm"] == a, "train_ms")
        npar = series_by_seed(data["a1"], seeds, lambda r, a=arm: r["arm"] == a, "n_params")
        a1[arm] = {
            "n_params": int(npar[0]),
            "train_ms": boot_ci(tms),
            "unseen_abduction_acc": boot_ci(acc_u),
            "unseen_abduction_f1": boot_ci(f1_u),
            "train_abduction_acc": boot_ci(acc_t),
            "collision_free_rate": boot_ci(cfr),
            "init_spacing_r": boot_ci(r0),
            "learned_spacing_r": boot_ci(r1),
            "field_contrast_snr": boot_ci(snr),
            "_raw_acc": acc_u,
        }
    summary["suites"]["A1"] = a1

    # --- Suite A2 ---
    a2 = {}
    for arm in ARMS_7:
        rmse = series_by_seed(data["a2"], seeds, lambda r, a=arm: r["arm"] == a, "unseen_pred_rmse")
        cls = series_by_seed(data["a2"], seeds, lambda r, a=arm: r["arm"] == a, "unseen_class_acc")
        cf_a = series_by_seed(data["a2"], seeds, lambda r, a=arm: r["arm"] == a, "cf_exact_acc")
        cf_r = series_by_seed(data["a2"], seeds, lambda r, a=arm: r["arm"] == a, "cf_pred_rmse")
        a2[arm] = {
            "unseen_pred_rmse": boot_ci(rmse),
            "unseen_class_acc": boot_ci(cls),
            "cf_exact_acc": boot_ci(cf_a),
            "cf_pred_rmse": boot_ci(cf_r),
            "_raw_rmse": rmse,
            "_raw_cls": cls,
            "_raw_cf": cf_a,
        }
    summary["suites"]["A2"] = a2

    # --- Suite A3 ---
    a3_an = {}
    for arm in ARMS_7:
        acc = series_by_seed(data["a3"], seeds, lambda r, a=arm: r["subtask"] == "analogy" and r["arm"] == a, "exact_acc")
        f1 = series_by_seed(data["a3"], seeds, lambda r, a=arm: r["subtask"] == "analogy" and r["arm"] == a, "partial_f1")
        rm = series_by_seed(data["a3"], seeds, lambda r, a=arm: r["subtask"] == "analogy" and r["arm"] == a, "pred_rmse")
        a3_an[arm] = {
            "exact_acc": boot_ci(acc),
            "partial_f1": boot_ci(f1),
            "pred_rmse": boot_ci(rm),
            "_raw_acc": acc,
        }

    a3_noise = {}
    for arm in ["adaptive_spectral", "riemann_fixed", "fourier_fixed", "mlp_conventional"]:
        a3_noise[arm] = {}
        for sig in ["0.0000", "0.0250", "0.0500", "0.1000"]:
            acc = series_by_seed(
                data["a3"],
                seeds,
                lambda r, a=arm, s=sig: r["subtask"] == "noise_sweep" and r["arm"] == a and r["sigma_x"] == s,
                "exact_acc",
            )
            rm = series_by_seed(
                data["a3"],
                seeds,
                lambda r, a=arm, s=sig: r["subtask"] == "noise_sweep" and r["arm"] == a and r["sigma_x"] == s,
                "pred_rmse",
            )
            a3_noise[arm][sig] = {"exact_acc": boot_ci(acc), "pred_rmse": boot_ci(rm), "_raw_acc": acc}
    summary["suites"]["A3"] = {"analogy": a3_an, "noise_sweep": a3_noise}

    # --- Suite A4 ---
    a4 = {}
    for arm in ARMS_7:
        ams = series_by_seed(data["a4"], seeds, lambda r, a=arm: r["arm"] == a, "adapt_ms")
        oa = series_by_seed(data["a4"], seeds, lambda r, a=arm: r["arm"] == a, "old_unseen_acc")
        or_ = series_by_seed(data["a4"], seeds, lambda r, a=arm: r["arm"] == a, "old_unseen_rmse")
        noa = series_by_seed(data["a4"], seeds, lambda r, a=arm: r["arm"] == a, "novel_old_acc")
        nor = series_by_seed(data["a4"], seeds, lambda r, a=arm: r["arm"] == a, "novel_old_rmse")
        nna = series_by_seed(data["a4"], seeds, lambda r, a=arm: r["arm"] == a, "novel_novel_acc")
        nnr = series_by_seed(data["a4"], seeds, lambda r, a=arm: r["arm"] == a, "novel_novel_rmse")
        naa = series_by_seed(data["a4"], seeds, lambda r, a=arm: r["arm"] == a, "novel_all_acc")
        nar = series_by_seed(data["a4"], seeds, lambda r, a=arm: r["arm"] == a, "novel_all_rmse")
        a4[arm] = {
            "adapt_ms": boot_ci(ams),
            "old_unseen_acc": boot_ci(oa),
            "old_unseen_rmse": boot_ci(or_),
            "novel_old_acc": boot_ci(noa),
            "novel_old_rmse": boot_ci(nor),
            "novel_novel_acc": boot_ci(nna),
            "novel_novel_rmse": boot_ci(nnr),
            "novel_all_acc": boot_ci(naa),
            "novel_all_rmse": boot_ci(nar),
            "_raw_old_acc": oa,
            "_raw_nov_acc": naa,
            "_raw_nov_rmse": nar,
        }
    summary["suites"]["A4"] = a4

    # --- Evaluate all 8 Preregistered Criteria (PLAN.md §4) ---
    d_a1_pois = paired_diff_ci(a1["adaptive_spectral"]["_raw_acc"], a1["poisson_fixed"]["_raw_acc"])
    d_a1_four = paired_diff_ci(a1["adaptive_spectral"]["_raw_acc"], a1["fourier_fixed"]["_raw_acc"])
    h_a1a = (
        a1["adaptive_spectral"]["unseen_abduction_acc"]["ci95_lo"] > 0.80
        and d_a1_pois["ci95_lo"] > 0.05
        and d_a1_four["ci95_lo"] > 0.05
    )

    d_a1_riem = paired_diff_ci(a1["adaptive_spectral"]["_raw_acc"], a1["riemann_fixed"]["_raw_acc"])
    d_a1_dys = paired_diff_ci(a1["adaptive_spectral"]["_raw_acc"], a1["dyson_gue_fixed"]["_raw_acc"])
    d_a1_qmc = paired_diff_ci(a1["adaptive_spectral"]["_raw_acc"], a1["qmc_halton_fixed"]["_raw_acc"])
    h_a1b = (
        d_a1_riem["ci95_lo"] > -0.02
        and d_a1_dys["ci95_lo"] > -0.02
        and d_a1_qmc["ci95_lo"] > -0.02
    )

    d_a1_mlp = paired_diff_ci(a1["adaptive_spectral"]["_raw_acc"], a1["mlp_conventional"]["_raw_acc"])
    h_a1c = d_a1_mlp["ci95_lo"] > 0.05

    d_a2_rmse_mlp = paired_diff_ci(a2["adaptive_spectral"]["_raw_rmse"], a2["mlp_conventional"]["_raw_rmse"])
    d_a2_cls_mlp = paired_diff_ci(a2["adaptive_spectral"]["_raw_cls"], a2["mlp_conventional"]["_raw_cls"])
    h_a2a = d_a2_rmse_mlp["ci95_hi"] < -0.10 and d_a2_cls_mlp["ci95_lo"] > 0.05

    d_a2_cf_mlp = paired_diff_ci(a2["adaptive_spectral"]["_raw_cf"], a2["mlp_conventional"]["_raw_cf"])
    d_a2_cf_four = paired_diff_ci(a2["adaptive_spectral"]["_raw_cf"], a2["fourier_fixed"]["_raw_cf"])
    h_a2b = (
        a2["adaptive_spectral"]["cf_exact_acc"]["ci95_lo"] > 0.75
        and d_a2_cf_mlp["ci95_lo"] > 0.10
        and d_a2_cf_four["ci95_lo"] > 0.05
    )

    d_a3_an_mlp = paired_diff_ci(a3_an["adaptive_spectral"]["_raw_acc"], a3_an["mlp_conventional"]["_raw_acc"])
    d_a3_an_pois = paired_diff_ci(a3_an["adaptive_spectral"]["_raw_acc"], a3_an["poisson_fixed"]["_raw_acc"])
    h_a3a = (
        a3_an["adaptive_spectral"]["exact_acc"]["ci95_lo"] > 0.75
        and d_a3_an_mlp["ci95_lo"] > 0.10
        and d_a3_an_pois["ci95_lo"] > 0.05
    )

    d_a3_noise = paired_diff_ci(
        a3_noise["adaptive_spectral"]["0.1000"]["_raw_acc"],
        a3_noise["adaptive_spectral"]["0.0000"]["_raw_acc"],
    )
    h_a3b = d_a3_noise["ci95_hi"] < -0.20

    d_a4_nov_acc = paired_diff_ci(a4["adaptive_spectral"]["_raw_nov_acc"], a4["mlp_conventional"]["_raw_nov_acc"])
    d_a4_nov_rmse = paired_diff_ci(a4["adaptive_spectral"]["_raw_nov_rmse"], a4["mlp_conventional"]["_raw_nov_rmse"])
    h_a4a = (
        d_a4_nov_acc["ci95_lo"] > 0.15
        and d_a4_nov_rmse["ci95_hi"] < -0.10
        and a4["adaptive_spectral"]["old_unseen_acc"]["ci95_lo"] > 0.75
    )

    summary["criteria"] = {
        "H_A1a": {"pass": h_a1a, "acc_adapt": a1["adaptive_spectral"]["unseen_abduction_acc"], "diff_vs_poisson": d_a1_pois, "diff_vs_fourier": d_a1_four},
        "H_A1b": {"pass": h_a1b, "diff_vs_riemann": d_a1_riem, "diff_vs_dyson": d_a1_dys, "diff_vs_qmc": d_a1_qmc},
        "H_A1c": {"pass": h_a1c, "diff_vs_mlp": d_a1_mlp},
        "H_A2a": {"pass": h_a2a, "rmse_diff_vs_mlp": d_a2_rmse_mlp, "cls_diff_vs_mlp": d_a2_cls_mlp},
        "H_A2b": {"pass": h_a2b, "cf_adapt": a2["adaptive_spectral"]["cf_exact_acc"], "cf_diff_vs_mlp": d_a2_cf_mlp, "cf_diff_vs_fourier": d_a2_cf_four},
        "H_A3a": {"pass": h_a3a, "an_adapt": a3_an["adaptive_spectral"]["exact_acc"], "an_diff_vs_mlp": d_a3_an_mlp, "an_diff_vs_poisson": d_a3_an_pois},
        "H_A3b": {"pass": h_a3b, "diff_0100_vs_0000": d_a3_noise},
        "H_A4a": {"pass": h_a4a, "nov_acc_diff_vs_mlp": d_a4_nov_acc, "nov_rmse_diff_vs_mlp": d_a4_nov_rmse, "old_ret_adapt": a4["adaptive_spectral"]["old_unseen_acc"]},
    }

    def strip_raw(obj):
        if isinstance(obj, dict):
            return {k: strip_raw(v) for k, v in obj.items() if not k.startswith("_raw")}
        return obj

    return strip_raw(summary)


def print_report(s: dict) -> None:
    print(f"\n================ {s['phase']} (N={s['n_seeds']} seeds, 10,000 paired bootstrap) ================")
    print("\n--- Suite A1: Entrada Desconocida, Autoorganización y Abducción de 166 Pares Nunca Vistos ---")
    for arm in ARMS_7:
        st = s["suites"]["A1"][arm]
        print(
            f"  {arm:20s} params={st['n_params']:4d} ms={st['train_ms']['mean']:6.1f}  "
            f"unseen={fmt(st['unseen_abduction_acc'], True)}  train={100*st['train_abduction_acc']['mean']:5.1f}%  "
            f"cfr={100*st['collision_free_rate']['mean']:5.1f}%  <r>={st['init_spacing_r']['mean']:.3f}->{st['learned_spacing_r']['mean']:.3f}  "
            f"snr={st['field_contrast_snr']['mean']:.2f}"
        )

    print("\n--- Suite A2: Predicción No Lineal 'Del Otro Lado' + Contrafactual do(c_a->c_new) [166 Pares Nunca Vistos] ---")
    for arm in ARMS_7:
        st = s["suites"]["A2"][arm]
        print(
            f"  {arm:20s} rmse={fmt(st['unseen_pred_rmse'])}  cls4={fmt(st['unseen_class_acc'], True)}  "
            f"cf_acc={fmt(st['cf_exact_acc'], True)}  cf_rmse={fmt(st['cf_pred_rmse'])}"
        )

    print("\n--- Suite A3a: Analogía Composicional A:B::C:? [100 Destinos Nunca Vistos] ---")
    for arm in ARMS_7:
        st = s["suites"]["A3"]["analogy"][arm]
        print(f"  {arm:20s} exact={fmt(st['exact_acc'], True)}  f1={fmt(st['partial_f1'], True)}  rmse={fmt(st['pred_rmse'])}")

    print("\n--- Suite A3b: Estrés de Ruido Sensorial sigma_x (166 Pares Nunca Vistos) ---")
    for arm, d in s["suites"]["A3"]["noise_sweep"].items():
        vals = "  ".join(f"s={float(sig):.3f}:{100*d[sig]['exact_acc']['mean']:5.1f}%" for sig in ["0.0000", "0.0250", "0.0500", "0.1000"])
        print(f"  {arm:20s} {vals}")

    print("\n--- Suite A4: Aprendizaje Continuo de 4 Conceptos Nuevos (K: 24 -> 28, Few-Shot) ---")
    for arm in ARMS_7:
        st = s["suites"]["A4"][arm]
        print(
            f"  {arm:20s} old_ret={fmt(st['old_unseen_acc'], True)}  "
            f"nov_old={100*st['novel_old_acc']['mean']:5.1f}%  nov_nov={100*st['novel_novel_acc']['mean']:5.1f}%  "
            f"nov_all={fmt(st['novel_all_acc'], True)}  nov_rmse={fmt(st['novel_all_rmse'])}"
        )

    print("\n--- Veredicto de los 8 Criterios Preregistrados ---")
    passed = 0
    for cid, c in s["criteria"].items():
        if c["pass"]:
            passed += 1
        print(f"  {cid}: {'PASS' if c['pass'] else 'FAIL'}")
    print(f"TOTAL {s['phase']}: {passed}/{len(s['criteria'])} PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()

    dev_s = evaluate_phase("DEV", DEV_SEEDS)
    print_report(dev_s)
    if not args.dev_only:
        conf_s = evaluate_phase("CONFIRM", CONFIRM_SEEDS)
        print_report(conf_s)
        out = DOCS_DIR / "summary_confirm.json"
        out.write_text(json.dumps({"DEV": dev_s, "CONFIRM": conf_s}, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved structured summary to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
