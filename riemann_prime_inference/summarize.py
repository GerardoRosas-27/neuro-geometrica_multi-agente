#!/usr/bin/env python3
"""Paired bootstrap (10,000 resamples, seed=12345) and evaluation of all 8
preregistered criteria (`H_I1a`–`H_I4c`) from `PLAN.md` (stdlib only).
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
DOCS_DIR = REPO_ROOT / "docs" / "riemann_prime_inference_v1"
RAW_DIR = DOCS_DIR / "raw"

DEV_SEEDS = [f"0x{0xA900 + i:X}" for i in range(16)]
CONFIRM_SEEDS = [f"0x{0xB900 + i:X}" for i in range(16)]

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
    lo = means[int(0.025 * N_BOOT)]
    hi = means[int(0.975 * N_BOOT)]
    return {"mean": mean(xs), "ci95_lo": lo, "ci95_hi": hi}


def paired_diff_ci(a: list[float], b: list[float], seed: int = BOOT_SEED) -> dict[str, float]:
    assert len(a) == len(b)
    diffs = [x - y for x, y in zip(a, b)]
    return boot_ci(diffs, seed=seed)


def load_phase(phase: str, seeds: list[str]) -> dict[str, list[dict]]:
    phase_dir = RAW_DIR / phase.lower()
    tables = {"i1": [], "i2": [], "i3": [], "i4": []}
    for seed in seeds:
        sdir = phase_dir / f"seed_{seed}"
        for key in tables:
            p = sdir / f"metrics_{key}.csv"
            with p.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    tables[key].append(row)
    return tables


def series_by_seed(
    rows: list[dict], seeds: list[str], predicate, col: str
) -> list[float]:
    by_seed = {}
    for row in rows:
        if predicate(row):
            by_seed[row["seed"]] = float(row[col])
    return [by_seed[s] for s in seeds]


def fmt_stat(st: dict[str, float], pct: bool = False) -> str:
    if pct:
        return f"{100*st['mean']:.2f}% [{100*st['ci95_lo']:.2f}%, {100*st['ci95_hi']:.2f}%]"
    return f"{st['mean']:.4f} [{st['ci95_lo']:.4f}, {st['ci95_hi']:.4f}]"


def evaluate_phase(phase: str, seeds: list[str]) -> dict:
    data = load_phase(phase, seeds)
    arms_4 = [
        "riemann_prime_collapse",
        "random_collapse",
        "dyson_gue_collapse",
        "fourier_linear_collapse",
    ]
    arms_i3 = [
        "riemann_bayes_collapse",
        "riemann_uniform_collapse",
        "random_bayes_collapse",
        "prior_only_ctrl",
    ]

    summary: dict = {"phase": phase, "n_seeds": len(seeds), "suites": {}, "criteria": {}}

    # --- Suite I1 ---
    i1_stats: dict = {}
    for sig in ["0.0000", "0.0020", "0.0060"]:
        i1_stats[sig] = {}
        for arm in arms_4:
            acc = series_by_seed(
                data["i1"], seeds, lambda r, a=arm, s=sig: r["arm"] == a and r["sigma"] == s, "origin_exact_acc"
            )
            f1 = series_by_seed(
                data["i1"], seeds, lambda r, a=arm, s=sig: r["arm"] == a and r["sigma"] == s, "origin_partial_f1"
            )
            snr = series_by_seed(
                data["i1"], seeds, lambda r, a=arm, s=sig: r["arm"] == a and r["sigma"] == s, "peak_margin_snr"
            )
            i1_stats[sig][arm] = {
                "origin_exact_acc": boot_ci(acc),
                "origin_partial_f1": boot_ci(f1),
                "peak_margin_snr": boot_ci(snr),
                "_raw_acc": acc,
            }
    summary["suites"]["I1"] = i1_stats

    # --- Suite I2 ---
    i2_stats: dict = {}
    for arm in arms_4:
        acc = series_by_seed(data["i2"], seeds, lambda r, a=arm: r["arm"] == a, "origin_exact_acc")
        rmse = series_by_seed(data["i2"], seeds, lambda r, a=arm: r["arm"] == a, "pred_rmse")
        cls = series_by_seed(data["i2"], seeds, lambda r, a=arm: r["arm"] == a, "class_acc")
        cf_acc = series_by_seed(data["i2"], seeds, lambda r, a=arm: r["arm"] == a, "cf_exact_acc")
        cf_rmse = series_by_seed(data["i2"], seeds, lambda r, a=arm: r["arm"] == a, "cf_rmse")
        i2_stats[arm] = {
            "origin_exact_acc": boot_ci(acc),
            "pred_rmse": boot_ci(rmse),
            "class_acc": boot_ci(cls),
            "cf_exact_acc": boot_ci(cf_acc),
            "cf_rmse": boot_ci(cf_rmse),
            "_raw_rmse": rmse,
            "_raw_cf_acc": cf_acc,
        }
    summary["suites"]["I2"] = i2_stats

    # --- Suite I3 ---
    i3_stats: dict = {}
    for arm in arms_i3:
        acc = series_by_seed(data["i3"], seeds, lambda r, a=arm: r["arm"] == a, "origin_exact_acc")
        f1 = series_by_seed(data["i3"], seeds, lambda r, a=arm: r["arm"] == a, "origin_partial_f1")
        i3_stats[arm] = {
            "origin_exact_acc": boot_ci(acc),
            "origin_partial_f1": boot_ci(f1),
            "_raw_acc": acc,
        }
    summary["suites"]["I3"] = i3_stats

    # --- Suite I4 ---
    i4_analogy: dict = {}
    for arm in arms_4:
        acc = series_by_seed(
            data["i4"], seeds, lambda r, a=arm: r["subtask"] == "analogy" and r["arm"] == a, "exact_acc"
        )
        f1 = series_by_seed(
            data["i4"], seeds, lambda r, a=arm: r["subtask"] == "analogy" and r["arm"] == a, "aux_metric"
        )
        i4_analogy[arm] = {"exact_acc": boot_ci(acc), "partial_f1": boot_ci(f1), "_raw_acc": acc}

    i4_noise: dict = {}
    for sig in ["0.0000", "0.0010", "0.0030", "0.0080", "0.0150", "0.0300"]:
        acc = series_by_seed(
            data["i4"],
            seeds,
            lambda r, s=sig: r["subtask"] == "noise_sweep" and r["param_value"] == s,
            "exact_acc",
        )
        snr = series_by_seed(
            data["i4"],
            seeds,
            lambda r, s=sig: r["subtask"] == "noise_sweep" and r["param_value"] == s,
            "aux_metric",
        )
        i4_noise[sig] = {"exact_acc": boot_ci(acc), "peak_margin_snr": boot_ci(snr), "_raw_acc": acc}

    i4_cap: dict = {}
    for kv in ["8", "12", "18"]:
        acc = series_by_seed(
            data["i4"],
            seeds,
            lambda r, k=kv: r["subtask"] == "capacity_sweep" and r["param_value"] == k,
            "exact_acc",
        )
        gap = series_by_seed(
            data["i4"],
            seeds,
            lambda r, k=kv: r["subtask"] == "capacity_sweep" and r["param_value"] == k,
            "aux_metric",
        )
        i4_cap[kv] = {
            "exact_acc": boot_ci(acc),
            "min_log_gap": gap[0],
            "_raw_acc": acc,
        }

    summary["suites"]["I4"] = {
        "analogy": i4_analogy,
        "noise_sweep": i4_noise,
        "capacity_sweep": i4_cap,
    }

    # --- Evaluate all 8 Preregistered Criteria (PLAN.md §3) ---
    r_i1 = i1_stats["0.0020"]["riemann_prime_collapse"]["_raw_acc"]
    rand_i1 = i1_stats["0.0020"]["random_collapse"]["_raw_acc"]
    dys_i1 = i1_stats["0.0020"]["dyson_gue_collapse"]["_raw_acc"]
    four_i1 = i1_stats["0.0020"]["fourier_linear_collapse"]["_raw_acc"]

    d_r_rand = paired_diff_ci(r_i1, rand_i1)
    d_r_dys = paired_diff_ci(r_i1, dys_i1)
    d_r_four = paired_diff_ci(r_i1, four_i1)

    h_i1a_pass = (
        i1_stats["0.0020"]["riemann_prime_collapse"]["origin_exact_acc"]["ci95_lo"] > 0.90
        and d_r_rand["ci95_lo"] > 0.80
        and d_r_dys["ci95_lo"] > 0.80
    )
    h_i1b_pass = d_r_four["ci95_lo"] > 0.45

    d_rmse_rand = paired_diff_ci(
        i2_stats["riemann_prime_collapse"]["_raw_rmse"],
        i2_stats["random_collapse"]["_raw_rmse"],
    )
    d_rmse_four = paired_diff_ci(
        i2_stats["riemann_prime_collapse"]["_raw_rmse"],
        i2_stats["fourier_linear_collapse"]["_raw_rmse"],
    )
    h_i2a_pass = (
        i2_stats["riemann_prime_collapse"]["class_acc"]["ci95_lo"] > 0.90
        and d_rmse_rand["ci95_hi"] < -0.50
        and d_rmse_four["ci95_hi"] < -0.25
    )

    d_cf_four = paired_diff_ci(
        i2_stats["riemann_prime_collapse"]["_raw_cf_acc"],
        i2_stats["fourier_linear_collapse"]["_raw_cf_acc"],
    )
    h_i2b_pass = (
        i2_stats["riemann_prime_collapse"]["cf_exact_acc"]["ci95_lo"] > 0.85
        and d_cf_four["ci95_lo"] > 0.40
    )

    d_i3_unif = paired_diff_ci(
        i3_stats["riemann_bayes_collapse"]["_raw_acc"],
        i3_stats["riemann_uniform_collapse"]["_raw_acc"],
    )
    d_i3_rand = paired_diff_ci(
        i3_stats["riemann_bayes_collapse"]["_raw_acc"],
        i3_stats["random_bayes_collapse"]["_raw_acc"],
    )
    d_i3_prior = paired_diff_ci(
        i3_stats["riemann_bayes_collapse"]["_raw_acc"],
        i3_stats["prior_only_ctrl"]["_raw_acc"],
    )
    h_i3a_pass = (
        d_i3_unif["ci95_lo"] > 0.05
        and d_i3_rand["ci95_lo"] > 0.30
        and d_i3_prior["ci95_lo"] > 0.30
    )

    d_an_rand = paired_diff_ci(
        i4_analogy["riemann_prime_collapse"]["_raw_acc"],
        i4_analogy["random_collapse"]["_raw_acc"],
    )
    d_an_four = paired_diff_ci(
        i4_analogy["riemann_prime_collapse"]["_raw_acc"],
        i4_analogy["fourier_linear_collapse"]["_raw_acc"],
    )
    h_i4a_pass = d_an_rand["ci95_lo"] > 0.70 and d_an_four["ci95_lo"] > 0.30

    d_noise_deg = paired_diff_ci(
        i4_noise["0.0300"]["_raw_acc"],
        i4_noise["0.0000"]["_raw_acc"],
    )
    h_i4b_pass = d_noise_deg["ci95_hi"] < -0.30

    d_cap_deg = paired_diff_ci(
        i4_cap["18"]["_raw_acc"],
        i4_cap["8"]["_raw_acc"],
    )
    h_i4c_pass = d_cap_deg["ci95_hi"] < 0.0

    summary["criteria"] = {
        "H_I1a": {
            "pass": h_i1a_pass,
            "riemann_acc": i1_stats["0.0020"]["riemann_prime_collapse"]["origin_exact_acc"],
            "diff_vs_random": d_r_rand,
            "diff_vs_dyson": d_r_dys,
        },
        "H_I1b": {
            "pass": h_i1b_pass,
            "diff_vs_fourier": d_r_four,
        },
        "H_I2a": {
            "pass": h_i2a_pass,
            "class_acc_riemann": i2_stats["riemann_prime_collapse"]["class_acc"],
            "rmse_diff_vs_random": d_rmse_rand,
            "rmse_diff_vs_fourier": d_rmse_four,
        },
        "H_I2b": {
            "pass": h_i2b_pass,
            "cf_acc_riemann": i2_stats["riemann_prime_collapse"]["cf_exact_acc"],
            "cf_diff_vs_fourier": d_cf_four,
        },
        "H_I3a": {
            "pass": h_i3a_pass,
            "diff_vs_uniform": d_i3_unif,
            "diff_vs_random_bayes": d_i3_rand,
            "diff_vs_prior_only": d_i3_prior,
        },
        "H_I4a": {
            "pass": h_i4a_pass,
            "diff_vs_random": d_an_rand,
            "diff_vs_fourier": d_an_four,
        },
        "H_I4b": {
            "pass": h_i4b_pass,
            "diff_030_vs_000": d_noise_deg,
        },
        "H_I4c": {
            "pass": h_i4c_pass,
            "diff_K18_vs_K8": d_cap_deg,
            "acc_K8": i4_cap["8"]["exact_acc"],
            "acc_K12": i4_cap["12"]["exact_acc"],
            "acc_K18": i4_cap["18"]["exact_acc"],
        },
    }

    # Strip private raw arrays before serialization
    def strip_raw(obj):
        if isinstance(obj, dict):
            return {k: strip_raw(v) for k, v in obj.items() if not k.startswith("_raw")}
        return obj

    return strip_raw(summary)


def print_report(summary: dict) -> None:
    phase = summary["phase"]
    print(f"\n================ {phase} (N={summary['n_seeds']} seeds, 10,000 paired bootstrap) ================")

    print("\n--- Suite I1: Rastreo Retrodictivo de Origen ---")
    for sig, arms in summary["suites"]["I1"].items():
        print(f"  sigma = {sig}:")
        for arm, st in arms.items():
            print(
                f"    {arm:25s} acc={fmt_stat(st['origin_exact_acc'], True)}  "
                f"f1={fmt_stat(st['origin_partial_f1'], True)}  snr={fmt_stat(st['peak_margin_snr'])}"
            )

    print("\n--- Suite I2: Predicción 'Del Otro Lado' + Contrafactual do(c_a -> c_new) ---")
    for arm, st in summary["suites"]["I2"].items():
        print(
            f"  {arm:25s} orig={fmt_stat(st['origin_exact_acc'], True)}  "
            f"cls4={fmt_stat(st['class_acc'], True)}  rmse={fmt_stat(st['pred_rmse'])}  "
            f"cf_acc={fmt_stat(st['cf_exact_acc'], True)}  cf_rmse={fmt_stat(st['cf_rmse'])}"
        )

    print("\n--- Suite I3: Desambiguación Bayesiana Pasado x Futuro (sigma=0.015) ---")
    for arm, st in summary["suites"]["I3"].items():
        print(
            f"  {arm:25s} acc={fmt_stat(st['origin_exact_acc'], True)}  "
            f"f1={fmt_stat(st['origin_partial_f1'], True)}"
        )

    print("\n--- Suite I4a: Transferencia Analógica (A:B::C:?) ---")
    for arm, st in summary["suites"]["I4"]["analogy"].items():
        print(
            f"  {arm:25s} exact={fmt_stat(st['exact_acc'], True)}  f1={fmt_stat(st['partial_f1'], True)}"
        )

    print("\n--- Suite I4b: Curva de Ruido (Riemann Prime, K=12) ---")
    for sig, st in summary["suites"]["I4"]["noise_sweep"].items():
        print(f"  sigma={sig}  exact={fmt_stat(st['exact_acc'], True)}  snr={fmt_stat(st['peak_margin_snr'])}")

    print("\n--- Suite I4c: Curva de Capacidad 1D (sigma=0.003, M=128) ---")
    for kv, st in summary["suites"]["I4"]["capacity_sweep"].items():
        print(
            f"  K={kv:>2s}  min_log_gap={st['min_log_gap']:.5f}  exact={fmt_stat(st['exact_acc'], True)}"
        )

    print("\n--- Veredicto de los 8 Criterios Preregistrados ---")
    passed = 0
    for cid, c in summary["criteria"].items():
        tag = "PASS" if c["pass"] else "FAIL"
        if c["pass"]:
            passed += 1
        print(f"  {cid}: {tag}")
    print(f"TOTAL {phase}: {passed}/{len(summary['criteria'])} PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()

    dev_summary = evaluate_phase("DEV", DEV_SEEDS)
    print_report(dev_summary)

    if not args.dev_only:
        confirm_summary = evaluate_phase("CONFIRM", CONFIRM_SEEDS)
        print_report(confirm_summary)
        out_path = DOCS_DIR / "summary_confirm.json"
        out_path.write_text(
            json.dumps({"DEV": dev_summary, "CONFIRM": confirm_summary}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved structured summary to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
