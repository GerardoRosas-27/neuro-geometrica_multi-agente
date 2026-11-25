#!/usr/bin/env python3
"""Paired bootstrap statistical summary (10,000 resamples, seed=12345, IC 95%)
and verbatim evaluation of the 8 preregistered criteria H_T1..H_T8."""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "spectral_transfer_v1"


def percentile(sorted_vals: list[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_mean_ci(vals: list[float], n_boot: int = 10000, seed: int = 12345) -> dict:
    n = len(vals)
    mean = sum(vals) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    return {
        "mean": mean,
        "ci_lo": percentile(boots, 0.025),
        "ci_hi": percentile(boots, 0.975),
        "min": min(vals),
        "max": max(vals),
    }


def bootstrap_paired_diff(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 12345) -> dict:
    assert len(a) == len(b)
    diffs = [x - y for x, y in zip(a, b)]
    st = bootstrap_mean_ci(diffs, n_boot=n_boot, seed=seed)
    st["wins_a"] = sum(1 for d in diffs if d > 0.0)
    st["wins_b"] = sum(1 for d in diffs if d < 0.0)
    st["ties"] = sum(1 for d in diffs if d == 0.0)
    return st


def load_suite(split: str, filename: str, key_cols: list[str]) -> tuple[list[str], dict]:
    raw_dir = DOCS / "raw" / split
    seed_dirs = sorted(raw_dir.glob("seed_0x*"))
    seeds = [d.name for d in seed_dirs]
    data: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sd in seed_dirs:
        with (sd / filename).open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = tuple(row[c] for c in key_cols)
                for col, val in row.items():
                    if col in ("seed", "arm", "axis", "n_few"):
                        continue
                    data[key][col].append(float(val))
    return seeds, data


def summarize_split(split: str) -> dict:
    seeds, t1 = load_suite(split, "metrics_t1.csv", ["arm"])
    _, t2 = load_suite(split, "metrics_t2.csv", ["arm"])
    _, t3 = load_suite(split, "metrics_t3.csv", ["n_few", "arm"])
    _, t4 = load_suite(split, "metrics_t4.csv", ["n_few", "arm"])

    def agg_table(raw: dict) -> dict:
        out = {}
        for k, metrics in raw.items():
            ks = "|".join(k)
            out[ks] = {m: bootstrap_mean_ci(vs) for m, vs in metrics.items()}
        return out

    # Key paired differences for H_T1..H_T8
    d = {}
    # T1
    d["t1_acc_adapt_vs_riemann"] = bootstrap_paired_diff(t1[("adaptive_spectral",)]["cls_acc"], t1[("riemann_fixed",)]["cls_acc"])
    d["t1_acc_adapt_vs_poisson"] = bootstrap_paired_diff(t1[("adaptive_spectral",)]["cls_acc"], t1[("poisson_fixed",)]["cls_acc"])
    d["t1_acc_adapt_vs_fourier"] = bootstrap_paired_diff(t1[("adaptive_spectral",)]["cls_acc"], t1[("fourier_fixed",)]["cls_acc"])
    d["t1_acc_adapt_vs_mlp"] = bootstrap_paired_diff(t1[("adaptive_spectral",)]["cls_acc"], t1[("mlp_conventional",)]["cls_acc"])
    d["t1_r_gain"] = bootstrap_paired_diff(t1[("adaptive_spectral",)]["spec_r_final"], t1[("adaptive_spectral",)]["spec_r_init"])
    d["t1_abd_adapt_vs_mlp"] = bootstrap_paired_diff(t1[("adaptive_spectral",)]["abd_top1"], t1[("mlp_conventional",)]["abd_top1"])
    d["t1_rmse_adapt_vs_mlp"] = bootstrap_paired_diff(t1[("adaptive_spectral",)]["reg_rmse"], t1[("mlp_conventional",)]["reg_rmse"])
    d["t1_nmi_adapt"] = bootstrap_mean_ci(t1[("adaptive_spectral",)]["nmi_unsup"])
    d["t1_cfr_adapt"] = bootstrap_mean_ci(t1[("adaptive_spectral",)]["cfr"])

    # T2
    d["t2_acc_full_vs_phase_scramble"] = bootstrap_paired_diff(t2[("full_adaptive",)]["cls_acc"], t2[("ablate_phase_scramble",)]["cls_acc"])
    d["t2_acc_full_vs_zero_phase"] = bootstrap_paired_diff(t2[("full_adaptive",)]["cls_acc"], t2[("train_zero_phase",)]["cls_acc"])
    d["t2_asym_full_vs_zero_phase"] = bootstrap_paired_diff(t2[("full_adaptive",)]["acc_asym"], t2[("train_zero_phase",)]["acc_asym"])
    d["t2_acc_full_vs_additive"] = bootstrap_paired_diff(t2[("full_adaptive",)]["cls_acc"], t2[("ablate_pairing_additive",)]["cls_acc"])
    d["t2_acc_full_vs_permuted"] = bootstrap_paired_diff(t2[("full_adaptive",)]["cls_acc"], t2[("ablate_pairing_permuted",)]["cls_acc"])

    # T3 (N=48 main + N=24 ultra few-shot)
    for nf in ("24", "48", "96"):
        for other in ("scratch_spectral", "scratch_mlp", "transfer_mlp_A", "riemann_A", "fourier_A", "poisson_A", "dyson_A", "neg_ctrl_noise_A", "neg_ctrl_phase_scramble", "untrained_poisson"):
            d[f"t3_n{nf}_acc_vs_{other}"] = bootstrap_paired_diff(
                t3[(nf, "transfer_spectral_A")]["cls_acc"], t3[(nf, other)]["cls_acc"]
            )
            d[f"t3_n{nf}_rmse_vs_{other}"] = bootstrap_paired_diff(
                t3[(nf, "transfer_spectral_A")]["reg_rmse"], t3[(nf, other)]["reg_rmse"]
            )

    # T4 (N=60 main + N=30 ultra few-shot)
    for nf in ("30", "60", "120"):
        for other in ("scratch_spectral", "scratch_mlp", "transfer_mlp_A", "riemann_A", "fourier_A", "poisson_A", "dyson_A", "neg_ctrl_noise_A", "neg_ctrl_phase_scramble", "untrained_poisson"):
            d[f"t4_n{nf}_acc_vs_{other}"] = bootstrap_paired_diff(
                t4[(nf, "transfer_spectral_A")]["cls_acc"], t4[(nf, other)]["cls_acc"]
            )
    d["t4_n60_nmi_vs_untrained"] = bootstrap_paired_diff(
        t4[("60", "transfer_spectral_A")]["nmi_unsup"], t4[("60", "untrained_poisson")]["nmi_unsup"]
    )
    d["t4_n60_cons_vs_transfer_mlp"] = bootstrap_paired_diff(
        t4[("60", "transfer_spectral_A")]["alg_consistency"], t4[("60", "transfer_mlp_A")]["alg_consistency"]
    )
    d["t4_n60_abd_vs_untrained"] = bootstrap_paired_diff(
        t4[("60", "transfer_spectral_A")]["abd_top1"], t4[("60", "untrained_poisson")]["abd_top1"]
    )

    # Exact verbatim evaluation of H_T1..H_T8 from PLAN.md
    criteria = {
        "H_T1": (
            d["t1_nmi_adapt"]["ci_lo"] > 0.35
            and d["t1_acc_adapt_vs_riemann"]["ci_lo"] > 0.02
            and d["t1_acc_adapt_vs_poisson"]["ci_lo"] > 0.04
            and d["t1_r_gain"]["ci_lo"] > 0.08
        ),
        "H_T2": (
            d["t1_abd_adapt_vs_mlp"]["ci_lo"] > 0.10
            and d["t1_rmse_adapt_vs_mlp"]["ci_hi"] < -0.03
            and d["t1_cfr_adapt"]["ci_lo"] >= 0.90
        ),
        "H_T3": (
            d["t2_acc_full_vs_phase_scramble"]["ci_lo"] > 0.15
            and d["t2_acc_full_vs_additive"]["ci_lo"] > 0.20
            and d["t2_acc_full_vs_permuted"]["ci_lo"] > 0.20
        ),
        "H_T4": (
            d["t3_n48_acc_vs_scratch_spectral"]["ci_lo"] > 0.05
            and d["t3_n48_rmse_vs_scratch_spectral"]["ci_hi"] < -0.04
            and d["t3_n48_acc_vs_neg_ctrl_noise_A"]["ci_lo"] > 0.05
            and d["t3_n48_acc_vs_untrained_poisson"]["ci_lo"] > 0.08
        ),
        "H_T5": (
            d["t3_n48_acc_vs_transfer_mlp_A"]["ci_lo"] > 0.04
            and d["t3_n48_rmse_vs_transfer_mlp_A"]["ci_hi"] < -0.02
            and d["t3_n48_acc_vs_riemann_A"]["ci_lo"] > 0.02
            and d["t3_n48_acc_vs_poisson_A"]["ci_lo"] > 0.03
        ),
        "H_T6": (
            d["t4_n60_acc_vs_scratch_spectral"]["ci_lo"] > 0.05
            and d["t4_n60_acc_vs_neg_ctrl_noise_A"]["ci_lo"] > 0.05
            and d["t4_n60_acc_vs_neg_ctrl_phase_scramble"]["ci_lo"] > 0.10
            and d["t4_n60_nmi_vs_untrained"]["ci_lo"] > 0.08
        ),
        "H_T7": (
            d["t4_n60_acc_vs_transfer_mlp_A"]["ci_lo"] > 0.04
            and d["t4_n60_acc_vs_riemann_A"]["ci_lo"] > 0.02
            and d["t4_n60_cons_vs_transfer_mlp"]["ci_lo"] > 0.03
            and d["t4_n60_abd_vs_untrained"]["ci_lo"] > 0.10
        ),
        "H_T8": (
            d["t3_n24_acc_vs_scratch_spectral"]["ci_lo"] > 0.06
            and d["t3_n24_acc_vs_scratch_mlp"]["ci_lo"] > 0.06
            and d["t4_n30_acc_vs_scratch_spectral"]["ci_lo"] > 0.06
            and d["t4_n30_acc_vs_scratch_mlp"]["ci_lo"] > 0.06
        ),
    }
    n_pass = sum(1 for v in criteria.values() if v)
    return {
        "split": split,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_pass": n_pass,
        "n_total": len(criteria),
        "criteria": criteria,
        "paired_diffs": d,
        "suite_t1": agg_table(t1),
        "suite_t2": agg_table(t2),
        "suite_t3": agg_table(t3),
        "suite_t4": agg_table(t4),
    }


def print_report(s: dict) -> str:
    lines = []
    lines.append(f"=== SUMMARY {s['split'].upper()} (N={s['n_seeds']} seeds) : {s['n_pass']}/{s['n_total']} PASS ===")
    for k, ok in s["criteria"].items():
        lines.append(f"  {k}: {'PASS' if ok else 'FAIL'}")

    def fmt_ci(obj: dict, pct: bool = True) -> str:
        m = 100.0 if pct else 1.0
        u = "%" if pct else ""
        return f"{obj['mean']*m:+.2f}{u} [{obj['ci_lo']*m:+.2f}, {obj['ci_hi']*m:+.2f}]" if pct else f"{obj['mean']:.4f} [{obj['ci_lo']:.4f}, {obj['ci_hi']:.4f}]"

    lines.append("\n--- T1 (Domain A) ---")
    for arm in ("adaptive_spectral", "riemann_fixed", "fourier_fixed", "poisson_fixed", "dyson_fixed", "mlp_conventional"):
        r = s["suite_t1"][arm]
        lines.append(
            f"  {arm:18s} | acc={r['cls_acc']['mean']*100:5.2f}% rmse={r['reg_rmse']['mean']:.4f} "
            f"nmi={r['nmi_unsup']['mean']:.3f} abd={r['abd_top1']['mean']*100:5.2f}% "
            f"cfr={r['cfr']['mean']*100:5.1f}% <r>={r['spec_r_init']['mean']:.3f}->{r['spec_r_final']['mean']:.3f}"
        )

    lines.append("\n--- T2 (Causal Dissection: Frequencies, Phases, Pairings) ---")
    for arm in ("full_adaptive", "ablate_phase_scramble", "train_zero_phase", "ablate_pairing_additive", "ablate_pairing_permuted", "ablate_freq_riemann", "ablate_freq_fourier", "ablate_freq_poisson"):
        r = s["suite_t2"][arm]
        lines.append(
            f"  {arm:24s} | acc={r['cls_acc']['mean']*100:5.2f}% sym={r['acc_sym']['mean']*100:5.2f}% "
            f"asym={r['acc_asym']['mean']*100:5.2f}% rmse={r['reg_rmse']['mean']:.4f} "
            f"abd={r['abd_top1']['mean']*100:5.2f}% nmi={r['nmi_unsup']['mean']:.3f}"
        )

    lines.append("\n--- T3 (Transfer A -> B Continuous Oscillators, N_few=48) ---")
    for arm in ("transfer_spectral_A", "riemann_A", "fourier_A", "poisson_A", "dyson_A", "transfer_mlp_A", "neg_ctrl_noise_A", "neg_ctrl_phase_scramble", "untrained_poisson", "scratch_spectral", "scratch_mlp"):
        r = s["suite_t3"][f"48|{arm}"]
        lines.append(
            f"  {arm:24s} | acc={r['cls_acc']['mean']*100:5.2f}% rmse={r['reg_rmse']['mean']:.4f} "
            f"nmi={r['nmi_unsup']['mean']:.3f} abd={r['abd_top1']['mean']*100:5.2f}% (N24={s['suite_t3'][f'24|{arm}']['cls_acc']['mean']*100:5.2f}%, N96={s['suite_t3'][f'96|{arm}']['cls_acc']['mean']*100:5.2f}%)"
        )

    lines.append("\n--- T4 (Transfer A -> C Discrete Relational Graph, N_few=60) ---")
    for arm in ("transfer_spectral_A", "riemann_A", "fourier_A", "poisson_A", "dyson_A", "transfer_mlp_A", "neg_ctrl_noise_A", "neg_ctrl_phase_scramble", "untrained_poisson", "scratch_spectral", "scratch_mlp"):
        r = s["suite_t4"][f"60|{arm}"]
        lines.append(
            f"  {arm:24s} | acc={r['cls_acc']['mean']*100:5.2f}% cons={r['alg_consistency']['mean']*100:5.2f}% "
            f"nmi={r['nmi_unsup']['mean']:.3f} abd={r['abd_top1']['mean']*100:5.2f}% (N30={s['suite_t4'][f'30|{arm}']['cls_acc']['mean']*100:5.2f}%, N120={s['suite_t4'][f'120|{arm}']['cls_acc']['mean']*100:5.2f}%)"
        )

    lines.append("\n--- Key Paired Diffs (IC 95%) ---")
    for k in (
        "t1_acc_adapt_vs_riemann", "t1_acc_adapt_vs_poisson", "t1_r_gain",
        "t1_abd_adapt_vs_mlp", "t1_rmse_adapt_vs_mlp",
        "t2_acc_full_vs_phase_scramble", "t2_acc_full_vs_additive", "t2_acc_full_vs_permuted",
        "t3_n48_acc_vs_scratch_spectral", "t3_n48_rmse_vs_scratch_spectral",
        "t3_n48_acc_vs_transfer_mlp_A", "t3_n48_rmse_vs_transfer_mlp_A",
        "t3_n48_acc_vs_riemann_A", "t3_n48_acc_vs_poisson_A",
        "t3_n48_acc_vs_neg_ctrl_noise_A", "t3_n48_acc_vs_untrained_poisson",
        "t4_n60_acc_vs_scratch_spectral", "t4_n60_acc_vs_transfer_mlp_A",
        "t4_n60_acc_vs_riemann_A", "t4_n60_acc_vs_neg_ctrl_noise_A",
        "t4_n60_acc_vs_neg_ctrl_phase_scramble", "t4_n60_nmi_vs_untrained",
        "t4_n60_cons_vs_transfer_mlp", "t4_n60_abd_vs_untrained",
        "t3_n24_acc_vs_scratch_spectral", "t3_n24_acc_vs_scratch_mlp",
        "t4_n30_acc_vs_scratch_spectral", "t4_n30_acc_vs_scratch_mlp",
    ):
        is_pct = not ("rmse" in k or "r_gain" in k or "nmi" in k)
        lines.append(f"  {k:40s}: {fmt_ci(s['paired_diffs'][k], pct=is_pct)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "confirm"], required=True)
    args = ap.parse_args()
    summary = summarize_split(args.split)
    out_json = DOCS / f"summary_{args.split}.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    text = print_report(summary)
    print(text)
    if args.split == "confirm":
        (DOCS / "analysis.log").write_text(text + "\n")


if __name__ == "__main__":
    main()
