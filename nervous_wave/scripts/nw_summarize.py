#!/usr/bin/env python3
"""Paired bootstrap summarizer (10,000 resamples, seed=12345, IC 95%)
and exact verbatim evaluation of H_N1..H_N6 from docs/nervous_wave_v1/PLAN.md."""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "nervous_wave_v1"


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


def load_suite(split: str, filename: str) -> tuple[list[str], dict]:
    raw_dir = DOCS / "raw" / split
    seed_dirs = sorted(raw_dir.glob("seed_0x*"))
    seeds = [d.name for d in seed_dirs]
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sd in seed_dirs:
        with (sd / filename).open() as f:
            for row in csv.DictReader(f):
                arm = row["arm"]
                for col, val in row.items():
                    if col in ("seed", "arm"):
                        continue
                    data[arm][col].append(float(val))
    return seeds, data


def summarize_split(split: str, n_boot: int = 10000) -> dict:
    seeds, p1 = load_suite(split, "metrics_p1_physics.csv")
    _, p2 = load_suite(split, "metrics_p2_inference.csv")
    _, p3 = load_suite(split, "metrics_p3_resilience.csv")
    _, p4 = load_suite(split, "metrics_p4_transfer.csv")

    def agg(raw: dict) -> dict:
        return {
            arm: {m: bootstrap_mean_ci(vs, n_boot=min(n_boot, 4000)) for m, vs in metrics.items()}
            for arm, metrics in raw.items()
        }

    d = {}
    # P1 physics
    d["p1_coinc_nervous_vs_grid"] = bootstrap_paired_diff(p1["nervous_system"]["collision_coinc"], p1["grid_local_lattice"]["collision_coinc"], n_boot)
    d["p1_coinc_nervous_vs_random"] = bootstrap_paired_diff(p1["nervous_system"]["collision_coinc"], p1["random_erdos_renyi"]["collision_coinc"], n_boot)
    d["p1_loc_nervous_vs_random"] = bootstrap_paired_diff(p1["nervous_system"]["localization_snr"], p1["random_erdos_renyi"]["localization_snr"], n_boot)
    d["p1_tarr_nervous"] = bootstrap_mean_ci(p1["nervous_system"]["t_arrival"], n_boot)
    d["p1_tarr_grid"] = bootstrap_mean_ci(p1["grid_local_lattice"]["t_arrival"], n_boot)
    d["p1_efinal_nervous"] = bootstrap_mean_ci(p1["nervous_system"]["e_final"], n_boot)
    d["p1_ehub_t0_max"] = max(p1["nervous_system"]["e_hub_t0"])

    # P2 inference
    for other in ("grid_local_lattice", "random_erdos_renyi", "all_to_all_flat", "nervous_no_delay", "nervous_no_inhibition"):
        d[f"p2_acc_a_vs_{other}"] = bootstrap_paired_diff(p2["nervous_system"]["acc_a"], p2[other]["acc_a"], n_boot)
        d[f"p2_rmse_a_vs_{other}"] = bootstrap_paired_diff(p2["nervous_system"]["rmse_a"], p2[other]["rmse_a"], n_boot)
        d[f"p2_abd_a_vs_{other}"] = bootstrap_paired_diff(p2["nervous_system"]["abd_a"], p2[other]["abd_a"], n_boot)
        d[f"p2_acc_c_vs_{other}"] = bootstrap_paired_diff(p2["nervous_system"]["acc_c"], p2[other]["acc_c"], n_boot)

    # P3 resilience
    for other in ("grid_local_lattice", "random_erdos_renyi", "all_to_all_flat", "nervous_no_delay", "nervous_no_inhibition"):
        d[f"p3_l15_vs_{other}"] = bootstrap_paired_diff(p3["nervous_system"]["acc_lesion15"], p3[other]["acc_lesion15"], n_boot)
        d[f"p3_l30_vs_{other}"] = bootstrap_paired_diff(p3["nervous_system"]["acc_lesion30"], p3[other]["acc_lesion30"], n_boot)
        d[f"p3_noise_vs_{other}"] = bootstrap_paired_diff(p3["nervous_system"]["acc_noise50"], p3[other]["acc_noise50"], n_boot)

    # P4 myelin transfer A -> C
    for other in ("grid_transfer", "random_transfer", "flat_transfer", "ctrl_permuted_synapses"):
        d[f"p4_acc_vs_{other}"] = bootstrap_paired_diff(p4["nervous_transfer"]["cls_acc"], p4[other]["cls_acc"], n_boot)

    # Verbatim H_N1..H_N6 from PLAN.md
    criteria = {
        "H_N1": (
            d["p1_ehub_t0_max"] == 0.0
            and d["p1_tarr_nervous"]["mean"] <= 2.5
            and d["p1_tarr_nervous"]["mean"] < d["p1_tarr_grid"]["mean"]
            and 0.15 <= d["p1_efinal_nervous"]["mean"] <= 1.25
            and d["p1_coinc_nervous_vs_grid"]["ci_lo"] > 0.05
            and d["p1_coinc_nervous_vs_random"]["ci_lo"] > 0.05
        ),
        "H_N2": (
            d["p2_acc_a_vs_grid_local_lattice"]["ci_lo"] > 0.03
            and d["p2_rmse_a_vs_grid_local_lattice"]["ci_hi"] < -0.04
            and d["p2_acc_a_vs_random_erdos_renyi"]["ci_lo"] > 0.03
            and d["p2_rmse_a_vs_random_erdos_renyi"]["ci_hi"] < -0.04
        ),
        "H_N3": (
            d["p2_acc_a_vs_all_to_all_flat"]["ci_lo"] > -0.02
            and d["p2_rmse_a_vs_all_to_all_flat"]["ci_hi"] < 0.02
            and d["p2_acc_c_vs_all_to_all_flat"]["ci_lo"] > -0.02
        ),
        "H_N4": (
            d["p2_acc_a_vs_nervous_no_delay"]["ci_lo"] > 0.05
            and d["p2_abd_a_vs_nervous_no_delay"]["ci_lo"] > 0.08
            and d["p3_noise_vs_nervous_no_inhibition"]["ci_lo"] > 0.02
        ),
        "H_N5": (
            d["p3_l15_vs_grid_local_lattice"]["ci_lo"] > 0.03
            and d["p3_l15_vs_random_erdos_renyi"]["ci_lo"] > 0.03
            and d["p3_l30_vs_random_erdos_renyi"]["ci_lo"] > 0.03
            and d["p3_noise_vs_grid_local_lattice"]["ci_lo"] > 0.02
            and d["p3_noise_vs_all_to_all_flat"]["ci_lo"] > 0.02
        ),
        "H_N6": (
            d["p4_acc_vs_grid_transfer"]["ci_lo"] > 0.03
            and d["p4_acc_vs_random_transfer"]["ci_lo"] > 0.03
            and d["p4_acc_vs_ctrl_permuted_synapses"]["ci_lo"] > 0.05
        ),
    }

    return {
        "split": split,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_pass": sum(1 for v in criteria.values() if v),
        "n_total": len(criteria),
        "criteria": criteria,
        "paired_diffs": d,
        "suite_p1": agg(p1),
        "suite_p2": agg(p2),
        "suite_p3": agg(p3),
        "suite_p4": agg(p4),
    }


def print_report(s: dict) -> str:
    lines = [f"=== PRUEBA A — SUMMARY {s['split'].upper()} (N={s['n_seeds']} seeds) : {s['n_pass']}/{s['n_total']} PASS ==="]
    for k, ok in s["criteria"].items():
        lines.append(f"  {k}: {'PASS' if ok else 'FAIL'}")

    arms = ("nervous_system", "grid_local_lattice", "random_erdos_renyi", "all_to_all_flat", "nervous_no_delay", "nervous_no_inhibition")
    lines.append("\n--- P1 (Physical Wave Propagation) ---")
    for a in arms:
        r = s["suite_p1"][a]
        lines.append(
            f"  {a:22s} | E={int(r['n_edges']['mean']):4d} E_hub0={r['e_hub_t0']['mean']:.4f} "
            f"t_arr={r['t_arrival']['mean']:.2f} hops={r['mean_hop_dist']['mean']:.2f} "
            f"E_fin={r['e_final']['mean']:.3f} locSNR={r['localization_snr']['mean']*100:5.1f}% coinc={r['collision_coinc']['mean']:.3f}"
        )

    lines.append("\n--- P2 (Relational Inference: Domain A & Domain C) ---")
    for a in arms:
        r = s["suite_p2"][a]
        lines.append(
            f"  {a:22s} | E={int(r['n_edges']['mean']):4d} A_acc={r['acc_a']['mean']*100:5.2f}% rmse={r['rmse_a']['mean']:.4f} "
            f"abd={r['abd_a']['mean']*100:5.2f}% 2hop={r['two_hop_a']['mean']*100:5.2f}% | C_acc={r['acc_c']['mean']*100:5.2f}% cons={r['cons_c']['mean']*100:5.2f}%"
        )

    lines.append("\n--- P3 (Neurological Resilience: 15%/30% Axonal Lesions & sigma=0.50 Noise) ---")
    for a in arms:
        r = s["suite_p3"][a]
        lines.append(
            f"  {a:22s} | clean={r['acc_clean']['mean']*100:5.2f}% L15={r['acc_lesion15']['mean']*100:5.2f}% ({r['ret_lesion15']['mean']*100:4.1f}%) "
            f"L30={r['acc_lesion30']['mean']*100:5.2f}% ({r['ret_lesion30']['mean']*100:4.1f}%) noise={r['acc_noise50']['mean']*100:5.2f}% ({r['ret_noise50']['mean']*100:4.1f}%)"
        )

    lines.append("\n--- P4 (Cross-Domain A -> C Frozen-Synapse Myelin Calibration Transfer, N=60) ---")
    for a in ("nervous_transfer", "grid_transfer", "random_transfer", "flat_transfer", "ctrl_permuted_synapses"):
        r = s["suite_p4"][a]
        lines.append(
            f"  {a:24s} | E={int(r['n_edges']['mean']):4d} acc={r['cls_acc']['mean']*100:5.2f}% cons={r['alg_consistency']['mean']*100:5.2f}% abd={r['abd_top1']['mean']*100:5.2f}%"
        )

    lines.append("\n--- Key Paired Diffs (IC 95%) ---")
    for k in (
        "p1_coinc_nervous_vs_grid", "p1_coinc_nervous_vs_random",
        "p2_acc_a_vs_grid_local_lattice", "p2_rmse_a_vs_grid_local_lattice",
        "p2_acc_a_vs_random_erdos_renyi", "p2_rmse_a_vs_random_erdos_renyi",
        "p2_acc_a_vs_all_to_all_flat", "p2_rmse_a_vs_all_to_all_flat", "p2_acc_c_vs_all_to_all_flat",
        "p2_acc_a_vs_nervous_no_delay", "p2_abd_a_vs_nervous_no_delay",
        "p3_noise_vs_nervous_no_inhibition", "p3_noise_vs_grid_local_lattice", "p3_noise_vs_all_to_all_flat",
        "p3_l15_vs_grid_local_lattice", "p3_l15_vs_random_erdos_renyi", "p3_l30_vs_random_erdos_renyi",
        "p4_acc_vs_grid_transfer", "p4_acc_vs_random_transfer", "p4_acc_vs_ctrl_permuted_synapses",
    ):
        v = s["paired_diffs"][k]
        is_pct = not ("rmse" in k or "coinc" in k)
        m = 100.0 if is_pct else 1.0
        u = "%" if is_pct else ""
        lines.append(f"  {k:38s}: {v['mean']*m:+.2f}{u} [{v['ci_lo']*m:+.2f}, {v['ci_hi']*m:+.2f}] ({v['wins_a']}/{s['n_seeds']} wins)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "confirm"], required=True)
    args = ap.parse_args()
    summary = summarize_split(args.split)
    (DOCS / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2) + "\n")
    rep = print_report(summary)
    print(rep)
    if args.split == "confirm":
        (DOCS / "analysis.log").write_text(rep + "\n")


if __name__ == "__main__":
    main()
