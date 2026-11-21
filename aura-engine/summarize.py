#!/usr/bin/env python3
"""Estadística bootstrap pareada por semilla (10,000 remuestreos, seed 12345, delta 0.05)
para AURA + CorticalFabric v1. Python NO entrena ni infiere; solo consolida métricas."""

import argparse
import csv
import json
import pathlib
import random
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/aura_v1"


def quantile(xs, p):
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (i - lo) * (xs[hi] - xs[lo])


def bootstrap_diff(d):
    rng = random.Random(12345)
    n = len(d)
    means = [sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000)]
    return {
        "mean": statistics.mean(d),
        "ci95": [quantile(means, 0.025), quantile(means, 0.975)],
        "seeds": n,
    }


def load_csv_group(raw_root, phases, filename, int_cols, float_cols):
    rows = []
    for phase in phases:
        for path in sorted((raw_root / phase).glob(f"*/{filename}")):
            for r in csv.DictReader(path.open()):
                for k in int_cols:
                    r[k] = int(r[k])
                for k in float_cols:
                    r[k] = float(r[k])
                rows.append(r)
    return rows


def paired_contrast(by_arm, arm_a, arm_b, metric):
    a_rows = sorted(by_arm[arm_a], key=lambda r: r["seed"])
    b_rows = sorted(by_arm[arm_b], key=lambda r: r["seed"])
    assert [r["seed"] for r in a_rows] == [r["seed"] for r in b_rows]
    diffs = [a[metric] - b[metric] for a, b in zip(a_rows, b_rows)]
    return bootstrap_diff(diffs)


def single_bootstrap(rows, metric):
    vals = [r[metric] for r in sorted(rows, key=lambda r: r["seed"])]
    return bootstrap_diff(vals)


def analyze_phase(raw_root, phase):
    e1 = load_csv_group(
        raw_root,
        [phase],
        "metrics_e1.csv",
        ["seed", "total_patches"],
        [
            "pred_bytes_per_patch",
            "high_ent_bytes_per_patch",
            "compression_ratio",
            "phase1_norm_err",
            "phase2_shift_norm_err",
            "phase2_recovery_steps",
            "phase3_static_norm_err",
            "phase3_static_coherence",
            "phase2_shift_coherence",
            "phase4_recall_norm_err",
            "familiar_epistemic",
            "static_epistemic",
            "ood_epistemic",
            "effective_rank",
            "mean_latent_norm",
            "eval_ms",
        ],
    )
    e2 = load_csv_group(
        raw_root,
        [phase],
        "metrics_e2.csv",
        ["seed"],
        [
            "phase1_return",
            "phase1_success",
            "phase2_shift_return",
            "phase2_shift_success",
            "phase3_ood_return",
            "combined_p2_p3_return",
            "abstain_rate_trap",
            "abstain_rate_familiar",
            "system1_frac_familiar",
            "mean_rollouts_per_step",
            "eval_ms",
        ],
    )
    e3 = load_csv_group(
        raw_root,
        [phase],
        "metrics_e3.csv",
        ["seed", "weight_flips"],
        [
            "acc_a_pre",
            "acc_a_post",
            "forgetting_a",
            "acc_b_post",
            "combined_ab",
            "nonzero_frac",
            "eval_ms",
        ],
    )
    e4 = load_csv_group(
        raw_root,
        [phase],
        "metrics_e4.csv",
        ["seed"],
        [
            "fe_regime1_early",
            "fe_regime1_late",
            "fe_regime2_early",
            "fe_regime2_late",
            "fe_late_mean",
            "sensory_fe_late",
            "mean_spike_rate_pct",
            "ne_pre_shift",
            "ne_post_shift",
            "ne_shift_delta",
            "ticks_per_sec",
            "eval_ms",
        ],
    )
    e5 = [json.loads(p.read_text()) for p in sorted((raw_root / phase).glob("*/bench_e5.json"))]

    assert len({r["seed"] for r in e1}) == 16
    assert len({r["seed"] for r in e2}) == 16
    assert len({r["seed"] for r in e3}) == 16
    assert len({r["seed"] for r in e4}) == 16
    assert len(e5) == 16

    def group(rows):
        out = {}
        for r in rows:
            out.setdefault(r["arm"], []).append(r)
        return out

    g1, g2, g3, g4 = group(e1), group(e2), group(e3), group(e4)

    def summarize_arms(g, cols):
        return {
            arm: {c: statistics.mean(r[c] for r in rs) for c in cols}
            for arm, rs in g.items()
        }

    # --- E1 ---
    e1_cols = [
        "total_patches",
        "pred_bytes_per_patch",
        "high_ent_bytes_per_patch",
        "compression_ratio",
        "phase1_norm_err",
        "phase2_shift_norm_err",
        "phase2_recovery_steps",
        "phase3_static_norm_err",
        "phase3_static_coherence",
        "phase2_shift_coherence",
        "phase4_recall_norm_err",
        "familiar_epistemic",
        "static_epistemic",
        "ood_epistemic",
        "effective_rank",
        "mean_latent_norm",
        "eval_ms",
    ]
    full1 = sorted(g1["aura_full"], key=lambda r: r["seed"])
    s1 = {
        "arms": summarize_arms(g1, e1_cols),
        "paired": {
            "H1a_shift_err_full_minus_no_fast": paired_contrast(g1, "aura_full", "no_fast_weights", "phase2_shift_norm_err"),
            "H1a_shift_err_full_minus_persistence": paired_contrast(g1, "aura_full", "persistence", "phase2_shift_norm_err"),
            "H1b_recall_err_full_minus_no_hip": paired_contrast(g1, "aura_full", "no_hippocampus", "phase4_recall_norm_err"),
            "H1c_erank_full_minus_no_vicreg": paired_contrast(g1, "aura_full", "no_vicreg", "effective_rank"),
            "H1c_norm_full_minus_no_vicreg": paired_contrast(g1, "aura_full", "no_vicreg", "mean_latent_norm"),
            "H1d_compression_full_minus_fixed": paired_contrast(g1, "aura_full", "fixed_patcher", "compression_ratio"),
            "H1e_ood_minus_familiar_epi": bootstrap_diff([r["ood_epistemic"] - r["familiar_epistemic"] for r in full1]),
            "H1e_shift_minus_static_coherence": bootstrap_diff([r["phase2_shift_coherence"] - r["phase3_static_coherence"] for r in full1]),
        },
    }

    # --- E2 ---
    e2_cols = [
        "phase1_return",
        "phase1_success",
        "phase2_shift_return",
        "phase2_shift_success",
        "phase3_ood_return",
        "combined_p2_p3_return",
        "abstain_rate_trap",
        "abstain_rate_familiar",
        "system1_frac_familiar",
        "mean_rollouts_per_step",
        "eval_ms",
    ]
    full2 = sorted(g2["aura_full"], key=lambda r: r["seed"])
    s2 = {
        "arms": summarize_arms(g2, e2_cols),
        "paired": {
            "H2a_combined_full_minus_sys1": paired_contrast(g2, "aura_full", "system1_only", "combined_p2_p3_return"),
            "H2a_combined_full_minus_no_fast": paired_contrast(g2, "aura_full", "no_fast_weights", "combined_p2_p3_return"),
            "H2a_combined_full_minus_random": paired_contrast(g2, "aura_full", "random_control", "combined_p2_p3_return"),
            "H2a_p2_succ_full_minus_sys1": paired_contrast(g2, "aura_full", "system1_only", "phase2_shift_success"),
            "H2a_p2_succ_full_minus_no_fast": paired_contrast(g2, "aura_full", "no_fast_weights", "phase2_shift_success"),
            "H2b_abstain_trap_minus_familiar": bootstrap_diff([r["abstain_rate_trap"] - r["abstain_rate_familiar"] for r in full2]),
        },
    }

    # --- E3 ---
    e3_cols = [
        "acc_a_pre",
        "acc_a_post",
        "forgetting_a",
        "acc_b_post",
        "combined_ab",
        "weight_flips",
        "nonzero_frac",
        "eval_ms",
    ]
    s3 = {
        "arms": summarize_arms(g3, e3_cols),
        "paired": {
            "H3a_combined_full_minus_no_replay": paired_contrast(g3, "aura_sleep_full", "ablation_no_replay", "combined_ab"),
            "H3a_forgetting_full_minus_no_replay": paired_contrast(g3, "aura_sleep_full", "ablation_no_replay", "forgetting_a"),
            "H3b_combined_full_minus_no_hysteresis": paired_contrast(g3, "aura_sleep_full", "ablation_no_hysteresis", "combined_ab"),
            "H3b_forgetting_full_minus_no_hysteresis": paired_contrast(g3, "aura_sleep_full", "ablation_no_hysteresis", "forgetting_a"),
            "H3_combined_full_minus_naive": paired_contrast(g3, "aura_sleep_full", "ablation_naive", "combined_ab"),
        },
    }

    # --- E4 ---
    e4_cols = [
        "fe_regime1_early",
        "fe_regime1_late",
        "fe_regime2_early",
        "fe_regime2_late",
        "fe_late_mean",
        "sensory_fe_late",
        "mean_spike_rate_pct",
        "ne_pre_shift",
        "ne_post_shift",
        "ne_shift_delta",
        "ticks_per_sec",
        "eval_ms",
    ]
    s4 = {
        "arms": summarize_arms(g4, e4_cols),
        "paired": {
            "H4a_fe_late_full_minus_frozen": paired_contrast(g4, "cortical_full", "cortical_frozen", "fe_late_mean"),
            "H4a_sens_fe_full_minus_frozen": paired_contrast(g4, "cortical_full", "cortical_frozen", "sensory_fe_late"),
            "H4b_ne_shift_delta_full": single_bootstrap(g4["cortical_full"], "ne_shift_delta"),
            "H4c_fe_late_full_minus_no_phase": paired_contrast(g4, "cortical_full", "cortical_no_phase_gate", "fe_late_mean"),
            "H4c_fe_late_full_minus_no_neuromod": paired_contrast(g4, "cortical_full", "cortical_no_neuromod", "fe_late_mean"),
        },
    }

    # --- E5 ---
    s5 = {
        "max_abs_diff_zero_mul": max(r["max_abs_diff_zero_mul"] for r in e5),
        "memory_ratio_min": min(r["memory_ratio_f32_over_packed"] for r in e5),
        "dim256_zero_mul_us_mean": statistics.mean(r["dim256_zero_mul_us"] for r in e5),
        "dim256_f32_dense_us_mean": statistics.mean(r["dim256_f32_dense_us"] for r in e5),
        "dim1024_zero_mul_us_mean": statistics.mean(r["dim1024_zero_mul_us"] for r in e5),
        "dim1024_f32_dense_us_mean": statistics.mean(r["dim1024_f32_dense_us"] for r in e5),
        "dim1024_packed_bytes": e5[0]["dim1024_packed_bytes"],
        "dim1024_f32_bytes": e5[0]["dim1024_f32_bytes"],
    }

    runtime = json.loads((raw_root / phase / "run.json").read_text())
    return {"e1": s1, "e2": s2, "e3": s3, "e4": s4, "e5": s5, "runtime": runtime}, (e1, e2, e3, e4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phases", nargs="+", default=["DEV", "CONFIRM"])
    p.add_argument("--raw", default="docs/aura_v1/raw")
    p.add_argument("--out", default="docs/aura_v1/summary.json")
    a = p.parse_args()

    raw_root = ROOT / a.raw
    summary = {
        "bootstrap": {
            "resamples": 10000,
            "seed": 12345,
            "delta": 0.05,
            "unit": "16 independent seeds per phase; paired environments within each seed",
            "method": "percentile with linearly interpolated quantiles",
        },
        "phases": {},
    }

    all_rows = {1: [], 2: [], 3: [], 4: []}
    for phase in a.phases:
        res, (e1, e2, e3, e4) = analyze_phase(raw_root, phase)
        summary["phases"][phase] = res
        all_rows[1].extend(e1)
        all_rows[2].extend(e2)
        all_rows[3].extend(e3)
        all_rows[4].extend(e4)

    # Escribir CSVs consolidados solo si se incluye CONFIRM o se corre completo.
    for idx in (1, 2, 3, 4):
        rows = all_rows[idx]
        if rows:
            with (DOC / f"metrics_e{idx}.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)

    if "CONFIRM" in summary["phases"]:
        c = summary["phases"]["CONFIRM"]
        verdicts = {
            "H1a_shift_recovery": c["e1"]["paired"]["H1a_shift_err_full_minus_no_fast"]["ci95"][1] < 0.0,
            "H1b_hippocampal_recall": c["e1"]["paired"]["H1b_recall_err_full_minus_no_hip"]["ci95"][1] < 0.0,
            "H1c_anti_collapse_norm": c["e1"]["paired"]["H1c_norm_full_minus_no_vicreg"]["ci95"][0] > 0.0
            and c["e1"]["arms"]["aura_full"]["effective_rank"] > 8.0,
            "H1d_entropy_compression": c["e1"]["arms"]["aura_full"]["compression_ratio"] > 1.2
            and c["e1"]["paired"]["H1d_compression_full_minus_fixed"]["ci95"][0] > 0.0,
            "H2a_active_world_vs_sys1": c["e2"]["paired"]["H2a_combined_full_minus_sys1"]["ci95"][0] > 0.0,
            "H2a_active_world_vs_no_fast": c["e2"]["paired"]["H2a_combined_full_minus_no_fast"]["ci95"][0] > 0.0,
            "H2b_abstain_calibration": c["e2"]["paired"]["H2b_abstain_trap_minus_familiar"]["ci95"][0] > 0.0,
            "H3a_sleep_combined_vs_no_replay": c["e3"]["paired"]["H3a_combined_full_minus_no_replay"]["ci95"][0] > 0.0,
            "H3a_sleep_forgetting_vs_no_replay": c["e3"]["paired"]["H3a_forgetting_full_minus_no_replay"]["ci95"][1] < 0.0,
            "H3b_hysteresis_combined": c["e3"]["paired"]["H3b_combined_full_minus_no_hysteresis"]["ci95"][0] > 0.0,
            "H4a_cortical_fe_vs_frozen": c["e4"]["paired"]["H4a_fe_late_full_minus_frozen"]["ci95"][1] < 0.0
            and 1.0 <= c["e4"]["arms"]["cortical_full"]["mean_spike_rate_pct"] <= 25.0,
            "H4b_cortical_ne_spike": c["e4"]["paired"]["H4b_ne_shift_delta_full"]["ci95"][0] > 0.0,
            "E5_zero_mul_exact_16x": c["e5"]["max_abs_diff_zero_mul"] == 0 and c["e5"]["memory_ratio_min"] == 16,
        }
        summary["confirm_verdicts"] = verdicts

    out_path = ROOT / a.out
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
