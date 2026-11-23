#!/usr/bin/env python3
"""Paired bootstrap (10,000 resamples, seed 12345, alpha=0.05) evaluator for AURA v2 B1-B6."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "aura_v2"
N_BOOT = 10_000
BOOT_SEED = 12345


def load_phase_csv(phase_dir: Path, fname: str) -> list[dict]:
    rows = []
    for seed_dir in sorted(phase_dir.glob("0x*")):
        p = seed_dir / fname
        if not p.exists():
            continue
        with p.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def paired_bootstrap(diffs: np.ndarray, seed: int = BOOT_SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    mean = float(np.mean(diffs))
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "n": int(n),
        "mean": mean,
        "std": float(np.std(diffs, ddof=1)) if n > 1 else 0.0,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
    }


def group_by_arm(rows: list[dict], metric: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for r in rows:
        arm = r["arm"]
        seed = int(r["seed"])
        val = float(r[metric])
        out.setdefault(arm, {})[seed] = val
    return out


def paired_diff(rows: list[dict], metric: str, arm_a: str, arm_b: str) -> dict:
    g = group_by_arm(rows, metric)
    seeds = sorted(set(g[arm_a].keys()) & set(g[arm_b].keys()))
    diffs = np.array([g[arm_a][s] - g[arm_b][s] for s in seeds], dtype=np.float64)
    return paired_bootstrap(diffs)


def paired_ratio(rows: list[dict], metric: str, arm_num: str, arm_den: str) -> dict:
    g = group_by_arm(rows, metric)
    seeds = sorted(set(g[arm_num].keys()) & set(g[arm_den].keys()))
    ratios = np.array([g[arm_num][s] / max(g[arm_den][s], 1e-12) for s in seeds], dtype=np.float64)
    return paired_bootstrap(ratios)


def single_arm_stat(rows: list[dict], metric: str, arm: str) -> dict:
    g = group_by_arm(rows, metric)
    seeds = sorted(g[arm].keys())
    vals = np.array([g[arm][s] for s in seeds], dtype=np.float64)
    return paired_bootstrap(vals)


def within_arm_diff(rows: list[dict], arm: str, col_a: str, col_b: str) -> dict:
    ga = group_by_arm(rows, col_a)[arm]
    gb = group_by_arm(rows, col_b)[arm]
    seeds = sorted(set(ga.keys()) & set(gb.keys()))
    diffs = np.array([ga[s] - gb[s] for s in seeds], dtype=np.float64)
    return paired_bootstrap(diffs)


def arm_means(rows: list[dict], numeric_cols: list[str]) -> dict:
    arms = sorted(set(r["arm"] for r in rows))
    out = {}
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        out[a] = {c: float(np.mean([float(r[c]) for r in sub])) for c in numeric_cols}
    return out


def analyze_phase(phase_dir: Path) -> dict:
    b1 = load_phase_csv(phase_dir, "b1.csv")
    b2 = load_phase_csv(phase_dir, "b2.csv")
    b3 = load_phase_csv(phase_dir, "b3.csv")
    b4 = load_phase_csv(phase_dir, "b4.csv")
    b5 = load_phase_csv(phase_dir, "b5.csv")
    b6 = load_phase_csv(phase_dir, "b6.csv")

    res = {}

    # B1: Silicon Mismatch
    b1_cols = [
        "max_abs_diff", "mem_ratio_1024", "us_256_d67_naive", "us_256_d67_f32", "us_256_d67_lut4_avx2",
        "us_1024_d30_naive", "us_1024_d30_f32", "us_1024_d30_lut4_avx2",
        "us_1024_d67_naive", "us_1024_d67_f32", "us_1024_d67_lut4_avx2",
        "us_1024_b16_naive", "us_1024_b16_f32", "us_1024_b16_lut4_tiled",
        "speedup_b16_vs_naive", "speedup_b16_vs_f32",
        "fabric_scatter_tps", "fabric_tiled_tps", "fabric_tiled_speedup",
        "fw_naive_ups", "fw_nlms_ups", "fw_nlms_speedup", "fw_naive_large_err", "fw_nlms_large_err",
    ]
    b1_means = {c: float(np.mean([float(r[c]) for r in b1])) for c in b1_cols}
    b1_max_diff = int(max(int(r["max_abs_diff"]) for r in b1))
    b1_sp_naive = paired_bootstrap(np.array([float(r["speedup_b16_vs_naive"]) for r in b1]))
    b1_sp_f32 = paired_bootstrap(np.array([float(r["speedup_b16_vs_f32"]) for r in b1]))
    b1_fab_sp = paired_bootstrap(np.array([float(r["fabric_tiled_speedup"]) for r in b1]))
    b1_fw_sp = paired_bootstrap(np.array([float(r["fw_nlms_speedup"]) for r in b1]))
    b1_fw_err_diff = paired_bootstrap(
        np.array([float(r["fw_nlms_large_err"]) - float(r["fw_naive_large_err"]) for r in b1])
    )
    res["b1"] = {
        "means": b1_means,
        "max_abs_diff": b1_max_diff,
        "speedup_b16_vs_naive": b1_sp_naive,
        "speedup_b16_vs_f32": b1_sp_f32,
        "fabric_tiled_speedup": b1_fab_sp,
        "fw_nlms_speedup": b1_fw_sp,
        "fw_large_err_diff": b1_fw_err_diff,
    }

    # B2: Settling Time & Parallel Scan
    b2_cols = [
        "avg_ticks_per_window", "total_ticks", "wall_ms", "late_fe", "late_sensory_fe",
        "firing_rate_pct", "ne_pre_shift", "ne_post_shift", "ne_shift_delta",
        "scan_fwd_max_diff", "scan_bwd_max_diff",
    ]
    b2_tick_speedup = paired_ratio(b2, "avg_ticks_per_window", "naive_cold_k16", "amortized_early_exit")
    b2_fe_vs_cold16 = paired_diff(b2, "late_fe", "amortized_early_exit", "naive_cold_k16")
    b2_fe_vs_cold4 = paired_diff(b2, "late_fe", "amortized_early_exit", "cold_k4_truncated")
    b2_fe_vs_frozen = paired_diff(b2, "late_fe", "amortized_early_exit", "frozen_control")
    b2_fe_ratio_no_nm = paired_ratio(b2, "late_fe", "amortized_early_exit", "no_neuromod_control")
    b2_ne_delta = single_arm_stat(b2, "ne_shift_delta", "amortized_early_exit")
    b2_scan_fwd_max = float(max(float(r["scan_fwd_max_diff"]) for r in b2))
    b2_scan_bwd_max = float(max(float(r["scan_bwd_max_diff"]) for r in b2))
    res["b2"] = {
        "means": arm_means(b2, b2_cols),
        "tick_speedup_vs_cold16": b2_tick_speedup,
        "fe_vs_cold16": b2_fe_vs_cold16,
        "fe_vs_cold4": b2_fe_vs_cold4,
        "fe_vs_frozen": b2_fe_vs_frozen,
        "fe_ratio_vs_no_neuromod": b2_fe_ratio_no_nm,
        "ne_shift_delta": b2_ne_delta,
        "scan_fwd_max_diff": b2_scan_fwd_max,
        "scan_bwd_max_diff": b2_scan_bwd_max,
    }

    # B3: Capacity Ceiling & Sleep Anti-Forgetting
    b3_cols = ["acc_a_pre", "acc_a_post", "forgetting_a", "acc_b_post", "combined_ab", "weight_flips", "nonzero_frac"]
    b3_vs_v1 = paired_diff(b3, "combined_ab", "v2_aura_sleep_full", "v1_naive_ternary")
    b3_comb_vs_norep = paired_diff(b3, "combined_ab", "v2_aura_sleep_full", "v2_bitnet_ste_no_replay")
    b3_forg_vs_norep = paired_diff(b3, "forgetting_a", "v2_aura_sleep_full", "v2_bitnet_ste_no_replay")
    b3_flips_vs_nohyst = paired_diff(b3, "weight_flips", "v2_aura_sleep_full", "v2_bitnet_ste_no_hysteresis")
    b3_forg_vs_nohyst = paired_diff(b3, "forgetting_a", "v2_aura_sleep_full", "v2_bitnet_ste_no_hysteresis")
    b3_comb_vs_nohyst = paired_diff(b3, "combined_ab", "v2_aura_sleep_full", "v2_bitnet_ste_no_hysteresis")
    res["b3"] = {
        "means": arm_means(b3, b3_cols),
        "combined_vs_v1": b3_vs_v1,
        "combined_vs_no_replay": b3_comb_vs_norep,
        "forgetting_vs_no_replay": b3_forg_vs_norep,
        "flips_vs_no_hysteresis": b3_flips_vs_nohyst,
        "forgetting_vs_no_hysteresis": b3_forg_vs_nohyst,
        "combined_vs_no_hysteresis": b3_comb_vs_nohyst,
    }

    # B4: 5,120-Byte Long-Horizon Credit Assignment
    b4_cols = ["acc_d128", "acc_d512", "acc_d2048", "acc_d5120", "hipp_store_frac_pct"]
    b4_vs_local = paired_diff(b4, "acc_d5120", "aura_v2_hipp_credit", "local_trace_only")
    b4_vs_ssm = paired_diff(b4, "acc_d5120", "aura_v2_hipp_credit", "ssm_only_no_hipp")
    b4_store_frac = single_arm_stat(b4, "hipp_store_frac_pct", "aura_v2_hipp_credit")
    res["b4"] = {
        "means": arm_means(b4, b4_cols),
        "d5120_vs_local": b4_vs_local,
        "d5120_vs_ssm": b4_vs_ssm,
        "hipp_store_frac_pct": b4_store_frac,
    }

    # B5: Latent Stability, TV-Static Trap & Phase-Aware Patching
    b5_cols = [
        "total_patches", "compression_ratio", "bpp_predictable", "bpp_static",
        "p1_err", "p2_err", "p4a_err", "effective_rank",
        "familiar_reducible_epi", "static_reducible_epi", "ood_reducible_epi",
        "static_aleatoric", "ood_aleatoric", "static_hipp_writes", "ood_hipp_writes",
    ]
    b5_ood_vs_static = within_arm_diff(b5, "aura_v2_full", "ood_reducible_epi", "static_reducible_epi")
    b5_writes_stat_vs_ood = within_arm_diff(b5, "aura_v2_full", "static_hipp_writes", "ood_hipp_writes")
    b5_p2_vs_v1 = paired_diff(b5, "p2_err", "aura_v2_full", "aura_v1_baseline")
    b5_p1_vs_v1 = paired_diff(b5, "p1_err", "aura_v2_full", "aura_v1_baseline")
    b5_p2_vs_fixed = paired_diff(b5, "p2_err", "aura_v2_full", "fixed_patcher_4")
    b5_erank_vs_novic = paired_diff(b5, "effective_rank", "aura_v2_full", "no_vicreg")
    b5_comp = single_arm_stat(b5, "compression_ratio", "aura_v2_full")
    res["b5"] = {
        "means": arm_means(b5, b5_cols),
        "ood_minus_static_epi": b5_ood_vs_static,
        "static_minus_ood_writes": b5_writes_stat_vs_ood,
        "p2_err_vs_v1": b5_p2_vs_v1,
        "p1_err_vs_v1": b5_p1_vs_v1,
        "p2_err_vs_fixed": b5_p2_vs_fixed,
        "erank_vs_no_vicreg": b5_erank_vs_novic,
        "compression_ratio": b5_comp,
    }

    # B6: End-to-End Real Byte Training + Online Markdown -> Rust Domain Shift
    b6_cols = [
        "train_init_bpb", "train_final_bpb", "eval_md_bpb", "shift_rust_bpb",
        "compression_ratio", "effective_rank", "train_bytes_per_sec",
        "packed_ternary_bytes", "dense_f32_bytes", "total_flips",
    ]
    b6_md_vs_v1 = paired_diff(b6, "eval_md_bpb", "aura_v2_online_full", "v1_unscaled_ternary")
    b6_md_vs_bigram = paired_diff(b6, "eval_md_bpb", "aura_v2_online_full", "bigram_byte_baseline")
    b6_rust_online_vs_frozen = paired_diff(b6, "shift_rust_bpb", "aura_v2_online_full", "aura_v2_frozen_infer")
    b6_comp = single_arm_stat(b6, "compression_ratio", "aura_v2_online_full")
    b6_erank = single_arm_stat(b6, "effective_rank", "aura_v2_online_full")
    res["b6"] = {
        "means": arm_means(b6, b6_cols),
        "eval_md_vs_v1": b6_md_vs_v1,
        "eval_md_vs_bigram": b6_md_vs_bigram,
        "shift_rust_online_vs_frozen": b6_rust_online_vs_frozen,
        "compression_ratio": b6_comp,
        "effective_rank": b6_erank,
    }

    # Preregistered Verdicts (17 criteria H1a-H6c matching docs/aura_v2/PLAN.md)
    verdicts = {
        "H1a_exact_parity_and_16x_mem": (b1_max_diff == 0) and (abs(b1_means["mem_ratio_1024"] - 16.0) < 1e-6),
        "H1b_lut4_b16_speedup_gt_3x_naive_and_1.5x_f32": (b1_sp_naive["ci95_lo"] > 3.0) and (b1_sp_f32["ci95_lo"] > 1.5),
        "H1c_tiled_csr_and_nlms_speedup_and_stability": (b1_fab_sp["ci95_lo"] > 1.0) and (b1_fw_sp["ci95_lo"] > 1.0) and (b1_fw_err_diff["ci95_hi"] < 0.0),
        "H2a_amortized_settling_ge_2.5x_ticks_and_lower_fe_vs_cold4_frozen": (
            b2_tick_speedup["ci95_lo"] >= 2.5
            and b2_fe_vs_cold4["ci95_hi"] < 0.0
            and b2_fe_vs_frozen["ci95_hi"] < 0.0
        ),
        "H2b_ne_shift_burst_and_no_neuromod_paradox_fixed": (
            b2_ne_delta["ci95_lo"] > 0.0
            and b2_fe_ratio_no_nm["ci95_hi"] <= 1.05
        ),
        "H2c_blelloch_parallel_scan_exact_parity": (b2_scan_fwd_max < 1e-5) and (b2_scan_bwd_max < 1e-5),
        "H3a_break_v1_capacity_ceiling_gt_10pp": b3_vs_v1["ci95_lo"] > 10.0,
        "H3b_sleep_replay_reduces_forgetting_and_boosts_combined": (
            b3_forg_vs_norep["ci95_hi"] < 0.0 and b3_comb_vs_norep["ci95_lo"] > 0.0
        ),
        "H3c_importance_hysteresis_reduces_flips_and_forgetting": (
            b3_flips_vs_nohyst["ci95_hi"] < 0.0 and b3_forg_vs_nohyst["ci95_hi"] < 0.0
        ),
        "H4a_long_horizon_5120B_beats_local_and_ssm": (
            b4_vs_local["ci95_lo"] > 40.0 and b4_vs_ssm["ci95_lo"] > 15.0
        ),
        "H4b_sublinear_hippocampus_storage_lt_5pct": b4_store_frac["ci95_hi"] < 5.0,
        "H5a_solves_tv_static_trap_and_write_gating": (
            b5_ood_vs_static["ci95_lo"] > 0.0 and b5_writes_stat_vs_ood["ci95_hi"] < 0.0
        ),
        "H5b_phase_aware_compression_and_beats_v1_error": (
            b5_comp["ci95_lo"] >= 2.5 and b5_p2_vs_v1["ci95_hi"] < 0.0
        ),
        "H5c_vicreg_prevents_dimensional_collapse": b5_erank_vs_novic["ci95_lo"] > 0.0,
        "H6a_real_md_bpb_beats_v1_and_bigram": (
            b6_md_vs_v1["ci95_hi"] < 0.0 and b6_md_vs_bigram["ci95_hi"] < 0.0
        ),
        "H6b_online_rust_shift_beats_frozen": b6_rust_online_vs_frozen["ci95_hi"] < 0.0,
        "H6c_real_text_compression_and_healthy_erank": (
            b6_comp["ci95_lo"] > 1.5 and b6_erank["ci95_lo"] > 8.0
        ),
    }
    res["verdicts"] = verdicts
    res["verdict_pass_count"] = sum(1 for v in verdicts.values() if v)
    res["verdict_total"] = len(verdicts)
    return res


def export_consolidated_csvs(raw_root: Path, out_dir: Path, phases: list[str]):
    for b in ["b1", "b2", "b3", "b4", "b5", "b6"]:
        all_rows = []
        for ph in phases:
            pdir = raw_root / ph
            if not pdir.exists():
                continue
            for r in load_phase_csv(pdir, f"{b}.csv"):
                r2 = {"phase": ph}
                r2.update(r)
                all_rows.append(r2)
        if not all_rows:
            continue
        out_p = out_dir / f"metrics_{b}.csv"
        with out_p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)


def print_summary(phase: str, data: dict):
    print(f"\n================ {phase} SUMMARY ({data['verdict_pass_count']}/{data['verdict_total']} PASS) ================")
    b1 = data["b1"]
    print(
        f"[B1] diff={b1['max_abs_diff']} mem={b1['means']['mem_ratio_1024']:.1f}x | "
        f"1024 B=16: lut4={b1['means']['us_1024_b16_lut4_tiled']:.1f}us vs naive={b1['means']['us_1024_b16_naive']:.1f}us "
        f"({b1['speedup_b16_vs_naive']['mean']:.2f}x [{b1['speedup_b16_vs_naive']['ci95_lo']:.2f},{b1['speedup_b16_vs_naive']['ci95_hi']:.2f}]) "
        f"vs f32={b1['means']['us_1024_b16_f32']:.1f}us "
        f"({b1['speedup_b16_vs_f32']['mean']:.2f}x [{b1['speedup_b16_vs_f32']['ci95_lo']:.2f},{b1['speedup_b16_vs_f32']['ci95_hi']:.2f}]) | "
        f"CSR tiled={b1['fabric_tiled_speedup']['mean']:.2f}x | NLMS={b1['fw_nlms_speedup']['mean']:.2f}x "
        f"(err {b1['means']['fw_nlms_large_err']:.0f} vs {b1['means']['fw_naive_large_err']:.0f})"
    )
    b2 = data["b2"]
    print(
        f"[B2] ticks: amort={b2['means']['amortized_early_exit']['avg_ticks_per_window']:.2f} vs cold16=16.0 "
        f"({b2['tick_speedup_vs_cold16']['mean']:.2f}x) | FE: amort={b2['means']['amortized_early_exit']['late_fe']:.1f} "
        f"cold16={b2['means']['naive_cold_k16']['late_fe']:.1f} cold4={b2['means']['cold_k4_truncated']['late_fe']:.1f} "
        f"frozen={b2['means']['frozen_control']['late_fe']:.1f} no_nm={b2['means']['no_neuromod_control']['late_fe']:.1f} | "
        f"NE delta={b2['ne_shift_delta']['mean']:+.2f} [{b2['ne_shift_delta']['ci95_lo']:+.2f},{b2['ne_shift_delta']['ci95_hi']:+.2f}] | "
        f"scan_diff=({b2['scan_fwd_max_diff']:.2e},{b2['scan_bwd_max_diff']:.2e})"
    )
    b3 = data["b3"]
    print(
        f"[B3] combined A+B: v2_full={b3['means']['v2_aura_sleep_full']['combined_ab']:.2f}% "
        f"v1={b3['means']['v1_naive_ternary']['combined_ab']:.2f}% "
        f"no_rep={b3['means']['v2_bitnet_ste_no_replay']['combined_ab']:.2f}% "
        f"no_hyst={b3['means']['v2_bitnet_ste_no_hysteresis']['combined_ab']:.2f}% | "
        f"forgetting_A: full={b3['means']['v2_aura_sleep_full']['forgetting_a']:.2f}pp "
        f"no_rep={b3['means']['v2_bitnet_ste_no_replay']['forgetting_a']:.2f}pp "
        f"no_hyst={b3['means']['v2_bitnet_ste_no_hysteresis']['forgetting_a']:.2f}pp | "
        f"flips: full={b3['means']['v2_aura_sleep_full']['weight_flips']:.0f} vs no_hyst={b3['means']['v2_bitnet_ste_no_hysteresis']['weight_flips']:.0f}"
    )
    b4 = data["b4"]
    print(
        f"[B4] D=5120B acc: aura_v2={b4['means']['aura_v2_hipp_credit']['acc_d5120']:.2f}% "
        f"ssm_only={b4['means']['ssm_only_no_hipp']['acc_d5120']:.2f}% "
        f"local_trace={b4['means']['local_trace_only']['acc_d5120']:.2f}% | "
        f"hipp_store={b4['means']['aura_v2_hipp_credit']['hipp_store_frac_pct']:.3f}%"
    )
    b5 = data["b5"]
    print(
        f"[B5] TV-static fix: ood_epi={b5['means']['aura_v2_full']['ood_reducible_epi']:.1f} vs "
        f"static_epi={b5['means']['aura_v2_full']['static_reducible_epi']:.1f} "
        f"(writes: static={b5['means']['aura_v2_full']['static_hipp_writes']:.1f} vs ood={b5['means']['aura_v2_full']['ood_hipp_writes']:.1f}) | "
        f"comp={b5['means']['aura_v2_full']['compression_ratio']:.2f}x | "
        f"p2_err: v2={b5['means']['aura_v2_full']['p2_err']:.4f} v1={b5['means']['aura_v1_baseline']['p2_err']:.4f} "
        f"fixed4={b5['means']['fixed_patcher_4']['p2_err']:.4f} | "
        f"erank: v2={b5['means']['aura_v2_full']['effective_rank']:.2f} no_vic={b5['means']['no_vicreg']['effective_rank']:.2f}"
    )
    b6 = data["b6"]
    print(
        f"[B6] Real MD BPB: v2={b6['means']['aura_v2_online_full']['eval_md_bpb']:.4f} "
        f"bigram={b6['means']['bigram_byte_baseline']['eval_md_bpb']:.4f} "
        f"v1_unscaled={b6['means']['v1_unscaled_ternary']['eval_md_bpb']:.4f} | "
        f"Rust shift BPB: online={b6['means']['aura_v2_online_full']['shift_rust_bpb']:.4f} vs "
        f"frozen={b6['means']['aura_v2_frozen_infer']['shift_rust_bpb']:.4f} | "
        f"comp={b6['means']['aura_v2_online_full']['compression_ratio']:.2f}x erank={b6['means']['aura_v2_online_full']['effective_rank']:.2f} "
        f"({b6['means']['aura_v2_online_full']['train_bytes_per_sec']:.0f} B/s)"
    )
    for k, ok in data["verdicts"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DOCS_DIR / "raw")
    ap.add_argument("--out", type=Path, default=DOCS_DIR / "summary.json")
    ap.add_argument("--phases", nargs="+", default=["DEV", "CONFIRM"])
    args = ap.parse_args()

    out_summary = {}
    present = []
    for ph in args.phases:
        pdir = args.raw / ph
        if pdir.exists() and list(pdir.glob("0x*")):
            present.append(ph)
            out_summary[ph] = analyze_phase(pdir)
            print_summary(ph, out_summary[ph])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_summary, indent=2) + "\n")
    export_consolidated_csvs(args.raw, args.out.parent, present)
    print(f"\nSaved summary -> {args.out}")


if __name__ == "__main__":
    main()
