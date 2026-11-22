#!/usr/bin/env python3
"""Statistical analysis (paired bootstrap 95% CI, 4000 resamples) and preregistered verdicts
(15 criteria H_G1a-H_G5c) for AURA v3 Genesis."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from run import REPO_ROOT

DOCS_V3_DIR = REPO_ROOT / "docs" / "aura_v3"


def bootstrap_ci(vals: np.ndarray, n_boot: int = 4000, seed: int = 202603) -> dict:
    vals = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(vals)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(n),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if n > 1 else 0.0,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
    }


def load_phase_csv(phase_dir: Path, filename: str) -> list[dict]:
    rows = []
    for sd in sorted(phase_dir.iterdir()):
        if not sd.is_dir():
            continue
        p = sd / filename
        if p.exists():
            with p.open() as f:
                rows.extend(csv.DictReader(f))
    return rows


def by_seed_arm(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["seed"], r["arm"]): r for r in rows}


def seeds_of(rows: list[dict]) -> list[str]:
    return sorted({r["seed"] for r in rows}, key=lambda x: int(x))


def paired_diff(rows: list[dict], col: str, arm_a: str, arm_b: str) -> dict:
    m = by_seed_arm(rows)
    vals = [float(m[(s, arm_a)][col]) - float(m[(s, arm_b)][col]) for s in seeds_of(rows)]
    return bootstrap_ci(np.array(vals))


def paired_abs_diff(rows: list[dict], col: str, arm_a: str, arm_b: str) -> dict:
    m = by_seed_arm(rows)
    vals = [abs(float(m[(s, arm_a)][col]) - float(m[(s, arm_b)][col])) for s in seeds_of(rows)]
    return bootstrap_ci(np.array(vals))


def paired_ratio(rows: list[dict], col: str, arm_a: str, arm_b: str) -> dict:
    m = by_seed_arm(rows)
    vals = [float(m[(s, arm_a)][col]) / max(float(m[(s, arm_b)][col]), 1e-12) for s in seeds_of(rows)]
    return bootstrap_ci(np.array(vals))


def single_arm_stat(rows: list[dict], col: str, arm: str) -> dict:
    m = by_seed_arm(rows)
    vals = [float(m[(s, arm)][col]) for s in seeds_of(rows)]
    return bootstrap_ci(np.array(vals))


def arm_means(rows: list[dict], cols: list[str]) -> dict:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["arm"]].append(r)
    out = {}
    for arm, rlist in sorted(grouped.items()):
        out[arm] = {c: float(np.mean([float(r[c]) for r in rlist])) for c in cols}
    return out


def analyze_phase(phase_dir: Path) -> dict:
    g1 = load_phase_csv(phase_dir, "g1.csv")
    g2 = load_phase_csv(phase_dir, "g2.csv")
    g3 = load_phase_csv(phase_dir, "g3.csv")
    g4 = load_phase_csv(phase_dir, "g4.csv")
    g5f = load_phase_csv(phase_dir, "g5_fabric.csv")
    g5b = load_phase_csv(phase_dir, "g5_bytes.csv")

    res = {}

    # G1: MorphogenRouter (Subroutine 1)
    g1_cols = [
        "genome_bytes", "csr_bytes_2048", "compression_ratio_2048",
        "topo_corr_2048", "spatial_late_fe", "spatial_sensory_fe", "avg_ticks",
    ]
    g1_comp = single_arm_stat(g1, "compression_ratio_2048", "morphogen_router_v3")
    g1_gbytes = single_arm_stat(g1, "genome_bytes", "morphogen_router_v3")
    g1_topo = single_arm_stat(g1, "topo_corr_2048", "morphogen_router_v3")
    g1_topo_diff = paired_diff(g1, "topo_corr_2048", "morphogen_router_v3", "random_v2_wiring")
    g1_fe_diff = paired_diff(g1, "spatial_late_fe", "morphogen_router_v3", "random_v2_wiring")
    res["g1"] = {
        "means": arm_means(g1, g1_cols),
        "genome_bytes": g1_gbytes,
        "compression_ratio_2048": g1_comp,
        "topo_corr_2048": g1_topo,
        "topo_diff_vs_random": g1_topo_diff,
        "spatial_fe_diff_vs_random": g1_fe_diff,
    }

    # G2: Canonical Cortical Microcircuit & Mriganka Sur Re-Wiring (Subroutine 2)
    g2_cols = ["template_bytes", "test_acc_pct", "osi_score"]
    g2_v1 = single_arm_stat(g2, "test_acc_pct", "v1_canonical_visual")
    g2_a1 = single_arm_stat(g2, "test_acc_pct", "a1_canonical_audio")
    g2_lang = single_arm_stat(g2, "test_acc_pct", "lang_canonical_bytes")
    g2_rew_osi = single_arm_stat(g2, "osi_score", "a1_rewired_visual_sur")
    g2_rew_vs_rand = paired_diff(g2, "test_acc_pct", "a1_rewired_visual_sur", "random_unstructured_visual")
    g2_rew_vs_v1_abs = paired_abs_diff(g2, "test_acc_pct", "a1_rewired_visual_sur", "v1_canonical_visual")
    res["g2"] = {
        "means": arm_means(g2, g2_cols),
        "acc_v1": g2_v1,
        "acc_a1_audio": g2_a1,
        "acc_lang_bytes": g2_lang,
        "osi_a1_rewired": g2_rew_osi,
        "acc_rewired_minus_random": g2_rew_vs_rand,
        "abs_diff_rewired_vs_v1": g2_rew_vs_v1_abs,
    }

    # G3: Conel Overproduction + Microglia Pruning + Myelination (Subroutine 3)
    g3_cols = [
        "synapses_newborn", "synapses_peak", "synapses_final", "csr_bytes_final",
        "prune_frac", "mean_axon_delay", "eval_fe", "eval_sensory_fe",
        "avg_ticks", "throughput_tps_2048",
    ]
    g3_means = arm_means(g3, g3_cols)
    g3_prune = single_arm_stat(g3, "prune_frac", "compiled_myelinated")
    g3_fe_vs_sparse = paired_diff(g3, "eval_fe", "compiled_myelinated", "sparse_from_scratch")
    g3_fe_vs_peak = paired_diff(g3, "eval_fe", "compiled_myelinated", "peak_unpruned")
    g3_ticks_vs_peak = paired_diff(g3, "avg_ticks", "compiled_myelinated", "peak_unpruned")
    g3_tps_ratio = paired_ratio(g3, "throughput_tps_2048", "compiled_myelinated", "peak_unpruned")
    res["g3"] = {
        "means": g3_means,
        "prune_frac": g3_prune,
        "fe_vs_sparse_scratch": g3_fe_vs_sparse,
        "fe_vs_peak": g3_fe_vs_peak,
        "ticks_vs_peak": g3_ticks_vs_peak,
        "tps_ratio_vs_peak": g3_tps_ratio,
    }

    # G4: Prenatal Synthetic Curriculum (Subroutine 4)
    g4_cols = ["osi_at_birth", "early_visual_fe", "early_sm_err"]
    g4_osi_vs_none = paired_diff(g4, "osi_at_birth", "prenatal_waves_and_twitches", "no_prenatal")
    g4_fe_vs_none = paired_diff(g4, "early_visual_fe", "prenatal_waves_and_twitches", "no_prenatal")
    g4_osi_vs_noise = paired_diff(g4, "osi_at_birth", "prenatal_waves_and_twitches", "white_noise_prenatal")
    g4_fe_vs_noise = paired_diff(g4, "early_visual_fe", "prenatal_waves_and_twitches", "white_noise_prenatal")
    g4_sm_vs_none = paired_diff(g4, "early_sm_err", "prenatal_waves_and_twitches", "no_prenatal")
    g4_sm_vs_noise = paired_diff(g4, "early_sm_err", "prenatal_waves_and_twitches", "white_noise_prenatal")
    res["g4"] = {
        "means": arm_means(g4, g4_cols),
        "osi_vs_none": g4_osi_vs_none,
        "early_fe_vs_none": g4_fe_vs_none,
        "osi_vs_white_noise": g4_osi_vs_noise,
        "early_fe_vs_white_noise": g4_fe_vs_noise,
        "sm_err_vs_none": g4_sm_vs_none,
        "sm_err_vs_white_noise": g4_sm_vs_noise,
    }

    # G5a: CorticalFabric v3 Genesis vs v2
    g5f_cols = ["genome_comp_2048", "synapses_final", "early_fe", "late_fe", "avg_ticks"]
    g5f_early_diff = paired_diff(g5f, "early_fe", "aura_v3_genesis_fabric", "aura_v2_fabric")
    g5f_late_diff = paired_diff(g5f, "late_fe", "aura_v3_genesis_fabric", "aura_v2_fabric")
    g5f_ticks_diff = paired_diff(g5f, "avg_ticks", "aura_v3_genesis_fabric", "aura_v2_fabric")
    g5f_comp = single_arm_stat(g5f, "genome_comp_2048", "aura_v3_genesis_fabric")
    res["g5_fabric"] = {
        "means": arm_means(g5f, g5f_cols),
        "early_fe_v3_minus_v2": g5f_early_diff,
        "late_fe_v3_minus_v2": g5f_late_diff,
        "ticks_v3_minus_v2": g5f_ticks_diff,
        "genome_comp_2048": g5f_comp,
    }

    # G5b: AuraByteModel v3 Genesis vs v2 on Real Repo Markdown & Rust
    g5b_cols = [
        "early_md_bpb", "eval_md_bpb", "shift_rust_bpb", "nonzero_peak_frac",
        "nonzero_final_frac", "pruned_synapses", "compression_ratio",
        "effective_rank", "train_bytes_per_sec",
    ]
    g5b_early_vs_v2 = paired_diff(g5b, "early_md_bpb", "aura_v3_genesis_full", "aura_v2_baseline")
    g5b_final_vs_v2 = paired_diff(g5b, "eval_md_bpb", "aura_v3_genesis_full", "aura_v2_baseline")
    g5b_final_vs_bg = paired_diff(g5b, "eval_md_bpb", "aura_v3_genesis_full", "bigram_byte_baseline")
    g5b_nz_vs_v2 = paired_diff(g5b, "nonzero_final_frac", "aura_v3_genesis_full", "aura_v2_baseline")
    g5b_rust_online_vs_frozen = paired_diff(g5b, "shift_rust_bpb", "aura_v3_genesis_full", "aura_v3_genesis_frozen_infer")
    g5b_rust_vs_bg = paired_diff(g5b, "shift_rust_bpb", "aura_v3_genesis_full", "bigram_byte_baseline")
    res["g5_bytes"] = {
        "means": arm_means(g5b, g5b_cols),
        "early_md_v3_minus_v2": g5b_early_vs_v2,
        "eval_md_v3_minus_v2": g5b_final_vs_v2,
        "eval_md_v3_minus_bigram": g5b_final_vs_bg,
        "nonzero_v3_minus_v2": g5b_nz_vs_v2,
        "rust_online_minus_frozen": g5b_rust_online_vs_frozen,
        "rust_v3_minus_bigram": g5b_rust_vs_bg,
    }

    # 15 Preregistered Criteria (H_G1a - H_G5c matching docs/aura_v3/PLAN.md)
    m_comp = g3_means["compiled_myelinated"]
    verdicts = {
        "H_G1a_genome_le_256B_and_comp_gt_1000x": (
            g1_gbytes["mean"] <= 256.0 and g1_comp["ci95_lo"] > 1000.0
        ),
        "H_G1b_retinotopic_corr_gt_0.85_and_beats_random": (
            g1_topo["ci95_lo"] > 0.85 and g1_topo_diff["ci95_lo"] > 0.80
        ),
        "H_G1c_morphogen_spatial_fe_beats_random": g1_fe_diff["ci95_hi"] < 0.0,
        "H_G2a_canonical_universal_3_modalities_gt_70pct": (
            g2_v1["ci95_lo"] > 70.0 and g2_a1["ci95_lo"] > 70.0 and g2_lang["ci95_lo"] > 70.0
        ),
        "H_G2b_sur_rewiring_osi_gt_0.35_and_beats_random_gt_10pp": (
            g2_rew_osi["ci95_lo"] > 0.35 and g2_rew_vs_rand["ci95_lo"] > 10.0
        ),
        "H_G2c_sur_rewired_a1_matches_v1_within_6pp": g2_rew_vs_v1_abs["ci95_hi"] < 6.0,
        "H_G3a_conel_curve_overproduce_ge_2.5x_and_prune_ge_50pct": (
            m_comp["synapses_peak"] >= 2.5 * m_comp["synapses_newborn"]
            and g3_prune["ci95_lo"] >= 0.50
        ),
        "H_G3b_lottery_ticket_compiled_beats_sparse_from_scratch": (
            g3_fe_vs_sparse["ci95_hi"] < 0.0 and g3_fe_vs_peak["ci95_hi"] <= 0.0
        ),
        "H_G3c_myelination_reduces_ticks_and_boosts_tps_gt_1.25x": (
            g3_ticks_vs_peak["ci95_hi"] < 0.0 and g3_tps_ratio["ci95_lo"] > 1.25
        ),
        "H_G4a_prenatal_waves_beat_no_prenatal_fe_and_osi": (
            g4_fe_vs_none["ci95_hi"] < 0.0 and g4_osi_vs_none["ci95_lo"] > 0.0
        ),
        "H_G4b_prenatal_waves_beat_white_noise_negative_control": (
            g4_fe_vs_noise["ci95_hi"] < 0.0 and g4_osi_vs_noise["ci95_lo"] > 0.0
        ),
        "H_G4c_fetal_motor_twitches_beat_none_and_white_noise": (
            g4_sm_vs_none["ci95_hi"] < 0.0 and g4_sm_vs_noise["ci95_hi"] < 0.0
        ),
        "H_G5a_fabric_v3_genesis_beats_v2_early_ticks_late_and_comp": (
            g5f_early_diff["ci95_hi"] < 0.0
            and g5f_ticks_diff["ci95_hi"] < 0.0
            and g5f_late_diff["ci95_hi"] <= 0.0
            and g5f_comp["ci95_lo"] > 1000.0
        ),
        "H_G5b_bytes_v3_genesis_beats_v2_early_bpb_and_bigram": (
            g5b_early_vs_v2["ci95_hi"] < 0.0
            and g5b_final_vs_bg["ci95_hi"] < 0.0
            and g5b_final_vs_v2["ci95_hi"] <= 0.05
        ),
        "H_G5c_bytes_v3_sparser_than_v2_and_online_rust_shift": (
            g5b_nz_vs_v2["ci95_hi"] < 0.0
            and g5b_rust_online_vs_frozen["ci95_hi"] < 0.0
            and g5b_rust_vs_bg["ci95_hi"] < 0.0
        ),
    }
    res["verdicts"] = verdicts
    res["verdict_pass_count"] = sum(1 for v in verdicts.values() if v)
    res["verdict_total"] = len(verdicts)
    return res


def export_consolidated_csvs(raw_root: Path, out_dir: Path, phases: list[str]):
    for name in ["g1", "g2", "g3", "g4", "g5_fabric", "g5_bytes"]:
        all_rows = []
        for ph in phases:
            pdir = raw_root / ph
            if not pdir.exists():
                continue
            for r in load_phase_csv(pdir, f"{name}.csv"):
                r2 = {"phase": ph}
                r2.update(r)
                all_rows.append(r2)
        if not all_rows:
            continue
        out_p = out_dir / f"metrics_{name}.csv"
        with out_p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)


def print_phase_report(name: str, data: dict):
    print(f"\n================ {name} AURA v3 GENESIS SUMMARY ({data['verdict_pass_count']}/{data['verdict_total']} PASS) ================")
    g1m = data["g1"]["means"]["morphogen_router_v3"]
    g1r = data["g1"]["means"]["random_v2_wiring"]
    print(f"[G1] genome={g1m['genome_bytes']:.0f}B comp={g1m['compression_ratio_2048']:.1f}x | topo: morpho={g1m['topo_corr_2048']:.4f} vs rand={g1r['topo_corr_2048']:.4f} | spatial FE: morpho={g1m['spatial_late_fe']:.1f} vs rand={g1r['spatial_late_fe']:.1f}")

    g2m = data["g2"]["means"]
    print(f"[G2] 1 template ({g2m['v1_canonical_visual']['template_bytes']:.0f}B) -> V1={g2m['v1_canonical_visual']['test_acc_pct']:.2f}% A1={g2m['a1_canonical_audio']['test_acc_pct']:.2f}% Bytes={g2m['lang_canonical_bytes']['test_acc_pct']:.2f}% | Sur A1->Vis={g2m['a1_rewired_visual_sur']['test_acc_pct']:.2f}% (OSI={g2m['a1_rewired_visual_sur']['osi_score']:.3f}) vs Rand={g2m['random_unstructured_visual']['test_acc_pct']:.2f}%")

    g3m = data["g3"]["means"]
    c3 = g3m["compiled_myelinated"]
    p3 = g3m["peak_unpruned"]
    s3 = g3m["sparse_from_scratch"]
    print(f"[G3] Conel: new={c3['synapses_newborn']:.0f} -> peak={c3['synapses_peak']:.0f} -> comp={c3['synapses_final']:.0f} (prune={c3['prune_frac']*100:.1f}%) | delay={p3['mean_axon_delay']:.2f}->{c3['mean_axon_delay']:.2f} | FE: comp={c3['eval_fe']:.1f} sparse={s3['eval_fe']:.1f} peak={p3['eval_fe']:.1f} | ticks={c3['avg_ticks']:.2f} vs {p3['avg_ticks']:.2f} | tps={data['g3']['tps_ratio_vs_peak']['mean']:.2f}x")

    g4m = data["g4"]["means"]
    print(f"[G4] OSI@birth: waves={g4m['prenatal_waves_and_twitches']['osi_at_birth']:.3f} noise={g4m['white_noise_prenatal']['osi_at_birth']:.3f} none={g4m['no_prenatal']['osi_at_birth']:.3f} | early FE: {g4m['prenatal_waves_and_twitches']['early_visual_fe']:.1f} vs noise={g4m['white_noise_prenatal']['early_visual_fe']:.1f} vs none={g4m['no_prenatal']['early_visual_fe']:.1f} | SM err: {g4m['prenatal_waves_and_twitches']['early_sm_err']:.4f} vs noise={g4m['white_noise_prenatal']['early_sm_err']:.4f} vs none={g4m['no_prenatal']['early_sm_err']:.4f}")

    g5fm = data["g5_fabric"]["means"]
    print(f"[G5a] Fabric v3 vs v2: early_fe={g5fm['aura_v3_genesis_fabric']['early_fe']:.1f} vs {g5fm['aura_v2_fabric']['early_fe']:.1f} | late_fe={g5fm['aura_v3_genesis_fabric']['late_fe']:.1f} vs {g5fm['aura_v2_fabric']['late_fe']:.1f} | ticks={g5fm['aura_v3_genesis_fabric']['avg_ticks']:.2f} vs {g5fm['aura_v2_fabric']['avg_ticks']:.2f} | comp={g5fm['aura_v3_genesis_fabric']['genome_comp_2048']:.1f}x")

    g5bm = data["g5_bytes"]["means"]
    v2b = g5bm["aura_v2_baseline"]
    v3b = g5bm["aura_v3_genesis_full"]
    v3f = g5bm["aura_v3_genesis_frozen_infer"]
    bg = g5bm["bigram_byte_baseline"]
    print(f"[G5b] Bytes v3 vs v2: early_MD={v3b['early_md_bpb']:.4f} vs v2={v2b['early_md_bpb']:.4f} | final_MD={v3b['eval_md_bpb']:.4f} vs v2={v2b['eval_md_bpb']:.4f} vs bg={bg['eval_md_bpb']:.4f} | Rust={v3b['shift_rust_bpb']:.4f} vs frozen={v3f['shift_rust_bpb']:.4f} vs bg={bg['shift_rust_bpb']:.4f} | nonzero={v3b['nonzero_final_frac']*100:.1f}% vs v2={v2b['nonzero_final_frac']*100:.1f}%")

    for k, ok in data["verdicts"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", default=["DEV"])
    ap.add_argument("--out", type=Path, default=DOCS_V3_DIR / "summary.json")
    args = ap.parse_args()

    raw_root = DOCS_V3_DIR / "raw"
    summary = {}
    for ph in args.phases:
        pdir = raw_root / ph
        if not pdir.exists():
            continue
        summary[ph] = analyze_phase(pdir)
        print_phase_report(ph, summary[ph])

    export_consolidated_csvs(raw_root, DOCS_V3_DIR, args.phases)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved summary -> {args.out}")


if __name__ == "__main__":
    main()
