#!/usr/bin/env python3
"""Paired bootstrap summarizer and preregistered H_B1..H_B6 evaluator for Prueba B."""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "nervous_wave_b_v1"
RUNS = DOCS / "runs"

ARMS = [
    "nervous_coherence_b",
    "nervous_no_coherence_gate",
    "nervous_no_delay",
    "grid_local_128",
    "random_er_128",
    "flat_dense_4096",
]
TRANSFER_ARMS = [
    "nervous_transfer",
    "grid_transfer",
    "random_transfer",
    "flat_transfer",
    "ctrl_permuted_synapses",
]


def mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(n - 1, 1)
    return m, math.sqrt(var)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 12345) -> dict:
    assert len(a) == len(b) and len(a) > 0
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    m = sum(diffs) / n
    wins = sum(1 for d in diffs if d > 0.0)
    ties = sum(1 for d in diffs if d == 0.0)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    ci_lo = boots[int(0.025 * n_boot)]
    ci_hi = boots[int(0.975 * n_boot)]
    return {
        "mean_diff": m,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wins": wins,
        "ties": ties,
        "n": n,
    }


def load_rows(split: str, filename: str) -> list[dict]:
    split_dir = RUNS / split
    rows = []
    for sdir in sorted(split_dir.glob("seed_0x*")):
        p = sdir / filename
        if not p.exists():
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def collect_by_arm(rows: list[dict], key_fn, metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        out[key_fn(r)][r["seed"]] = float(r[metric])
    return out


def paired_vals(d1: dict[str, float], d2: dict[str, float]) -> tuple[list[float], list[float]]:
    seeds = sorted(set(d1.keys()) & set(d2.keys()))
    return [d1[s] for s in seeds], [d2[s] for s in seeds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "confirm"], required=True)
    args = ap.parse_args()

    b1 = load_rows(args.split, "b1_physics.csv")
    b2 = load_rows(args.split, "b2_inference.csv")
    b3 = load_rows(args.split, "b3_resilience.csv")
    b4 = load_rows(args.split, "b4_transfer.csv")

    seeds = sorted({r["seed"] for r in b1})
    n_seeds = len(seeds)
    assert n_seeds > 0, f"No runs found in {RUNS / args.split}"

    summary = {"split": args.split, "n_seeds": n_seeds, "b1": {}, "b2": {}, "b3": {}, "b4": {}, "comparisons": {}, "hypotheses": {}}

    # B1 aggregates
    for arm in ARMS:
        sub = [r for r in b1 if r["arm"] == arm]
        summary["b1"][arm] = {}
        for col in ["mean_hops", "hub_energy_t0", "hub_energy_t1", "arrival_time", "final_hub_energy", "localization_snr", "coincidence", "hub_r_coh", "hub_d_incoh"]:
            m, sd = mean_sd([float(r[col]) for r in sub])
            summary["b1"][arm][col] = {"mean": m, "sd": sd}

    # B2 aggregates
    for arm in ARMS:
        sub = [r for r in b2 if r["arm"] == arm]
        summary["b2"][arm] = {}
        for col in ["domA_acc", "domA_rmse", "domA_abduction", "domA_twohop", "domC_acc", "domC_consistency", "domA_hub_rcoh"]:
            m, sd = mean_sd([float(r[col]) for r in sub])
            summary["b2"][arm][col] = {"mean": m, "sd": sd}

    # B3 aggregates
    for arm in ARMS:
        sub = [r for r in b3 if r["arm"] == arm]
        summary["b3"][arm] = {}
        for col in ["clean_acc", "lesion15_acc", "lesion15_ret", "lesion30_acc", "lesion30_ret", "noise50_acc", "noise50_ret", "noise50_rcoh", "sens_noise50_acc"]:
            m, sd = mean_sd([float(r[col]) for r in sub])
            summary["b3"][arm][col] = {"mean": m, "sd": sd}

    # B4 aggregates
    for dom in ["domain_b_oscillators", "domain_c_graph"]:
        summary["b4"][dom] = {}
        for arm in TRANSFER_ARMS:
            sub = [r for r in b4 if r["target_domain"] == dom and r["arm"] == arm]
            frozen_ok = all(int(r["frozen_ok"]) == 1 for r in sub)
            entry = {"frozen_all_ok": frozen_ok}
            for col in ["acc", "rmse", "consistency", "hub_r_coh"]:
                m, sd = mean_sd([float(r[col]) for r in sub])
                entry[col] = {"mean": m, "sd": sd}
            summary["b4"][dom][arm] = entry

    # Key paired bootstrap comparisons
    def cmp_b(rows, metric, arm_a, arm_b, key_fn=lambda r: r["arm"], seed_off=0):
        by_arm = collect_by_arm(rows, key_fn, metric)
        va, vb = paired_vals(by_arm[arm_a], by_arm[arm_b])
        return paired_bootstrap(va, vb, seed=12345 + seed_off)

    c = summary["comparisons"]
    c["b1_coinc_nervous_vs_grid"] = cmp_b(b1, "coincidence", "nervous_coherence_b", "grid_local_128", seed_off=1)
    c["b1_rcoh_nervous_vs_no_gate"] = cmp_b(b1, "hub_r_coh", "nervous_coherence_b", "nervous_no_coherence_gate", seed_off=2)

    c["b2_domA_acc_nervous_vs_no_delay"] = cmp_b(b2, "domA_acc", "nervous_coherence_b", "nervous_no_delay", seed_off=10)
    c["b2_domA_abd_nervous_vs_no_delay"] = cmp_b(b2, "domA_abduction", "nervous_coherence_b", "nervous_no_delay", seed_off=11)
    c["b2_domA_acc_nervous_vs_flat"] = cmp_b(b2, "domA_acc", "nervous_coherence_b", "flat_dense_4096", seed_off=12)
    c["b2_domA_acc_nervous_vs_grid"] = cmp_b(b2, "domA_acc", "nervous_coherence_b", "grid_local_128", seed_off=13)
    c["b2_domA_acc_nervous_vs_random"] = cmp_b(b2, "domA_acc", "nervous_coherence_b", "random_er_128", seed_off=14)
    c["b2_domA_acc_nervous_vs_no_gate"] = cmp_b(b2, "domA_acc", "nervous_coherence_b", "nervous_no_coherence_gate", seed_off=15)
    c["b2_domC_acc_nervous_vs_flat"] = cmp_b(b2, "domC_acc", "nervous_coherence_b", "flat_dense_4096", seed_off=16)

    c["b3_noise50_nervous_vs_no_gate"] = cmp_b(b3, "noise50_acc", "nervous_coherence_b", "nervous_no_coherence_gate", seed_off=20)
    c["b3_noise50_nervous_vs_grid"] = cmp_b(b3, "noise50_acc", "nervous_coherence_b", "grid_local_128", seed_off=21)
    c["b3_noise50_nervous_vs_random"] = cmp_b(b3, "noise50_acc", "nervous_coherence_b", "random_er_128", seed_off=22)
    c["b3_noise50_nervous_vs_flat"] = cmp_b(b3, "noise50_acc", "nervous_coherence_b", "flat_dense_4096", seed_off=23)
    c["b3_sens50_nervous_vs_no_gate"] = cmp_b(b3, "sens_noise50_acc", "nervous_coherence_b", "nervous_no_coherence_gate", seed_off=24)
    c["b3_sens50_nervous_vs_grid"] = cmp_b(b3, "sens_noise50_acc", "nervous_coherence_b", "grid_local_128", seed_off=25)
    c["b3_l15_nervous_vs_grid"] = cmp_b(b3, "lesion15_acc", "nervous_coherence_b", "grid_local_128", seed_off=26)
    c["b3_l30_nervous_vs_grid"] = cmp_b(b3, "lesion30_acc", "nervous_coherence_b", "grid_local_128", seed_off=27)
    c["b3_l30_nervous_vs_random"] = cmp_b(b3, "lesion30_acc", "nervous_coherence_b", "random_er_128", seed_off=28)

    b4_c = [r for r in b4 if r["target_domain"] == "domain_c_graph"]
    b4_b = [r for r in b4 if r["target_domain"] == "domain_b_oscillators"]
    c["b4_ac_nervous_vs_perm"] = cmp_b(b4_c, "acc", "nervous_transfer", "ctrl_permuted_synapses", seed_off=30)
    c["b4_ac_nervous_vs_grid"] = cmp_b(b4_c, "acc", "nervous_transfer", "grid_transfer", seed_off=31)
    c["b4_ac_nervous_vs_random"] = cmp_b(b4_c, "acc", "nervous_transfer", "random_transfer", seed_off=32)
    c["b4_ac_nervous_vs_flat"] = cmp_b(b4_c, "acc", "nervous_transfer", "flat_transfer", seed_off=33)

    c["b4_ab_nervous_vs_perm"] = cmp_b(b4_b, "acc", "nervous_transfer", "ctrl_permuted_synapses", seed_off=40)
    c["b4_ab_nervous_vs_grid"] = cmp_b(b4_b, "acc", "nervous_transfer", "grid_transfer", seed_off=41)
    c["b4_ab_nervous_vs_random"] = cmp_b(b4_b, "acc", "nervous_transfer", "random_transfer", seed_off=42)
    c["b4_ab_nervous_vs_flat"] = cmp_b(b4_b, "acc", "nervous_transfer", "flat_transfer", seed_off=43)

    # Evaluate verbatim preregistered hypotheses H_B1..H_B6
    n_b1 = summary["b1"]["nervous_coherence_b"]
    g_b1 = summary["b1"]["grid_local_128"]
    h = summary["hypotheses"]

    h["H_B1"] = {
        "pass": bool(
            n_b1["hub_energy_t0"]["mean"] == 0.0
            and n_b1["hub_energy_t1"]["mean"] == 0.0
            and n_b1["arrival_time"]["mean"] <= 2.5
            and g_b1["arrival_time"]["mean"] >= 3.0
            and n_b1["localization_snr"]["mean"] >= 0.85
            and n_b1["hub_r_coh"]["mean"] > 0.25
            and c["b1_coinc_nervous_vs_grid"]["ci_lo"] > 0.0
        ),
        "detail": f"E0={n_b1['hub_energy_t0']['mean']:.4f}, E1={n_b1['hub_energy_t1']['mean']:.4f}, t_arr={n_b1['arrival_time']['mean']:.2f} vs grid {g_b1['arrival_time']['mean']:.2f}, locSNR={n_b1['localization_snr']['mean']*100:.1f}%, R_coh={n_b1['hub_r_coh']['mean']:.3f}, coinc_diff={c['b1_coinc_nervous_vs_grid']['mean_diff']:+.4f} [{c['b1_coinc_nervous_vs_grid']['ci_lo']:+.4f},{c['b1_coinc_nervous_vs_grid']['ci_hi']:+.4f}]",
    }

    h["H_B2"] = {
        "pass": bool(
            c["b2_domA_acc_nervous_vs_no_delay"]["mean_diff"] >= 0.05
            and c["b2_domA_acc_nervous_vs_no_delay"]["ci_lo"] > 0.0
            and c["b2_domA_abd_nervous_vs_no_delay"]["mean_diff"] >= 0.06
            and c["b2_domA_abd_nervous_vs_no_delay"]["ci_lo"] > 0.0
            and c["b2_domA_acc_nervous_vs_flat"]["mean_diff"] >= -0.02
        ),
        "detail": f"vs_no_delay acc={c['b2_domA_acc_nervous_vs_no_delay']['mean_diff']*100:+.2f}pp [{c['b2_domA_acc_nervous_vs_no_delay']['ci_lo']*100:+.2f},{c['b2_domA_acc_nervous_vs_no_delay']['ci_hi']*100:+.2f}], abd={c['b2_domA_abd_nervous_vs_no_delay']['mean_diff']*100:+.2f}pp, vs_flat={c['b2_domA_acc_nervous_vs_flat']['mean_diff']*100:+.2f}pp",
    }

    h["H_B3"] = {
        "pass": bool(
            c["b3_noise50_nervous_vs_no_gate"]["ci_lo"] > 0.0
            and c["b3_noise50_nervous_vs_grid"]["mean_diff"] >= 0.02
            and c["b3_noise50_nervous_vs_grid"]["ci_lo"] > 0.0
        ),
        "detail": f"noise50 vs_no_gate={c['b3_noise50_nervous_vs_no_gate']['mean_diff']*100:+.2f}pp [{c['b3_noise50_nervous_vs_no_gate']['ci_lo']*100:+.2f},{c['b3_noise50_nervous_vs_no_gate']['ci_hi']*100:+.2f}], vs_grid={c['b3_noise50_nervous_vs_grid']['mean_diff']*100:+.2f}pp [{c['b3_noise50_nervous_vs_grid']['ci_lo']*100:+.2f},{c['b3_noise50_nervous_vs_grid']['ci_hi']*100:+.2f}]",
    }

    h["H_B4"] = {
        "pass": bool(
            c["b3_l15_nervous_vs_grid"]["ci_lo"] > 0.0
            and c["b3_l30_nervous_vs_grid"]["ci_lo"] > 0.0
            and c["b3_l30_nervous_vs_random"]["ci_lo"] > 0.0
        ),
        "detail": f"L15 vs_grid={c['b3_l15_nervous_vs_grid']['mean_diff']*100:+.2f}pp [{c['b3_l15_nervous_vs_grid']['ci_lo']*100:+.2f},{c['b3_l15_nervous_vs_grid']['ci_hi']*100:+.2f}], L30 vs_grid={c['b3_l30_nervous_vs_grid']['mean_diff']*100:+.2f}pp, L30 vs_rand={c['b3_l30_nervous_vs_random']['mean_diff']*100:+.2f}pp [{c['b3_l30_nervous_vs_random']['ci_lo']*100:+.2f},{c['b3_l30_nervous_vs_random']['ci_hi']*100:+.2f}]",
    }

    ac_mean = summary["b4"]["domain_c_graph"]["nervous_transfer"]["acc"]["mean"]
    ac_frozen = summary["b4"]["domain_c_graph"]["nervous_transfer"]["frozen_all_ok"]
    h["H_B5"] = {
        "pass": bool(
            ac_frozen
            and c["b4_ac_nervous_vs_perm"]["ci_lo"] > 0.0
            and ac_mean >= 0.55
        ),
        "detail": f"A->C acc={ac_mean*100:.2f}% (frozen={ac_frozen}), vs_perm={c['b4_ac_nervous_vs_perm']['mean_diff']*100:+.2f}pp [{c['b4_ac_nervous_vs_perm']['ci_lo']*100:+.2f},{c['b4_ac_nervous_vs_perm']['ci_hi']*100:+.2f}]",
    }

    ab_mean = summary["b4"]["domain_b_oscillators"]["nervous_transfer"]["acc"]["mean"]
    ab_frozen = summary["b4"]["domain_b_oscillators"]["nervous_transfer"]["frozen_all_ok"]
    h["H_B6"] = {
        "pass": bool(
            ab_frozen
            and c["b4_ab_nervous_vs_perm"]["ci_lo"] > 0.0
            and c["b4_ab_nervous_vs_grid"]["ci_lo"] > 0.0
        ),
        "detail": f"A->B acc={ab_mean*100:.2f}% (frozen={ab_frozen}), vs_perm={c['b4_ab_nervous_vs_perm']['mean_diff']*100:+.2f}pp [{c['b4_ab_nervous_vs_perm']['ci_lo']*100:+.2f},{c['b4_ab_nervous_vs_perm']['ci_hi']*100:+.2f}], vs_grid={c['b4_ab_nervous_vs_grid']['mean_diff']*100:+.2f}pp [{c['b4_ab_nervous_vs_grid']['ci_lo']*100:+.2f},{c['b4_ab_nervous_vs_grid']['ci_hi']*100:+.2f}]",
    }

    n_pass = sum(1 for v in h.values() if v["pass"])
    summary["n_pass"] = n_pass
    summary["n_total"] = len(h)

    out_path = DOCS / f"summary_{args.split}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"=== Prueba B ({args.split.upper()}, N={n_seeds}) — {n_pass}/{len(h)} PASS ===")
    for k, v in h.items():
        st = "PASS" if v["pass"] else "FAIL"
        print(f"  {k}: {st} | {v['detail']}")
    print("\nB2 Inference (Domain A / Domain C):")
    for arm in ARMS:
        d = summary["b2"][arm]
        print(f"  {arm:26s} A_acc={d['domA_acc']['mean']*100:5.2f}% rmse={d['domA_rmse']['mean']:.4f} abd={d['domA_abduction']['mean']*100:5.2f}% 2hop={d['domA_twohop']['mean']*100:5.2f}% | C_acc={d['domC_acc']['mean']*100:5.2f}% cons={d['domC_consistency']['mean']*100:5.2f}%")
    print("\nB3 Resilience (Zero Retraining):")
    for arm in ARMS:
        d = summary["b3"][arm]
        print(f"  {arm:26s} clean={d['clean_acc']['mean']*100:5.2f}% L15={d['lesion15_acc']['mean']*100:5.2f}% L30={d['lesion30_acc']['mean']*100:5.2f}% phase50={d['noise50_acc']['mean']*100:5.2f}% (ret {d['noise50_ret']['mean']*100:5.1f}%) sens50={d['sens_noise50_acc']['mean']*100:5.2f}%")
    print("\nB4 Frozen-Synapse Resonant Transfer:")
    for dom in ["domain_b_oscillators", "domain_c_graph"]:
        print(f"  [{dom}]")
        for arm in TRANSFER_ARMS:
            d = summary["b4"][dom][arm]
            print(f"    {arm:24s} acc={d['acc']['mean']*100:5.2f}% rmse={d['rmse']['mean']:.4f} cons={d['consistency']['mean']*100:5.2f}% frozen={d['frozen_all_ok']}")


if __name__ == "__main__":
    main()
