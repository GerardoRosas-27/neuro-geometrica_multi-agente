#!/usr/bin/env python3
"""Paired 10,000-resample seed-bootstrap analysis and preregistered H_R1a..H_R5c evaluation."""
import argparse
import csv
import json
import pathlib
import random
import statistics

CRATE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CRATE_DIR.parent
DOCS_DIR = REPO_ROOT / "docs" / "riemann_wave_v1"

STR_COLS = {"seed", "phase", "arm", "family", "task"}


def quantile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    i = (len(s) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def paired_bootstrap(diffs: list[float], resamples: int = 10000, seed: int = 12345) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    means = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    return {
        "mean_diff": statistics.mean(diffs),
        "median_diff": statistics.median(diffs),
        "ci95": [quantile(means, 0.025), quantile(means, 0.975)],
        "seeds": n,
    }


def load_suite(suite_num: int, phases: list[str]) -> list[dict]:
    rows = []
    for phase in phases:
        for p in sorted((DOCS_DIR / "raw" / phase).glob(f"0x*/metrics_r{suite_num}.csv")):
            for r in csv.DictReader(p.open()):
                out = {}
                for k, v in r.items():
                    if k == "seed":
                        out[k] = int(v)
                    elif k in STR_COLS:
                        out[k] = v
                    else:
                        out[k] = float(v)
                rows.append(out)
    return rows


def write_agg_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def group_by_key(rows: list[dict], key_col: str, extra_filter=None) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if extra_filter and not extra_filter(r):
            continue
        groups.setdefault(r[key_col], []).append(r)
    for k in groups:
        groups[k].sort(key=lambda x: x["seed"])
    return groups


def paired_diff(groups: dict[str, list[dict]], a_key: str, b_key: str, metric: str) -> dict:
    ga, gb = groups[a_key], groups[b_key]
    assert [r["seed"] for r in ga] == [r["seed"] for r in gb], f"Seed mismatch {a_key} vs {b_key}"
    return paired_bootstrap([x[metric] - y[metric] for x, y in zip(ga, gb)])


def mean_std(rows: list[dict], cols: list[str]) -> dict:
    out = {}
    for c in cols:
        vals = [r[c] for r in rows]
        out[c] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return out


def analyze_phase(phase: str, s1, s2, s3, s4, s5) -> dict:
    r1 = [r for r in s1 if r["phase"] == phase]
    r2 = [r for r in s2 if r["phase"] == phase]
    r3 = [r for r in s3 if r["phase"] == phase]
    r4 = [r for r in s4 if r["phase"] == phase]
    r5 = [r for r in s5 if r["phase"] == phase]

    g1 = group_by_key(r1, "arm")
    g2 = group_by_key(r2, "family")
    g3 = group_by_key(r3, "family")
    g4 = group_by_key(r4, "arm")
    g5_1d = group_by_key(r5, "arm", lambda r: r["task"] == "r5_1d")
    g5_2d = group_by_key(r5, "arm", lambda r: r["task"] == "r5_2d")

    # Descriptive means per suite/arm
    desc = {
        "r1": {k: mean_std(v, ["prime_auc", "prime_snr", "ks_vs_gue", "ks_vs_poisson", "p_small_02"]) for k, v in g1.items()},
        "r2": {k: mean_std(v, ["log10_cond", "df_ridge", "norm_ols", "norm_ridge", "rmse_ols_015", "rmse_ols_050", "rmse_ridge_015", "rmse_ridge_050", "rmse_ridge_000", "min_gap"]) for k, v in g2.items()},
        "r3": {k: mean_std(v, ["in_domain_rmse", "ghost_peak", "ghost_rms", "periodic_echo_corr", "joint_score", "irreg_log10_cond", "irreg_rmse_ols", "irreg_rmse_ridge"]) for k, v in g3.items()},
        "r4": {k: mean_std(v, ["stream_rmse", "first_concept_rmse", "log10_cond", "weight_norm", "min_gap"]) for k, v in g4.items()},
        "r5_1d": {k: mean_std(v, ["kernel_rel_frob", "rmse_1step", "rmse_rollout8"]) for k, v in g5_1d.items()},
        "r5_2d": {k: mean_std(v, ["kernel_rel_frob", "rmse_1step", "rmse_rollout8"]) for k, v in g5_2d.items()},
    }

    # Paired bootstrap comparisons
    paired = {
        # R1
        "r1_snr_riemann_vs_poisson": paired_diff(g1, "riemann_zeros", "poisson_prime_ctrl", "prime_snr"),
        "r1_ks_dyson_gue_minus_poisson": paired_bootstrap([x["ks_vs_gue"] - x["ks_vs_poisson"] for x in g1["dyson_gue"]]),
        "r1_psmall_dyson_vs_poisson": paired_diff(g1, "dyson_gue", "poisson_prime_ctrl", "p_small_02"),
        "r1_ks_guemat_gue_minus_poisson": paired_bootstrap([x["ks_vs_gue"] - x["ks_vs_poisson"] for x in g1["gue_matrix"]]),
        # R2
        "r2_cond_dyson_vs_poisson": paired_diff(g2, "dyson_gue", "poisson", "log10_cond"),
        "r2_cond_riemann_vs_poisson": paired_diff(g2, "riemann_zeros", "poisson", "log10_cond"),
        "r2_rmse_ols015_dyson_vs_poisson": paired_diff(g2, "dyson_gue", "poisson", "rmse_ols_015"),
        "r2_rmse_ols015_riemann_vs_poisson": paired_diff(g2, "riemann_zeros", "poisson", "rmse_ols_015"),
        "r2_rmse_ols050_dyson_vs_poisson": paired_diff(g2, "dyson_gue", "poisson", "rmse_ols_050"),
        "r2_rmse_ridge015_dyson_vs_poisson": paired_diff(g2, "dyson_gue", "poisson", "rmse_ridge_015"),
        "r2_rmse_ridge015_riemann_vs_poisson": paired_diff(g2, "riemann_zeros", "poisson", "rmse_ridge_015"),
        "r2_rmse_ridge015_dyson_vs_fourier": paired_diff(g2, "dyson_gue", "fourier_uniform", "rmse_ridge_015"),
        "r2_cond_dyson_vs_fourier": paired_diff(g2, "dyson_gue", "fourier_uniform", "log10_cond"),
        "r2_rmse_ridge015_dyson_vs_golden": paired_diff(g2, "dyson_gue", "golden_jitter", "rmse_ridge_015"),
        # R3
        "r3_ghost_peak_dyson_vs_fourier": paired_diff(g3, "dyson_gue", "fourier_uniform", "ghost_peak"),
        "r3_ghost_peak_riemann_vs_fourier": paired_diff(g3, "riemann_zeros", "fourier_uniform", "ghost_peak"),
        "r3_ghost_peak_dyson_vs_golden": paired_diff(g3, "dyson_gue", "golden_jitter", "ghost_peak"),
        "r3_echo_corr_dyson_vs_fourier": paired_diff(g3, "dyson_gue", "fourier_uniform", "periodic_echo_corr"),
        "r3_echo_corr_riemann_vs_fourier": paired_diff(g3, "riemann_zeros", "fourier_uniform", "periodic_echo_corr"),
        "r3_joint_dyson_vs_fourier": paired_diff(g3, "dyson_gue", "fourier_uniform", "joint_score"),
        "r3_joint_dyson_vs_poisson": paired_diff(g3, "dyson_gue", "poisson", "joint_score"),
        "r3_irreg_cond_dyson_vs_poisson": paired_diff(g3, "dyson_gue", "poisson", "irreg_log10_cond"),
        "r3_irreg_rmse_ridge_dyson_vs_poisson": paired_diff(g3, "dyson_gue", "poisson", "irreg_rmse_ridge"),
        "r3_irreg_rmse_ridge_riemann_vs_poisson": paired_diff(g3, "riemann_zeros", "poisson", "irreg_rmse_ridge"),
        "r3_irreg_rmse_ridge_dyson_vs_fourier": paired_diff(g3, "dyson_gue", "fourier_uniform", "irreg_rmse_ridge"),
        # R4
        "r4_stream_rmse_dyson_vs_poisson": paired_diff(g4, "dyson_gue", "poisson", "stream_rmse"),
        "r4_first_rmse_dyson_vs_poisson": paired_diff(g4, "dyson_gue", "poisson", "first_concept_rmse"),
        "r4_stream_rmse_riemann_vs_poisson": paired_diff(g4, "riemann_zeros", "poisson", "stream_rmse"),
        "r4_stream_rmse_adapt_dyson_vs_norep": paired_diff(g4, "adaptive_dyson", "adaptive_no_repulsion", "stream_rmse"),
        "r4_first_rmse_adapt_dyson_vs_norep": paired_diff(g4, "adaptive_dyson", "adaptive_no_repulsion", "first_concept_rmse"),
        "r4_mingap_adapt_dyson_vs_norep": paired_diff(g4, "adaptive_dyson", "adaptive_no_repulsion", "min_gap"),
        "r4_cond_adapt_dyson_vs_norep": paired_diff(g4, "adaptive_dyson", "adaptive_no_repulsion", "log10_cond"),
        # R5 1D
        "r5_1d_kerr_dyson_vs_poisson": paired_diff(g5_1d, "dyson_rff", "poisson_rff", "kernel_rel_frob"),
        "r5_1d_kerr_riemann_vs_poisson": paired_diff(g5_1d, "riemann_rff", "poisson_rff", "kernel_rel_frob"),
        "r5_1d_kerr_dyson_vs_qmc": paired_diff(g5_1d, "dyson_rff", "qmc_rff", "kernel_rel_frob"),
        "r5_1d_1step_dyson_vs_poisson": paired_diff(g5_1d, "dyson_rff", "poisson_rff", "rmse_1step"),
        "r5_1d_roll8_dyson_vs_poisson": paired_diff(g5_1d, "dyson_rff", "poisson_rff", "rmse_rollout8"),
        "r5_1d_roll8_riemann_vs_poisson": paired_diff(g5_1d, "riemann_rff", "poisson_rff", "rmse_rollout8"),
        "r5_1d_roll8_dyson_vs_qmc": paired_diff(g5_1d, "dyson_rff", "qmc_rff", "rmse_rollout8"),
        # R5 2D
        "r5_2d_kerr_ginibre_vs_poisson": paired_diff(g5_2d, "ginibre_2d_rff", "poisson_2d_rff", "kernel_rel_frob"),
        "r5_2d_kerr_ginibre_vs_qmc": paired_diff(g5_2d, "ginibre_2d_rff", "qmc_2d_rff", "kernel_rel_frob"),
        "r5_2d_1step_ginibre_vs_poisson": paired_diff(g5_2d, "ginibre_2d_rff", "poisson_2d_rff", "rmse_1step"),
        "r5_2d_roll8_ginibre_vs_poisson": paired_diff(g5_2d, "ginibre_2d_rff", "poisson_2d_rff", "rmse_rollout8"),
        "r5_2d_roll8_ginibre_vs_qmc": paired_diff(g5_2d, "ginibre_2d_rff", "qmc_2d_rff", "rmse_rollout8"),
    }

    # Evaluate the 12 preregistered criteria H_R1a..H_R5c
    criteria = {
        "H_R1a": (
            desc["r1"]["riemann_zeros"]["prime_auc"]["mean"] > 0.95
            and paired["r1_snr_riemann_vs_poisson"]["ci95"][0] > 2.0
        ),
        "H_R1b": (
            paired["r1_ks_dyson_gue_minus_poisson"]["ci95"][1] < -0.15
            and paired["r1_psmall_dyson_vs_poisson"]["ci95"][1] < -0.10
        ),
        "H_R2a": (
            paired["r2_cond_dyson_vs_poisson"]["ci95"][1] < -1.0
            and paired["r2_rmse_ols015_dyson_vs_poisson"]["ci95"][1] < 0.0
            and paired["r2_rmse_ols015_riemann_vs_poisson"]["ci95"][1] < 0.0
        ),
        "H_R2b": (
            paired["r2_rmse_ridge015_dyson_vs_poisson"]["ci95"][1] < 0.0
            and paired["r2_rmse_ridge015_riemann_vs_poisson"]["ci95"][1] < 0.0
        ),
        "H_R2c": (
            # Honesty verification: check if fourier_uniform indeed achieves strictly better conditioning in periodic [0, 2pi)
            paired["r2_cond_dyson_vs_fourier"]["ci95"][0] > 0.0
        ),
        "H_R3a": (
            paired["r3_ghost_peak_dyson_vs_fourier"]["ci95"][1] < -0.30
            and paired["r3_ghost_peak_riemann_vs_fourier"]["ci95"][1] < -0.30
        ),
        "H_R3b": (
            paired["r3_irreg_cond_dyson_vs_poisson"]["ci95"][1] < -0.5
            and paired["r3_irreg_rmse_ridge_dyson_vs_poisson"]["ci95"][1] < 0.0
        ),
        "H_R4a": (
            paired["r4_stream_rmse_dyson_vs_poisson"]["ci95"][1] < 0.0
        ),
        "H_R4b": (
            paired["r4_stream_rmse_adapt_dyson_vs_norep"]["ci95"][1] < 0.0
            and paired["r4_mingap_adapt_dyson_vs_norep"]["ci95"][0] > 0.0
        ),
        "H_R5a": (
            paired["r5_1d_kerr_dyson_vs_poisson"]["ci95"][1] < 0.0
            and paired["r5_1d_kerr_riemann_vs_poisson"]["ci95"][1] < 0.0
        ),
        "H_R5b": (
            paired["r5_1d_roll8_dyson_vs_poisson"]["ci95"][1] < 0.0
        ),
        "H_R5c": (
            paired["r5_2d_roll8_ginibre_vs_poisson"]["ci95"][1] < 0.0
        ),
    }

    run_meta = json.loads((DOCS_DIR / "raw" / phase / "run.json").read_text())
    return {
        "runtime": run_meta,
        "descriptive": desc,
        "paired": paired,
        "criteria": criteria,
        "passed_count": sum(1 for v in criteria.values() if v),
        "total_criteria": len(criteria),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dev-only", action="store_true")
    a = p.parse_args()
    phases = ["DEV"] if a.dev_only else ["DEV", "CONFIRM"]

    s1 = load_suite(1, phases)
    s2 = load_suite(2, phases)
    s3 = load_suite(3, phases)
    s4 = load_suite(4, phases)
    s5 = load_suite(5, phases)

    if not a.dev_only:
        write_agg_csv(DOCS_DIR / "metrics_r1.csv", s1)
        write_agg_csv(DOCS_DIR / "metrics_r2.csv", s2)
        write_agg_csv(DOCS_DIR / "metrics_r3.csv", s3)
        write_agg_csv(DOCS_DIR / "metrics_r4.csv", s4)
        write_agg_csv(DOCS_DIR / "metrics_r5.csv", s5)

    summary = {
        "bootstrap": {
            "resamples": 10000,
            "seed": 12345,
            "delta": 0.05,
            "method": "paired seed-level percentile bootstrap with linear interpolation",
        },
        "phases": {},
    }
    for ph in phases:
        summary["phases"][ph] = analyze_phase(ph, s1, s2, s3, s4, s5)

    out_name = "summary_dev.json" if a.dev_only else "summary.json"
    (DOCS_DIR / out_name).write_text(json.dumps(summary, indent=2) + "\n")

    for ph in phases:
        res = summary["phases"][ph]
        print(f"\n=== {ph} ({res['passed_count']}/{res['total_criteria']} criteria PASS) ===")
        for cid, ok in res["criteria"].items():
            print(f"  {cid}: {'PASS' if ok else 'FAIL'}")
        print("Key paired deltas:")
        for k in [
            "r1_snr_riemann_vs_poisson",
            "r1_ks_dyson_gue_minus_poisson",
            "r2_cond_dyson_vs_poisson",
            "r2_rmse_ols015_dyson_vs_poisson",
            "r2_rmse_ridge015_dyson_vs_poisson",
            "r2_rmse_ridge015_dyson_vs_fourier",
            "r3_ghost_peak_dyson_vs_fourier",
            "r3_ghost_peak_riemann_vs_fourier",
            "r3_irreg_rmse_ridge_dyson_vs_poisson",
            "r4_stream_rmse_dyson_vs_poisson",
            "r4_stream_rmse_adapt_dyson_vs_norep",
            "r5_1d_kerr_dyson_vs_poisson",
            "r5_1d_roll8_dyson_vs_poisson",
            "r5_1d_roll8_dyson_vs_qmc",
            "r5_2d_kerr_ginibre_vs_poisson",
            "r5_2d_roll8_ginibre_vs_poisson",
            "r5_2d_roll8_ginibre_vs_qmc",
        ]:
            d = res["paired"][k]
            print(f"  {k:40s} = {d['mean_diff']:+.5f} [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}]")


if __name__ == "__main__":
    main()
