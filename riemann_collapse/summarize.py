#!/usr/bin/env python3
"""Paired 10,000-resample seed-bootstrap analysis and preregistered H_C1a..H_C4b evaluation."""
import argparse
import csv
import json
import pathlib
import random
import statistics

CRATE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CRATE_DIR.parent
DOCS_DIR = REPO_ROOT / "docs" / "riemann_collapse_v1"

STR_COLS = {"seed", "phase", "operator", "arm", "probe"}


def quantile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    i = (len(s) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def paired_bootstrap(vals: list[float], resamples: int = 10000, seed: int = 12345) -> dict:
    rng = random.Random(seed)
    n = len(vals)
    means = [sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "ci95": [quantile(means, 0.025), quantile(means, 0.975)],
        "seeds": n,
    }


def load_suite(num: int, phases: list[str]) -> list[dict]:
    rows = []
    for phase in phases:
        for p in sorted((DOCS_DIR / "raw" / phase).glob(f"0x*/metrics_c{num}.csv")):
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


def group_by(rows: list[dict], key_col: str) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}
    for r in rows:
        g.setdefault(r[key_col], []).append(r)
    for k in g:
        g[k].sort(key=lambda x: x["seed"])
    return g


def paired_diff(groups: dict[str, list[dict]], a_key: str, b_key: str, metric: str) -> dict:
    ga, gb = groups[a_key], groups[b_key]
    assert [r["seed"] for r in ga] == [r["seed"] for r in gb]
    return paired_bootstrap([x[metric] - y[metric] for x, y in zip(ga, gb)])


def single_boot(groups: dict[str, list[dict]], key: str, metric: str) -> dict:
    return paired_bootstrap([x[metric] for x in groups[key]])


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


def analyze_phase(phase: str, s1, s2, s3, s4) -> dict:
    c1 = [r for r in s1 if r["phase"] == phase]
    c2 = [r for r in s2 if r["phase"] == phase]
    c3 = [r for r in s3 if r["phase"] == phase]
    c4 = [r for r in s4 if r["phase"] == phase]

    g1 = group_by(c1, "operator")
    g2 = group_by(c2, "arm")
    g3 = group_by(c3, "probe")
    g4 = group_by(c4, "arm")

    desc = {
        "c1": {k: mean_std(v, ["prime_only_auc", "prime_power_auc", "prime_snr", "past_alone_prime_auc", "future_alone_prime_auc", "off_time_auc", "emergence_gain"]) for k, v in g1.items()},
        "c2": {k: mean_std(v, ["tau", "k_sum", "delta_phi", "sigma_tot", "beta_disp", "p_win", "v_asym", "err_tau", "err_k_sum", "err_phi", "err_beta", "train_obj", "train_auc", "heldout_pp_auc", "heldout_p_auc", "heldout_snr"]) for k, v in g2.items()},
        "c3": {k: mean_std(v, ["param_val", "auc_2_100", "auc_61_150", "snr_2_100"]) for k, v in g3.items()},
        "c4": {k: mean_std(v, ["distinct_zeros_found", "mean_zero_err", "auc_2_45", "snr_2_45", "extrap_psi_corr"]) for k, v in g4.items()},
    }

    paired = {
        # C1
        "c1_snr_trans_vs_poisson": paired_diff(g1, "transactional_half_freq", "poisson_freq_ctrl", "prime_snr"),
        "c1_auc_naive_vs_trans": paired_diff(g1, "naive_bilinear_ctrl", "transactional_half_freq", "prime_power_auc"),
        # C2
        "c2_heldout_auc_opt": single_boot(g2, "collapse_optimizer", "heldout_pp_auc"),
        "c2_heldout_auc_opt_vs_rand": paired_diff(g2, "collapse_optimizer", "random_search_ctrl", "heldout_pp_auc"),
        "c2_heldout_auc_opt_vs_init": paired_diff(g2, "collapse_optimizer", "init_random", "heldout_pp_auc"),
        "c2_heldout_snr_opt_vs_neghalf": paired_diff(g2, "collapse_optimizer", "fixed_sigma_neg_half", "heldout_snr"),
        "c2_err_tau_opt_vs_rand": paired_diff(g2, "collapse_optimizer", "random_search_ctrl", "err_tau"),
        "c2_err_ksum_opt_vs_rand": paired_diff(g2, "collapse_optimizer", "random_search_ctrl", "err_k_sum"),
        # C4
        "c4_found_dyson": single_boot(g4, "blind_collapse_dyson", "distinct_zeros_found"),
        "c4_found_dyson_vs_norep": paired_diff(g4, "blind_collapse_dyson", "blind_collapse_no_repulsion", "distinct_zeros_found"),
        "c4_err_dyson_vs_norep": paired_diff(g4, "blind_collapse_dyson", "blind_collapse_no_repulsion", "mean_zero_err"),
        "c4_auc45_dyson_vs_norep": paired_diff(g4, "blind_collapse_dyson", "blind_collapse_no_repulsion", "auc_2_45"),
        "c4_auc45_dyson_vs_rand": paired_diff(g4, "blind_collapse_dyson", "random_freq_ctrl", "auc_2_45"),
        "c4_psicorr_dyson": single_boot(g4, "blind_collapse_dyson", "extrap_psi_corr"),
        "c4_psicorr_dyson_vs_rand": paired_diff(g4, "blind_collapse_dyson", "random_freq_ctrl", "extrap_psi_corr"),
        "c4_psicorr_dyson_vs_norep": paired_diff(g4, "blind_collapse_dyson", "blind_collapse_no_repulsion", "extrap_psi_corr"),
    }

    t1 = desc["c1"]["transactional_half_freq"]
    l1 = desc["c1"]["linear_counterprop"]
    n1 = desc["c1"]["naive_bilinear_ctrl"]
    o2 = desc["c2"]["collapse_optimizer"]

    criteria = {
        "H_C1a": (
            t1["prime_only_auc"]["mean"] > 0.98
            and t1["past_alone_prime_auc"]["mean"] < 0.60
            and t1["future_alone_prime_auc"]["mean"] < 0.60
            and t1["emergence_gain"]["mean"] > 0.40
            and paired["c1_snr_trans_vs_poisson"]["ci95"][0] > 10.0
        ),
        "H_C1b": (
            n1["prime_power_auc"]["mean"] < 0.65
            and paired["c1_auc_naive_vs_trans"]["ci95"][1] < -0.30
        ),
        "H_C1c": (
            t1["off_time_auc"]["mean"] < 0.65
            and l1["off_time_auc"]["mean"] < 0.65
        ),
        "H_C2a": (
            o2["err_tau"]["mean"] < 0.005
            and o2["err_k_sum"]["mean"] < 0.002
            and o2["err_phi"]["mean"] < 0.15
            and o2["err_beta"]["mean"] < 0.002
        ),
        "H_C2b": (
            paired["c2_heldout_auc_opt"]["ci95"][0] > 0.95
            and paired["c2_heldout_auc_opt_vs_rand"]["ci95"][0] > 0.25
        ),
        "H_C2c": (
            paired["c2_heldout_snr_opt_vs_neghalf"]["ci95"][0] > 0.0
        ),
        "H_C3a": (
            desc["c3"]["phase_pi"]["auc_2_100"]["mean"] > 0.98
            and desc["c3"]["phase_zero"]["auc_2_100"]["mean"] < 0.02
            and desc["c3"]["detune_k_0.020"]["auc_2_100"]["mean"] < 0.65
        ),
        "H_C3b": (
            desc["c3"]["modes_M_256"]["auc_61_150"]["mean"] > 0.99
            and desc["c3"]["modes_M_16"]["auc_61_150"]["mean"] < 0.70
        ),
        "H_C4a": (
            paired["c4_found_dyson"]["ci95"][0] >= 15.0
            and paired["c4_found_dyson_vs_norep"]["ci95"][0] > 5.0
            and paired["c4_err_dyson_vs_norep"]["ci95"][1] < -0.5
        ),
        "H_C4b": (
            paired["c4_auc45_dyson_vs_rand"]["ci95"][0] > 0.25
            and paired["c4_psicorr_dyson"]["ci95"][0] > 0.50
            and paired["c4_psicorr_dyson_vs_rand"]["ci95"][0] > 0.45
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

    if not a.dev_only:
        write_agg_csv(DOCS_DIR / "metrics_c1.csv", s1)
        write_agg_csv(DOCS_DIR / "metrics_c2.csv", s2)
        write_agg_csv(DOCS_DIR / "metrics_c3.csv", s3)
        write_agg_csv(DOCS_DIR / "metrics_c4.csv", s4)

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
        summary["phases"][ph] = analyze_phase(ph, s1, s2, s3, s4)

    out_name = "summary_dev.json" if a.dev_only else "summary.json"
    (DOCS_DIR / out_name).write_text(json.dumps(summary, indent=2) + "\n")

    for ph in phases:
        res = summary["phases"][ph]
        print(f"\n=== {ph} ({res['passed_count']}/{res['total_criteria']} criteria PASS) ===")
        for cid, ok in res["criteria"].items():
            print(f"  {cid}: {'PASS' if ok else 'FAIL'}")
        o2 = res["descriptive"]["c2"]["collapse_optimizer"]
        print(f"  C2 discovered exact config: tau={o2['tau']['mean']:+.6f} (|err|={o2['err_tau']['mean']:.6f}), k_sum={o2['k_sum']['mean']:.6f} (|err|={o2['err_k_sum']['mean']:.6f}), dphi={o2['delta_phi']['mean']:+.5f} (|err|={o2['err_phi']['mean']:.5f}), sigma={o2['sigma_tot']['mean']:+.4f}, beta={o2['beta_disp']['mean']:+.6f}, p_win={o2['p_win']['mean']:.3f}")
        print(f"  C2 heldout [61..150] AUC: opt={o2['heldout_pp_auc']['mean']:.5f} vs rand={res['descriptive']['c2']['random_search_ctrl']['heldout_pp_auc']['mean']:.5f} | SNR opt={o2['heldout_snr']['mean']:.3f} vs sigma=-0.5: {res['descriptive']['c2']['fixed_sigma_neg_half']['heldout_snr']['mean']:.3f}")
        d4 = res["descriptive"]["c4"]
        print(f"  C4 blind zeros found (/20): dyson={d4['blind_collapse_dyson']['distinct_zeros_found']['mean']:.2f} (err={d4['blind_collapse_dyson']['mean_zero_err']['mean']:.4f}) vs no_rep={d4['blind_collapse_no_repulsion']['distinct_zeros_found']['mean']:.2f} (err={d4['blind_collapse_no_repulsion']['mean_zero_err']['mean']:.4f})")
        print(f"  C4 blind extrap psi(x)-x corr [601..1200]: dyson={d4['blind_collapse_dyson']['extrap_psi_corr']['mean']:.4f} vs no_rep={d4['blind_collapse_no_repulsion']['extrap_psi_corr']['mean']:.4f} vs rand={d4['random_freq_ctrl']['extrap_psi_corr']['mean']:.4f} vs exact20={d4['exact_20_zeros']['extrap_psi_corr']['mean']:.4f}")


if __name__ == "__main__":
    main()
