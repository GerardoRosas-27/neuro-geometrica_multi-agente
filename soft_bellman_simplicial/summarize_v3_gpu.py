#!/usr/bin/env python3
"""Estadística bootstrap pareada (10,000 remuestreos, seed 12345) para Soft-Bellman Simplicial 3D v3 GPU."""
import csv
import json
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/soft_bellman_simplicial_v3_gpu"

rows = []
for path in sorted((DOC / "raw").glob("*/*/metrics.csv")):
    for r in csv.DictReader(path.open()):
        for k in ["nll", "accuracy", "success", "regret", "interior_fraction", "eval_ms"]:
            r[k] = float(r[k])
        for k in ["seed", "maps", "starts", "states", "actions"]:
            r[k] = int(r[k])
        rows.append(r)

MODELS = [
    "full",
    "diagonal",
    "edge_diagonal",
    "no_phi",
    "untrained",
    "euclidean",
    "oracle",
    "v3_full_gpu",
    "v3_continuous_gpu",
    "v3_zeroshot_gpu",
    "v3_adapted_gpu",
]

assert len(rows) == 32 * 5 * len(MODELS), f"Expected {32*5*len(MODELS)} rows, got {len(rows)}"

with (DOC / "metrics_v3_gpu.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

def quantile(xs, p):
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (i - lo) * (xs[hi] - xs[lo])

def bootstrap(d):
    rng = random.Random(12345)
    n = len(d)
    means = [sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000)]
    return {
        "difference": statistics.mean(d),
        "ci95": [quantile(means, 0.025), quantile(means, 0.975)],
        "seeds": n,
    }

summary = {
    "bootstrap": {
        "resamples": 10000,
        "seed": 12345,
        "delta": 0.05,
        "unit": "independent trained model seed; paired maps within seed",
    },
    "phases": {},
}

for phase in ["DEV", "CONFIRM"]:
    ps = {}
    rs = [r for r in rows if r["phase"] == phase]
    for split in ["train", "val", "test", "ood_geometry", "changed_law"]:
        ss = {}
        lookup = {}
        for model in MODELS:
            a = sorted([r for r in rs if r["split"] == split and r["model"] == model], key=lambda r: r["seed"])
            lookup[model] = a
            ss[model] = {
                k: statistics.mean(r[k] for r in a)
                for k in ["nll", "accuracy", "success", "regret", "interior_fraction", "eval_ms"]
            }
            ss[model]["maps_total"] = sum(r["maps"] for r in a)
            ss[model]["states_total"] = sum(r["states"] for r in a)
            ss[model]["starts_total"] = sum(r["starts"] for r in a)
        ss["paired_vs_v3_full"] = {}
        for control in ["full", "diagonal", "edge_diagonal", "no_phi", "untrained", "euclidean"]:
            ss["paired_vs_v3_full"][control] = {}
            for metric in ["regret", "nll", "success"]:
                d = [a[metric] - b[metric] for a, b in zip(lookup["v3_full_gpu"], lookup[control])]
                ss["paired_vs_v3_full"][control][metric] = bootstrap(d)
        ss["paired_vs_v3_adapted"] = {}
        for control in ["full", "diagonal", "no_phi", "euclidean", "v3_zeroshot_gpu"]:
            ss["paired_vs_v3_adapted"][control] = {}
            for metric in ["regret", "nll"]:
                d = [a[metric] - b[metric] for a, b in zip(lookup["v3_adapted_gpu"], lookup[control])]
                ss["paired_vs_v3_adapted"][control][metric] = bootstrap(d)
        ps[split] = ss
    summary["phases"][phase] = ps

# Resumen de geometría, refinamiento multinivel y Snell continuo sobre las 48 mallas diagnósticas CONFIRM
geom_rows = []
for path in sorted((DOC / "raw" / "CONFIRM").glob("*/geometry.csv")):
    for r in csv.DictReader(path.open()):
        geom_rows.append({k: float(v) for k, v in r.items()})

coarse_vs_edge = [1.0 - r["coarse_oracle_mean"] / r["edge_oracle_mean"] for r in geom_rows]
refined_vs_coarse = [1.0 - r["refined_oracle_mean"] / r["coarse_oracle_mean"] for r in geom_rows]
ultrafine_vs_coarse = [1.0 - r["ultrafine_oracle_mean"] / r["coarse_oracle_mean"] for r in geom_rows]
ultrafine_vs_refined = [1.0 - r["ultrafine_oracle_mean"] / r["refined_oracle_mean"] for r in geom_rows]
cont_vs_coarse = [1.0 - r["continuous_snell_mean"] / r["coarse_oracle_mean"] for r in geom_rows]
cont_vs_refined = [1.0 - r["continuous_snell_mean"] / r["refined_oracle_mean"] for r in geom_rows]
cont_gap_vs_ultrafine = [abs(r["continuous_snell_mean"] / r["ultrafine_oracle_mean"] - 1.0) for r in geom_rows]

summary["geometry_confirm_48"] = {
    "coarse_reduction_vs_edge_pct": statistics.mean(coarse_vs_edge) * 100.0,
    "refined_reduction_vs_coarse_pct": statistics.mean(refined_vs_coarse) * 100.0,
    "ultrafine_reduction_vs_coarse_pct": statistics.mean(ultrafine_vs_coarse) * 100.0,
    "ultrafine_reduction_vs_refined_pct": statistics.mean(ultrafine_vs_refined) * 100.0,
    "continuous_snell_reduction_vs_coarse_pct": statistics.mean(cont_vs_coarse) * 100.0,
    "continuous_snell_reduction_vs_refined_pct": statistics.mean(cont_vs_refined) * 100.0,
    "residual_gap_continuous_vs_ultrafine_pct": statistics.mean(cont_gap_vs_ultrafine) * 100.0,
    "v3_mean_intrinsic_regge_action": statistics.mean(r["v3_intrinsic_regge"] for r in geom_rows),
    "v3_mean_cm_log_barrier": statistics.mean(r["v3_cm_barrier"] for r in geom_rows),
    "v3_mean_hodge_dirichlet_energy": statistics.mean(r["v3_hodge_energy"] for r in geom_rows),
    "min_cm_volume2": min(r["min_cm_volume2"] for r in geom_rows),
}

(DOC / "summary_v3_gpu.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({
    "CONFIRM_test_v2_full": summary["phases"]["CONFIRM"]["test"]["full"],
    "CONFIRM_test_v3_full_gpu": summary["phases"]["CONFIRM"]["test"]["v3_full_gpu"],
    "CONFIRM_test_v3_continuous_gpu": summary["phases"]["CONFIRM"]["test"]["v3_continuous_gpu"],
    "CONFIRM_changed_law_v2_full": summary["phases"]["CONFIRM"]["changed_law"]["full"],
    "CONFIRM_changed_law_euclidean": summary["phases"]["CONFIRM"]["changed_law"]["euclidean"],
    "CONFIRM_changed_law_v3_zeroshot_gpu": summary["phases"]["CONFIRM"]["changed_law"]["v3_zeroshot_gpu"],
    "CONFIRM_changed_law_v3_adapted_gpu": summary["phases"]["CONFIRM"]["changed_law"]["v3_adapted_gpu"],
    "geometry_confirm_48": summary["geometry_confirm_48"],
}, indent=2))
