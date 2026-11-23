#!/usr/bin/env python3
"""Run and analyze the Head-to-Head Tournament of the 5 Best Architectures
under Strictly Equal Training Conditions across 16 CONFIRM seeds (0xB600-0xB60F)."""
import concurrent.futures
import csv
import json
import random
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = Path(__file__).resolve().parent
DOCS_V3_DIR = REPO_ROOT / "docs" / "aura_v3"
RAW_TOP5_DIR = DOCS_V3_DIR / "raw" / "TOP5_CONFIRM"

SEEDS = [0xB600 + i for i in range(16)]

ARCH_ORDER_BYTES = [
    "aura_v3_genesis",
    "aura_v2",
    "metafe_3s",
    "metafe_v2",
    "bq10",
    "constelacion_v4",
    "atlas_fw",
    "orbita_v1",
    "aura_v1",
]

ARCH_ORDER_DYN = [
    "aura_v3_genesis",
    "bq10",
    "aura_v2",
    "constelacion_v4",
    "atlas_fw",
    "orbita_v1",
]


def bootstrap_stat(vals: list[float], n_boot: int = 4000, seed: int = 20261122) -> dict:
    n = len(vals)
    mean = sum(vals) / max(n, 1)
    var = sum((x - mean) ** 2 for x in vals) / max(n - 1, 1)
    std = var ** 0.5
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return {"n": n, "mean": mean, "std": std, "ci95_lo": lo, "ci95_hi": hi}


def paired_diff(map_a: dict[int, float], map_b: dict[int, float]) -> dict:
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    diffs = [map_a[k] - map_b[k] for k in keys]
    wins_a_lower = sum(1 for d in diffs if d < 0)
    wins_a_higher = sum(1 for d in diffs if d > 0)
    st = bootstrap_stat(diffs)
    st["wins_a_lower"] = wins_a_lower
    st["wins_a_higher"] = wins_a_higher
    return st


def run_one_seed(bin_path: Path, seed: int) -> tuple[str, float]:
    s_hex = f"0x{seed:04X}"
    out_dir = RAW_TOP5_DIR / s_hex
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(
        ["nice", "-n", "10", str(bin_path), s_hex, str(out_dir), "--repo", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        check=True,
    )
    return s_hex, time.time() - t0


def main():
    bin_path = ENGINE_DIR / "target" / "release" / "top5-arena"
    subprocess.run(
        [
            "nice",
            "-n",
            "10",
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(ENGINE_DIR / "top5-arena" / "Cargo.toml"),
            "--target-dir",
            str(ENGINE_DIR / "target"),
            "-j",
            "2",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    RAW_TOP5_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[TOP5-ARENA] Running 16 CONFIRM seeds (0xB600..0xB60F) under equal training conditions...")
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_one_seed, bin_path, s) for s in SEEDS]
        for fut in concurrent.futures.as_completed(futs):
            sh, dt = fut.result()
            print(f"  done {sh} ({dt:.2f}s)")
    total_sec = time.time() - t_start
    print(f"[TOP5-ARENA] Completed 16 seeds in {total_sec:.2f}s")

    # Consolidate CSVs
    bytes_rows = []
    dyn_rows = []
    for s in SEEDS:
        s_hex = f"0x{s:04X}"
        with (RAW_TOP5_DIR / s_hex / "arena1_bytes.csv").open() as f:
            for r in csv.DictReader(f):
                bytes_rows.append(r)
        with (RAW_TOP5_DIR / s_hex / "arena2_dynamics.csv").open() as f:
            for r in csv.DictReader(f):
                dyn_rows.append(r)

    out_b_csv = DOCS_V3_DIR / "top5_equal_arena1_bytes.csv"
    with out_b_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bytes_rows[0].keys()))
        w.writeheader()
        w.writerows(bytes_rows)

    out_d_csv = DOCS_V3_DIR / "top5_equal_arena2_dynamics.csv"
    with out_d_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dyn_rows[0].keys()))
        w.writeheader()
        w.writerows(dyn_rows)

    # Group by architecture
    b_by_arch: dict[str, dict[str, dict[int, float]]] = {}
    for r in bytes_rows:
        a = r["arch"]
        sd = int(r["seed"])
        b_by_arch.setdefault(a, {})
        for col in [
            "early_md_bpb",
            "eval_md_bpb",
            "shift_rust_bpb",
            "active_weight_bytes",
            "nonzero_frac",
            "effective_rank",
            "train_bytes_per_sec",
        ]:
            b_by_arch[a].setdefault(col, {})[sd] = float(r[col])

    d_by_arch: dict[str, dict[str, dict[int, float]]] = {}
    for r in dyn_rows:
        a = r["arch"]
        sd = int(r["seed"])
        d_by_arch.setdefault(a, {})
        for col in [
            "new_med_log",
            "known_med_log",
            "known_acc16",
            "new_acc16",
            "active_synapses",
            "wall_ms",
        ]:
            d_by_arch[a].setdefault(col, {})[sd] = float(r[col])

    arena1_stats = {}
    for a in ARCH_ORDER_BYTES:
        arena1_stats[a] = {
            col: bootstrap_stat(list(b_by_arch[a][col].values()))
            for col in b_by_arch[a]
        }

    arena2_stats = {}
    for a in ARCH_ORDER_DYN:
        arena2_stats[a] = {
            col: bootstrap_stat(list(d_by_arch[a][col].values()))
            for col in d_by_arch[a]
        }

    # Head-to-head (`vs`) contrasts of AURA v3 Genesis against every architecture
    arena1_vs = {}
    for other in ["aura_v2", "metafe_3s", "metafe_v2", "bq10", "atlas_fw", "constelacion_v4", "orbita_v1", "aura_v1"]:
        arena1_vs[f"aura_v3_genesis_vs_{other}"] = {
            "eval_md_bpb": paired_diff(
                b_by_arch["aura_v3_genesis"]["eval_md_bpb"],
                b_by_arch[other]["eval_md_bpb"],
            ),
            "shift_rust_bpb": paired_diff(
                b_by_arch["aura_v3_genesis"]["shift_rust_bpb"],
                b_by_arch[other]["shift_rust_bpb"],
            ),
            "early_md_bpb": paired_diff(
                b_by_arch["aura_v3_genesis"]["early_md_bpb"],
                b_by_arch[other]["early_md_bpb"],
            ),
        }

    arena2_vs = {}
    for other in ["bq10", "aura_v2", "atlas_fw", "constelacion_v4", "orbita_v1"]:
        arena2_vs[f"aura_v3_genesis_vs_{other}"] = {
            "new_med_log": paired_diff(
                d_by_arch["aura_v3_genesis"]["new_med_log"],
                d_by_arch[other]["new_med_log"],
            ),
            "known_med_log": paired_diff(
                d_by_arch["aura_v3_genesis"]["known_med_log"],
                d_by_arch[other]["known_med_log"],
            ),
            "known_acc16": paired_diff(
                d_by_arch["aura_v3_genesis"]["known_acc16"],
                d_by_arch[other]["known_acc16"],
            ),
            "new_acc16": paired_diff(
                d_by_arch["aura_v3_genesis"]["new_acc16"],
                d_by_arch[other]["new_acc16"],
            ),
        }

    summary = {
        "seeds": [f"0x{s:04X}" for s in SEEDS],
        "elapsed_sec": round(total_sec, 3),
        "arena1_bytes": {
            "per_arch": arena1_stats,
            "aura_v3_vs_others": arena1_vs,
        },
        "arena2_dynamics": {
            "per_arch": arena2_stats,
            "aura_v3_vs_others": arena2_vs,
        },
    }
    out_json = DOCS_V3_DIR / "top5_equal_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n================ ARENA 1: BYTES REALES (Markdown -> Rust, 4 epocas iguales, N=16) ================")
    for a in ARCH_ORDER_BYTES:
        s = arena1_stats[a]
        print(
            f"  {a:17s} | eval_MD={s['eval_md_bpb']['mean']:.4f} [{s['eval_md_bpb']['ci95_lo']:.4f}, {s['eval_md_bpb']['ci95_hi']:.4f}]"
            f" | Rust_shift={s['shift_rust_bpb']['mean']:.4f} [{s['shift_rust_bpb']['ci95_lo']:.4f}, {s['shift_rust_bpb']['ci95_hi']:.4f}]"
            f" | early_MD={s['early_md_bpb']['mean']:.4f}"
            f" | RAM={int(s['active_weight_bytes']['mean']):6d}B (nz={s['nonzero_frac']['mean']*100:5.1f}%)"
        )

    print("\n--- Contrastes Pareados Arena 1 (AURA v3 Genesis menos Otra; < 0 gana AURA v3) ---")
    for k, v in arena1_vs.items():
        em = v["eval_md_bpb"]
        rs = v["shift_rust_bpb"]
        print(
            f"  {k:34s} | Δeval_MD={em['mean']:+.4f} [{em['ci95_lo']:+.4f}, {em['ci95_hi']:+.4f}] (gana {em['wins_a_lower']}/16)"
            f" | ΔRust={rs['mean']:+.4f} [{rs['ci95_lo']:+.4f}, {rs['ci95_hi']:+.4f}] (gana {rs['wins_a_lower']}/16)"
        )

    print("\n================ ARENA 2: DINAMICA GEOMETRICA 2D CST4 (K=4 -> h=16, 320 eps iguales, N=16) ================")
    for a in ARCH_ORDER_DYN:
        s = arena2_stats[a]
        print(
            f"  {a:17s} | new_med_log={s['new_med_log']['mean']:+.4f} [{s['new_med_log']['ci95_lo']:+.4f}, {s['new_med_log']['ci95_hi']:+.4f}]"
            f" | known_med_log={s['known_med_log']['mean']:+.4f}"
            f" | known_acc16={s['known_acc16']['mean']*100:5.2f}%"
            f" | new_acc16={s['new_acc16']['mean']*100:5.2f}%"
            f" | syn={s['active_synapses']['mean']:.1f}"
        )

    print("\n--- Contrastes Pareados Arena 2 (AURA v3 Genesis menos Otra) ---")
    for k, v in arena2_vs.items():
        nm = v["new_med_log"]
        ka = v["known_acc16"]
        na = v["new_acc16"]
        print(
            f"  {k:34s} | Δnew_med_log={nm['mean']:+.4f} [{nm['ci95_lo']:+.4f}, {nm['ci95_hi']:+.4f}] (gana {nm['wins_a_lower']}/16)"
            f" | Δknown_acc16={ka['mean']*100:+.2f}pp | Δnew_acc16={na['mean']*100:+.2f}pp [{na['ci95_lo']*100:+.2f}, {na['ci95_hi']*100:+.2f}]"
        )


if __name__ == "__main__":
    main()
