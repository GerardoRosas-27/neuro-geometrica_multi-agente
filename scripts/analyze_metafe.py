#!/usr/bin/env python3
"""Análisis preregistrado de la Familia METAFE (docs/metafe/PREREG.md).

Bootstrap pareado por semilla (10 000 réplicas, seed 12345, IC95 percentil, δ=0.05),
siguiendo exactamente la convención canónica de scripts/analyze_bq10.py.

Empareja semilla a semilla artifacts/metafe/{dev,confirm}/metrics_0x*.json contra
artifacts/bio_quantum_v10/{bq_dev,bq_confirm}/metrics_0x*.json verificando
test_sha256 bit-idéntico y test_queries == 0 en las 32 semillas.

Uso:
  python3 scripts/analyze_metafe.py              # imprime JSON completo
  python3 scripts/analyze_metafe.py --md         # imprime resumen en Markdown
"""
import glob
import json
import sys
import numpy as np

D = 0.05
NB = 10000
METAFE_ROOT = "artifacts/metafe"
BQ_ROOT = "artifacts/bio_quantum_v10"
FROZEN = json.load(open("docs/metafe/frozen.json"))

METAFE_ARMS = [
    "FE-ARD",
    "GP-Cart-FE",
    "GP-Symp-FE",
    "GP-Polar-FE",
    "METAFE-NoFE",
    "METAFE-NoRoll",
    "METAFE",
]
BASELINES = ["ORBITA", "v4", "CORE", "ALL", "BQ10"]
HELD_KEYS = ("new_med_log", "new_acc16", "known_med_log", "known_acc16")
SYN_KEYS = ("syn_ood_med_log", "syn_ood_acc16")
METACOG_METRICS = (
    "auroc",
    "brier",
    "ece",
    "aurc",
    "err_pred_corr",
    "err_pred_spearman",
    "fe_reduction_mean",
    "fe_reduced_frac",
    "mean_epi",
    "ens_disagree",
)


def ci(x):
    """Exact canonical paired bootstrap (matches scripts/analyze_bq10.py)."""
    arr = np.asarray(x, float)
    assert np.all(np.isfinite(arr)), f"Non-finite values in bootstrap input: {arr}"
    rng = np.random.default_rng(12345)
    boot = arr[rng.integers(0, len(arr), (NB, len(arr)))].mean(1)
    return [float(arr.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def load_phase_dir(pattern):
    data = {}
    for path in sorted(glob.glob(pattern)):
        m = json.load(open(path))
        data[m["seed"]] = m
    return data


def get_base_arm(bq_seed_rec, ref):
    if ref == "ORBITA":
        return {
            "held": bq_seed_rec["orbita"],
            "e33": bq_seed_rec["e33_orbita"],
            "nan": False,
            "wall_s": None,
            "total_e2e_s": None,
        }
    arm = bq_seed_rec["arms"][ref]
    v4_train_s = float(bq_seed_rec["v4_train"]["train_s"])
    return {
        "held": arm["held"],
        "e33": arm["e33"],
        "e33_v3head": arm.get("e33_v3head"),
        "nan": bool(arm["nan"]),
        "wall_s": float(arm["wall_s"]),
        "total_e2e_s": v4_train_s + float(arm["wall_s"]),
    }


def agg_metacog_section(recs, path_fn):
    out = {}
    for k in METACOG_METRICS:
        vals = [path_fn(r).get(k) for r in recs]
        if all(v is not None and np.isfinite(v) for v in vals):
            out[k] = ci(vals)
    # Optional extra fields present in overall
    for extra in ("fe_null_mean", "fe_post_mean", "err_pred_raw_corr", "post_entropy_mean"):
        vals = [path_fn(r).get(extra) for r in recs]
        if all(v is not None and np.isfinite(v) for v in vals):
            out[extra] = ci(vals)
    # Hypothesis selection fractions if present
    sample = path_fn(recs[0])
    if "hyp_selection" in sample:
        all_hyps = sorted({h for r in recs for h in path_fn(r).get("hyp_selection", {}).keys()})
        out["hyp_selection"] = {
            h: float(np.mean([path_fn(r).get("hyp_selection", {}).get(h, 0.0) for r in recs]))
            for h in all_hyps
        }
    return out


def analyze_phase(metafe_ph, bq_ph):
    M = load_phase_dir(f"{METAFE_ROOT}/{metafe_ph}/metrics_0x*.json")
    B = load_phase_dir(f"{BQ_ROOT}/{bq_ph}/metrics_0x*.json")
    S = sorted(M.keys())
    SB = sorted(B.keys())
    assert len(S) == 16, f"Expected 16 seeds in {metafe_ph}, got {len(S)}"
    assert S == SB, f"Seed mismatch between {metafe_ph} and {bq_ph}: {S} vs {SB}"

    sha_matches = sum(M[s]["test_sha256"] == B[s]["test_sha256"] for s in S)
    assert sha_matches == 16, f"test_sha256 mismatch in {metafe_ph}: {sha_matches}/16"
    assert all(M[s]["test_queries"] == 0 for s in S), f"Non-zero test_queries in METAFE {metafe_ph}"
    assert all(B[s]["test_queries"] == 0 for s in S), f"Non-zero test_queries in BQ {bq_ph}"

    out = {
        "phase": metafe_ph,
        "seeds": S,
        "n_seeds": len(S),
        "test_sha256_match": f"{sha_matches}/{len(S)}",
        "test_queries_total": int(sum(M[s]["test_queries"] for s in S)),
        "git_shas": sorted({M[s]["git_sha"] for s in S}),
        "baselines": {},
        "arms": {},
        "ablations": {},
    }

    # Baselines from bq_{dev,confirm}
    for ref in BASELINES:
        recs = [get_base_arm(B[s], ref) for s in S]
        entry = {
            "means": {k: ci([r["held"][k] for r in recs]) for k in HELD_KEYS},
            "e33_pass": sum(bool(r["e33"]["pass"]) for r in recs),
            "nan": sum(bool(r["nan"]) for r in recs),
        }
        if recs[0]["wall_s"] is not None:
            entry["wall_s"] = ci([r["wall_s"] for r in recs])
            entry["total_e2e_s"] = ci([r["total_e2e_s"] for r in recs])
        if recs[0].get("e33_v3head") is not None:
            entry["e33_v3head_pass"] = sum(bool(r["e33_v3head"]["pass"]) for r in recs)
            entry["e33_v3head_h1"] = ci([r["e33_v3head"]["acc"][0] for r in recs])
        out["baselines"][ref] = entry

    # METAFE arms
    for a in METAFE_ARMS:
        recs = [M[s]["arms"][a] for s in S]
        entry = {
            "means": {k: ci([r["held"][k] for r in recs]) for k in HELD_KEYS},
            "syn_ood": {k: ci([r["syn_ood"][k] for r in recs]) for k in SYN_KEYS},
            "e33_pass": sum(bool(r["e33"]["pass"]) for r in recs),
            "e33_h1": ci([r["e33"]["acc"][0] for r in recs]),
            "e33_h16": ci([r["e33"]["acc"][4] for r in recs]),
            "e33_auto_pass": sum(bool(r["e33_autonomous"]["pass"]) for r in recs),
            "e33_auto_h1": ci([r["e33_autonomous"]["acc"][0] for r in recs]),
            "e33_auto_h16": ci([r["e33_autonomous"]["acc"][4] for r in recs]),
            "e33_raw_pass": sum(bool(r["e33_raw"]["pass"]) for r in recs),
            "e33_raw_h1": ci([r["e33_raw"]["acc"][0] for r in recs]),
            "nan": sum(bool(r["nan"]) for r in recs),
            "wall_s": ci([float(r["wall_s"]) for r in recs]),
            "train_s": ci([float(r["train_s"]) for r in recs]),
            "eval_s": ci([float(r["eval_s"]) for r in recs]),
            "per_fam_med_log": {},
            "metacog": {
                "overall": agg_metacog_section(recs, lambda r: r["metacog"]["overall"]),
                "known": agg_metacog_section(recs, lambda r: r["metacog"]["known_metacog"]),
                "new": agg_metacog_section(recs, lambda r: r["metacog"]["new_metacog"]),
                "syn_ood": agg_metacog_section(recs, lambda r: r["syn_metacog"]["syn_ood_metacog"]),
            },
        }
        # Per-family median log error
        fams = sorted(
            recs[0]["metacog"]["overall"]["per_fam_med_log"].keys(),
            key=lambda x: int(x.split("_")[1]),
        )
        for fam in fams:
            entry["per_fam_med_log"][fam] = ci(
                [r["metacog"]["overall"]["per_fam_med_log"][fam] for r in recs]
            )

        # Paired contrasts vs every baseline
        for ref in BASELINES:
            b_recs = [get_base_arm(B[s], ref) for s in S]
            entry[f"vs_{ref}"] = {
                k: ci([r["held"][k] - br["held"][k] for r, br in zip(recs, b_recs)])
                for k in HELD_KEYS
            }

        # Speedup vs BQ10 and ALL (end-to-end and head-only)
        bq_e2e = np.mean([get_base_arm(B[s], "BQ10")["total_e2e_s"] for s in S])
        bq_head = np.mean([get_base_arm(B[s], "BQ10")["wall_s"] for s in S])
        all_e2e = np.mean([get_base_arm(B[s], "ALL")["total_e2e_s"] for s in S])
        all_head = np.mean([get_base_arm(B[s], "ALL")["wall_s"] for s in S])
        arm_wall = entry["wall_s"][0]
        entry["speedup"] = {
            "vs_BQ10_e2e": float(bq_e2e / arm_wall),
            "vs_BQ10_head": float(bq_head / arm_wall),
            "vs_ALL_e2e": float(all_e2e / arm_wall),
            "vs_ALL_head": float(all_head / arm_wall),
        }
        out["arms"][a] = entry

    # Causal ablation paired contrasts
    ablation_pairs = [
        ("METAFE", "METAFE-NoFE"),
        ("METAFE", "METAFE-NoRoll"),
        ("METAFE-NoRoll", "METAFE-NoFE"),
        ("METAFE", "FE-ARD"),
        ("METAFE", "GP-Cart-FE"),
        ("METAFE", "GP-Symp-FE"),
        ("METAFE", "GP-Polar-FE"),
    ]
    for a, b in ablation_pairs:
        ra = [M[s]["arms"][a] for s in S]
        rb = [M[s]["arms"][b] for s in S]
        pair_key = f"{a}_minus_{b}"
        res = {
            k: ci([x["held"][k] - y["held"][k] for x, y in zip(ra, rb)])
            for k in HELD_KEYS
        }
        for k in SYN_KEYS:
            res[k] = ci([x["syn_ood"][k] - y["syn_ood"][k] for x, y in zip(ra, rb)])
        for mk in ("auroc", "brier", "ece", "aurc"):
            res[f"overall_{mk}"] = ci(
                [x["metacog"]["overall"][mk] - y["metacog"]["overall"][mk] for x, y in zip(ra, rb)]
            )
            res[f"new_{mk}"] = ci(
                [x["metacog"]["new_metacog"][mk] - y["metacog"]["new_metacog"][mk] for x, y in zip(ra, rb)]
            )
        out["ablations"][pair_key] = res

    return out


def win_vs_ref(arm_phase, ref):
    v = arm_phase[f"vs_{ref}"]
    return (
        v["new_med_log"][2] < 0.0
        and abs(v["new_med_log"][0]) >= D
        and v["known_acc16"][1] > -D
        and arm_phase["e33_pass"] == 16
        and arm_phase["nan"] == 0
    )


def win_delta_neg(delta_ci):
    return delta_ci[2] < 0.0 and abs(delta_ci[0]) >= D


def evaluate_verdicts(dev, confirm):
    P = FROZEN["primary_arm"]
    verdicts = {}

    # H1, H2: Primary METAFE vs BQ10, ALL, CORE, v4, ORBITA
    for ref in ("BQ10", "ALL", "CORE", "v4", "ORBITA"):
        ok_d = win_vs_ref(dev["arms"][P], ref)
        ok_c = win_vs_ref(confirm["arms"][P], ref)
        verdicts[f"{P}_vs_{ref}"] = (
            "PASA (DEV y CONFIRM)"
            if (ok_d and ok_c)
            else ("PASA SOLO CONFIRM" if ok_c else ("PASA SOLO DEV" if ok_d else "NO PASA"))
        )

    # Secondary arms vs BQ10 and v4
    for a in FROZEN["secondary_arms"]:
        for ref in ("BQ10", "ALL", "v4"):
            ok_d = win_vs_ref(dev["arms"][a], ref)
            ok_c = win_vs_ref(confirm["arms"][a], ref)
            verdicts[f"{a}_vs_{ref}"] = (
                "PASA (DEV y CONFIRM)"
                if (ok_d and ok_c)
                else ("PASA SOLO CONFIRM" if ok_c else ("PASA SOLO DEV" if ok_d else "NO PASA"))
            )

    # H3: Causal Free Energy (METAFE - METAFE-NoFE)
    h3_d = win_delta_neg(dev["ablations"]["METAFE_minus_METAFE-NoFE"]["new_med_log"])
    h3_c = win_delta_neg(confirm["ablations"]["METAFE_minus_METAFE-NoFE"]["new_med_log"])
    verdicts["H3_causal_FE_new_med_log"] = (
        "PASA (DEV y CONFIRM)" if (h3_d and h3_c) else f"DEV={h3_d}, CONFIRM={h3_c}"
    )

    # H4: Causal Active Multi-Step Inference (METAFE - METAFE-NoRoll)
    h4_new_d = dev["ablations"]["METAFE_minus_METAFE-NoRoll"]["new_med_log"]
    h4_new_c = confirm["ablations"]["METAFE_minus_METAFE-NoRoll"]["new_med_log"]
    h4_acc_d = dev["ablations"]["METAFE_minus_METAFE-NoRoll"]["new_acc16"]
    h4_acc_c = confirm["ablations"]["METAFE_minus_METAFE-NoRoll"]["new_acc16"]
    h4_syn_d = dev["ablations"]["METAFE_minus_METAFE-NoRoll"]["syn_ood_med_log"]
    h4_syn_c = confirm["ablations"]["METAFE_minus_METAFE-NoRoll"]["syn_ood_med_log"]
    verdicts["H4_NoRoll_new_med_log"] = (
        "PASA (DEV y CONFIRM)"
        if (win_delta_neg(h4_new_d) and win_delta_neg(h4_new_c))
        else (
            "MEJORA SIGNIFICATIVA (IC<0 ambos, |Δ|<0.05 en al menos una fase)"
            if (h4_new_d[2] < 0 and h4_new_c[2] < 0)
            else f"NO PASA UMBRAL δ=0.05 (DEV={h4_new_d[0]:+.4f} [{h4_new_d[1]:+.4f},{h4_new_d[2]:+.4f}], CONFIRM={h4_new_c[0]:+.4f} [{h4_new_c[1]:+.4f},{h4_new_c[2]:+.4f}])"
        )
    )
    verdicts["H4_NoRoll_new_acc16_ci_gt0"] = (
        "PASA (DEV y CONFIRM)"
        if (h4_acc_d[1] > 0 and h4_acc_c[1] > 0)
        else f"DEV={h4_acc_d[0]:+.4f} [{h4_acc_d[1]:+.4f},{h4_acc_d[2]:+.4f}], CONFIRM={h4_acc_c[0]:+.4f} [{h4_acc_c[1]:+.4f},{h4_acc_c[2]:+.4f}]"
    )
    verdicts["H4_NoRoll_syn_ood_med_log"] = (
        "PASA (DEV y CONFIRM)"
        if (win_delta_neg(h4_syn_d) and win_delta_neg(h4_syn_c))
        else f"DEV={h4_syn_d[0]:+.4f} [{h4_syn_d[1]:+.4f},{h4_syn_d[2]:+.4f}], CONFIRM={h4_syn_c[0]:+.4f} [{h4_syn_c[1]:+.4f},{h4_syn_c[2]:+.4f}]"
    )

    # Safety & Autonomy gates
    verdicts["gate_e33_canonical"] = (
        "PASA (16/16 DEV y 16/16 CONFIRM)"
        if (dev["arms"][P]["e33_pass"] == 16 and confirm["arms"][P]["e33_pass"] == 16)
        else f"FALLA (DEV={dev['arms'][P]['e33_pass']}/16, CONFIRM={confirm['arms'][P]['e33_pass']}/16)"
    )
    verdicts["gate_e33_autonomous"] = (
        "PASA (16/16 DEV y 16/16 CONFIRM)"
        if (dev["arms"][P]["e33_auto_pass"] == 16 and confirm["arms"][P]["e33_auto_pass"] == 16)
        else f"FALLA (DEV={dev['arms'][P]['e33_auto_pass']}/16, CONFIRM={confirm['arms'][P]['e33_auto_pass']}/16)"
    )
    verdicts["gate_no_nan_zero_test"] = (
        "PASA (0 NaN, 0 TEST, 32/32 SHA-256)"
        if (
            dev["arms"][P]["nan"] == 0
            and confirm["arms"][P]["nan"] == 0
            and dev["test_queries_total"] == 0
            and confirm["test_queries_total"] == 0
        )
        else "FALLA"
    )

    # Metacognition & Free Energy gates from PREREG.md §5
    for ph_name, ph in (("dev", dev), ("confirm", confirm)):
        mc = ph["arms"][P]["metacog"]
        fe_frac = mc["overall"]["fe_reduced_frac"][0]
        ov_auroc = mc["overall"]["auroc"][0]
        ov_brier = mc["overall"]["brier"][0]
        nw_auroc = mc["new"]["auroc"][0]
        nw_r = mc["new"]["err_pred_corr"][0]
        ok = (
            fe_frac >= 0.99
            and ov_auroc >= 0.90
            and ov_brier <= 0.08
            and nw_auroc >= 0.75
            and nw_r >= 0.70
        )
        verdicts[f"gate_metacog_fe_{ph_name}"] = (
            f"PASA (FE_frac={fe_frac:.4f}, ov_AUROC={ov_auroc:.4f}, ov_Brier={ov_brier:.4f}, new_AUROC={nw_auroc:.4f}, new_r={nw_r:.4f})"
            if ok
            else f"REVISAR (FE_frac={fe_frac:.4f}, ov_AUROC={ov_auroc:.4f}, ov_Brier={ov_brier:.4f}, new_AUROC={nw_auroc:.4f}, new_r={nw_r:.4f})"
        )

    return verdicts


def fmt_ci(v, signed=False, digits=4):
    s = "+" if signed else ""
    return f"{v[0]:{s}.{digits}f} [{v[1]:{s}.{digits}f}, {v[2]:{s}.{digits}f}]"


def render_md(out):
    L = [
        "# Familia METAFE — Resumen Estadístico Preregistrado (DEV y CONFIRM)",
        "",
        f"Primario congelado: `{FROZEN['primary_arm']}`. Bootstrap pareado por semilla (10 000 réplicas, seed 12345, IC95 percentil, δ=0.05). Verificación criptográfica `test_sha256`: DEV {out['dev']['test_sha256_match']}, CONFIRM {out['confirm']['test_sha256_match']}; `test_queries = 0`.",
        "",
    ]
    for ph_name in ("dev", "confirm"):
        ph = out[ph_name]
        L += [
            f"## {ph_name.upper()} ({ph['seeds'][0]}–{ph['seeds'][-1]}, n={ph['n_seeds']})",
            "",
            "| Arquitectura | new_med_log | new_acc16 | known_acc16 | Δ vs BQ10 new_med_log | Δ vs ALL new_med_log | E33 | E33 Auto | wall_s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for ref in BASELINES:
            b = ph["baselines"][ref]
            w = f"{b['wall_s'][0]:.1f} ({b['total_e2e_s'][0]:.0f} e2e)" if "wall_s" in b else "—"
            auto = "—" if ref != "ORBITA" else f"{b['e33_pass']}/16"
            if "e33_v3head_pass" in b:
                auto = f"{b['e33_v3head_pass']}/16 (v3h)"
            L.append(
                f"| {ref} | {fmt_ci(b['means']['new_med_log'], True)} | {b['means']['new_acc16'][0]:.4f} | {b['means']['known_acc16'][0]:.4f} | — | — | {b['e33_pass']}/16 | {auto} | {w} |"
            )
        for a in METAFE_ARMS:
            x = ph["arms"][a]
            name = f"**{a}**" if a == "METAFE" else a
            L.append(
                f"| {name} | {fmt_ci(x['means']['new_med_log'], True)} | {x['means']['new_acc16'][0]:.4f} | {x['means']['known_acc16'][0]:.4f} | {fmt_ci(x['vs_BQ10']['new_med_log'], True)} | {fmt_ci(x['vs_ALL']['new_med_log'], True)} | {x['e33_pass']}/16 | {x['e33_auto_pass']}/16 ({x['e33_auto_h1'][0]:.3f}) | {x['wall_s'][0]:.1f} |"
            )
        L += [
            "",
            "### Ablaciones Causales Pareadas",
            "",
            "| Contraste | Δ new_med_log | Δ new_acc16 | Δ known_acc16 | Δ syn_ood_med_log | Δ syn_ood_acc16 |",
            "|---|---|---|---|---|---|",
        ]
        for pk in ("METAFE_minus_METAFE-NoFE", "METAFE_minus_METAFE-NoRoll", "METAFE-NoRoll_minus_METAFE-NoFE"):
            d = ph["ablations"][pk]
            L.append(
                f"| `{pk}` | {fmt_ci(d['new_med_log'], True)} | {fmt_ci(d['new_acc16'], True)} | {fmt_ci(d['known_acc16'], True)} | {fmt_ci(d['syn_ood_med_log'], True)} | {fmt_ci(d['syn_ood_acc16'], True)} |"
            )
        L += [
            "",
            "### Metacognición y Reducción de Energía Libre (`METAFE`)",
            "",
            "| Subconjunto | AUROC | Brier | ECE | AURC | Pearson r | Spearman ρ | ΔF (nats) | % ΔF>0 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        mc = ph["arms"]["METAFE"]["metacog"]
        for sub in ("overall", "known", "new", "syn_ood"):
            s = mc[sub]
            L.append(
                f"| {sub} | {s['auroc'][0]:.4f} | {s['brier'][0]:.4f} | {s['ece'][0]:.4f} | {s['aurc'][0]:.4f} | {s['err_pred_corr'][0]:.4f} | {s['err_pred_spearman'][0]:.4f} | {s['fe_reduction_mean'][0]:+.2f} | {100*s['fe_reduced_frac'][0]:.2f}% |"
            )
        L.append("")

    L += ["## Veredictos Preregistrados", ""]
    for k, v in out["verdict"].items():
        L.append(f"- **{k}:** {v}")
    return "\n".join(L)


def main():
    dev = analyze_phase("dev", "bq_dev")
    confirm = analyze_phase("confirm", "bq_confirm")
    out = {
        "frozen": FROZEN,
        "dev": dev,
        "confirm": confirm,
        "verdict": evaluate_verdicts(dev, confirm),
    }
    if "--md" in sys.argv:
        print(render_md(out))
    else:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
