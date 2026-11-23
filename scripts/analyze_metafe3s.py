#!/usr/bin/env python3
"""Análisis preregistrado de METAFE-3S (Control Metacognitivo de Tres Escalas sobre Energía Libre).

Bootstrap pareado por semilla (10 000 réplicas, seed 12345, IC95 percentil, δ=0.05),
idéntico al protocolo canónico de scripts/analyze_metafe.py y scripts/analyze_bq10.py.

Empareja semilla a semilla:
  - artifacts/metafe3s/{dev,confirm}/metrics_0x*.json
  - artifacts/metafe/{dev,confirm}/metrics_0x*.json  (verifica paridad bit a bit de METAFE-v2)
  - artifacts/bio_quantum_v10/{bq_dev,bq_confirm}/metrics_0x*.json
verificando test_sha256 bit-idéntico 32/32 y test_queries == 0 en las 32 semillas.

Uso:
  python3 scripts/analyze_metafe3s.py --tune      # resumen rápido de 4 semillas tune (0xC4B0-0xC4B3)
  python3 scripts/analyze_metafe3s.py             # JSON completo DEV + CONFIRM + veredictos
  python3 scripts/analyze_metafe3s.py --md        # resumen Markdown DEV + CONFIRM + veredictos
"""
import glob
import json
import os
import sys
import numpy as np

D = 0.05
NB = 10000
ROOT_3S = "artifacts/metafe3s"
ROOT_V2 = "artifacts/metafe"
ROOT_BQ = "artifacts/bio_quantum_v10"

ARMS_3S = [
    "METAFE-v2",
    "METAFE-3S",
    "AURA-v3-Genesis",
    "METAFE-3S-NoFast",
    "METAFE-3S-NoRefute",
    "METAFE-3S-NoNM",
]
BASELINES = ["ORBITA", "v4", "CORE", "ALL", "BQ10"]
HELD_KEYS = ("new_med_log", "new_acc16", "known_med_log", "known_acc16")
SYN_KEYS = ("syn_ood_med_log", "syn_ood_acc16")
SHIFT_KEYS = (
    "rs_a_known_med_log",
    "rs_a_known_acc16",
    "rs_a_ood_med_log",
    "rs_a_ood_acc16",
    "rs_a_all_med_log",
    "rs_a_reset_rate",
    "rs_a_min_lambda_mean",
    "rs_b_frozen_w0_med_log",
    "rs_b_frozen_w0_mean_log",
    "rs_b_frozen_w0_acc16",
)
METACOG_METRICS = (
    "auroc",
    "brier",
    "ece",
    "aurc",
    "err_pred_corr",
    "err_pred_spearman",
    "fe_reduction_mean",
    "fe_reduced_frac",
)
EXTRA_3S_METACOG = (
    "mean_refute",
    "fast_used_frac",
    "mean_alpha",
    "mean_beta",
    "cycle_accepted_total",
    "fe_null_mean",
    "fe_post_mean",
    "err_pred_raw_corr",
)


def ci(x):
    """Exact canonical paired bootstrap (matches scripts/analyze_bq10.py & analyze_metafe.py)."""
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
    for k in METACOG_METRICS + EXTRA_3S_METACOG:
        vals = [path_fn(r).get(k) for r in recs]
        if all(v is not None and np.isfinite(v) for v in vals):
            out[k] = ci(vals)
    sample = path_fn(recs[0])
    if "hyp_selection" in sample:
        all_hyps = sorted({h for r in recs for h in path_fn(r).get("hyp_selection", {}).keys()})
        out["hyp_selection"] = {
            h: float(np.mean([path_fn(r).get("hyp_selection", {}).get(h, 0.0) for r in recs]))
            for h in all_hyps
        }
    return out


def analyze_phase(ph_3s, ph_v2, ph_bq, expected_n=16):
    M = load_phase_dir(f"{ROOT_3S}/{ph_3s}/metrics_0x*.json")
    S = sorted(M.keys())
    assert len(S) == expected_n, f"Expected {expected_n} seeds in {ph_3s}, got {len(S)}"
    assert all(M[s]["test_queries"] == 0 for s in S), f"Non-zero test_queries in {ph_3s}"

    B = load_phase_dir(f"{ROOT_BQ}/{ph_bq}/metrics_0x*.json") if ph_bq else {}
    V2 = load_phase_dir(f"{ROOT_V2}/{ph_v2}/metrics_0x*.json") if ph_v2 else {}

    has_bq = len(B) == len(S) and sorted(B.keys()) == S
    has_v2 = len(V2) == len(S) and sorted(V2.keys()) == S
    if expected_n == 16:
        assert has_bq, f"Missing or mismatched BQ10 artifacts for {ph_bq}"
        assert has_v2, f"Missing or mismatched METAFE v2 artifacts for {ph_v2}"

    sha_bq = sum(M[s]["test_sha256"] == B[s]["test_sha256"] for s in S) if has_bq else 0
    sha_v2 = sum(M[s]["test_sha256"] == V2[s]["test_sha256"] for s in S) if has_v2 else 0
    if has_bq:
        assert sha_bq == len(S), f"test_sha256 mismatch vs BQ10 in {ph_3s}: {sha_bq}/{len(S)}"
        assert all(B[s]["test_queries"] == 0 for s in S)
    if has_v2:
        assert sha_v2 == len(S), f"test_sha256 mismatch vs METAFE v2 in {ph_3s}: {sha_v2}/{len(S)}"

    # Verify bit-parity of paired METAFE-v2 against frozen artifacts/metafe/
    v2_parity_max_diff = None
    if has_v2 and "METAFE-v2" in M[S[0]]["arms"]:
        diffs = []
        for s in S:
            for k in HELD_KEYS:
                a = M[s]["arms"]["METAFE-v2"]["held"][k]
                b = V2[s]["arms"]["METAFE"]["held"][k]
                diffs.append(abs(a - b))
        v2_parity_max_diff = float(max(diffs))

    out = {
        "phase": ph_3s,
        "seeds": S,
        "n_seeds": len(S),
        "test_sha256_match_bq": f"{sha_bq}/{len(S)}" if has_bq else "n/a",
        "test_sha256_match_v2": f"{sha_v2}/{len(S)}" if has_v2 else "n/a",
        "v2_parity_max_diff": v2_parity_max_diff,
        "test_queries_total": int(sum(M[s]["test_queries"] for s in S)),
        "git_shas": sorted({M[s]["git_sha"] for s in S}),
        "baselines": {},
        "arms": {},
        "contrasts": {},
        "continual_ewc": {},
    }

    if has_bq:
        for ref in BASELINES:
            if ref not in B[S[0]]["arms"] and ref != "ORBITA":
                continue
            recs = [get_base_arm(B[s], ref) for s in S]
            entry = {
                "means": {k: ci([r["held"][k] for r in recs]) for k in HELD_KEYS},
                "e33_pass": sum(bool(r["e33"]["pass"]) for r in recs),
                "nan": sum(bool(r["nan"]) for r in recs),
            }
            if recs[0]["wall_s"] is not None:
                entry["wall_s"] = ci([r["wall_s"] for r in recs])
                entry["total_e2e_s"] = ci([r["total_e2e_s"] for r in recs])
            out["baselines"][ref] = entry

    avail_arms = [a for a in ARMS_3S if a in M[S[0]]["arms"]]
    for a in avail_arms:
        recs = [M[s]["arms"][a] for s in S]
        entry = {
            "means": {k: ci([r["held"][k] for r in recs]) for k in HELD_KEYS},
            "syn_ood": {k: ci([r["syn_ood"][k] for r in recs]) for k in SYN_KEYS},
            "shift": {k: ci([r["shift"][k] for r in recs]) for k in SHIFT_KEYS},
            "e33_pass": sum(bool(r["e33"]["pass"]) for r in recs),
            "e33_h1": ci([r["e33"]["acc"][0] for r in recs]),
            "e33_h16": ci([r["e33"]["acc"][4] for r in recs]),
            "e33_auto_pass": sum(bool(r["e33_autonomous"]["pass"]) for r in recs),
            "e33_auto_h1": ci([r["e33_autonomous"]["acc"][0] for r in recs]),
            "e33_auto_h16": ci([r["e33_autonomous"]["acc"][4] for r in recs]),
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
        fams = sorted(
            recs[0]["metacog"]["overall"]["per_fam_med_log"].keys(),
            key=lambda x: int(x.split("_")[1]),
        )
        for fam in fams:
            entry["per_fam_med_log"][fam] = ci(
                [r["metacog"]["overall"]["per_fam_med_log"][fam] for r in recs]
            )
        if has_bq:
            for ref in out["baselines"].keys():
                b_recs = [get_base_arm(B[s], ref) for s in S]
                entry[f"vs_{ref}"] = {
                    k: ci([r["held"][k] - br["held"][k] for r, br in zip(recs, b_recs)])
                    for k in HELD_KEYS
                }
            if "BQ10" in out["baselines"]:
                bq_e2e = np.mean([get_base_arm(B[s], "BQ10")["total_e2e_s"] for s in S])
                entry["speedup_vs_BQ10_e2e"] = float(bq_e2e / entry["wall_s"][0])
        out["arms"][a] = entry

    # Paired contrasts among 3S arms, AURA-v3-Genesis, and METAFE-v2
    pairs = [
        ("METAFE-3S", "METAFE-v2"),
        ("AURA-v3-Genesis", "METAFE-3S"),
        ("AURA-v3-Genesis", "METAFE-v2"),
        ("METAFE-3S", "METAFE-3S-NoFast"),
        ("METAFE-3S", "METAFE-3S-NoRefute"),
        ("METAFE-3S", "METAFE-3S-NoNM"),
    ]
    for a, b in pairs:
        if a not in avail_arms or b not in avail_arms:
            continue
        ra = [M[s]["arms"][a] for s in S]
        rb = [M[s]["arms"][b] for s in S]
        res = {}
        for k in HELD_KEYS:
            res[k] = ci([x["held"][k] - y["held"][k] for x, y in zip(ra, rb)])
        for k in SYN_KEYS:
            res[k] = ci([x["syn_ood"][k] - y["syn_ood"][k] for x, y in zip(ra, rb)])
        for k in SHIFT_KEYS:
            res[k] = ci([x["shift"][k] - y["shift"][k] for x, y in zip(ra, rb)])
        for sub, path in (
            ("overall", lambda r: r["metacog"]["overall"]),
            ("new", lambda r: r["metacog"]["new_metacog"]),
            ("syn", lambda r: r["syn_metacog"]["syn_ood_metacog"]),
        ):
            for mk in ("auroc", "brier", "ece", "aurc", "err_pred_corr", "err_pred_spearman"):
                va = [path(x).get(mk) for x in ra]
                vb = [path(y).get(mk) for y in rb]
                if all(v is not None and np.isfinite(v) for v in va + vb):
                    res[f"{sub}_{mk}"] = ci([x - y for x, y in zip(va, vb)])
        res["win_seeds_new_med_log"] = int(
            sum(x["held"]["new_med_log"] < y["held"]["new_med_log"] for x, y in zip(ra, rb))
        )
        res["win_seeds_rs_a_all"] = int(
            sum(x["shift"]["rs_a_all_med_log"] < y["shift"]["rs_a_all_med_log"] for x, y in zip(ra, rb))
        )
        res["win_seeds_rs_b"] = int(
            sum(
                x["shift"]["rs_b_frozen_w0_med_log"] < y["shift"]["rs_b_frozen_w0_med_log"]
                for x, y in zip(ra, rb)
            )
        )
        out["contrasts"][f"{a}_minus_{b}"] = res

    # Continual EWC vs NoEWC offline consolidation
    ce = [M[s]["continual_ewc"] for s in S]
    out["continual_ewc"] = {
        "ewc_t1_forgetting_ce": ci([r["ewc"]["t1_forgetting_ce"] for r in ce]),
        "noewc_t1_forgetting_ce": ci([r["no_ewc"]["t1_forgetting_ce"] for r in ce]),
        "delta_forgetting_ewc_minus_noewc": ci([r["delta_forgetting_ewc_minus_noewc"] for r in ce]),
        "ewc_avg_ce": ci([r["ewc"]["avg_ce"] for r in ce]),
        "noewc_avg_ce": ci([r["no_ewc"]["avg_ce"] for r in ce]),
        "delta_avg_ce_ewc_minus_noewc": ci([r["delta_avg_ce_ewc_minus_noewc"] for r in ce]),
        "ewc_avg_acc": ci([r["ewc"]["avg_acc"] for r in ce]),
        "noewc_avg_acc": ci([r["no_ewc"]["avg_acc"] for r in ce]),
        "delta_avg_acc_ewc_minus_noewc": ci(
            [r["ewc"]["avg_acc"] - r["no_ewc"]["avg_acc"] for r in ce]
        ),
        "ewc_wins_forgetting_count": int(sum(bool(r["ewc_wins_forgetting"]) for r in ce)),
    }

    return out


def fmt_ci(v, signed=False, digits=4):
    s = "+" if signed else ""
    return f"{v[0]:{s}.{digits}f} [{v[1]:{s}.{digits}f}, {v[2]:{s}.{digits}f}]"


def win_neg(delta_ci, thresh=D):
    return delta_ci[2] < 0.0 and abs(delta_ci[0]) >= thresh


def evaluate_verdicts(dev, confirm, frozen):
    P = frozen["primary_arm"]
    verdicts = {}

    # H1: METAFE-3S vs repo baselines (BQ10, ALL, CORE, v4, ORBITA)
    for ref in ("BQ10", "ALL", "CORE", "v4", "ORBITA"):
        ok_d = (
            win_neg(dev["arms"][P][f"vs_{ref}"]["new_med_log"])
            and dev["arms"][P][f"vs_{ref}"]["known_acc16"][1] > -D
        )
        ok_c = (
            win_neg(confirm["arms"][P][f"vs_{ref}"]["new_med_log"])
            and confirm["arms"][P][f"vs_{ref}"]["known_acc16"][1] > -D
        )
        verdicts[f"H1_{P}_vs_{ref}"] = (
            "PASA (DEV y CONFIRM)"
            if (ok_d and ok_c)
            else ("PASA SOLO CONFIRM" if ok_c else ("PASA SOLO DEV" if ok_d else "NO PASA"))
        )

    # H2: Head-to-head METAFE-3S vs METAFE-v2 (canonical + OOD metacog)
    c_d = dev["contrasts"]["METAFE-3S_minus_METAFE-v2"]
    c_c = confirm["contrasts"]["METAFE-3S_minus_METAFE-v2"]
    verdicts["H2a_3S_vs_v2_new_med_log"] = (
        "PASA (DEV y CONFIRM, IC<0 y |Δ|≥0.05)"
        if (win_neg(c_d["new_med_log"]) and win_neg(c_c["new_med_log"]))
        else (
            "MEJORA SIGNIFICATIVA (IC<0 ambos)"
            if (c_d["new_med_log"][2] < 0 and c_c["new_med_log"][2] < 0)
            else f"SIN SUPERACIÓN ESTRICTA δ=0.05 (DEV={fmt_ci(c_d['new_med_log'], True)}, CONFIRM={fmt_ci(c_c['new_med_log'], True)})"
        )
    )
    verdicts["H2b_3S_vs_v2_known_no_harm"] = (
        "PASA (DEV y CONFIRM, IC inf > -0.05)"
        if (c_d["known_acc16"][1] > -D and c_c["known_acc16"][1] > -D)
        else f"REVISAR (DEV={fmt_ci(c_d['known_acc16'], True)}, CONFIRM={fmt_ci(c_c['known_acc16'], True)})"
    )
    verdicts["H2c_3S_vs_v2_new_brier"] = (
        "PASA (DEV y CONFIRM, IC sup < 0)"
        if (c_d["new_brier"][2] < 0 and c_c["new_brier"][2] < 0)
        else f"NO PASA AMBOS (DEV={fmt_ci(c_d['new_brier'], True)}, CONFIRM={fmt_ci(c_c['new_brier'], True)})"
    )
    verdicts["H2d_3S_vs_v2_new_auroc"] = (
        "PASA (DEV y CONFIRM, IC inf > 0)"
        if (c_d["new_auroc"][1] > 0 and c_c["new_auroc"][1] > 0)
        else f"NO PASA AMBOS (DEV={fmt_ci(c_d['new_auroc'], True)}, CONFIRM={fmt_ci(c_c['new_auroc'], True)})"
    )

    # H3: Online Non-Stationary Regime Shift RS-A (abrupt law switch: ACC + MetaNet λ_t reset)
    h3_v2_d = win_neg(c_d["rs_a_all_med_log"])
    h3_v2_c = win_neg(c_c["rs_a_all_med_log"])
    nm_d = dev["contrasts"]["METAFE-3S_minus_METAFE-3S-NoNM"]
    nm_c = confirm["contrasts"]["METAFE-3S_minus_METAFE-3S-NoNM"]
    h3_nm_d = win_neg(nm_d["rs_a_all_med_log"])
    h3_nm_c = win_neg(nm_c["rs_a_all_med_log"])
    verdicts["H3_RS_A_vs_v2"] = (
        "PASA (DEV y CONFIRM)" if (h3_v2_d and h3_v2_c) else f"DEV={h3_v2_d}, CONFIRM={h3_v2_c}"
    )
    verdicts["H3_RS_A_causal_NoNM"] = (
        "PASA (DEV y CONFIRM)" if (h3_nm_d and h3_nm_c) else f"DEV={h3_nm_d}, CONFIRM={h3_nm_c}"
    )

    # H4: Online Frozen W_0 Perturbation RS-B (Fast Weights F_t + 3-factor E_t + MetaNet)
    nf_d = dev["contrasts"]["METAFE-3S_minus_METAFE-3S-NoFast"]
    nf_c = confirm["contrasts"]["METAFE-3S_minus_METAFE-3S-NoFast"]
    verdicts["H4_RS_B_vs_v2"] = (
        "PASA (DEV y CONFIRM)"
        if (win_neg(c_d["rs_b_frozen_w0_med_log"]) and win_neg(c_c["rs_b_frozen_w0_med_log"]))
        else "NO PASA"
    )
    verdicts["H4_RS_B_causal_NoFast"] = (
        "PASA (DEV y CONFIRM)"
        if (win_neg(nf_d["rs_b_frozen_w0_med_log"]) and win_neg(nf_c["rs_b_frozen_w0_med_log"]))
        else "NO PASA"
    )
    verdicts["H4_RS_B_causal_NoNM"] = (
        "PASA (DEV y CONFIRM)"
        if (win_neg(nm_d["rs_b_frozen_w0_med_log"]) and win_neg(nm_c["rs_b_frozen_w0_med_log"]))
        else "NO PASA"
    )

    # H5: Causal Cross-Branch Refuter (METAFE-3S - METAFE-3S-NoRefute)
    nr_d = dev["contrasts"]["METAFE-3S_minus_METAFE-3S-NoRefute"]
    nr_c = confirm["contrasts"]["METAFE-3S_minus_METAFE-3S-NoRefute"]
    verdicts["H5_causal_NoRefute_new_med_log"] = (
        "PASA (DEV y CONFIRM)"
        if (win_neg(nr_d["new_med_log"]) and win_neg(nr_c["new_med_log"]))
        else f"DEV={fmt_ci(nr_d['new_med_log'], True)}, CONFIRM={fmt_ci(nr_c['new_med_log'], True)}"
    )
    verdicts["H5_causal_NoRefute_new_brier"] = (
        "PASA (DEV y CONFIRM, IC<0)"
        if (nr_d["new_brier"][2] < 0 and nr_c["new_brier"][2] < 0)
        else f"DEV={fmt_ci(nr_d['new_brier'], True)}, CONFIRM={fmt_ci(nr_c['new_brier'], True)}"
    )

    # H6: Offline Continual Consolidation EWC vs NoEWC
    ewc_d = dev["continual_ewc"]["delta_forgetting_ewc_minus_noewc"]
    ewc_c = confirm["continual_ewc"]["delta_forgetting_ewc_minus_noewc"]
    verdicts["H6_EWC_reduces_forgetting"] = (
        "PASA (DEV y CONFIRM, IC sup < 0)"
        if (ewc_d[2] < 0.0 and ewc_c[2] < 0.0)
        else f"DEV={fmt_ci(ewc_d, True)}, CONFIRM={fmt_ci(ewc_c, True)}"
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
    verdicts["gate_zero_nan_zero_test_parity"] = (
        "PASA (0 NaN, 0 TEST, 32/32 SHA-256, v2 bit-exact)"
        if (
            dev["arms"][P]["nan"] == 0
            and confirm["arms"][P]["nan"] == 0
            and dev["test_queries_total"] == 0
            and confirm["test_queries_total"] == 0
            and (dev["v2_parity_max_diff"] or 0.0) == 0.0
            and (confirm["v2_parity_max_diff"] or 0.0) == 0.0
        )
        else "REVISAR"
    )
    for ph_name, ph in (("dev", dev), ("confirm", confirm)):
        mc = ph["arms"][P]["metacog"]
        ok = (
            mc["overall"]["fe_reduced_frac"][0] >= 0.99
            and mc["overall"]["auroc"][0] >= 0.90
            and mc["overall"]["brier"][0] <= 0.08
            and mc["new"]["auroc"][0] >= 0.78
            and mc["new"]["err_pred_corr"][0] >= 0.75
        )
        verdicts[f"gate_metacog_fe_{ph_name}"] = (
            f"PASA (FE={mc['overall']['fe_reduced_frac'][0]:.4f}, ov_AUROC={mc['overall']['auroc'][0]:.4f}, ov_Brier={mc['overall']['brier'][0]:.4f}, new_AUROC={mc['new']['auroc'][0]:.4f}, new_r={mc['new']['err_pred_corr'][0]:.4f})"
            if ok
            else f"REVISAR (FE={mc['overall']['fe_reduced_frac'][0]:.4f}, ov_AUROC={mc['overall']['auroc'][0]:.4f}, ov_Brier={mc['overall']['brier'][0]:.4f}, new_AUROC={mc['new']['auroc'][0]:.4f}, new_r={mc['new']['err_pred_corr'][0]:.4f})"
        )

    return verdicts


def render_tune_md(t):
    L = [
        f"# METAFE-3S — Diagnóstico TUNE ({t['seeds'][0]}–{t['seeds'][-1]}, n={t['n_seeds']})",
        f"test_sha256 vs BQ10: {t['test_sha256_match_bq']}, vs METAFE v2: {t['test_sha256_match_v2']}, v2 parity diff: {t['v2_parity_max_diff']}",
        "",
        "## Canónico + Sintético OOD + No-Estacionario (RS-A / RS-B)",
        "| Brazo | new_med_log | new_acc16 | known_acc16 | syn_ood_med | RS-A all | RS-A known_acc | RS-A ood_acc | RS-A reset | RS-B med | RS-B acc | E33/Auto | wall_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, x in t["arms"].items():
        sh = x["shift"]
        L.append(
            f"| {a} | {x['means']['new_med_log'][0]:+.4f} | {x['means']['new_acc16'][0]:.4f} | {x['means']['known_acc16'][0]:.4f} | {x['syn_ood']['syn_ood_med_log'][0]:+.4f} | {sh['rs_a_all_med_log'][0]:+.4f} | {sh['rs_a_known_acc16'][0]:.3f} | {sh['rs_a_ood_acc16'][0]:.3f} | {sh['rs_a_reset_rate'][0]:.3f} | {sh['rs_b_frozen_w0_med_log'][0]:+.4f} | {sh['rs_b_frozen_w0_acc16'][0]:.3f} | {x['e33_pass']}/{x['e33_auto_pass']} | {x['wall_s'][0]:.1f} |"
        )
    L += [
        "",
        "## Metacognición (overall / new / syn_ood)",
        "| Brazo | ov_AUROC | ov_Brier | ov_ECE | ov_r | new_AUROC | new_Brier | new_ECE | new_AURC | new_r | new_ρ | syn_AUROC | syn_Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, x in t["arms"].items():
        ov = x["metacog"]["overall"]
        nw = x["metacog"]["new"]
        sy = x["metacog"]["syn_ood"]
        L.append(
            f"| {a} | {ov['auroc'][0]:.4f} | {ov['brier'][0]:.4f} | {ov['ece'][0]:.4f} | {ov['err_pred_corr'][0]:.4f} | {nw['auroc'][0]:.4f} | {nw['brier'][0]:.4f} | {nw['ece'][0]:.4f} | {nw['aurc'][0]:.4f} | {nw['err_pred_corr'][0]:.4f} | {nw['err_pred_spearman'][0]:.4f} | {sy['auroc'][0]:.4f} | {sy['brier'][0]:.4f} |"
        )
    L += [
        "",
        "## Contrastes pareados clave en TUNE",
        "| Contraste | Δ new_med_log | Δ new_acc16 | Δ new_AUROC | Δ new_Brier | Δ RS-A all | Δ RS-B med | wins(new/rsA/rsB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, d in t["contrasts"].items():
        L.append(
            f"| `{k}` | {d['new_med_log'][0]:+.4f} | {d['new_acc16'][0]:+.4f} | {d['new_auroc'][0]:+.4f} | {d['new_brier'][0]:+.4f} | {d['rs_a_all_med_log'][0]:+.4f} | {d['rs_b_frozen_w0_med_log'][0]:+.4f} | {d['win_seeds_new_med_log']}/{d['win_seeds_rs_a_all']}/{d['win_seeds_rs_b']} |"
        )
    ce = t["continual_ewc"]
    L += [
        "",
        f"## Consolidación Continua Offline (EWC vs NoEWC): Δ forgetting = {ce['delta_forgetting_ewc_minus_noewc'][0]:+.4f} (EWC={ce['ewc_t1_forgetting_ce'][0]:.4f} vs NoEWC={ce['noewc_t1_forgetting_ce'][0]:.4f}, gana {ce['ewc_wins_forgetting_count']}/{t['n_seeds']}), Δ avg_acc = {ce['delta_avg_acc_ewc_minus_noewc'][0]:+.4f}",
    ]
    return "\n".join(L)


def render_md(out):
    L = [
        "# METAFE-3S — Resumen Estadístico Preregistrado (DEV y CONFIRM)",
        "",
        f"Primario congelado: `{out['frozen']['primary_arm']}`. Bootstrap pareado (10 000 réplicas, seed 12345, IC95 percentil, δ=0.05). `test_sha256` idéntico: DEV {out['dev']['test_sha256_match_bq']} (BQ10) / {out['dev']['test_sha256_match_v2']} (METAFE v2), CONFIRM {out['confirm']['test_sha256_match_bq']} (BQ10) / {out['confirm']['test_sha256_match_v2']} (METAFE v2); paridad `METAFE-v2` máx. diff = {out['dev']['v2_parity_max_diff']:.2e} (DEV) / {out['confirm']['v2_parity_max_diff']:.2e} (CONFIRM); `test_queries = 0`.",
        "",
    ]
    for ph_name in ("dev", "confirm"):
        ph = out[ph_name]
        L += [
            f"## {ph_name.upper()} ({ph['seeds'][0]}–{ph['seeds'][-1]}, n={ph['n_seeds']})",
            "",
            "### 1. Benchmark Canónico + OOD Sintético + Arranque en Frío E33",
            "| Arquitectura | new_med_log | new_acc16 | known_acc16 | syn_ood_med | Δ vs BQ10 new_med | E33 | E33 Auto | wall_s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for ref in BASELINES:
            b = ph["baselines"][ref]
            w = f"{b['wall_s'][0]:.1f}" if "wall_s" in b else "—"
            L.append(
                f"| {ref} | {fmt_ci(b['means']['new_med_log'], True)} | {b['means']['new_acc16'][0]:.4f} | {b['means']['known_acc16'][0]:.4f} | — | — | {b['e33_pass']}/16 | — | {w} |"
            )
        for a in ARMS_3S:
            x = ph["arms"][a]
            nm = f"**{a}**" if a == "METAFE-3S" else a
            L.append(
                f"| {nm} | {fmt_ci(x['means']['new_med_log'], True)} | {x['means']['new_acc16'][0]:.4f} | {x['means']['known_acc16'][0]:.4f} | {fmt_ci(x['syn_ood']['syn_ood_med_log'], True)} | {fmt_ci(x['vs_BQ10']['new_med_log'], True)} | {x['e33_pass']}/16 | {x['e33_auto_pass']}/16 ({x['e33_auto_h1'][0]:.3f}) | {x['wall_s'][0]:.1f} |"
            )
        L += [
            "",
            "### 2. Cambio de Régimen Online (RS-A abrupto y RS-B W_0 congelado + F_t)",
            "| Brazo | RS-A all_med | RS-A known_med (acc) | RS-A ood_med (acc) | RS-A reset | RS-B frozen W_0 med | RS-B acc16 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for a in ARMS_3S:
            sh = ph["arms"][a]["shift"]
            nm = f"**{a}**" if a == "METAFE-3S" else a
            L.append(
                f"| {nm} | {fmt_ci(sh['rs_a_all_med_log'], True)} | {sh['rs_a_known_med_log'][0]:+.4f} ({sh['rs_a_known_acc16'][0]:.3f}) | {sh['rs_a_ood_med_log'][0]:+.4f} ({sh['rs_a_ood_acc16'][0]:.3f}) | {sh['rs_a_reset_rate'][0]:.3f} | {fmt_ci(sh['rs_b_frozen_w0_med_log'], True)} | {sh['rs_b_frozen_w0_acc16'][0]:.4f} |"
            )
        L += [
            "",
            "### 3. Contrastes Pareados (Cabeza a Cabeza y Ablaciones Causales)",
            "| Contraste | Δ new_med_log | Δ new_acc16 | Δ new_AUROC | Δ new_Brier | Δ RS-A all | Δ RS-B med |",
            "|---|---|---|---|---|---|---|",
        ]
        for pk, d in ph["contrasts"].items():
            L.append(
                f"| `{pk}` | {fmt_ci(d['new_med_log'], True)} | {fmt_ci(d['new_acc16'], True)} | {fmt_ci(d['new_auroc'], True)} | {fmt_ci(d['new_brier'], True)} | {fmt_ci(d['rs_a_all_med_log'], True)} | {fmt_ci(d['rs_b_frozen_w0_med_log'], True)} |"
            )
        ce = ph["continual_ewc"]
        L += [
            "",
            "### 4. Consolidación Continua Offline (Módulo 5: EWC vs NoEWC)",
            f"- `EWC` olvido T1 (CE): {fmt_ci(ce['ewc_t1_forgetting_ce'])} vs `NoEWC`: {fmt_ci(ce['noewc_t1_forgetting_ce'])} → **Δ = {fmt_ci(ce['delta_forgetting_ewc_minus_noewc'], True)}** (EWC gana **{ce['ewc_wins_forgetting_count']}/16** semillas).",
            f"- `EWC` exactitud media T1–T3: {fmt_ci(ce['ewc_avg_acc'])} vs `NoEWC`: {fmt_ci(ce['noewc_avg_acc'])} → **Δ = {fmt_ci(ce['delta_avg_acc_ewc_minus_noewc'], True)}**.",
            "",
        ]
    L += ["## Veredictos Preregistrados", ""]
    for k, v in out["verdict"].items():
        L.append(f"- **{k}:** {v}")
    return "\n".join(L)


def main():
    if "--tune" in sys.argv:
        t = analyze_phase("tune", "tune", "bq_tune", expected_n=4)
        if "--json" in sys.argv:
            print(json.dumps(t, indent=1))
        else:
            print(render_tune_md(t))
        return

    frozen = json.load(open("docs/metafe3s/frozen.json"))
    dev = analyze_phase("dev", "dev", "bq_dev", expected_n=16)
    confirm = analyze_phase("confirm", "confirm", "bq_confirm", expected_n=16)
    out = {
        "frozen": frozen,
        "dev": dev,
        "confirm": confirm,
        "verdict": evaluate_verdicts(dev, confirm, frozen),
    }
    if "--md" in sys.argv:
        print(render_md(out))
    else:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
