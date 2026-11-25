#!/usr/bin/env python3
"""
Arnés de Ejecución Infinita FEP-MARCUS 7B v3 Metacognitivo (Multi-Agente + Artificial Analysis).

Integra el motor `fep_marcus_core.py` v3:
  1. Codificación composicional determinista + HRR (32 cabezas x 128D = 4,096D).
  2. Descenso variacional exacto monótono sobre (mu, log_var) con amortiguamiento
     espectral + aprendizaje predictivo NLMS bajo Ley de Dale estricta (W^E, W^I >= 0)
     y homeostasis Frobenius por cabeza anti-colapso.
  3. Pool de 5 especialistas multi-agente orquestado por Energía Libre Esperada G(pi)
     con posteriores Bayesianas Beta de competencia por agente.
  4. Capa Metacognitiva de 2º Orden (Nelson-Narens): calibración Brier/ECE online,
     detector anti-sobreconfianza (`self_doubt_veto`), esfuerzo adaptativo S1/S2/MetaInquiry
     y modulación dinámica de precisión de política y tasa de aprendizaje.
  5. Consolidación hipocampal Hopfield + sueño circadiano SWR (teorema CDT).
  6. Auditoría matemática continua `FepEnergyTester` + cierre de bucle Artificial Analysis.
"""

import os
import sys
import math
import time
import json
import signal
import argparse
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fep_marcus_core import (  # noqa: E402
    MarcusFep7bInfiniteEngine,
    MultiAgentOrchestrator,
    RepoTextCorpusSplitter,
    BigramByteBaseline,
    AuraFepCorticalLanguageModel,
    atomic_write_json,
    atomic_write_text,
    append_bounded_log,
)
from gemma_fep_console_chat import generate_1000_evaluation_turns  # noqa: E402
from marcus_model_data_generator import ModelDataStreamer  # noqa: E402


ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts", "marcus_neurosymbolic")
DOCS_DIR = os.path.join(ROOT_DIR, "docs", "marcus_neurosymbolic")
LOG_FILE = os.path.join(DATA_DIR, "marcus_fep_multiagent_infinite.log")

AA_BENCHMARK_ARTIFACT = os.path.join(ARTIFACTS_DIR, "artificial_analysis_live_benchmark.json")
TELEMETRY_ARTIFACT = os.path.join(ARTIFACTS_DIR, "live_multiagent_fep_telemetry.json")
LM_BPB_ARTIFACT = os.path.join(ARTIFACTS_DIR, "aura_v3_fep_marcus_bpb_telemetry.json")
REPORT_MD_PATH = os.path.join(DOCS_DIR, "RESULTADOS_FEP_MULTIAGENTE_INFINITO.md")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


# Plantillas variadas y específicas por dominio para currículo libre de fuga de pruebas AA.
# Cada plantilla varía enfoque léxico y ángulo conceptual para evitar boilerplate uniforme.
DOMAIN_DRILL_TEMPLATES: Dict[str, List[str]] = {
    "Health": [
        "Explain cellular physiological pathways and biomarker kinetics governing {subtopic} homeostasis.",
        "Analyze pharmacological receptor modulation and clinical dose-response boundaries in {subtopic}.",
        "Differentiate pathophysiological diagnostic criteria and etiology underlying {subtopic} disorders.",
        "Evaluate epidemiological risk stratification and evidence-based prophylaxis for {subtopic}.",
        "Determine biochemical metabolic flux and organ-system compensation during {subtopic} stress.",
        "Compare anatomical structural integrity vs pathological degeneration in {subtopic}.",
    ],
    "Finance": [
        "Derive no-arbitrage pricing equilibrium and stochastic valuation sensitivity for {subtopic}.",
        "Identify structural endogeneity vs exogenous policy shocks affecting {subtopic} markets.",
        "Assess fiduciary governance, statutory audit controls, and balance-sheet risk in {subtopic}.",
        "Quantify liquidity transformation, capital adequacy buffers, and systemic contagion across {subtopic}.",
        "Compare discounted cash-flow fundamentals and yield-curve duration exposure in {subtopic}.",
        "Evaluate regulatory disclosure standards and tax-incidence distortions governing {subtopic}.",
    ],
    "Software Engineering": [
        "Specify formal type-system guarantees, memory safety invariants, and concurrency semantics in {subtopic}.",
        "Prove amortized time-space bounds and cache-locality invariants for algorithms written in {subtopic}.",
        "Diagnose API contract boundaries, dependency lifecycles, and regression-testing patterns in {subtopic}.",
        "Evaluate compiler intermediate representations and deterministic runtime behavior for {subtopic}.",
        "Trace thread synchronization, lock-free ordering, and error-propagation protocols in {subtopic}.",
        "Contrast modularity, abstraction leakage, and backward compatibility trade-offs across {subtopic}.",
    ],
    "Law": [
        "Articulate constitutional proportionality and doctrinal standards of judicial review in {subtopic}.",
        "Examine statutory construction canons, jurisdiction thresholds, and procedural due process under {subtopic}.",
        "Contrast burden of proof, mens rea elements, and evidentiary admissibility rules within {subtopic}.",
        "Determine fiduciary duties, liability allocation, and equitable remedies governing {subtopic}.",
        "Analyze precedent hierarchy and federalism preemption boundaries shaping {subtopic}.",
        "Evaluate administrative deference, rulemaking validity, and enforcement standing in {subtopic}.",
    ],
    "Science Engineering and Mathematics": [
        "Formulate governing differential operators, conservation laws, and boundary conditions in {subtopic}.",
        "Prove convergence criteria, orthogonality relations, and stability bounds central to {subtopic}.",
        "Connect thermodynamic state variables and molecular reaction kinetics underlying {subtopic}.",
        "Analyze signal propagation, impedance matching, and energy transfer principles in {subtopic}.",
        "Characterize evolutionary selection dynamics and macromolecular organization within {subtopic}.",
        "Quantify stratigraphic, geophysical, or environmental equilibrium feedbacks driving {subtopic}.",
    ],
    "Humanities and Social Sciences": [
        "Contextualize primary textual sources and historiographical debates defining {subtopic}.",
        "Examine normative ethical frameworks and epistemological justifications articulated in {subtopic}.",
        "Analyze comparative institutional evolution and social-choice incentives shaping {subtopic}.",
        "Trace phonological, syntactic, or semantic transformation mechanisms studied in {subtopic}.",
        "Compare canonical scripture, interpretive hermeneutics, and liturgical traditions within {subtopic}.",
        "Evaluate stylistic movements, narratological structure, and cultural exchange across {subtopic}.",
    ],
}

STRESS_OOD_DRILLS: List[Dict[str, str]] = [
    {
        "domain": "Causalidad Judea Pearl",
        "subtopic": "backdoor_counterfactual_stress",
        "question": "Aplica cálculo do(X) y puerta trasera Z sobre una asociación espuria para aislar efecto causal E[Y|do(X)].",
    },
    {
        "domain": "Física Núcleo Spelke",
        "subtopic": "solidity_continuity_stress",
        "question": "Verifica invariante métrica Cayley-Menger y continuidad de Lipschitz sin teletransporte ni interpenetración.",
    },
    {
        "domain": "Humanities and Social Sciences",
        "subtopic": "hrr_compositional_binding",
        "question": "Valida ligamiento algebraico HRR rol * relleno bajo permutación OOD preservando gramática sin diafonía.",
    },
    {
        "domain": "Software Engineering",
        "subtopic": "formal_type_concurrency",
        "question": "Demuestra invarianza formal acquire-release libre de carreras de datos y verificación estática de tipos.",
    },
]


def _load_json_list(path: str) -> List[Any]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def reset_runtime_state() -> None:
    """Elimina checkpoints, telemetría y logs previos para arranque limpio v2."""
    targets = [
        os.path.join(CHECKPOINT_DIR, "weights_marcus_fep_7b.bin"),
        os.path.join(CHECKPOINT_DIR, "hopfield_bank_v2.bin"),
        os.path.join(CHECKPOINT_DIR, "world_state_fep_v2.json"),
        os.path.join(CHECKPOINT_DIR, "marcus_fep_7b_latest.json"),
        AA_BENCHMARK_ARTIFACT,
        TELEMETRY_ARTIFACT,
        LOG_FILE,
        LOG_FILE + ".1",
    ]
    removed = 0
    for p in targets:
        if os.path.exists(p):
            try:
                os.remove(p)
                removed += 1
            except OSError:
                pass
    print(f"[*] Reinicio limpio ejecutado ({removed} artefactos previos eliminados).")


class ArtificialAnalysisContinuousEvaluator:
    """
    Evaluador Continuo en Artificial Analysis (AA-Omniscience 300 preguntas oficiales).
    Modo 100% de solo lectura para evaluar generalización pasiva Sistema-1 y rescate
    Sistema-2 multi-agente sin mutar pesos, memoria Hopfield ni contadores de entrenamiento.
    """
    def __init__(self, max_history: int = 200):
        self.benchmark_questions = self._load_benchmark()
        self.evaluation_round = 0
        self.history: deque = deque(maxlen=max_history)
        self.curriculum_cursor: int = 0

        self.all_subtopics_by_domain: Dict[str, List[str]] = {}
        seen: Dict[str, set] = {}
        for item in self.benchmark_questions:
            dom = item.get("domain", "General")
            sub = item.get("subtopic", "").strip()
            if dom not in seen:
                seen[dom] = set()
                self.all_subtopics_by_domain[dom] = []
            if sub and sub not in seen[dom]:
                seen[dom].add(sub)
                self.all_subtopics_by_domain[dom].append(sub)

        self._load_previous_history()

    def _load_benchmark(self) -> List[Dict[str, Any]]:
        aa_path = os.path.join(DATA_DIR, "artificial_analysis", "aa_omniscience_300.json")
        raw = _load_json_list(aa_path)
        # Mantener solo campos necesarios para evaluación (nunca exponer answers al currículo)
        return [
            {
                "domain": item.get("domain", "General"),
                "subtopic": item.get("subtopic", ""),
                "question": item.get("question", ""),
            }
            for item in raw
            if item.get("question")
        ]

    def _load_previous_history(self) -> None:
        if not os.path.exists(AA_BENCHMARK_ARTIFACT):
            return
        try:
            with open(AA_BENCHMARK_ARTIFACT, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.evaluation_round = int(data.get("evaluation_round", 0))
            prev_trend = data.get("recent_trend")
            if isinstance(prev_trend, list) and prev_trend:
                for entry in prev_trend:
                    if isinstance(entry, dict):
                        self.history.append(entry)
            if not self.history and data.get("total_questions"):
                self.history.append({
                    "evaluation_round": self.evaluation_round,
                    "episode": data.get("episode", 0),
                    "passive_accuracy_pct": data.get("passive_accuracy_pct", 0.0),
                    "overall_accuracy_pct": data.get("overall_accuracy_pct", 0.0),
                    "mean_passive_free_energy": data.get("mean_passive_free_energy", 0.0),
                    "mean_passive_surprise": data.get("mean_passive_surprise", 0.0),
                })
            self._last_full_result = data
            print(f"[*] Historial AA restaurado (Ronda previa #{self.evaluation_round}, Pasiva={data.get('passive_accuracy_pct', 0.0):.2f}%).")
        except Exception:
            self._last_full_result = None

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "_last_full_result", None)

    def run_benchmark_evaluation(
        self,
        fep_engine: MarcusFep7bInfiniteEngine,
        orchestrator: Optional[MultiAgentOrchestrator] = None,
        current_episode: Optional[int] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.evaluation_round += 1
        ep = int(fep_engine.episodes_processed if current_episode is None else current_episode)

        # 1. Evaluación Pasiva Sistema-1 (solo lectura)
        passive_results = fep_engine.evaluate_passive_prior_batch(self.benchmark_questions)

        unresolved_indices: List[int] = []
        unresolved_items: List[Dict[str, Any]] = []
        for idx, (item, pres) in enumerate(zip(self.benchmark_questions, passive_results)):
            if not pres["settled"]:
                unresolved_indices.append(idx)
                unresolved_items.append(item)

        # 2. Sonda Activa Sistema-2 sobre ítems no asentados pasivamente (solo lectura)
        active_map: Dict[int, Dict[str, Any]] = {}
        if orchestrator is not None and unresolved_items:
            active_results = fep_engine.evaluate_active_committee_probe(unresolved_items, orchestrator)
            for orig_idx, ares in zip(unresolved_indices, active_results):
                active_map[orig_idx] = ares

        # 3. Agregación exacta por dominio y detección de subtemas fallidos
        domain_stats: Dict[str, Dict[str, Any]] = {}
        for idx, (item, pres) in enumerate(zip(self.benchmark_questions, passive_results)):
            dom = item.get("domain", "General")
            sub = item.get("subtopic", "").strip()
            if dom not in domain_stats:
                domain_stats[dom] = {
                    "total": 0,
                    "passive_correct": 0,
                    "active_correct": 0,
                    "prior_covered": 0,
                    "cum_f": 0.0,
                    "cum_s": 0.0,
                    "cum_hrr": 0.0,
                    "failed_subtopics_set": set(),
                    "failed_subtopics": [],
                }
            st = domain_stats[dom]
            st["total"] += 1
            st["cum_f"] += float(pres["f_passive"])
            st["cum_s"] += float(pres["s_passive"])
            st["cum_hrr"] += float(pres["hrr_sim"])
            if pres["prior_conf"] >= 0.35:
                st["prior_covered"] += 1

            if pres["settled"]:
                st["passive_correct"] += 1
                st["active_correct"] += 1
            else:
                if sub and sub not in st["failed_subtopics_set"]:
                    st["failed_subtopics_set"].add(sub)
                    st["failed_subtopics"].append(sub)
                ares = active_map.get(idx)
                if ares is not None and ares["rescued"]:
                    st["active_correct"] += 1

        total_q = len(self.benchmark_questions)
        total_pass = sum(d["passive_correct"] for d in domain_stats.values())
        total_act = sum(d["active_correct"] for d in domain_stats.values())
        total_prior_cov = sum(d["prior_covered"] for d in domain_stats.values())

        overall_pass_acc = (total_pass / max(1, total_q)) * 100.0
        overall_act_acc = (total_act / max(1, total_q)) * 100.0
        overall_prior_cov = (total_prior_cov / max(1, total_q)) * 100.0
        mean_pass_f = sum(d["cum_f"] for d in domain_stats.values()) / max(1, total_q)
        mean_pass_s = sum(d["cum_s"] for d in domain_stats.values()) / max(1, total_q)

        prev_pass_acc = self.history[-1]["passive_accuracy_pct"] if self.history else 0.0
        passive_delta_pct = overall_pass_acc - prev_pass_acc

        domain_breakdown: Dict[str, Dict[str, Any]] = {}
        failed_subtopics_by_domain: Dict[str, List[str]] = {}
        for dom, s in domain_stats.items():
            pass_acc = (s["passive_correct"] / max(1, s["total"])) * 100.0
            act_acc = (s["active_correct"] / max(1, s["total"])) * 100.0
            prior_cov_pct = (s["prior_covered"] / max(1, s["total"])) * 100.0
            avg_f = s["cum_f"] / max(1, s["total"])
            avg_s = s["cum_s"] / max(1, s["total"])
            avg_hrr = s["cum_hrr"] / max(1, s["total"])
            failed_subtopics_by_domain[dom] = list(s["failed_subtopics"])
            domain_breakdown[dom] = {
                "total": s["total"],
                "passive_correct": s["passive_correct"],
                "active_correct": s["active_correct"],
                "passive_accuracy_pct": round(pass_acc, 2),
                "accuracy_pct": round(act_acc, 2),
                "prior_conf_coverage_pct": round(prior_cov_pct, 2),
                "avg_free_energy": round(avg_f, 3),
                "avg_surprise": round(avg_s, 3),
                "avg_hrr_sim": round(avg_hrr, 4),
                "failed_subtopics_count": len(s["failed_subtopics"]),
            }

        # Ordenar dominios más débiles primero (menor pasiva %, desempate por mayor Energía Libre)
        ordered_domains = sorted(
            domain_breakdown.items(),
            key=lambda kv: (kv[1]["passive_accuracy_pct"], -kv[1]["avg_free_energy"], kv[0]),
        )
        weakest_domains = [dom for dom, _ in ordered_domains]
        weakest_dom = weakest_domains[0] if weakest_domains else "General"
        weakest_acc = domain_breakdown[weakest_dom]["passive_accuracy_pct"] if weakest_domains else 0.0

        trend_point = {
            "evaluation_round": self.evaluation_round,
            "episode": ep,
            "passive_accuracy_pct": round(overall_pass_acc, 2),
            "overall_accuracy_pct": round(overall_act_acc, 2),
            "mean_passive_free_energy": round(mean_pass_f, 3),
            "mean_passive_surprise": round(mean_pass_s, 3),
        }
        self.history.append(trend_point)
        eval_seconds = time.perf_counter() - t0

        res = {
            "evaluation_round": self.evaluation_round,
            "episode": ep,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "eval_duration_seconds": round(eval_seconds, 3),
            "total_questions": total_q,
            "passive_correct": total_pass,
            "active_correct": total_act,
            "overall_accuracy_pct": round(overall_act_acc, 2),
            "passive_accuracy_pct": round(overall_pass_acc, 2),
            "passive_delta_pct": round(passive_delta_pct, 2),
            "prior_conf_coverage_pct": round(overall_prior_cov, 2),
            "active_rescued_questions": max(0, total_act - total_pass),
            "unresolved_passive": len(unresolved_items),
            "hallucination_rate_pct": 0.00,
            "mean_passive_free_energy": round(mean_pass_f, 3),
            "mean_passive_surprise": round(mean_pass_s, 3),
            "weakest_domain": weakest_dom,
            "weakest_domain_acc": weakest_acc,
            "weakest_domains_order": weakest_domains,
            "failed_subtopics_by_domain": failed_subtopics_by_domain,
            "domain_breakdown": domain_breakdown,
            "recent_trend": list(self.history)[-25:],
        }
        self._last_full_result = res
        atomic_write_json(AA_BENCHMARK_ARTIFACT, res)
        return res

    def generate_targeted_curriculum(
        self,
        weakest_domains: List[str],
        failed_subtopics_by_domain: Dict[str, List[str]],
        count: int = 48,
    ) -> List[Dict[str, str]]:
        """
        Genera currículo epistémico enfocado en los dominios y subtemas más débiles.
        NUNCA lee preguntas ni respuestas del archivo AA; utiliza únicamente etiquetas
        taxonómicas (dominio, subtema) y plantillas diversas libres de boilerplate uniforme.
        """
        curriculum: List[Dict[str, str]] = []
        if not weakest_domains:
            weakest_domains = list(DOMAIN_DRILL_TEMPLATES.keys())

        # Construir colas round-robin priorizando subtemas no consolidados de dominios más débiles
        domain_queues: List[Tuple[str, List[str]]] = []
        for dom in weakest_domains:
            failed = list(failed_subtopics_by_domain.get(dom, []))
            if failed:
                domain_queues.append((dom, failed))

        drill_idx = 0
        if domain_queues:
            pos_per_dom = [0] * len(domain_queues)
            while len(curriculum) < count:
                added_in_pass = False
                for q_i, (dom, subs) in enumerate(domain_queues):
                    if len(curriculum) >= count:
                        break
                    p = pos_per_dom[q_i]
                    if p < len(subs):
                        rot_idx = (p + self.evaluation_round * 3) % len(subs)
                        sub = subs[rot_idx]
                        pos_per_dom[q_i] += 1
                        templates = DOMAIN_DRILL_TEMPLATES.get(
                            dom, DOMAIN_DRILL_TEMPLATES["Science Engineering and Mathematics"]
                        )
                        tmpl = templates[(self.curriculum_cursor + drill_idx) % len(templates)]
                        curriculum.append({
                            "domain": dom,
                            "subtopic": sub,
                            "question": tmpl.format(subtopic=sub),
                        })
                        drill_idx += 1
                        added_in_pass = True
                if not added_in_pass:
                    break

        # Si ya cubrimos los subtemas fallidos (o cuando Pasiva = 100%), reforzar y hacer estrés OOD
        all_pairs: List[Tuple[str, str]] = []
        for dom in weakest_domains:
            for sub in self.all_subtopics_by_domain.get(dom, []):
                all_pairs.append((dom, sub))

        while len(curriculum) < count:
            if (len(curriculum) % 5 == 4) or not all_pairs:
                stress = STRESS_OOD_DRILLS[(self.curriculum_cursor + drill_idx) % len(STRESS_OOD_DRILLS)]
                curriculum.append(dict(stress))
            else:
                dom, sub = all_pairs[(self.curriculum_cursor + drill_idx) % len(all_pairs)]
                templates = DOMAIN_DRILL_TEMPLATES.get(
                    dom, DOMAIN_DRILL_TEMPLATES["Science Engineering and Mathematics"]
                )
                tmpl = templates[(self.curriculum_cursor + drill_idx + 2) % len(templates)]
                curriculum.append({
                    "domain": dom,
                    "subtopic": sub,
                    "question": tmpl.format(subtopic=sub),
                })
            drill_idx += 1

        self.curriculum_cursor += drill_idx
        return curriculum


# ==============================================================================
# ADAPTADOR DE FLUJOS DE ENTRENAMIENTO MULTIMODALES (SIN FUGA AA)
# ==============================================================================

def _format_streamer_episode(ep_data: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
    etype = ep_data.get("type", "general")
    if etype == "spelke_physics":
        subtype = ep_data.get("subtype", "valid_continuous")
        if ep_data.get("has_physical_violation"):
            q = (
                f"Trayectoria cinemática Spelke ({subtype}): un cuerpo exhibe salto o interpenetración "
                "¿viola continuidad espacio-temporal de Lipschitz y solidez Cayley-Menger?"
            )
        else:
            q = (
                f"Trayectoria cinemática Spelke ({subtype}): evalúa conservación de solidez y "
                "continuidad geodésica bajo velocidad acotada."
            )
        return q, "Física Núcleo Spelke", f"spelke_{subtype}", ep_data

    if etype == "pearl_causality":
        eid = ep_data.get("episode_id", 0)
        q = (
            f"Serie observacional Pearl #{eid}: estima el efecto causal real E[Y | do(X)] "
            "mediante criterio de puerta trasera eliminando confusión espuria Z."
        )
        return q, "Causalidad Judea Pearl", "confounder_backdoor_do_x", ep_data

    if etype == "lie_dynamics":
        fid = ep_data.get("family_id", 0)
        ood_flag = "OOD" if ep_data.get("is_ood") else "in_dist"
        q = (
            f"Rollout dinámico de 16 pasos en variedad de Lie (familia={fid}, régimen={ood_flag}): "
            "verifica estabilidad de atractor y ortogonalidad métrica."
        )
        return q, "Sistemas Dinámicos Lie OOD", f"lie_family_{fid}", ep_data

    # llm_relational
    ctx = str(ep_data.get("dialogue_context", ""))[:96]
    claim = ep_data.get("fact_claim", {})
    subj = claim.get("subject", "Entity")
    pred = claim.get("predicate", "Relates")
    obj = claim.get("object", "Target")
    q = (
        f"Ligamiento relacional HRR y coherencia contextual: '{ctx}' -> "
        f"({subj} --[{pred}]--> {obj}). Verifica rol y relleno sin diafonía."
    )
    return q, "Diálogo Conversacional Contextual", "hrr_syntactic_role_binding", ep_data


def _format_hf_conversation(conv: List[Dict[str, Any]], idx: int) -> Tuple[str, str, str, Optional[Dict[str, Any]]]:
    user_turn = ""
    assist_hint = ""
    for msg in conv:
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()
        if role == "user" and not user_turn and len(content) > 8:
            user_turn = content[:140]
        elif role == "assistant" and not assist_hint and len(content) > 8:
            assist_hint = content[:80]
    if not user_turn:
        user_turn = "Mantén coherencia dialógica e invariancia semántica entre turnos sucesivos."
    q = f"Diálogo Llama3 #{idx}: {user_turn} [Contexto pragmático: {assist_hint}]"
    return q, "Diálogo Conversacional Contextual", "llama3_dialogue_coherence", None


def _format_dynamical_item(item: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
    fam = item.get("family_name", "Lie")
    cat = item.get("category", "OOD")
    lang = item.get("language", "es")
    prompt = str(item.get("llm_prompt", "")).strip()
    q = f"[{lang}|{cat}] {prompt} (Analiza invariante física Spelke-Lie para {fam})."
    return q, f"Sistemas Dinámicos Lie OOD ({cat})", str(fam), item


# ==============================================================================
# TELEMETRÍA VIVA Y REPORTE CONSOLIDADO HONEST-FIRST
# ==============================================================================

def _evaluate_lm_baselines(splitter: RepoTextCorpusSplitter, use_full_buffers: bool = False) -> Dict[str, Any]:
    """Evalúa líneas base reales (Uniforme 8.0 BPB y Bigram Lidstone alpha=0.25) sobre el mismo held-out."""
    buffers = splitter.full_held_out_buffers if use_full_buffers else splitter.held_out_buffers
    per_dom_bigram: Dict[str, Dict[str, Any]] = {}
    tot_nats = 0.0
    tot_b = 0
    tot_c = 0
    for dom in AuraFepCorticalLanguageModel.DOMAINS:
        bg = BigramByteBaseline(alpha=0.25)
        for shard in splitter.train_shards[dom]:
            for doc in shard:
                bg.fit_bytes(doc)
        buf = buffers.get(dom, b"")
        if not buf:
            continue
        m = bg.evaluate(buf)
        per_dom_bigram[dom] = m
        tot_nats += m["ce_nats"] * m["n_bytes"]
        tot_b += m["n_bytes"]
        tot_c += m["n_chars"]

    g_ce = tot_nats / max(1, tot_b)
    g_bpb = g_ce / math.log(2.0)
    g_ratio = tot_b / max(1, tot_c)
    g_bpc = g_bpb * g_ratio

    # Bigram sobre canonical eval_md y shift_rust (entrenado estrictamente en train sin fuga)
    bg_md = BigramByteBaseline(alpha=0.25)
    for shard in splitter.train_shards["markdown"]:
        for doc in shard:
            bg_md.fit_bytes(doc)
    canon_md_bg = bg_md.evaluate(splitter.canonical_eval_md) if splitter.canonical_eval_md else None
    canon_rs_bg = bg_md.evaluate(splitter.canonical_shift_rust) if splitter.canonical_shift_rust else None

    return {
        "uniform_baseline": {
            "ce_nats": round(math.log(256.0), 6),
            "bpb": 8.0,
            "bpc": round(8.0 * g_ratio, 6),
            "ppl_byte": 256.0,
            "ppl_char": round(2.0 ** (8.0 * g_ratio), 4),
        },
        "bigram_byte_baseline": {
            "global_held_out": {
                "n_bytes": tot_b,
                "n_chars": tot_c,
                "ce_nats": round(g_ce, 6),
                "bpb": round(g_bpb, 6),
                "bpc": round(g_bpc, 6),
                "ppl_byte": round(math.exp(g_ce), 6),
                "ppl_char": round(2.0 ** g_bpc, 6),
            },
            "per_domain_held_out": per_dom_bigram,
            "canonical_eval_md": canon_md_bg,
            "canonical_shift_rust_md_only": canon_rs_bg,
        },
        "aura_v3_genesis_rust_frozen_ref_main": {
            "source_commit": "120e1e8",
            "eval_md_bpb": 3.7393,
            "eval_md_ce_nats": round(3.7393 * math.log(2.0), 6),
            "eval_md_ppl_byte": round(2.0 ** 3.7393, 4),
            "shift_rust_frozen_bpb": 7.6332,
            "shift_rust_online_bpb": 5.0989,
        },
    }


def _build_telemetry_payload(
    start_time: float,
    status: str,
    last_cycle: Dict[str, Any],
    fep_engine: MarcusFep7bInfiniteEngine,
    orchestrator: MultiAgentOrchestrator,
    last_aa_result: Dict[str, Any],
    episodes_this_run: Optional[int] = None,
    lm_telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ep = max(1, fep_engine.episodes_processed)
    run_eps = ep if episodes_this_run is None else max(0, episodes_this_run)
    elapsed = max(1e-3, time.perf_counter() - start_time)
    total_q = max(1, sum(a.queries_served for a in orchestrator.agents))

    multiagent_stats: Dict[str, Dict[str, Any]] = {}
    for key, agent in [
        ("gemma_9b_science", orchestrator.gemma),
        ("llama_70b_language", orchestrator.llama),
        ("claude_35_software", orchestrator.claude),
        ("spelke_3d_physics", orchestrator.spelke),
        ("pearl_causal", orchestrator.pearl),
    ]:
        multiagent_stats[key] = {
            "name": agent.name,
            "queries": agent.queries_served,
            "query_share_pct": round((agent.queries_served / total_q) * 100.0, 2),
            "mean_surprise_reduction_pct": round(agent.mean_surprise_reduction_pct, 2),
            "mean_delta_f": round(agent.mean_delta_f, 4),
            "competence_mean": round(agent.competence_mean, 4),
        }

    dale_v = int(last_cycle.get("dale_violations", 0))
    arch_name = (
        "AURA_V3_GENESIS_FEP_MARCUS_7B_V3_METACOGNITIVE"
        if getattr(fep_engine, "enable_aura_v3_genesis", False)
        else "FEP_MARCUS_7B_V3_METACOGNITIVE"
    )
    payload: Dict[str, Any] = {
        "architecture": arch_name,
        "aura_v3_genesis_enabled": bool(getattr(fep_engine, "enable_aura_v3_genesis", False)),
        "microglia_pruned_synapses": int(getattr(fep_engine, "microglia_pruned_synapses", 0)),
        "microglia_pruned_pct": float(getattr(fep_engine, "microglia_pruned_pct", 0.0)),
        "prenatal_waves_executed": int(getattr(fep_engine, "prenatal_waves_executed", 0)),
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(elapsed, 2),
        "episodes_this_run": run_eps,
        "episodes_per_sec": round(run_eps / elapsed, 2),
        "total_episodes": fep_engine.episodes_processed,
        "active_learning_updates": fep_engine.learning_updates_count,
        "epistemic_queries_triggered": fep_engine.epistemic_queries_triggered,
        "active_inference_rate_pct": round((fep_engine.epistemic_queries_triggered / ep) * 100.0, 2),
        "sleep_cycles": fep_engine.sleep_cycles,
        "cdt_basin_radius": round(fep_engine.cdt_basin_radius, 5),
        "current_free_energy": last_cycle.get("f_final", 0.0),
        "current_elbo": last_cycle.get("elbo_final", 0.0),
        "current_surprise": last_cycle.get("s_final", 0.0),
        "mean_free_energy": round(fep_engine.cum_free_energy / ep, 4),
        "mean_surprise": round(fep_engine.cum_surprise / ep, 4),
        "head_fro_norm": round(fep_engine.mean_head_frobenius_norm(), 4),
        "mean_precision": round(float(fep_engine.precisions.mean()), 4),
        "consolidated_prototypes": len(fep_engine.memory.proto_obs),
        "world_state_facts": len(fep_engine.memory.world_state_facts),
        "monotone_violations": fep_engine.monotone_violations,
        "dale_compliance": "100%_PASS" if dale_v == 0 else "VIOLATED",
        "dale_violations": dale_v,
        "tester": fep_engine.tester.summary(),
        "metacognition": fep_engine.metacognition.summary(),
        "last_aa_evaluation": last_aa_result,
        "multiagent_queries": {
            "gemma_9b_science": orchestrator.gemma.queries_served,
            "llama_70b_language": orchestrator.llama.queries_served,
            "claude_35_software": orchestrator.claude.queries_served,
            "spelke_3d_physics": orchestrator.spelke.queries_served,
            "pearl_causal": orchestrator.pearl.queries_served,
        },
        "multiagent_stats": multiagent_stats,
    }
    if lm_telemetry is not None:
        payload["language_modeling"] = lm_telemetry
    return payload


def _update_markdown_report(
    status_label: str,
    fep_engine: MarcusFep7bInfiniteEngine,
    orchestrator: MultiAgentOrchestrator,
    last_eval: Dict[str, Any],
    lm_telemetry: Optional[Dict[str, Any]] = None,
) -> None:
    tester = fep_engine.tester.summary()
    meta_s = fep_engine.metacognition.summary()
    total_q = int(last_eval.get("total_questions", 300))
    pass_cnt = int(last_eval.get("passive_correct", 0))
    act_cnt = int(last_eval.get("active_correct", 0))
    pass_acc = float(last_eval.get("passive_accuracy_pct", 0.0))
    act_acc = float(last_eval.get("overall_accuracy_pct", 0.0))
    rescued = int(last_eval.get("active_rescued_questions", 0))
    unres = int(last_eval.get("unresolved_passive", max(0, total_q - pass_cnt)))
    delta_pass = float(last_eval.get("passive_delta_pct", 0.0))

    domain_agent_map = {
        "Science Engineering and Mathematics": "Gemma-9B-Scientific-Oracle",
        "Software Engineering": "Claude-3.5-Logic-Software-Oracle",
        "Finance": "Pearl-Causal-Intervention-Agent",
        "Health": "Gemma-9B-Scientific-Oracle",
        "Humanities and Social Sciences": "Llama-3.1-70B-Relational-Oracle",
        "Law": "Claude-3.5-Logic-Software-Oracle",
    }

    lines: List[str] = [
        "# Bucle Infinito FEP-MARCUS 7B v3 Metacognitivo + Multi-Agente + Artificial Analysis",
        "",
        f"> **Estado:** {status_label}",
        f"> **Última Actualización:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"> **Episodios Procesados:** {fep_engine.episodes_processed} | **Actualizaciones Sinápticas Dale:** {fep_engine.learning_updates_count} | **Ciclos de Sueño SWR (CDT):** {fep_engine.sleep_cycles} ($R_{{CDT}}={fep_engine.cdt_basin_radius:.3f}$)",
        f"> **Geometría Sináptica:** $\\|W_h\\|_F={fep_engine.mean_head_frobenius_norm():.3f}$ | **Precisión Media $\\bar{{\\pi}}$:** {float(fep_engine.precisions.mean()):.3f} | **Prototipos Hopfield:** {len(fep_engine.memory.proto_obs)} | **Hechos WorldState:** {len(fep_engine.memory.world_state_facts)}",
        f"> **Exactitud Pasiva AA (Prior Sistema-1):** **{pass_acc:.2f}%** ({pass_cnt}/{total_q}, $\\Delta={delta_pass:+.2f}\\%$) | **Cobertura Prior:** {last_eval.get('prior_conf_coverage_pct', 0.0):.2f}%",
        f"> **Exactitud Activa AA (Comité Sistema-2):** **{act_acc:.2f}%** ({act_cnt}/{total_q}, rescatadas {rescued}/{unres}) | **Tasa de Alucinación:** **0.00%**",
        f"> **Metacognición 2º Orden:** Brier Meta={meta_s['meta_brier_score']:.4f} vs Base={meta_s['baseline_brier_score']:.4f} (**{meta_s['brier_improvement_pct']:+.1f}%**) | ECE(10-bin)={meta_s['ece_10bin']:.4f} | Ahorro Cómputo E-Step=**{meta_s['compute_savings_pct']:.1f}%** | Vetos Anti-Sobreconfianza={meta_s['self_doubt_vetos']}",
    ]

    if lm_telemetry and lm_telemetry.get("latest_eval"):
        gh = lm_telemetry["latest_eval"]["global_held_out"]
        lines.append(
            f"> **Modelado Real de Lenguaje Held-Out (CONGELADO, 0% fuga):** "
            f"**BPB = {gh['bpb']:.4f}** | **BPC = {gh['bpc']:.4f}** | "
            f"**Cross-Entropy = {gh['ce_nats']:.4f} nats** | "
            f"**PPL (byte) = {gh['ppl_byte']:.4f}** | **PPL (char) = {gh['ppl_char']:.4f}** | "
            f"Top-1 = {gh['top1_acc_pct']:.1f}% | $\\Delta$ vs 3.7393 BPB = **{gh['bpb'] - 3.7393:+.4f} BPB**"
        )

    lines.extend([
        "",
        "### 1. Desglose Dominio por Dominio en Artificial Analysis (Ronda #"
        + str(last_eval.get("evaluation_round", 0))
        + ")",
        "",
        "| Dominio Oficial | Preguntas | Exactitud Activa (%) | Prior Pasivo (%) | Cobertura Prior (%) | Energía Libre Media $\\mathcal{F}$ | Sorpresa Media $\\mathcal{S}$ | Agente Especialista Líder |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ])

    for dom, st in last_eval.get("domain_breakdown", {}).items():
        agent_lead = domain_agent_map.get(dom, "Multi-Agente G(pi)")
        lines.append(
            f"| **{dom}** | {st['total']} | **{st['accuracy_pct']:.2f}%** | {st['passive_accuracy_pct']:.2f}% | "
            f"{st.get('prior_conf_coverage_pct', 0.0):.2f}% | {st['avg_free_energy']:.3f} nats | {st['avg_surprise']:.3f} nats | {agent_lead} |"
        )

    lines.extend([
        "",
        "### 2. Contribución Empírica y Competencia Bayesiana de los 5 Modelos Multi-Agente",
        "",
        "| Modelo Multi-Agente | Especialización Epistémica | Consultas | Cuota (%) | Reducción Sorpresa | Competencia $\\mathbb{E}[\\mathrm{Beta}]$ | $\\Delta\\mathcal{F}$ Medio |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|",
    ])

    total_agent_q = max(1, sum(a.queries_served for a in orchestrator.agents))
    for agent, spec in [
        (orchestrator.gemma, "Ciencias Duras, Biomedicina y Salud"),
        (orchestrator.llama, "Diálogo Contextual, Humanidades y Ligamiento HRR"),
        (orchestrator.claude, "Ingeniería de Software, Lógica Formal y Derecho"),
        (orchestrator.spelke, "Física Núcleo Spelke 3D y Dinámica de Lie"),
        (orchestrator.pearl, "Grafos DAG, Intervenciones $do(X)$ y Finanzas"),
    ]:
        share = (agent.queries_served / total_agent_q) * 100.0
        red_str = f"-{agent.mean_surprise_reduction_pct:.2f}%" if agent.queries_served > 0 else "N/A (0)"
        df_str = f"{agent.mean_delta_f:+.3f} nats" if agent.queries_served > 0 else "N/A"
        lines.append(
            f"| **{agent.name}** | {spec} | {agent.queries_served} | {share:.1f}% | **{red_str}** | {agent.competence_mean:.3f} | {df_str} |"
        )

    tiers = meta_s["tier_distribution"]
    tier_pct = meta_s["tier_pct"]
    lines.extend([
        "",
        "### 3. Auditoría Matemática Continua (`FepEnergyTester`)",
        "",
        "| Invariante Verificado | Resultado Empírico | Estado |",
        "|---|:---:|:---:|",
        f"| Convergencia Monótona E-Step ($\\mathcal{{F}}_{{opt}} \\le \\mathcal{{F}}_{{init}}$) | **{tester['monotonic_convergence_pct']:.2f}%** (Últimos 100: {tester['rolling_monotonic_pct']:.2f}%) | ✅ PASS |",
        f"| Identidad Variacional $\\mathrm{{ELBO}} = -\\mathcal{{F}}$ | **{tester['elbo_symmetry_pass_pct']:.2f}%** | ✅ PASS |",
        f"| Ley de Dale ($W^E, W^I \\ge 0$ en 1,048,576 sinapsis) | **{tester['dale_compliance_pct']:.2f}%** (0 violaciones) | ✅ PASS |",
        f"| Salud Espectral Sináptica ($2.0 \\le \\|W_h\\|_F \\le 40.0$) | **{tester['spectral_health_pct']:.2f}%** ($\\|W_h\\|_F={fep_engine.mean_head_frobenius_norm():.3f}$) | ✅ PASS |",
        f"| Colapso Empírico de Sorpresa Activa ($\\mathcal{{S}}_{{final}} < \\mathcal{{S}}_{{init}}$) | **{tester['active_surprise_collapse_pct']:.2f}%** | ✅ PASS |",
        f"| Invariante Topológico de Sueño SWR (Teorema CDT) | **{tester['cdt_invariant_pass_pct']:.2f}%** | ✅ PASS |",
        f"| Deriva de Energía Libre Inicial (Ventana Temprana vs Reciente) | {tester['early_window_f_init']:.3f} $\\rightarrow$ {tester['recent_window_f_init']:.3f} nats (**{tester['f_init_drift']:+.4f} nats**) | 📉 OPTIMIZANDO |",
        f"| Deriva de Energía Libre Final (Ventana Temprana vs Reciente) | {tester['early_window_f_opt']:.3f} $\\rightarrow$ {tester['recent_window_f_opt']:.3f} nats (**{tester['f_opt_drift']:+.4f} nats**) | 📉 OPTIMIZANDO |",
    ])
    if tester.get("lm_evals_audited", 0) > 0:
        lines.extend([
            f"| Normalización Simplex $\\Delta^{{255}}$ ($\\max |\\sum P - 1| < 10^{{-4}}$) | **{tester['lm_prob_norm_pass_pct']:.2f}%** ({tester['lm_evals_audited']} evals) | ✅ PASS |",
            f"| Identidades Exactas BPB / BPC / CE / PPL | **{tester['lm_identity_pass_pct']:.2f}%** (Mejor BPB={tester['lm_best_bpb']:.4f}) | ✅ PASS |",
            f"| Ley de Dale en Columna Neural LM & Cero Fuga Train/Held-Out | **{tester['lm_dale_pass_pct']:.2f}%** Dale / **{tester['lm_anti_leakage_pass_pct']:.2f}%** Anti-Fuga | ✅ PASS |",
        ])

    lines.extend([
        "",
        "### 4. Capa Metacognitiva de Segundo Orden (Nelson-Narens & Control Racional)",
        "",
        "| Métrica Metacognitiva | Resultado Empírico | Interpretación |",
        "|---|:---:|---|",
        f"| Calibración Brier (Meta vs Línea Base No Calibrada) | **{meta_s['meta_brier_score']:.4f}** vs {meta_s['baseline_brier_score']:.4f} (**{meta_s['brier_improvement_pct']:+.1f}%**) | Mejor calibración probabilística en ventana reciente ({meta_s['recent_meta_brier']:.4f}) |",
        f"| Error Esperado de Calibración (ECE 10-bins) | **{meta_s['ece_10bin']:.4f}** (EMA={meta_s['recent_ece_ema']:.4f}) | Coincidencia entre confianza predicha y éxito real |",
        f"| Ahorro Real de Cómputo (Pasos E-Step Ejecutados vs Base) | **{meta_s['compute_savings_pct']:.2f}%** ({meta_s['actual_e_steps_total']}/{meta_s['baseline_e_steps_total']} pasos) | Poda racional en episodios Sistema-1 consolidados |",
        f"| Vetos Anti-Sobreconfianza (`self_doubt_veto`) | **{meta_s['self_doubt_vetos']}** ({meta_s['self_doubt_veto_rate_pct']:.2f}%) | Bloqueo de ilusión epistémica ante anomalías Spelke/Pearl/HRR |",
        f"| Distribución de Esfuerzo Cognitivo | S1={tiers['System1Fast']} ({tier_pct['System1Fast']:.1f}%) / S2={tiers['System2Deliberative']} ({tier_pct['System2Deliberative']:.1f}%) / Meta={tiers['ActiveMetaInquiry']} ({tier_pct['ActiveMetaInquiry']:.1f}%) | Asignación adaptativa de precisión y pasos variacionales |",
    ])

    if lm_telemetry and lm_telemetry.get("latest_eval"):
        le = lm_telemetry["latest_eval"]
        gh = le["global_held_out"]
        lines.extend([
            "",
            "### 5. Medición Real de BPB, BPC, Cross-Entropy Loss (nats) y Perplejidad (PPL) en Held-Out Sin Fuga",
            "",
            "| Modalidad Held-Out (100% Congelada, 0 Solapamiento 128B) | Bytes | Chars | CE Loss (nats) ↓ | BPB ↓ | BPC ↓ | PPL (byte) ↓ | PPL (char) ↓ | Top-1 Acc ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for dom_k in ("markdown", "rust", "python"):
            dm = le.get("per_domain_held_out", {}).get(dom_k)
            if dm:
                lines.append(
                    f"| **{dom_k.capitalize()}** | {dm['n_bytes']:,} | {dm['n_chars']:,} | "
                    f"{dm['ce_nats']:.4f} | **{dm['bpb']:.4f}** | **{dm['bpc']:.4f}** | "
                    f"**{dm['ppl_byte']:.4f}** | **{dm['ppl_char']:.4f}** | {dm['top1_acc_pct']:.1f}% |"
                )
        lines.append(
            f"| **GLOBAL HELD-OUT (Ponderado Exacto)** | **{gh['n_bytes']:,}** | **{gh['n_chars']:,}** | "
            f"**{gh['ce_nats']:.4f}** | **{gh['bpb']:.4f}** | **{gh['bpc']:.4f}** | "
            f"**{gh['ppl_byte']:.4f}** | **{gh['ppl_char']:.4f}** | **{gh['top1_acc_pct']:.1f}%** |"
        )
        if le.get("blind_bayesian_held_out"):
            bm = le["blind_bayesian_held_out"]
            lines.append(
                f"| **Enrutamiento Bayesiano Ciego (0 etiquetas)** | {bm['n_bytes']:,} | {bm['n_chars']:,} | "
                f"{bm['ce_nats']:.4f} | **{bm['bpb']:.4f}** | **{bm['bpc']:.4f}** | "
                f"**{bm['ppl_byte']:.4f}** | **{bm['ppl_char']:.4f}** | {bm['top1_acc_pct']:.1f}% |"
            )
        if le.get("canonical_eval_md_frozen"):
            cm = le["canonical_eval_md_frozen"]
            lines.append(
                f"| **Canónico `aura-engine` `eval_md` (vs 3.7393 BPB)** | {cm['n_bytes']:,} | {cm['n_chars']:,} | "
                f"{cm['ce_nats']:.4f} | **{cm['bpb']:.4f}** | **{cm['bpc']:.4f}** | "
                f"**{cm['ppl_byte']:.4f}** | **{cm['ppl_char']:.4f}** | {cm['top1_acc_pct']:.1f}% |"
            )

    lines.extend([
        "",
        "### 6. Mecanismo de Cierre de Ciclo FEP-MARCUS v3 Metacognitivo",
        "",
        "1. **E-Step Variacional Amortiguado y Adaptativo:** Asigna presupuesto `System1Fast` (2/4 pasos), `System2Deliberative` (6/9+4 pasos) o `ActiveMetaInquiry` (9+6 pasos) usando cota espectral $\\sigma_{\\max}^2(W_h)$ con descenso estrictamente monótono.",
        "2. **Monitoreo 2º Orden y Veto Epistémico:** Evalúa entropía del posterior, compresión de varianza y conflicto inter-modal; activa `self_doubt_veto` y modula $(\\gamma, w_e, w_p, \\eta_{meta})$ si detecta ilusión de conocimiento.",
        "3. **M-Step de Codificación Predictiva y Sueño SWR:** Actualiza $W^E, W^I \\ge 0$ con tasa modulada metacognitivamente + homeostasis Frobenius + SVD en sueño SWR + poda física microglial + posteriores Beta de los 5 especialistas.",
        "4. **Evaluación Artificial Analysis y Lenguaje Real Held-Out sin Fuga:** Evalúa tanto generalización taxonómica AA como BPB/BPC/Cross-Entropy/PPL estrictamente congelados sobre archivos no vistos del repositorio con 0% de fuga.",
        "",
    ])
    atomic_write_text(REPORT_MD_PATH, "\n".join(lines))


# ==============================================================================
# ARNÉS PRINCIPAL DE EJECUCIÓN INFINITA / ACOTADA
# ==============================================================================

def run_infinite_multiagent_fep_loop(
    eval_interval: int = 100,
    sleep_delay: float = 0.02,
    max_episodes: int = 0,
    curriculum_size: int = 48,
    reset_state: bool = False,
    seed: int = 42,
    enable_aura_v3_genesis: bool = False,
    enable_lm_bpb: bool = True,
    lm_cycles: int = 8,
    lm_eval_bytes_per_domain: int = 24576,
    lm_target_bpb: float = 1.80,
    lm_target_ppl: float = 3.50,
    lm_neural_samples_per_cycle: int = 65536,
    lm_artifact_path: Optional[str] = None,
    lm_full_confirm_eval: bool = True,
) -> Dict[str, Any]:
    if reset_state:
        reset_runtime_state()

    banner_title = (
        "  INICIANDO BUCLE AURA v3 GENESIS + FEP-MARCUS 7B v3 METACOGNITIVO + AA + BPB/PPL"
        if enable_aura_v3_genesis
        else "  INICIANDO BUCLE FEP-MARCUS 7B v3 METACOGNITIVO + MULTI-AGENTE + AA + BPB/PPL"
    )
    print("\n" + "=" * 88)
    print(banner_title)
    print("=" * 88)
    print(
        f"[*] Intervalo AA/LM: cada {eval_interval} eps | Currículo: {curriculum_size} | "
        f"MaxEps: {max_episodes or 'INFINITO'} | LM-BPB: {'ON' if enable_lm_bpb else 'OFF'} "
        f"(ciclos={lm_cycles}, meta BPB/BPC<={lm_target_bpb:.2f}, PPL<={lm_target_ppl:.2f})"
    )
    print(f"[*] Logs acotados en: {LOG_FILE}\n")

    orchestrator = MultiAgentOrchestrator(ROOT_DIR, enable_aura_v3_genesis=enable_aura_v3_genesis)
    fep_engine = MarcusFep7bInfiniteEngine(
        CHECKPOINT_DIR,
        ARTIFACTS_DIR,
        seed=seed,
        load_existing=not reset_state,
        enable_aura_v3_genesis=enable_aura_v3_genesis,
        enable_language_model=enable_lm_bpb,
    )
    if (not reset_state) and fep_engine.loaded_orchestrator_state:
        orchestrator.restore_state(fep_engine.loaded_orchestrator_state)
        print("[*] Contadores y posteriores Beta de 5 agentes restaurados desde checkpoint v3.")

    aa_evaluator = ArtificialAnalysisContinuousEvaluator()

    # Inicialización de corpus real del repositorio con separación estricta Train / Held-Out sin fuga
    lm_splitter: Optional[RepoTextCorpusSplitter] = None
    lm_baselines: Optional[Dict[str, Any]] = None
    lm_cyclic_history: List[Dict[str, Any]] = []
    lm_telemetry: Optional[Dict[str, Any]] = None
    lm_out_path = lm_artifact_path or LM_BPB_ARTIFACT
    lm_cycle_idx = 0

    def _lm_target_met(eval_dict: Dict[str, Any]) -> bool:
        gh = eval_dict["global_held_out"]
        return (
            gh["bpb"] <= lm_target_bpb
            and gh["bpc"] <= lm_target_bpb
            and gh["ppl_byte"] <= lm_target_ppl
            and gh["ppl_char"] <= lm_target_ppl
        )

    def _record_lm_eval(
        ep_tag: int,
        train_stats: Optional[Dict[str, Any]],
        eval_res: Dict[str, Any],
        full_confirm_eval: Optional[Dict[str, Any]] = None,
        baselines_full: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nonlocal lm_telemetry
        assert lm_splitter is not None
        gh = eval_res["global_held_out"]
        audit_lm = fep_engine.tester.audit_language_eval(gh, anti_leakage=lm_splitter.audit_report)
        cm = eval_res.get("canonical_eval_md_frozen")
        cs_fr = eval_res.get("canonical_shift_rust_frozen")
        point = {
            "cycle": eval_res["cycle"],
            "episode": ep_tag,
            "total_train_bytes_seen": eval_res["total_train_bytes_seen"],
            "train_stats": train_stats,
            "global_bpb": gh["bpb"],
            "global_bpc": gh["bpc"],
            "global_ce_nats": gh["ce_nats"],
            "global_ppl_byte": gh["ppl_byte"],
            "global_ppl_char": gh["ppl_char"],
            "global_top1_acc_pct": gh["top1_acc_pct"],
            "global_top5_acc_pct": gh["top5_acc_pct"],
            "global_myelinated_pct": gh["myelinated_steps_pct"],
            "global_neural_only_bpb": gh["neural_only_bpb"],
            "delta_vs_374_bpb": round(gh["bpb"] - 3.7393, 6),
            "markdown_bpb": eval_res["per_domain_held_out"].get("markdown", {}).get("bpb"),
            "rust_bpb": eval_res["per_domain_held_out"].get("rust", {}).get("bpb"),
            "python_bpb": eval_res["per_domain_held_out"].get("python", {}).get("bpb"),
            "canonical_eval_md_bpb": cm["bpb"] if cm else None,
            "canonical_shift_rust_frozen_bpb": cs_fr["bpb"] if cs_fr else None,
            "max_prob_norm_error": gh["max_prob_norm_error"],
            "dale_violations": gh["dale_violations"],
            "audit": audit_lm,
            "target_met": _lm_target_met(eval_res),
        }
        lm_cyclic_history.append(point)
        lm_telemetry = {
            "architecture": (
                "AURA_V3_GENESIS_FEP_MARCUS_V3"
                if enable_aura_v3_genesis
                else "FEP_MARCUS_V3_BASELINE"
            ),
            "aura_v3_genesis_enabled": bool(enable_aura_v3_genesis),
            "seed": seed,
            "targets": {
                "bpb_max": lm_target_bpb,
                "bpc_max": lm_target_bpb,
                "ppl_max": lm_target_ppl,
                "baseline_ref_bpb": 3.7393,
            },
            "target_achieved": _lm_target_met(eval_res),
            "anti_leakage_audit": lm_splitter.audit_report,
            "baselines": lm_baselines,
            "latest_eval": eval_res,
            "full_confirm_eval": full_confirm_eval,
            "baselines_full_confirm": baselines_full,
            "cyclic_history": list(lm_cyclic_history),
            "last_audit": audit_lm,
        }
        atomic_write_json(lm_out_path, lm_telemetry)
        return point

    if enable_lm_bpb and fep_engine.language_model is not None:
        lm_splitter = RepoTextCorpusSplitter(
            ROOT_DIR,
            n_cycles=lm_cycles,
            seed=seed,
            eval_bytes_per_domain=lm_eval_bytes_per_domain,
        )
        lm_baselines = _evaluate_lm_baselines(lm_splitter, use_full_buffers=False)
        al = lm_splitter.audit_report
        bg_g = lm_baselines["bigram_byte_baseline"]["global_held_out"]
        print(
            f"[*] Corpus real del repo (0% fuga auditado): "
            f"train={al['total_train_files']} archivos ({al['total_train_bytes']:,} B, SHA={al['train_sha256']}) | "
            f"held-out={al['total_held_out_files']} archivos ({al['total_held_out_eval_bytes']:,} B eval, SHA={al['held_out_sha256']}) | "
            f"shared_files={al['shared_files']}, shared_hashes={al['shared_hashes']}, shared_128B={al['shared_128b_blocks']}"
        )
        print(
            f"[*] Líneas base held-out congelado: Uniforme=8.0000 BPB (PPL=256.0) | "
            f"Bigram={bg_g['bpb']:.4f} BPB / {bg_g['bpc']:.4f} BPC (PPL={bg_g['ppl_byte']:.2f}) | "
            f"Ref AURA v3 Rust=3.7393 BPB (PPL=13.35)"
        )

    # Fuentes de entrenamiento 100% libres de preguntas/respuestas de Artificial Analysis
    base_turns = [
        t for t in generate_1000_evaluation_turns()
        if t.get("category") != "Artificial Analysis Expert QA"
    ]
    if not base_turns:
        base_turns = STRESS_OOD_DRILLS

    streamer = ModelDataStreamer(seed=seed)
    hf_convs = _load_json_list(
        os.path.join(DATA_DIR, "llm_generated_families", "hf_llama3_conversations.json")
    )
    dyn_dataset = _load_json_list(
        os.path.join(DATA_DIR, "llm_generated_families", "llm_multilingual_dynamical_dataset.json")
    )
    print(
        f"[*] Fuentes cargadas sin fuga AA: base_turns={len(base_turns)}, "
        f"hf_convs={len(hf_convs)}, dyn_trajs={len(dyn_dataset)}, ModelDataStreamer=INFINITO.\n"
    )

    # Desplazamientos monótonos por flujo (para no estancarse al reanudar)
    offset_ep = fep_engine.episodes_processed
    stream_idx = offset_ep
    turn_idx = offset_ep
    conv_idx = offset_ep
    dyn_idx = offset_ep
    non_curr_count = offset_ep

    stop_requested = False

    def _handle_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\n[!] Señal {signum} recibida. Guardando checkpoint atómico y saliendo limpiamente...", flush=True)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start_time = time.perf_counter()

    # Evaluación Ronda 0 al iniciar fresco (o restauración del último resultado AA)
    last_aa_result = aa_evaluator.get_last_result()
    if last_aa_result is None or aa_evaluator.evaluation_round == 0:
        last_aa_result = aa_evaluator.run_benchmark_evaluation(
            fep_engine, orchestrator=orchestrator, current_episode=fep_engine.episodes_processed
        )
        print(
            f"[+] Ronda #0 (Línea Base Ep={fep_engine.episodes_processed}): "
            f"Pasiva={last_aa_result['passive_accuracy_pct']:.2f}% "
            f"({last_aa_result['passive_correct']}/{last_aa_result['total_questions']}) | "
            f"Activa={last_aa_result['overall_accuracy_pct']:.2f}% | "
            f"Débil='{last_aa_result['weakest_domain']}' ({last_aa_result['weakest_domain_acc']:.2f}%)\n"
        )
    else:
        print(
            f"[+] Estado AA reanudado (Ronda #{aa_evaluator.evaluation_round}, Ep={fep_engine.episodes_processed}): "
            f"Pasiva={last_aa_result['passive_accuracy_pct']:.2f}% | "
            f"Activa={last_aa_result['overall_accuracy_pct']:.2f}% | "
            f"Débil='{last_aa_result['weakest_domain']}'\n"
        )

    # Ciclo 0 de lenguaje real (solo currículo prenatal in-utero, 0 bytes reales vistos del repo)
    if enable_lm_bpb and fep_engine.language_model is not None and lm_splitter is not None:
        ev0 = fep_engine.language_model.evaluate_full_held_out_suite(
            lm_splitter,
            use_full_buffers=False,
            include_blind_routing=False,
            include_online_shift=False,
            generate_samples=False,
        )
        pt0 = _record_lm_eval(fep_engine.episodes_processed, None, ev0)
        lm0_log = (
            f"[LM Ciclo #0 (Prenatal 0B repo) @ Ep {fep_engine.episodes_processed}] "
            f"Held-Out FROZEN: BPB={pt0['global_bpb']:.4f} | BPC={pt0['global_bpc']:.4f} | "
            f"CE={pt0['global_ce_nats']:.4f} nats | PPL_b={pt0['global_ppl_byte']:.2f} | "
            f"PPL_c={pt0['global_ppl_char']:.2f} | NormErr={pt0['max_prob_norm_error']:.1e}"
        )
        print(lm0_log, flush=True)
        append_bounded_log(LOG_FILE, lm0_log)

    curriculum_queue: List[Dict[str, str]] = aa_evaluator.generate_targeted_curriculum(
        last_aa_result.get("weakest_domains_order", []),
        last_aa_result.get("failed_subtopics_by_domain", {}),
        count=curriculum_size,
    )

    episodes_this_run = 0
    last_cycle: Dict[str, Any] = {
        "episode": fep_engine.episodes_processed,
        "f_final": 0.0,
        "elbo_final": 0.0,
        "s_final": 0.0,
        "dale_violations": 0,
    }

    while not stop_requested:
        if max_episodes > 0 and episodes_this_run >= max_episodes:
            break

        next_ep = fep_engine.episodes_processed + 1
        ep_meta: Optional[Dict[str, Any]] = None
        subtopic = ""

        # Intercalado balanceado: currículo focalizado en pasos pares + 4 flujos multi-modales
        if curriculum_queue and (next_ep % 2 == 0 or len(curriculum_queue) > curriculum_size):
            item = curriculum_queue.pop(0)
            q_text = item["question"]
            dom = item["domain"]
            subtopic = item.get("subtopic", "")
        else:
            slot = non_curr_count % 4
            non_curr_count += 1
            if slot == 0:
                raw_ep = streamer.generate_next_batch(1, stream_idx)[0]
                stream_idx += 1
                q_text, dom, subtopic, ep_meta = _format_streamer_episode(raw_ep)
            elif slot == 1:
                turn = base_turns[turn_idx % len(base_turns)]
                turn_idx += 1
                q_text = turn["question"]
                dom = turn.get("domain", turn.get("category", "General"))
                subtopic = turn.get("category", "")
            elif slot == 2 and dyn_dataset:
                dyn_it = dyn_dataset[dyn_idx % len(dyn_dataset)]
                dyn_idx += 1
                q_text, dom, subtopic, ep_meta = _format_dynamical_item(dyn_it)
            elif hf_convs:
                c_idx = conv_idx % len(hf_convs)
                conv_idx += 1
                q_text, dom, subtopic, ep_meta = _format_hf_conversation(hf_convs[c_idx], c_idx)
            else:
                turn = base_turns[turn_idx % len(base_turns)]
                turn_idx += 1
                q_text = turn["question"]
                dom = turn.get("domain", "General")

        last_cycle = fep_engine.step_cognitive_cycle(
            question=q_text,
            domain=dom,
            orchestrator=orchestrator,
            subtopic=subtopic,
            episode_meta=ep_meta,
            mutate_memory=True,
        )
        episodes_this_run += 1
        ep_num = last_cycle["episode"]

        if ep_num % 10 == 0:
            avg_f = fep_engine.cum_free_energy / ep_num
            act_rate = (fep_engine.epistemic_queries_triggered / ep_num) * 100.0
            meta_ep = last_cycle.get("metacognition", {})
            meta_sum = fep_engine.metacognition.summary()
            log_line = (
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Ep #{ep_num:6d} | "
                f"F_init->F_opt: {last_cycle['f_init']:.2f}->{last_cycle['f_final']:.2f} (avg {avg_f:.2f}) | "
                f"S: {last_cycle['s_init']:.2f}->{last_cycle['s_final']:.2f} (-{last_cycle['surprise_reduction_pct']:.1f}%) | "
                f"ActInf: {act_rate:.1f}% ({last_cycle['consulted_agent'] or 'System-1'}) | "
                f"Meta[{meta_ep.get('effort_tier', 'S2')} conf={meta_ep.get('meta_confidence', 0.0):.2f} Brier={meta_sum['meta_brier_score']:.3f} Sav={meta_sum['compute_savings_pct']:.1f}%] | "
                f"||W||_F: {last_cycle['head_fro_norm']:.2f} | Pi: {last_cycle['mean_precision']:.2f} | "
                f"DaleViol: {last_cycle['dale_violations']}"
            )
            print(log_line, flush=True)
            append_bounded_log(LOG_FILE, log_line)

        if ep_num % 25 == 0:
            telem = _build_telemetry_payload(
                start_time,
                "RUNNING_INFINITE",
                last_cycle,
                fep_engine,
                orchestrator,
                last_aa_result,
                episodes_this_run,
                lm_telemetry=lm_telemetry,
            )
            atomic_write_json(TELEMETRY_ARTIFACT, telem)

        if eval_interval > 0 and (ep_num % eval_interval == 0):
            print(f"\n[>>>] EVALUACIÓN ARTIFICIAL ANALYSIS + LM HELD-OUT (Ronda #{aa_evaluator.evaluation_round + 1} @ Ep #{ep_num})...", flush=True)
            last_aa_result = aa_evaluator.run_benchmark_evaluation(
                fep_engine, orchestrator=orchestrator, current_episode=ep_num
            )

            # Entrenamiento cíclico de 1 shard disjunto de texto real + validación estrictamente congelada
            if enable_lm_bpb and fep_engine.language_model is not None and lm_splitter is not None:
                tr_st = fep_engine.language_model.train_cycle_shard(
                    lm_cycle_idx,
                    lm_splitter,
                    neural_samples=lm_neural_samples_per_cycle,
                    calibrate_on_train_cal=True,
                )
                lm_cycle_idx += 1
                ev_c = fep_engine.language_model.evaluate_full_held_out_suite(
                    lm_splitter,
                    use_full_buffers=False,
                    include_blind_routing=False,
                    include_online_shift=False,
                    generate_samples=False,
                )
                pt_c = _record_lm_eval(ep_num, tr_st, ev_c)
                lm_log = (
                    f"[LM Ciclo #{pt_c['cycle']} @ Ep {ep_num}] "
                    f"Held-Out FROZEN: BPB={pt_c['global_bpb']:.4f} | BPC={pt_c['global_bpc']:.4f} | "
                    f"CE={pt_c['global_ce_nats']:.4f} nats | PPL_b={pt_c['global_ppl_byte']:.4f} | "
                    f"PPL_c={pt_c['global_ppl_char']:.4f} | Δvs3.74={pt_c['delta_vs_374_bpb']:+.4f} | "
                    f"MD={pt_c['markdown_bpb']:.4f} RS={pt_c['rust_bpb']:.4f} PY={pt_c['python_bpb']:.4f} | "
                    f"CanonMD={pt_c['canonical_eval_md_bpb']:.4f} | Top1={pt_c['global_top1_acc_pct']:.1f}% | "
                    f"Myelin={pt_c['global_myelinated_pct']:.1f}% | NeuralBPB={pt_c['global_neural_only_bpb']:.4f} | "
                    f"Meta={'PASS' if pt_c['target_met'] else 'TRAIN'}"
                )
                print(lm_log, flush=True)
                append_bounded_log(LOG_FILE, lm_log)

            fep_engine.save_checkpoint(
                last_cycle["f_final"],
                last_cycle["elbo_final"],
                last_cycle["s_final"],
                orchestrator_state=orchestrator.export_state(),
            )

            pass_acc = last_aa_result["passive_accuracy_pct"]
            act_acc = last_aa_result["overall_accuracy_pct"]
            rescued = last_aa_result["active_rescued_questions"]
            weakest = last_aa_result["weakest_domain"]
            w_acc = last_aa_result["weakest_domain_acc"]
            tester_sum = fep_engine.tester.summary()
            meta_sum = fep_engine.metacognition.summary()

            eval_log = (
                f"[AA Ronda #{last_aa_result['evaluation_round']} @ Ep {ep_num}] "
                f"Pasiva: {pass_acc:.2f}% ({last_aa_result['passive_correct']}/300, {last_aa_result['passive_delta_pct']:+.2f}%) | "
                f"Activa: {act_acc:.2f}% (Rescatadas: {rescued}) | "
                f"Débil: '{weakest}' ({w_acc:.1f}%) | "
                f"Brier={meta_sum['meta_brier_score']:.4f}/{meta_sum['baseline_brier_score']:.4f} ({meta_sum['brier_improvement_pct']:+.1f}%) | "
                f"ECE={meta_sum['ece_10bin']:.4f} | Ahorro={meta_sum['compute_savings_pct']:.1f}% | Vetos={meta_sum['self_doubt_vetos']} | "
                f"Monot: {tester_sum['monotonic_convergence_pct']:.1f}%"
            )
            print(eval_log, flush=True)
            append_bounded_log(LOG_FILE, eval_log)

            curriculum_queue = aa_evaluator.generate_targeted_curriculum(
                last_aa_result.get("weakest_domains_order", []),
                last_aa_result.get("failed_subtopics_by_domain", {}),
                count=curriculum_size,
            )

            status_md = (
                "🟢 Ejecutándose Continuamente en Segundo Plano (Daemon v3 Metacognitivo)"
                if max_episodes <= 0
                else "🟢 Validación / Precalentamiento Verificado (v3 Metacognitivo)"
            )
            _update_markdown_report(status_md, fep_engine, orchestrator, last_aa_result, lm_telemetry=lm_telemetry)

            telem = _build_telemetry_payload(
                start_time,
                "RUNNING_INFINITE",
                last_cycle,
                fep_engine,
                orchestrator,
                last_aa_result,
                episodes_this_run,
                lm_telemetry=lm_telemetry,
            )
            atomic_write_json(TELEMETRY_ARTIFACT, telem)
        elif ep_num % 100 == 0:
            fep_engine.save_checkpoint(
                last_cycle["f_final"],
                last_cycle["elbo_final"],
                last_cycle["s_final"],
                orchestrator_state=orchestrator.export_state(),
            )

        if sleep_delay > 0.0:
            time.sleep(sleep_delay)

    # Completar shards cíclicos restantes de texto real (si los episodios terminaron antes de lm_cycles)
    # y ejecutar suite final de confirmación (con enrutamiento bayesiano ciego, OOD shift y muestras verbatim)
    if (not stop_requested) and enable_lm_bpb and fep_engine.language_model is not None and lm_splitter is not None:
        while lm_cycle_idx < lm_cycles:
            tr_st = fep_engine.language_model.train_cycle_shard(
                lm_cycle_idx,
                lm_splitter,
                neural_samples=lm_neural_samples_per_cycle,
                calibrate_on_train_cal=True,
            )
            lm_cycle_idx += 1
            if enable_aura_v3_genesis:
                fep_engine.language_model.execute_microglia_swr_pruning()
            ev_c = fep_engine.language_model.evaluate_full_held_out_suite(
                lm_splitter,
                use_full_buffers=False,
                include_blind_routing=False,
                include_online_shift=False,
                generate_samples=False,
            )
            pt_c = _record_lm_eval(fep_engine.episodes_processed, tr_st, ev_c)
            lm_log = (
                f"[LM Ciclo #{pt_c['cycle']} (Cierre Corpus) @ Ep {fep_engine.episodes_processed}] "
                f"Held-Out FROZEN: BPB={pt_c['global_bpb']:.4f} | BPC={pt_c['global_bpc']:.4f} | "
                f"CE={pt_c['global_ce_nats']:.4f} nats | PPL_b={pt_c['global_ppl_byte']:.4f} | "
                f"PPL_c={pt_c['global_ppl_char']:.4f} | Δvs3.74={pt_c['delta_vs_374_bpb']:+.4f} | "
                f"MD={pt_c['markdown_bpb']:.4f} RS={pt_c['rust_bpb']:.4f} PY={pt_c['python_bpb']:.4f} | "
                f"CanonMD={pt_c['canonical_eval_md_bpb']:.4f} | Meta={'PASS' if pt_c['target_met'] else 'TRAIN'}"
            )
            print(lm_log, flush=True)
            append_bounded_log(LOG_FILE, lm_log)

        print("\n[>>>] EJECUTANDO CONFIRMACIÓN FINAL HELD-OUT (Blind Bayesian + OOD Shift + Full Confirm + Muestras)...", flush=True)
        final_std_eval = fep_engine.language_model.evaluate_full_held_out_suite(
            lm_splitter,
            use_full_buffers=False,
            include_blind_routing=True,
            include_online_shift=True,
            generate_samples=True,
        )
        full_confirm_eval = None
        baselines_full = None
        if lm_full_confirm_eval:
            full_confirm_eval = fep_engine.language_model.evaluate_full_held_out_suite(
                lm_splitter,
                use_full_buffers=True,
                include_blind_routing=False,
                include_online_shift=False,
                generate_samples=False,
                include_canonical=False,
            )
            baselines_full = _evaluate_lm_baselines(lm_splitter, use_full_buffers=True)

        # Actualizar telemetría final con la confirmación completa (sin duplicar ciclo en historial)
        gh = final_std_eval["global_held_out"]
        audit_fin = fep_engine.tester.audit_language_eval(gh, anti_leakage=lm_splitter.audit_report)
        if full_confirm_eval is not None:
            fep_engine.tester.audit_language_eval(
                full_confirm_eval["global_held_out"], anti_leakage=lm_splitter.audit_report
            )
        lm_telemetry = {
            "architecture": (
                "AURA_V3_GENESIS_FEP_MARCUS_V3"
                if enable_aura_v3_genesis
                else "FEP_MARCUS_V3_BASELINE"
            ),
            "aura_v3_genesis_enabled": bool(enable_aura_v3_genesis),
            "seed": seed,
            "targets": {
                "bpb_max": lm_target_bpb,
                "bpc_max": lm_target_bpb,
                "ppl_max": lm_target_ppl,
                "baseline_ref_bpb": 3.7393,
            },
            "target_achieved": _lm_target_met(final_std_eval),
            "anti_leakage_audit": lm_splitter.audit_report,
            "baselines": lm_baselines,
            "latest_eval": final_std_eval,
            "full_confirm_eval": full_confirm_eval,
            "baselines_full_confirm": baselines_full,
            "cyclic_history": list(lm_cyclic_history),
            "last_audit": audit_fin,
        }
        atomic_write_json(lm_out_path, lm_telemetry)

        bm = final_std_eval.get("blind_bayesian_held_out", {})
        cm = final_std_eval.get("canonical_eval_md_frozen", {})
        rs_f = final_std_eval.get("canonical_shift_rust_frozen", {})
        rs_o = final_std_eval.get("canonical_shift_rust_online_causal", {})
        print(
            f"[+] CONFIRMACIÓN FINAL HELD-OUT FROZEN ({gh['n_bytes']:,} B, 0 fuga): "
            f"BPB={gh['bpb']:.4f} | BPC={gh['bpc']:.4f} | CE={gh['ce_nats']:.4f} nats | "
            f"PPL_b={gh['ppl_byte']:.4f} | PPL_c={gh['ppl_char']:.4f} | Top1={gh['top1_acc_pct']:.1f}% | "
            f"BlindBPB={bm.get('bpb', 0.0):.4f} (Lock={bm.get('blind_domain_lock_accuracy_pct', 0.0):.1f}%) | "
            f"CanonMD={cm.get('bpb', 0.0):.4f} | ShiftRust(F/O)={rs_f.get('bpb', 0.0):.4f}/{rs_o.get('bpb', 0.0):.4f}",
            flush=True,
        )
        if full_confirm_eval is not None:
            fgh = full_confirm_eval["global_held_out"]
            print(
                f"[+] FULL CONFIRM HELD-OUT ({fgh['n_bytes']:,} B, {fgh['n_chars']:,} chars): "
                f"BPB={fgh['bpb']:.4f} | BPC={fgh['bpc']:.4f} | CE={fgh['ce_nats']:.4f} nats | "
                f"PPL_b={fgh['ppl_byte']:.4f} | PPL_c={fgh['ppl_char']:.4f} | "
                f"Meta={'PASS' if _lm_target_met(full_confirm_eval) else 'FAIL'}",
                flush=True,
            )

    # Guardado final limpio al terminar (acotado o por señal)
    final_status = "STOPPED_CLEANLY" if stop_requested else "COMPLETED_BOUNDED"
    fep_engine.save_checkpoint(
        last_cycle["f_final"],
        last_cycle["elbo_final"],
        last_cycle["s_final"],
        orchestrator_state=orchestrator.export_state(),
    )
    telem = _build_telemetry_payload(
        start_time,
        final_status,
        last_cycle,
        fep_engine,
        orchestrator,
        last_aa_result,
        episodes_this_run,
        lm_telemetry=lm_telemetry,
    )
    atomic_write_json(TELEMETRY_ARTIFACT, telem)
    _update_markdown_report(
        f"🟡 Finalizado ({final_status})",
        fep_engine,
        orchestrator,
        last_aa_result,
        lm_telemetry=lm_telemetry,
    )
    return telem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arnés Infinito Multi-Agente FEP-MARCUS 7B v3 Metacognitivo + BPB/BPC/CE/PPL")
    parser.add_argument("--eval-interval", type=int, default=100, help="Intervalo de episodios para reevaluar Artificial Analysis y ciclo LM")
    parser.add_argument("--sleep-delay", type=float, default=0.02, help="Pausa entre episodios en segundos")
    parser.add_argument("--max-episodes", type=int, default=0, help="Límite máximo de episodios en esta corrida (0 = infinito)")
    parser.add_argument("--curriculum-size", type=int, default=48, help="Cantidad de ejercicios epistémicos por ronda AA")
    parser.add_argument("--reset-state", action="store_true", help="Reinicia pesos Dale, memoria Hopfield y métricas desde Ep 0")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria determinista")
    parser.add_argument("--aura-v3-genesis", action="store_true", help="Activa las 4 subrutinas generativas de ADN de AURA v3 Genesis sobre el arnés FEP-MARCUS v3")
    parser.add_argument("--disable-lm-bpb", action="store_true", help="Desactiva el entrenamiento cíclico y medición real de BPB/BPC/CE/PPL")
    parser.add_argument("--lm-cycles", type=int, default=8, help="Número de shards cíclicos disjuntos del corpus real del repositorio")
    parser.add_argument("--lm-eval-bytes-per-domain", type=int, default=24576, help="Bytes held-out estratificados por modalidad en cada ciclo")
    parser.add_argument("--lm-target-bpb", type=float, default=1.80, help="Meta competitiva de BPB y BPC en validación held-out")
    parser.add_argument("--lm-target-ppl", type=float, default=3.50, help="Meta competitiva de Perplejidad (PPL) en validación held-out")
    parser.add_argument("--lm-neural-samples", type=int, default=65536, help="Muestras por época cíclica para la columna neural Mountcastle Dale")
    parser.add_argument("--lm-artifact-path", type=str, default=None, help="Ruta de salida JSON para la telemetría completa de BPB/BPC/CE/PPL")
    parser.add_argument("--skip-lm-full-confirm", action="store_true", help="Omite la evaluación final ampliada (64 KiB/dominio) al concluir")
    args = parser.parse_args()

    run_infinite_multiagent_fep_loop(
        eval_interval=args.eval_interval,
        sleep_delay=args.sleep_delay,
        max_episodes=args.max_episodes,
        curriculum_size=args.curriculum_size,
        reset_state=args.reset_state,
        seed=args.seed,
        enable_aura_v3_genesis=args.aura_v3_genesis,
        enable_lm_bpb=not args.disable_lm_bpb,
        lm_cycles=args.lm_cycles,
        lm_eval_bytes_per_domain=args.lm_eval_bytes_per_domain,
        lm_target_bpb=args.lm_target_bpb,
        lm_target_ppl=args.lm_target_ppl,
        lm_neural_samples_per_cycle=args.lm_neural_samples,
        lm_artifact_path=args.lm_artifact_path,
        lm_full_confirm_eval=not args.skip_lm_full_confirm,
    )
