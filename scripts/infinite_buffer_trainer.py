#!/usr/bin/env python3
"""
Arnés de Entrenamiento Infinito Impulsado por Buffer con Guardado Atómico Automático
(`AURA v3 Genesis + FEP-MARCUS 7B v3 Metacognitivo + Cortical/Neural LM`).

Cada vez que el buffer acotado (`BoundedLearningBuffer`: episodios cognitivos + bytes `train`
sin fuga held-out) se llena al 100%, el arnés ejecuta automáticamente:
  1. Drenado del buffer en RAM (mantiene memoria estrictamente acotada en ejecución infinita).
  2. Consolidación cognitiva (SWR + homeostasis SVD + poda microglial bajo Ley de Dale estricta)
     y consolidación de lenguaje (ingesta de shards `train` 0..7 y luego consolidación infinita
     con LR coseno con reinicios cálidos + recalibración monótona de Friston sobre `train_cal`).
  3. Guardado atómico a disco de TODO lo aprendido:
       - `data/checkpoints/weights_marcus_fep_7b.bin` (1,048,576 sinapsis cognitivas Dale)
       - `data/checkpoints/hopfield_bank_v2.bin` (prototipos Hopfield)
       - `data/checkpoints/world_state_fep_v2.json` (WorldState, metacognición, orquestador y cursores)
       - `data/checkpoints/aura_v3_lm_learned_state.npz` + `aura_v3_lm_meta.json` (804,864 sinapsis LM
         Dale + momentos Adam + precisiones Friston W/K/B/alpha + curvas morfogenéticas)
       - Instantáneas `best_*` cada vez que se mejora el mejor BPB held-out congelado
       - Recibo criptográfico SHA-256 en `data/stream_buffer/buffer_flush_ledger.jsonl`
  4. Auditoría periódica 100% congelada (Artificial Analysis 300 preguntas + Held-Out BPB/BPC/CE/PPL
     con 0% fuga de datos) y reanudación transparente tras cualquier reinicio.
"""

import os
import sys
import time
import json
import math
import shutil
import signal
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fep_marcus_core import (  # noqa: E402
    MarcusFep7bInfiniteEngine,
    MultiAgentOrchestrator,
    RepoTextCorpusSplitter,
    AuraFepCorticalLanguageModel,
    BoundedLearningBuffer,
    atomic_write_json,
    append_bounded_log,
    file_sha256_prefix,
)
from gemma_fep_console_chat import generate_1000_evaluation_turns  # noqa: E402
from marcus_model_data_generator import ModelDataStreamer  # noqa: E402
from marcus_fep_multiagent_infinite_loop import (  # noqa: E402
    ROOT_DIR,
    DATA_DIR,
    CHECKPOINT_DIR,
    ARTIFACTS_DIR,
    LOG_FILE,
    TELEMETRY_ARTIFACT,
    STRESS_OOD_DRILLS,
    ArtificialAnalysisContinuousEvaluator,
    _load_json_list,
    _format_streamer_episode,
    _format_hf_conversation,
    _format_dynamical_item,
    _evaluate_lm_baselines,
    _build_telemetry_payload,
    _update_markdown_report,
    reset_runtime_state,
)

STREAM_BUFFER_DIR = os.path.join(DATA_DIR, "stream_buffer")
FLUSH_LEDGER_PATH = os.path.join(STREAM_BUFFER_DIR, "buffer_flush_ledger.jsonl")
PID_FILE_PATH = os.path.join(STREAM_BUFFER_DIR, "infinite_buffer_trainer.pid")
INFINITE_STATUS_ARTIFACT = os.path.join(ARTIFACTS_DIR, "infinite_buffer_training_status.json")
INFINITE_LM_ARTIFACT = os.path.join(ARTIFACTS_DIR, "infinite_buffer_lm_bpb_telemetry.json")

os.makedirs(STREAM_BUFFER_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def reset_infinite_buffer_state() -> None:
    """Limpia checkpoints previos, estado LM, ledger de flushes y telemetría viva para arranque limpio."""
    reset_runtime_state()
    extra_targets = [
        os.path.join(CHECKPOINT_DIR, "aura_v3_lm_learned_state.npz"),
        os.path.join(CHECKPOINT_DIR, "aura_v3_lm_meta.json"),
        os.path.join(CHECKPOINT_DIR, "best_aura_v3_lm_learned_state.npz"),
        os.path.join(CHECKPOINT_DIR, "best_aura_v3_lm_meta.json"),
        os.path.join(CHECKPOINT_DIR, "best_weights_marcus_fep_7b.bin"),
        FLUSH_LEDGER_PATH,
        FLUSH_LEDGER_PATH + ".1",
        INFINITE_STATUS_ARTIFACT,
        INFINITE_LM_ARTIFACT,
    ]
    removed = 0
    for path in extra_targets:
        if os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    print(f"[*] Estado del buffer infinito y checkpoints LM limpiados ({removed} archivos adicionales).", flush=True)


def _extract_rolling_train_chunk(pool: bytes, cursor: int, length: int) -> tuple[bytes, int]:
    """Extrae un bloque contiguo circular de `pool` (exclusivamente bytes `train`) avanzando el cursor."""
    n = len(pool)
    if n == 0 or length <= 0:
        return b"", 0
    start = cursor % n
    end = start + length
    if end <= n:
        return pool[start:end], end % n
    part1 = pool[start:]
    rem = length - len(part1)
    # Envolver tantas veces como haga falta si el pool es menor al bloque
    reps, tail = divmod(rem, n)
    chunk = part1 + (pool * reps) + pool[:tail]
    return chunk, tail % n


def run_infinite_buffer_training(
    buffer_episodes: int = 64,
    buffer_bytes: int = 524_288,
    max_flushes: int = 0,
    eval_every_flushes: int = 4,
    curriculum_size: int = 48,
    sleep_delay: float = 0.0,
    seed: int = 42,
    reset_state: bool = False,
    enable_aura_v3_genesis: bool = True,
    enable_lm_bpb: bool = True,
    lm_shards: int = 8,
    lm_eval_bytes_per_domain: int = 16384,
    lm_neural_samples: int = 65536,
    lm_target_bpb: float = 1.80,
    lm_target_ppl: float = 3.50,
) -> Dict[str, Any]:
    """
    Ejecuta entrenamiento infinito (o acotado a `max_flushes > 0` para pruebas) impulsado por
    `BoundedLearningBuffer`, guardando atómicamente todo lo aprendido cada vez que el buffer se llena.
    """
    if reset_state:
        reset_infinite_buffer_state()

    pid = os.getpid()
    with open(PID_FILE_PATH, "w", encoding="utf-8") as pf:
        pf.write(f"{pid}\n")

    print("\n" + "=" * 96, flush=True)
    print("  ENTRENAMIENTO INFINITO POR BUFFER CON GUARDADO ATÓMICO (AURA v3 Genesis + FEP-MARCUS v3)", flush=True)
    print("=" * 96, flush=True)
    print(
        f"[*] PID={pid} | Modo={'INFINITO' if max_flushes <= 0 else f'ACOTADO({max_flushes} flushes)'} | "
        f"Capacidad Buffer: {buffer_episodes} eps OR {buffer_bytes / 1024.0:.1f} KiB train | "
        f"AURA v3 Genesis={'ON' if enable_aura_v3_genesis else 'OFF'} | LM-BPB={'ON' if enable_lm_bpb else 'OFF'}",
        flush=True,
    )
    print(f"[*] Checkpoints atómicos: {CHECKPOINT_DIR}", flush=True)
    print(f"[*] Ledger de guardados : {FLUSH_LEDGER_PATH}", flush=True)
    print(f"[*] Estado en vivo      : {INFINITE_STATUS_ARTIFACT}\n", flush=True)

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
        print("[*] Posteriores Beta de 5 agentes restaurados desde checkpoint.", flush=True)

    aa_evaluator = ArtificialAnalysisContinuousEvaluator()
    learning_buffer = BoundedLearningBuffer(max_episodes=buffer_episodes, max_bytes=buffer_bytes)

    # Corpus real del repositorio (0% fuga auditado) y pool rodante exclusivamente de `train`
    lm_splitter: Optional[RepoTextCorpusSplitter] = None
    lm_baselines: Optional[Dict[str, Any]] = None
    train_byte_pools: Dict[str, bytes] = {d: b"" for d in AuraFepCorticalLanguageModel.DOMAINS}
    lm_cyclic_history: List[Dict[str, Any]] = []
    lm_telemetry: Optional[Dict[str, Any]] = None
    last_lm_eval: Optional[Dict[str, Any]] = None

    # Restaurar cursores y contadores del loop si existen
    lstate = (fep_engine.loaded_loop_state or {}) if not reset_state else {}
    learning_buffer.restore_counters(
        flushes_completed=int(lstate.get("buffer_flushes_completed", 0)),
        total_episodes_flushed=int(lstate.get("total_episodes_flushed", 0)),
        total_bytes_flushed=int(lstate.get("total_bytes_flushed", 0)),
    )
    lm_cycle_idx = int(lstate.get("lm_cycle_idx", 0))
    best_held_out_bpb = float(lstate.get("best_held_out_bpb", 99.0))
    best_held_out_cycle = int(lstate.get("best_held_out_cycle", 0))
    lm_byte_cursors: Dict[str, int] = {
        d: int(lstate.get("lm_byte_cursors", {}).get(d, 0))
        for d in AuraFepCorticalLanguageModel.DOMAINS
    }

    offset_ep = fep_engine.episodes_processed
    stream_idx = int(lstate.get("stream_idx", offset_ep))
    turn_idx = int(lstate.get("turn_idx", offset_ep))
    conv_idx = int(lstate.get("conv_idx", offset_ep))
    dyn_idx = int(lstate.get("dyn_idx", offset_ep))
    non_curr_count = int(lstate.get("non_curr_count", offset_ep))

    if enable_lm_bpb and fep_engine.language_model is not None:
        lm_splitter = RepoTextCorpusSplitter(
            ROOT_DIR,
            n_cycles=lm_shards,
            seed=seed,
            eval_bytes_per_domain=lm_eval_bytes_per_domain,
        )
        lm_baselines = _evaluate_lm_baselines(lm_splitter, use_full_buffers=False)
        for dom in AuraFepCorticalLanguageModel.DOMAINS:
            docs_all = [doc for shard in lm_splitter.train_shards[dom] for doc in shard]
            train_byte_pools[dom] = b"\n\n".join(docs_all)

        al = lm_splitter.audit_report
        print(
            f"[*] Corpus real inmutable (0% fuga): train={al['total_train_files']} arch ({al['total_train_bytes']:,} B, SHA={al['train_sha256']}) | "
            f"held-out={al['total_held_out_files']} arch ({al['total_held_out_eval_bytes']:,} B, SHA={al['held_out_sha256']}) | "
            f"shared_128B={al['shared_128b_blocks']}",
            flush=True,
        )

        # Restaurar estado aprendido del LM si estamos reanudando
        if not reset_state:
            restored_meta = fep_engine.language_model.load_learned_state(
                CHECKPOINT_DIR, splitter=lm_splitter, tag="latest"
            )
            if restored_meta is not None:
                lm_cycle_idx = max(lm_cycle_idx, int(restored_meta.get("cycles_trained", 0)))
                extra = restored_meta.get("extra", {})
                if isinstance(extra, dict) and extra.get("best_held_out_bpb") is not None:
                    best_held_out_bpb = min(best_held_out_bpb, float(extra["best_held_out_bpb"]))
                    best_held_out_cycle = int(extra.get("best_held_out_cycle", best_held_out_cycle))
                print(
                    f"[*] Checkpoint LM restaurado: cycles={fep_engine.language_model.cycles_trained} | "
                    f"shards={sorted(fep_engine.language_model.ingested_shards)} | "
                    f"npz_sha={restored_meta.get('npz_sha256')} | DaleViol={fep_engine.language_model._total_dale_violations()}",
                    flush=True,
                )
            if os.path.exists(INFINITE_LM_ARTIFACT):
                try:
                    with open(INFINITE_LM_ARTIFACT, "r", encoding="utf-8") as lf:
                        prev_lm = json.load(lf)
                    if isinstance(prev_lm.get("cyclic_history"), list):
                        lm_cyclic_history = list(prev_lm["cyclic_history"])[-128:]
                    if isinstance(prev_lm.get("latest_eval"), dict):
                        last_lm_eval = prev_lm["latest_eval"]
                    lm_telemetry = prev_lm
                except Exception:
                    pass

    # Fuentes de entrenamiento cognitivo (0% fuga AA)
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

    stop_requested = False

    def _handle_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(
            f"\n[!] Señal {signum} recibida. Drenando buffer parcial y guardando todo lo aprendido...",
            flush=True,
        )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start_time = time.perf_counter()

    # Línea base inicial AA si es arranque limpio
    last_aa_result = aa_evaluator.get_last_result()
    if last_aa_result is None or aa_evaluator.evaluation_round == 0:
        last_aa_result = aa_evaluator.run_benchmark_evaluation(
            fep_engine, orchestrator=orchestrator, current_episode=fep_engine.episodes_processed
        )
        print(
            f"[+] Ronda AA #0 @ Ep {fep_engine.episodes_processed}: "
            f"Pasiva={last_aa_result['passive_accuracy_pct']:.2f}% | Activa={last_aa_result['overall_accuracy_pct']:.2f}%",
            flush=True,
        )

    def _lm_target_met(eval_dict: Dict[str, Any]) -> bool:
        gh = eval_dict["global_held_out"]
        return (
            gh["bpb"] <= lm_target_bpb
            and gh["bpc"] <= lm_target_bpb
            and gh["ppl_byte"] <= lm_target_ppl
            and gh["ppl_char"] <= lm_target_ppl
        )

    def _record_lm_eval(ep_tag: int, train_stats: Optional[Dict[str, Any]], eval_res: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal lm_telemetry, last_lm_eval
        assert lm_splitter is not None
        last_lm_eval = eval_res
        gh = eval_res["global_held_out"]
        audit_lm = fep_engine.tester.audit_language_eval(gh, anti_leakage=lm_splitter.audit_report)
        cm = eval_res.get("canonical_eval_md_frozen")
        point = {
            "cycle": eval_res["cycle"],
            "flush_id": learning_buffer.flushes_completed,
            "episode": ep_tag,
            "total_train_bytes_seen": eval_res["total_train_bytes_seen"],
            "train_stats": train_stats,
            "global_bpb": gh["bpb"],
            "global_bpc": gh["bpc"],
            "global_ce_nats": gh["ce_nats"],
            "global_ppl_byte": gh["ppl_byte"],
            "global_ppl_char": gh["ppl_char"],
            "global_top1_acc_pct": gh["top1_acc_pct"],
            "global_ece": gh["next_byte_ece_10bin"],
            "global_neural_only_bpb": gh["neural_only_bpb"],
            "delta_vs_374_bpb": round(gh["bpb"] - 3.7393, 6),
            "markdown_bpb": eval_res["per_domain_held_out"].get("markdown", {}).get("bpb"),
            "rust_bpb": eval_res["per_domain_held_out"].get("rust", {}).get("bpb"),
            "python_bpb": eval_res["per_domain_held_out"].get("python", {}).get("bpb"),
            "canonical_eval_md_bpb": cm["bpb"] if cm else None,
            "max_prob_norm_error": gh["max_prob_norm_error"],
            "dale_violations": gh["dale_violations"],
            "audit": audit_lm,
            "target_met": _lm_target_met(eval_res),
        }
        lm_cyclic_history.append(point)
        if len(lm_cyclic_history) > 256:
            lm_cyclic_history[:] = lm_cyclic_history[-256:]
        lm_telemetry = {
            "architecture": (
                "AURA_V3_GENESIS_FEP_MARCUS_V3_INFINITE_BUFFER"
                if enable_aura_v3_genesis
                else "FEP_MARCUS_V3_INFINITE_BUFFER"
            ),
            "aura_v3_genesis_enabled": bool(enable_aura_v3_genesis),
            "seed": seed,
            "targets": {
                "bpb_max": lm_target_bpb,
                "bpc_max": lm_target_bpb,
                "ppl_max": lm_target_ppl,
                "baseline_ref_bpb": 3.7393,
            },
            "best_held_out_bpb": round(best_held_out_bpb, 6) if best_held_out_bpb < 90.0 else None,
            "best_held_out_cycle": best_held_out_cycle,
            "target_achieved": _lm_target_met(eval_res),
            "anti_leakage_audit": lm_splitter.audit_report,
            "baselines": lm_baselines,
            "latest_eval": eval_res,
            "cyclic_history": list(lm_cyclic_history),
            "last_audit": audit_lm,
        }
        atomic_write_json(INFINITE_LM_ARTIFACT, lm_telemetry)
        return point

    # Evaluación Ciclo 0 si arranque completamente nuevo
    if enable_lm_bpb and fep_engine.language_model is not None and lm_splitter is not None and lm_cycle_idx == 0 and not lm_cyclic_history:
        ev0 = fep_engine.language_model.evaluate_full_held_out_suite(
            lm_splitter,
            use_full_buffers=False,
            include_blind_routing=False,
            include_online_shift=False,
            generate_samples=False,
        )
        pt0 = _record_lm_eval(fep_engine.episodes_processed, None, ev0)
        best_held_out_bpb = min(best_held_out_bpb, float(pt0["global_bpb"]))
        print(
            f"[LM Ciclo #0 (Prenatal 0B repo) @ Ep {fep_engine.episodes_processed}] "
            f"BPB={pt0['global_bpb']:.4f} | BPC={pt0['global_bpc']:.4f} | PPL_b={pt0['global_ppl_byte']:.2f}",
            flush=True,
        )

    curriculum_queue: List[Dict[str, str]] = aa_evaluator.generate_targeted_curriculum(
        last_aa_result.get("weakest_domains_order", []),
        last_aa_result.get("failed_subtopics_by_domain", {}),
        count=curriculum_size,
    )

    episodes_this_run = 0
    flushes_this_run = 0
    last_flush_receipt: Optional[Dict[str, Any]] = None
    last_cycle: Dict[str, Any] = {
        "episode": fep_engine.episodes_processed,
        "f_final": 0.0,
        "elbo_final": 0.0,
        "s_final": 0.0,
        "dale_violations": 0,
    }

    # Tamaño del fragmento de texto `train` por episodio para que episodios y bytes llenen en sincronía
    bytes_per_ep_dom = max(256, (buffer_bytes + 3 * max(1, buffer_episodes) - 1) // (3 * max(1, buffer_episodes)))

    def _export_loop_state() -> Dict[str, Any]:
        return {
            "buffer_flushes_completed": learning_buffer.flushes_completed,
            "total_episodes_flushed": learning_buffer.total_episodes_flushed,
            "total_bytes_flushed": learning_buffer.total_bytes_flushed,
            "lm_cycle_idx": lm_cycle_idx,
            "best_held_out_bpb": round(best_held_out_bpb, 6),
            "best_held_out_cycle": best_held_out_cycle,
            "stream_idx": stream_idx,
            "turn_idx": turn_idx,
            "conv_idx": conv_idx,
            "dyn_idx": dyn_idx,
            "non_curr_count": non_curr_count,
            "lm_byte_cursors": dict(lm_byte_cursors),
        }

    def _write_live_status(status_str: str) -> Dict[str, Any]:
        elapsed = max(1e-3, time.perf_counter() - start_time)
        meta_sum = fep_engine.metacognition.summary()
        payload = {
            "status": status_str,
            "pid": pid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": round(elapsed, 2),
            "mode": "INFINITE_BUFFER_DRIVEN" if max_flushes <= 0 else f"BOUNDED_BUFFER_DRIVEN({max_flushes})",
            "aura_v3_genesis_enabled": bool(enable_aura_v3_genesis),
            "episodes_total": fep_engine.episodes_processed,
            "episodes_this_run": episodes_this_run,
            "flushes_this_run": flushes_this_run,
            "buffer": learning_buffer.fill_status(),
            "cognitive": {
                "mean_free_energy": round(fep_engine.cum_free_energy / max(1, fep_engine.episodes_processed), 4),
                "current_free_energy": last_cycle.get("f_final", 0.0),
                "current_surprise": last_cycle.get("s_final", 0.0),
                "sleep_cycles": fep_engine.sleep_cycles,
                "consolidated_prototypes": len(fep_engine.memory.proto_obs),
                "world_state_facts": len(fep_engine.memory.world_state_facts),
                "aa_passive_pct": last_aa_result.get("passive_accuracy_pct", 0.0),
                "aa_active_pct": last_aa_result.get("overall_accuracy_pct", 0.0),
                "meta_brier": meta_sum["meta_brier_score"],
                "compute_savings_pct": meta_sum["compute_savings_pct"],
                "microglia_pruned_pct": fep_engine.microglia_pruned_pct,
                "dale_violations": int(last_cycle.get("dale_violations", 0)),
            },
            "language_model": {
                "enabled": bool(enable_lm_bpb),
                "cycles_trained": fep_engine.language_model.cycles_trained if fep_engine.language_model else 0,
                "ingested_shards": (
                    sorted(fep_engine.language_model.ingested_shards)
                    if fep_engine.language_model
                    else []
                ),
                "total_train_bytes_seen": (
                    fep_engine.language_model.total_train_bytes_seen
                    if fep_engine.language_model
                    else 0
                ),
                "last_neural_train_bpb": (
                    round(fep_engine.language_model.last_neural_epoch_bpb, 4)
                    if fep_engine.language_model
                    else None
                ),
                "last_held_out_bpb": (
                    last_lm_eval["global_held_out"]["bpb"]
                    if last_lm_eval
                    else None
                ),
                "last_held_out_bpc": (
                    last_lm_eval["global_held_out"]["bpc"]
                    if last_lm_eval
                    else None
                ),
                "last_held_out_ppl_byte": (
                    last_lm_eval["global_held_out"]["ppl_byte"]
                    if last_lm_eval
                    else None
                ),
                "best_held_out_bpb": round(best_held_out_bpb, 6) if best_held_out_bpb < 90.0 else None,
                "best_held_out_cycle": best_held_out_cycle,
                "dale_violations": (
                    fep_engine.language_model._total_dale_violations()
                    if fep_engine.language_model
                    else 0
                ),
            },
            "last_flush_receipt": last_flush_receipt,
        }
        atomic_write_json(INFINITE_STATUS_ARTIFACT, payload)
        return payload

    def _flush_consolidate_and_save(override_reason: Optional[str] = None) -> Dict[str, Any]:
        nonlocal lm_cycle_idx, best_held_out_bpb, best_held_out_cycle
        nonlocal last_aa_result, curriculum_queue, flushes_this_run, last_flush_receipt

        t_flush0 = time.perf_counter()
        batch = learning_buffer.drain(override_reason=override_reason)
        flush_id = batch["flush_id"]
        flushes_this_run += 1

        # 1. Consolidación cognitiva SWR si corresponde al vaciar el buffer
        swr_delta_r = 0.0
        if batch["episodes_count"] > 0 and (fep_engine.episodes_processed % 64 != 0):
            swr_delta_r = fep_engine.execute_circadian_swr_sleep()

        # 2. Consolidación de lenguaje estrictamente sobre `train` (0% held-out)
        lm_train_stats: Optional[Dict[str, Any]] = None
        lm_eval_point: Optional[Dict[str, Any]] = None
        new_best_saved = False

        if enable_lm_bpb and fep_engine.language_model is not None and lm_splitter is not None:
            if lm_cycle_idx < lm_shards:
                lm_train_stats = fep_engine.language_model.train_cycle_shard(
                    lm_cycle_idx,
                    lm_splitter,
                    neural_samples=lm_neural_samples,
                    calibrate_on_train_cal=True,
                    total_cycles=lm_shards,
                )
                lm_cycle_idx += 1
            else:
                recal_friston = ((flush_id - lm_shards) % max(2, eval_every_flushes) == 2)
                lm_train_stats = fep_engine.language_model.consolidate_from_buffer(
                    batch["lm_bytes_by_domain"],
                    lm_splitter,
                    neural_samples=lm_neural_samples,
                    recalibrate_friston=recal_friston,
                )
                lm_cycle_idx = fep_engine.language_model.cycles_trained

            # 3. Evaluación periódica congelada (AA 300 preguntas + Held-Out BPB/BPC/CE/PPL)
            should_audit = (
                override_reason is None
                and (
                    flush_id in (1, 4, lm_shards)
                    or (flush_id > lm_shards and (flush_id - lm_shards) % max(1, eval_every_flushes) == 0)
                )
            )
            if should_audit:
                last_aa_result = aa_evaluator.run_benchmark_evaluation(
                    fep_engine, orchestrator=orchestrator, current_episode=fep_engine.episodes_processed
                )
                curriculum_queue = aa_evaluator.generate_targeted_curriculum(
                    last_aa_result.get("weakest_domains_order", []),
                    last_aa_result.get("failed_subtopics_by_domain", {}),
                    count=curriculum_size,
                )
                include_blind = (
                    flush_id == lm_shards
                    or (flush_id > lm_shards and (flush_id - lm_shards) % max(2, eval_every_flushes * 2) == 0)
                )
                ev_c = fep_engine.language_model.evaluate_full_held_out_suite(
                    lm_splitter,
                    use_full_buffers=False,
                    include_blind_routing=include_blind,
                    include_online_shift=False,
                    generate_samples=False,
                )
                lm_eval_point = _record_lm_eval(fep_engine.episodes_processed, lm_train_stats, ev_c)
                cur_bpb = float(lm_eval_point["global_bpb"])
                if cur_bpb < best_held_out_bpb:
                    best_held_out_bpb = cur_bpb
                    best_held_out_cycle = int(lm_eval_point["cycle"])
                    if lm_telemetry is not None:
                        lm_telemetry["best_held_out_bpb"] = round(best_held_out_bpb, 6)
                        lm_telemetry["best_held_out_cycle"] = best_held_out_cycle
                        atomic_write_json(INFINITE_LM_ARTIFACT, lm_telemetry)
                    new_best_saved = True
        elif override_reason is None and (flush_id == 1 or flush_id % max(1, eval_every_flushes) == 0):
            last_aa_result = aa_evaluator.run_benchmark_evaluation(
                fep_engine, orchestrator=orchestrator, current_episode=fep_engine.episodes_processed
            )
            curriculum_queue = aa_evaluator.generate_targeted_curriculum(
                last_aa_result.get("weakest_domains_order", []),
                last_aa_result.get("failed_subtopics_by_domain", {}),
                count=curriculum_size,
            )

        # 4. GUARDADO ATÓMICO OBLIGATORIO AL LLENARSE EL BUFFER (Cognitivo + Hopfield + WorldState + LM)
        loop_st = _export_loop_state()
        cog_receipt = fep_engine.save_checkpoint(
            float(last_cycle.get("f_final", 0.0)),
            float(last_cycle.get("elbo_final", 0.0)),
            float(last_cycle.get("s_final", 0.0)),
            orchestrator_state=orchestrator.export_state(),
            loop_state=loop_st,
        )
        lm_receipt: Optional[Dict[str, Any]] = None
        if enable_lm_bpb and fep_engine.language_model is not None:
            lm_extra = {
                "buffer_flushes_completed": learning_buffer.flushes_completed,
                "best_held_out_bpb": round(best_held_out_bpb, 6) if best_held_out_bpb < 90.0 else None,
                "best_held_out_cycle": best_held_out_cycle,
                "episodes_processed": fep_engine.episodes_processed,
            }
            lm_receipt = fep_engine.language_model.save_learned_state(
                CHECKPOINT_DIR, tag="latest", extra_meta=lm_extra
            )
            if new_best_saved:
                fep_engine.language_model.save_learned_state(
                    CHECKPOINT_DIR, tag="best", extra_meta=lm_extra
                )
                w_src = os.path.join(CHECKPOINT_DIR, "weights_marcus_fep_7b.bin")
                w_best = os.path.join(CHECKPOINT_DIR, "best_weights_marcus_fep_7b.bin")
                if os.path.exists(w_src):
                    shutil.copy2(w_src, w_best)

        # 5. Registro en ledger auditable de flushes + actualización viva de telemetría y reporte
        total_save_ms = round(
            cog_receipt["save_ms"] + (lm_receipt["save_ms"] if lm_receipt else 0.0), 2
        )
        mean_f_batch = (
            sum(e["f_final"] for e in batch["episodes"]) / len(batch["episodes"])
            if batch["episodes"]
            else float(last_cycle.get("f_final", 0.0))
        )
        meta_sum = fep_engine.metacognition.summary()
        receipt = {
            "flush_id": flush_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": batch["trigger"],
            "episode": fep_engine.episodes_processed,
            "episodes_drained": batch["episodes_count"],
            "bytes_drained": batch["bytes_count"],
            "mean_f_batch": round(mean_f_batch, 4),
            "swr_delta_r": round(float(swr_delta_r), 5),
            "aa_passive_pct": last_aa_result.get("passive_accuracy_pct", 0.0),
            "aa_active_pct": last_aa_result.get("overall_accuracy_pct", 0.0),
            "compute_savings_pct": meta_sum["compute_savings_pct"],
            "meta_brier": meta_sum["meta_brier_score"],
            "lm_phase": lm_train_stats.get("phase") if lm_train_stats else None,
            "lm_cycle": fep_engine.language_model.cycles_trained if fep_engine.language_model else 0,
            "lm_neural_train_bpb": lm_train_stats.get("neural_train_bpb") if lm_train_stats else None,
            "lm_held_out_bpb": lm_eval_point["global_bpb"] if lm_eval_point else (
                last_lm_eval["global_held_out"]["bpb"] if last_lm_eval else None
            ),
            "lm_held_out_ppl_byte": lm_eval_point["global_ppl_byte"] if lm_eval_point else (
                last_lm_eval["global_held_out"]["ppl_byte"] if last_lm_eval else None
            ),
            "best_held_out_bpb": round(best_held_out_bpb, 6) if best_held_out_bpb < 90.0 else None,
            "new_best_snapshot": new_best_saved,
            "dale_violations_total": (
                cog_receipt["dale_violations"]
                + (lm_receipt["dale_violations"] if lm_receipt else 0)
            ),
            "weights_sha256": cog_receipt["weights_sha256"],
            "world_state_sha256": cog_receipt["world_state_sha256"],
            "lm_npz_sha256": lm_receipt["npz_sha256"] if lm_receipt else None,
            "lm_npz_size_mb": round(lm_receipt["npz_size_bytes"] / (1024.0 * 1024.0), 2) if lm_receipt else 0.0,
            "save_ms": total_save_ms,
            "flush_total_seconds": round(time.perf_counter() - t_flush0, 2),
        }
        last_flush_receipt = receipt
        BoundedLearningBuffer.append_flush_ledger(FLUSH_LEDGER_PATH, receipt)

        status_label = (
            "🟢 Entrenamiento Infinito Activo (Guardado Automático por Buffer)"
            if override_reason is None and max_flushes <= 0
            else "🟡 Guardado Atómico por Buffer"
        )
        telem = _build_telemetry_payload(
            start_time,
            "RUNNING_INFINITE_BUFFER" if override_reason is None else "FLUSH_SAVED",
            last_cycle,
            fep_engine,
            orchestrator,
            last_aa_result,
            episodes_this_run,
            lm_telemetry=lm_telemetry,
        )
        telem["buffer_training"] = {
            "fill_status": learning_buffer.fill_status(),
            "last_flush_receipt": receipt,
        }
        atomic_write_json(TELEMETRY_ARTIFACT, telem)
        _update_markdown_report(status_label, fep_engine, orchestrator, last_aa_result, lm_telemetry=lm_telemetry)
        _write_live_status("RUNNING_INFINITE_BUFFER" if override_reason is None else "SAVED_ON_SHUTDOWN")

        bpb_txt = (
            f"BPB={receipt['lm_held_out_bpb']:.4f} (best={receipt['best_held_out_bpb']:.4f})"
            if receipt["lm_held_out_bpb"] is not None
            else f"NeuralTr={receipt['lm_neural_train_bpb']}"
        )
        msg = (
            f"[BUFFER FULL -> GUARDADO #{flush_id:04d} ({receipt['trigger']})] "
            f"Ep={receipt['episode']} | Drenado={receipt['episodes_drained']} eps, {receipt['bytes_drained']/1024.0:.1f} KiB | "
            f"FaseLM={receipt['lm_phase']} (c={receipt['lm_cycle']}) | {bpb_txt} | "
            f"AA={receipt['aa_passive_pct']:.1f}%/{receipt['aa_active_pct']:.1f}% | "
            f"F_lote={receipt['mean_f_batch']:.2f} | Sav={receipt['compute_savings_pct']:.1f}% | "
            f"SHA[W={receipt['weights_sha256'][:8]}, LM={str(receipt['lm_npz_sha256'])[:8]}] "
            f"({receipt['save_ms']:.1f}ms, DaleViol={receipt['dale_violations_total']})"
            + (" ★NEW_BEST" if new_best_saved else "")
        )
        print(msg, flush=True)
        append_bounded_log(LOG_FILE, msg)
        return receipt

    _write_live_status("RUNNING_INFINITE_BUFFER")

    try:
        while not stop_requested:
            if max_flushes > 0 and flushes_this_run >= max_flushes:
                break

            next_ep = fep_engine.episodes_processed + 1
            ep_meta: Optional[Dict[str, Any]] = None
            subtopic = ""

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

            # 1. Insertar episodio aprendido en el buffer acotado
            learning_buffer.push_episode({
                "episode": last_cycle["episode"],
                "domain": dom,
                "f_init": last_cycle["f_init"],
                "f_final": last_cycle["f_final"],
                "s_init": last_cycle["s_init"],
                "s_final": last_cycle["s_final"],
                "consulted_agent": last_cycle.get("consulted_agent"),
            })

            # 2. Alimentar el buffer LM con bloques rodantes estrictamente de `train` (0% held-out)
            if enable_lm_bpb and lm_splitter is not None:
                for lm_dom in AuraFepCorticalLanguageModel.DOMAINS:
                    chunk, new_cur = _extract_rolling_train_chunk(
                        train_byte_pools[lm_dom], lm_byte_cursors[lm_dom], bytes_per_ep_dom
                    )
                    lm_byte_cursors[lm_dom] = new_cur
                    learning_buffer.push_lm_bytes(lm_dom, chunk)

            # 3. Cada vez que el buffer se llena -> consolidar, guardar atómicamente y vaciar RAM
            if learning_buffer.is_full():
                _flush_consolidate_and_save()
            elif episodes_this_run % 16 == 0:
                _write_live_status("RUNNING_INFINITE_BUFFER")

            if sleep_delay > 0.0:
                time.sleep(sleep_delay)
    finally:
        # Si se interrumpe con datos pendientes en el buffer, vaciar y guardar para no perder nada
        if not learning_buffer.is_empty():
            _flush_consolidate_and_save(override_reason="shutdown_partial")
        final_status = "STOPPED_CLEANLY" if stop_requested else "COMPLETED_BOUNDED_FLUSHES"
        final_payload = _write_live_status(final_status)
        if os.path.exists(PID_FILE_PATH):
            try:
                os.remove(PID_FILE_PATH)
            except OSError:
                pass
        print(
            f"[*] Arnés finalizado limpiamente ({final_status}) | "
            f"Episodios={fep_engine.episodes_processed} | Flushes guardados={learning_buffer.flushes_completed}",
            flush=True,
        )

    return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento Infinito Impulsado por Buffer con Guardado Atómico (AURA v3 Genesis + FEP-MARCUS v3)"
    )
    parser.add_argument("--buffer-episodes", type=int, default=64, help="Capacidad máxima de episodios del buffer antes de consolidar y guardar")
    parser.add_argument("--buffer-bytes", type=int, default=524288, help="Capacidad máxima en bytes de texto train del buffer antes de consolidar y guardar")
    parser.add_argument("--max-flushes", type=int, default=0, help="Límite de flushes de buffer en esta corrida (0 = INFINITO)")
    parser.add_argument("--eval-every-flushes", type=int, default=4, help="Frecuencia de auditoría AA + Held-Out BPB congelada tras ingerir los shards")
    parser.add_argument("--curriculum-size", type=int, default=48, help="Ejercicios epistémicos por ronda AA")
    parser.add_argument("--sleep-delay", type=float, default=0.0, help="Pausa opcional entre episodios (segundos)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla determinista")
    parser.add_argument("--reset-state", action="store_true", help="Reinicia checkpoints, buffer ledger y pesos desde Ep 0")
    parser.add_argument("--no-aura-v3-genesis", action="store_true", help="Desactiva AURA v3 Genesis (por defecto ACTIVADO)")
    parser.add_argument("--disable-lm-bpb", action="store_true", help="Desactiva el subsistema LM de bytes")
    parser.add_argument("--lm-shards", type=int, default=8, help="Número de shards disjuntos del corpus train (8 por paridad)")
    parser.add_argument("--lm-eval-bytes-per-domain", type=int, default=16384, help="Bytes held-out por modalidad en auditoría congelada")
    parser.add_argument("--lm-neural-samples", type=int, default=65536, help="Muestras por época de consolidación neural Dale en cada flush")
    args = parser.parse_args()

    run_infinite_buffer_training(
        buffer_episodes=args.buffer_episodes,
        buffer_bytes=args.buffer_bytes,
        max_flushes=args.max_flushes,
        eval_every_flushes=args.eval_every_flushes,
        curriculum_size=args.curriculum_size,
        sleep_delay=args.sleep_delay,
        seed=args.seed,
        reset_state=args.reset_state,
        enable_aura_v3_genesis=not args.no_aura_v3_genesis,
        enable_lm_bpb=not args.disable_lm_bpb,
        lm_shards=args.lm_shards,
        lm_eval_bytes_per_domain=args.lm_eval_bytes_per_domain,
        lm_neural_samples=args.lm_neural_samples,
    )
