//! Entrenamiento sintético reanudable del motor unificado.

use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::matrix_free_cognitive_substrate::LatentConceptId;
use cdt_rqm_epr::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use cdt_rqm_epr::relational_field::ObserverId;
use cdt_rqm_epr::symmetry_guided_rqm_epr::{RqmPhaseRelationState, RqmRelationKey};
use cdt_rqm_epr::unified_spin_cognitive_engine::{
    ConsolidatedKnowledge, KnowledgeKey, UnifiedSpinCognitiveConfig, UnifiedSpinCognitiveEngine,
};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const VERSION: u32 = 1;
const DEFAULT_ROOT: &str = "data/unified_infinite_training";
const CORE_OBSERVER: ObserverId = ObserverId(998_100);
const ORBIT_OBSERVER: ObserverId = ObserverId(998_101);

#[derive(Clone, Debug)]
struct TrainerConfig {
    duration: Option<Duration>,
    batch_size: usize,
    validate_every: u64,
    checkpoint_every: u64,
    milestone_every: u64,
    homeostasis_cooling_steps: usize,
    max_batches: Option<u64>,
    root: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RelationSnapshot {
    observer: usize,
    source: usize,
    target: usize,
    amplitude: f64,
    phase: f64,
    coherence: f64,
    uncertainty: f64,
    eligibility: f64,
    prediction_error: f64,
    exposures: u64,
    consolidated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct KnowledgeSnapshot {
    observer: usize,
    source: usize,
    target: usize,
    confidence: f64,
    topological_symmetry: f64,
    spin_entropy: f64,
    prediction_error: f64,
    consolidations: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainingCheckpoint {
    version: u32,
    batch: u64,
    total_examples: u64,
    rng_state: u64,
    first_emergent_batch: Option<u64>,
    sustained_emergent_validations: u64,
    amplitudes: Vec<[f64; 2]>,
    relations: Vec<RelationSnapshot>,
    epr_state: String,
    knowledge: Vec<KnowledgeSnapshot>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct ValidationMetrics {
    batch: u64,
    total_examples: u64,
    composition_accuracy: f64,
    direct_composed_relation_absence: f64,
    orbit_accuracy: f64,
    ood_abstention: f64,
    topological_symmetry: f64,
    spin_entropy: f64,
    entangled_edges: usize,
    knowledge: usize,
    relations: usize,
    epr_links: usize,
    emergent_cognition_gate: bool,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct TrainedLegacyEvaluation {
    checkpoint_batch: u64,
    checkpoint_examples: u64,
    unified_direct_accuracy: f64,
    legacy_direct_accuracy: f64,
    unified_composition_accuracy: f64,
    legacy_composition_accuracy: f64,
    unified_orbit_accuracy: f64,
    legacy_orbit_accuracy: f64,
    unified_ood_abstention: f64,
    legacy_ood_abstention: f64,
    unified_core_knowledge_coverage: f64,
    unified_knowledge_selectivity: f64,
    unified_knowledge: usize,
    unified_relations: usize,
    legacy_relations: usize,
    unified_epr_links: usize,
    legacy_epr_links: usize,
    unified_query_ms: f64,
    legacy_query_ms: f64,
    latency_ratio: f64,
    unified_quantum_entropy: f64,
    unified_entangled_edges: usize,
    unified_topological_symmetry: f64,
    decision: &'static str,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrainerConfig::from_env();
    fs::create_dir_all(config.root.join("checkpoints"))?;
    let latest_path = config.root.join("latest.json");
    let (mut engine, mut state) = if latest_path.exists() {
        let checkpoint: TrainingCheckpoint = serde_json::from_slice(&fs::read(&latest_path)?)?;
        let mut engine = fresh_engine()?;
        restore_checkpoint(&mut engine, &checkpoint)?;
        println!(
            "resume=true batch={} examples={} relations={} knowledge={}",
            checkpoint.batch,
            checkpoint.total_examples,
            checkpoint.relations.len(),
            checkpoint.knowledge.len()
        );
        (engine, checkpoint)
    } else {
        println!("resume=false fresh=true");
        (
            fresh_engine()?,
            TrainingCheckpoint {
                version: VERSION,
                batch: 0,
                total_examples: 0,
                rng_state: 0x1A5A_C06E_2026_0718,
                first_emergent_batch: None,
                sustained_emergent_validations: 0,
                amplitudes: Vec::new(),
                relations: Vec::new(),
                epr_state: String::new(),
                knowledge: Vec::new(),
            },
        )
    };
    if env::args().any(|argument| argument == "--evaluate-legacy") {
        let evaluation =
            evaluate_trained_against_legacy(&engine, state.batch, state.total_examples);
        fs::write(
            config.root.join("legacy_comparison.json"),
            serde_json::to_vec_pretty(&evaluation)?,
        )?;
        println!("{}", serde_json::to_string_pretty(&evaluation)?);
        return Ok(());
    }
    let mut rng = SplitMix64::new(state.rng_state);
    let started = Instant::now();
    let metrics_path = config.root.join("metrics.jsonl");
    let mut last_metrics = ValidationMetrics::default();

    loop {
        state.batch += 1;
        train_batch(&mut engine, &mut rng, config.batch_size);
        state.total_examples += (config.batch_size + 72) as u64;

        if state.batch % config.validate_every == 0 {
            last_metrics = validate(&engine, state.batch, state.total_examples);
            if !last_metrics.emergent_cognition_gate
                && last_metrics.composition_accuracy >= 0.95
                && last_metrics.orbit_accuracy >= 0.95
            {
                engine.spin_liquid.cool(config.homeostasis_cooling_steps);
                last_metrics = validate(&engine, state.batch, state.total_examples);
                println!(
                    "event=quantum_homeostasis batch={} cooling_steps={} entangled_edges={} gate={}",
                    state.batch,
                    config.homeostasis_cooling_steps,
                    last_metrics.entangled_edges,
                    last_metrics.emergent_cognition_gate
                );
            }
            if last_metrics.emergent_cognition_gate {
                if state.first_emergent_batch.is_none() {
                    state.first_emergent_batch = Some(state.batch);
                    println!("event=first_emergent_cognition batch={}", state.batch);
                }
                state.sustained_emergent_validations += 1;
            } else {
                state.sustained_emergent_validations = 0;
            }
            append_jsonl(&metrics_path, &last_metrics)?;
            println!(
                "batch={} examples={} composition={:.3} orbit={:.3} ood={:.3} direct_absence={:.3} knowledge={} relations={} epr={} entropy={:.4} entangled_edges={} gate={} sustained={}",
                last_metrics.batch,
                last_metrics.total_examples,
                last_metrics.composition_accuracy,
                last_metrics.orbit_accuracy,
                last_metrics.ood_abstention,
                last_metrics.direct_composed_relation_absence,
                last_metrics.knowledge,
                last_metrics.relations,
                last_metrics.epr_links,
                last_metrics.spin_entropy,
                last_metrics.entangled_edges,
                last_metrics.emergent_cognition_gate,
                state.sustained_emergent_validations,
            );
        }

        if state.batch % config.checkpoint_every == 0
            || state.first_emergent_batch == Some(state.batch)
        {
            state.rng_state = rng.state;
            capture_checkpoint(&engine, &mut state);
            save_checkpoint(&config.root, &state, false)?;
        }
        if state.batch % config.milestone_every == 0 {
            state.rng_state = rng.state;
            capture_checkpoint(&engine, &mut state);
            save_checkpoint(&config.root, &state, true)?;
        }

        let reached_time = config
            .duration
            .is_some_and(|duration| started.elapsed() >= duration);
        let reached_batches = config
            .max_batches
            .is_some_and(|maximum| state.batch >= maximum);
        if reached_time || reached_batches {
            break;
        }
    }

    state.rng_state = rng.state;
    capture_checkpoint(&engine, &mut state);
    save_checkpoint(&config.root, &state, true)?;
    fs::write(
        config.root.join("summary.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "version": VERSION,
            "batch": state.batch,
            "total_examples": state.total_examples,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "first_emergent_batch": state.first_emergent_batch,
            "sustained_emergent_validations": state.sustained_emergent_validations,
            "last_metrics": last_metrics,
            "decision": if last_metrics.emergent_cognition_gate {
                "EMERGENT_COGNITION_SUSTAINED"
            } else {
                "NEEDS_MORE_TRAINING"
            }
        }))?,
    )?;
    println!(
        "finished=true batch={} examples={} elapsed_s={:.3} first_emergent={:?} sustained={} decision={}",
        state.batch,
        state.total_examples,
        started.elapsed().as_secs_f64(),
        state.first_emergent_batch,
        state.sustained_emergent_validations,
        if last_metrics.emergent_cognition_gate {
            "EMERGENT_COGNITION_SUSTAINED"
        } else {
            "NEEDS_MORE_TRAINING"
        }
    );
    Ok(())
}

impl TrainerConfig {
    fn from_env() -> Self {
        let hours = env_f64("UNIFIED_TRAIN_HOURS", 5.0);
        Self {
            duration: (hours > 0.0).then(|| Duration::from_secs_f64(hours * 3_600.0)),
            batch_size: env_usize("UNIFIED_TRAIN_BATCH_SIZE", 128).max(1),
            validate_every: env_u64("UNIFIED_TRAIN_VALIDATE_EVERY", 10).max(1),
            checkpoint_every: env_u64("UNIFIED_TRAIN_CHECKPOINT_EVERY", 50).max(1),
            milestone_every: env_u64("UNIFIED_TRAIN_MILESTONE_EVERY", 500).max(1),
            homeostasis_cooling_steps: env_usize("UNIFIED_TRAIN_HOMEOSTASIS_COOLING", 120).max(1),
            max_batches: env::var("UNIFIED_TRAIN_MAX_BATCHES")
                .ok()
                .and_then(|value| value.parse().ok()),
            root: PathBuf::from(
                env::var("UNIFIED_TRAIN_ROOT").unwrap_or_else(|_| DEFAULT_ROOT.to_string()),
            ),
        }
    }
}

fn fresh_engine() -> Result<UnifiedSpinCognitiveEngine, Box<dyn std::error::Error>> {
    Ok(UnifiedSpinCognitiveEngine::periodic_pyrochlore(
        2,
        1,
        1,
        UnifiedSpinCognitiveConfig {
            bootstrap_cooling_steps: 180,
            cooling_steps_per_observation: 0,
            real_steps_per_observation: 0,
            ..UnifiedSpinCognitiveConfig::default()
        },
    )?)
}

fn train_batch(
    engine: &mut UnifiedSpinCognitiveEngine,
    rng: &mut SplitMix64,
    random_examples: usize,
) {
    for task in 0..32 {
        let base = task * 3;
        let phase = task as f64 * 0.071;
        engine.train_relation(
            CORE_OBSERVER,
            LatentConceptId(base),
            LatentConceptId(base + 1),
            phase,
            1.0,
            0.0,
            &[],
            1,
        );
        engine.train_relation(
            CORE_OBSERVER,
            LatentConceptId(base + 1),
            LatentConceptId(base + 2),
            phase,
            1.0,
            0.0,
            &[],
            1,
        );
    }
    for task in 0..8 {
        let base = 100 + task * 4;
        let orbit = [(LatentConceptId(base + 2), LatentConceptId(base + 3))];
        engine.train_relation(
            ORBIT_OBSERVER,
            LatentConceptId(base),
            LatentConceptId(base + 1),
            task as f64 * 0.093,
            1.0,
            1.0,
            &orbit,
            1,
        );
    }
    for _ in 0..random_examples {
        let source = 200 + rng.index(56);
        let mut target = 200 + rng.index(56);
        if target == source {
            target = 200 + (target + 1 - 200) % 56;
        }
        engine.train_relation(
            ObserverId(998_200 + rng.index(4)),
            LatentConceptId(source),
            LatentConceptId(target),
            rng.unit() * std::f64::consts::TAU,
            0.35 + 0.65 * rng.unit(),
            0.0,
            &[],
            1,
        );
    }
}

fn validate(
    engine: &UnifiedSpinCognitiveEngine,
    batch: u64,
    total_examples: u64,
) -> ValidationMetrics {
    let mut composition = 0;
    let mut direct_absence = 0;
    for task in 0..32 {
        let base = task * 3;
        let phase = task as f64 * 0.071;
        composition += usize::from(
            engine
                .infer(CORE_OBSERVER, LatentConceptId(base), phase, 2)
                .is_some_and(|inference| inference.path.last() == Some(&LatentConceptId(base + 2))),
        );
        direct_absence += usize::from(
            engine
                .cognition
                .workspace
                .relation(
                    CORE_OBSERVER,
                    LatentConceptId(base),
                    LatentConceptId(base + 2),
                )
                .is_none(),
        );
    }
    let mut orbit = 0;
    for task in 0..8 {
        let base = 100 + task * 4;
        orbit += usize::from(
            engine
                .infer(
                    ORBIT_OBSERVER,
                    LatentConceptId(base + 2),
                    task as f64 * 0.093,
                    1,
                )
                .is_some_and(|inference| inference.path.last() == Some(&LatentConceptId(base + 3))),
        );
    }
    let ood_abstention = usize::from(
        engine
            .infer(CORE_OBSERVER, LatentConceptId(999), 0.0, 3)
            .is_none(),
    ) as f64;
    let report = engine.report();
    let composition_accuracy = composition as f64 / 32.0;
    let direct_composed_relation_absence = direct_absence as f64 / 32.0;
    let orbit_accuracy = orbit as f64 / 8.0;
    let emergent_cognition_gate = composition_accuracy >= 0.95
        && direct_composed_relation_absence == 1.0
        && orbit_accuracy >= 0.95
        && ood_abstention == 1.0
        && report.topological_symmetry >= 0.99
        && report.quantum.entangled_edges > 0;
    ValidationMetrics {
        batch,
        total_examples,
        composition_accuracy,
        direct_composed_relation_absence,
        orbit_accuracy,
        ood_abstention,
        topological_symmetry: report.topological_symmetry,
        spin_entropy: report.quantum.mean_single_spin_entropy,
        entangled_edges: report.quantum.entangled_edges,
        knowledge: report.consolidated_knowledge,
        relations: report.rqm_relations,
        epr_links: report.epr_links,
        emergent_cognition_gate,
    }
}

fn evaluate_trained_against_legacy(
    engine: &UnifiedSpinCognitiveEngine,
    checkpoint_batch: u64,
    checkpoint_examples: u64,
) -> TrainedLegacyEvaluation {
    let mut legacy = legacy_fixture();
    for task in 0..32 {
        let base = task * 3;
        let phase = task as f32 * 0.071;
        train_legacy(&mut legacy, CORE_OBSERVER, phase, base, base + 1);
        train_legacy(&mut legacy, CORE_OBSERVER, phase, base + 1, base + 2);
    }
    for task in 0..8 {
        let base = 100 + task * 4;
        train_legacy(
            &mut legacy,
            ORBIT_OBSERVER,
            task as f32 * 0.093,
            base,
            base + 1,
        );
    }

    let mut unified_direct = 0;
    let mut legacy_direct = 0;
    let mut unified_composition = 0;
    let mut legacy_composition = 0;
    for task in 0..32 {
        let base = task * 3;
        let phase = task as f64 * 0.071;
        unified_direct += usize::from(
            engine
                .cognition
                .workspace
                .query(CORE_OBSERVER, LatentConceptId(base), phase)
                .first()
                .is_some_and(|candidate| candidate.concept == LatentConceptId(base + 1)),
        );
        unified_composition += usize::from(
            engine
                .infer(CORE_OBSERVER, LatentConceptId(base), phase, 2)
                .is_some_and(|inference| inference.path.last() == Some(&LatentConceptId(base + 2))),
        );
        legacy_direct += usize::from(legacy_has_candidate(
            &mut legacy,
            CORE_OBSERVER,
            phase as f32,
            base,
            base + 1,
        ));
        let middle = legacy_top_candidate(&mut legacy, CORE_OBSERVER, phase as f32, base);
        let end = middle.and_then(|middle| {
            legacy_top_candidate(&mut legacy, CORE_OBSERVER, phase as f32, middle)
        });
        legacy_composition += usize::from(middle == Some(base + 1) && end == Some(base + 2));
    }

    let mut unified_orbit = 0;
    let mut legacy_orbit = 0;
    for task in 0..8 {
        let base = 100 + task * 4;
        let phase = task as f64 * 0.093;
        unified_orbit += usize::from(
            engine
                .infer(ORBIT_OBSERVER, LatentConceptId(base + 2), phase, 1)
                .is_some_and(|inference| inference.path.last() == Some(&LatentConceptId(base + 3))),
        );
        legacy_orbit += usize::from(legacy_has_candidate(
            &mut legacy,
            ORBIT_OBSERVER,
            phase as f32,
            base + 2,
            base + 3,
        ));
    }

    let unified_ood = usize::from(
        engine
            .infer(CORE_OBSERVER, LatentConceptId(999), 0.0, 2)
            .is_none(),
    ) as f64;
    let legacy_ood = usize::from(
        legacy
            .query(CORE_OBSERVER, 0.0, &[255])
            .candidates
            .is_empty(),
    ) as f64;
    let mut covered = 0;
    for task in 0..32 {
        let base = task * 3;
        for (source, target) in [(base, base + 1), (base + 1, base + 2)] {
            covered += usize::from(engine.knowledge.contains_key(&KnowledgeKey {
                observer: CORE_OBSERVER.0,
                source: LatentConceptId(source),
                target: LatentConceptId(target),
            }));
        }
    }
    for task in 0..8 {
        let base = 100 + task * 4;
        for (source, target) in [(base, base + 1), (base + 2, base + 3)] {
            covered += usize::from(engine.knowledge.contains_key(&KnowledgeKey {
                observer: ORBIT_OBSERVER.0,
                source: LatentConceptId(source),
                target: LatentConceptId(target),
            }));
        }
    }

    let repeats = 100;
    let new_started = Instant::now();
    for _ in 0..repeats {
        for task in 0..32 {
            let base = task * 3;
            let _ = engine.infer(CORE_OBSERVER, LatentConceptId(base), task as f64 * 0.071, 2);
        }
    }
    let unified_query_ms = new_started.elapsed().as_secs_f64() * 1_000.0;
    let legacy_started = Instant::now();
    for _ in 0..repeats {
        for task in 0..32 {
            let base = task * 3;
            let phase = task as f32 * 0.071;
            if let Some(middle) = legacy_top_candidate(&mut legacy, CORE_OBSERVER, phase, base) {
                let _ = legacy_top_candidate(&mut legacy, CORE_OBSERVER, phase, middle);
            }
        }
    }
    let legacy_query_ms = legacy_started.elapsed().as_secs_f64() * 1_000.0;
    let report = engine.report();
    let selectivity = report.consolidated_knowledge as f64 / report.rqm_relations.max(1) as f64;
    TrainedLegacyEvaluation {
        checkpoint_batch,
        checkpoint_examples,
        unified_direct_accuracy: unified_direct as f64 / 32.0,
        legacy_direct_accuracy: legacy_direct as f64 / 32.0,
        unified_composition_accuracy: unified_composition as f64 / 32.0,
        legacy_composition_accuracy: legacy_composition as f64 / 32.0,
        unified_orbit_accuracy: unified_orbit as f64 / 8.0,
        legacy_orbit_accuracy: legacy_orbit as f64 / 8.0,
        unified_ood_abstention: unified_ood,
        legacy_ood_abstention: legacy_ood,
        unified_core_knowledge_coverage: covered as f64 / 80.0,
        unified_knowledge_selectivity: selectivity,
        unified_knowledge: report.consolidated_knowledge,
        unified_relations: report.rqm_relations,
        legacy_relations: legacy.relation_count(),
        unified_epr_links: report.epr_links,
        legacy_epr_links: legacy.entanglement.active_count(),
        unified_query_ms,
        legacy_query_ms,
        latency_ratio: unified_query_ms / legacy_query_ms.max(f64::EPSILON),
        unified_quantum_entropy: report.quantum.mean_single_spin_entropy,
        unified_entangled_edges: report.quantum.entangled_edges,
        unified_topological_symmetry: report.topological_symmetry,
        decision: if unified_orbit == 8 && legacy_orbit == 0 && selectivity >= 0.95 {
            "FUNCTIONALLY_SUPERIOR_BUT_OVERCONSOLIDATED_AND_SLOWER"
        } else {
            "NEEDS_FURTHER_VALIDATION"
        },
    }
}

fn legacy_fixture() -> NativeThermoRqmEprSubstrate {
    NativeThermoRqmEprSubstrate::new(
        NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: 256,
            temperature: 0.0,
            ..NativeThermoCdtConfig::default()
        },
        NativeThermoRqmConfig {
            thermal_steps_per_train: 0,
            thermal_steps_per_query: 0,
            collect_query_diagnostics: false,
            ..NativeThermoRqmConfig::default()
        },
        EntanglementConfig {
            create_threshold: 0.75,
            ..EntanglementConfig::default()
        },
    )
}

fn train_legacy(
    legacy: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    phase: f32,
    source: usize,
    target: usize,
) {
    for _ in 0..24 {
        legacy.train_observed_transition(observer, phase, &[source], &[target], 1.0);
    }
}

fn legacy_has_candidate(
    legacy: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    phase: f32,
    source: usize,
    target: usize,
) -> bool {
    legacy
        .query(observer, phase, &[source])
        .candidates
        .iter()
        .any(|candidate| candidate.agent == target)
}

fn legacy_top_candidate(
    legacy: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    phase: f32,
    source: usize,
) -> Option<usize> {
    legacy
        .query(observer, phase, &[source])
        .candidates
        .first()
        .map(|candidate| candidate.agent)
}

fn capture_checkpoint(engine: &UnifiedSpinCognitiveEngine, checkpoint: &mut TrainingCheckpoint) {
    checkpoint.amplitudes = engine
        .spin_liquid
        .amplitudes()
        .iter()
        .map(|amplitude| [amplitude.re, amplitude.im])
        .collect();
    checkpoint.relations = engine
        .cognition
        .workspace
        .relation_entries()
        .map(|(key, state)| RelationSnapshot::from_parts(key, state))
        .collect();
    checkpoint.epr_state = engine
        .cognition
        .workspace
        .entanglement
        .serialize_persistent_state();
    checkpoint.knowledge = engine
        .knowledge
        .values()
        .copied()
        .map(KnowledgeSnapshot::from)
        .collect();
}

fn restore_checkpoint(
    engine: &mut UnifiedSpinCognitiveEngine,
    checkpoint: &TrainingCheckpoint,
) -> Result<(), Box<dyn std::error::Error>> {
    if checkpoint.version != VERSION {
        return Err(format!(
            "checkpoint version {} incompatible con {}",
            checkpoint.version, VERSION
        )
        .into());
    }
    let amplitudes = checkpoint
        .amplitudes
        .iter()
        .map(|value| Complex64::new(value[0], value[1]))
        .collect::<Vec<_>>();
    if !amplitudes.is_empty() {
        engine.spin_liquid.set_amplitudes(&amplitudes)?;
    }
    for relation in &checkpoint.relations {
        engine
            .cognition
            .workspace
            .import_relation(relation.key(), relation.state());
    }
    if !checkpoint.epr_state.is_empty() {
        engine
            .cognition
            .workspace
            .entanglement
            .apply_persistent_state(&checkpoint.epr_state)
            .map_err(io::Error::other)?;
    }
    for knowledge in &checkpoint.knowledge {
        let value = knowledge.value();
        engine.knowledge.insert(value.key, value);
    }
    Ok(())
}

impl RelationSnapshot {
    fn from_parts(key: RqmRelationKey, state: RqmPhaseRelationState) -> Self {
        Self {
            observer: key.observer,
            source: key.source.0,
            target: key.target.0,
            amplitude: state.amplitude,
            phase: state.phase,
            coherence: state.coherence,
            uncertainty: state.uncertainty,
            eligibility: state.eligibility,
            prediction_error: state.prediction_error,
            exposures: state.exposures,
            consolidated: state.consolidated,
        }
    }

    fn key(&self) -> RqmRelationKey {
        RqmRelationKey {
            observer: self.observer,
            source: LatentConceptId(self.source),
            target: LatentConceptId(self.target),
        }
    }

    fn state(&self) -> RqmPhaseRelationState {
        RqmPhaseRelationState {
            amplitude: self.amplitude,
            phase: self.phase,
            coherence: self.coherence,
            uncertainty: self.uncertainty,
            eligibility: self.eligibility,
            prediction_error: self.prediction_error,
            exposures: self.exposures,
            consolidated: self.consolidated,
        }
    }
}

impl From<ConsolidatedKnowledge> for KnowledgeSnapshot {
    fn from(value: ConsolidatedKnowledge) -> Self {
        Self {
            observer: value.key.observer,
            source: value.key.source.0,
            target: value.key.target.0,
            confidence: value.confidence,
            topological_symmetry: value.topological_symmetry,
            spin_entropy: value.spin_entropy,
            prediction_error: value.prediction_error,
            consolidations: value.consolidations,
        }
    }
}

impl KnowledgeSnapshot {
    fn value(&self) -> ConsolidatedKnowledge {
        ConsolidatedKnowledge {
            key: KnowledgeKey {
                observer: self.observer,
                source: LatentConceptId(self.source),
                target: LatentConceptId(self.target),
            },
            confidence: self.confidence,
            topological_symmetry: self.topological_symmetry,
            spin_entropy: self.spin_entropy,
            prediction_error: self.prediction_error,
            consolidations: self.consolidations,
        }
    }
}

fn save_checkpoint(
    root: &Path,
    checkpoint: &TrainingCheckpoint,
    milestone: bool,
) -> io::Result<()> {
    let body = serde_json::to_vec(checkpoint).map_err(io::Error::other)?;
    let latest = root.join("latest.json");
    let temporary = root.join("latest.tmp");
    fs::write(&temporary, &body)?;
    if latest.exists() {
        fs::remove_file(&latest)?;
    }
    fs::rename(&temporary, &latest)?;
    if milestone {
        fs::write(
            root.join("checkpoints")
                .join(format!("batch-{:012}.json", checkpoint.batch)),
            body,
        )?;
    }
    Ok(())
}

fn append_jsonl(path: &Path, value: &impl Serialize) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value).map_err(io::Error::other)?;
    file.write_all(b"\n")
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper.max(1)
    }
}
