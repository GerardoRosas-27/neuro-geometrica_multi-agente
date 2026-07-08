use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use cdt_rqm_epr::native_thermodynamic_engine::{
    native_sleep_consolidate, Lesson, LessonKind, DEFAULT_NODES_PER_SLICE, DEFAULT_OBSERVER,
};
use cdt_rqm_epr::relational_field::ObserverId;
use std::collections::HashMap;
use std::env;
use std::hash::{Hash, Hasher};
use std::process::Command;
use std::time::Instant;

#[derive(Clone, Copy)]
struct ConceptSpec {
    id: &'static str,
    aliases: &'static [(&'static str, &'static str)],
    attributes: &'static [&'static str],
    distractor: &'static str,
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    cases: usize,
    correct: usize,
    leakage_sum: f32,
    margin_sum: f32,
    gemma_ok: usize,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32, gemma_ok: bool) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.gemma_ok += usize::from(gemma_ok);
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
    }

    fn accuracy(self) -> f32 {
        self.correct as f32 / self.cases.max(1) as f32
    }

    fn leakage(self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }

    fn margin(self) -> f32 {
        self.margin_sum / self.cases.max(1) as f32
    }

    fn gemma_accuracy(self) -> f32 {
        self.gemma_ok as f32 / self.cases.max(1) as f32
    }
}

struct GemmaPeripheral {
    model: String,
    cache: HashMap<(String, String), String>,
}

fn main() {
    let model = env::var("GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string());
    let epochs = env_usize("GEMMA_PHASE1_EPOCHS", 4).max(1);
    let sleep_attempts = env_usize("GEMMA_PHASE1_SLEEP_ATTEMPTS", 4);
    let sleep_replay_passes = env_usize("GEMMA_PHASE1_SLEEP_REPLAY_PASSES", 2).max(1);
    let mut gemma = GemmaPeripheral::new(model);
    let mut substrate = clean_substrate();
    let start = Instant::now();

    println!("Native Gemma phase 1 trainer");
    println!(
        "model={} concepts={} epochs={} sleep_attempts={} sleep_replay_passes={}",
        gemma.model,
        concepts().len(),
        epochs,
        sleep_attempts,
        sleep_replay_passes
    );

    for epoch in 0..epochs {
        train_epoch(&mut substrate, &mut gemma, epoch);
        let lessons = sleep_lessons();
        let (slept, sleep) =
            native_sleep_consolidate(substrate, &lessons, sleep_attempts, sleep_replay_passes);
        substrate = slept;
        let eval = evaluate_cross_lingual(&mut substrate, &mut gemma);
        println!(
            "epoch={} sleep_accepted={} cross_lingual_acc={:.1}% gemma_map={:.1}% leakage={:.1}% margin={:.3} relations={} epr_links={} elapsed_ms={:.1}",
            epoch + 1,
            sleep.accepted,
            eval.accuracy() * 100.0,
            eval.gemma_accuracy() * 100.0,
            eval.leakage() * 100.0,
            eval.margin(),
            substrate.relation_count(),
            substrate.entanglement.summary().active_links,
            start.elapsed().as_secs_f64() * 1_000.0
        );
    }

    let final_eval = evaluate_cross_lingual(&mut substrate, &mut gemma);
    println!(
        "final: accuracy={:.1}% gemma_map={:.1}% leakage={:.1}% margin={:.3} relations={} epr_links={} decision={}",
        final_eval.accuracy() * 100.0,
        final_eval.gemma_accuracy() * 100.0,
        final_eval.leakage() * 100.0,
        final_eval.margin(),
        substrate.relation_count(),
        substrate.entanglement.summary().active_links,
        if final_eval.accuracy() >= 0.90 && final_eval.leakage() <= 0.05 {
            "phase1_pass"
        } else {
            "phase1_needs_tuning"
        }
    );
}

impl GemmaPeripheral {
    fn new(model: String) -> Self {
        Self {
            model,
            cache: HashMap::new(),
        }
    }

    fn map_alias(&mut self, language: &str, surface: &str) -> Option<String> {
        let key = (language.to_string(), surface.to_string());
        if let Some(value) = self.cache.get(&key) {
            return Some(value.clone());
        }

        let allowed = concepts()
            .iter()
            .map(|concept| concept.id)
            .collect::<Vec<_>>()
            .join(", ");
        let prompt = format!(
            "Map this word or phrase to exactly one concept id from this list: {allowed}.\nLanguage: {language}\nText: {surface}\nReturn only the concept id, no explanation."
        );
        let output = Command::new("ollama")
            .arg("run")
            .arg(&self.model)
            .arg(prompt)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout).to_uppercase();
        let mapped = concepts()
            .iter()
            .find(|concept| text.contains(concept.id))
            .map(|concept| concept.id.to_string())?;
        self.cache.insert(key, mapped.clone());
        Some(mapped)
    }
}

fn train_epoch(
    substrate: &mut NativeThermoRqmEprSubstrate,
    gemma: &mut GemmaPeripheral,
    epoch: usize,
) {
    for concept in concepts() {
        let Some(mapped) = concept
            .aliases
            .iter()
            .find(|(language, _)| *language == "es")
            .and_then(|(language, alias)| gemma.map_alias(language, alias))
        else {
            continue;
        };
        if mapped != concept.id {
            continue;
        }

        let concept_pattern = pattern("concept", concept.id, 1);
        let spanish_alias = concept
            .aliases
            .iter()
            .find(|(language, _)| *language == "es")
            .map(|(_, alias)| pattern("alias", alias, 0))
            .unwrap_or_default();
        substrate.train_observed_transition(
            DEFAULT_OBSERVER,
            0.0,
            &spanish_alias,
            &concept_pattern,
            1.0,
        );
        for attribute in concept.attributes {
            let attr = pattern("attribute", attribute, 2);
            substrate.train_observed_transition(
                DEFAULT_OBSERVER,
                0.0,
                &concept_pattern,
                &attr,
                1.0,
            );
            substrate.train_observed_transition(
                typed_observer(concept.id),
                phase_for_epoch(epoch),
                &concept_pattern,
                &attr,
                0.95,
            );
        }
        let distractor = pattern("attribute", concept.distractor, 2);
        for source in &concept_pattern {
            for target in &distractor {
                substrate.attenuate_relation(DEFAULT_OBSERVER, *source, *target, 0.35);
            }
        }
    }
}

fn evaluate_cross_lingual(
    substrate: &mut NativeThermoRqmEprSubstrate,
    gemma: &mut GemmaPeripheral,
) -> Metrics {
    let mut metrics = Metrics::default();
    for concept in concepts() {
        for (language, alias) in concept.aliases {
            if *language == "es" {
                continue;
            }
            let mapped = gemma.map_alias(language, alias);
            let gemma_ok = mapped.as_deref() == Some(concept.id);
            let query_id = mapped.unwrap_or_else(|| "UNKNOWN".to_string());
            let cue = pattern("concept", &query_id, 1);
            let report = substrate.query(DEFAULT_OBSERVER, 0.0, &cue);
            let expected = concept
                .attributes
                .iter()
                .map(|attribute| score(&report.candidates, &pattern("attribute", attribute, 2)))
                .sum::<f32>();
            let distractor = score(
                &report.candidates,
                &pattern("attribute", concept.distractor, 2),
            );
            metrics.record(expected, distractor, gemma_ok);
        }
    }
    metrics
}

fn sleep_lessons() -> Vec<Lesson> {
    concepts()
        .iter()
        .map(|concept| Lesson {
            kind: LessonKind::Semantic,
            local: pattern("concept", concept.id, 1),
            action: pattern("action", "cross_lingual_recall", 1),
            remote: concept
                .attributes
                .iter()
                .flat_map(|attribute| pattern("attribute", attribute, 2))
                .collect(),
            distractor: pattern("attribute", concept.distractor, 2),
        })
        .collect()
}

fn clean_substrate() -> NativeThermoRqmEprSubstrate {
    NativeThermoRqmEprSubstrate::new(
        NativeThermoCdtConfig {
            slices: 4,
            nodes_per_slice: DEFAULT_NODES_PER_SLICE,
            spatial_degree: 4,
            temporal_degree: 2,
            temperature: 0.24,
            dt: 0.01,
            diffusion: 0.20,
            confinement: 0.05,
            pilot_gain: 0.50,
            phase_coupling: 0.18,
            amplitude_decay: 0.003,
            state_clamp: 3.0,
            seed: 0x9E44_0001,
        },
        NativeThermoRqmConfig {
            thermal_steps_per_train: 0,
            thermal_steps_per_query: 2,
            thermal_score_gain: 0.35,
            thermal_activation_margin: f32::MAX,
            max_candidates: 128,
            max_pilot_window_nodes: 96,
            sampling_block_size: 16,
            sampling_schedule_rounds: 2,
            max_sampling_blocks: 8,
            collect_query_diagnostics: true,
            ..NativeThermoRqmConfig::default()
        },
        EntanglementConfig {
            create_threshold: 1.0,
            max_links_per_node: 8,
            max_syncs_per_step: 512,
            contradiction_gain: 0.55,
            max_entropy: 0.9,
            max_heat: 0.9,
            ..EntanglementConfig::default()
        },
    )
}

fn concepts() -> &'static [ConceptSpec] {
    &[
        ConceptSpec {
            id: "DOG",
            aliases: &[("es", "perro"), ("en", "dog"), ("fr", "chien")],
            attributes: &["mammal", "pet", "barks"],
            distractor: "metal",
        },
        ConceptSpec {
            id: "WATER",
            aliases: &[("es", "agua"), ("en", "water"), ("fr", "eau")],
            attributes: &["liquid", "life", "wet"],
            distractor: "fire",
        },
        ConceptSpec {
            id: "FIRE",
            aliases: &[("es", "fuego"), ("en", "fire"), ("fr", "feu")],
            attributes: &["heat", "burns", "light"],
            distractor: "wet",
        },
        ConceptSpec {
            id: "PLANT",
            aliases: &[("es", "planta"), ("en", "plant"), ("fr", "plante")],
            attributes: &["grows", "roots", "oxygen"],
            distractor: "machine",
        },
        ConceptSpec {
            id: "CODE",
            aliases: &[("es", "codigo"), ("en", "code"), ("fr", "code")],
            attributes: &["program", "logic", "debug"],
            distractor: "animal",
        },
    ]
}

fn pattern(label: &str, value: &str, slice: usize) -> Vec<usize> {
    let mut out = (0..10)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            label.hash(&mut hasher);
            value.hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * DEFAULT_NODES_PER_SLICE + (hasher.finish() as usize % DEFAULT_NODES_PER_SLICE)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn score(candidates: &[NativeCandidateScore], targets: &[usize]) -> f32 {
    candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn typed_observer(id: &str) -> ObserverId {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut hasher);
    ObserverId(810_000 + (hasher.finish() as usize % 20_000))
}

fn phase_for_epoch(epoch: usize) -> f32 {
    match epoch % 4 {
        0 => 0.0,
        1 => std::f32::consts::FRAC_PI_2,
        2 => std::f32::consts::PI,
        _ => -std::f32::consts::FRAC_PI_2,
    }
}

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}
