use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const CONCEPTS: usize = 10_000;
const EPOCHS: usize = 3;
const EVAL_SAMPLES: usize = 100;
const RECALL_STEPS: usize = 4;
const LANGUAGE_TERMS: usize = 3;
const VISION_TERMS: usize = 4;
const AUDIO_TERMS: usize = 2;
const ACTIVE_THRESHOLD: f32 = 0.08;

#[derive(Clone, Copy)]
enum Modality {
    Language,
    Vision,
    Audio,
}

impl Modality {
    fn band(self) -> usize {
        match self {
            Self::Language => 0,
            Self::Vision => 1,
            Self::Audio => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Language => "language",
            Self::Vision => "vision",
            Self::Audio => "audio",
        }
    }
}

#[derive(Default)]
struct Aggregate {
    recall: f32,
    precision: f32,
    leakage: f32,
    max_active: usize,
    samples: usize,
}

struct ConceptPattern {
    language: Vec<usize>,
    sensory: Vec<usize>,
}

fn main() {
    let config = large_config();
    let mut network = SimplicialNetwork::grid(config);
    let patterns = build_patterns(&network);
    let sensory_counts = build_sensory_counts(network.agents.len(), &patterns);

    println!("SNGA large synthetic validation");
    println!("conceptos={CONCEPTS}");
    println!("epocas={EPOCHS}");
    println!("muestras_eval={EVAL_SAMPLES}");
    println!("nodos={}", network.agents.len());
    println!(
        "inhibicion=max_active:{} max_spikes:{} decay:{:.2}",
        network.config.max_active_agents,
        network.config.max_spikes_per_step,
        network.config.inhibition_decay
    );
    println!();

    train(&mut network, &patterns);
    network.clear_activity();
    println!("aristas_totales_post_entrenamiento={}", network.edges.len());
    println!();

    let mut aggregate = Aggregate::default();
    for concept_id in 0..EVAL_SAMPLES.min(patterns.len()) {
        let report = evaluate(&mut network, concept_id, &patterns, &sensory_counts);
        aggregate.recall += report.recall;
        aggregate.precision += report.precision;
        aggregate.leakage += report.leakage;
        aggregate.max_active = aggregate.max_active.max(report.max_active);
        aggregate.samples += 1;

        if concept_id < 8 {
            println!(
                "concepto={concept_id:04} recall={:.1}% precision={:.1}% fuga={:.3}% activos_max={}",
                report.recall * 100.0,
                report.precision * 100.0,
                report.leakage * 100.0,
                report.max_active
            );
        }
    }

    let n = aggregate.samples.max(1) as f32;
    println!();
    println!(
        "resumen: recall_medio={:.1}% precision_media={:.1}% fuga_media={:.3}% activos_max_observado={}",
        aggregate.recall / n * 100.0,
        aggregate.precision / n * 100.0,
        aggregate.leakage / n * 100.0,
        aggregate.max_active
    );
    println!(
        "lectura: {}",
        if aggregate.recall / n > 0.85 && aggregate.leakage / n < 0.02 {
            "estable con inhibicion; viable como memoria asociativa escalable inicial"
        } else {
            "aprende, pero requiere mejor separacion/inhibicion para escalar con robustez"
        }
    );
}

fn train(network: &mut SimplicialNetwork, patterns: &[ConceptPattern]) {
    for _ in 0..EPOCHS {
        for pattern in patterns {
            let mut fused = pattern.language.clone();
            fused.extend(pattern.sensory.iter().copied());
            fused.sort_unstable();
            fused.dedup();
            network.reinforce_coactivation(&fused, 0.12);
        }
    }
}

struct EvalReport {
    recall: f32,
    precision: f32,
    leakage: f32,
    max_active: usize,
}

fn evaluate(
    network: &mut SimplicialNetwork,
    concept_id: usize,
    patterns: &[ConceptPattern],
    sensory_counts: &[u16],
) -> EvalReport {
    let target = &patterns[concept_id].sensory;
    let mut target_marker = vec![false; network.agents.len()];
    for &idx in target {
        target_marker[idx] = true;
    }

    network.clear_activity();
    network.inject_pattern(&patterns[concept_id].language, 1.35, 2);

    let mut max_active = 0;
    for _ in 0..RECALL_STEPS {
        let stats = network.step();
        max_active = max_active.max(stats.active_agents);
    }

    let active_targets = count_active(network, target);
    let mut active_distractors = 0;
    let mut distractor_nodes = 0;
    for (idx, &count) in sensory_counts.iter().enumerate() {
        if count == 0 || target_marker[idx] {
            continue;
        }
        distractor_nodes += 1;
        if network.agents[idx].surprise > ACTIVE_THRESHOLD {
            active_distractors += 1;
        }
    }
    let recall = active_targets as f32 / target.len().max(1) as f32;
    let leakage = active_distractors as f32 / distractor_nodes.max(1) as f32;
    let precision = active_targets as f32 / (active_targets + active_distractors).max(1) as f32;

    EvalReport {
        recall,
        precision,
        leakage,
        max_active,
    }
}

fn count_active(network: &SimplicialNetwork, pattern: &[usize]) -> usize {
    pattern
        .iter()
        .filter(|&&idx| network.agents[idx].surprise > ACTIVE_THRESHOLD)
        .count()
}

fn build_patterns(network: &SimplicialNetwork) -> Vec<ConceptPattern> {
    (0..CONCEPTS)
        .map(|concept_id| ConceptPattern {
            language: encode_terms(network, concept_id, Modality::Language, LANGUAGE_TERMS),
            sensory: sensory_pattern(network, concept_id),
        })
        .collect()
}

fn sensory_pattern(network: &SimplicialNetwork, concept_id: usize) -> Vec<usize> {
    let mut pattern = encode_terms(network, concept_id, Modality::Vision, VISION_TERMS);
    pattern.extend(encode_terms(
        network,
        concept_id,
        Modality::Audio,
        AUDIO_TERMS,
    ));
    pattern.sort_unstable();
    pattern.dedup();
    pattern
}

fn build_sensory_counts(nodes: usize, patterns: &[ConceptPattern]) -> Vec<u16> {
    let mut counts = vec![0_u16; nodes];
    for pattern in patterns {
        for &idx in &pattern.sensory {
            counts[idx] = counts[idx].saturating_add(1);
        }
    }
    counts
}

fn encode_terms(
    network: &SimplicialNetwork,
    concept_id: usize,
    modality: Modality,
    terms: usize,
) -> Vec<usize> {
    let len = network.agents.len().max(1);
    let band_size = (len / 3).max(1);
    let band_start = modality.band() * band_size;
    let band_end = if modality.band() == 2 {
        len
    } else {
        ((modality.band() + 1) * band_size).min(len)
    };
    let span = (band_end - band_start).max(1);

    (0..terms)
        .map(|term_id| {
            let mut hasher = DefaultHasher::new();
            modality.name().hash(&mut hasher);
            concept_id.hash(&mut hasher);
            term_id.hash(&mut hasher);
            band_start + (hasher.finish() as usize % span)
        })
        .collect()
}

fn large_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 600,
        height: 300,
        spacing: 3.5,
        elasticity: 0.003,
        damping: 0.88,
        activation_threshold: 0.68,
        simplex_area_weight: 0.0001,
        max_active_agents: 32,
        inhibition_decay: 0.02,
        max_spikes_per_step: 128,
        seed: 23,
    }
}
