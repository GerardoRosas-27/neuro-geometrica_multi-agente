use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::entanglement::EntanglementConfig;
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use std::env;
use std::hash::{Hash, Hasher};

const DEFAULT_STATE: &str = "data/cdt_rqm_epr_small_sleep.cdt_rqm";
const DEFAULT_NODES_PER_SLICE: usize = 2048;

#[derive(Clone, Copy)]
struct Sample {
    kind: SampleKind,
    input: &'static str,
    target: &'static str,
    distractor: &'static str,
}

#[derive(Clone, Copy)]
enum SampleKind {
    Concept,
    Causal,
    Skill,
    Episode,
    Correlation,
}

#[derive(Default, Clone, Copy)]
struct Metrics {
    cases: usize,
    correct: usize,
    leakage_sum: f32,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
    }

    fn accuracy(self) -> f32 {
        self.correct as f32 / self.cases.max(1) as f32
    }

    fn leakage(self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }
}

fn main() {
    let state = env::var("CDT_RQM_EPR_SLEEP_STATE").unwrap_or_else(|_| DEFAULT_STATE.to_string());
    let attempts = env_usize("CDT_RQM_EPR_SLEEP_ATTEMPTS", 4).max(1);
    let nodes_per_slice =
        env_usize("CDT_RQM_EPR_SMALL_NODES_PER_SLICE", DEFAULT_NODES_PER_SLICE).max(256);
    let mut substrate = CdtRqmUniverseSubstrate::new(config(nodes_per_slice));
    match substrate.load_consolidated_state(&state) {
        Ok(()) => println!("loaded=true state={state}"),
        Err(err) => {
            println!("loaded=false state={} error={err}", state);
            return;
        }
    }
    if substrate.entanglement.is_none() {
        substrate.enable_entanglement(epr_config());
    }

    let before = evaluate(&substrate);
    let before_edges = active_edges(&substrate);
    let before_regge = substrate.hardware.regge_action();
    let before_epr = substrate.entanglement_summary();
    let validation = validation_set();
    let protected = validation
        .iter()
        .flat_map(|(_, _, cue, expected, _)| {
            cue.iter()
                .flat_map(move |source| expected.iter().map(move |target| (*source, *target)))
        })
        .collect::<Vec<_>>();
    let mut accepted = 0;
    let mut best_regge = before_regge;
    let mut best_edges = before_edges;
    for _ in 0..attempts {
        let backup = substrate.clone();
        substrate.hardware.anneal_geometry_step(&protected);
        let candidate = evaluate(&substrate);
        let candidate_edges = active_edges(&substrate);
        let candidate_regge = substrate.hardware.regge_action();
        let preserves = candidate.accuracy() + 0.0001 >= before.accuracy()
            && candidate.leakage() <= before.leakage() + 0.0001;
        let improves = candidate_regge < best_regge || candidate_edges < best_edges;
        if preserves && improves {
            accepted += 1;
            best_regge = candidate_regge;
            best_edges = candidate_edges;
        } else {
            substrate = backup;
        }
    }
    let after = evaluate(&substrate);
    let after_edges = active_edges(&substrate);
    let after_regge = substrate.hardware.regge_action();
    let after_epr = substrate.entanglement_summary();

    let preserves = after.accuracy() + 0.0001 >= before.accuracy()
        && after.leakage() <= before.leakage() + 0.0001;
    if preserves {
        match substrate.save_consolidated_state(&state) {
            Ok(()) => println!("saved=true state={state}"),
            Err(err) => println!("saved=false state={} error={err}", state),
        }
    } else {
        println!("saved=false reason=memory_degraded");
    }

    println!(
        "memory: accuracy={:.1}%->{:.1}% leakage={:.1}%->{:.1}% preserved={}",
        before.accuracy() * 100.0,
        after.accuracy() * 100.0,
        before.leakage() * 100.0,
        after.leakage() * 100.0,
        preserves
    );
    println!(
        "graphity: attempts={} accepted={} regge={:.1}->{:.1} edges={} -> {} causality_violations={}",
        attempts,
        accepted,
        before_regge,
        after_regge,
        before_edges,
        after_edges,
        substrate.hardware.causality_violations()
    );
    if let (Some(before), Some(after)) = (before_epr, after_epr) {
        println!(
            "epr: links={} -> {} coherence={:.3}->{:.3} entropy={:.3}->{:.3}",
            before.active_links,
            after.active_links,
            before.mean_coherence,
            after.mean_coherence,
            before.mean_entropy,
            after.mean_entropy
        );
    }
}

fn evaluate(substrate: &CdtRqmUniverseSubstrate) -> Metrics {
    let mut software = substrate.software.clone();
    let mut metrics = Metrics::default();
    for sample in dataset().iter().take(8) {
        let input = pattern("input", sample.input, 0);
        let target = pattern("target", sample.target, 1);
        let distractor = pattern("distractor", sample.distractor, 1);
        let report = software.observe_pattern(
            observer(sample.kind, sample.input),
            &input,
            phase_for_kind(sample.kind),
            96,
        );
        let expected = report
            .candidates
            .iter()
            .filter(|candidate| target.contains(&candidate.agent))
            .map(|candidate| candidate.score)
            .sum::<f32>();
        let leak = report
            .candidates
            .iter()
            .filter(|candidate| distractor.contains(&candidate.agent))
            .map(|candidate| candidate.score)
            .sum::<f32>();
        metrics.record(expected, leak);
    }
    metrics
}

fn validation_set() -> Vec<(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)> {
    dataset()
        .iter()
        .take(8)
        .map(|sample| {
            (
                observer(sample.kind, sample.input),
                phase_for_kind(sample.kind),
                pattern("input", sample.input, 0),
                pattern("target", sample.target, 1),
                pattern("distractor", sample.distractor, 1),
            )
        })
        .collect()
}

fn active_edges(substrate: &CdtRqmUniverseSubstrate) -> usize {
    substrate
        .hardware
        .edges
        .iter()
        .filter(|edge| edge.active)
        .count()
}

fn pattern(prefix: &str, value: &str, slice: usize) -> Vec<usize> {
    let nodes_per_slice =
        env_usize("CDT_RQM_EPR_SMALL_NODES_PER_SLICE", DEFAULT_NODES_PER_SLICE).max(256);
    let mut out = (0..16)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prefix.hash(&mut hasher);
            normalize(value).hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * nodes_per_slice + (hasher.finish() as usize % nodes_per_slice)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn observer(kind: SampleKind, value: &str) -> ObserverId {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    sample_kind_label(kind).hash(&mut hasher);
    normalize(value).hash(&mut hasher);
    ObserverId(500_000 + (hasher.finish() as usize % 400_000))
}

fn phase_for_kind(kind: SampleKind) -> f32 {
    match kind {
        SampleKind::Concept => 0.0,
        SampleKind::Causal => std::f32::consts::FRAC_PI_2,
        SampleKind::Skill => std::f32::consts::PI,
        SampleKind::Episode => -std::f32::consts::FRAC_PI_2,
        SampleKind::Correlation => 0.25,
    }
}

fn sample_kind_label(kind: SampleKind) -> &'static str {
    match kind {
        SampleKind::Concept => "concept",
        SampleKind::Causal => "causal",
        SampleKind::Skill => "skill",
        SampleKind::Episode => "episode",
        SampleKind::Correlation => "correlation",
    }
}

fn normalize(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|ch| match ch {
            'á' | 'à' | 'ä' | 'â' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            'ñ' => 'n',
            other => other,
        })
        .filter(|ch| ch.is_ascii_alphanumeric() || ch.is_whitespace())
        .collect()
}

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn epr_config() -> EntanglementConfig {
    EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node: 8,
        max_syncs_per_step: 512,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    }
}

fn config(nodes_per_slice: usize) -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice,
            initial_spatial_connectivity: 0.00004,
            initial_temporal_connectivity: 0.00002,
            target_spatial_degree: 5,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 12,
            seed: 92_001,
        },
        rqm: RelationalFieldConfig {
            amplitude_learning_rate: 0.09,
            phase_learning_rate: 0.22,
            coherence_learning_rate: 0.12,
            uncertainty_learning_rate: 0.10,
            amplitude_decay: 0.001,
            coherence_decay: 0.0005,
            uncertainty_recovery: 0.002,
            activation_threshold: 0.025,
        },
        max_quantum_candidates: 96,
        rqm_feedback_gain: 0.40,
    }
}

fn dataset() -> &'static [Sample] {
    &[
        Sample {
            kind: SampleKind::Concept,
            input: "perro",
            target: "mamifero domestico que ladra y puede ser mascota",
            distractor: "fuego produce calor",
        },
        Sample {
            kind: SampleKind::Concept,
            input: "agua",
            target: "liquido que moja y sostiene vida",
            distractor: "vidrio fragil",
        },
        Sample {
            kind: SampleKind::Causal,
            input: "fuego",
            target: "calor",
            distractor: "suelo mojado",
        },
        Sample {
            kind: SampleKind::Causal,
            input: "lluvia",
            target: "suelo mojado",
            distractor: "hambre saciada",
        },
        Sample {
            kind: SampleKind::Causal,
            input: "golpear vidrio",
            target: "vidrio roto",
            distractor: "planta crece",
        },
        Sample {
            kind: SampleKind::Skill,
            input: "sumar",
            target: "leer numeros alinear cantidades combinar resultado",
            distractor: "dibujar contorno",
        },
        Sample {
            kind: SampleKind::Skill,
            input: "programar",
            target: "definir objetivo diseñar pasos escribir codigo probar corregir",
            distractor: "caminar equilibrar pie",
        },
        Sample {
            kind: SampleKind::Episode,
            input: "vi gato negro bajo lluvia",
            target: "gato negro existe y lluvia cambia entorno",
            distractor: "fuego calienta",
        },
        Sample {
            kind: SampleKind::Episode,
            input: "comi y dejo de doler hambre",
            target: "comer sacia hambre",
            distractor: "lluvia moja suelo",
        },
        Sample {
            kind: SampleKind::Correlation,
            input: "perro humano",
            target: "mascota relacion social",
            distractor: "vidrio roto",
        },
        Sample {
            kind: SampleKind::Correlation,
            input: "agua planta",
            target: "agua ayuda crecimiento planta",
            distractor: "programar codigo",
        },
    ]
}
