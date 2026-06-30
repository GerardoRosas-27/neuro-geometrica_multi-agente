use snga::relational_field::{
    CollapseReport, ObserverId, RelationalFieldConfig, RelationalFieldSubstrate,
};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI};

const TRAINING_EPOCHS: usize = 18;
const PATTERN_STRIDE: usize = 8;

#[derive(Clone, Copy)]
struct FrameSpec {
    observer: ObserverId,
    phase: f32,
    context: &'static str,
    primary: &'static str,
    support_a: &'static str,
    support_b: &'static str,
}

#[derive(Clone, Copy)]
struct AmbiguousCase {
    cue: &'static str,
    left: FrameSpec,
    right: FrameSpec,
}

#[derive(Default)]
struct Metrics {
    frames: usize,
    top_correct: usize,
    purity_sum: f32,
    leakage_sum: f32,
    margin_sum: f32,
    coherence_sum: f32,
    incompatible_tension_sum: f32,
}

impl Metrics {
    fn record(
        &mut self,
        expected_score: f32,
        distractor_score: f32,
        coherence: f32,
        incompatible_tension: f32,
    ) {
        let total = expected_score + distractor_score;
        let purity = if total > f32::EPSILON {
            expected_score / total
        } else {
            0.0
        };
        let leakage = if total > f32::EPSILON {
            distractor_score / total
        } else {
            1.0
        };
        self.frames += 1;
        self.top_correct += usize::from(expected_score > distractor_score);
        self.purity_sum += purity;
        self.leakage_sum += leakage;
        self.margin_sum += expected_score - distractor_score;
        self.coherence_sum += coherence;
        self.incompatible_tension_sum += incompatible_tension;
    }

    fn accuracy(&self) -> f32 {
        self.top_correct as f32 / self.frames.max(1) as f32
    }

    fn mean_purity(&self) -> f32 {
        self.purity_sum / self.frames.max(1) as f32
    }

    fn mean_leakage(&self) -> f32 {
        self.leakage_sum / self.frames.max(1) as f32
    }

    fn mean_margin(&self) -> f32 {
        self.margin_sum / self.frames.max(1) as f32
    }

    fn mean_coherence(&self) -> f32 {
        self.coherence_sum / self.frames.max(1) as f32
    }

    fn mean_incompatible_tension(&self) -> f32 {
        self.incompatible_tension_sum / self.frames.max(1) as f32
    }
}

struct SymbolTable {
    ids: HashMap<&'static str, usize>,
}

impl SymbolTable {
    fn from_cases(cases: &[AmbiguousCase]) -> Self {
        let mut ids = HashMap::new();
        for case in cases {
            insert_symbol(&mut ids, case.cue);
            for frame in [case.left, case.right] {
                insert_symbol(&mut ids, frame.context);
                insert_symbol(&mut ids, frame.primary);
                insert_symbol(&mut ids, frame.support_a);
                insert_symbol(&mut ids, frame.support_b);
            }
        }
        Self { ids }
    }

    fn id(&self, symbol: &'static str) -> usize {
        *self.ids.get(symbol).expect("symbol must exist")
    }

    fn pattern(&self, symbol: &'static str) -> Vec<usize> {
        let start = self.id(symbol) * PATTERN_STRIDE;
        vec![start, start + 1, start + 3, start + 5]
    }

    fn frame_pattern(&self, frame: FrameSpec) -> Vec<usize> {
        let mut pattern = self.pattern(frame.primary);
        pattern.extend(self.pattern(frame.support_a));
        pattern.extend(self.pattern(frame.support_b));
        pattern.sort_unstable();
        pattern.dedup();
        pattern
    }
}

fn main() {
    let cases = cases();
    let symbols = SymbolTable::from_cases(&cases);
    let rqf = train_relational_field(&cases, &symbols);
    let legacy = train_legacy_snga(&cases, &symbols);

    let rqf_metrics = evaluate_relational_field(&rqf, &cases, &symbols);
    let legacy_cue_only = evaluate_legacy_snga(&legacy, &cases, &symbols, LegacyQuery::CueOnly);
    let legacy_with_context =
        evaluate_legacy_snga(&legacy, &cases, &symbols, LegacyQuery::CueAndContext);

    println!("RQF-SNGA comparison benchmark");
    println!(
        "cases={} frames={} training_epochs={} symbols={} rqf_relations={}",
        cases.len(),
        cases.len() * 2,
        TRAINING_EPOCHS,
        symbols.ids.len(),
        rqf.relation_count()
    );
    print_metrics("rqf_observer_relational", &rqf_metrics);
    print_metrics("legacy_snga_cue_only", &legacy_cue_only);
    print_metrics("legacy_snga_cue_and_context", &legacy_with_context);
    println!(
        "gain_vs_cue_only: accuracy={:.1}% purity={:.1}% leakage_reduction={:.1}%",
        (rqf_metrics.accuracy() - legacy_cue_only.accuracy()) * 100.0,
        (rqf_metrics.mean_purity() - legacy_cue_only.mean_purity()) * 100.0,
        (legacy_cue_only.mean_leakage() - rqf_metrics.mean_leakage()) * 100.0
    );
    println!(
        "gain_vs_contextual_legacy: accuracy={:.1}% purity={:.1}% leakage_reduction={:.1}%",
        (rqf_metrics.accuracy() - legacy_with_context.accuracy()) * 100.0,
        (rqf_metrics.mean_purity() - legacy_with_context.mean_purity()) * 100.0,
        (legacy_with_context.mean_leakage() - rqf_metrics.mean_leakage()) * 100.0
    );
    println!(
        "lectura: {}",
        if rqf_metrics.mean_leakage() < legacy_with_context.mean_leakage()
            && rqf_metrics.accuracy() >= legacy_with_context.accuracy()
        {
            "el observador relacional conserva o mejora el acierto y reduce fuga semantica frente al sustrato previo"
        } else {
            "el sustrato relacional aprende, pero aun no supera claramente a la linea base contextual"
        }
    );
}

fn train_relational_field(
    cases: &[AmbiguousCase],
    symbols: &SymbolTable,
) -> RelationalFieldSubstrate {
    let mut substrate = RelationalFieldSubstrate::new(relational_config());
    for _ in 0..TRAINING_EPOCHS {
        for case in cases {
            train_frame_relations(&mut substrate, symbols, case.cue, case.left, case.right);
            train_frame_relations(&mut substrate, symbols, case.cue, case.right, case.left);
        }
    }
    substrate
}

fn train_frame_relations(
    substrate: &mut RelationalFieldSubstrate,
    symbols: &SymbolTable,
    cue: &'static str,
    frame: FrameSpec,
    competing_frame: FrameSpec,
) {
    let cue = symbols.id(cue);
    let primary = symbols.id(frame.primary);
    let support_a = symbols.id(frame.support_a);
    let support_b = symbols.id(frame.support_b);
    let competing_primary = symbols.id(competing_frame.primary);
    let competing_a = symbols.id(competing_frame.support_a);
    let competing_b = symbols.id(competing_frame.support_b);

    substrate.reinforce_relation(frame.observer, cue, primary, frame.phase, 1.0);
    substrate.reinforce_relation(frame.observer, cue, support_a, frame.phase, 1.0);
    substrate.reinforce_relation(frame.observer, cue, support_b, frame.phase, 1.0);
    substrate.reinforce_relation(frame.observer, primary, support_a, 0.0, 1.0);
    substrate.reinforce_relation(frame.observer, support_a, support_b, 0.0, 1.0);
    substrate.reinforce_relation(frame.observer, support_b, primary, 0.0, 1.0);

    let incompatible_phase = frame.phase + PI;
    substrate.reinforce_relation(
        frame.observer,
        cue,
        competing_primary,
        incompatible_phase,
        0.20,
    );
    substrate.reinforce_relation(frame.observer, cue, competing_a, incompatible_phase, 0.20);
    substrate.reinforce_relation(frame.observer, cue, competing_b, incompatible_phase, 0.20);
    substrate.reinforce_relation(
        frame.observer,
        competing_primary,
        competing_a,
        incompatible_phase,
        0.20,
    );
    substrate.reinforce_relation(
        frame.observer,
        competing_a,
        competing_b,
        incompatible_phase,
        0.20,
    );
}

fn train_legacy_snga(cases: &[AmbiguousCase], symbols: &SymbolTable) -> SimplicialNetwork {
    let mut network = SimplicialNetwork::grid_3d(legacy_config(symbols.ids.len()), 2);
    for _ in 0..TRAINING_EPOCHS {
        for case in cases {
            train_legacy_frame(&mut network, symbols, case.cue, case.left);
            train_legacy_frame(&mut network, symbols, case.cue, case.right);
        }
    }
    network
}

fn train_legacy_frame(
    network: &mut SimplicialNetwork,
    symbols: &SymbolTable,
    cue: &'static str,
    frame: FrameSpec,
) {
    let cue_pattern = symbols.pattern(cue);
    let context_pattern = symbols.pattern(frame.context);
    let frame_pattern = symbols.frame_pattern(frame);
    let mut fused = cue_pattern.clone();
    fused.extend(context_pattern.iter().copied());
    fused.extend(frame_pattern.iter().copied());
    fused.sort_unstable();
    fused.dedup();

    network.learn_transition(&cue_pattern, &frame_pattern);
    network.learn_transition(&context_pattern, &frame_pattern);
    network.reinforce_coactivation_if_useful(&fused, 0.04, 0.92);
}

fn evaluate_relational_field(
    substrate: &RelationalFieldSubstrate,
    cases: &[AmbiguousCase],
    symbols: &SymbolTable,
) -> Metrics {
    let mut substrate = substrate.clone();
    let mut metrics = Metrics::default();
    for case in cases {
        evaluate_relational_frame(
            &mut substrate,
            symbols,
            case.cue,
            case.left,
            case.right,
            &mut metrics,
        );
        evaluate_relational_frame(
            &mut substrate,
            symbols,
            case.cue,
            case.right,
            case.left,
            &mut metrics,
        );
    }
    metrics
}

fn evaluate_relational_frame(
    substrate: &mut RelationalFieldSubstrate,
    symbols: &SymbolTable,
    cue: &'static str,
    frame: FrameSpec,
    competing_frame: FrameSpec,
    metrics: &mut Metrics,
) {
    let report = substrate.observe_pattern(frame.observer, &[symbols.id(cue)], frame.phase, 12);
    let expected = [
        symbols.id(frame.primary),
        symbols.id(frame.support_a),
        symbols.id(frame.support_b),
    ];
    let distractors = [
        symbols.id(competing_frame.primary),
        symbols.id(competing_frame.support_a),
        symbols.id(competing_frame.support_b),
    ];
    let expected_score = score_report(&report, &expected);
    let distractor_score = score_report(&report, &distractors);
    let coherence = substrate
        .simplex_phase_report(
            frame.observer,
            symbols.id(cue),
            symbols.id(frame.primary),
            symbols.id(frame.support_a),
        )
        .map(|report| report.coherence)
        .unwrap_or(0.0);
    let incompatible_tension = substrate
        .simplex_phase_report(
            frame.observer,
            symbols.id(cue),
            symbols.id(competing_frame.primary),
            symbols.id(competing_frame.support_a),
        )
        .map(|report| report.tension)
        .unwrap_or(0.0);
    metrics.record(
        expected_score,
        distractor_score,
        coherence,
        incompatible_tension,
    );
}

#[derive(Clone, Copy)]
enum LegacyQuery {
    CueOnly,
    CueAndContext,
}

fn evaluate_legacy_snga(
    network: &SimplicialNetwork,
    cases: &[AmbiguousCase],
    symbols: &SymbolTable,
    query: LegacyQuery,
) -> Metrics {
    let mut metrics = Metrics::default();
    for case in cases {
        evaluate_legacy_frame(
            network,
            symbols,
            case.cue,
            case.left,
            case.right,
            query,
            &mut metrics,
        );
        evaluate_legacy_frame(
            network,
            symbols,
            case.cue,
            case.right,
            case.left,
            query,
            &mut metrics,
        );
    }
    metrics
}

fn evaluate_legacy_frame(
    network: &SimplicialNetwork,
    symbols: &SymbolTable,
    cue: &'static str,
    frame: FrameSpec,
    competing_frame: FrameSpec,
    query: LegacyQuery,
    metrics: &mut Metrics,
) {
    let mut query_pattern = symbols.pattern(cue);
    if matches!(query, LegacyQuery::CueAndContext) {
        query_pattern.extend(symbols.pattern(frame.context));
        query_pattern.sort_unstable();
        query_pattern.dedup();
    }
    let prediction = network.predict_from(&query_pattern, 96);
    let expected = symbols.frame_pattern(frame);
    let distractors = symbols.frame_pattern(competing_frame);
    let expected_score = score_prediction(&prediction, &expected);
    let distractor_score = score_prediction(&prediction, &distractors);
    metrics.record(expected_score, distractor_score, 0.0, 0.0);
}

fn score_report(report: &CollapseReport, targets: &[usize]) -> f32 {
    report
        .candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn score_prediction(prediction: &[(usize, f32)], targets: &[usize]) -> f32 {
    prediction
        .iter()
        .filter(|(agent, _)| targets.contains(agent))
        .map(|(_, score)| *score)
        .sum()
}

fn print_metrics(label: &str, metrics: &Metrics) {
    println!(
        "{}: frames={} accuracy={:.1}% purity={:.1}% leakage={:.1}% margin={:.3} simplex_coherence={:.3} incompatible_tension={:.3}",
        label,
        metrics.frames,
        metrics.accuracy() * 100.0,
        metrics.mean_purity() * 100.0,
        metrics.mean_leakage() * 100.0,
        metrics.mean_margin(),
        metrics.mean_coherence(),
        metrics.mean_incompatible_tension()
    );
}

fn insert_symbol(ids: &mut HashMap<&'static str, usize>, symbol: &'static str) {
    let next = ids.len();
    ids.entry(symbol).or_insert(next);
}

fn relational_config() -> RelationalFieldConfig {
    RelationalFieldConfig {
        amplitude_learning_rate: 0.075,
        phase_learning_rate: 0.20,
        coherence_learning_rate: 0.11,
        uncertainty_learning_rate: 0.10,
        amplitude_decay: 0.002,
        coherence_decay: 0.001,
        uncertainty_recovery: 0.003,
        activation_threshold: 0.03,
    }
}

fn legacy_config(symbol_count: usize) -> SimplicialConfig {
    let width = 48;
    let height = ((symbol_count * PATTERN_STRIDE) / width + 4).max(18);
    SimplicialConfig {
        width,
        height,
        spacing: 8.0,
        elasticity: 0.006,
        damping: 0.88,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0002,
        max_active_agents: 64,
        inhibition_decay: 0.08,
        max_spikes_per_step: 256,
        local_inhibition_decay: 0.70,
        refractory_ticks: 1,
        rhythm_period: 16,
        rhythm_amplitude: 0.0,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 128,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.075,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 811,
    }
}

fn cases() -> Vec<AmbiguousCase> {
    vec![
        ambiguous(
            "banco",
            frame(100, 0.0, "ctx_finanzas", "dinero", "credito", "institucion"),
            frame(101, FRAC_PI_2, "ctx_parque", "sentarse", "madera", "plaza"),
        ),
        ambiguous(
            "planta",
            frame(102, 0.0, "ctx_botanica", "vegetal", "hoja", "raiz"),
            frame(
                103,
                FRAC_PI_2,
                "ctx_industria",
                "fabrica",
                "maquina",
                "produccion",
            ),
        ),
        ambiguous(
            "raton",
            frame(104, 0.0, "ctx_animal", "roedor", "cola_animal", "queso"),
            frame(
                105,
                FRAC_PI_2,
                "ctx_computadora",
                "periferico",
                "cursor",
                "usb",
            ),
        ),
        ambiguous(
            "cola",
            frame(106, 0.0, "ctx_cuerpo", "apendice", "animal", "movimiento"),
            frame(107, FRAC_PI_2, "ctx_fila", "espera", "personas", "turno"),
        ),
        ambiguous(
            "carta",
            frame(108, 0.0, "ctx_correo", "mensaje", "sobre", "sello"),
            frame(
                109,
                FRAC_PI_2,
                "ctx_restaurante",
                "menu",
                "comida",
                "precio",
            ),
        ),
        ambiguous(
            "vela",
            frame(110, 0.0, "ctx_luz", "cera", "llama", "mecha"),
            frame(
                111,
                FRAC_PI_2,
                "ctx_barco",
                "navegacion",
                "viento",
                "mastil",
            ),
        ),
        ambiguous(
            "sierra",
            frame(112, 0.0, "ctx_taller", "herramienta", "corte", "diente"),
            frame(
                113,
                FRAC_PI_2,
                "ctx_montana",
                "cordillera",
                "altura",
                "valle",
            ),
        ),
        ambiguous(
            "copa",
            frame(114, 0.0, "ctx_bebida", "vaso", "liquido", "brindis"),
            frame(115, FRAC_PI_2, "ctx_deporte", "trofeo", "campeon", "torneo"),
        ),
        ambiguous(
            "radio",
            frame(116, 0.0, "ctx_audio", "receptor", "frecuencia", "antena"),
            frame(
                117,
                FRAC_PI_2,
                "ctx_geometria",
                "circulo",
                "centro",
                "distancia",
            ),
        ),
        ambiguous(
            "red",
            frame(118, 0.0, "ctx_internet", "conexion", "servidor", "paquete"),
            frame(119, FRAC_PI_2, "ctx_pesca", "malla", "pez", "agua"),
        ),
        ambiguous(
            "llave",
            frame(120, 0.0, "ctx_puerta", "cerradura", "abrir", "metal"),
            frame(121, FRAC_PI_2, "ctx_musica", "tonalidad", "nota", "armonia"),
        ),
        ambiguous(
            "corriente",
            frame(122, 0.0, "ctx_electricidad", "electron", "voltaje", "cable"),
            frame(123, FRAC_PI_2, "ctx_agua", "rio", "flujo", "orilla"),
        ),
    ]
}

fn ambiguous(cue: &'static str, left: FrameSpec, right: FrameSpec) -> AmbiguousCase {
    AmbiguousCase { cue, left, right }
}

fn frame(
    observer: usize,
    phase: f32,
    context: &'static str,
    primary: &'static str,
    support_a: &'static str,
    support_b: &'static str,
) -> FrameSpec {
    FrameSpec {
        observer: ObserverId(observer),
        phase,
        context,
        primary,
        support_a,
        support_b,
    }
}
