use snga::relational_field::{ObserverId, RelationalFieldConfig, RelationalFieldSubstrate};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI};

const TRAINING_EPOCHS: usize = 18;
const PATTERN_STRIDE: usize = 8;
const PATTERN_WIDTH: usize = 4;
const EVAL_STEPS: usize = 5;

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
    active_agents_sum: usize,
    active_spikes_sum: usize,
    free_energy_sum: f32,
    stability_sum: f32,
    incompatible_tension_sum: f32,
}

impl Metrics {
    fn record(
        &mut self,
        expected_score: f32,
        distractor_score: f32,
        active_agents: usize,
        active_spikes: usize,
        free_energy: f32,
        stability: f32,
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
        self.active_agents_sum += active_agents;
        self.active_spikes_sum += active_spikes;
        self.free_energy_sum += free_energy;
        self.stability_sum += stability;
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

    fn mean_active_agents(&self) -> f32 {
        self.active_agents_sum as f32 / self.frames.max(1) as f32
    }

    fn mean_active_spikes(&self) -> f32 {
        self.active_spikes_sum as f32 / self.frames.max(1) as f32
    }

    fn mean_free_energy(&self) -> f32 {
        self.free_energy_sum / self.frames.max(1) as f32
    }

    fn mean_stability(&self) -> f32 {
        self.stability_sum / self.frames.max(1) as f32
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
        (0..PATTERN_WIDTH)
            .map(|offset| start + offset * 2)
            .collect()
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
    let pure_snga = train_snga(&cases, &symbols, false);
    let contextual_snga = train_snga(&cases, &symbols, false);
    let hybrid = train_snga(&cases, &symbols, true);
    let rqf = train_rqf(&cases, &symbols);

    let pure_snga_metrics = evaluate_snga(&pure_snga, &cases, &symbols, EvalMode::CueOnly);
    let rqf_metrics = evaluate_rqf(&rqf, &cases, &symbols);
    let contextual_snga_metrics =
        evaluate_snga(&contextual_snga, &cases, &symbols, EvalMode::CueAndContext);
    let hybrid_metrics = evaluate_snga(&hybrid, &cases, &symbols, EvalMode::HybridRelational);

    println!("SNGA + RQF hybrid relational benchmark");
    println!(
        "cases={} frames={} epochs={} symbols={} hybrid_relations={}",
        cases.len(),
        cases.len() * 2,
        TRAINING_EPOCHS,
        symbols.ids.len(),
        hybrid.relational_relation_count()
    );
    print_metrics("1_snga_puro", &pure_snga_metrics);
    print_metrics("2_rqf_puro", &rqf_metrics);
    print_metrics("3_snga_contexto_explicito", &contextual_snga_metrics);
    print_metrics("4_snga_rqf_integrado", &hybrid_metrics);
    println!(
        "lectura: {}",
        if hybrid_metrics.accuracy() >= contextual_snga_metrics.accuracy()
            && hybrid_metrics.mean_leakage() < contextual_snga_metrics.mean_leakage()
            && hybrid_metrics.mean_active_agents() <= contextual_snga_metrics.mean_active_agents()
        {
            "SNGA+RQF conserva la memoria geometrica, reduce fuga y usa menos actividad que el contexto explicito"
        } else {
            "SNGA+RQF ya integra la modulacion relacional, pero requiere ajustar acoplamientos para superar todas las metricas"
        }
    );
}

fn train_snga(
    cases: &[AmbiguousCase],
    symbols: &SymbolTable,
    enable_rqf: bool,
) -> SimplicialNetwork {
    let mut network = SimplicialNetwork::grid_3d(config(symbols.ids.len()), 2);
    if enable_rqf {
        network.enable_relational_field(relational_config());
    }
    for _ in 0..TRAINING_EPOCHS {
        for case in cases {
            train_snga_frame(
                &mut network,
                symbols,
                case.cue,
                case.left,
                case.right,
                enable_rqf,
            );
            train_snga_frame(
                &mut network,
                symbols,
                case.cue,
                case.right,
                case.left,
                enable_rqf,
            );
        }
    }
    network
}

fn train_snga_frame(
    network: &mut SimplicialNetwork,
    symbols: &SymbolTable,
    cue: &'static str,
    frame: FrameSpec,
    competing_frame: FrameSpec,
    enable_rqf: bool,
) {
    let cue_pattern = symbols.pattern(cue);
    let context_pattern = symbols.pattern(frame.context);
    let frame_pattern = symbols.frame_pattern(frame);
    let competing_pattern = symbols.frame_pattern(competing_frame);
    let mut fused = cue_pattern.clone();
    fused.extend(context_pattern.iter().copied());
    fused.extend(frame_pattern.iter().copied());
    fused.sort_unstable();
    fused.dedup();

    network.learn_transition(&cue_pattern, &frame_pattern);
    network.learn_transition(&context_pattern, &frame_pattern);
    network.reinforce_coactivation_if_useful(&fused, 0.04, 0.92);

    if enable_rqf {
        network.reinforce_relational_links(
            frame.observer,
            &cue_pattern,
            &frame_pattern,
            frame.phase,
            1.0,
        );
        network.reinforce_relational_pattern(frame.observer, &frame_pattern, 0.0, 1.0);
        network.reinforce_relational_links(
            frame.observer,
            &cue_pattern,
            &competing_pattern,
            frame.phase + PI,
            0.20,
        );
        network.reinforce_relational_pattern(
            frame.observer,
            &competing_pattern,
            frame.phase + PI,
            0.20,
        );
    }
}

fn train_rqf(cases: &[AmbiguousCase], symbols: &SymbolTable) -> RelationalFieldSubstrate {
    let mut rqf = RelationalFieldSubstrate::new(relational_config());
    for _ in 0..TRAINING_EPOCHS {
        for case in cases {
            train_rqf_frame(&mut rqf, symbols, case.cue, case.left, case.right);
            train_rqf_frame(&mut rqf, symbols, case.cue, case.right, case.left);
        }
    }
    rqf
}

fn train_rqf_frame(
    rqf: &mut RelationalFieldSubstrate,
    symbols: &SymbolTable,
    cue: &'static str,
    frame: FrameSpec,
    competing_frame: FrameSpec,
) {
    let cue = symbols.id(cue);
    let targets = [
        symbols.id(frame.primary),
        symbols.id(frame.support_a),
        symbols.id(frame.support_b),
    ];
    let competitors = [
        symbols.id(competing_frame.primary),
        symbols.id(competing_frame.support_a),
        symbols.id(competing_frame.support_b),
    ];
    for target in targets {
        rqf.reinforce_relation(frame.observer, cue, target, frame.phase, 1.0);
    }
    rqf.reinforce_relation(frame.observer, targets[0], targets[1], 0.0, 1.0);
    rqf.reinforce_relation(frame.observer, targets[1], targets[2], 0.0, 1.0);
    rqf.reinforce_relation(frame.observer, targets[2], targets[0], 0.0, 1.0);
    for competitor in competitors {
        rqf.reinforce_relation(frame.observer, cue, competitor, frame.phase + PI, 0.20);
    }
    rqf.reinforce_relation(
        frame.observer,
        competitors[0],
        competitors[1],
        frame.phase + PI,
        0.20,
    );
    rqf.reinforce_relation(
        frame.observer,
        competitors[1],
        competitors[2],
        frame.phase + PI,
        0.20,
    );
}

#[derive(Clone, Copy)]
enum EvalMode {
    CueOnly,
    CueAndContext,
    HybridRelational,
}

fn evaluate_snga(
    network: &SimplicialNetwork,
    cases: &[AmbiguousCase],
    symbols: &SymbolTable,
    mode: EvalMode,
) -> Metrics {
    let mut metrics = Metrics::default();
    for case in cases {
        evaluate_snga_frame(
            network,
            symbols,
            case.cue,
            case.left,
            case.right,
            mode,
            &mut metrics,
        );
        evaluate_snga_frame(
            network,
            symbols,
            case.cue,
            case.right,
            case.left,
            mode,
            &mut metrics,
        );
    }
    metrics
}

fn evaluate_snga_frame(
    network: &SimplicialNetwork,
    symbols: &SymbolTable,
    cue: &'static str,
    frame: FrameSpec,
    competing_frame: FrameSpec,
    mode: EvalMode,
    metrics: &mut Metrics,
) {
    let mut trial = network.clone();
    trial.clear_activity();
    let mut query = symbols.pattern(cue);
    if matches!(mode, EvalMode::CueAndContext) {
        query.extend(symbols.pattern(frame.context));
        query.sort_unstable();
        query.dedup();
    }
    if matches!(mode, EvalMode::HybridRelational) {
        trial.set_relational_observer(frame.observer, frame.phase);
    }
    trial.inject_pattern(&query, 1.2, 3);
    let mut active_spikes = 0;
    let mut last_energy = 0.0;
    for _ in 0..EVAL_STEPS {
        let stats = trial.step();
        active_spikes += stats.active_spikes;
        last_energy = stats.total_free_energy;
    }
    let before_stability = active_set(&trial, 0.10);
    let next_energy = trial.step().total_free_energy;
    let after_stability = active_set(&trial, 0.10);
    let stability = jaccard(&before_stability, &after_stability)
        * (1.0 - ((next_energy - last_energy).abs() / (last_energy.abs() + 1.0)).min(1.0));

    let expected = symbols.frame_pattern(frame);
    let distractors = symbols.frame_pattern(competing_frame);
    let expected_score = surprise_score(&trial, &expected);
    let distractor_score = surprise_score(&trial, &distractors);
    let incompatible_tension = if matches!(mode, EvalMode::HybridRelational) {
        trial
            .relational_simplex_phase_report(
                frame.observer,
                symbols.pattern(cue)[0],
                symbols.pattern(competing_frame.primary)[0],
                symbols.pattern(competing_frame.support_a)[0],
            )
            .map(|report| report.tension)
            .unwrap_or(0.0)
    } else {
        0.0
    };

    metrics.record(
        expected_score,
        distractor_score,
        trial
            .agents
            .iter()
            .filter(|agent| agent.surprise > 0.0)
            .count(),
        active_spikes,
        last_energy,
        stability,
        incompatible_tension,
    );
}

fn evaluate_rqf(
    rqf: &RelationalFieldSubstrate,
    cases: &[AmbiguousCase],
    symbols: &SymbolTable,
) -> Metrics {
    let mut rqf = rqf.clone();
    let mut metrics = Metrics::default();
    for case in cases {
        evaluate_rqf_frame(
            &mut rqf,
            symbols,
            case.cue,
            case.left,
            case.right,
            &mut metrics,
        );
        evaluate_rqf_frame(
            &mut rqf,
            symbols,
            case.cue,
            case.right,
            case.left,
            &mut metrics,
        );
    }
    metrics
}

fn evaluate_rqf_frame(
    rqf: &mut RelationalFieldSubstrate,
    symbols: &SymbolTable,
    cue: &'static str,
    frame: FrameSpec,
    competing_frame: FrameSpec,
    metrics: &mut Metrics,
) {
    let report = rqf.observe_pattern(frame.observer, &[symbols.id(cue)], frame.phase, 8);
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
    let expected_score = report_score(&report, &expected);
    let distractor_score = report_score(&report, &distractors);
    let incompatible_tension = rqf
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
        report.candidates.len(),
        0,
        0.0,
        1.0,
        incompatible_tension,
    );
}

fn surprise_score(network: &SimplicialNetwork, targets: &[usize]) -> f32 {
    targets
        .iter()
        .filter_map(|&idx| network.agents.get(idx))
        .map(|agent| agent.surprise)
        .sum()
}

fn report_score(report: &snga::relational_field::CollapseReport, targets: &[usize]) -> f32 {
    report
        .candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn active_set(network: &SimplicialNetwork, threshold: f32) -> Vec<usize> {
    network
        .agents
        .iter()
        .filter(|agent| agent.surprise >= threshold)
        .map(|agent| agent.id)
        .collect()
}

fn jaccard(left: &[usize], right: &[usize]) -> f32 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let intersection = left.iter().filter(|idx| right.contains(idx)).count();
    let union = left.len() + right.len() - intersection;
    intersection as f32 / union.max(1) as f32
}

fn print_metrics(label: &str, metrics: &Metrics) {
    println!(
        "{}: frames={} accuracy={:.1}% purity={:.1}% leakage={:.1}% margin={:.3} active_agents={:.1} spikes={:.1} energy={:.3} stability={:.3} incompatible_tension={:.3}",
        label,
        metrics.frames,
        metrics.accuracy() * 100.0,
        metrics.mean_purity() * 100.0,
        metrics.mean_leakage() * 100.0,
        metrics.mean_margin(),
        metrics.mean_active_agents(),
        metrics.mean_active_spikes(),
        metrics.mean_free_energy(),
        metrics.mean_stability(),
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
        amplitude_decay: 0.001,
        coherence_decay: 0.0005,
        uncertainty_recovery: 0.002,
        activation_threshold: 0.03,
    }
}

fn config(symbol_count: usize) -> SimplicialConfig {
    let width = 64;
    let height = ((symbol_count * PATTERN_STRIDE) / width + 6).max(20);
    SimplicialConfig {
        width,
        height,
        spacing: 8.0,
        elasticity: 0.006,
        damping: 0.88,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0002,
        max_active_agents: 40,
        inhibition_decay: 0.05,
        max_spikes_per_step: 96,
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
        seed: 912,
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
