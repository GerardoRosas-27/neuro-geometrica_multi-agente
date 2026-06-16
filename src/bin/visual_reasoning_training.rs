use macroquad::prelude::*;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const CAUSAL_CHAINS: usize = 5_000;
const HIERARCHY_CHAINS: usize = 3_000;
const CONTRADICTIONS: usize = 3_000;
const EVAL_CAUSAL: usize = 600;
const EVAL_HIERARCHY: usize = 400;
const EVAL_CONTRADICTIONS: usize = 400;
const PATTERN_SIZE: usize = 5;
const TRANSITIVE_LIMIT: usize = 512;
const TRAIN_BATCH: usize = 60;
const EVAL_BATCH: usize = 12;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    TrainCausal,
    TrainHierarchy,
    TrainContradiction,
    EvaluateCausal,
    EvaluateHierarchy,
    EvaluateContradiction,
    Done,
}

#[derive(Default)]
struct ReasoningAggregate {
    direct_recall: f32,
    broad_recall: f32,
    broad_precision: f32,
    optimized_recall: f32,
    optimized_precision: f32,
    rewarded_paths: usize,
    evaporated_paths: usize,
    samples: usize,
}

#[derive(Default)]
struct ContradictionAggregate {
    tension: f32,
    energy_delta: f32,
    samples: usize,
}

struct VisualExperiment {
    network: SimplicialNetwork,
    phase: Phase,
    cursor: usize,
    causal: ReasoningAggregate,
    hierarchy: ReasoningAggregate,
    contradiction: ContradictionAggregate,
    final_frames: u32,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "SNGA - entrenamiento visual de razonamiento topologico".to_string(),
        window_width: 1280,
        window_height: 820,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut experiment = VisualExperiment {
        network: SimplicialNetwork::grid(benchmark_config()),
        phase: Phase::TrainCausal,
        cursor: 0,
        causal: ReasoningAggregate::default(),
        hierarchy: ReasoningAggregate::default(),
        contradiction: ContradictionAggregate::default(),
        final_frames: 0,
    };

    loop {
        for _ in 0..work_units_per_frame() {
            experiment.step();
            if experiment.phase == Phase::Done {
                break;
            }
        }

        draw_experiment(&experiment);
        next_frame().await;

        if experiment.phase == Phase::Done {
            experiment.final_frames += 1;
            if experiment.final_frames > 600 {
                break;
            }
        }
    }
}

impl VisualExperiment {
    fn step(&mut self) {
        match self.phase {
            Phase::TrainCausal => {
                let end = (self.cursor + TRAIN_BATCH).min(CAUSAL_CHAINS);
                for chain_id in self.cursor..end {
                    train_causal_chain(&mut self.network, chain_id);
                }
                self.advance_or_switch(end, CAUSAL_CHAINS, Phase::TrainHierarchy);
            }
            Phase::TrainHierarchy => {
                let end = (self.cursor + TRAIN_BATCH).min(HIERARCHY_CHAINS);
                for chain_id in self.cursor..end {
                    train_hierarchy(&mut self.network, chain_id);
                }
                self.advance_or_switch(end, HIERARCHY_CHAINS, Phase::TrainContradiction);
            }
            Phase::TrainContradiction => {
                let end = (self.cursor + TRAIN_BATCH).min(CONTRADICTIONS);
                for pair_id in self.cursor..end {
                    train_contradiction(&mut self.network, pair_id);
                }
                self.advance_or_switch(end, CONTRADICTIONS, Phase::EvaluateCausal);
            }
            Phase::EvaluateCausal => {
                let end = (self.cursor + EVAL_BATCH).min(EVAL_CAUSAL);
                for chain_id in self.cursor..end {
                    evaluate_causal(&mut self.network, chain_id, &mut self.causal);
                }
                self.advance_or_switch(end, EVAL_CAUSAL, Phase::EvaluateHierarchy);
            }
            Phase::EvaluateHierarchy => {
                let end = (self.cursor + EVAL_BATCH).min(EVAL_HIERARCHY);
                for chain_id in self.cursor..end {
                    evaluate_hierarchy(&mut self.network, chain_id, &mut self.hierarchy);
                }
                self.advance_or_switch(end, EVAL_HIERARCHY, Phase::EvaluateContradiction);
            }
            Phase::EvaluateContradiction => {
                let end = (self.cursor + EVAL_BATCH).min(EVAL_CONTRADICTIONS);
                for pair_id in self.cursor..end {
                    evaluate_contradiction(&mut self.network, pair_id, &mut self.contradiction);
                }
                self.advance_or_switch(end, EVAL_CONTRADICTIONS, Phase::Done);
            }
            Phase::Done => {}
        }
    }

    fn advance_or_switch(&mut self, end: usize, total: usize, next: Phase) {
        self.cursor = end;
        if self.cursor >= total {
            self.cursor = 0;
            self.phase = next;
        }
    }

    fn progress(&self) -> f32 {
        let total = phase_total(self.phase).max(1);
        self.cursor as f32 / total as f32
    }
}

fn work_units_per_frame() -> usize {
    if is_key_down(KeyCode::LeftShift) {
        8
    } else {
        2
    }
}

fn train_causal_chain(network: &mut SimplicialNetwork, chain_id: usize) {
    let a = pattern("causal", chain_id, 0, network.agents.len());
    let b = pattern("causal", chain_id, 1, network.agents.len());
    let c = pattern("causal", chain_id, 2, network.agents.len());
    let d = pattern("causal", chain_id, 3, network.agents.len());

    network.learn_transition(&a, &b);
    network.learn_transition(&b, &c);
    network.learn_transition(&c, &d);
    reinforce_states(network, [&a, &b, &c, &d]);
}

fn train_hierarchy(network: &mut SimplicialNetwork, chain_id: usize) {
    let leaf = pattern("hierarchy", chain_id, 0, network.agents.len());
    let parent = pattern("hierarchy", chain_id, 1, network.agents.len());
    let root = pattern("hierarchy", chain_id, 2, network.agents.len());

    network.learn_transition(&leaf, &parent);
    network.learn_transition(&parent, &root);
    reinforce_states(network, [&leaf, &parent, &root]);
}

fn train_contradiction(network: &mut SimplicialNetwork, pair_id: usize) {
    let left = pattern("contradiction", pair_id, 0, network.agents.len());
    let right = pattern("contradiction", pair_id, 1, network.agents.len());

    network.learn_contradiction(&left, &right);
    reinforce_states(network, [&left, &right]);
}

fn reinforce_states<const N: usize>(network: &mut SimplicialNetwork, states: [&Vec<usize>; N]) {
    for state in states {
        network.reinforce_coactivation(state, 0.08);
    }
}

fn evaluate_causal(
    network: &mut SimplicialNetwork,
    chain_id: usize,
    aggregate: &mut ReasoningAggregate,
) {
    let a = pattern("causal", chain_id, 0, network.agents.len());
    let d = pattern("causal", chain_id, 3, network.agents.len());
    let direct = network.evaluate_prediction(&a, &d, d.len());
    let broad = network.evaluate_transitive_prediction(&a, &d, 3, TRANSITIVE_LIMIT);
    let optimized = network.optimize_routes_to_expected(&a, &d, 3, TRANSITIVE_LIMIT, 0.08, 0.04);

    aggregate.direct_recall += direct.recall;
    aggregate.broad_recall += broad.recall;
    aggregate.broad_precision += broad.precision;
    aggregate.optimized_recall += optimized.prediction.recall;
    aggregate.optimized_precision += optimized.prediction.precision;
    aggregate.rewarded_paths += optimized.rewarded_paths;
    aggregate.evaporated_paths += optimized.evaporated_paths;
    aggregate.samples += 1;
}

fn evaluate_hierarchy(
    network: &mut SimplicialNetwork,
    chain_id: usize,
    aggregate: &mut ReasoningAggregate,
) {
    let leaf = pattern("hierarchy", chain_id, 0, network.agents.len());
    let root = pattern("hierarchy", chain_id, 2, network.agents.len());
    let direct = network.evaluate_prediction(&leaf, &root, root.len());
    let broad = network.evaluate_transitive_prediction(&leaf, &root, 2, TRANSITIVE_LIMIT);
    let optimized =
        network.optimize_routes_to_expected(&leaf, &root, 2, TRANSITIVE_LIMIT, 0.08, 0.04);

    aggregate.direct_recall += direct.recall;
    aggregate.broad_recall += broad.recall;
    aggregate.broad_precision += broad.precision;
    aggregate.optimized_recall += optimized.prediction.recall;
    aggregate.optimized_precision += optimized.prediction.precision;
    aggregate.rewarded_paths += optimized.rewarded_paths;
    aggregate.evaporated_paths += optimized.evaporated_paths;
    aggregate.samples += 1;
}

fn evaluate_contradiction(
    network: &mut SimplicialNetwork,
    pair_id: usize,
    aggregate: &mut ContradictionAggregate,
) {
    let left = pattern("contradiction", pair_id, 0, network.agents.len());
    let right = pattern("contradiction", pair_id, 1, network.agents.len());
    let tension = network.contradiction_tension(&left, &right);
    aggregate.tension += tension;
    aggregate.energy_delta += tension * network.config.contradiction_energy_weight;
    aggregate.samples += 1;
}

fn draw_experiment(experiment: &VisualExperiment) {
    clear_background(Color::from_rgba(5, 7, 13, 255));
    draw_header(experiment);
    draw_route_panel(experiment);
    draw_metrics_panel(experiment);
    draw_network_stats(experiment);
    draw_footer();
}

fn draw_header(experiment: &VisualExperiment) {
    let phase = phase_name(experiment.phase);
    draw_text(
        "SNGA - Entrenamiento visual de rutas neuro-geometricas",
        24.0,
        34.0,
        28.0,
        WHITE,
    );
    draw_text(
        &format!(
            "fase: {phase} | progreso: {:.1}% | Shift = acelerar | auto-cierre tras finalizar",
            experiment.progress() * 100.0
        ),
        24.0,
        64.0,
        22.0,
        Color::from_rgba(200, 220, 240, 255),
    );
    draw_progress_bar(24.0, 80.0, 720.0, 16.0, experiment.progress(), BLUE);
}

fn draw_route_panel(experiment: &VisualExperiment) {
    let y = 170.0;
    draw_text(
        "Proceso de razonamiento: exploracion -> evaporacion -> ruta optima",
        24.0,
        y - 36.0,
        24.0,
        WHITE,
    );

    let nodes = [
        ("A", 100.0, y),
        ("B", 310.0, y),
        ("C", 520.0, y),
        ("D", 730.0, y),
    ];

    for window in nodes.windows(2) {
        let (_, x1, y1) = window[0];
        let (_, x2, y2) = window[1];
        draw_line(
            x1 + 26.0,
            y1,
            x2 - 26.0,
            y2,
            5.0,
            Color::from_rgba(60, 230, 160, 230),
        );
    }

    let noisy_routes = if experiment.phase == Phase::EvaluateCausal
        || experiment.phase == Phase::EvaluateHierarchy
    {
        10
    } else {
        4
    };
    for i in 0..noisy_routes {
        let offset = (i as f32 - noisy_routes as f32 / 2.0) * 9.0;
        draw_line(
            126.0,
            y,
            704.0,
            y + offset,
            1.0,
            Color::from_rgba(180, 80, 80, 80),
        );
    }

    for (label, x, y) in nodes {
        draw_circle(x, y, 28.0, Color::from_rgba(255, 180, 70, 255));
        draw_text(label, x - 8.0, y + 8.0, 28.0, BLACK);
    }

    draw_text(
        "rutas candidatas",
        92.0,
        y + 70.0,
        18.0,
        Color::from_rgba(200, 100, 100, 255),
    );
    draw_text(
        "ruta reforzada",
        520.0,
        y + 70.0,
        18.0,
        Color::from_rgba(80, 255, 180, 255),
    );
}

fn draw_metrics_panel(experiment: &VisualExperiment) {
    let x = 820.0;
    let y = 110.0;
    draw_rectangle(
        x - 20.0,
        y - 45.0,
        420.0,
        365.0,
        Color::from_rgba(14, 20, 32, 235),
    );
    draw_text("Metricas reales agregadas", x, y - 14.0, 24.0, WHITE);

    metric_block("Causal", &experiment.causal, x, y + 20.0);
    metric_block("Jerarquia", &experiment.hierarchy, x, y + 145.0);

    let n = experiment.contradiction.samples.max(1) as f32;
    draw_text(
        &format!(
            "Contradiccion: tension={:.3} deltaE={:.3} muestras={}",
            experiment.contradiction.tension / n,
            experiment.contradiction.energy_delta / n,
            experiment.contradiction.samples
        ),
        x,
        y + 285.0,
        20.0,
        Color::from_rgba(240, 210, 120, 255),
    );
}

fn metric_block(title: &str, aggregate: &ReasoningAggregate, x: f32, y: f32) {
    let n = aggregate.samples.max(1) as f32;
    let broad_precision = aggregate.broad_precision / n;
    let optimized_precision = aggregate.optimized_precision / n;
    let optimized_recall = aggregate.optimized_recall / n;

    draw_text(
        &format!("{title}: muestras={}", aggregate.samples),
        x,
        y,
        20.0,
        Color::from_rgba(220, 230, 255, 255),
    );
    draw_metric_bar("precision amplia", broad_precision, x, y + 24.0, RED);
    draw_metric_bar(
        "precision optimizada",
        optimized_precision,
        x,
        y + 54.0,
        GREEN,
    );
    draw_metric_bar("recall optimizado", optimized_recall, x, y + 84.0, SKYBLUE);
    draw_text(
        &format!(
            "reforzadas={} evaporadas={}",
            aggregate.rewarded_paths, aggregate.evaporated_paths
        ),
        x,
        y + 112.0,
        18.0,
        GRAY,
    );
}

fn draw_network_stats(experiment: &VisualExperiment) {
    let stats = experiment.network.plasticity_stats();
    let y = 520.0;
    draw_text("Estado estructural del nucleo", 24.0, y, 24.0, WHITE);
    let lines = [
        format!("nodos: {}", experiment.network.agents.len()),
        format!("aristas activas: {}", stats.active_edges),
        format!("aristas asociativas: {}", stats.associative_edges),
        format!("aristas causales: {}", stats.causal_edges),
        format!("contradicciones: {}", stats.contradiction_edges),
        format!("episodios: {}", stats.episodes),
    ];
    for (i, line) in lines.iter().enumerate() {
        draw_text(
            line,
            30.0,
            y + 35.0 + i as f32 * 24.0,
            20.0,
            Color::from_rgba(210, 225, 240, 255),
        );
    }
}

fn draw_footer() {
    draw_text(
        "Nota: se visualiza una proyeccion conceptual; el entrenamiento usa el grafo SNGA real del benchmark.",
        24.0,
        screen_height() - 28.0,
        18.0,
        Color::from_rgba(180, 190, 205, 255),
    );
}

fn draw_progress_bar(x: f32, y: f32, w: f32, h: f32, value: f32, color: Color) {
    draw_rectangle(x, y, w, h, Color::from_rgba(45, 55, 70, 255));
    draw_rectangle(x, y, w * value.clamp(0.0, 1.0), h, color);
}

fn draw_metric_bar(label: &str, value: f32, x: f32, y: f32, color: Color) {
    draw_text(
        label,
        x,
        y + 15.0,
        17.0,
        Color::from_rgba(210, 215, 225, 255),
    );
    draw_progress_bar(x + 150.0, y + 2.0, 190.0, 15.0, value, color);
    draw_text(
        &format!("{:.1}%", value * 100.0),
        x + 350.0,
        y + 15.0,
        17.0,
        WHITE,
    );
}

fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::TrainCausal => "entrenando cadenas causales",
        Phase::TrainHierarchy => "entrenando jerarquias",
        Phase::TrainContradiction => "entrenando contradicciones",
        Phase::EvaluateCausal => "evaluando/optimizando rutas causales",
        Phase::EvaluateHierarchy => "evaluando/optimizando rutas jerarquicas",
        Phase::EvaluateContradiction => "evaluando tension de contradiccion",
        Phase::Done => "finalizado",
    }
}

fn phase_total(phase: Phase) -> usize {
    match phase {
        Phase::TrainCausal => CAUSAL_CHAINS,
        Phase::TrainHierarchy => HIERARCHY_CHAINS,
        Phase::TrainContradiction => CONTRADICTIONS,
        Phase::EvaluateCausal => EVAL_CAUSAL,
        Phase::EvaluateHierarchy => EVAL_HIERARCHY,
        Phase::EvaluateContradiction => EVAL_CONTRADICTIONS,
        Phase::Done => 1,
    }
}

fn pattern(domain: &str, id: usize, role: usize, nodes: usize) -> Vec<usize> {
    (0..PATTERN_SIZE)
        .map(|term| {
            let mut hasher = DefaultHasher::new();
            domain.hash(&mut hasher);
            id.hash(&mut hasher);
            role.hash(&mut hasher);
            term.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn benchmark_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 720,
        height: 360,
        spacing: 3.0,
        elasticity: 0.0025,
        damping: 0.88,
        activation_threshold: 0.66,
        simplex_area_weight: 0.00008,
        max_active_agents: 64,
        inhibition_decay: 0.02,
        max_spikes_per_step: 256,
        local_inhibition_decay: 0.80,
        refractory_ticks: 0,
        rhythm_period: 32,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 4,
        consolidated_forgetting_scale: 0.2,
        max_episodes: 256,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.18,
        contradiction_learning_rate: 0.25,
        contradiction_energy_weight: 4.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 53,
    }
}
