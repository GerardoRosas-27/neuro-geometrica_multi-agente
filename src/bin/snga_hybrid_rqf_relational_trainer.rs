use snga::relational_field::{ObserverId, RelationalFieldConfig};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::HashMap;
use std::env;
use std::f32::consts::{FRAC_PI_2, PI};

const DEFAULT_SNGA_OUTPUT: &str = "data/snga_hybrid_rqf_relational.snga";
const DEFAULT_RQF_OUTPUT: &str = "data/snga_hybrid_rqf_relational.rqf";
const PATTERN_STRIDE: usize = 8;
const PATTERN_WIDTH: usize = 4;

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

    fn pattern(&self, symbol: &'static str) -> Vec<usize> {
        let start = self.ids[symbol] * PATTERN_STRIDE;
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
    let snga_output =
        env::var("SNGA_HYBRID_RQF_OUTPUT").unwrap_or_else(|_| DEFAULT_SNGA_OUTPUT.to_string());
    let rqf_output =
        env::var("SNGA_HYBRID_RQF_FIELD_OUTPUT").unwrap_or_else(|_| DEFAULT_RQF_OUTPUT.to_string());
    let epochs = env::var("SNGA_HYBRID_RQF_EPOCHS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(24)
        .max(1);

    let cases = cases();
    let symbols = SymbolTable::from_cases(&cases);
    let mut network = SimplicialNetwork::grid_3d(config(symbols.ids.len()), 2);
    network.enable_relational_field(relational_config());

    println!("SNGA hybrid RQF relational trainer");
    println!(
        "cases={} frames={} symbols={} epochs={} output={} rqf_output={}",
        cases.len(),
        cases.len() * 2,
        symbols.ids.len(),
        epochs,
        snga_output,
        rqf_output
    );

    for epoch in 0..epochs {
        for case in &cases {
            train_frame(&mut network, &symbols, case.cue, case.left, case.right);
            train_frame(&mut network, &symbols, case.cue, case.right, case.left);
        }
        if epoch % 4 == 3 || epoch + 1 == epochs {
            println!(
                "epoch={} edges={} relations={} energy={:.3}",
                epoch + 1,
                network.plasticity_stats().associative_edges,
                network.relational_relation_count(),
                network.total_free_energy()
            );
        }
    }

    match network.save_persistent_state(&snga_output) {
        Ok(report) => println!(
            "snga_saved=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("snga_saved=false error={err}");
            return;
        }
    }

    match network
        .relational_field()
        .expect("relational field enabled")
        .save_persistent_state(&rqf_output)
    {
        Ok(()) => println!(
            "rqf_saved=true relations={} path={}",
            network.relational_relation_count(),
            rqf_output
        ),
        Err(err) => println!("rqf_saved=false error={err}"),
    }
}

fn train_frame(
    network: &mut SimplicialNetwork,
    symbols: &SymbolTable,
    cue: &'static str,
    frame: FrameSpec,
    competing_frame: FrameSpec,
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
    network.reinforce_coactivation_if_useful(&fused, 0.045, 0.94);
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
        seed: 913,
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
