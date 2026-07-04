use cdt_rqm_epr::relational_field::{
    CollapseReport, ObserverId, RelationalFieldConfig, RelationalFieldSubstrate,
};
use std::f32::consts::{FRAC_PI_2, PI};

const BANCO: usize = 1;
const DINERO: usize = 2;
const CREDITO: usize = 3;
const SENTARSE: usize = 4;
const PARQUE: usize = 5;
const INSTITUCION: usize = 6;
const MADERA: usize = 7;

const FINANCIAL_OBSERVER: ObserverId = ObserverId(10);
const PARK_OBSERVER: ObserverId = ObserverId(20);
const NEUTRAL_OBSERVER: ObserverId = ObserverId(30);

fn main() {
    let mut substrate = RelationalFieldSubstrate::new(config());

    for _ in 0..16 {
        train_financial_frame(&mut substrate);
        train_park_frame(&mut substrate);
        train_neutral_ambiguity(&mut substrate);
    }

    let financial = substrate.observe_pattern(FINANCIAL_OBSERVER, &[BANCO], 0.0, 5);
    let park = substrate.observe_pattern(PARK_OBSERVER, &[BANCO], FRAC_PI_2, 5);
    let neutral = substrate.observe_pattern(NEUTRAL_OBSERVER, &[BANCO], 0.0, 5);

    println!("CDT-RQM-EPR relational field substrate experiment");
    println!("relations={}", substrate.relation_count());
    print_report("observador_financiero", &financial);
    print_report("observador_parque", &park);
    print_report("observador_neutral", &neutral);

    if let Some(simplex) =
        substrate.simplex_phase_report(FINANCIAL_OBSERVER, BANCO, DINERO, CREDITO)
    {
        println!(
            "simplex_financiero banco-dinero-credito: closure={:.3} coherence={:.3} tension={:.3}",
            simplex.phase_closure, simplex.coherence, simplex.tension
        );
    }
    if let Some(simplex) = substrate.simplex_phase_report(PARK_OBSERVER, BANCO, SENTARSE, PARQUE) {
        println!(
            "simplex_parque banco-sentarse-parque: closure={:.3} coherence={:.3} tension={:.3}",
            simplex.phase_closure, simplex.coherence, simplex.tension
        );
    }
    if let Some(simplex) =
        substrate.simplex_phase_report(FINANCIAL_OBSERVER, BANCO, SENTARSE, PARQUE)
    {
        println!(
            "simplex_incompatible_financiero banco-sentarse-parque: closure={:.3} coherence={:.3} tension={:.3}",
            simplex.phase_closure, simplex.coherence, simplex.tension
        );
    }

    let financial_top = top_agent(&financial);
    let park_top = top_agent(&park);
    println!(
        "lectura: {}",
        if financial_top == Some(DINERO) && park_top == Some(SENTARSE) {
            "el significado no queda en el nodo banco; colapsa distinto segun el observador relacional"
        } else {
            "el campo relacional funciona, pero requiere ajustar fases/amplitudes para separar mejor los marcos"
        }
    );
}

fn train_financial_frame(substrate: &mut RelationalFieldSubstrate) {
    reinforce_cycle(
        substrate,
        FINANCIAL_OBSERVER,
        [BANCO, DINERO, CREDITO],
        [0.0, 0.0, 0.0],
        1.0,
    );
    substrate.reinforce_relation(FINANCIAL_OBSERVER, BANCO, INSTITUCION, 0.0, 1.0);
    substrate.reinforce_relation(FINANCIAL_OBSERVER, BANCO, SENTARSE, PI, 0.35);
    substrate.reinforce_relation(FINANCIAL_OBSERVER, BANCO, PARQUE, PI, 0.25);
    substrate.reinforce_relation(FINANCIAL_OBSERVER, SENTARSE, PARQUE, PI, 0.25);
}

fn train_park_frame(substrate: &mut RelationalFieldSubstrate) {
    reinforce_cycle(
        substrate,
        PARK_OBSERVER,
        [BANCO, SENTARSE, PARQUE],
        [FRAC_PI_2, FRAC_PI_2, -PI],
        1.0,
    );
    substrate.reinforce_relation(PARK_OBSERVER, BANCO, MADERA, FRAC_PI_2, 1.0);
    substrate.reinforce_relation(PARK_OBSERVER, BANCO, DINERO, -FRAC_PI_2, 0.25);
    substrate.reinforce_relation(PARK_OBSERVER, DINERO, CREDITO, -FRAC_PI_2, 0.25);
}

fn train_neutral_ambiguity(substrate: &mut RelationalFieldSubstrate) {
    substrate.reinforce_relation(NEUTRAL_OBSERVER, BANCO, DINERO, 0.0, 0.70);
    substrate.reinforce_relation(NEUTRAL_OBSERVER, BANCO, SENTARSE, 0.0, 0.70);
    substrate.reinforce_relation(NEUTRAL_OBSERVER, BANCO, INSTITUCION, 0.0, 0.55);
    substrate.reinforce_relation(NEUTRAL_OBSERVER, BANCO, PARQUE, 0.0, 0.55);
}

fn reinforce_cycle(
    substrate: &mut RelationalFieldSubstrate,
    observer: ObserverId,
    vertices: [usize; 3],
    phases: [f32; 3],
    success: f32,
) {
    substrate.reinforce_relation(observer, vertices[0], vertices[1], phases[0], success);
    substrate.reinforce_relation(observer, vertices[1], vertices[2], phases[1], success);
    substrate.reinforce_relation(observer, vertices[2], vertices[0], phases[2], success);
}

fn print_report(label: &str, report: &CollapseReport) {
    println!(
        "{}: total_interference={:.3} mean_coherence={:.3} mean_uncertainty={:.3}",
        label, report.total_interference, report.mean_coherence, report.mean_uncertainty
    );
    for candidate in &report.candidates {
        println!(
            "  -> {} score={:.3} interference={:.3} probability={:.3} coherence={:.3} uncertainty={:.3}",
            name(candidate.agent),
            candidate.score,
            candidate.interference,
            candidate.probability,
            candidate.mean_coherence,
            candidate.mean_uncertainty
        );
    }
}

fn top_agent(report: &CollapseReport) -> Option<usize> {
    report.candidates.first().map(|candidate| candidate.agent)
}

fn name(agent: usize) -> &'static str {
    match agent {
        BANCO => "banco",
        DINERO => "dinero",
        CREDITO => "credito",
        SENTARSE => "sentarse",
        PARQUE => "parque",
        INSTITUCION => "institucion",
        MADERA => "madera",
        _ => "desconocido",
    }
}

fn config() -> RelationalFieldConfig {
    RelationalFieldConfig {
        amplitude_learning_rate: 0.09,
        phase_learning_rate: 0.22,
        coherence_learning_rate: 0.12,
        uncertainty_learning_rate: 0.10,
        amplitude_decay: 0.002,
        coherence_decay: 0.001,
        uncertainty_recovery: 0.004,
        activation_threshold: 0.05,
    }
}
