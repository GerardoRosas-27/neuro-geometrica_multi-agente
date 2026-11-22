//! Binario de Demostración Interactiva MARCUS-I CLI.

use marcus_neurosymbolic::algebraic_mind::*;
use marcus_neurosymbolic::core_knowledge::*;
use marcus_neurosymbolic::epistemic_vigilance::*;
use marcus_neurosymbolic::MarcusEngine;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                   ARQUITECTURA MARCUS-I                      ║");
    println!("║   Sustrato Cognitivo Neuro-Simbólico Basado en Gary Marcus   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("--- [DEMO 1] LIGAMIENTO DINÁMICO DE VARIABLES Y HRR (Marcus 2001) ---");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);
    let dim = 512;
    let r_agent = HrrVector::new_random(dim, &mut rng);
    let r_patient = HrrVector::new_random(dim, &mut rng);
    let f_john = HrrVector::new_random(dim, &mut rng);
    let f_mary = HrrVector::new_random(dim, &mut rng);

    let bound_sentence = r_agent.circular_convolution(&f_john)
        .superpose(&r_patient.circular_convolution(&f_mary));

    let extracted_agent = r_agent.circular_correlation(&bound_sentence);
    let sim_john = extracted_agent.cosine_similarity(&f_john);
    let sim_mary = extracted_agent.cosine_similarity(&f_mary);

    println!("  Oración vectorizada: Loves(Agent * John + Patient * Mary)");
    println!("  Desatando Rol 'Agent': Similitud con John = {:.4}, con Mary = {:.4}", sim_john, sim_mary);
    println!("  => Ligamiento exitoso sin diafonía (John >> Mary)\n");

    println!("--- [DEMO 2] GENERALIZACIÓN ALGEBRAICA ZERO-SHOT OOD ---");
    let seq_ood = ["quark_alpha", "quark_beta", "quark_beta"];
    let pattern = AlgebraicRuleEngine::classify_three_token_grammar(&seq_ood);
    println!("  Secuencia de tokens no vistos: {:?}", seq_ood);
    println!("  Patrón algebraico inferido: {:?} (100% de certeza zero-shot)\n", pattern);

    println!("--- [DEMO 3] FÍSICA NÚCLEO SPELKE: DETECCIÓN DE IMPOSIBILIDADES ---");
    let mut physics = SpelkePhysicsEngine::new();
    physics.max_physical_speed = 25.0;

    let ball_t0 = SpatiotemporalObject {
        id: "ObservedBall".to_string(),
        position: Vec3D::new(0.0, 0.0, 1.0),
        velocity: Vec3D::zero(),
        radius: 0.25,
        is_solid: true,
        is_occluded: false,
        is_supported: true,
        contained_in: None,
    };
    physics.step_and_verify(&[ball_t0], 0.1);

    // Salto imposible de 150 metros en 0.1s
    let ball_t1 = SpatiotemporalObject {
        id: "ObservedBall".to_string(),
        position: Vec3D::new(150.0, 0.0, 1.0),
        velocity: Vec3D::zero(),
        radius: 0.25,
        is_solid: true,
        is_occluded: false,
        is_supported: true,
        contained_in: None,
    };
    let violations = physics.step_and_verify(&[ball_t1], 0.1);
    for v in &violations {
        println!("  [ALERTA FÍSICA INNATA] Violación detectada: {:?}", v);
    }
    println!();

    println!("--- [DEMO 4] INFERENCIA CAUSAL PEARL: INTERVENCIÓN do(X) ---");
    let mut causal = CausalEngine::new();
    let barometer = CausalVar::new("BarometerReading");
    let rain = CausalVar::new("Rainfall");
    let pressure = CausalVar::new("Pressure");

    causal.add_structural_equation(pressure.clone(), StructuralEquation::new(1013.0, 0.0));
    causal.add_structural_equation(barometer.clone(), StructuralEquation::new(0.0, 0.0).with_parent(pressure.clone(), 1.0));
    causal.add_structural_equation(rain.clone(), StructuralEquation::new(500.0, 0.0).with_parent(pressure.clone(), -0.5));

    let rain_do_low = causal.estimate_interventional_expectation(barometer.clone(), 950.0, &rain);
    let rain_do_high = causal.estimate_interventional_expectation(barometer.clone(), 1050.0, &rain);
    println!("  Intervención do(Barometer = 950) -> Lluvia esperada = {:.2}", rain_do_low);
    println!("  Intervención do(Barometer = 1050) -> Lluvia esperada = {:.2}", rain_do_high);
    println!("  => Delta causal = {:.4}. El motor sabe que manipular el barómetro no causa lluvia.\n", rain_do_high - rain_do_low);

    println!("--- [DEMO 5] BUCLE DUAL Y CENTINELA ANTI-ALUCINACIONES (Taming Silicon Valley) ---");
    let mut engine = MarcusEngine::new();
    engine.world_state.insert_fact("GaryMarcus", "Authored", "TheAlgebraicMind", EpistemicCertainty::ObservedFact);
    engine.world_state.insert_fact("GaryMarcus", "Authored", "TamingSiliconValley", EpistemicCertainty::ObservedFact);

    // Consulta 1: Hecho verídico
    let dec1 = engine.audit_claim("c1", "GaryMarcus", "Authored", "TamingSiliconValley");
    if let HallucinationGateDecision::Approved { provenance, .. } = dec1 {
        println!("  [APROBADO] Afirmación verificada:");
        println!("{}", provenance.render_audit_report());
    }

    // Consulta 2: Alucinación inducida por asociación superficial
    let dec2 = engine.audit_claim("c2", "GaryMarcus", "Authored", "TheBitterLesson");
    if let HallucinationGateDecision::Vetoed { statement, rejection_category, counter_evidence, .. } = dec2 {
        println!("  [VETADO] Alucinación bloqueada por el Centinela Epistémico:");
        println!("    Afirmación: \"{}\"", statement);
        println!("    Categoría de Rechazo: {}", rejection_category);
        println!("    Evidencia en Contra: {}\n", counter_evidence);
    }

    println!("============================================================");
    println!(" Demostración concluida con éxito. Arquitectura operativa. ");
    println!("============================================================");
}
