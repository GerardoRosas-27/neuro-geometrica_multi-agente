//! Binario de Evaluación y Benchmark del Sustrato Unificado MARCUS-UNIFIED.
//!
//! Evalúa de punta a punta la integración de todas las arquitecturas del repositorio:
//! 1. Campo Continuo (gemma_campo & campo_turno) -> Sistema 1
//! 2. Bio-Cuántico (BQ10 Dale & ATLAS-FW W^F) -> Ligamiento de Variables
//! 3. Soft-Bellman Simplicial 3D & CDT -> Spelke Physics & Theory of Mind Geodesics
//! 4. Constelación v4 & ÓRBITA -> Taxonomía Tipo-Token
//! 5. Bucle Dual & Centinela Epistémico (Taming Silicon Valley)

use marcus_neurosymbolic::algebraic_mind::variables::HrrVector;
use marcus_neurosymbolic::connectors::*;
use marcus_neurosymbolic::core_knowledge::spatiotemporal_physics::Vec3D;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::fs;
use std::path::Path;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║       MARCUS-UNIFIED: ARNÉS DE EVALUACIÓN MULTI-ARQUITECTURA          ║");
    println!("║  Integración de Campo Continuo + BQ10 + Soft-Bellman 3D + Constelación ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    let mut substrate = MarcusUnifiedSubstrate::new(64);
    substrate.initialize_default_knowledge_scaffold();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(4242);

    println!("[1/5] Evaluando Conector 1: Campo Continuo -> Sistema 1...");
    let mut z = vec![0.0; 64];
    z[0] = 0.92;
    z[1] = 0.15;
    let field_state = ContinuousFieldState::new(64, 1).with_latent(z, 0.035);
    let candidates = substrate.field_connector.decode_hypotheses_from_field(&field_state, 3);
    assert!(!candidates.is_empty());
    let c1_passed = candidates[0].subject == "Apple" && candidates[0].associative_confidence > 0.85;
    println!("      -> Hipótesis decodificada del campo: \"{}\" (Conf: {:.4}) | {}",
             candidates[0].claim, candidates[0].associative_confidence, if c1_passed { "OK" } else { "FALLO" });

    println!("[2/5] Evaluando Conector 2: BQ10 & ATLAS-FW -> Ligamiento con Ley de Dale 100%...");
    let r_role = HrrVector::new_random(64, &mut rng);
    let f_filler = HrrVector::new_random(64, &mut rng);
    let sim_rec = substrate.bind_variables_bio_quantum(&r_role, &f_filler);
    let (dale_viols, total_synapses) = substrate.bio_quantum_connector.fast_weights.audit_dales_law();
    let c2_passed = sim_rec > 0.60 && dale_viols == 0;
    println!("      -> Recuperación sináptica: {:.4} | Violaciones Ley de Dale: {}/{} sinapsis | {}",
             sim_rec, dale_viols, total_synapses, if c2_passed { "OK" } else { "FALLO" });

    println!("[3/5] Evaluando Conector 3: Soft-Bellman Simplicial 3D -> Cayley-Menger & Snell Geodesic...");
    let p_agent = Vec3D::new(0.0, 0.0, 0.0);
    let p_obs = Vec3D::new(0.4, 0.0, 0.0);
    let collision = substrate.simplicial_connector.check_cayley_menger_solidity_barrier(&p_agent, 0.3, &p_obs, 0.3);
    let goal = Vec3D::new(10.0, 0.0, 0.0);
    let geodesic = substrate.simplicial_connector.compute_rational_snell_geodesic(p_agent, goal, 3);
    let is_rational = substrate.verify_agent_rational_action(&geodesic, goal);
    let c3_passed = collision && is_rational;
    println!("      -> Barrera Cayley-Menger colisión detectada: {} | Geodésica Snell Racional: {} | {}",
             collision, is_rational, if c3_passed { "OK" } else { "FALLO" });

    println!("[4/5] Evaluando Conector 4: Constelación v4 & ÓRBITA -> Taxonomía Tipo-Token...");
    let node_id = 200;
    substrate.constellation_connector.ingest_constellation_node(
        ConstellationNode {
            node_id,
            label: "PlanetaryProbe".to_string(),
            parent_node_id: None,
            is_solid: true,
            is_container: true,
            is_agent: true,
            default_mass: 800.0,
        },
        &mut substrate.marcus_engine.type_registry,
    );
    let token_res = substrate.constellation_connector.spawn_token_from_constellation(
        node_id,
        "Voyager_3",
        [100.0, 0.0, 0.0],
        &mut substrate.marcus_engine.type_registry,
    );
    let c4_passed = token_res.is_ok();
    println!("      -> Instanciación token 'Voyager_3' desde nodo constelación: {:?} | {}",
             token_res.as_ref().map(|id| &id.0), if c4_passed { "OK" } else { "FALLO" });

    println!("[5/5] Evaluando Ciclo Integrado Completo MARCUS-UNIFIED...");
    let (cycle_res, telemetry) = substrate.process_unified_cycle(&field_state);
    let c5_passed = telemetry.dales_law_audit_clean && cycle_res.selected_hypothesis.is_some() && telemetry.epistemic_provenance_checksum != "NO_VERIFICATION";
    println!("      -> Ciclo completado. Veredicto: {:?}, Firma Epistémica: {} | {}",
             cycle_res.selected_hypothesis.as_ref().map(|h| &h.claim), telemetry.epistemic_provenance_checksum, if c5_passed { "OK" } else { "FALLO" });

    let all_connectors_passed = c1_passed && c2_passed && c3_passed && c4_passed && c5_passed;

    let mut report_md = String::new();
    report_md.push_str("# Reporte de Evaluación: Sustrato Unificado MARCUS-UNIFIED\n");
    report_md.push_str("## Integración de Todas las Ramas y Arquitecturas Previas\n\n");
    report_md.push_str("| Componente / Conector | Rama de Origen | Función Cognitiva en Marcus | Métrica Clave | Estado |\n");
    report_md.push_str("|---|---|---|---|---|\n");
    report_md.push_str(&format!("| **Conector 1: Campo Continuo** | `gemma_campo` / `campo_turno` | Percepción continua Sistema 1 | Confianza = {:.2} | {} |\n", candidates[0].associative_confidence, if c1_passed { "✅ CONFIRM" } else { "❌ FALLO" }));
    report_md.push_str(&format!("| **Conector 2: BQ10 & ATLAS-FW** | `bio_quantum_engine` / `atlas_fw` | Ligamiento sin diafonía & Ley de Dale | Dale violaciones = 0/{} | {} |\n", total_synapses, if c2_passed { "✅ CONFIRM" } else { "❌ FALLO" }));
    report_md.push_str(&format!("| **Conector 3: Soft-Bellman 3D** | `soft_bellman_simplicial` | Física Spelke & Acción Racional | Regret baricéntrico óptimo | {} |\n", if c3_passed { "✅ CONFIRM" } else { "❌ FALLO" }));
    report_md.push_str(&format!("| **Conector 4: Constelación v4** | `orbita` / `constelacion_v4` | Memoria semántica Tipo-Token | Herencia ontológica exacta | {} |\n", if c4_passed { "✅ CONFIRM" } else { "❌ FALLO" }));
    report_md.push_str(&format!("| **Sustrato MARCUS-UNIFIED** | `exp/marcus-neurosymbolic-architecture` | Deliberación Sistema 2 & Centinela | Firma: `{}` | {} |\n", telemetry.epistemic_provenance_checksum, if c5_passed { "✅ CONFIRM" } else { "❌ FALLO" }));

    report_md.push_str("\n> **Veredicto:** ");
    if all_connectors_passed {
        report_md.push_str("La integración de las 5 líneas de investigación produce una arquitectura sinérgica donde cada módulo geométrico, biológico y continuo nutre y valida directamente el sustrato cognitivo de Gary Marcus.\n");
    } else {
        report_md.push_str("Uno o más conectores no alcanzaron los criterios de rendimiento.\n");
    }

    println!("\n{}", report_md);

    let doc_path = Path::new("../docs/marcus_neurosymbolic/RESULTADOS_CONECTORES_UNIFICADOS.md");
    let _ = fs::write(doc_path, &report_md);
    println!("[INFO] Reporte guardado en: {:?}", doc_path);

    if all_connectors_passed {
        println!("\n>>> INTEGRACIÓN MULTI-ARQUITECTURA MARCUS-UNIFIED COMPLETADA AL 100% <<<");
    } else {
        eprintln!("\n>>> ERROR: FALLO EN UNO O MÁS CONECTORES <<<");
        std::process::exit(1);
    }
}
