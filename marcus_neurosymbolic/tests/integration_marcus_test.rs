//! Test de integración de extremo a extremo para MARCUS-I.

use marcus_neurosymbolic::*;

#[test]
fn test_integration_full_marcus_pipeline() {
    let mut engine = MarcusEngine::new();

    // 1. Andamiaje innato: Relación de contención de Spelke
    engine.physics_engine.set_containment("Key", "SmallPouch");
    engine.physics_engine.set_containment("SmallPouch", "Luggage");
    assert!(engine.physics_engine.is_transitively_inside("Key", "Luggage"));

    // 2. Estado epistémico
    engine.world_state.insert_fact("Luggage", "Location", "Trunk", EpistemicCertainty::ObservedFact);
    assert!(engine.world_state.fact_exists("Luggage", "Location", "Trunk"));

    // 3. Sistema 1 propone hipótesis basadas en co-ocurrencia
    engine.system1.add_association(CandidateHypothesis::new(
        "hyp_1",
        "Key is in Luggage",
        0.88,
        "Key",
        "IsIn",
        "Luggage",
    ));
    engine.system1.add_association(CandidateHypothesis::new(
        "hyp_2",
        "Key is in Microwave (Hallucination)",
        0.75,
        "Key",
        "IsIn",
        "Microwave",
    ));

    // 4. Ciclo Dual: Sistema 2 verifica y aprueba hyp_1 mientras veta hyp_2
    let result = engine.process_query("Where is the key?");
    assert!(result.selected_hypothesis.is_some());
    let selected = result.selected_hypothesis.unwrap();
    assert_eq!(selected.id, "hyp_1");
    assert_eq!(selected.object, "Luggage");

    // 5. Centinela de alucinaciones
    let gate_pass = engine.audit_claim("claim_trunk", "Luggage", "Location", "Trunk");
    assert!(matches!(gate_pass, HallucinationGateDecision::Approved { .. }));

    let gate_fail = engine.audit_claim("claim_fake", "Luggage", "Location", "Stratosphere");
    assert!(matches!(gate_fail, HallucinationGateDecision::Vetoed { .. }));
}
