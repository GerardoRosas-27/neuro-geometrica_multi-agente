use cdt_rqm_epr::native_multi_operator_core::{
    FlowDemand, NativeMultiOperatorCore, OperatorRecipe, OperatorSolution, OrientedFace,
    PairFactor, RequestedOperator, UnaryFactor, VariableDomain, VariableSpec,
};
use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    DEFAULT_PHASOR_NODES_PER_SLICE, DEFAULT_PHASOR_STARTUP_SLICES,
};
use cdt_rqm_epr::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use std::time::Instant;

const SEEDS: usize = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices: DEFAULT_PHASOR_STARTUP_SLICES,
        nodes_per_slice: DEFAULT_PHASOR_NODES_PER_SLICE,
        temperature: 0.0,
        seed: 0xA11C_71CE_5EED,
        ..NativeThermoCdtConfig::default()
    });
    let mut engine = NativeMultiOperatorCore::default();
    let mut verified = 0usize;
    let mut total = 0usize;
    let mut l0_ms = 0.0;
    let mut qubo_ms = 0.0;
    let mut l1_ms = 0.0;

    for seed in 0..SEEDS {
        for recipe in [l0_recipe(seed), qubo_recipe(seed), l1_recipe(seed)] {
            let started = Instant::now();
            let solved = engine.solve(&recipe, &core)?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            match &solved.solution {
                OperatorSolution::L0(solution) => {
                    l0_ms += elapsed_ms;
                    if solution.verified
                        && solution.final_energy <= solution.initial_energy + 1.0e-5
                    {
                        verified += 1;
                    }
                }
                OperatorSolution::Qubo(solution) => {
                    qubo_ms += elapsed_ms;
                    if solution.verified && solution.exact && solution.local_optimum {
                        verified += 1;
                    }
                }
                OperatorSolution::L1(solution) => {
                    l1_ms += elapsed_ms;
                    if solution.verified && solution.residual <= 1.0e-4 {
                        verified += 1;
                    }
                }
            }
            if solved.working_set.global_nodes.len() >= core.node_count() {
                return Err("el working set dejó de ser sparse".into());
            }
            total += 1;
        }
    }

    for recipe in [l0_recipe(99), qubo_recipe(99), l1_recipe(99)] {
        engine.solve_and_consolidate(&recipe, &mut core, 0.18)?;
    }

    println!(
        "multioperador_cpu seeds={} casos={} verificados={}/{}",
        SEEDS, total, verified, total
    );
    println!(
        "latencia_media_ms L0={:.3} QUBO={:.3} L1={:.3}",
        l0_ms / SEEDS as f64,
        qubo_ms / SEEDS as f64,
        l1_ms / SEEDS as f64
    );
    println!(
        "escala core_nodos={} core_aristas={} working_sets=L0:512,QUBO:14,L1:192",
        core.node_count(),
        core.edge_count()
    );
    println!(
        "memoria recetas={} nodos_delta={} aristas_delta={}",
        engine.snapshot.accepted_recipes.len(),
        engine.snapshot.node_deltas.len(),
        engine.snapshot.edge_deltas.len()
    );
    if verified != total {
        return Err(format!("fallaron {} casos", total - verified).into());
    }
    Ok(())
}

fn l0_recipe(seed: usize) -> OperatorRecipe {
    let variables = (0..512)
        .map(|node| VariableSpec {
            name: format!("l0_{seed}_{node}"),
            domain: VariableDomain::Phasor,
        })
        .collect::<Vec<_>>();
    let pair_factors = (0..512)
        .map(|node| PairFactor {
            a: variables[node].name.clone(),
            b: variables[(node + 1) % variables.len()].name.clone(),
            weight: 1.0,
            phase: 0.0,
        })
        .collect();
    OperatorRecipe {
        name: format!("memoria_l0_{seed}"),
        requested_operator: RequestedOperator::Auto,
        unary_factors: vec![UnaryFactor {
            variable: variables[seed % variables.len()].name.clone(),
            weight: 1.5,
            phase: 0.0,
        }],
        variables,
        pair_factors,
        oriented_faces: Vec::new(),
        flow_demands: Vec::new(),
        max_working_set: 8_192,
        ridge: 1.0e-3,
    }
}

fn qubo_recipe(seed: usize) -> OperatorRecipe {
    let variables = (0..14)
        .map(|node| VariableSpec {
            name: format!("q_{seed}_{node}"),
            domain: VariableDomain::Binary,
        })
        .collect::<Vec<_>>();
    let unary_factors = variables
        .iter()
        .enumerate()
        .map(|(node, variable)| UnaryFactor {
            variable: variable.name.clone(),
            weight: if (node + seed) % 3 == 0 { -1.0 } else { 0.35 },
            phase: 0.0,
        })
        .collect();
    let pair_factors = (0..13)
        .map(|node| PairFactor {
            a: variables[node].name.clone(),
            b: variables[node + 1].name.clone(),
            weight: if (node + seed) % 2 == 0 { 0.8 } else { -0.25 },
            phase: 0.0,
        })
        .collect();
    OperatorRecipe {
        name: format!("qubo_{seed}"),
        requested_operator: RequestedOperator::Auto,
        variables,
        unary_factors,
        pair_factors,
        oriented_faces: Vec::new(),
        flow_demands: Vec::new(),
        max_working_set: 512,
        ridge: 1.0e-3,
    }
}

fn l1_recipe(seed: usize) -> OperatorRecipe {
    let variables = (0..192)
        .map(|node| VariableSpec {
            name: format!("v_{seed}_{node}"),
            domain: VariableDomain::Complex,
        })
        .collect::<Vec<_>>();
    let mut pair_factors = Vec::new();
    let mut oriented_faces = Vec::new();
    let mut flow_demands = Vec::new();
    for triangle in 0..64 {
        let a = variables[triangle * 3].name.clone();
        let b = variables[triangle * 3 + 1].name.clone();
        let c = variables[triangle * 3 + 2].name.clone();
        pair_factors.extend([
            PairFactor {
                a: a.clone(),
                b: b.clone(),
                weight: 1.0 + triangle as f32 * 0.001,
                phase: 0.0,
            },
            PairFactor {
                a: b.clone(),
                b: c.clone(),
                weight: 1.0,
                phase: 0.0,
            },
            PairFactor {
                a: c.clone(),
                b: a.clone(),
                weight: 1.0,
                phase: 0.0,
            },
        ]);
        oriented_faces.push(OrientedFace {
            vertices: [a.clone(), b.clone(), c],
        });
        flow_demands.push(FlowDemand {
            from: a,
            to: b,
            real: 1.0,
            imag: if triangle == seed % 64 { 0.25 } else { 0.0 },
        });
    }
    OperatorRecipe {
        name: format!("flujo_l1_{seed}"),
        requested_operator: RequestedOperator::Auto,
        variables,
        unary_factors: Vec::new(),
        pair_factors,
        oriented_faces,
        flow_demands,
        max_working_set: 8_192,
        ridge: 0.05,
    }
}
