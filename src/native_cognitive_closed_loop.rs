//! Ciclo cognitivo externo: memoria episódica, feedback para Gemma y sueños L0.
//!
//! Gemma permanece congelada. El aprendizaje durable vive en el CDT y en los
//! episodios recuperables del snapshot; los sueños son variaciones fasoriales
//! acotadas que siempre deben volver a pasar por el solver y sus verificadores.

use crate::native_multi_operator_core::{
    CognitiveEpisode, L0Solution, OperatorDeltaSnapshot, OperatorRecipe, OperatorSolution,
    QuboSolution, RequestedOperator, SolvedRecipe, UnaryFactor,
};
use std::collections::BTreeSet;

const MAX_EPISODES: usize = 256;

pub fn summarize_solution(solved: &SolvedRecipe) -> String {
    match &solved.solution {
        OperatorSolution::L0(solution) => summarize_l0(solution),
        OperatorSolution::Qubo(solution) => summarize_qubo(solution),
        OperatorSolution::L1(solution) => format!(
            "L1 aristas={} residual={:.6e} iteraciones={} verificado={}",
            solution.edge_flows.len(),
            solution.residual,
            solution.iterations,
            solution.verified
        ),
    }
}

fn summarize_l0(solution: &L0Solution) -> String {
    let coherent = solution
        .amplitudes
        .iter()
        .filter(|&&amplitude| amplitude >= 0.5)
        .count();
    format!(
        "L0 energia_inicial={:.6} energia_final={:.6} delta={:.6} residual={:.6e} \
         nodos_coherentes={}/{} verificado={}",
        solution.initial_energy,
        solution.final_energy,
        solution.initial_energy - solution.final_energy,
        solution.residual,
        coherent,
        solution.amplitudes.len(),
        solution.verified
    )
}

fn summarize_qubo(solution: &QuboSolution) -> String {
    let active = solution.bits.iter().filter(|&&bit| bit).count();
    format!(
        "QUBO energia={:.6} bits_activos={}/{} exacto={} optimo_local={} verificado={}",
        solution.energy,
        active,
        solution.bits.len(),
        solution.exact,
        solution.local_optimum,
        solution.verified
    )
}

pub fn episode_from_solution(
    prompt: &str,
    recipe: &OperatorRecipe,
    solved: &SolvedRecipe,
) -> CognitiveEpisode {
    let key = format!("{}\0{}\0{:?}", prompt.trim(), recipe.name, solved.operator);
    CognitiveEpisode {
        id: format!("{:016x}", stable_hash(key.as_bytes())),
        prompt: prompt.trim().to_string(),
        recipe_name: recipe.name.clone(),
        operator: solved.operator,
        solution_summary: summarize_solution(solved),
        verified: solved.solution.verified(),
        recalls: 0,
    }
}

pub fn record_episode(snapshot: &mut OperatorDeltaSnapshot, episode: CognitiveEpisode) {
    if let Some(stored) = snapshot
        .episodes
        .iter_mut()
        .find(|stored| stored.id == episode.id)
    {
        let recalls = stored.recalls;
        *stored = episode;
        stored.recalls = recalls;
        return;
    }
    snapshot.episodes.push(episode);
    if snapshot.episodes.len() > MAX_EPISODES {
        let remove = snapshot
            .episodes
            .iter()
            .enumerate()
            .min_by_key(|(_, episode)| (episode.recalls, episode.id.clone()))
            .map(|(index, _)| index)
            .unwrap_or(0);
        snapshot.episodes.remove(remove);
    }
}

/// Recupera episodios por solapamiento léxico determinista y registra el recall.
pub fn retrieve_episodes(
    snapshot: &mut OperatorDeltaSnapshot,
    query: &str,
    limit: usize,
) -> Vec<CognitiveEpisode> {
    let query_tokens = tokens(query);
    if query_tokens.is_empty() || limit == 0 {
        return Vec::new();
    }
    let mut ranked = snapshot
        .episodes
        .iter()
        .enumerate()
        .filter_map(|(index, episode)| {
            if !episode.verified {
                return None;
            }
            let mut document = episode.prompt.clone();
            document.push(' ');
            document.push_str(&episode.recipe_name);
            document.push(' ');
            document.push_str(&episode.solution_summary);
            let document_tokens = tokens(&document);
            let overlap = query_tokens.intersection(&document_tokens).count();
            (overlap > 0).then_some((index, overlap, episode.recalls))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.0.cmp(&right.0))
    });
    let selected = ranked
        .into_iter()
        .take(limit)
        .map(|(index, _, _)| index)
        .collect::<Vec<_>>();
    for &index in &selected {
        snapshot.episodes[index].recalls = snapshot.episodes[index].recalls.saturating_add(1);
    }
    selected
        .into_iter()
        .map(|index| snapshot.episodes[index].clone())
        .collect()
}

pub fn memory_context(episodes: &[CognitiveEpisode]) -> String {
    if episodes.is_empty() {
        return "sin episodios relevantes".to_string();
    }
    episodes
        .iter()
        .enumerate()
        .map(|(index, episode)| {
            format!(
                "{}. tarea={:?}; receta={}; operador={:?}; resultado={}",
                index + 1,
                episode.prompt,
                episode.recipe_name,
                episode.operator,
                episode.solution_summary
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Genera experiencias L0 nuevas pero acotadas a partir de recuerdos aceptados.
///
/// Los nombres de slot son estables para mantener el snapshot limitado. No se
/// consolidan aquí: el llamador debe resolver, verificar y usar una tasa menor.
pub fn generate_dream_recipes(
    snapshot: &OperatorDeltaSnapshot,
    dream_cycle: u64,
    count: usize,
    seed: u64,
) -> Vec<OperatorRecipe> {
    let bases = snapshot
        .accepted_recipes
        .iter()
        .filter(|recipe| !recipe.name.starts_with("dream_"))
        .filter(|recipe| recipe.selected_operator() == Ok(RequestedOperator::L0))
        .collect::<Vec<_>>();
    if bases.is_empty() || count == 0 {
        return Vec::new();
    }
    (0..count)
        .filter_map(|slot| {
            let base_index = stable_hash(&(seed ^ dream_cycle ^ slot as u64).to_le_bytes())
                as usize
                % bases.len();
            let base = bases[base_index];
            let mut dream = base.clone();
            dream.name = format!("dream_{}_slot_{}", base.name, slot);
            let perturbation_seed =
                stable_hash(&(seed ^ dream_cycle.rotate_left(17) ^ slot as u64).to_le_bytes());
            for (index, unary) in dream.unary_factors.iter_mut().enumerate() {
                unary.phase = (unary.phase + signed_jitter(perturbation_seed, index, 0.18))
                    .rem_euclid(std::f32::consts::TAU);
            }
            for (index, pair) in dream.pair_factors.iter_mut().enumerate() {
                pair.phase = (pair.phase
                    + signed_jitter(perturbation_seed ^ 0x0A11_CE55, index, 0.08))
                .rem_euclid(std::f32::consts::TAU);
                pair.weight = (pair.weight
                    * (1.0 + signed_jitter(perturbation_seed, index + 97, 0.04)))
                .max(1.0e-4);
            }
            if dream.unary_factors.is_empty() {
                let variable = dream.variables.first()?.name.clone();
                dream.unary_factors.push(UnaryFactor {
                    variable,
                    weight: 0.25,
                    phase: signed_jitter(perturbation_seed, 0, std::f32::consts::PI)
                        .rem_euclid(std::f32::consts::TAU),
                });
            }
            dream.validate().ok().map(|_| dream)
        })
        .collect()
}

fn tokens(text: &str) -> BTreeSet<String> {
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| token.len() >= 3)
        .map(str::to_string)
        .collect()
}

fn signed_jitter(seed: u64, index: usize, scale: f32) -> f32 {
    let mixed = stable_hash(&(seed ^ index as u64).to_le_bytes());
    let unit = (mixed >> 11) as f64 / ((1u64 << 53) - 1) as f64;
    ((unit as f32 * 2.0) - 1.0) * scale
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_multi_operator_core::{
        PairFactor, SparseWorkingSet, VariableDomain, VariableNodeMapping, VariableSpec,
    };

    fn recipe() -> OperatorRecipe {
        OperatorRecipe {
            name: "memoria_fasorial".to_string(),
            requested_operator: RequestedOperator::L0,
            variables: vec![
                VariableSpec {
                    name: "ruta".to_string(),
                    domain: VariableDomain::Phasor,
                },
                VariableSpec {
                    name: "destino".to_string(),
                    domain: VariableDomain::Phasor,
                },
            ],
            unary_factors: vec![UnaryFactor {
                variable: "ruta".to_string(),
                weight: 1.0,
                phase: 0.0,
            }],
            pair_factors: vec![PairFactor {
                a: "ruta".to_string(),
                b: "destino".to_string(),
                weight: 1.0,
                phase: 0.0,
            }],
            oriented_faces: Vec::new(),
            flow_demands: Vec::new(),
            max_working_set: 8,
            ridge: 1.0e-3,
        }
    }

    fn solved() -> SolvedRecipe {
        SolvedRecipe {
            operator: RequestedOperator::L0,
            working_set: SparseWorkingSet {
                mappings: vec![VariableNodeMapping {
                    variable: "ruta".to_string(),
                    global_node: 1,
                }],
                global_nodes: vec![1],
            },
            solution: OperatorSolution::L0(L0Solution {
                amplitudes: vec![1.0, 0.8],
                phases: vec![0.0, 0.1],
                initial_energy: 4.0,
                final_energy: -1.0,
                residual: 1.0e-4,
                verified: true,
            }),
        }
    }

    #[test]
    fn records_and_retrieves_verified_episode() {
        let mut snapshot = OperatorDeltaSnapshot::default();
        record_episode(
            &mut snapshot,
            episode_from_solution("planifica una ruta", &recipe(), &solved()),
        );
        let found = retrieve_episodes(&mut snapshot, "ruta nueva", 2);
        assert_eq!(found.len(), 1);
        assert_eq!(snapshot.episodes[0].recalls, 1);
    }

    #[test]
    fn dreams_are_bounded_valid_l0_variants() {
        let mut snapshot = OperatorDeltaSnapshot::default();
        snapshot.accepted_recipes.push(recipe());
        let dreams = generate_dream_recipes(&snapshot, 7, 2, 11);
        assert_eq!(dreams.len(), 2);
        assert!(dreams.iter().all(|dream| dream.validate().is_ok()));
        assert!(dreams.iter().all(|dream| dream.name.starts_with("dream_")));
    }
}
