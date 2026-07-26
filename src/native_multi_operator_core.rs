//! Núcleo CPU sparse para recetas L0 magnéticas, QUBO y Hodge L1.
//! Las recetas usan nombres lógicos; los índices físicos se asignan aquí.

use crate::native_checkpoint::atomic_write;
use crate::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorMinimizerConfig, NativePhasorThermodynamicEngine,
};
use crate::native_thermodynamic_cdt::{
    NativeCdtEdgeKind, NativeEdgeUpsert, NativeNodeDelta, NativeThermoCdtConfig,
    NativeThermoCdtSubstrate,
};
use num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

const DEFAULT_MAX_WORKING_SET: usize = 16_384;
const EPSILON: f32 = 1.0e-7;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestedOperator {
    Auto,
    L0,
    Qubo,
    L1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariableDomain {
    Binary,
    Phasor,
    Complex,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariableSpec {
    pub name: String,
    pub domain: VariableDomain,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnaryFactor {
    pub variable: String,
    pub weight: f32,
    #[serde(default)]
    pub phase: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairFactor {
    pub a: String,
    pub b: String,
    pub weight: f32,
    #[serde(default)]
    pub phase: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrientedFace {
    pub vertices: [String; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlowDemand {
    pub from: String,
    pub to: String,
    pub real: f32,
    #[serde(default)]
    pub imag: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorRecipe {
    pub name: String,
    pub requested_operator: RequestedOperator,
    pub variables: Vec<VariableSpec>,
    #[serde(default)]
    pub unary_factors: Vec<UnaryFactor>,
    #[serde(default)]
    pub pair_factors: Vec<PairFactor>,
    #[serde(default)]
    pub oriented_faces: Vec<OrientedFace>,
    #[serde(default)]
    pub flow_demands: Vec<FlowDemand>,
    pub max_working_set: usize,
    #[serde(default = "default_ridge")]
    pub ridge: f32,
}

fn default_ridge() -> f32 {
    1.0e-3
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperatorRecipeError {
    EmptyName(&'static str),
    DuplicateVariable(String),
    UnknownVariable(String),
    DuplicateEdge(String, String),
    SelfEdge(String),
    NonFinite(&'static str),
    InvalidFace(usize),
    MissingFaceEdge(String, String),
    EmptyRecipe,
    WorkingSetLimit { requested: usize, available: usize },
    IncompatibleOperator(&'static str),
}

impl fmt::Display for OperatorRecipeError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName(kind) => write!(output, "{kind} tiene nombre vacío"),
            Self::DuplicateVariable(name) => write!(output, "variable duplicada: {name}"),
            Self::UnknownVariable(name) => {
                write!(output, "referencia a variable desconocida: {name}")
            }
            Self::DuplicateEdge(a, b) => write!(output, "arista duplicada: {a}-{b}"),
            Self::SelfEdge(name) => write!(output, "autoarista inválida: {name}"),
            Self::NonFinite(field) => write!(output, "valor no finito en {field}"),
            Self::InvalidFace(face) => write!(output, "cara orientada {face} inválida"),
            Self::MissingFaceEdge(a, b) => write!(output, "falta arista de cara: {a}-{b}"),
            Self::EmptyRecipe => write!(output, "la receta no contiene variables"),
            Self::WorkingSetLimit {
                requested,
                available,
            } => {
                write!(output, "working set {requested} excede límite {available}")
            }
            Self::IncompatibleOperator(reason) => write!(output, "operador incompatible: {reason}"),
        }
    }
}

impl std::error::Error for OperatorRecipeError {}

impl OperatorRecipe {
    pub fn validate(&self) -> Result<(), OperatorRecipeError> {
        if self.name.trim().is_empty() {
            return Err(OperatorRecipeError::EmptyName("receta"));
        }
        if self.variables.is_empty() {
            return Err(OperatorRecipeError::EmptyRecipe);
        }
        if self.max_working_set == 0 || self.variables.len() > self.max_working_set {
            return Err(OperatorRecipeError::WorkingSetLimit {
                requested: self.variables.len(),
                available: self.max_working_set,
            });
        }
        if !self.ridge.is_finite() || self.ridge < 0.0 {
            return Err(OperatorRecipeError::NonFinite("ridge"));
        }
        let mut names = BTreeSet::new();
        for variable in &self.variables {
            if variable.name.trim().is_empty() {
                return Err(OperatorRecipeError::EmptyName("variable"));
            }
            if !names.insert(variable.name.clone()) {
                return Err(OperatorRecipeError::DuplicateVariable(
                    variable.name.clone(),
                ));
            }
        }
        let ensure_name = |name: &str| {
            if names.contains(name) {
                Ok(())
            } else {
                Err(OperatorRecipeError::UnknownVariable(name.to_string()))
            }
        };
        for unary in &self.unary_factors {
            ensure_name(&unary.variable)?;
            if !unary.weight.is_finite() || !unary.phase.is_finite() {
                return Err(OperatorRecipeError::NonFinite("factor unario"));
            }
        }
        let mut edges = BTreeSet::new();
        for pair in &self.pair_factors {
            ensure_name(&pair.a)?;
            ensure_name(&pair.b)?;
            if pair.a == pair.b {
                return Err(OperatorRecipeError::SelfEdge(pair.a.clone()));
            }
            if !pair.weight.is_finite() || !pair.phase.is_finite() {
                return Err(OperatorRecipeError::NonFinite("factor par"));
            }
            let key = undirected_key(&pair.a, &pair.b);
            if !edges.insert(key.clone()) {
                return Err(OperatorRecipeError::DuplicateEdge(key.0, key.1));
            }
        }
        for (index, face) in self.oriented_faces.iter().enumerate() {
            let [a, b, c] = &face.vertices;
            ensure_name(a)?;
            ensure_name(b)?;
            ensure_name(c)?;
            if a == b || b == c || a == c {
                return Err(OperatorRecipeError::InvalidFace(index));
            }
            for (from, to) in [(a, b), (b, c), (c, a)] {
                if !edges.contains(&undirected_key(from, to)) {
                    return Err(OperatorRecipeError::MissingFaceEdge(
                        from.clone(),
                        to.clone(),
                    ));
                }
            }
        }
        for demand in &self.flow_demands {
            ensure_name(&demand.from)?;
            ensure_name(&demand.to)?;
            if !demand.real.is_finite() || !demand.imag.is_finite() {
                return Err(OperatorRecipeError::NonFinite("demanda de flujo"));
            }
            if !edges.contains(&undirected_key(&demand.from, &demand.to)) {
                return Err(OperatorRecipeError::MissingFaceEdge(
                    demand.from.clone(),
                    demand.to.clone(),
                ));
            }
        }
        let selected = self.selected_operator()?;
        match selected {
            RequestedOperator::Qubo
                if self
                    .variables
                    .iter()
                    .any(|variable| variable.domain != VariableDomain::Binary) =>
            {
                Err(OperatorRecipeError::IncompatibleOperator(
                    "QUBO requiere variables binarias",
                ))
            }
            RequestedOperator::Qubo
                if !self.oriented_faces.is_empty()
                    || !self.flow_demands.is_empty()
                    || self
                        .unary_factors
                        .iter()
                        .any(|factor| factor.phase.abs() > EPSILON)
                    || self
                        .pair_factors
                        .iter()
                        .any(|factor| factor.phase.abs() > EPSILON) =>
            {
                Err(OperatorRecipeError::IncompatibleOperator(
                    "QUBO no admite fases, caras ni demandas de flujo",
                ))
            }
            RequestedOperator::L1 if self.pair_factors.is_empty() => Err(
                OperatorRecipeError::IncompatibleOperator("L1 requiere aristas"),
            ),
            RequestedOperator::L1
                if self.pair_factors.iter().any(|factor| factor.weight <= 0.0) =>
            {
                Err(OperatorRecipeError::IncompatibleOperator(
                    "L1 requiere pesos de arista positivos",
                ))
            }
            RequestedOperator::L0 if self.pair_factors.is_empty() => Err(
                OperatorRecipeError::IncompatibleOperator("L0 requiere aristas"),
            ),
            RequestedOperator::L0
                if self
                    .variables
                    .iter()
                    .any(|variable| variable.domain == VariableDomain::Binary) =>
            {
                Err(OperatorRecipeError::IncompatibleOperator(
                    "L0 no acepta variables binarias",
                ))
            }
            _ => Ok(()),
        }
    }

    pub fn selected_operator(&self) -> Result<RequestedOperator, OperatorRecipeError> {
        let selected = match self.requested_operator {
            RequestedOperator::Auto
                if !self.oriented_faces.is_empty() || !self.flow_demands.is_empty() =>
            {
                RequestedOperator::L1
            }
            RequestedOperator::Auto
                if self
                    .variables
                    .iter()
                    .all(|variable| variable.domain == VariableDomain::Binary) =>
            {
                RequestedOperator::Qubo
            }
            RequestedOperator::Auto => RequestedOperator::L0,
            explicit => explicit,
        };
        Ok(selected)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariableNodeMapping {
    pub variable: String,
    pub global_node: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseWorkingSet {
    pub mappings: Vec<VariableNodeMapping>,
    pub global_nodes: Vec<usize>,
}

impl SparseWorkingSet {
    pub fn compile(
        recipe: &OperatorRecipe,
        core: &NativeThermoCdtSubstrate,
        hard_limit: usize,
    ) -> Result<Self, OperatorRecipeError> {
        recipe.validate()?;
        let available = recipe
            .max_working_set
            .min(hard_limit)
            .min(core.node_count());
        if recipe.variables.len() > available {
            return Err(OperatorRecipeError::WorkingSetLimit {
                requested: recipe.variables.len(),
                available,
            });
        }
        let mut sorted_names = recipe
            .variables
            .iter()
            .map(|variable| variable.name.clone())
            .collect::<Vec<_>>();
        sorted_names.sort();
        let mut assigned = BTreeMap::new();
        let mut occupied = BTreeSet::new();
        for name in sorted_names {
            let mut node = stable_hash(name.as_bytes()) as usize % core.node_count();
            while occupied.contains(&node) {
                node = (node + 1) % core.node_count();
            }
            occupied.insert(node);
            assigned.insert(name, node);
        }
        let mappings = recipe
            .variables
            .iter()
            .map(|variable| VariableNodeMapping {
                variable: variable.name.clone(),
                global_node: assigned[&variable.name],
            })
            .collect::<Vec<_>>();
        let global_nodes = mappings.iter().map(|mapping| mapping.global_node).collect();
        Ok(Self {
            mappings,
            global_nodes,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct L0Solution {
    pub amplitudes: Vec<f32>,
    pub phases: Vec<f32>,
    pub initial_energy: f32,
    pub final_energy: f32,
    pub residual: f32,
    pub verified: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuboSolution {
    pub bits: Vec<bool>,
    pub energy: f32,
    pub starts: usize,
    pub exact: bool,
    pub local_optimum: bool,
    pub verified: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct L1Solution {
    pub edge_flows: Vec<Complex32>,
    pub residual: f32,
    pub iterations: usize,
    pub verified: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OperatorSolution {
    L0(L0Solution),
    Qubo(QuboSolution),
    L1(L1Solution),
}

impl OperatorSolution {
    pub fn verified(&self) -> bool {
        match self {
            Self::L0(solution) => solution.verified,
            Self::Qubo(solution) => solution.verified,
            Self::L1(solution) => solution.verified,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SolvedRecipe {
    pub operator: RequestedOperator,
    pub working_set: SparseWorkingSet,
    pub solution: OperatorSolution,
}

#[derive(Clone, Debug)]
pub struct HodgeL1Operator {
    vertex_count: usize,
    edges: Vec<(usize, usize)>,
    edge_scales: Vec<f32>,
    faces: Vec<[(usize, f32); 3]>,
    ridge: f32,
}

impl HodgeL1Operator {
    pub fn from_recipe(recipe: &OperatorRecipe) -> Result<Self, OperatorRecipeError> {
        recipe.validate()?;
        let indices = recipe
            .variables
            .iter()
            .enumerate()
            .map(|(index, variable)| (variable.name.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        let edges = recipe
            .pair_factors
            .iter()
            .map(|pair| (indices[pair.a.as_str()], indices[pair.b.as_str()]))
            .collect::<Vec<_>>();
        let edge_scales = recipe
            .pair_factors
            .iter()
            .map(|pair| pair.weight.sqrt())
            .collect::<Vec<_>>();
        let edge_indices = recipe
            .pair_factors
            .iter()
            .enumerate()
            .map(|(index, pair)| (undirected_key(&pair.a, &pair.b), index))
            .collect::<BTreeMap<_, _>>();
        let mut faces = Vec::with_capacity(recipe.oriented_faces.len());
        for face in &recipe.oriented_faces {
            let [a, b, c] = &face.vertices;
            let mut incidence = [(0, 0.0); 3];
            for (slot, (from, to)) in [(a, b), (b, c), (c, a)].into_iter().enumerate() {
                let edge = edge_indices[&undirected_key(from, to)];
                let pair = &recipe.pair_factors[edge];
                incidence[slot] = (
                    edge,
                    if pair.a == *from && pair.b == *to {
                        1.0
                    } else {
                        -1.0
                    },
                );
            }
            faces.push(incidence);
        }
        Ok(Self {
            vertex_count: recipe.variables.len(),
            edges,
            edge_scales,
            faces,
            ridge: recipe.ridge,
        })
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Comprueba localmente ∂1∂2=0 para cada cara orientada.
    pub fn chain_complex_valid(&self) -> bool {
        self.faces.iter().all(|face| {
            let mut boundary = vec![0.0f32; self.vertex_count];
            for &(edge, face_sign) in face {
                let (a, b) = self.edges[edge];
                boundary[a] -= face_sign;
                boundary[b] += face_sign;
            }
            boundary.into_iter().all(|value| value.abs() <= EPSILON)
        })
    }

    /// Aplica B1^T(B1 z) + B2(B2^T z) + ridge z sin materializar L1.
    pub fn apply(&self, input: &[Complex32], output: &mut [Complex32]) {
        assert_eq!(input.len(), self.edges.len());
        assert_eq!(output.len(), self.edges.len());
        let mut divergence = vec![Complex32::new(0.0, 0.0); self.vertex_count];
        for (edge, &(a, b)) in self.edges.iter().enumerate() {
            let weighted = self.edge_scales[edge] * input[edge];
            divergence[a] -= weighted;
            divergence[b] += weighted;
        }
        for (edge, &(a, b)) in self.edges.iter().enumerate() {
            output[edge] = self.edge_scales[edge] * (-divergence[a] + divergence[b])
                + self.ridge * input[edge];
        }
        for face in &self.faces {
            let circulation = face
                .iter()
                .fold(Complex32::new(0.0, 0.0), |sum, &(edge, sign)| {
                    sum + sign * self.edge_scales[edge] * input[edge]
                });
            for &(edge, sign) in face {
                output[edge] += sign * self.edge_scales[edge] * circulation;
            }
        }
    }

    pub fn quadratic_form(&self, input: &[Complex32]) -> Complex32 {
        let mut output = vec![Complex32::new(0.0, 0.0); input.len()];
        self.apply(input, &mut output);
        input
            .iter()
            .zip(output)
            .map(|(&left, right)| left.conj() * right)
            .sum()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnedNodeDelta {
    pub variable: String,
    pub global_node: usize,
    pub thermal_state: Option<f32>,
    pub amplitude: Option<f32>,
    pub phase: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnedEdgeDelta {
    pub a: usize,
    pub b: usize,
    pub weight: f32,
    pub phase: f32,
}

/// Episodio simbólico recuperable por Gemma en inferencias posteriores.
///
/// No modifica los pesos del LLM: conserva la tarea, la receta compilada y un
/// resumen verificable del solver como memoria externa del ciclo cognitivo.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveEpisode {
    pub id: String,
    pub prompt: String,
    pub recipe_name: String,
    pub operator: RequestedOperator,
    pub solution_summary: String,
    pub verified: bool,
    #[serde(default)]
    pub recalls: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct OperatorDeltaSnapshot {
    pub mappings: Vec<VariableNodeMapping>,
    pub node_deltas: Vec<LearnedNodeDelta>,
    pub edge_deltas: Vec<LearnedEdgeDelta>,
    pub accepted_recipes: Vec<OperatorRecipe>,
    #[serde(default)]
    pub episodes: Vec<CognitiveEpisode>,
}

impl OperatorDeltaSnapshot {
    pub fn save(&self, destination: impl AsRef<Path>) -> Result<(), String> {
        let body = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        atomic_write(destination.as_ref(), &body)
    }

    pub fn load(source: impl AsRef<Path>) -> Result<Self, String> {
        let body = std::fs::read(source).map_err(|error| error.to_string())?;
        serde_json::from_slice(&body).map_err(|error| error.to_string())
    }

    pub fn apply_to_core(
        &self,
        core: &mut NativeThermoCdtSubstrate,
        learning_rate: f32,
    ) -> Result<(), String> {
        let node_deltas = self
            .node_deltas
            .iter()
            .map(|delta| NativeNodeDelta {
                node: delta.global_node,
                thermal_state: delta.thermal_state,
                amplitude: delta.amplitude,
                phase: delta.phase,
            })
            .collect::<Vec<_>>();
        let edge_upserts = self
            .edge_deltas
            .iter()
            .map(|delta| NativeEdgeUpsert {
                a: delta.a,
                b: delta.b,
                kind: NativeCdtEdgeKind::Spatial,
                weight: delta.weight,
                phase: delta.phase,
                stability: 1.0,
            })
            .collect::<Vec<_>>();
        core.batch_upsert_sparse(&node_deltas, &edge_upserts, learning_rate)
    }
}

#[derive(Clone, Debug)]
pub struct NativeMultiOperatorCore {
    pub max_working_set: usize,
    pub snapshot: OperatorDeltaSnapshot,
}

impl Default for NativeMultiOperatorCore {
    fn default() -> Self {
        Self {
            max_working_set: DEFAULT_MAX_WORKING_SET,
            snapshot: OperatorDeltaSnapshot::default(),
        }
    }
}

impl NativeMultiOperatorCore {
    pub fn solve(
        &self,
        recipe: &OperatorRecipe,
        global_core: &NativeThermoCdtSubstrate,
    ) -> Result<SolvedRecipe, String> {
        let working_set = SparseWorkingSet::compile(recipe, global_core, self.max_working_set)
            .map_err(|error| error.to_string())?;
        let operator = recipe
            .selected_operator()
            .map_err(|error| error.to_string())?;
        let solution = match operator {
            RequestedOperator::L0 => {
                OperatorSolution::L0(solve_l0(recipe, global_core, &working_set)?)
            }
            RequestedOperator::Qubo => OperatorSolution::Qubo(solve_qubo(recipe)),
            RequestedOperator::L1 => OperatorSolution::L1(solve_l1(recipe)?),
            RequestedOperator::Auto => unreachable!("Auto siempre se resuelve"),
        };
        Ok(SolvedRecipe {
            operator,
            working_set,
            solution,
        })
    }

    pub fn consolidate(
        &mut self,
        recipe: &OperatorRecipe,
        solved: &SolvedRecipe,
        global_core: &mut NativeThermoCdtSubstrate,
        learning_rate: f32,
    ) -> Result<(), String> {
        if !solved.solution.verified() {
            return Err("la solución no pasó verificación; no se consolida".to_string());
        }
        let name_to_local = recipe
            .variables
            .iter()
            .enumerate()
            .map(|(index, variable)| (variable.name.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        let name_to_global = solved
            .working_set
            .mappings
            .iter()
            .map(|mapping| (mapping.variable.as_str(), mapping.global_node))
            .collect::<BTreeMap<_, _>>();
        let mut node_updates = Vec::new();
        let mut learned_nodes = Vec::new();
        match &solved.solution {
            OperatorSolution::L0(solution) => {
                for mapping in &solved.working_set.mappings {
                    let local = name_to_local[mapping.variable.as_str()];
                    node_updates.push(NativeNodeDelta {
                        node: mapping.global_node,
                        thermal_state: None,
                        amplitude: Some(solution.amplitudes[local]),
                        phase: Some(solution.phases[local]),
                    });
                    learned_nodes.push(LearnedNodeDelta {
                        variable: mapping.variable.clone(),
                        global_node: mapping.global_node,
                        thermal_state: None,
                        amplitude: Some(solution.amplitudes[local]),
                        phase: Some(solution.phases[local]),
                    });
                }
            }
            OperatorSolution::Qubo(solution) => {
                for mapping in &solved.working_set.mappings {
                    let local = name_to_local[mapping.variable.as_str()];
                    let state = if solution.bits[local] { 1.0 } else { -1.0 };
                    node_updates.push(NativeNodeDelta {
                        node: mapping.global_node,
                        thermal_state: Some(state),
                        amplitude: None,
                        phase: None,
                    });
                    learned_nodes.push(LearnedNodeDelta {
                        variable: mapping.variable.clone(),
                        global_node: mapping.global_node,
                        thermal_state: Some(state),
                        amplitude: None,
                        phase: None,
                    });
                }
            }
            OperatorSolution::L1(solution) => {
                let mut accumulation = vec![Complex32::new(0.0, 0.0); recipe.variables.len()];
                let mut degree = vec![0usize; recipe.variables.len()];
                for (edge, pair) in recipe.pair_factors.iter().enumerate() {
                    let a = name_to_local[pair.a.as_str()];
                    let b = name_to_local[pair.b.as_str()];
                    accumulation[a] -= solution.edge_flows[edge];
                    accumulation[b] += solution.edge_flows[edge];
                    degree[a] += 1;
                    degree[b] += 1;
                }
                for mapping in &solved.working_set.mappings {
                    let local = name_to_local[mapping.variable.as_str()];
                    let state = accumulation[local].norm() / degree[local].max(1) as f32;
                    node_updates.push(NativeNodeDelta {
                        node: mapping.global_node,
                        thermal_state: Some(state),
                        amplitude: None,
                        phase: None,
                    });
                    learned_nodes.push(LearnedNodeDelta {
                        variable: mapping.variable.clone(),
                        global_node: mapping.global_node,
                        thermal_state: Some(state),
                        amplitude: None,
                        phase: None,
                    });
                }
            }
        }
        let edge_updates = recipe
            .pair_factors
            .iter()
            .map(|pair| {
                let (weight, phase) = magnetic_weight_phase(pair.weight, pair.phase);
                NativeEdgeUpsert {
                    a: name_to_global[pair.a.as_str()],
                    b: name_to_global[pair.b.as_str()],
                    kind: NativeCdtEdgeKind::Spatial,
                    weight,
                    phase,
                    stability: 1.0,
                }
            })
            .collect::<Vec<_>>();
        global_core.batch_upsert_sparse(&node_updates, &edge_updates, learning_rate)?;

        for mapping in &solved.working_set.mappings {
            if let Some(stored) = self
                .snapshot
                .mappings
                .iter_mut()
                .find(|stored| stored.variable == mapping.variable)
            {
                *stored = mapping.clone();
            } else {
                self.snapshot.mappings.push(mapping.clone());
            }
        }
        for delta in learned_nodes {
            if let Some(stored) = self
                .snapshot
                .node_deltas
                .iter_mut()
                .find(|stored| stored.global_node == delta.global_node)
            {
                *stored = delta;
            } else {
                self.snapshot.node_deltas.push(delta);
            }
        }
        for delta in edge_updates.iter().map(|edge| LearnedEdgeDelta {
            a: edge.a,
            b: edge.b,
            weight: edge.weight,
            phase: edge.phase,
        }) {
            if let Some(stored) = self.snapshot.edge_deltas.iter_mut().find(|stored| {
                (stored.a == delta.a && stored.b == delta.b)
                    || (stored.a == delta.b && stored.b == delta.a)
            }) {
                *stored = delta;
            } else {
                self.snapshot.edge_deltas.push(delta);
            }
        }
        if let Some(stored) = self
            .snapshot
            .accepted_recipes
            .iter_mut()
            .find(|stored| stored.name == recipe.name)
        {
            *stored = recipe.clone();
        } else {
            self.snapshot.accepted_recipes.push(recipe.clone());
        }
        Ok(())
    }

    pub fn solve_and_consolidate(
        &mut self,
        recipe: &OperatorRecipe,
        global_core: &mut NativeThermoCdtSubstrate,
        learning_rate: f32,
    ) -> Result<SolvedRecipe, String> {
        let solved = self.solve(recipe, global_core)?;
        self.consolidate(recipe, &solved, global_core, learning_rate)?;
        Ok(solved)
    }
}

fn solve_l0(
    recipe: &OperatorRecipe,
    global_core: &NativeThermoCdtSubstrate,
    working_set: &SparseWorkingSet,
) -> Result<L0Solution, String> {
    let nodes = recipe.variables.len();
    let mut local = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices: 1,
        nodes_per_slice: nodes,
        spatial_degree: 1,
        temporal_degree: 1,
        temperature: 0.0,
        seed: stable_hash(recipe.name.as_bytes()),
        ..NativeThermoCdtConfig::default()
    });
    let local_index = recipe
        .variables
        .iter()
        .enumerate()
        .map(|(index, variable)| (variable.name.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    for (local_node, &global_node) in working_set.global_nodes.iter().enumerate() {
        local.amplitude[local_node] = global_core.amplitude[global_node].max(0.1);
        local.phase[local_node] = global_core.phase[global_node];
    }
    local.replace_edges(recipe.pair_factors.iter().map(|pair| {
        let (weight, phase) = magnetic_weight_phase(pair.weight, pair.phase);
        (
            local_index[pair.a.as_str()],
            local_index[pair.b.as_str()],
            NativeCdtEdgeKind::Spatial,
            weight,
            phase,
            1.0,
        )
    }));
    let mut engine = NativePhasorThermodynamicEngine::from_core(
        &local,
        NativePhasorConfig {
            temperature_scale: 0.0,
            noise_scale: 0.0,
            seed: stable_hash(recipe.name.as_bytes()),
            ..NativePhasorConfig::default()
        },
    )
    .map_err(|error| error.to_string())?;
    for unary in &recipe.unary_factors {
        engine.inject_pattern(
            &[local_index[unary.variable.as_str()]],
            unary.weight.abs(),
            unary.phase
                + if unary.weight < 0.0 {
                    std::f32::consts::PI
                } else {
                    0.0
                },
        );
    }
    let initial_energy = engine.report().free_energy;
    let report = engine.minimize_free_energy(NativePhasorMinimizerConfig {
        max_iterations: 600,
        residual_tolerance: 5.0e-4,
        ..NativePhasorMinimizerConfig::default()
    });
    let final_energy = report.final_report.free_energy;
    Ok(L0Solution {
        amplitudes: engine.phasors.iter().map(|value| value.norm()).collect(),
        phases: engine.phasors.iter().map(|value| value.arg()).collect(),
        initial_energy,
        final_energy,
        residual: report.final_report.gradient_residual,
        verified: final_energy.is_finite()
            && final_energy <= initial_energy + 1.0e-5
            && report.final_report.gradient_residual.is_finite()
            && report.final_report.gradient_residual <= 5.0e-3,
    })
}

pub fn evaluate_qubo(recipe: &OperatorRecipe, bits: &[bool]) -> f32 {
    let indices = recipe
        .variables
        .iter()
        .enumerate()
        .map(|(index, variable)| (variable.name.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut energy = 0.0;
    for unary in &recipe.unary_factors {
        if bits[indices[unary.variable.as_str()]] {
            energy += unary.weight;
        }
    }
    for pair in &recipe.pair_factors {
        if bits[indices[pair.a.as_str()]] && bits[indices[pair.b.as_str()]] {
            energy += pair.weight;
        }
    }
    energy
}

fn solve_qubo(recipe: &OperatorRecipe) -> QuboSolution {
    let nodes = recipe.variables.len();
    let indices = recipe
        .variables
        .iter()
        .enumerate()
        .map(|(index, variable)| (variable.name.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut unary = vec![0.0; nodes];
    let mut adjacency = vec![Vec::<(usize, f32)>::new(); nodes];
    for factor in &recipe.unary_factors {
        unary[indices[factor.variable.as_str()]] += factor.weight;
    }
    for factor in &recipe.pair_factors {
        let a = indices[factor.a.as_str()];
        let b = indices[factor.b.as_str()];
        adjacency[a].push((b, factor.weight));
        adjacency[b].push((a, factor.weight));
    }

    let mut starts = Vec::new();
    starts.push(vec![false; nodes]);
    starts.push(vec![true; nodes]);
    for start in 0..8 {
        starts.push(
            (0..nodes)
                .map(|node| {
                    stable_hash(format!("{}:{start}:{node}", recipe.name).as_bytes()) & 1 == 1
                })
                .collect(),
        );
    }
    // Relajación continua sparse proyectada y redondeo.
    let mut relaxed = vec![0.5f32; nodes];
    for iteration in 0..128 {
        let step = 0.15 / (1.0 + iteration as f32 * 0.03);
        for node in 0..nodes {
            let gradient = unary[node]
                + adjacency[node]
                    .iter()
                    .map(|&(other, weight)| weight * relaxed[other])
                    .sum::<f32>();
            relaxed[node] = (relaxed[node] - step * gradient).clamp(0.0, 1.0);
        }
    }
    starts.push(relaxed.iter().map(|&value| value >= 0.5).collect());
    let mut best = starts[0].clone();
    let mut best_energy = evaluate_qubo(recipe, &best);
    for mut bits in starts.iter().cloned() {
        coordinate_descent(&mut bits, &unary, &adjacency);
        let energy = evaluate_qubo(recipe, &bits);
        if energy < best_energy {
            best = bits;
            best_energy = energy;
        }
    }
    // Para fixtures pequeños se verifica exhaustivamente; la API no afirma
    // optimalidad global para problemas generales.
    let exact = nodes <= 16;
    if exact {
        for mask in 0u64..(1u64 << nodes) {
            let bits = (0..nodes)
                .map(|node| mask & (1u64 << node) != 0)
                .collect::<Vec<_>>();
            let energy = evaluate_qubo(recipe, &bits);
            if energy < best_energy {
                best = bits;
                best_energy = energy;
            }
        }
    }
    let verified_energy = evaluate_qubo(recipe, &best);
    let local_optimum = (0..nodes).all(|node| {
        let mut neighbor = best.clone();
        neighbor[node] = !neighbor[node];
        evaluate_qubo(recipe, &neighbor) + 1.0e-6 >= verified_energy
    });
    QuboSolution {
        bits: best,
        energy: best_energy,
        starts: starts.len(),
        exact,
        local_optimum,
        verified: verified_energy.is_finite()
            && (verified_energy - best_energy).abs() <= 1.0e-6
            && local_optimum,
    }
}

fn coordinate_descent(bits: &mut [bool], unary: &[f32], adjacency: &[Vec<(usize, f32)>]) {
    for _ in 0..(bits.len() * 4).max(1) {
        let mut changed = false;
        for node in 0..bits.len() {
            let field = unary[node]
                + adjacency[node]
                    .iter()
                    .filter(|(other, _)| bits[*other])
                    .map(|(_, weight)| *weight)
                    .sum::<f32>();
            let desired = field < 0.0;
            if desired != bits[node] && field.abs() > EPSILON {
                bits[node] = desired;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn solve_l1(recipe: &OperatorRecipe) -> Result<L1Solution, String> {
    let operator = HodgeL1Operator::from_recipe(recipe).map_err(|error| error.to_string())?;
    if !operator.chain_complex_valid() {
        return Err("el complejo L1 no satisface B1·B2=0".to_string());
    }
    let edge_indices = recipe
        .pair_factors
        .iter()
        .enumerate()
        .map(|(index, pair)| (undirected_key(&pair.a, &pair.b), index))
        .collect::<BTreeMap<_, _>>();
    let mut rhs = vec![Complex32::new(0.0, 0.0); operator.edge_count()];
    for demand in &recipe.flow_demands {
        let edge = edge_indices[&undirected_key(&demand.from, &demand.to)];
        let pair = &recipe.pair_factors[edge];
        let sign = if pair.a == demand.from && pair.b == demand.to {
            1.0
        } else {
            -1.0
        };
        rhs[edge] += sign * Complex32::new(demand.real, demand.imag);
    }
    let mut solution = vec![Complex32::new(0.0, 0.0); operator.edge_count()];
    let mut residual = rhs.clone();
    let mut direction = residual.clone();
    let mut residual_sqr = norm_sqr(&residual);
    let rhs_norm = norm_sqr(&rhs).sqrt().max(EPSILON);
    let mut iterations = 0;
    let max_iterations = (operator.edge_count() * 20).clamp(32, 4_000);
    let mut applied = vec![Complex32::new(0.0, 0.0); operator.edge_count()];
    while iterations < max_iterations && residual_sqr.sqrt() / rhs_norm > 1.0e-5 {
        operator.apply(&direction, &mut applied);
        let denominator = real_dot(&direction, &applied);
        if denominator <= EPSILON {
            break;
        }
        let alpha = residual_sqr / denominator;
        for edge in 0..solution.len() {
            solution[edge] += alpha * direction[edge];
            residual[edge] -= alpha * applied[edge];
        }
        let next_sqr = norm_sqr(&residual);
        let beta = next_sqr / residual_sqr.max(EPSILON);
        for edge in 0..direction.len() {
            direction[edge] = residual[edge] + beta * direction[edge];
        }
        residual_sqr = next_sqr;
        iterations += 1;
    }
    operator.apply(&solution, &mut applied);
    for edge in 0..residual.len() {
        residual[edge] = rhs[edge] - applied[edge];
    }
    let relative_residual = norm_sqr(&residual).sqrt() / rhs_norm;
    Ok(L1Solution {
        edge_flows: solution,
        residual: relative_residual,
        iterations,
        verified: relative_residual.is_finite() && relative_residual <= 1.0e-4,
    })
}

fn norm_sqr(values: &[Complex32]) -> f32 {
    values.iter().map(|value| value.norm_sqr()).sum()
}

fn real_dot(left: &[Complex32], right: &[Complex32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (a.conj() * b).re)
        .sum()
}

fn magnetic_weight_phase(weight: f32, phase: f32) -> (f32, f32) {
    if weight < 0.0 {
        (weight.abs(), phase + std::f32::consts::PI)
    } else {
        (weight, phase)
    }
}

fn undirected_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn core(nodes: usize) -> NativeThermoCdtSubstrate {
        NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: nodes,
            ..NativeThermoCdtConfig::default()
        })
    }

    fn recipe(domains: &[VariableDomain]) -> OperatorRecipe {
        OperatorRecipe {
            name: "fixture".to_string(),
            requested_operator: RequestedOperator::Auto,
            variables: domains
                .iter()
                .enumerate()
                .map(|(index, &domain)| VariableSpec {
                    name: format!("v{index}"),
                    domain,
                })
                .collect(),
            unary_factors: Vec::new(),
            pair_factors: Vec::new(),
            oriented_faces: Vec::new(),
            flow_demands: Vec::new(),
            max_working_set: 32,
            ridge: 0.1,
        }
    }

    #[test]
    fn invalid_recipe_is_rejected_and_auto_selects() {
        let mut invalid = recipe(&[VariableDomain::Binary]);
        invalid.variables.push(invalid.variables[0].clone());
        assert!(matches!(
            invalid.validate(),
            Err(OperatorRecipeError::DuplicateVariable(_))
        ));
        assert_eq!(
            recipe(&[VariableDomain::Binary])
                .selected_operator()
                .unwrap(),
            RequestedOperator::Qubo
        );
        assert_eq!(
            recipe(&[VariableDomain::Phasor])
                .selected_operator()
                .unwrap(),
            RequestedOperator::L0
        );
    }

    #[test]
    fn l0_reduces_energy() {
        let mut input = recipe(&[VariableDomain::Phasor, VariableDomain::Phasor]);
        input.pair_factors.push(PairFactor {
            a: "v0".into(),
            b: "v1".into(),
            weight: -2.0,
            phase: 0.0,
        });
        let solved = NativeMultiOperatorCore::default()
            .solve(&input, &core(64))
            .unwrap();
        let OperatorSolution::L0(solution) = solved.solution else {
            panic!("operador incorrecto")
        };
        assert!(solution.verified);
        assert!(solution.final_energy <= solution.initial_energy + 1.0e-5);
    }

    #[test]
    fn qubo_matches_brute_force_energy() {
        let mut input = recipe(&[
            VariableDomain::Binary,
            VariableDomain::Binary,
            VariableDomain::Binary,
        ]);
        input.unary_factors = vec![
            UnaryFactor {
                variable: "v0".into(),
                weight: -1.0,
                phase: 0.0,
            },
            UnaryFactor {
                variable: "v1".into(),
                weight: -0.7,
                phase: 0.0,
            },
        ];
        input.pair_factors = vec![
            PairFactor {
                a: "v0".into(),
                b: "v1".into(),
                weight: 1.6,
                phase: 0.0,
            },
            PairFactor {
                a: "v1".into(),
                b: "v2".into(),
                weight: -0.4,
                phase: 0.0,
            },
        ];
        let solved = NativeMultiOperatorCore::default()
            .solve(&input, &core(64))
            .unwrap();
        let OperatorSolution::Qubo(solution) = solved.solution else {
            panic!("operador incorrecto")
        };
        let brute = (0..8)
            .map(|mask| {
                let bits = (0..3).map(|bit| mask & (1 << bit) != 0).collect::<Vec<_>>();
                evaluate_qubo(&input, &bits)
            })
            .fold(f32::INFINITY, f32::min);
        assert!((solution.energy - brute).abs() < 1.0e-6);
    }

    fn triangle_recipe() -> OperatorRecipe {
        let mut input = recipe(&[
            VariableDomain::Complex,
            VariableDomain::Complex,
            VariableDomain::Complex,
        ]);
        input.pair_factors = vec![
            PairFactor {
                a: "v0".into(),
                b: "v1".into(),
                weight: 1.0,
                phase: 0.0,
            },
            PairFactor {
                a: "v1".into(),
                b: "v2".into(),
                weight: 1.0,
                phase: 0.0,
            },
            PairFactor {
                a: "v2".into(),
                b: "v0".into(),
                weight: 1.0,
                phase: 0.0,
            },
        ];
        input.oriented_faces.push(OrientedFace {
            vertices: ["v0".into(), "v1".into(), "v2".into()],
        });
        input.flow_demands.push(FlowDemand {
            from: "v0".into(),
            to: "v1".into(),
            real: 1.0,
            imag: 0.25,
        });
        input
    }

    #[test]
    fn l1_is_hermitian_psd_and_cg_residual_is_low() {
        let input = triangle_recipe();
        let operator = HodgeL1Operator::from_recipe(&input).unwrap();
        assert!(operator.chain_complex_valid());
        let x = vec![
            Complex32::new(1.0, 0.2),
            Complex32::new(-0.3, 0.7),
            Complex32::new(0.4, -0.5),
        ];
        let y = vec![
            Complex32::new(0.1, 0.8),
            Complex32::new(0.6, -0.2),
            Complex32::new(-0.4, 0.3),
        ];
        let mut ax = vec![Complex32::new(0.0, 0.0); 3];
        let mut ay = ax.clone();
        operator.apply(&x, &mut ax);
        operator.apply(&y, &mut ay);
        let lhs: Complex32 = x.iter().zip(&ay).map(|(&a, &b)| a.conj() * b).sum();
        let rhs: Complex32 = ax.iter().zip(&y).map(|(&a, &b)| a.conj() * b).sum();
        assert!((lhs - rhs).norm() < 1.0e-5);
        let form = operator.quadratic_form(&x);
        assert!(form.re >= -1.0e-5 && form.im.abs() < 1.0e-5);
        let solved = NativeMultiOperatorCore::default()
            .solve(&input, &core(64))
            .unwrap();
        let OperatorSolution::L1(solution) = solved.solution else {
            panic!("operador incorrecto")
        };
        assert!(solution.verified, "residual={}", solution.residual);
    }

    #[test]
    fn working_set_and_consolidation_are_sparse() {
        let mut global = core(128);
        let before_states = global.thermal_state.clone();
        let before_edges = global.edge_count();
        let mut input = recipe(&[VariableDomain::Binary, VariableDomain::Binary]);
        input.unary_factors.push(UnaryFactor {
            variable: "v0".into(),
            weight: -1.0,
            phase: 0.0,
        });
        input.pair_factors.push(PairFactor {
            a: "v0".into(),
            b: "v1".into(),
            weight: -0.5,
            phase: 0.0,
        });
        let mut engine = NativeMultiOperatorCore::default();
        let solved = engine
            .solve_and_consolidate(&input, &mut global, 0.5)
            .unwrap();
        assert!(solved.working_set.global_nodes.len() < global.node_count());
        let selected = solved
            .working_set
            .global_nodes
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        for node in 0..global.node_count() {
            if !selected.contains(&node) {
                assert_eq!(global.thermal_state[node], before_states[node]);
            }
        }
        assert!(global.edge_count() >= before_edges);
        assert_eq!(engine.snapshot.node_deltas.len(), 2);
    }

    #[test]
    fn compact_snapshot_round_trip() {
        let mut snapshot = OperatorDeltaSnapshot::default();
        snapshot.accepted_recipes.push(triangle_recipe());
        snapshot.mappings.push(VariableNodeMapping {
            variable: "v0".into(),
            global_node: 17,
        });
        let path = std::env::temp_dir().join(format!(
            "operator-delta-{}-{}.json",
            std::process::id(),
            stable_hash(b"round-trip")
        ));
        snapshot.save(&path).unwrap();
        let encoded = std::fs::read(&path).unwrap();
        let loaded = OperatorDeltaSnapshot::load(&path).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(loaded, snapshot);
        assert!(encoded.len() < 8_192);
    }
}
