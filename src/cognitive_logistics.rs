//! Planificación logística tipada sobre CDT-RQM-EPR.
//!
//! El sustrato aprende priors procedurales de transiciones primitivas. La
//! memoria de trabajo ejecutiva compone esas habilidades mediante búsqueda
//! beam y un modelo de mundo explícito. La corrección final se comprueba con
//! un verificador independiente.

use crate::native_thermo_rqm_epr::NativeThermoRqmEprSubstrate;
use crate::relational_field::ObserverId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

const PROCEDURAL_OBSERVER: ObserverId = ObserverId(890_004);
const PROCEDURAL_PHASE: f32 = std::f32::consts::PI;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Location(pub u8);

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Package(pub u8);

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct LogisticsState {
    pub robot_at: Location,
    pub package_at: Vec<Option<Location>>,
    pub carrying: Option<Package>,
    pub has_key: bool,
    pub connections: Vec<(Location, Location)>,
    pub locked_edges: Vec<(Location, Location)>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum LogisticsAction {
    Move(Location),
    Pickup(Package),
    Drop(Package),
    Unlock(Location),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct LogisticsGoal {
    pub package: Package,
    pub destination: Location,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LogisticsTask {
    pub id: String,
    pub initial: LogisticsState,
    pub goal: LogisticsGoal,
    pub max_steps: usize,
}

#[derive(Clone, Debug)]
pub struct LogisticsDecision {
    pub plan: Option<Vec<LogisticsAction>>,
    pub expanded_states: usize,
    pub confidence: f32,
    pub abstained: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct LogisticsPlannerConfig {
    pub beam_width: usize,
    pub pattern_width: usize,
    pub procedural_gain: f32,
    pub max_expansions: usize,
    pub use_handcrafted_schemas: bool,
    pub use_learned_schemas: bool,
}

impl Default for LogisticsPlannerConfig {
    fn default() -> Self {
        Self {
            beam_width: 24,
            pattern_width: 4,
            procedural_gain: 0.12,
            max_expansions: 4_096,
            use_handcrafted_schemas: true,
            use_learned_schemas: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PrimitiveEpisode {
    pub before: LogisticsState,
    pub action: LogisticsAction,
    pub after: LogisticsState,
    pub reward: f32,
}

#[derive(Clone, Debug)]
pub struct LogisticsController {
    pub substrate: NativeThermoRqmEprSubstrate,
    pub config: LogisticsPlannerConfig,
    patterns: HashMap<String, Vec<usize>>,
    next_pattern_slot: usize,
    episodes: Vec<PrimitiveEpisode>,
    learned_schemas: HashSet<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LogisticsControllerSnapshot {
    pub schema_version: u32,
    pub config: LogisticsPlannerConfig,
    pub patterns: HashMap<String, Vec<usize>>,
    pub next_pattern_slot: usize,
    pub episodes: Vec<PrimitiveEpisode>,
    pub learned_schemas: HashSet<String>,
}

#[derive(Clone)]
struct PlanNode {
    state: LogisticsState,
    plan: Vec<LogisticsAction>,
    score: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Verification {
    pub actions_valid: bool,
    pub goal_reached: bool,
    pub executed_steps: usize,
}

impl LogisticsState {
    pub fn canonicalize(&mut self) {
        for edge in &mut self.connections {
            *edge = ordered_edge(edge.0, edge.1);
        }
        self.connections.sort_unstable();
        self.connections.dedup();
        for edge in &mut self.locked_edges {
            *edge = ordered_edge(edge.0, edge.1);
        }
        self.locked_edges.sort_unstable();
        self.locked_edges.dedup();
    }

    pub fn goal_reached(&self, goal: LogisticsGoal) -> bool {
        self.carrying != Some(goal.package)
            && self
                .package_at
                .get(goal.package.0 as usize)
                .copied()
                .flatten()
                == Some(goal.destination)
    }

    pub fn valid_actions(&self) -> Vec<LogisticsAction> {
        let mut actions = Vec::new();
        for &(a, b) in &self.connections {
            let destination = if a == self.robot_at {
                Some(b)
            } else if b == self.robot_at {
                Some(a)
            } else {
                None
            };
            let Some(destination) = destination else {
                continue;
            };
            if self
                .locked_edges
                .contains(&ordered_edge(self.robot_at, destination))
            {
                if self.has_key {
                    actions.push(LogisticsAction::Unlock(destination));
                }
            } else {
                actions.push(LogisticsAction::Move(destination));
            }
        }
        if self.carrying.is_none() {
            for (index, location) in self.package_at.iter().enumerate() {
                if *location == Some(self.robot_at) {
                    actions.push(LogisticsAction::Pickup(Package(index as u8)));
                }
            }
        } else if let Some(package) = self.carrying {
            actions.push(LogisticsAction::Drop(package));
        }
        actions.sort_by_key(action_sort_key);
        actions
    }

    pub fn apply(&self, action: LogisticsAction) -> Option<Self> {
        if !self.valid_actions().contains(&action) {
            return None;
        }
        let mut next = self.clone();
        match action {
            LogisticsAction::Move(destination) => next.robot_at = destination,
            LogisticsAction::Pickup(package) => {
                next.package_at[package.0 as usize] = None;
                next.carrying = Some(package);
            }
            LogisticsAction::Drop(package) => {
                next.package_at[package.0 as usize] = Some(next.robot_at);
                next.carrying = None;
            }
            LogisticsAction::Unlock(destination) => {
                let edge = ordered_edge(next.robot_at, destination);
                next.locked_edges.retain(|candidate| *candidate != edge);
            }
        }
        Some(next)
    }
}

impl LogisticsController {
    pub fn new(substrate: NativeThermoRqmEprSubstrate, config: LogisticsPlannerConfig) -> Self {
        Self {
            substrate,
            config,
            patterns: HashMap::new(),
            next_pattern_slot: 0,
            episodes: Vec::new(),
            learned_schemas: HashSet::new(),
        }
    }

    pub fn from_snapshot(
        substrate: NativeThermoRqmEprSubstrate,
        snapshot: LogisticsControllerSnapshot,
    ) -> Result<Self, String> {
        if snapshot.schema_version != 1 {
            return Err(format!(
                "versión de snapshot cognitivo no soportada: {}",
                snapshot.schema_version
            ));
        }
        Ok(Self {
            substrate,
            config: snapshot.config,
            patterns: snapshot.patterns,
            next_pattern_slot: snapshot.next_pattern_slot,
            episodes: snapshot.episodes,
            learned_schemas: snapshot.learned_schemas,
        })
    }

    pub fn snapshot(&self) -> LogisticsControllerSnapshot {
        LogisticsControllerSnapshot {
            schema_version: 1,
            config: self.config,
            patterns: self.patterns.clone(),
            next_pattern_slot: self.next_pattern_slot,
            episodes: self.episodes.clone(),
            learned_schemas: self.learned_schemas.clone(),
        }
    }

    pub fn pattern_entries(&self) -> impl Iterator<Item = (&str, &[usize])> {
        self.patterns
            .iter()
            .map(|(key, nodes)| (key.as_str(), nodes.as_slice()))
    }

    pub fn learned_schema_entries(&self) -> impl Iterator<Item = &str> {
        self.learned_schemas.iter().map(String::as_str)
    }

    pub fn episodes(&self) -> &[PrimitiveEpisode] {
        &self.episodes
    }

    pub fn learned_schema_count(&self) -> usize {
        self.learned_schemas.len()
    }

    pub fn observe(&mut self, episode: PrimitiveEpisode) {
        self.observe_with_goal(episode, None);
    }

    /// Memoria procedural condicionada por una meta. Sigue siendo una
    /// transición primitiva; no almacena ni recibe el plan completo.
    pub fn observe_for_goal(&mut self, episode: PrimitiveEpisode, goal: LogisticsGoal) {
        self.observe_with_goal(episode, Some(goal));
    }

    /// Consolidación térmica de una acción que recibió recompensa externa.
    /// No toca relaciones RQM, permitiendo una ablación causal pareada.
    pub fn incubate_action_thermal(
        &mut self,
        action: LogisticsAction,
        strength: f32,
        pulses: usize,
    ) {
        let nodes = self.action_nodes(action);
        for pulse in 0..pulses.max(1) {
            let phase = PROCEDURAL_PHASE + pulse as f32 * 0.01;
            for &node in &nodes {
                self.substrate.thermal.inject_local_node(
                    node,
                    2.0 + 2.0 * strength.clamp(0.0, 1.0),
                    phase,
                    strength.clamp(0.0, 1.0),
                );
            }
            self.substrate.thermal.step();
        }
    }

    pub fn action_thermal_signal(&mut self, action: LogisticsAction) -> f32 {
        let nodes = self.action_nodes(action);
        self.mean_thermal_signal(&nodes)
    }

    /// Sueño sobre un esquema abstracto (por ejemplo, `move:open_progress`).
    /// La memoria resultante puede reutilizarse con otros IDs de entidades.
    pub fn incubate_action_schema_thermal(
        &mut self,
        state: &LogisticsState,
        goal: LogisticsGoal,
        action: LogisticsAction,
        strength: f32,
        pulses: usize,
    ) {
        let nodes = self.action_schema_nodes(state, goal, action);
        for pulse in 0..pulses.max(1) {
            let phase = PROCEDURAL_PHASE + pulse as f32 * 0.01;
            for &node in &nodes {
                self.substrate.thermal.inject_local_node(
                    node,
                    2.0 + 2.0 * strength.clamp(0.0, 1.0),
                    phase,
                    strength.clamp(0.0, 1.0),
                );
            }
            self.substrate.thermal.step();
        }
    }

    pub fn action_schema_thermal_signal(
        &mut self,
        state: &LogisticsState,
        goal: LogisticsGoal,
        action: LogisticsAction,
    ) -> f32 {
        let nodes = self.action_schema_nodes(state, goal, action);
        self.mean_thermal_signal(&nodes)
    }

    /// Consolida el prototipo inducido por los efectos de una transición. El
    /// nombre y los IDs concretos no forman parte de la firma.
    pub fn incubate_learned_schema_thermal(
        &mut self,
        state: &LogisticsState,
        goal: LogisticsGoal,
        action: LogisticsAction,
        strength: f32,
        pulses: usize,
    ) -> bool {
        let Some(key) = self.learned_schema_key(state, goal, action) else {
            return false;
        };
        if !self.learned_schemas.contains(&key) {
            return false;
        }
        let nodes = self.pattern_nodes(format!("learned_action_schema:{key}"));
        for pulse in 0..pulses.max(1) {
            let phase = PROCEDURAL_PHASE + pulse as f32 * 0.01;
            for &node in &nodes {
                self.substrate.thermal.inject_local_node(
                    node,
                    2.0 + 2.0 * strength.clamp(0.0, 1.0),
                    phase,
                    strength.clamp(0.0, 1.0),
                );
            }
            self.substrate.thermal.step();
        }
        // Cierre homeostático: la última operación escribe memoria estable y
        // no vuelve a añadir ruido después de consolidarla.
        for &node in &nodes {
            self.substrate
                .thermal
                .inject_local_node(node, 4.0, PROCEDURAL_PHASE, 1.0);
        }
        true
    }

    pub fn learned_schema_thermal_signal(
        &mut self,
        state: &LogisticsState,
        goal: LogisticsGoal,
        action: LogisticsAction,
    ) -> Option<f32> {
        let key = self.learned_schema_key(state, goal, action)?;
        if !self.learned_schemas.contains(&key) {
            return None;
        }
        let nodes = self.pattern_nodes(format!("learned_action_schema:{key}"));
        Some(self.mean_thermal_signal(&nodes))
    }

    fn mean_thermal_signal(&self, nodes: &[usize]) -> f32 {
        nodes
            .iter()
            .map(|&node| {
                self.substrate.thermal.thermal_state[node].tanh()
                    + 0.1 * self.substrate.thermal.amplitude[node]
            })
            .sum::<f32>()
            / nodes.len().max(1) as f32
    }

    fn observe_with_goal(&mut self, episode: PrimitiveEpisode, goal: Option<LogisticsGoal>) {
        if self.config.use_learned_schemas {
            if let Some(goal) = goal {
                if let Some(key) =
                    effect_signature_key(&episode.before, &episode.after, goal, episode.action)
                {
                    self.learned_schemas.insert(key);
                }
            }
        }
        let mut seeds = self.state_feature_nodes(&episode.before, goal);
        seeds.extend(self.transition_feature_nodes(&episode.before, episode.action));
        seeds.sort_unstable();
        seeds.dedup();
        let action_nodes = match goal {
            Some(goal) => self.decision_nodes(&episode.before, goal, episode.action),
            None => self.action_nodes(episode.action),
        };
        self.substrate.train_observed_transition(
            PROCEDURAL_OBSERVER,
            PROCEDURAL_PHASE,
            &seeds,
            &action_nodes,
            episode.reward.clamp(0.0, 1.0),
        );
        self.episodes.push(episode);
    }

    /// Replay offline de experiencias primitivas. No usa casos de test.
    pub fn dream_replay(&mut self, passes: usize) {
        let episodes = self.episodes.clone();
        for _ in 0..passes.max(1) {
            for episode in &episodes {
                let mut seeds = self.state_feature_nodes(&episode.before, None);
                seeds.extend(self.transition_feature_nodes(&episode.before, episode.action));
                seeds.sort_unstable();
                seeds.dedup();
                let action_nodes = self.action_nodes(episode.action);
                self.substrate.train_observed_transition(
                    PROCEDURAL_OBSERVER,
                    PROCEDURAL_PHASE,
                    &seeds,
                    &action_nodes,
                    episode.reward.clamp(0.0, 1.0),
                );
            }
        }
        self.substrate.thermal.run_until_stable(6, 1.0e-5, 1.0e-5);
    }

    /// Planifica sin expected, distractor ni plan óptimo.
    pub fn plan(&mut self, task: &LogisticsTask) -> LogisticsDecision {
        if task.initial.goal_reached(task.goal) {
            return LogisticsDecision {
                plan: Some(Vec::new()),
                expanded_states: 0,
                confidence: 1.0,
                abstained: false,
            };
        }
        let mut initial = task.initial.clone();
        initial.canonicalize();
        let mut frontier = vec![PlanNode {
            score: -(self.estimated_remaining(&initial, task.goal) as f32),
            state: initial.clone(),
            plan: Vec::new(),
        }];
        let mut visited = HashMap::<LogisticsState, usize>::new();
        visited.insert(initial, 0);
        let mut expanded_states = 0usize;

        for depth in 0..task.max_steps {
            let mut next_frontier = Vec::new();
            for node in frontier {
                if expanded_states >= self.config.max_expansions {
                    break;
                }
                let actions = node.state.valid_actions();
                let priors = self.action_priors(&node.state, task.goal, &actions);
                for action in actions {
                    let Some(next_state) = node.state.apply(action) else {
                        continue;
                    };
                    expanded_states += 1;
                    let next_depth = depth + 1;
                    if visited
                        .get(&next_state)
                        .is_some_and(|&seen_depth| seen_depth <= next_depth)
                    {
                        continue;
                    }
                    visited.insert(next_state.clone(), next_depth);
                    let mut plan = node.plan.clone();
                    plan.push(action);
                    if next_state.goal_reached(task.goal) {
                        let confidence = (1.0 / (1.0 + 0.08 * plan.len() as f32)).clamp(0.0, 1.0);
                        return LogisticsDecision {
                            plan: Some(plan),
                            expanded_states,
                            confidence,
                            abstained: false,
                        };
                    }
                    let prior = priors.get(&action).copied().unwrap_or(0.0);
                    let remaining = self.estimated_remaining(&next_state, task.goal) as f32;
                    next_frontier.push(PlanNode {
                        state: next_state,
                        plan,
                        score: -remaining - 0.08 * next_depth as f32
                            + self.config.procedural_gain * (1.0 + prior).ln(),
                    });
                }
            }
            next_frontier.sort_by(|left, right| {
                right
                    .score
                    .total_cmp(&left.score)
                    .then_with(|| left.plan.len().cmp(&right.plan.len()))
            });
            next_frontier.truncate(self.config.beam_width.max(1));
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }
        LogisticsDecision {
            plan: None,
            expanded_states,
            confidence: 0.0,
            abstained: true,
        }
    }

    pub fn verify(task: &LogisticsTask, plan: &[LogisticsAction]) -> Verification {
        let mut state = task.initial.clone();
        state.canonicalize();
        for (index, &action) in plan.iter().enumerate() {
            let Some(next) = state.apply(action) else {
                return Verification {
                    actions_valid: false,
                    goal_reached: false,
                    executed_steps: index,
                };
            };
            state = next;
        }
        Verification {
            actions_valid: true,
            goal_reached: state.goal_reached(task.goal),
            executed_steps: plan.len(),
        }
    }

    fn action_priors(
        &mut self,
        state: &LogisticsState,
        goal: LogisticsGoal,
        actions: &[LogisticsAction],
    ) -> HashMap<LogisticsAction, f32> {
        if actions.is_empty() {
            return HashMap::new();
        }
        let mut seeds = self.state_feature_nodes(state, Some(goal));
        for &action in actions {
            seeds.extend(self.transition_feature_nodes(state, action));
        }
        seeds.sort_unstable();
        seeds.dedup();
        let report = self
            .substrate
            .query(PROCEDURAL_OBSERVER, PROCEDURAL_PHASE, &seeds);
        let node_scores = report
            .candidates
            .iter()
            .map(|candidate| (candidate.agent, candidate.score))
            .collect::<HashMap<_, _>>();
        actions
            .iter()
            .copied()
            .map(|action| {
                let score = self
                    .decision_nodes(state, goal, action)
                    .iter()
                    .filter_map(|node| node_scores.get(node))
                    .sum();
                (action, score)
            })
            .collect()
    }

    fn estimated_remaining(&self, state: &LogisticsState, goal: LogisticsGoal) -> usize {
        if state.goal_reached(goal) {
            return 0;
        }
        if state.carrying == Some(goal.package) {
            return optimistic_distance(state, state.robot_at, goal.destination).unwrap_or(64) + 1;
        }
        let Some(package_location) = state
            .package_at
            .get(goal.package.0 as usize)
            .copied()
            .flatten()
        else {
            return 64;
        };
        let to_package = optimistic_distance(state, state.robot_at, package_location).unwrap_or(64);
        let to_goal = optimistic_distance(state, package_location, goal.destination).unwrap_or(64);
        to_package + 1 + to_goal + 1
    }

    fn state_feature_nodes(
        &mut self,
        state: &LogisticsState,
        goal: Option<LogisticsGoal>,
    ) -> Vec<usize> {
        let mut keys = vec![
            format!("robot_at:{}", state.robot_at.0),
            format!("has_key:{}", state.has_key),
            match state.carrying {
                Some(package) => format!("carrying:{}", package.0),
                None => "carrying:none".to_string(),
            },
        ];
        for (index, location) in state.package_at.iter().enumerate() {
            if let Some(location) = location {
                keys.push(format!("package:{index}:at:{}", location.0));
            }
        }
        for &(a, b) in &state.locked_edges {
            keys.push(format!("locked:{}:{}", a.0, b.0));
        }
        if let Some(goal) = goal {
            keys.push(format!("goal:{}:at:{}", goal.package.0, goal.destination.0));
            if self.config.use_learned_schemas {
                keys.push("abstract:any_goal_state".to_string());
            }
            keys.push(format!(
                "abstract:carrying_goal:{}",
                state.carrying == Some(goal.package)
            ));
            keys.push(format!(
                "abstract:goal_package_here:{}",
                state
                    .package_at
                    .get(goal.package.0 as usize)
                    .copied()
                    .flatten()
                    == Some(state.robot_at)
            ));
            keys.push(format!(
                "abstract:robot_at_goal:{}",
                state.robot_at == goal.destination
            ));
            let open_distance =
                open_distance(state, state.robot_at, goal.destination).unwrap_or(usize::MAX);
            keys.push(format!(
                "abstract:goal_open_distance:{}",
                open_distance.min(8)
            ));
        }
        keys.push(format!(
            "abstract:has_locked_edges:{}",
            !state.locked_edges.is_empty()
        ));
        keys.push(format!("abstract:has_key:{}", state.has_key));
        keys.into_iter()
            .flat_map(|key| self.pattern_nodes(key))
            .collect()
    }

    fn transition_feature_nodes(
        &mut self,
        state: &LogisticsState,
        action: LogisticsAction,
    ) -> Vec<usize> {
        let key = match action {
            LogisticsAction::Move(destination) => {
                format!("can_move:{}:{}", state.robot_at.0, destination.0)
            }
            LogisticsAction::Pickup(package) => {
                format!("can_pickup:{}:{}", package.0, state.robot_at.0)
            }
            LogisticsAction::Drop(package) => {
                format!("can_drop:{}:{}", package.0, state.robot_at.0)
            }
            LogisticsAction::Unlock(destination) => {
                format!("can_unlock:{}:{}", state.robot_at.0, destination.0)
            }
        };
        self.pattern_nodes(key)
    }

    fn action_nodes(&mut self, action: LogisticsAction) -> Vec<usize> {
        self.pattern_nodes(format!("action:{action:?}"))
    }

    fn decision_nodes(
        &mut self,
        state: &LogisticsState,
        goal: LogisticsGoal,
        action: LogisticsAction,
    ) -> Vec<usize> {
        let mut nodes = self.action_nodes(action);
        if self.config.use_handcrafted_schemas {
            nodes.extend(self.action_schema_nodes(state, goal, action));
        }
        if self.config.use_learned_schemas {
            if let Some(key) = self.learned_schema_key(state, goal, action) {
                if self.learned_schemas.contains(&key) {
                    nodes.extend(self.pattern_nodes(format!("learned_action_schema:{key}")));
                }
            }
        }
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    fn action_schema_nodes(
        &mut self,
        state: &LogisticsState,
        goal: LogisticsGoal,
        action: LogisticsAction,
    ) -> Vec<usize> {
        let role = action_schema_role(state, goal, action);
        self.pattern_nodes(format!("action_schema:{role}"))
    }

    fn learned_schema_key(
        &self,
        state: &LogisticsState,
        goal: LogisticsGoal,
        action: LogisticsAction,
    ) -> Option<String> {
        let after = state.apply(action)?;
        effect_signature_key(state, &after, goal, action)
    }

    fn pattern_nodes(&mut self, key: String) -> Vec<usize> {
        if let Some(nodes) = self.patterns.get(&key) {
            return nodes.clone();
        }
        let width = self.config.pattern_width.max(1);
        let node_count = self.substrate.thermal.node_count().max(1);
        let capacity = (node_count / width).max(1);
        let slot = self.next_pattern_slot;
        let nodes = if slot < capacity {
            (0..width)
                .map(|projection| projection * capacity + slot)
                .filter(|&node| node < node_count)
                .collect::<Vec<_>>()
        } else {
            // El benchmark dimensiona el sustrato para no alcanzar esta rama.
            (0..width)
                .map(|projection| (slot + projection * 977) % node_count)
                .collect()
        };
        self.next_pattern_slot = self.next_pattern_slot.saturating_add(1);
        self.patterns.insert(key, nodes.clone());
        nodes
    }
}

fn optimistic_distance(
    state: &LogisticsState,
    start: Location,
    destination: Location,
) -> Option<usize> {
    if start == destination {
        return Some(0);
    }
    let mut queue = VecDeque::from([(start, 0usize)]);
    let mut seen = HashSet::from([start]);
    while let Some((location, distance)) = queue.pop_front() {
        for &(a, b) in &state.connections {
            let next = if a == location {
                Some(b)
            } else if b == location {
                Some(a)
            } else {
                None
            };
            let Some(next) = next else {
                continue;
            };
            if next == destination {
                return Some(distance + 1);
            }
            if seen.insert(next) {
                queue.push_back((next, distance + 1));
            }
        }
    }
    None
}

fn open_distance(state: &LogisticsState, start: Location, destination: Location) -> Option<usize> {
    if start == destination {
        return Some(0);
    }
    let mut queue = VecDeque::from([(start, 0usize)]);
    let mut seen = HashSet::from([start]);
    while let Some((location, distance)) = queue.pop_front() {
        for &(a, b) in &state.connections {
            if state.locked_edges.contains(&ordered_edge(a, b)) {
                continue;
            }
            let next = if a == location {
                Some(b)
            } else if b == location {
                Some(a)
            } else {
                None
            };
            let Some(next) = next else {
                continue;
            };
            if next == destination {
                return Some(distance + 1);
            }
            if seen.insert(next) {
                queue.push_back((next, distance + 1));
            }
        }
    }
    None
}

fn action_schema_role(
    state: &LogisticsState,
    goal: LogisticsGoal,
    action: LogisticsAction,
) -> &'static str {
    match action {
        LogisticsAction::Move(destination) => {
            let target = if state.carrying == Some(goal.package) {
                Some(goal.destination)
            } else {
                state
                    .package_at
                    .get(goal.package.0 as usize)
                    .copied()
                    .flatten()
            };
            let Some(target) = target else {
                return "move:unknown";
            };
            let before = open_distance(state, state.robot_at, target).unwrap_or(usize::MAX);
            let after = open_distance(state, destination, target).unwrap_or(usize::MAX);
            if after < before {
                "move:open_progress"
            } else if after > before {
                "move:open_regress"
            } else {
                "move:open_lateral"
            }
        }
        LogisticsAction::Pickup(package) => {
            if package == goal.package {
                "pickup:goal_package"
            } else {
                "pickup:other_package"
            }
        }
        LogisticsAction::Drop(package) => {
            if package == goal.package && state.robot_at == goal.destination {
                "drop:goal_reached"
            } else {
                "drop:premature"
            }
        }
        LogisticsAction::Unlock(destination) => {
            let Some(next) = state.apply(action) else {
                return "unlock:invalid";
            };
            let before =
                open_distance(state, state.robot_at, goal.destination).unwrap_or(usize::MAX);
            let after = open_distance(&next, destination, goal.destination).unwrap_or(usize::MAX);
            if after < before {
                "unlock:opens_progress"
            } else {
                "unlock:no_progress"
            }
        }
    }
}

/// Firma inducida por diferencias observables. No contiene IDs de ubicaciones,
/// paquetes ni nombres de acciones grounded.
fn effect_signature_key(
    before: &LogisticsState,
    after: &LogisticsState,
    goal: LogisticsGoal,
    action: LogisticsAction,
) -> Option<String> {
    let action_kind = match action {
        LogisticsAction::Move(_) => 0,
        LogisticsAction::Pickup(_) => 1,
        LogisticsAction::Drop(_) => 2,
        LogisticsAction::Unlock(_) => 3,
    };
    let target_before = relevant_target(before, goal)?;
    let target_after = relevant_target(after, goal).unwrap_or(target_before);
    let distance_before = open_distance(before, before.robot_at, target_before);
    let distance_after = open_distance(after, after.robot_at, target_after);
    let distance_delta = signed_option_delta(distance_before, distance_after);
    let carrying_before = carrying_role(before, goal);
    let carrying_after = carrying_role(after, goal);
    let goal_delta = bool_delta(before.goal_reached(goal), after.goal_reached(goal));
    let locked_delta = signed_delta(before.locked_edges.len(), after.locked_edges.len());
    Some(format!(
        "kind={action_kind};distance={distance_delta};carry={carrying_before}>{carrying_after};goal={goal_delta};locks={locked_delta}"
    ))
}

fn relevant_target(state: &LogisticsState, goal: LogisticsGoal) -> Option<Location> {
    if state.carrying == Some(goal.package) {
        Some(goal.destination)
    } else {
        state
            .package_at
            .get(goal.package.0 as usize)
            .copied()
            .flatten()
    }
}

fn carrying_role(state: &LogisticsState, goal: LogisticsGoal) -> i8 {
    match state.carrying {
        Some(package) if package == goal.package => 1,
        Some(_) => -1,
        None => 0,
    }
}

fn bool_delta(before: bool, after: bool) -> i8 {
    i8::from(after) - i8::from(before)
}

fn signed_delta(before: usize, after: usize) -> i8 {
    match after.cmp(&before) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

fn signed_option_delta(before: Option<usize>, after: Option<usize>) -> i8 {
    match (before, after) {
        (Some(before), Some(after)) => signed_delta(before, after),
        (None, Some(_)) => -1,
        (Some(_), None) => 1,
        (None, None) => 0,
    }
}

fn ordered_edge(a: Location, b: Location) -> (Location, Location) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn action_sort_key(action: &LogisticsAction) -> (u8, u8) {
    match action {
        LogisticsAction::Move(location) => (0, location.0),
        LogisticsAction::Pickup(package) => (1, package.0),
        LogisticsAction::Drop(package) => (2, package.0),
        LogisticsAction::Unlock(location) => (3, location.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entanglement::EntanglementConfig;
    use crate::native_thermo_rqm_epr::NativeThermoRqmConfig;
    use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;

    fn controller() -> LogisticsController {
        LogisticsController::new(
            NativeThermoRqmEprSubstrate::new(
                NativeThermoCdtConfig {
                    slices: 4,
                    nodes_per_slice: 256,
                    seed: 19,
                    ..NativeThermoCdtConfig::default()
                },
                NativeThermoRqmConfig {
                    thermal_steps_per_train: 0,
                    thermal_steps_per_query: 0,
                    ..NativeThermoRqmConfig::default()
                },
                EntanglementConfig {
                    max_syncs_per_step: 0,
                    ..EntanglementConfig::default()
                },
            ),
            LogisticsPlannerConfig::default(),
        )
    }

    fn line_state() -> LogisticsState {
        LogisticsState {
            robot_at: Location(0),
            package_at: vec![Some(Location(0))],
            carrying: None,
            has_key: false,
            connections: vec![(Location(0), Location(1)), (Location(1), Location(2))],
            locked_edges: Vec::new(),
        }
    }

    #[test]
    fn plans_four_step_delivery_and_verifies_it() {
        let task = LogisticsTask {
            id: "delivery".into(),
            initial: line_state(),
            goal: LogisticsGoal {
                package: Package(0),
                destination: Location(2),
            },
            max_steps: 4,
        };
        let decision = controller().plan(&task);
        let plan = decision.plan.unwrap();
        assert_eq!(plan.len(), 4);
        let verification = LogisticsController::verify(&task, &plan);
        assert!(verification.actions_valid && verification.goal_reached);
    }

    #[test]
    fn abstains_when_destination_is_disconnected() {
        let task = LogisticsTask {
            id: "impossible".into(),
            initial: line_state(),
            goal: LogisticsGoal {
                package: Package(0),
                destination: Location(9),
            },
            max_steps: 6,
        };
        let decision = controller().plan(&task);
        assert!(decision.abstained);
        assert!(decision.plan.is_none());
    }

    #[test]
    fn induces_distinct_effect_schemas_without_grounded_ids() {
        let mut controller = controller();
        controller.config.use_handcrafted_schemas = false;
        controller.config.use_learned_schemas = true;
        let mut before = LogisticsState {
            robot_at: Location(0),
            package_at: vec![None],
            carrying: Some(Package(0)),
            has_key: false,
            connections: vec![
                (Location(0), Location(1)),
                (Location(1), Location(3)),
                (Location(0), Location(4)),
                (Location(4), Location(5)),
                (Location(5), Location(3)),
            ],
            locked_edges: vec![(Location(1), Location(3))],
        };
        before.canonicalize();
        let goal = LogisticsGoal {
            package: Package(0),
            destination: Location(3),
        };
        for action in [
            LogisticsAction::Move(Location(1)),
            LogisticsAction::Move(Location(4)),
        ] {
            controller.observe_for_goal(
                PrimitiveEpisode {
                    before: before.clone(),
                    action,
                    after: before.apply(action).unwrap(),
                    reward: 1.0,
                },
                goal,
            );
        }
        assert_eq!(controller.learned_schema_count(), 2);
    }
}
