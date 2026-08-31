//! Grafo de 6 agentes-nodo sobre un unico runtime Gemma.
//!
//! El router no llama al Transformer. El camino feliz es un forward.

use crate::gemma_operator_bridge::parse_operator_recipe;
use crate::layer_route_cache::{fingerprint_overlap, LayerRouteCache};
use crate::native_checkpoint::atomic_write;
use crate::native_gemma2::LayerExecutionMask;
use crate::native_gemma2_runtime::GEMMA2_FORCED_LANGUAGE;
use crate::native_rng::{splitmix64, unit_from_u64};
use crate::thermo_router::ActivationFingerprint;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;

pub const AGENT_GRAPH_FILE: &str = "agent-graph.json";
pub const ABSTAIN_REPLY: &str = "No dispongo de un resultado fiable. Me abstengo.";

const STATE_VERSION: u32 = 1;
const CLAIM_OVERLAP: f32 = 0.35;

pub const ROUTER_ID: NodeId = NodeId(0);
pub const FAST_TALKER_ID: NodeId = NodeId(1);
pub const DENSE_TALKER_ID: NodeId = NodeId(2);
pub const VERIFIER_ID: NodeId = NodeId(3);
pub const COMPILER_ID: NodeId = NodeId(4);
pub const MEMORY_ID: NodeId = NodeId(5);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct NodeId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AgentRole {
    Router,
    FastTalker,
    DenseTalker,
    Verifier,
    Compiler,
    Memory,
}

impl AgentRole {
    pub fn is_speaker(self) -> bool {
        matches!(self, Self::FastTalker | Self::DenseTalker | Self::Compiler)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentBudget {
    pub max_new_tokens: usize,
    pub max_executed_layers: usize,
    pub timeout_ms: u64,
}

impl Default for AgentBudget {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            max_executed_layers: 26,
            timeout_ms: 120_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentStats {
    pub calls: u64,
    pub accepts: u64,
    pub rejects: u64,
    pub mean_ms: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MessageKind {
    Query,
    Reply,
    Reject,
    Metric,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentMessage {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: MessageKind,
    pub tokens: Option<Vec<u32>>,
    pub text: Option<String>,
    pub route: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentNode {
    pub id: NodeId,
    pub role: AgentRole,
    pub route_pref: Option<u64>,
    pub claim: ActivationFingerprint,
    pub budget: AgentBudget,
    pub stats: AgentStats,
    #[serde(skip)]
    pub mailbox: VecDeque<AgentMessage>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentEdge {
    pub src: NodeId,
    pub dst: NodeId,
    pub weight: f32,
    pub successes: u32,
    pub failures: u32,
    pub mean_latency_ms: f32,
    pub last_used: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentGraphConfig {
    pub min_claim_overlap: f32,
    pub epsilon: f32,
    pub alpha: f32,
    pub lambda_quality: f32,
    pub mu_fallback: f32,
    pub seed: u64,
}

impl Default for AgentGraphConfig {
    fn default() -> Self {
        Self {
            min_claim_overlap: CLAIM_OVERLAP,
            epsilon: 0.05,
            alpha: 0.10,
            lambda_quality: 1.0,
            mu_fallback: 0.50,
            seed: 0xA6E0_76A0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentGraphState {
    version: u32,
    generation: u64,
    config: AgentGraphConfig,
    nodes: Vec<AgentNode>,
    edges: Vec<AgentEdge>,
}

#[derive(Clone, Debug)]
pub struct AgentGraph {
    generation: u64,
    config: AgentGraphConfig,
    nodes: Vec<AgentNode>,
    edges: Vec<AgentEdge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouteDecision {
    pub speaker: AgentRole,
    pub speaker_id: NodeId,
    pub mask: LayerExecutionMask,
    pub layer_route_id: Option<u64>,
    pub compiler: bool,
    pub lrc_hit: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyStatus {
    Pass,
    Fail,
    Abstain,
}

impl VerifyStatus {
    pub fn passed(self) -> bool {
        matches!(self, Self::Pass)
    }

    pub fn should_abstain(self) -> bool {
        matches!(self, Self::Abstain)
    }
}

impl Default for AgentGraph {
    fn default() -> Self {
        Self::new(AgentGraphConfig::default())
    }
}

impl AgentGraph {
    pub fn new(config: AgentGraphConfig) -> Self {
        Self {
            generation: 0,
            config,
            nodes: initial_nodes(),
            edges: initial_edges(),
        }
    }

    pub fn load_or_new(path: impl AsRef<Path>, config: AgentGraphConfig) -> Self {
        let path = path.as_ref();
        let Ok(body) = std::fs::read(path) else {
            return Self::new(config);
        };
        let Ok(state) = serde_json::from_slice::<AgentGraphState>(&body) else {
            return Self::new(config);
        };
        if state.version != STATE_VERSION || state.nodes.len() != 6 {
            return Self::new(config);
        }
        Self {
            generation: state.generation,
            config: state.config,
            nodes: state.nodes,
            edges: state.edges,
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let state = AgentGraphState {
            version: STATE_VERSION,
            generation: self.generation,
            config: self.config.clone(),
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
        };
        let body = serde_json::to_vec_pretty(&state).map_err(|error| error.to_string())?;
        atomic_write(path.as_ref(), &body)
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn nodes(&self) -> &[AgentNode] {
        &self.nodes
    }

    pub fn edges(&self) -> &[AgentEdge] {
        &self.edges
    }

    pub fn node(&self, id: NodeId) -> Option<&AgentNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    fn node_mut(&mut self, id: NodeId) -> Option<&mut AgentNode> {
        self.nodes.iter_mut().find(|node| node.id == id)
    }

    pub fn post(&mut self, message: AgentMessage) {
        if let Some(node) = self.node_mut(message.to) {
            node.stats.calls = node.stats.calls.saturating_add(1);
            node.mailbox.push_back(message);
        }
    }

    pub fn drain(&mut self, id: NodeId) -> Vec<AgentMessage> {
        self.node_mut(id)
            .map(|node| node.mailbox.drain(..).collect())
            .unwrap_or_default()
    }

    /// Router sin LLM: lookup LRC + grafo. Costo conceptual < 1 ms.
    pub fn plan_turn(
        &mut self,
        fingerprint: &ActivationFingerprint,
        user_text: &str,
        lrc: &LayerRouteCache,
        layer_count: usize,
    ) -> RouteDecision {
        let dense = LayerExecutionMask::all(layer_count.max(1));
        let lrc_hit_now = lrc
            .lookup_confident(fingerprint)
            .is_some_and(|route| route.mask.layer_count() == layer_count);
        let recipe = looks_like_recipe(user_text);
        let claim_hits = self.claim_speakers(fingerprint);

        let (speaker_id, speaker) = if lrc_hit_now {
            (FAST_TALKER_ID, AgentRole::FastTalker)
        } else if recipe {
            (COMPILER_ID, AgentRole::Compiler)
        } else if claim_hits.len() == 1 {
            let id = claim_hits[0];
            (id, self.role_of(id))
        } else if claim_hits.len() > 1 {
            let id = self.epsilon_greedy(&claim_hits);
            (id, self.role_of(id))
        } else {
            (DENSE_TALKER_ID, AgentRole::DenseTalker)
        };

        let (mask, layer_route_id, lrc_hit) = if speaker == AgentRole::FastTalker {
            if let Some(route) = lrc.lookup_confident(fingerprint) {
                (route.mask.clone(), Some(route.id), true)
            } else {
                (dense.clone(), None, false)
            }
        } else {
            (dense, None, false)
        };

        self.post(AgentMessage {
            from: ROUTER_ID,
            to: speaker_id,
            kind: MessageKind::Query,
            tokens: None,
            text: Some(user_text.to_string()),
            route: layer_route_id,
        });
        self.post(AgentMessage {
            from: ROUTER_ID,
            to: MEMORY_ID,
            kind: MessageKind::Query,
            tokens: None,
            text: None,
            route: layer_route_id,
        });
        self.drain(MEMORY_ID);

        RouteDecision {
            speaker,
            speaker_id,
            mask,
            layer_route_id,
            compiler: speaker == AgentRole::Compiler,
            lrc_hit,
        }
    }

    pub fn verify_reply(
        &mut self,
        speaker_id: NodeId,
        text: &str,
        compiler_mode: bool,
    ) -> VerifyStatus {
        let status = verify_text(text, compiler_mode);
        self.post(AgentMessage {
            from: speaker_id,
            to: VERIFIER_ID,
            kind: if status.passed() {
                MessageKind::Reply
            } else {
                MessageKind::Reject
            },
            tokens: None,
            text: Some(text.to_string()),
            route: None,
        });
        let _ = self.drain(VERIFIER_ID);
        status
    }

    pub fn observe_turn(
        &mut self,
        speaker_id: NodeId,
        verifier_passed: bool,
        latency_ms: f32,
        dense_fallback: bool,
    ) {
        self.generation = self.generation.saturating_add(1);
        let quality = f32::from(verifier_passed);
        let reward = -(latency_ms / 1_000.0).clamp(0.0, 1.0) + self.config.lambda_quality * quality
            - self.config.mu_fallback * f32::from(dense_fallback);
        self.update_edge(ROUTER_ID, speaker_id, reward, latency_ms, verifier_passed);
        self.update_verifier_edge(speaker_id, verifier_passed, latency_ms);
        if let Some(node) = self.node_mut(speaker_id) {
            if verifier_passed {
                node.stats.accepts = node.stats.accepts.saturating_add(1);
            } else {
                node.stats.rejects = node.stats.rejects.saturating_add(1);
            }
            let calls = (node.stats.accepts + node.stats.rejects).max(1) as f32;
            node.stats.mean_ms = (node.stats.mean_ms * (calls - 1.0) + latency_ms.max(0.0)) / calls;
        }
    }

    fn role_of(&self, id: NodeId) -> AgentRole {
        self.node(id)
            .map(|node| node.role)
            .unwrap_or(AgentRole::DenseTalker)
    }

    fn claim_speakers(&self, fingerprint: &ActivationFingerprint) -> Vec<NodeId> {
        let mut hits = self
            .nodes
            .iter()
            .filter(|node| node.role.is_speaker())
            .filter_map(|node| {
                let overlap = fingerprint_overlap(fingerprint, &node.claim);
                (overlap + 1.0e-6 >= self.config.min_claim_overlap).then_some((node.id, overlap))
            })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0 .0.cmp(&right.0 .0)));
        hits.into_iter().map(|(id, _)| id).collect()
    }

    fn epsilon_greedy(&self, candidates: &[NodeId]) -> NodeId {
        if candidates.is_empty() {
            return DENSE_TALKER_ID;
        }
        let sample = unit_from_u64(splitmix64(
            self.config.seed ^ self.generation.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        ));
        if sample < self.config.epsilon {
            let index = (splitmix64(self.config.seed ^ self.generation.wrapping_add(1)) as usize)
                % candidates.len();
            return candidates[index];
        }
        candidates
            .iter()
            .copied()
            .max_by(|left, right| {
                self.edge_weight(ROUTER_ID, *left)
                    .total_cmp(&self.edge_weight(ROUTER_ID, *right))
                    .then(left.0.cmp(&right.0))
            })
            .unwrap_or(DENSE_TALKER_ID)
    }

    fn edge_weight(&self, src: NodeId, dst: NodeId) -> f32 {
        self.edges
            .iter()
            .find(|edge| edge.src == src && edge.dst == dst)
            .map(|edge| edge.weight)
            .unwrap_or(0.0)
    }

    fn update_edge(
        &mut self,
        src: NodeId,
        dst: NodeId,
        reward: f32,
        latency_ms: f32,
        success: bool,
    ) {
        let alpha = self.config.alpha.clamp(0.0, 1.0);
        let generation = self.generation;
        if let Some(edge) = self
            .edges
            .iter_mut()
            .find(|edge| edge.src == src && edge.dst == dst)
        {
            edge.weight = (1.0 - alpha) * edge.weight + alpha * reward;
            if success {
                edge.successes = edge.successes.saturating_add(1);
            } else {
                edge.failures = edge.failures.saturating_add(1);
            }
            let n = (edge.successes + edge.failures).max(1) as f32;
            edge.mean_latency_ms = (edge.mean_latency_ms * (n - 1.0) + latency_ms.max(0.0)) / n;
            edge.last_used = generation;
        }
    }

    fn update_verifier_edge(&mut self, speaker_id: NodeId, passed: bool, latency_ms: f32) {
        let reward = if passed { 1.0 } else { 0.0 };
        self.update_edge(speaker_id, VERIFIER_ID, reward, latency_ms, passed);
    }
}

fn initial_nodes() -> Vec<AgentNode> {
    vec![
        node(ROUTER_ID, AgentRole::Router, 0, 0),
        node(FAST_TALKER_ID, AgentRole::FastTalker, 256, 22),
        node(DENSE_TALKER_ID, AgentRole::DenseTalker, 256, 26),
        node(VERIFIER_ID, AgentRole::Verifier, 0, 0),
        AgentNode {
            claim: compiler_claim_sketch(),
            ..node(COMPILER_ID, AgentRole::Compiler, 768, 26)
        },
        node(MEMORY_ID, AgentRole::Memory, 0, 0),
    ]
}

fn node(
    id: NodeId,
    role: AgentRole,
    max_new_tokens: usize,
    max_executed_layers: usize,
) -> AgentNode {
    AgentNode {
        id,
        role,
        route_pref: None,
        claim: empty_fingerprint(),
        budget: AgentBudget {
            max_new_tokens,
            max_executed_layers,
            timeout_ms: 120_000,
        },
        stats: AgentStats::default(),
        mailbox: VecDeque::new(),
    }
}

fn initial_edges() -> Vec<AgentEdge> {
    vec![
        edge(ROUTER_ID, FAST_TALKER_ID, 1.0),
        edge(ROUTER_ID, DENSE_TALKER_ID, 0.80),
        edge(ROUTER_ID, COMPILER_ID, 0.70),
        edge(ROUTER_ID, MEMORY_ID, 1.0),
        edge(FAST_TALKER_ID, VERIFIER_ID, 0.50),
        edge(DENSE_TALKER_ID, VERIFIER_ID, 0.50),
        edge(COMPILER_ID, VERIFIER_ID, 0.50),
    ]
}

fn edge(src: NodeId, dst: NodeId, weight: f32) -> AgentEdge {
    AgentEdge {
        src,
        dst,
        weight,
        successes: 0,
        failures: 0,
        mean_latency_ms: 0.0,
        last_used: 0,
    }
}

fn empty_fingerprint() -> ActivationFingerprint {
    ActivationFingerprint {
        entries: Vec::new(),
        confidence: 0.0,
        entropy: 0.0,
    }
}

pub fn compiler_claim_sketch() -> ActivationFingerprint {
    ActivationFingerprint {
        entries: vec![
            (u32::from_be_bytes(*b"QUBO"), 1.0),
            (u32::from_be_bytes(*b"{r?}"), 0.80),
        ],
        confidence: 0.85,
        entropy: 0.40,
    }
}

pub fn looks_like_recipe(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let has_object = text.contains('{') && text.contains('}');
    has_object
        || lower.contains("qubo")
        || lower.contains("operator=")
        || lower.contains("variables=")
        || lower.contains("unary=")
        || lower.contains("pairs=")
        || lower.contains("ising")
}

pub fn verify_text(text: &str, compiler_mode: bool) -> VerifyStatus {
    if !latin_and_spanish_ok(text) {
        return VerifyStatus::Fail;
    }
    if is_llm_disclaimer(text) {
        return VerifyStatus::Fail;
    }
    if compiler_mode {
        return if parse_operator_recipe(text).is_ok() {
            VerifyStatus::Pass
        } else {
            VerifyStatus::Abstain
        };
    }
    if text.trim().is_empty() {
        return VerifyStatus::Fail;
    }
    VerifyStatus::Pass
}

fn is_llm_disclaimer(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("i'm a large language model")
        || lower.contains("i am a large language model")
        || lower.contains("soy un modelo de lenguaje")
}

fn latin_and_spanish_ok(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let latin = trimmed
        .chars()
        .filter(|character| character.is_ascii_alphabetic())
        .count();
    let letters = trimmed
        .chars()
        .filter(|character| character.is_alphabetic())
        .count();
    if letters > 0 && latin * 4 < letters {
        return false;
    }
    if GEMMA2_FORCED_LANGUAGE != "es" {
        return true;
    }
    let padded = format!(" {} ", trimmed.to_lowercase());
    let spanish = SPANISH_MARKERS
        .iter()
        .filter(|marker| padded.contains(*marker))
        .count();
    let english = ENGLISH_MARKERS
        .iter()
        .filter(|marker| padded.contains(*marker))
        .count();
    if english >= 2 && english > spanish {
        return false;
    }
    if spanish == 0 && english > 0 {
        return false;
    }
    true
}

const SPANISH_MARKERS: &[&str] = &[
    " el ",
    " la ",
    " de ",
    " que ",
    " y ",
    " en ",
    " un ",
    " una ",
    " es ",
    " se ",
    " no ",
    " por ",
    " con ",
    " para ",
    " del ",
    " los ",
    " las ",
    " como ",
    " mas ",
    " más ",
    " pero ",
    " este ",
    " esta ",
    " hay ",
    " son ",
    " esta ",
    " muy ",
    " hola ",
    " gracias ",
    " aqui ",
    " aquí ",
    " tambien ",
    " también ",
    " porque ",
    " si ",
    " sí ",
];
const ENGLISH_MARKERS: &[&str] = &[
    " the ", " and ", " is ", " of ", " to ", " you ", " that ", " it ", " for ", " are ",
    " with ", " this ", " have ", " not ", " i'm ", " i am ", " your ", " we ", " they ",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer_route_cache::{fingerprint_wake, LayerRouteCache, LayerRouteCacheConfig};

    fn sparse(layers: usize) -> LayerExecutionMask {
        let mut enabled = vec![true; layers];
        if layers > 2 {
            enabled[layers / 2] = false;
        }
        LayerExecutionMask::from_enabled(enabled)
    }

    #[test]
    fn graph_has_exactly_six_nodes() {
        let graph = AgentGraph::default();
        assert_eq!(graph.nodes().len(), 6);
        let roles = graph
            .nodes()
            .iter()
            .map(|node| node.role)
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            vec![
                AgentRole::Router,
                AgentRole::FastTalker,
                AgentRole::DenseTalker,
                AgentRole::Verifier,
                AgentRole::Compiler,
                AgentRole::Memory,
            ]
        );
    }

    #[test]
    fn router_picks_fast_talker_on_confident_lrc() {
        let mut lrc = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let tokens = [1u32, 2, 3, 4, 5, 6];
        let fp = fingerprint_wake(&tokens);
        assert!(lrc.promote(fp.clone(), sparse(8), 0.04, 1.0, 1));
        let mut graph = AgentGraph::default();
        let decision = graph.plan_turn(&fp, "hola, como estas", &lrc, 8);
        assert_eq!(decision.speaker, AgentRole::FastTalker);
        assert!(decision.lrc_hit);
        assert_eq!(decision.mask.executed_count(), 7);
        assert!(!decision.compiler);
    }

    #[test]
    fn router_picks_compiler_on_recipe_text() {
        let lrc = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let mut graph = AgentGraph::default();
        let fp = fingerprint_wake(&[90, 91, 92]);
        let decision = graph.plan_turn(
            &fp,
            "operator=qubo name=demo variables=a:binary,b:binary unary=a:-1:0",
            &lrc,
            8,
        );
        assert_eq!(decision.speaker, AgentRole::Compiler);
        assert!(decision.compiler);
        assert_eq!(decision.mask.executed_count(), 8);
    }

    #[test]
    fn router_falls_back_to_dense_without_lrc_or_recipe() {
        let lrc = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let mut graph = AgentGraph::default();
        let fp = fingerprint_wake(&[11, 12, 13, 14]);
        let decision = graph.plan_turn(&fp, "explica el residual de una capa", &lrc, 8);
        assert_eq!(decision.speaker, AgentRole::DenseTalker);
        assert!(!decision.lrc_hit);
    }

    #[test]
    fn router_matches_compiler_claim_overlap() {
        let lrc = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let mut graph = AgentGraph::default();
        let fp = compiler_claim_sketch();
        let decision = graph.plan_turn(&fp, "planifica el circuito", &lrc, 8);
        assert_eq!(decision.speaker, AgentRole::Compiler);
    }

    #[test]
    fn epsilon_greedy_tie_picks_heavier_edge_when_epsilon_is_zero() {
        let mut graph = AgentGraph::new(AgentGraphConfig {
            epsilon: 0.0,
            ..AgentGraphConfig::default()
        });
        let sketch = fingerprint_wake(&[7, 8, 9, 10, 11]);
        if let Some(node) = graph.node_mut(FAST_TALKER_ID) {
            node.claim = sketch.clone();
        }
        if let Some(node) = graph.node_mut(DENSE_TALKER_ID) {
            node.claim = sketch.clone();
        }
        if let Some(edge) = graph
            .edges
            .iter_mut()
            .find(|edge| edge.src == ROUTER_ID && edge.dst == FAST_TALKER_ID)
        {
            edge.weight = 0.10;
        }
        if let Some(edge) = graph
            .edges
            .iter_mut()
            .find(|edge| edge.src == ROUTER_ID && edge.dst == DENSE_TALKER_ID)
        {
            edge.weight = 0.90;
        }
        let lrc = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let decision = graph.plan_turn(&sketch, "pregunta empatada", &lrc, 8);
        assert_eq!(decision.speaker, AgentRole::DenseTalker);
    }

    #[test]
    fn verifier_accepts_spanish_and_rejects_english_and_llm_disclaimer() {
        assert_eq!(
            verify_text("Hola. El residual de la capa se conserva.", false),
            VerifyStatus::Pass
        );
        assert_eq!(
            verify_text("Sure, I can help you with that problem.", false),
            VerifyStatus::Fail
        );
        assert_eq!(
            verify_text("Soy un modelo de lenguaje y no tengo opiniones.", false),
            VerifyStatus::Fail
        );
        assert_eq!(
            verify_text("I'm a large language model trained by Google.", false),
            VerifyStatus::Fail
        );
        assert_eq!(verify_text("", false), VerifyStatus::Fail);
    }

    #[test]
    fn verifier_requires_parseable_recipe_in_compiler_mode() {
        let recipe = concat!(
            "operator=qubo\n",
            "name=ejemplo_bits\n",
            "variables=a:binary,b:binary\n",
            "unary=a:-1:0,b:-1:0\n",
            "pairs=a:b:2:0\n",
            "faces=\n",
            "flows=\n",
            "max_working_set=8192\n",
            "ridge=0.001\n",
            "END\n"
        );
        assert_eq!(verify_text(recipe, true), VerifyStatus::Pass);
        assert_eq!(
            verify_text("Hola. No se como formular el QUBO.", true),
            VerifyStatus::Abstain
        );
    }

    #[test]
    fn mailbox_is_synchronous_and_observe_updates_router_edge() {
        let mut graph = AgentGraph::default();
        let lrc = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let fp = fingerprint_wake(&[3, 1, 4]);
        let decision = graph.plan_turn(&fp, "hola", &lrc, 8);
        assert_eq!(decision.speaker, AgentRole::DenseTalker);
        let queued = graph.drain(DENSE_TALKER_ID);
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].kind, MessageKind::Query);
        graph.observe_turn(DENSE_TALKER_ID, true, 12.0, false);
        let edge = graph
            .edges()
            .iter()
            .find(|edge| edge.src == ROUTER_ID && edge.dst == DENSE_TALKER_ID)
            .unwrap();
        assert_eq!(edge.successes, 1);
        let verifier = graph
            .edges()
            .iter()
            .find(|edge| edge.src == DENSE_TALKER_ID && edge.dst == VERIFIER_ID)
            .unwrap();
        assert_eq!(verifier.successes, 1);
    }

    #[test]
    fn roundtrip_json_keeps_six_nodes() {
        let dir = std::env::temp_dir().join(format!(
            "agent-graph-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(AGENT_GRAPH_FILE);
        let mut graph = AgentGraph::default();
        graph.observe_turn(FAST_TALKER_ID, true, 8.0, false);
        graph.save(&path).unwrap();
        let loaded = AgentGraph::load_or_new(&path, AgentGraphConfig::default());
        assert_eq!(loaded.nodes().len(), 6);
        assert_eq!(loaded.generation(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore]
    fn ignored_gguf_router_does_not_require_a_forward() {
        let lrc = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let mut graph = AgentGraph::default();
        let fp = fingerprint_wake(&[1, 2, 3]);
        let decision = graph.plan_turn(&fp, "hola", &lrc, 26);
        assert!(decision.speaker.is_speaker());
        assert_ne!(decision.speaker, AgentRole::Router);
    }
}
