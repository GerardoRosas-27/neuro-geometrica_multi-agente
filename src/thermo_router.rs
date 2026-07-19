//! Router asociativo bidireccional entre activaciones Transformer y CDT-RQM-EPR.
//! El sustrato conserva rutas y dinámica; los payloads opacos viven en el vault.

use crate::native_thermo_rqm_epr::{NativeThermoRqmEprSubstrate, RealtimeUpdateConfig};
use crate::relational_field::ObserverId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

pub const ROUTER_OBSERVER: ObserverId = ObserverId(778_003);

#[derive(Clone, Debug)]
pub struct ActivationFingerprint {
    pub entries: Vec<(u32, f32)>,
    pub confidence: f32,
    pub entropy: f32,
}

#[derive(Clone, Debug)]
pub struct SparseAssembly {
    pub nodes: Vec<usize>,
    pub signature: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct RouteId(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct MemoryRef(pub u64);

#[derive(Clone, Debug)]
pub struct ContextInjection {
    pub route_id: RouteId,
    pub context_ids: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct RouterOutcome {
    pub assembly_nodes: Vec<usize>,
    pub route_id: Option<RouteId>,
    pub route_created: bool,
    pub recalled: Option<ContextInjection>,
    pub route_score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouterConfig {
    pub schema_version: u32,
    pub node_span: usize,
    pub query_width: usize,
    pub route_width: usize,
    pub max_routes: usize,
    pub min_similarity: f32,
    pub min_route_margin: f32,
    pub min_evidence: u32,
    pub min_confidence: f32,
    pub context_limit: usize,
    pub seed: u64,
}

impl RouterConfig {
    pub fn for_substrate(nodes: usize) -> Self {
        Self {
            schema_version: 8,
            node_span: nodes.max(32),
            query_width: 24,
            route_width: 2,
            max_routes: 512,
            min_similarity: 0.15,
            min_route_margin: 3.10,
            min_evidence: 3,
            min_confidence: 0.22,
            context_limit: 32,
            seed: 0xA550_C1A7_E5,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouteRecord {
    pub id: RouteId,
    pub assembly_nodes: Vec<usize>,
    pub feature_ids: Vec<u32>,
    pub route_nodes: Vec<usize>,
    pub memory_ref: MemoryRef,
    pub evidence: u32,
    pub successes: u32,
    pub failures: u32,
    pub confidence: f32,
    pub utility: f32,
    pub last_generation: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpaqueMemoryCapsule {
    pub id: MemoryRef,
    pub model_id: String,
    pub context_ids: Vec<u32>,
    pub created_generation: u64,
    pub uses: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseAssemblyAdapter {
    config: RouterConfig,
}

impl SparseAssemblyAdapter {
    pub fn new(config: RouterConfig) -> Self {
        Self { config }
    }

    pub fn encode(&self, fingerprint: &ActivationFingerprint) -> SparseAssembly {
        let route_reserve = (self.config.node_span / 4).max(self.config.route_width);
        let query_span = self.config.node_span.saturating_sub(route_reserve).max(1);
        let mut nodes = Vec::with_capacity(self.config.query_width);
        for &(token, _) in &fingerprint.entries {
            for projection in 0..2usize {
                let node = stable_hash(&(self.config.seed, "query", token, projection)) as usize
                    % query_span;
                if !nodes.contains(&node) {
                    nodes.push(node);
                    if nodes.len() >= self.config.query_width {
                        break;
                    }
                }
            }
            if nodes.len() >= self.config.query_width {
                break;
            }
        }
        nodes.sort_unstable();
        let signature = stable_hash(&(self.config.seed, &nodes));
        SparseAssembly { nodes, signature }
    }

    pub fn route_nodes(&self, route_id: RouteId) -> Vec<usize> {
        let route_reserve = (self.config.node_span / 4).max(self.config.route_width);
        let start = self.config.node_span.saturating_sub(route_reserve);
        (0..self.config.route_width)
            .map(|projection| {
                start
                    + stable_hash(&(self.config.seed, "route", route_id.0, projection)) as usize
                        % route_reserve
            })
            .collect()
    }

    pub fn similarity(&self, left: &[usize], right: &[usize]) -> f32 {
        if left.is_empty() || right.is_empty() {
            return 0.0;
        }
        let right = right.iter().copied().collect::<HashSet<_>>();
        let overlap = left.iter().filter(|node| right.contains(node)).count();
        overlap as f32 / left.len().max(right.len()) as f32
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RouteRegistry {
    routes: Vec<RouteRecord>,
    next_route: u64,
}

impl RouteRegistry {
    pub fn routes(&self) -> &[RouteRecord] {
        &self.routes
    }

    fn exact_assembly(&self, nodes: &[usize]) -> Option<RouteId> {
        self.routes
            .iter()
            .find(|route| route.assembly_nodes == nodes)
            .map(|route| route.id)
    }

    pub fn get(&self, id: RouteId) -> Option<&RouteRecord> {
        self.routes.iter().find(|route| route.id == id)
    }

    pub fn get_mut(&mut self, id: RouteId) -> Option<&mut RouteRecord> {
        self.routes.iter_mut().find(|route| route.id == id)
    }

    fn create(
        &mut self,
        assembly: &SparseAssembly,
        fingerprint: &ActivationFingerprint,
        route_nodes: Vec<usize>,
        memory_ref: MemoryRef,
        generation: u64,
    ) -> RouteId {
        self.next_route = self.next_route.wrapping_add(1).max(1);
        let id = RouteId(self.next_route);
        self.routes.push(RouteRecord {
            id,
            assembly_nodes: assembly.nodes.clone(),
            feature_ids: fingerprint.entries.iter().map(|(id, _)| *id).collect(),
            route_nodes,
            memory_ref,
            evidence: 1,
            successes: 0,
            failures: 0,
            confidence: 0.20,
            utility: 0.20,
            last_generation: generation,
        });
        id
    }

    fn prune_low_utility(&mut self, max_routes: usize) -> Vec<MemoryRef> {
        if self.routes.len() <= max_routes {
            return Vec::new();
        }
        self.routes.sort_by(|left, right| {
            right
                .utility
                .total_cmp(&left.utility)
                .then(right.evidence.cmp(&left.evidence))
        });
        self.routes
            .drain(max_routes..)
            .map(|route| route.memory_ref)
            .collect()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MemoryVaultAdapter {
    capsules: HashMap<MemoryRef, OpaqueMemoryCapsule>,
    next_memory: u64,
}

impl MemoryVaultAdapter {
    pub fn store(
        &mut self,
        model_id: &str,
        context: &[u32],
        generation: u64,
        limit: usize,
    ) -> MemoryRef {
        self.next_memory = self.next_memory.wrapping_add(1).max(1);
        let id = MemoryRef(self.next_memory);
        let tail = &context[context.len().saturating_sub(limit)..];
        self.capsules.insert(
            id,
            OpaqueMemoryCapsule {
                id,
                model_id: model_id.to_string(),
                context_ids: tail.to_vec(),
                created_generation: generation,
                uses: 0,
            },
        );
        id
    }

    pub fn recall(&mut self, id: MemoryRef, model_id: &str) -> Option<Vec<u32>> {
        let capsule = self.capsules.get_mut(&id)?;
        if capsule.model_id != model_id {
            return None;
        }
        capsule.uses = capsule.uses.saturating_add(1);
        Some(capsule.context_ids.clone())
    }

    fn replace_context(&mut self, id: MemoryRef, context: &[u32], limit: usize) {
        if let Some(capsule) = self.capsules.get_mut(&id) {
            let tail = &context[context.len().saturating_sub(limit)..];
            capsule.context_ids = tail.to_vec();
        }
    }

    fn remove(&mut self, id: MemoryRef) {
        self.capsules.remove(&id);
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TransformerInjectionAdapter;

impl TransformerInjectionAdapter {
    pub fn prepare(
        &self,
        route: &RouteRecord,
        vault: &mut MemoryVaultAdapter,
        model_id: &str,
    ) -> Option<ContextInjection> {
        Some(ContextInjection {
            route_id: route.id,
            context_ids: vault.recall(route.memory_ref, model_id)?,
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FeedbackAdapter;

impl FeedbackAdapter {
    pub fn quality(confidence: f32, entropy: f32) -> f32 {
        (0.30 * confidence + 0.70 * (1.0 - entropy)).clamp(0.0, 1.0)
    }

    fn apply(route: &mut RouteRecord, quality: f32, generation: u64) -> bool {
        route.last_generation = generation;
        if quality >= 0.25 {
            route.successes = route.successes.saturating_add(1);
            route.evidence = route.evidence.saturating_add(1);
            route.confidence = (route.confidence * 0.85 + quality * 0.15).min(1.0);
            route.utility = (route.utility * 0.92 + quality * 0.08).min(1.0);
            true
        } else {
            route.failures = route.failures.saturating_add(1);
            route.confidence *= 0.92;
            route.utility *= 0.90;
            false
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermoAssociativeRouter {
    pub config: RouterConfig,
    pub model_id: String,
    pub assembly_adapter: SparseAssemblyAdapter,
    pub registry: RouteRegistry,
    pub vault: MemoryVaultAdapter,
    pub injection_adapter: TransformerInjectionAdapter,
    pub feedback_adapter: FeedbackAdapter,
    pub observations: u64,
    pub recalls: u64,
    pub creations: u64,
    pub abstentions: u64,
    pub last_recall_score: f32,
    pub last_recall_margin: f32,
}

impl ThermoAssociativeRouter {
    pub fn new(model_id: impl Into<String>, config: RouterConfig) -> Self {
        Self {
            assembly_adapter: SparseAssemblyAdapter::new(config.clone()),
            config,
            model_id: model_id.into(),
            registry: RouteRegistry::default(),
            vault: MemoryVaultAdapter::default(),
            injection_adapter: TransformerInjectionAdapter,
            feedback_adapter: FeedbackAdapter,
            observations: 0,
            recalls: 0,
            creations: 0,
            abstentions: 0,
            last_recall_score: 0.0,
            last_recall_margin: 0.0,
        }
    }

    /// Crea una ruta a partir de una memoria ya verificada. No entrena pesos:
    /// vincula una firma Transformer con una referencia opaca en RQM/EPR.
    pub fn bind_verified(
        &mut self,
        substrate: &mut NativeThermoRqmEprSubstrate,
        fingerprint: &ActivationFingerprint,
        payload: &[u32],
        generation: u64,
    ) -> RouteId {
        let assembly = self.assembly_adapter.encode(fingerprint);
        let route_id = if let Some(route_id) = self.registry.exact_assembly(&assembly.nodes) {
            if let Some(route) = self.registry.get_mut(route_id) {
                route.evidence = route.evidence.max(self.config.min_evidence);
                route.successes = route.successes.saturating_add(1);
                route.confidence = 0.95;
                route.utility = 0.95;
                route.last_generation = generation;
                route.feature_ids = fingerprint.entries.iter().map(|(id, _)| *id).collect();
                self.vault
                    .replace_context(route.memory_ref, payload, self.config.context_limit);
            }
            route_id
        } else {
            let memory_ref = self.vault.store(
                &self.model_id,
                payload,
                generation,
                self.config.context_limit,
            );
            let provisional = RouteId(self.registry.next_route.wrapping_add(1).max(1));
            let route_nodes = self.assembly_adapter.route_nodes(provisional);
            let route_id =
                self.registry
                    .create(&assembly, fingerprint, route_nodes, memory_ref, generation);
            if let Some(route) = self.registry.get_mut(route_id) {
                route.evidence = self.config.min_evidence;
                route.successes = 1;
                route.confidence = 0.95;
                route.utility = 0.95;
            }
            self.creations = self.creations.saturating_add(1);
            route_id
        };
        if let Some(route) = self.registry.get(route_id) {
            substrate.train_observed_transition_realtime(
                ROUTER_OBSERVER,
                0.0,
                &assembly.nodes,
                &route.route_nodes,
                1.0,
                RealtimeUpdateConfig {
                    thermal_microsteps: 0,
                    ..RealtimeUpdateConfig::default()
                },
            );
        }
        route_id
    }

    /// Recuperación pura: no crea rutas ni altera el vault salvo el contador de uso.
    pub fn recall(
        &mut self,
        substrate: &mut NativeThermoRqmEprSubstrate,
        fingerprint: &ActivationFingerprint,
    ) -> Option<ContextInjection> {
        let assembly = self.assembly_adapter.encode(fingerprint);
        let report = substrate.query(ROUTER_OBSERVER, 0.0, &assembly.nodes);
        let candidate_scores = report
            .candidates
            .iter()
            .map(|candidate| (candidate.agent, candidate.score))
            .collect::<HashMap<_, _>>();
        let mut ranked = self
            .registry
            .routes()
            .iter()
            .filter(|route| {
                route.evidence >= self.config.min_evidence
                    && route.confidence >= self.config.min_confidence
            })
            .map(|route| {
                let rqm_score = route
                    .route_nodes
                    .iter()
                    .filter_map(|node| candidate_scores.get(node))
                    .sum::<f32>()
                    * route.utility;
                let similarity = self
                    .assembly_adapter
                    .similarity(&assembly.nodes, &route.assembly_nodes);
                let features = feature_similarity(fingerprint, route);
                let normalized_rqm = rqm_score / (1.0 + rqm_score.abs());
                (
                    route.id,
                    features,
                    similarity,
                    features * 20.0 + similarity * 3.0 + normalized_rqm,
                )
            })
            .filter(|(_, features, similarity, _)| {
                *features >= 0.05 || *similarity >= self.config.min_similarity
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.3.total_cmp(&left.3));
        let best = ranked.first()?;
        let second_score = ranked.get(1).map(|item| item.3).unwrap_or(0.0);
        self.last_recall_score = best.3;
        self.last_recall_margin = best.3 - second_score;
        if self.last_recall_margin < self.config.min_route_margin {
            self.abstentions = self.abstentions.saturating_add(1);
            return None;
        }
        let route_id = best.0;
        let route = self.registry.get(route_id).cloned()?;
        let injection = self
            .injection_adapter
            .prepare(&route, &mut self.vault, &self.model_id)?;
        self.recalls = self.recalls.saturating_add(1);
        Some(injection)
    }

    pub fn process(
        &mut self,
        substrate: &mut NativeThermoRqmEprSubstrate,
        fingerprint: &ActivationFingerprint,
        context: &[u32],
        generation: u64,
        feedback_route: Option<RouteId>,
    ) -> RouterOutcome {
        self.observations = self.observations.saturating_add(1);
        let quality = FeedbackAdapter::quality(fingerprint.confidence, fingerprint.entropy);
        if let Some(route_id) = feedback_route {
            self.apply_feedback(substrate, route_id, quality, generation);
        }

        let assembly = self.assembly_adapter.encode(fingerprint);
        for &node in &assembly.nodes {
            if node < substrate.thermal.node_count() {
                substrate.thermal.inject_local_node(
                    node,
                    0.25 + quality * 0.55,
                    assembly.signature as f32 * 1.0e-6,
                    0.20 + quality * 0.45,
                );
            }
        }

        let report = substrate.query(ROUTER_OBSERVER, 0.0, &assembly.nodes);
        let candidate_scores = report
            .candidates
            .iter()
            .map(|candidate| (candidate.agent, candidate.score))
            .collect::<HashMap<_, _>>();
        let mut ranked = self
            .registry
            .routes()
            .iter()
            .map(|route| {
                let rqm_score = route
                    .route_nodes
                    .iter()
                    .filter_map(|node| candidate_scores.get(node))
                    .sum::<f32>()
                    * route.utility.max(0.05);
                let similarity = self
                    .assembly_adapter
                    .similarity(&assembly.nodes, &route.assembly_nodes);
                let features = feature_similarity(fingerprint, route);
                let normalized_rqm = rqm_score / (1.0 + rqm_score.abs());
                (
                    route.id,
                    features,
                    similarity,
                    features * 20.0 + similarity * 3.0 + normalized_rqm,
                )
            })
            .filter(|(_, features, similarity, _)| {
                *features >= 0.05 || *similarity >= self.config.min_similarity
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.3.total_cmp(&left.3));
        let route_choice = ranked.first().map(|best| {
            let second = ranked.get(1).map(|item| item.3).unwrap_or(0.0);
            (best.0, best.3, best.3 - second)
        });
        let route_id = route_choice.map(|(id, _, _)| id);

        let mut created = false;
        let route_id = if let Some(route_id) = route_id {
            if let Some(route) = self.registry.get_mut(route_id) {
                route.evidence = route.evidence.saturating_add(1);
                route.last_generation = generation;
                route.confidence = route.confidence * 0.90 + quality * 0.10;
                route.utility = route.utility * 0.95 + quality * 0.05;
            }
            route_id
        } else {
            let memory_ref = self.vault.store(
                &self.model_id,
                context,
                generation,
                self.config.context_limit,
            );
            let provisional = RouteId(self.registry.next_route.wrapping_add(1).max(1));
            let route_nodes = self.assembly_adapter.route_nodes(provisional);
            let route_id =
                self.registry
                    .create(&assembly, fingerprint, route_nodes, memory_ref, generation);
            self.creations = self.creations.saturating_add(1);
            created = true;
            route_id
        };

        let (route_nodes, mature) = self
            .registry
            .get(route_id)
            .map(|route| {
                (
                    route.route_nodes.clone(),
                    route.evidence >= self.config.min_evidence
                        && route.confidence >= self.config.min_confidence,
                )
            })
            .unwrap_or_default();
        substrate.train_observed_transition_realtime(
            ROUTER_OBSERVER,
            0.0,
            &assembly.nodes,
            &route_nodes,
            quality.max(0.10),
            RealtimeUpdateConfig {
                max_relation_updates: self.config.query_width * self.config.route_width,
                max_epr_observations: self.config.route_width * 2,
                thermal_microsteps: 0,
                ..RealtimeUpdateConfig::default()
            },
        );

        let (route_score, route_margin) = route_choice
            .map(|(_, score, margin)| (score, margin))
            .unwrap_or((0.0, f32::INFINITY));
        let recalled = if mature && route_margin >= self.config.min_route_margin {
            let route = self.registry.get(route_id).cloned();
            route.and_then(|route| {
                self.injection_adapter
                    .prepare(&route, &mut self.vault, &self.model_id)
            })
        } else {
            if mature {
                self.abstentions = self.abstentions.saturating_add(1);
            }
            None
        };
        if recalled.is_some() {
            self.recalls = self.recalls.saturating_add(1);
        }

        for memory_ref in self.registry.prune_low_utility(self.config.max_routes) {
            self.vault.remove(memory_ref);
        }
        RouterOutcome {
            assembly_nodes: assembly.nodes,
            route_id: Some(route_id),
            route_created: created,
            recalled,
            route_score,
        }
    }

    fn apply_feedback(
        &mut self,
        substrate: &mut NativeThermoRqmEprSubstrate,
        route_id: RouteId,
        quality: f32,
        generation: u64,
    ) {
        let Some(route) = self.registry.get_mut(route_id) else {
            return;
        };
        let success = FeedbackAdapter::apply(route, quality, generation);
        let assembly = route.assembly_nodes.clone();
        let targets = route.route_nodes.clone();
        if success {
            substrate.train_observed_transition_realtime(
                ROUTER_OBSERVER,
                0.0,
                &assembly,
                &targets,
                quality,
                RealtimeUpdateConfig {
                    thermal_microsteps: 0,
                    ..RealtimeUpdateConfig::default()
                },
            );
        } else {
            for &source in &assembly {
                for &target in &targets {
                    substrate.attenuate_relation(ROUTER_OBSERVER, source, target, 0.35);
                }
            }
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let temporary = path.with_extension("tmp");
        let body = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        fs::write(&temporary, body).map_err(|error| error.to_string())?;
        if path.exists() {
            fs::remove_file(path).map_err(|error| error.to_string())?;
        }
        fs::rename(temporary, path).map_err(|error| error.to_string())
    }

    pub fn load_or_new(
        path: impl AsRef<Path>,
        model_id: impl Into<String>,
        config: RouterConfig,
    ) -> Self {
        let path = path.as_ref();
        let model_id = model_id.into();
        fs::read(path)
            .ok()
            .and_then(|body| serde_json::from_slice::<Self>(&body).ok())
            .filter(|router| {
                router.model_id == model_id && router.config.schema_version == config.schema_version
            })
            .unwrap_or_else(|| Self::new(model_id, config))
    }
}

#[derive(Clone, Debug)]
pub struct TransformerActivationAdapter {
    top_k: usize,
}

impl TransformerActivationAdapter {
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k: top_k.max(1),
        }
    }

    pub fn capture(&self, logits: &[f32]) -> ActivationFingerprint {
        if logits.is_empty() {
            return ActivationFingerprint {
                entries: Vec::new(),
                confidence: 0.0,
                entropy: 1.0,
            };
        }
        let mut ranked = logits.iter().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(left.1));
        ranked.truncate(self.top_k);
        let maximum = *ranked[0].1;
        let sum = logits
            .iter()
            .map(|logit| ((*logit - maximum) as f64).exp())
            .sum::<f64>()
            .max(f64::EPSILON);
        let entries = ranked
            .into_iter()
            .map(|(id, logit)| (id as u32, ((*logit - maximum) as f64).exp() as f32))
            .collect::<Vec<_>>();
        let confidence = 1.0 / sum as f32;
        let entropy = {
            let mut entropy = 0.0f64;
            for &logit in logits {
                let probability = ((logit - maximum) as f64).exp() / sum;
                if probability > 0.0 {
                    entropy -= probability * probability.ln();
                }
            }
            (entropy / (logits.len().max(2) as f64).ln()).clamp(0.0, 1.0) as f32
        };
        ActivationFingerprint {
            entries,
            confidence,
            entropy,
        }
    }

    /// Combina la distribución de salida con una huella opaca del contexto.
    /// El bit alto separa IDs contextuales de IDs de logits; el sustrato solo
    /// recibe sus proyecciones dispersas, no la secuencia lingüística.
    pub fn capture_with_context(
        &self,
        logits: &[f32],
        context_ids: &[u32],
    ) -> ActivationFingerprint {
        let base = self.capture(logits);
        let mut entries = Vec::with_capacity(8 + self.top_k);
        for &token in context_ids.iter().rev() {
            if token <= 2 {
                continue;
            }
            let contextual = token | 0x8000_0000;
            if entries.iter().any(|(id, _)| *id == contextual) {
                continue;
            }
            entries.push((contextual, 1.0));
            if entries.len() >= 8 {
                break;
            }
        }
        entries.extend(base.entries);
        ActivationFingerprint {
            entries,
            confidence: base.confidence,
            entropy: base.entropy,
        }
    }
}

fn stable_hash(value: &impl Hash) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn feature_similarity(fingerprint: &ActivationFingerprint, route: &RouteRecord) -> f32 {
    let query = fingerprint
        .entries
        .iter()
        .map(|(id, _)| *id)
        .collect::<HashSet<_>>();
    let stored = route.feature_ids.iter().copied().collect::<HashSet<_>>();
    let weight = |id: u32| if id & 0x8000_0000 != 0 { 3.0 } else { 1.0 };
    let overlap = query
        .intersection(&stored)
        .map(|id| weight(*id))
        .sum::<f32>();
    let query_weight = query.iter().map(|id| weight(*id)).sum::<f32>();
    let stored_weight = stored.iter().map(|id| weight(*id)).sum::<f32>();
    overlap / query_weight.min(stored_weight).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entanglement::EntanglementConfig;
    use crate::native_thermo_rqm_epr::NativeThermoRqmConfig;
    use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;

    #[test]
    fn route_matures_and_recalls_opaque_context() {
        let mut substrate = NativeThermoRqmEprSubstrate::new(
            NativeThermoCdtConfig {
                slices: 4,
                nodes_per_slice: 32,
                ..NativeThermoCdtConfig::default()
            },
            NativeThermoRqmConfig {
                thermal_steps_per_query: 0,
                ..NativeThermoRqmConfig::default()
            },
            EntanglementConfig::default(),
        );
        let mut router = ThermoAssociativeRouter::new(
            "test-model",
            RouterConfig::for_substrate(substrate.thermal.node_count()),
        );
        let fingerprint = ActivationFingerprint {
            entries: vec![(10, 1.0), (20, 0.8), (30, 0.5), (40, 0.3)],
            confidence: 0.8,
            entropy: 0.1,
        };
        let context = vec![1, 10, 20, 30];
        let mut feedback = None;
        let mut recalled = None;
        for generation in 1..=5 {
            let outcome =
                router.process(&mut substrate, &fingerprint, &context, generation, feedback);
            feedback = outcome.route_id;
            recalled = outcome.recalled;
        }
        assert_eq!(router.registry.routes().len(), 1);
        assert_eq!(recalled.unwrap().context_ids, context);
        assert!(substrate.relation_count_for_observer(ROUTER_OBSERVER) > 0);
    }

    #[test]
    fn activation_adapter_captures_top_logits() {
        let adapter = TransformerActivationAdapter::new(2);
        let fingerprint = adapter.capture(&[0.1, 2.0, 1.5, -1.0]);
        assert_eq!(fingerprint.entries.len(), 2);
        assert_eq!(fingerprint.entries[0].0, 1);
        assert_eq!(fingerprint.entries[1].0, 2);
    }

    #[test]
    fn contextual_fingerprint_prioritizes_opaque_context_ids() {
        let adapter = TransformerActivationAdapter::new(2);
        let fingerprint = adapter.capture_with_context(&[0.1, 2.0, 1.5], &[1, 42, 99]);
        assert_eq!(fingerprint.entries[0].0, 99 | 0x8000_0000);
        assert_eq!(fingerprint.entries[1].0, 42 | 0x8000_0000);
        assert_eq!(fingerprint.entries[2].0, 1);
    }

    #[test]
    fn verified_route_recalls_from_overlapping_paraphrase_fingerprint() {
        let mut substrate = NativeThermoRqmEprSubstrate::new(
            NativeThermoCdtConfig {
                slices: 4,
                nodes_per_slice: 32,
                ..NativeThermoCdtConfig::default()
            },
            NativeThermoRqmConfig {
                thermal_steps_per_query: 0,
                ..NativeThermoRqmConfig::default()
            },
            EntanglementConfig::default(),
        );
        let mut router = ThermoAssociativeRouter::new(
            "test-model",
            RouterConfig::for_substrate(substrate.thermal.node_count()),
        );
        let learned = ActivationFingerprint {
            entries: vec![(10, 1.0), (20, 0.8), (30, 0.5), (40, 0.3)],
            confidence: 0.8,
            entropy: 0.1,
        };
        let paraphrase = ActivationFingerprint {
            entries: vec![(10, 0.9), (20, 0.7), (50, 0.5), (60, 0.4)],
            confidence: 0.7,
            entropy: 0.2,
        };
        router.bind_verified(&mut substrate, &learned, &[101, 102, 103], 1);
        let recalled = router.recall(&mut substrate, &paraphrase).unwrap();
        assert_eq!(recalled.context_ids, vec![101, 102, 103]);
    }
}
