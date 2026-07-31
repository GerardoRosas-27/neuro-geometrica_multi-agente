//! Enrutamiento adaptativo y memoria termodinámica de dos velocidades para Gemma 2.

use crate::entanglement::EntanglementConfig;
use crate::matrix_free_cognitive_substrate::LatentConceptId;
use crate::native_checkpoint::{atomic_write, save_native_checkpoint_transactional};
use crate::native_gemma2::{Gemma2ForwardTrace, LayerExecutionMask};
use crate::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::native_thermodynamic_engine::load_native_checkpoint;
use crate::relational_field::ObserverId;
use crate::thermo_router::{
    ActivationFingerprint, RouteId, RouterConfig, ThermoAssociativeRouter, ROUTER_OBSERVER,
};
use crate::unified_spin_cognitive_engine::{
    UnifiedSpinCognitiveConfig, UnifiedSpinCognitiveEngine,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

const STATE_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveGemma2Config {
    pub buffer_capacity: usize,
    pub min_verified_quality: f32,
    pub max_skip_fraction: f32,
    pub minimum_executed_layers: usize,
    pub revalidate_every: u64,
    pub max_routes: usize,
    pub sleep_decay: f32,
    pub protected_utility: f32,
    pub relation_budget: usize,
}

impl Default for AdaptiveGemma2Config {
    fn default() -> Self {
        Self {
            buffer_capacity: 24,
            min_verified_quality: 0.92,
            max_skip_fraction: 0.15,
            minimum_executed_layers: 8,
            revalidate_every: 16,
            max_routes: 2_048,
            sleep_decay: 0.995,
            protected_utility: 0.85,
            relation_budget: 16_384,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveExperience {
    pub context: ActivationFingerprint,
    pub activations: ActivationFingerprint,
    pub mask: LayerExecutionMask,
    pub memory_tokens: Vec<u32>,
    pub quality: f32,
    pub route_id: Option<RouteId>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FastWorkingMemoryBuffer {
    entries: Vec<AdaptiveExperience>,
    capacity: usize,
}

impl FastWorkingMemoryBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
        }
    }

    pub fn push(&mut self, experience: AdaptiveExperience) -> bool {
        self.entries.push(experience);
        self.is_full()
    }

    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct AdaptivePersistentState {
    version: u32,
    generation: u64,
    verified_routes: u64,
    fallbacks: u64,
    revalidations: u64,
    spin_gate_passes: u64,
    spin_gate_rejections: u64,
    buffer: FastWorkingMemoryBuffer,
}

#[derive(Clone, Debug)]
pub struct RecalledLayerRoute {
    pub route_id: RouteId,
    pub mask: LayerExecutionMask,
    pub memory_tokens: Vec<u32>,
    pub score: f32,
    pub margin: f32,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct SleepConsolidationReport {
    pub flushed: usize,
    pub pruned_routes: usize,
    pub pruned_relations: usize,
    pub remaining_routes: usize,
}

pub struct AdaptiveThermoMemory {
    pub config: AdaptiveGemma2Config,
    pub substrate: NativeThermoRqmEprSubstrate,
    pub router: ThermoAssociativeRouter,
    root: PathBuf,
    state: AdaptivePersistentState,
}

impl AdaptiveThermoMemory {
    pub fn load_or_new(
        root: impl AsRef<Path>,
        model_id: impl Into<String>,
        config: AdaptiveGemma2Config,
    ) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|error| error.to_string())?;
        let substrate_path = root.join("thermo-memory.cdt_native");
        let substrate =
            load_native_checkpoint(&substrate_path).unwrap_or_else(|_| fresh_adaptive_substrate());
        let mut router_config = RouterConfig::for_substrate(substrate.thermal.node_count());
        router_config.min_similarity = 0.35;
        router_config.min_route_margin = 0.05;
        router_config.max_routes = config.max_routes;
        router_config.context_limit = 160;
        let router =
            ThermoAssociativeRouter::load_or_new(root.join("routes.json"), model_id, router_config);
        let mut state = fs::read(root.join("adaptive-state.json"))
            .ok()
            .and_then(|body| serde_json::from_slice::<AdaptivePersistentState>(&body).ok())
            .filter(|state| state.version == STATE_VERSION)
            .unwrap_or_else(|| AdaptivePersistentState {
                version: STATE_VERSION,
                buffer: FastWorkingMemoryBuffer::new(config.buffer_capacity),
                ..AdaptivePersistentState::default()
            });
        state.buffer.capacity = config.buffer_capacity.max(1);
        Ok(Self {
            config,
            substrate,
            router,
            root,
            state,
        })
    }

    pub fn generation(&self) -> u64 {
        self.state.generation
    }

    pub fn should_revalidate(&self) -> bool {
        self.config.revalidate_every > 0
            && self
                .state
                .generation
                .is_multiple_of(self.config.revalidate_every)
    }

    pub fn context_fingerprint(&self, token_ids: &[u32]) -> ActivationFingerprint {
        context_fingerprint(token_ids)
    }

    pub fn activation_fingerprint(
        &self,
        context: &ActivationFingerprint,
        trace: &Gemma2ForwardTrace,
    ) -> ActivationFingerprint {
        activation_fingerprint(context, trace)
    }

    pub fn recall(
        &mut self,
        fingerprint: &ActivationFingerprint,
        layer_count: usize,
    ) -> Option<RecalledLayerRoute> {
        let injection = self.router.recall(&mut self.substrate, fingerprint)?;
        let (mask, memory_tokens) = decode_route_payload(&injection.context_ids)?;
        if mask.layer_count() != layer_count
            || mask.executed_count() < self.config.minimum_executed_layers.min(layer_count)
        {
            return None;
        }
        Some(RecalledLayerRoute {
            route_id: injection.route_id,
            mask,
            memory_tokens,
            score: self.router.last_recall_score,
            margin: self.router.last_recall_margin,
        })
    }

    pub fn candidate_mask(&self, trace: &Gemma2ForwardTrace) -> LayerExecutionMask {
        conservative_candidate_mask(trace, &self.config)
    }

    pub fn progressive_candidate_masks(
        &self,
        trace: &Gemma2ForwardTrace,
    ) -> Vec<LayerExecutionMask> {
        progressive_candidate_masks(trace, &self.config)
    }

    // API relacional posicional: agrupar los argumentos en un struct sólo
    // añadiría ruido en los sitios de llamada del benchmark adaptativo.
    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &mut self,
        context: ActivationFingerprint,
        activations: ActivationFingerprint,
        mask: LayerExecutionMask,
        memory_tokens: &[u32],
        quality: f32,
        route_id: Option<RouteId>,
        fallback: bool,
    ) -> Result<(), String> {
        self.state.generation = self.state.generation.saturating_add(1);
        if fallback {
            self.state.fallbacks = self.state.fallbacks.saturating_add(1);
        }
        if let Some(route_id) = route_id {
            self.router.feedback(
                &mut self.substrate,
                route_id,
                quality,
                self.state.generation,
            );
        }
        let full = self.state.buffer.push(AdaptiveExperience {
            context,
            activations,
            mask,
            memory_tokens: memory_tokens[memory_tokens.len().saturating_sub(128)..].to_vec(),
            quality: quality.clamp(0.0, 1.0),
            route_id,
        });
        if full {
            self.flush_fast_memory()?;
        }
        Ok(())
    }

    pub fn flush_fast_memory(&mut self) -> Result<usize, String> {
        let entries = std::mem::take(&mut self.state.buffer.entries);
        let mut consolidated = 0;
        let mut spin_gate = None;
        for entry in entries {
            if entry.quality < self.config.min_verified_quality {
                continue;
            }
            if spin_gate.is_none() {
                spin_gate = Some(new_offline_spin_gate()?);
            }
            let report = verify_with_offline_spin_gate(
                spin_gate.as_mut().expect("spin gate initialized"),
                &entry,
            );
            if !report {
                self.state.spin_gate_rejections = self.state.spin_gate_rejections.saturating_add(1);
                continue;
            }
            self.state.spin_gate_passes = self.state.spin_gate_passes.saturating_add(1);
            let payload = encode_route_payload(&entry.mask, &entry.memory_tokens);
            self.router.bind_verified(
                &mut self.substrate,
                &entry.context,
                &payload,
                self.state.generation,
            );
            self.router.bind_verified(
                &mut self.substrate,
                &entry.activations,
                &payload,
                self.state.generation,
            );
            consolidated += 1;
        }
        self.state.verified_routes = self
            .state
            .verified_routes
            .saturating_add(consolidated as u64);
        self.save()?;
        Ok(consolidated)
    }

    pub fn consolidate_sleep(&mut self) -> Result<SleepConsolidationReport, String> {
        let flushed = self.flush_fast_memory()?;
        let pruned_routes = self.router.sleep_decay_and_prune(
            self.config.max_routes,
            self.config.sleep_decay,
            self.config.protected_utility,
        );
        let pruned_relations = self
            .substrate
            .prune_observer_relations_to_budget(ROUTER_OBSERVER, self.config.relation_budget);
        self.save()?;
        Ok(SleepConsolidationReport {
            flushed,
            pruned_routes,
            pruned_relations,
            remaining_routes: self.router.registry.routes().len(),
        })
    }

    pub fn note_revalidation(&mut self) {
        self.state.revalidations = self.state.revalidations.saturating_add(1);
    }

    pub fn save(&self) -> Result<(), String> {
        save_native_checkpoint_transactional(
            &self.substrate,
            self.root.join("thermo-memory.cdt_native"),
        )?;
        self.router.save(self.root.join("routes.json"))?;
        let body = serde_json::to_vec_pretty(&self.state).map_err(|error| error.to_string())?;
        atomic_write(&self.root.join("adaptive-state.json"), &body)
    }
}

pub fn context_fingerprint(token_ids: &[u32]) -> ActivationFingerprint {
    let tail = &token_ids[token_ids.len().saturating_sub(128)..];
    let mut counts = HashMap::<u32, usize>::new();
    for token in tail {
        *counts.entry(*token).or_default() += 1;
    }
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
    ranked.truncate(20);
    let denominator = tail.len().max(1) as f32;
    let mut entries = ranked
        .into_iter()
        .map(|(token, count)| (token, count as f32 / denominator))
        .collect::<Vec<_>>();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tail.hash(&mut hasher);
    entries.push(((hasher.finish() & u32::MAX as u64) as u32, 1.0));
    let diversity = tail.iter().copied().collect::<HashSet<_>>().len() as f32 / denominator;
    ActivationFingerprint {
        entries,
        confidence: if tail.is_empty() { 0.0 } else { 0.85 },
        entropy: diversity.clamp(0.0, 1.0),
    }
}

pub fn activation_fingerprint(
    context: &ActivationFingerprint,
    trace: &Gemma2ForwardTrace,
) -> ActivationFingerprint {
    let maximum = trace
        .layers
        .iter()
        .map(|layer| layer.delta_rms)
        .fold(0.0f32, f32::max)
        .max(f32::EPSILON);
    let mut entries = context.entries.iter().take(12).copied().collect::<Vec<_>>();
    entries.extend(
        trace
            .layers
            .iter()
            .filter(|layer| layer.executed)
            .map(|layer| {
                let strength = (layer.delta_rms / maximum).clamp(0.0, 1.0);
                (0x8000_0000 | layer.layer as u32, strength)
            }),
    );
    ActivationFingerprint {
        entries,
        confidence: context.confidence,
        entropy: context.entropy,
    }
}

fn encode_route_payload(mask: &LayerExecutionMask, memory_tokens: &[u32]) -> Vec<u32> {
    let encoded_mask = mask.encode();
    let mut payload = Vec::with_capacity(encoded_mask.len() + memory_tokens.len() + 2);
    payload.push(encoded_mask.len() as u32);
    payload.extend(encoded_mask);
    payload.push(memory_tokens.len() as u32);
    payload.extend(memory_tokens);
    payload
}

fn decode_route_payload(payload: &[u32]) -> Option<(LayerExecutionMask, Vec<u32>)> {
    let mask_len = *payload.first()? as usize;
    if mask_len == 0 || payload.len() < 1 + mask_len + 1 {
        return None;
    }
    let mask = LayerExecutionMask::decode(&payload[1..1 + mask_len])?;
    let memory_len = payload[1 + mask_len] as usize;
    let memory_start = 1 + mask_len + 1;
    if payload.len() != memory_start + memory_len {
        return None;
    }
    Some((mask, payload[memory_start..].to_vec()))
}

pub fn conservative_candidate_mask(
    trace: &Gemma2ForwardTrace,
    config: &AdaptiveGemma2Config,
) -> LayerExecutionMask {
    let layer_count = trace.layers.len();
    if layer_count < 3 {
        return LayerExecutionMask::all(layer_count);
    }
    let minimum = config.minimum_executed_layers.min(layer_count);
    let skip_budget = ((layer_count as f32 * config.max_skip_fraction.clamp(0.0, 0.5)).floor()
        as usize)
        .min(layer_count.saturating_sub(minimum));
    let mut ranked = trace
        .layers
        .iter()
        .filter(|layer| layer.executed && layer.layer > 0 && layer.layer + 1 < layer_count)
        .map(|layer| (layer.layer, layer.delta_rms))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
    let mut enabled = vec![true; layer_count];
    let mut skipped = 0;
    for (layer, _) in ranked {
        if skipped >= skip_budget {
            break;
        }
        if !enabled[layer - 1] || !enabled[layer + 1] {
            continue;
        }
        enabled[layer] = false;
        skipped += 1;
    }
    LayerExecutionMask::from_enabled(enabled)
}

pub fn progressive_candidate_masks(
    trace: &Gemma2ForwardTrace,
    config: &AdaptiveGemma2Config,
) -> Vec<LayerExecutionMask> {
    let layer_count = trace.layers.len();
    if layer_count < 3 {
        return Vec::new();
    }
    let target = conservative_candidate_mask(trace, config);
    let skipped = (1..layer_count.saturating_sub(1))
        .filter(|layer| !target.executes(*layer))
        .collect::<Vec<_>>();
    let mut enabled = vec![true; layer_count];
    let mut masks = Vec::with_capacity(skipped.len());
    for layer in skipped {
        enabled[layer] = false;
        masks.push(LayerExecutionMask::from_enabled(enabled.clone()));
    }
    masks
}

fn fresh_adaptive_substrate() -> NativeThermoRqmEprSubstrate {
    NativeThermoRqmEprSubstrate::new(
        NativeThermoCdtConfig {
            slices: 8,
            nodes_per_slice: 256,
            temperature: 0.18,
            amplitude_decay: 0.001,
            ..NativeThermoCdtConfig::default()
        },
        NativeThermoRqmConfig {
            thermal_steps_per_train: 0,
            thermal_steps_per_query: 1,
            collect_query_diagnostics: false,
            max_candidates: 96,
            ..NativeThermoRqmConfig::default()
        },
        EntanglementConfig {
            max_links_per_node: 8,
            max_syncs_per_step: 0,
            create_threshold: 1.0,
            ..EntanglementConfig::default()
        },
    )
}

fn new_offline_spin_gate() -> Result<UnifiedSpinCognitiveEngine, String> {
    UnifiedSpinCognitiveEngine::periodic_pyrochlore(
        1,
        1,
        1,
        UnifiedSpinCognitiveConfig {
            bootstrap_cooling_steps: 64,
            cooling_steps_per_observation: 1,
            real_steps_per_observation: 1,
            backreaction_rate: 0.0,
            ..UnifiedSpinCognitiveConfig::default()
        },
    )
    .map_err(|error| error.to_string())
}

fn verify_with_offline_spin_gate(
    gate: &mut UnifiedSpinCognitiveEngine,
    experience: &AdaptiveExperience,
) -> bool {
    let mut source_hasher = std::collections::hash_map::DefaultHasher::new();
    for (feature, strength) in &experience.context.entries {
        feature.hash(&mut source_hasher);
        strength.to_bits().hash(&mut source_hasher);
    }
    let mut target_hasher = std::collections::hash_map::DefaultHasher::new();
    experience.mask.encode().hash(&mut target_hasher);
    let source = LatentConceptId(source_hasher.finish() as usize);
    let target = LatentConceptId(target_hasher.finish() as usize);
    let phase = (source.0 as f64 * 1.0e-6) % std::f64::consts::TAU;
    gate.train_relation(
        ObserverId(778_004),
        source,
        target,
        phase,
        experience.quality as f64,
        0.0,
        &[],
        32,
    )
    .gate
    .passed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_gemma2::LayerActivationSummary;

    #[test]
    fn context_fingerprints_are_deterministic() {
        let left = context_fingerprint(&[1, 2, 3, 2, 4]);
        let right = context_fingerprint(&[1, 2, 3, 2, 4]);
        assert_eq!(left.entries, right.entries);
    }

    #[test]
    fn candidate_masks_keep_boundaries_and_avoid_adjacent_skips() {
        let trace = Gemma2ForwardTrace {
            layers: (0..12)
                .map(|layer| LayerActivationSummary {
                    layer,
                    executed: true,
                    delta_rms: layer as f32,
                    ..LayerActivationSummary::default()
                })
                .collect(),
            executed_layers: 12,
            skipped_layers: 0,
        };
        let mask = conservative_candidate_mask(
            &trace,
            &AdaptiveGemma2Config {
                max_skip_fraction: 0.25,
                minimum_executed_layers: 6,
                ..AdaptiveGemma2Config::default()
            },
        );
        assert!(mask.executes(0));
        assert!(mask.executes(11));
        for layer in 1..11 {
            assert!(mask.executes(layer) || (mask.executes(layer - 1) && mask.executes(layer + 1)));
        }
    }

    #[test]
    fn fast_memory_signals_capacity() {
        let mut buffer = FastWorkingMemoryBuffer::new(1);
        assert!(buffer.push(AdaptiveExperience {
            context: context_fingerprint(&[1]),
            activations: context_fingerprint(&[2]),
            mask: LayerExecutionMask::all(4),
            memory_tokens: vec![1, 2],
            quality: 1.0,
            route_id: None,
        }));
    }

    #[test]
    fn progressive_masks_remove_one_additional_layer_at_a_time() {
        let trace = Gemma2ForwardTrace {
            layers: (0..20)
                .map(|layer| LayerActivationSummary {
                    layer,
                    executed: true,
                    delta_rms: layer as f32,
                    ..LayerActivationSummary::default()
                })
                .collect(),
            executed_layers: 20,
            skipped_layers: 0,
        };
        let masks = progressive_candidate_masks(
            &trace,
            &AdaptiveGemma2Config {
                max_skip_fraction: 0.20,
                minimum_executed_layers: 8,
                ..AdaptiveGemma2Config::default()
            },
        );
        assert!(!masks.is_empty());
        for (index, mask) in masks.iter().enumerate() {
            assert_eq!(mask.executed_count(), 20 - index - 1);
        }
    }

    #[test]
    fn verified_experience_passes_offline_spin_gate() {
        let mut gate = new_offline_spin_gate().unwrap();
        let experience = AdaptiveExperience {
            context: context_fingerprint(&[10, 20, 30, 20]),
            activations: context_fingerprint(&[40, 50]),
            mask: LayerExecutionMask::all(26),
            memory_tokens: vec![10, 20, 30],
            quality: 1.0,
            route_id: None,
        };
        assert!(verify_with_offline_spin_gate(&mut gate, &experience));
    }
}
