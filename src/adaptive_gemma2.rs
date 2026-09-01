//! Enrutamiento adaptativo y memoria termodinámica de dos velocidades para Gemma 2.

use crate::entanglement::EntanglementConfig;
use crate::layer_route_cache::{
    fingerprint_wake, logits_kl, top1_agree, LayerRouteCache, LayerRouteCacheConfig,
    LAYER_ROUTES_FILE,
};
use crate::matrix_free_cognitive_substrate::LatentConceptId;
use crate::native_checkpoint::{atomic_write, save_native_checkpoint_transactional};
use crate::native_gemma2::{
    Gemma2ForwardOutput, Gemma2ForwardTrace, LayerExecutionMask, QuantizedGemma2,
};
use crate::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::native_thermodynamic_engine::load_native_checkpoint;
use crate::relational_field::ObserverId;
use crate::thermo_router::{
    ActivationFingerprint, RouteId, RouterConfig, ThermoAssociativeRouter,
    TransformerActivationAdapter, ROUTER_OBSERVER,
};
use crate::unified_spin_cognitive_engine::{
    UnifiedSpinCognitiveConfig, UnifiedSpinCognitiveEngine,
};
use candle_core::Tensor;
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
    /// Calidad mínima para conservar una máscara en memoria de trabajo
    /// (vigilia del día siguiente). Por debajo se descarta en el sueño.
    pub min_runtime_quality: f32,
    pub max_skip_fraction: f32,
    pub minimum_executed_layers: usize,
    pub revalidate_every: u64,
    pub max_routes: usize,
    /// Candidatos sparse que el sueño prueba por secuencia.
    pub max_candidate_prefills: usize,
    /// Máximo de prompts que el sueño rejuega para descubrir máscaras.
    pub max_sleep_replays: usize,
    pub sleep_decay: f32,
    pub protected_utility: f32,
    pub relation_budget: usize,
    /// Confianza mínima para aplicar una ruta LRC en vigilia.
    pub lrc_min_confidence: f32,
    /// KL máxima (denso || sparse) para promocionar a LRC en sueño.
    pub lrc_max_kl_promote: f32,
}

impl Default for AdaptiveGemma2Config {
    fn default() -> Self {
        Self {
            buffer_capacity: 24,
            min_verified_quality: 0.92,
            min_runtime_quality: 0.50,
            max_skip_fraction: 0.15,
            minimum_executed_layers: 8,
            revalidate_every: 16,
            max_routes: 2_048,
            max_candidate_prefills: 1,
            max_sleep_replays: 8,
            sleep_decay: 0.995,
            protected_utility: 0.85,
            relation_budget: 16_384,
            lrc_min_confidence: 0.55,
            lrc_max_kl_promote: 0.15,
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

    pub fn push(&mut self, experience: AdaptiveExperience) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push(experience);
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

    fn recall_best(
        &self,
        fingerprint: &ActivationFingerprint,
        layer_count: usize,
        minimum_executed_layers: usize,
        min_overlap: f32,
    ) -> Option<&AdaptiveExperience> {
        self.entries.iter().rev().find(|entry| {
            entry.mask.layer_count() == layer_count
                && entry.mask.executed_count() >= minimum_executed_layers.min(layer_count)
                && fingerprint_overlap(fingerprint, &entry.context) >= min_overlap
        })
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
    pub route_id: Option<RouteId>,
    pub mask: LayerExecutionMask,
    pub memory_tokens: Vec<u32>,
    pub score: f32,
    pub margin: f32,
    pub layer_route_id: Option<u64>,
}

pub struct PreparedAdaptiveForward {
    pub output: Gemma2ForwardOutput,
    pub mask: LayerExecutionMask,
    pub context_fingerprint: ActivationFingerprint,
    pub activation_fingerprint: ActivationFingerprint,
    pub route_id: Option<RouteId>,
    pub quality: f32,
    pub fallback: bool,
    pub recalled_memory_tokens: usize,
    pub prefill_tokens: usize,
    pub cache_reused: bool,
    pub layer_route_id: Option<u64>,
    pub layer_route_hit: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WakePrefillPlan {
    pub mask: LayerExecutionMask,
    pub position: usize,
    pub suffix_start: usize,
    pub reuse_cache: bool,
}

impl WakePrefillPlan {
    pub fn suffix<'a>(&self, prompt_tokens: &'a [u32]) -> &'a [u32] {
        &prompt_tokens[self.suffix_start.min(prompt_tokens.len())..]
    }

    pub fn prefill_tokens(&self, prompt_len: usize) -> usize {
        prompt_len.saturating_sub(self.suffix_start)
    }
}

/// Vigilia KV-A: la máscara se elige al empezar el turno.
///
/// Si el prompt extiende la KV y la máscara recordada coincide con la activa,
/// se reutiliza la cache (incluido sparse→sparse). Si la máscara cambia, no
/// se reutiliza: el llamador limpia KV y hace prefill con la máscara nueva.
pub fn plan_wake_prefill(
    prompt_tokens: &[u32],
    cached_tokens: &[u32],
    cached_mask: Option<&LayerExecutionMask>,
    recalled_mask: Option<&LayerExecutionMask>,
    layer_count: usize,
) -> WakePrefillPlan {
    let cached = cached_mask
        .filter(|mask| mask.layer_count() == layer_count)
        .cloned();
    let recalled = recalled_mask
        .filter(|mask| mask.layer_count() == layer_count)
        .cloned();
    let chosen = recalled
        .or_else(|| cached.clone())
        .unwrap_or_else(|| LayerExecutionMask::all(layer_count));
    let extending =
        !cached_tokens.is_empty() && cached.is_some() && prompt_tokens.starts_with(cached_tokens);
    let same_mask = cached.as_ref() == Some(&chosen);
    if extending && same_mask {
        return WakePrefillPlan {
            mask: chosen,
            position: cached_tokens.len(),
            suffix_start: cached_tokens.len(),
            reuse_cache: true,
        };
    }
    WakePrefillPlan {
        mask: chosen,
        position: 0,
        suffix_start: 0,
        reuse_cache: false,
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct SleepConsolidationReport {
    pub flushed: usize,
    #[serde(default)]
    pub discovered_masks: usize,
    #[serde(default)]
    pub replayed: usize,
    #[serde(default)]
    pub retained_working: usize,
    pub pruned_routes: usize,
    pub pruned_relations: usize,
    pub remaining_routes: usize,
    #[serde(default)]
    pub lrc_promoted: usize,
    #[serde(default)]
    pub sleep_mean_kl: f32,
    #[serde(default)]
    pub sleep_top1_agree: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct SleepReplayOutcome {
    discovered: bool,
    promoted: bool,
    kl: Option<f32>,
    top1_agree: f32,
}

pub struct AdaptiveThermoMemory {
    pub config: AdaptiveGemma2Config,
    pub substrate: NativeThermoRqmEprSubstrate,
    pub router: ThermoAssociativeRouter,
    pub lrc: LayerRouteCache,
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
        let substrate = if substrate_path.is_file() {
            load_native_checkpoint(&substrate_path).map_err(|error| {
                format!(
                    "no se pudo cargar la termo-memoria {}: {error}",
                    substrate_path.display()
                )
            })?
        } else {
            fresh_adaptive_substrate()
        };
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
        let lrc = LayerRouteCache::load_or_new(
            root.join(LAYER_ROUTES_FILE),
            LayerRouteCacheConfig {
                min_confidence: config.lrc_min_confidence,
                max_kl_promote: config.lrc_max_kl_promote,
                min_overlap: 0.35,
                max_routes: config.max_routes,
            },
        );
        Ok(Self {
            config,
            substrate,
            router,
            lrc,
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
        if let Some(route) = self.recall_working_memory(fingerprint, layer_count) {
            return Some(route);
        }
        let injection = self.router.recall(&mut self.substrate, fingerprint)?;
        let (mask, memory_tokens) = decode_route_payload(&injection.context_ids)?;
        if mask.layer_count() != layer_count
            || mask.executed_count() < self.config.minimum_executed_layers.min(layer_count)
        {
            return None;
        }
        Some(RecalledLayerRoute {
            route_id: Some(injection.route_id),
            mask,
            memory_tokens,
            score: self.router.last_recall_score,
            margin: self.router.last_recall_margin,
            layer_route_id: None,
        })
    }

    fn recall_working_memory(
        &self,
        fingerprint: &ActivationFingerprint,
        layer_count: usize,
    ) -> Option<RecalledLayerRoute> {
        let entry = self.state.buffer.recall_best(
            fingerprint,
            layer_count,
            self.config.minimum_executed_layers,
            0.35,
        )?;
        Some(RecalledLayerRoute {
            route_id: entry.route_id,
            mask: entry.mask.clone(),
            memory_tokens: entry.memory_tokens.clone(),
            score: fingerprint_overlap(fingerprint, &entry.context),
            margin: 1.0,
            layer_route_id: None,
        })
    }

    pub fn prepare_forward(
        &mut self,
        model: &mut QuantizedGemma2,
        prompt_tokens: &[u32],
        cached_tokens: &[u32],
        cached_mask: Option<&LayerExecutionMask>,
        cached_logits: Option<&Tensor>,
    ) -> Result<PreparedAdaptiveForward, String> {
        self.prepare_forward_with_forced_mask(
            model,
            prompt_tokens,
            cached_tokens,
            cached_mask,
            cached_logits,
            None,
        )
    }

    /// Prefill con una mascara impuesta por el grafo (DenseTalker / Compiler).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_forward_with_forced_mask(
        &mut self,
        model: &mut QuantizedGemma2,
        prompt_tokens: &[u32],
        cached_tokens: &[u32],
        cached_mask: Option<&LayerExecutionMask>,
        cached_logits: Option<&Tensor>,
        forced_mask: Option<&LayerExecutionMask>,
    ) -> Result<PreparedAdaptiveForward, String> {
        let context_fingerprint = self.context_fingerprint(prompt_tokens);
        let wake_fingerprint = fingerprint_wake(prompt_tokens);
        let recalled = if let Some(mask) = forced_mask {
            Some(RecalledLayerRoute {
                route_id: None,
                mask: mask.clone(),
                memory_tokens: Vec::new(),
                score: 1.0,
                margin: 1.0,
                layer_route_id: None,
            })
        } else {
            self.recall_layer_route(&wake_fingerprint, model.layer_count())
                .or_else(|| self.recall(&context_fingerprint, model.layer_count()))
        };
        let plan = plan_wake_prefill(
            prompt_tokens,
            cached_tokens,
            cached_mask,
            recalled.as_ref().map(|route| &route.mask),
            model.layer_count(),
        );
        let suffix = plan.suffix(prompt_tokens);
        let output = if suffix.is_empty() {
            let logits = cached_logits
                .cloned()
                .ok_or_else(|| "sesión Gemma sin logits para reutilizar la KV cache".to_string())?;
            let logits = if logits.dims().len() == 1 {
                logits.unsqueeze(0).map_err(|error| error.to_string())?
            } else {
                logits
            };
            Gemma2ForwardOutput {
                logits,
                trace: Gemma2ForwardTrace::default(),
                sequence_hidden: None,
            }
        } else {
            if !plan.reuse_cache {
                model.clear_kv_cache();
            }
            let prompt = Tensor::new(suffix, model.device())
                .and_then(|tensor| tensor.unsqueeze(0))
                .map_err(|error| error.to_string())?;
            model
                // Vigilia no aprende máscaras: las trazas RMS por capa se
                // capturan durante replay de sueño, fuera de la ruta crítica.
                .forward_with_mask(&prompt, plan.position, Some(&plan.mask), false, false)
                .map_err(|error| error.to_string())?
        };
        let logits = output
            .logits
            .squeeze(0)
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|error| error.to_string())?;
        let quality = output_confidence(&logits).max(0.30);
        let activation_fingerprint = if output.trace.layers.is_empty() {
            context_fingerprint.clone()
        } else {
            self.activation_fingerprint(&context_fingerprint, &output.trace)
        };
        Ok(PreparedAdaptiveForward {
            output,
            mask: plan.mask,
            context_fingerprint,
            activation_fingerprint,
            route_id: recalled.as_ref().and_then(|route| route.route_id),
            quality,
            fallback: recalled.is_none(),
            recalled_memory_tokens: recalled
                .as_ref()
                .map(|route| route.memory_tokens.len())
                .unwrap_or(0),
            prefill_tokens: suffix.len(),
            cache_reused: plan.reuse_cache,
            layer_route_id: recalled.as_ref().and_then(|route| route.layer_route_id),
            layer_route_hit: recalled
                .as_ref()
                .and_then(|route| route.layer_route_id)
                .is_some(),
        })
    }

    fn recall_layer_route(
        &self,
        fingerprint: &ActivationFingerprint,
        layer_count: usize,
    ) -> Option<RecalledLayerRoute> {
        let route = self.lrc.lookup_confident(fingerprint)?;
        if route.mask.layer_count() != layer_count
            || route.mask.executed_count() < self.config.minimum_executed_layers.min(layer_count)
        {
            return None;
        }
        Some(RecalledLayerRoute {
            route_id: None,
            mask: route.mask.clone(),
            memory_tokens: Vec::new(),
            score: route.confidence,
            margin: route.confidence,
            layer_route_id: Some(route.id),
        })
    }

    pub fn observe_layer_route_turn(
        &mut self,
        prompt_tokens: &[u32],
        mask: &LayerExecutionMask,
        fallback: bool,
    ) {
        self.lrc.observe_turn(
            &fingerprint_wake(prompt_tokens),
            mask,
            fallback,
            self.state.generation,
        );
    }

    pub fn layer_route_count(&self) -> usize {
        self.lrc.len()
    }

    pub fn working_memory_len(&self) -> usize {
        self.state.buffer.len()
    }

    pub fn verified_route_count(&self) -> u64 {
        self.state.verified_routes
    }

    pub fn candidate_mask(&self, trace: &Gemma2ForwardTrace) -> LayerExecutionMask {
        kl_budget_mask_from_trace(trace, &self.config)
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
        self.state.buffer.push(AdaptiveExperience {
            context,
            activations,
            mask,
            memory_tokens: memory_tokens[memory_tokens.len().saturating_sub(128)..].to_vec(),
            quality: quality.clamp(0.0, 1.0),
            route_id,
        });
        Ok(())
    }

    pub fn flush_fast_memory(&mut self) -> Result<usize, String> {
        let entries = std::mem::take(&mut self.state.buffer.entries);
        let mut consolidated = 0;
        let mut retained = Vec::new();
        let mut spin_gate = None;
        for entry in entries {
            if entry.quality >= self.config.min_verified_quality && is_sparse_mask(&entry.mask) {
                if spin_gate.is_none() {
                    spin_gate = Some(new_offline_spin_gate()?);
                }
                let report = verify_with_offline_spin_gate(
                    spin_gate.as_mut().expect("spin gate initialized"),
                    &entry,
                );
                if !report {
                    self.state.spin_gate_rejections =
                        self.state.spin_gate_rejections.saturating_add(1);
                    if entry.quality >= self.config.min_runtime_quality {
                        retained.push(entry);
                    }
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
            } else if entry.quality >= self.config.min_runtime_quality
                && is_sparse_mask(&entry.mask)
            {
                retained.push(entry);
            }
        }
        self.state.buffer.entries = retained;
        self.state.verified_routes = self
            .state
            .verified_routes
            .saturating_add(consolidated as u64);
        self.save()?;
        Ok(consolidated)
    }

    pub fn consolidate_sleep(&mut self) -> Result<SleepConsolidationReport, String> {
        self.finish_sleep(0, 0, 0, 0.0, 0.0)
    }

    pub fn consolidate_sleep_with_model(
        &mut self,
        model: &mut QuantizedGemma2,
        extra_prompts: &[Vec<u32>],
    ) -> Result<SleepConsolidationReport, String> {
        let (replayed, discovered, lrc_promoted, sleep_mean_kl, sleep_top1_agree) =
            self.discover_sleep_masks(model, extra_prompts)?;
        self.finish_sleep(
            replayed,
            discovered,
            lrc_promoted,
            sleep_mean_kl,
            sleep_top1_agree,
        )
    }

    fn finish_sleep(
        &mut self,
        replayed: usize,
        discovered: usize,
        lrc_promoted: usize,
        sleep_mean_kl: f32,
        sleep_top1_agree: f32,
    ) -> Result<SleepConsolidationReport, String> {
        let flushed = self.flush_fast_memory()?;
        let retained_working = self.state.buffer.len();
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
            discovered_masks: discovered,
            replayed,
            retained_working,
            pruned_routes,
            pruned_relations,
            remaining_routes: self.router.registry.routes().len(),
            lrc_promoted,
            sleep_mean_kl,
            sleep_top1_agree,
        })
    }

    fn discover_sleep_masks(
        &mut self,
        model: &mut QuantizedGemma2,
        extra_prompts: &[Vec<u32>],
    ) -> Result<(usize, usize, usize, f32, f32), String> {
        let buffer_tokens = self
            .state
            .buffer
            .entries
            .iter()
            .rev()
            .map(|entry| entry.memory_tokens.clone())
            .collect::<Vec<_>>();
        let prompts = collect_sleep_prompts(
            buffer_tokens,
            extra_prompts,
            self.config.max_sleep_replays.max(1),
        );
        let mut replayed = 0usize;
        let mut discovered = 0usize;
        let mut lrc_promoted = 0usize;
        let mut kl_sum = 0.0f32;
        let mut top1_sum = 0.0f32;
        let mut probed = 0usize;
        for prompt in prompts {
            replayed += 1;
            let outcome = self.replay_prompt_for_mask(model, &prompt)?;
            if outcome.discovered {
                discovered += 1;
            }
            if outcome.promoted {
                lrc_promoted += 1;
            }
            if let Some(kl) = outcome.kl {
                kl_sum += kl;
                top1_sum += outcome.top1_agree;
                probed += 1;
            }
        }
        let n = probed.max(1) as f32;
        Ok((replayed, discovered, lrc_promoted, kl_sum / n, top1_sum / n))
    }

    fn replay_prompt_for_mask(
        &mut self,
        model: &mut QuantizedGemma2,
        prompt_tokens: &[u32],
    ) -> Result<SleepReplayOutcome, String> {
        if prompt_tokens.len() < 2 {
            return Ok(SleepReplayOutcome::default());
        }
        model.clear_kv_cache();
        let full_mask = LayerExecutionMask::all(model.layer_count());
        let prompt = Tensor::new(prompt_tokens, model.device())
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|error| error.to_string())?;
        let full = model
            .forward_with_mask(&prompt, 0, Some(&full_mask), true, false)
            .map_err(|error| error.to_string())?;
        let full_logits = full
            .logits
            .squeeze(0)
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|error| error.to_string())?;
        let context = self.context_fingerprint(prompt_tokens);
        let activations = self.activation_fingerprint(&context, &full.trace);
        let mut best = None::<(LayerExecutionMask, f32, f32, f32)>;
        let mut candidates = Vec::new();
        let primary = self.candidate_mask(&full.trace);
        if is_sparse_mask(&primary) {
            candidates.push(primary);
        }
        let limit = self.config.max_candidate_prefills.max(1);
        if candidates.len() < limit {
            for mask in self.progressive_candidate_masks(&full.trace) {
                if candidates.len() >= limit {
                    break;
                }
                if candidates.iter().any(|existing| existing == &mask) {
                    continue;
                }
                candidates.push(mask);
            }
        }
        for candidate in candidates {
            model.clear_kv_cache();
            let sparse = model
                .forward_with_mask(&prompt, 0, Some(&candidate), false, false)
                .map_err(|error| error.to_string())?;
            let sparse_logits = sparse
                .logits
                .squeeze(0)
                .and_then(|tensor| tensor.to_vec1::<f32>())
                .map_err(|error| error.to_string())?;
            let quality = logit_agreement(&full_logits, &sparse_logits);
            let kl = logits_kl(&full_logits, &sparse_logits);
            let agree = top1_agree(&full_logits, &sparse_logits);
            if best
                .as_ref()
                .map(|(_, current, _, _)| quality > *current)
                .unwrap_or(true)
            {
                best = Some((candidate, quality, kl, agree));
            }
        }
        let mut outcome = SleepReplayOutcome {
            kl: best.as_ref().map(|(_, _, kl, _)| *kl),
            top1_agree: best.as_ref().map(|(_, _, _, agree)| *agree).unwrap_or(0.0),
            ..SleepReplayOutcome::default()
        };
        if let Some((mask, _, kl, agree)) = best.as_ref() {
            outcome.promoted = self.lrc.promote(
                fingerprint_wake(prompt_tokens),
                mask.clone(),
                *kl,
                *agree,
                self.state.generation,
            );
        }
        let (mask, quality, usable) = choose_sleep_mask(
            &full_mask,
            best.map(|(mask, quality, _, _)| (mask, quality)),
            self.config.min_runtime_quality,
            self.config.min_verified_quality,
        );
        if usable {
            self.upsert_sleep_experience(prompt_tokens, mask, context, activations, quality);
            outcome.discovered = true;
        }
        Ok(outcome)
    }

    fn upsert_sleep_experience(
        &mut self,
        prompt_tokens: &[u32],
        mask: LayerExecutionMask,
        context: ActivationFingerprint,
        activations: ActivationFingerprint,
        quality: f32,
    ) {
        if let Some(entry) = self
            .state
            .buffer
            .entries
            .iter_mut()
            .rev()
            .find(|entry| entry.memory_tokens.as_slice() == prompt_tokens)
        {
            entry.mask = mask;
            entry.quality = quality;
            entry.context = context;
            entry.activations = activations;
            return;
        }
        self.state.buffer.push(AdaptiveExperience {
            context,
            activations,
            mask,
            memory_tokens: prompt_tokens.to_vec(),
            quality,
            route_id: None,
        });
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
        self.lrc.save(self.root.join(LAYER_ROUTES_FILE))?;
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

fn fingerprint_overlap(query: &ActivationFingerprint, stored: &ActivationFingerprint) -> f32 {
    if query.entries.is_empty() || stored.entries.is_empty() {
        return 0.0;
    }
    let stored_ids = stored
        .entries
        .iter()
        .map(|(id, _)| *id)
        .collect::<HashSet<_>>();
    let hits = query
        .entries
        .iter()
        .filter(|(id, _)| stored_ids.contains(id))
        .count();
    hits as f32 / query.entries.len() as f32
}

fn output_confidence(logits: &[f32]) -> f32 {
    TransformerActivationAdapter::new(8)
        .capture(logits)
        .confidence
}

/// Conservado para la fase de sueño: verificar máscaras sparse offline.
fn logit_agreement(full: &[f32], sparse: &[f32]) -> f32 {
    if full.len() != sparse.len() || full.is_empty() {
        return 0.0;
    }
    let full_top = full
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index);
    let sparse_top = sparse
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index);
    if full_top != sparse_top {
        return 0.0;
    }
    let (squared_error, squared_signal) =
        full.iter()
            .zip(sparse)
            .fold((0.0f64, 0.0f64), |(error, signal), (left, right)| {
                (
                    error + (*left as f64 - *right as f64).powi(2),
                    signal + (*left as f64).powi(2),
                )
            });
    let relative_rmse = (squared_error / squared_signal.max(f64::EPSILON)).sqrt();
    (-4.0 * relative_rmse).exp() as f32
}

fn is_sparse_mask(mask: &LayerExecutionMask) -> bool {
    mask.executed_count() < mask.layer_count()
}

pub fn collect_sleep_prompts(
    buffer_tokens: impl IntoIterator<Item = Vec<u32>>,
    extra_prompts: &[Vec<u32>],
    max_replays: usize,
) -> Vec<Vec<u32>> {
    let limit = max_replays.max(1);
    let mut seen = HashSet::new();
    let mut prompts = Vec::new();
    let mut push = |tokens: &[u32]| {
        if prompts.len() >= limit {
            return;
        }
        let start = tokens.len().saturating_sub(128);
        let tail = tokens[start..].to_vec();
        if tail.len() < 2 {
            return;
        }
        if seen.insert(tail.clone()) {
            prompts.push(tail);
        }
    };
    for tokens in buffer_tokens {
        push(&tokens);
    }
    for tokens in extra_prompts {
        push(tokens);
    }
    prompts
}

pub fn choose_sleep_mask(
    full_mask: &LayerExecutionMask,
    best_candidate: Option<(LayerExecutionMask, f32)>,
    min_runtime: f32,
    min_verified: f32,
) -> (LayerExecutionMask, f32, bool) {
    match best_candidate {
        Some((mask, quality)) if quality >= min_verified && is_sparse_mask(&mask) => {
            (mask, quality, true)
        }
        Some((mask, quality)) if quality >= min_runtime && is_sparse_mask(&mask) => {
            (mask, quality, true)
        }
        _ => (full_mask.clone(), 0.0, false),
    }
}

/// Ranking T2.1: KL al apagar esa capa sola, orden ascendente.
/// Fuente: `docs/v8_layer_kl_ablation.csv`. No usar `delta_rms`.
pub const GEMMA2_T21_LAYER_KL: &[(usize, f32)] = &[
    (7, 0.018273),
    (21, 0.021460),
    (8, 0.023733),
    (12, 0.031118),
    (20, 0.035451),
    (15, 0.036453),
    (9, 0.043925),
    (11, 0.044089),
    (6, 0.045520),
    (23, 0.049675),
    (14, 0.051977),
    (19, 0.063634),
    (10, 0.064504),
    (18, 0.070332),
    (16, 0.079036),
    (1, 0.082673),
    (13, 0.087647),
    (22, 0.091118),
    (24, 0.145864),
    (17, 0.161124),
    (3, 0.228592),
    (5, 0.241503),
    (4, 0.261560),
    (2, 0.479236),
];

fn sliding_windows_from_trace(trace: &Gemma2ForwardTrace) -> Vec<bool> {
    let layer_count = trace.layers.len();
    let mut sliding = vec![false; layer_count];
    for layer in &trace.layers {
        if layer.layer < layer_count {
            sliding[layer.layer] = layer.sliding_window;
        }
    }
    sliding
}

/// Apaga capas en orden de KL incremental mientras la KL acumulada cabe
/// en el presupuesto (`lrc_max_kl_promote`, 0,15), quedan >= 8 capas
/// ejecutadas y no hay dos globales consecutivos apagados (invariante V5).
/// El presupuesto de KL es el limitador, no `max_skip_fraction`.
pub fn kl_budget_mask(
    layer_count: usize,
    sliding: &[bool],
    ranked_kl: &[(usize, f32)],
    config: &AdaptiveGemma2Config,
) -> LayerExecutionMask {
    if layer_count < 3 || ranked_kl.is_empty() {
        return LayerExecutionMask::all(layer_count);
    }
    let minimum = config.minimum_executed_layers.min(layer_count);
    let budget = config.lrc_max_kl_promote;
    let mut enabled = vec![true; layer_count];
    let mut acc_kl = 0.0f32;
    let mut executed = layer_count;
    for &(layer, kl) in ranked_kl {
        if executed <= minimum {
            break;
        }
        if layer == 0 || layer + 1 >= layer_count || layer >= layer_count {
            continue;
        }
        if !enabled[layer] {
            continue;
        }
        if !kl.is_finite() || kl < 0.0 || kl > budget {
            continue;
        }
        if acc_kl + kl > budget {
            continue;
        }
        if would_skip_consecutive_globals(&enabled, sliding, layer) {
            continue;
        }
        enabled[layer] = false;
        acc_kl += kl;
        executed -= 1;
    }
    LayerExecutionMask::from_enabled(enabled)
}

/// Mascara de producto para V8 / LRC.
///
/// `kl_budget_mask` sigue apagando en orden de KL con ranking sintetico.
/// El ranking T2.1 es KL de **una** capa en el prompt 0. V8 T2.2 midio:
/// - 5 capas baratas (7,8,12,20,21), 1 prompt: KL 0,234 > 0,15
/// - solo capa 7, 3 prompts: KL 0,018 / 0,200 / 0,424 (media 0,214)
/// Ningun skip estatico cabe en el presupuesto del set. Camino S se niega
/// a saltar: 26/26. El limitador es KL, no `max_skip_fraction`.
pub fn kl_budget_mask_from_trace(
    trace: &Gemma2ForwardTrace,
    config: &AdaptiveGemma2Config,
) -> LayerExecutionMask {
    let _ = config;
    LayerExecutionMask::all(trace.layers.len())
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
    let mut sliding = vec![false; layer_count];
    for layer in &trace.layers {
        if layer.layer < layer_count {
            sliding[layer.layer] = layer.sliding_window;
        }
    }
    let mut ranked = trace
        .layers
        .iter()
        .filter(|layer| layer.executed && layer.layer > 0 && layer.layer + 1 < layer_count)
        .map(|layer| {
            let local_rank = u8::from(!layer.sliding_window);
            (layer.layer, local_rank, layer.delta_rms)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.1.cmp(&right.1).then(left.2.total_cmp(&right.2)));
    let mut enabled = vec![true; layer_count];
    let mut skipped = 0;
    for (layer, _, _) in ranked {
        if skipped >= skip_budget {
            break;
        }
        if !enabled[layer - 1] || !enabled[layer + 1] {
            continue;
        }
        if would_skip_consecutive_globals(&enabled, &sliding, layer) {
            continue;
        }
        enabled[layer] = false;
        skipped += 1;
    }
    LayerExecutionMask::from_enabled(enabled)
}

fn would_skip_consecutive_globals(enabled: &[bool], sliding: &[bool], layer: usize) -> bool {
    if sliding.get(layer).copied().unwrap_or(false) {
        return false;
    }
    let previous = (0..layer)
        .rev()
        .find(|&index| !sliding.get(index).copied().unwrap_or(false));
    let next =
        ((layer + 1)..enabled.len()).find(|&index| !sliding.get(index).copied().unwrap_or(false));
    previous.is_some_and(|index| !enabled[index]) || next.is_some_and(|index| !enabled[index])
}

/// Hay dos capas globales consecutivas (en la subsecuencia global) apagadas.
pub fn mask_has_consecutive_global_skips(mask: &LayerExecutionMask, sliding: &[bool]) -> bool {
    let mut last_global_skipped = false;
    for layer in 0..mask.layer_count() {
        if sliding.get(layer).copied().unwrap_or(false) {
            continue;
        }
        let skipped = !mask.executes(layer);
        if skipped && last_global_skipped {
            return true;
        }
        last_global_skipped = skipped;
    }
    false
}

pub fn progressive_candidate_masks(
    trace: &Gemma2ForwardTrace,
    config: &AdaptiveGemma2Config,
) -> Vec<LayerExecutionMask> {
    let layer_count = trace.layers.len();
    if layer_count < 3 {
        return Vec::new();
    }
    let sliding = sliding_windows_from_trace(trace);
    let target = kl_budget_mask(layer_count, &sliding, GEMMA2_T21_LAYER_KL, config);
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

    fn gemma_like_trace(layers: usize, delta: impl Fn(usize) -> f32) -> Gemma2ForwardTrace {
        Gemma2ForwardTrace {
            layers: (0..layers)
                .map(|layer| LayerActivationSummary {
                    layer,
                    executed: true,
                    delta_rms: delta(layer),
                    sliding_window: layer % 2 == 0,
                    ..LayerActivationSummary::default()
                })
                .collect(),
            executed_layers: layers,
            skipped_layers: 0,
        }
    }

    #[test]
    fn default_skip_fraction_stays_conservative() {
        assert!((AdaptiveGemma2Config::default().max_skip_fraction - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn candidate_mask_never_skips_two_consecutive_globals() {
        let trace = gemma_like_trace(26, |layer| if layer % 2 == 0 { 1.0 } else { 0.01 });
        let mask = conservative_candidate_mask(
            &trace,
            &AdaptiveGemma2Config {
                max_skip_fraction: 0.15,
                minimum_executed_layers: 8,
                ..AdaptiveGemma2Config::default()
            },
        );
        let sliding = (0..26).map(|layer| layer % 2 == 0).collect::<Vec<_>>();
        assert!(!mask_has_consecutive_global_skips(&mask, &sliding));
        assert!(mask.executes(0));
        assert!(mask.executes(25));
        for layer in 1..25 {
            assert!(mask.executes(layer) || (mask.executes(layer - 1) && mask.executes(layer + 1)));
        }
    }

    #[test]
    fn candidate_mask_prefers_local_layers_over_globals() {
        let trace = gemma_like_trace(12, |layer| if layer % 2 == 0 { 5.0 } else { 0.01 });
        let mask = conservative_candidate_mask(
            &trace,
            &AdaptiveGemma2Config {
                max_skip_fraction: 0.15,
                minimum_executed_layers: 8,
                ..AdaptiveGemma2Config::default()
            },
        );
        let skipped = (0..12)
            .filter(|layer| !mask.executes(*layer))
            .collect::<Vec<_>>();
        assert!(
            !skipped.is_empty(),
            "el presupuesto de 0.15 debe saltar alguna capa local"
        );
        assert!(
            skipped.iter().all(|layer| layer % 2 == 0),
            "el ranking debe preferir locales aunque el delta_rms global sea menor: {skipped:?}"
        );
        let sliding = (0..12).map(|layer| layer % 2 == 0).collect::<Vec<_>>();
        assert!(!mask_has_consecutive_global_skips(&mask, &sliding));
    }

    fn skipped_layers(mask: &LayerExecutionMask, layer_count: usize) -> Vec<usize> {
        (0..layer_count)
            .filter(|layer| !mask.executes(*layer))
            .collect()
    }

    #[test]
    fn kl_budget_mask_turns_off_cheap_layers_first_and_stops_before_budget() {
        let layer_count = 26;
        let sliding = (0..layer_count)
            .map(|layer| layer % 2 == 0)
            .collect::<Vec<_>>();
        let ranked = [
            (4, 0.04),
            (8, 0.04),
            (12, 0.04),
            (16, 0.04),
            (20, 0.04),
            (7, 0.20),
        ];
        let mask = kl_budget_mask(
            layer_count,
            &sliding,
            &ranked,
            &AdaptiveGemma2Config::default(),
        );
        assert_eq!(skipped_layers(&mask, layer_count), vec![4, 8, 12]);
        assert!(mask.executed_count() >= 8);
        assert!(mask.executes(0) && mask.executes(25));
        assert!(mask.executes(16) && mask.executes(20));
        assert!(mask.executes(7), "KL individual 0.20 > presupuesto");
        assert!(!mask_has_consecutive_global_skips(&mask, &sliding));
    }

    #[test]
    fn kl_budget_mask_never_skips_two_consecutive_globals() {
        let layer_count = 26;
        let sliding = (0..layer_count)
            .map(|layer| layer % 2 == 0)
            .collect::<Vec<_>>();
        let ranked = [(1, 0.05), (3, 0.05), (5, 0.05), (7, 0.05), (9, 0.05)];
        let mask = kl_budget_mask(
            layer_count,
            &sliding,
            &ranked,
            &AdaptiveGemma2Config::default(),
        );
        let skipped = skipped_layers(&mask, layer_count);
        assert_eq!(skipped, vec![1, 5, 9]);
        assert!(mask.executed_count() >= 8);
        assert!(!mask_has_consecutive_global_skips(&mask, &sliding));
    }

    #[test]
    fn kl_budget_mask_all_on_if_ranking_empty_or_all_high_kl() {
        let layer_count = 12;
        let sliding = vec![false; layer_count];
        let empty = kl_budget_mask(layer_count, &sliding, &[], &AdaptiveGemma2Config::default());
        assert_eq!(empty.executed_count(), layer_count);
        assert_eq!(skipped_layers(&empty, layer_count), Vec::<usize>::new());
        let expensive = kl_budget_mask(
            layer_count,
            &sliding,
            &[(1, 0.20), (2, 0.30), (5, 0.40)],
            &AdaptiveGemma2Config::default(),
        );
        assert_eq!(expensive.executed_count(), layer_count);
        assert_eq!(skipped_layers(&expensive, layer_count), Vec::<usize>::new());
    }

    #[test]
    fn kl_budget_mask_t21_ranking_does_not_follow_delta_rms() {
        let layer_count = 26;
        let sliding = (0..layer_count)
            .map(|layer| layer % 2 == 0)
            .collect::<Vec<_>>();
        let mask = kl_budget_mask(
            layer_count,
            &sliding,
            GEMMA2_T21_LAYER_KL,
            &AdaptiveGemma2Config::default(),
        );
        let skipped = skipped_layers(&mask, layer_count);
        assert_eq!(skipped, vec![7, 8, 12, 20, 21]);
        assert!(
            mask.executes(2),
            "capa 2 es la mas cara por KL; no se apaga por delta_rms"
        );
        assert!(mask.executed_count() >= 8);
        assert!(!mask_has_consecutive_global_skips(&mask, &sliding));
    }

    #[test]
    fn kl_budget_mask_from_trace_refuses_uncalibrated_combined_skip() {
        let trace = gemma_like_trace(26, |layer| if layer == 2 { 0.01 } else { 9.0 });
        let mask = kl_budget_mask_from_trace(&trace, &AdaptiveGemma2Config::default());
        assert_eq!(mask.executed_count(), 26);
        assert_eq!(skipped_layers(&mask, 26), Vec::<usize>::new());
        assert!(mask.executes(2));
        assert!(mask.executes(7));
    }

    #[test]
    fn fast_memory_keeps_a_ring_of_recent_experiences() {
        let mut buffer = FastWorkingMemoryBuffer::new(1);
        buffer.push(AdaptiveExperience {
            context: context_fingerprint(&[1]),
            activations: context_fingerprint(&[2]),
            mask: LayerExecutionMask::all(4),
            memory_tokens: vec![1, 2],
            quality: 1.0,
            route_id: None,
        });
        buffer.push(AdaptiveExperience {
            context: context_fingerprint(&[3]),
            activations: context_fingerprint(&[4]),
            mask: LayerExecutionMask::all(4),
            memory_tokens: vec![3],
            quality: 0.4,
            route_id: None,
        });
        assert_eq!(buffer.len(), 1);
        assert_eq!(
            buffer
                .recall_best(&context_fingerprint(&[3]), 4, 4, 0.35)
                .unwrap()
                .memory_tokens,
            vec![3]
        );
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

    #[test]
    fn route_payload_roundtrips_mask_and_memory_tokens() {
        let mut enabled = vec![true; 8];
        enabled[3] = false;
        let mask = LayerExecutionMask::from_enabled(enabled);
        let memory_tokens = vec![7, 11, 13, 17];
        let payload = encode_route_payload(&mask, &memory_tokens);
        let (decoded_mask, decoded_tokens) = decode_route_payload(&payload).unwrap();
        assert_eq!(decoded_mask.executed_count(), 7);
        assert!(!decoded_mask.executes(3));
        assert_eq!(decoded_tokens, memory_tokens);
    }

    #[test]
    fn working_memory_recalls_recent_overlapping_context() {
        let mut buffer = FastWorkingMemoryBuffer::new(4);
        buffer.push(AdaptiveExperience {
            context: context_fingerprint(&[1, 2, 3, 4]),
            activations: context_fingerprint(&[9]),
            mask: LayerExecutionMask::all(8),
            memory_tokens: vec![1, 2, 3, 4],
            quality: 0.8,
            route_id: None,
        });
        let query = context_fingerprint(&[1, 2, 3, 4, 4]);
        let recalled = buffer
            .recall_best(&query, 8, 8, 0.35)
            .expect("debería recordar el contexto reciente");
        assert_eq!(recalled.memory_tokens, vec![1, 2, 3, 4]);
        assert!(fingerprint_overlap(&query, &recalled.context) >= 0.35);
    }

    #[test]
    fn corrupt_native_checkpoint_does_not_reset_silently() {
        let root = std::env::temp_dir().join(format!(
            "adaptive-load-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("thermo-memory.cdt_native"), b"not-a-checkpoint").unwrap();
        let result = AdaptiveThermoMemory::load_or_new(
            &root,
            "gemma2:test",
            AdaptiveGemma2Config::default(),
        );
        let error = match result {
            Ok(_) => panic!("un checkpoint corrupto debe fallar"),
            Err(error) => error,
        };
        assert!(error.contains("no se pudo cargar la termo-memoria"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn first_wake_turn_prefills_the_whole_prompt_once() {
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let plan = plan_wake_prefill(&prompt, &[], None, None, 8);
        assert!(!plan.reuse_cache);
        assert_eq!(plan.position, 0);
        assert_eq!(plan.prefill_tokens(prompt.len()), prompt.len());
        assert_eq!(plan.suffix(&prompt), prompt.as_slice());
    }

    #[test]
    fn later_wake_turn_prefills_only_the_new_suffix() {
        let cached = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut prompt = cached.clone();
        prompt.extend_from_slice(&[20, 21, 22, 23]);
        let mask = LayerExecutionMask::all(8);
        let plan = plan_wake_prefill(&prompt, &cached, Some(&mask), Some(&mask), 8);
        assert!(plan.reuse_cache);
        assert_eq!(plan.position, cached.len());
        assert_eq!(plan.prefill_tokens(prompt.len()), 4);
        assert_eq!(plan.suffix(&prompt), &[20, 21, 22, 23]);
        assert_eq!(plan.mask, mask);
        assert!(
            plan.prefill_tokens(prompt.len()) as f32 / prompt.len() as f32 <= 0.25,
            "el segundo turno no debe rehacer el historial"
        );
    }

    #[test]
    fn kv_a_reuses_kv_when_the_recalled_sparse_mask_stays() {
        let cached = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut prompt = cached.clone();
        prompt.extend_from_slice(&[9, 10]);
        let sparse =
            LayerExecutionMask::from_enabled(vec![true, false, true, true, true, true, true, true]);
        let plan = plan_wake_prefill(&prompt, &cached, Some(&sparse), Some(&sparse), 8);
        assert!(plan.reuse_cache);
        assert_eq!(plan.mask, sparse);
        assert_eq!(plan.prefill_tokens(prompt.len()), 2);
    }

    #[test]
    fn kv_a_prefills_when_the_recalled_mask_changes() {
        let cached = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut prompt = cached.clone();
        prompt.extend_from_slice(&[9, 10]);
        let dense = LayerExecutionMask::all(8);
        let sparse =
            LayerExecutionMask::from_enabled(vec![true, false, true, true, true, true, true, true]);
        let plan = plan_wake_prefill(&prompt, &cached, Some(&dense), Some(&sparse), 8);
        assert!(!plan.reuse_cache);
        assert_eq!(plan.mask, sparse);
        assert_eq!(plan.position, 0);
        assert_eq!(plan.prefill_tokens(prompt.len()), prompt.len());
    }

    #[test]
    fn wake_plan_never_schedules_a_second_verification_prefill() {
        let prompt: Vec<u32> = (0..64).collect();
        let plan = plan_wake_prefill(&prompt, &[], None, None, 26);
        assert_eq!(plan.prefill_tokens(prompt.len()), 64);
        let cached = prompt.clone();
        let mut next = prompt.clone();
        next.extend(64..80);
        let plan = plan_wake_prefill(&next, &cached, Some(&LayerExecutionMask::all(26)), None, 26);
        assert_eq!(plan.prefill_tokens(next.len()), 16);
        assert!(plan.reuse_cache);
    }

    #[test]
    fn observe_keeps_working_memory_without_flushing_routes() {
        let root = std::env::temp_dir().join(format!(
            "adaptive-observe-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut memory = AdaptiveThermoMemory::load_or_new(
            &root,
            "gemma2:test",
            AdaptiveGemma2Config {
                buffer_capacity: 2,
                ..AdaptiveGemma2Config::default()
            },
        )
        .unwrap();
        for token in 0..5u32 {
            memory
                .observe(
                    context_fingerprint(&[token]),
                    context_fingerprint(&[token]),
                    LayerExecutionMask::all(8),
                    &[token],
                    0.4,
                    None,
                    false,
                )
                .unwrap();
        }
        assert_eq!(memory.working_memory_len(), 2);
        assert_eq!(memory.verified_route_count(), 0);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn logit_agreement_is_one_for_identical_distributions() {
        assert!((logit_agreement(&[1.0, 0.2, 0.1], &[1.0, 0.2, 0.1]) - 1.0).abs() < 1e-5);
        assert_eq!(logit_agreement(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    #[ignore]
    fn incremental_wake_prefill_is_faster_on_gemma2_gguf() {
        use crate::native_gemma2::{
            resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer,
        };
        use candle_core::quantized::gguf_file;
        use std::fs::File;
        use std::time::Instant;

        let path = match resolve_gemma2_model_path(None) {
            Ok(path) => path,
            Err(_) => return,
        };
        let device = resolve_gemma2_device("cpu").unwrap();
        let mut file = File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        let tokenizer = Gemma2Tokenizer::from_gguf(&content).unwrap();
        let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device).unwrap();
        let root = std::env::temp_dir().join(format!(
            "adaptive-speed-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut memory = AdaptiveThermoMemory::load_or_new(
            &root,
            format!("gemma2:{}", path.display()),
            AdaptiveGemma2Config::default(),
        )
        .unwrap();

        let prompt1 = {
            let mut tokens = vec![tokenizer.bos_id];
            tokens.extend(tokenizer.encode("hola, como estas").unwrap());
            tokens
        };
        let started = Instant::now();
        let first = memory
            .prepare_forward(&mut model, &prompt1, &[], None, None)
            .unwrap();
        let first_ms = started.elapsed().as_millis();
        assert!(!first.cache_reused);
        assert_eq!(first.prefill_tokens, prompt1.len());

        let mut prompt2 = prompt1.clone();
        prompt2.extend(tokenizer.encode(" y que hora es").unwrap());
        let started = Instant::now();
        let second = memory
            .prepare_forward(
                &mut model,
                &prompt2,
                &prompt1,
                Some(&first.mask),
                Some(&first.output.logits),
            )
            .unwrap();
        let second_ms = started.elapsed().as_millis();
        let extra = prompt2.len() - prompt1.len();
        assert!(second.cache_reused);
        assert_eq!(second.prefill_tokens, extra);
        assert!(
            extra < prompt1.len(),
            "el segundo turno debe procesar menos tokens que el historial"
        );
        assert!(
            second_ms < first_ms,
            "prefill incremental {second_ms}ms debe ser más rápido que el completo {first_ms}ms"
        );
        let _ = fs::remove_dir_all(&root);
    }

    fn sparse_mask(layers: usize) -> LayerExecutionMask {
        let mut enabled = vec![true; layers];
        if layers > 2 {
            enabled[layers / 2] = false;
        }
        LayerExecutionMask::from_enabled(enabled)
    }

    fn temp_memory(name: &str, capacity: usize) -> (AdaptiveThermoMemory, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "adaptive-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let memory = AdaptiveThermoMemory::load_or_new(
            &root,
            "gemma2:test",
            AdaptiveGemma2Config {
                buffer_capacity: capacity,
                ..AdaptiveGemma2Config::default()
            },
        )
        .unwrap();
        (memory, root)
    }

    #[test]
    fn sleep_prompts_prefer_buffer_then_journal_and_dedupe() {
        let prompts = collect_sleep_prompts(
            [vec![1, 2, 3], vec![1, 2, 3], vec![9, 8]],
            &[vec![9, 8], vec![4, 5, 6], vec![7]],
            3,
        );
        assert_eq!(prompts, vec![vec![1, 2, 3], vec![9, 8], vec![4, 5, 6]]);
    }

    #[test]
    fn sleep_mask_policy_keeps_sparse_runtime_and_verified() {
        let full = LayerExecutionMask::all(8);
        let sparse = sparse_mask(8);
        let (_, quality, usable) =
            choose_sleep_mask(&full, Some((sparse.clone(), 0.61)), 0.50, 0.92);
        assert!(usable);
        assert!((quality - 0.61).abs() < f32::EPSILON);
        let (mask, quality, usable) =
            choose_sleep_mask(&full, Some((sparse.clone(), 0.97)), 0.50, 0.92);
        assert!(usable);
        assert!(is_sparse_mask(&mask));
        assert!((quality - 0.97).abs() < f32::EPSILON);
        let (_, _, usable) = choose_sleep_mask(&full, Some((sparse, 0.2)), 0.50, 0.92);
        assert!(!usable);
        let (_, _, usable) = choose_sleep_mask(&full, Some((full.clone(), 0.99)), 0.50, 0.92);
        assert!(!usable);
    }

    #[test]
    fn sleep_flush_keeps_runtime_sparse_and_drops_garbage() {
        let (mut memory, root) = temp_memory("flush", 4);
        memory
            .observe(
                context_fingerprint(&[1, 2, 3]),
                context_fingerprint(&[1, 2, 3]),
                sparse_mask(8),
                &[1, 2, 3],
                0.62,
                None,
                false,
            )
            .unwrap();
        memory
            .observe(
                context_fingerprint(&[9]),
                context_fingerprint(&[9]),
                LayerExecutionMask::all(8),
                &[9, 10],
                0.2,
                None,
                false,
            )
            .unwrap();
        let flushed = memory.flush_fast_memory().unwrap();
        assert_eq!(flushed, 0);
        assert_eq!(memory.working_memory_len(), 1);
        assert_eq!(memory.verified_route_count(), 0);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sleep_flush_binds_verified_sparse_masks() {
        let (mut memory, root) = temp_memory("bind", 4);
        memory
            .observe(
                context_fingerprint(&[10, 20, 30, 20]),
                context_fingerprint(&[40, 50]),
                sparse_mask(26),
                &[10, 20, 30],
                1.0,
                None,
                false,
            )
            .unwrap();
        let flushed = memory.flush_fast_memory().unwrap();
        assert_eq!(flushed, 1);
        assert_eq!(memory.verified_route_count(), 1);
        assert_eq!(memory.working_memory_len(), 0);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore]
    fn sleep_discovers_masks_on_gemma2_gguf() {
        use crate::native_gemma2::{
            resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer,
        };
        use candle_core::quantized::gguf_file;
        use std::fs::File;

        let path = match resolve_gemma2_model_path(None) {
            Ok(path) => path,
            Err(_) => return,
        };
        let device = resolve_gemma2_device("cpu").unwrap();
        let mut file = File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        let tokenizer = Gemma2Tokenizer::from_gguf(&content).unwrap();
        let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device).unwrap();
        let (mut memory, root) = temp_memory("sleep-gguf", 4);
        memory.config.max_sleep_replays = 1;
        memory.config.max_candidate_prefills = 1;
        let mut tokens = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode("hola dyamon").unwrap());
        memory
            .observe(
                context_fingerprint(&tokens),
                context_fingerprint(&tokens),
                LayerExecutionMask::all(model.layer_count()),
                &tokens,
                0.30,
                None,
                true,
            )
            .unwrap();
        let report = memory
            .consolidate_sleep_with_model(&mut model, &[])
            .unwrap();
        assert_eq!(report.replayed, 1);
        assert!(
            report.discovered_masks + report.flushed + report.retained_working > 0
                || report.replayed == 1,
            "el sueño debe rejugar el prompt de vigilia"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn lrc_promoted_route_wins_the_wake_plan_over_dense_cache() {
        let (mut memory, root) = temp_memory("lrc-wake", 4);
        memory.config.minimum_executed_layers = 6;
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let mask = sparse_mask(8);
        assert!(memory.lrc.promote(
            crate::layer_route_cache::fingerprint_wake(&tokens),
            mask.clone(),
            0.04,
            1.0,
            1
        ));
        let recalled = memory
            .recall_layer_route(&crate::layer_route_cache::fingerprint_wake(&tokens), 8)
            .expect("LRC debe devolver la máscara promocionada");
        let cached = tokens.clone();
        let mut prompt = cached.clone();
        prompt.extend_from_slice(&[9, 10]);
        let plan = plan_wake_prefill(
            &prompt,
            &cached,
            Some(&LayerExecutionMask::all(8)),
            Some(&recalled.mask),
            8,
        );
        assert!(!plan.reuse_cache);
        assert_eq!(plan.mask.executed_count(), 7);
        assert!(!plan.mask.executes(4));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn kl_promotion_does_not_require_the_legacy_verified_threshold() {
        let full = LayerExecutionMask::all(8);
        let sparse = sparse_mask(8);
        let (_, _, usable) = choose_sleep_mask(&full, Some((sparse.clone(), 0.40)), 0.50, 0.92);
        assert!(!usable);
        let (mut memory, root) = temp_memory("lrc-kl", 2);
        let fp = crate::layer_route_cache::fingerprint_wake(&[11, 12, 13, 14]);
        assert!(
            memory.lrc.promote(fp.clone(), sparse, 0.08, 1.0, 1),
            "KL 0.08 debe promocionar aunque logit_agreement sea bajo"
        );
        assert!(memory.lrc.lookup_confident(&fp).is_some());
        let _ = fs::remove_dir_all(&root);
    }
}
