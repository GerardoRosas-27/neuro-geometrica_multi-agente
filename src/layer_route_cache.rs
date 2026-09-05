//! Tabla de rutas de capas (LRC): huella barata → máscara, sin tocar pesos.
//!
//! El lookup no hace matmul. La calidad (KL vs denso) se mide en sueño.

use crate::native_checkpoint::atomic_write;
use crate::native_gemma2::LayerExecutionMask;
use crate::thermo_router::ActivationFingerprint;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::Path;

pub const LAYER_ROUTES_FILE: &str = "layer-routes.json";
const STATE_VERSION: u32 = 1;
const WAKE_TURN_WINDOW: usize = 48;
const WAKE_HASH_WINDOW: usize = 32;
const WAKE_TOP_K: usize = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkipKind {
    #[default]
    MiddleSkip,
    EarlyExit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerRoute {
    pub id: u64,
    pub fingerprint: ActivationFingerprint,
    pub mask: LayerExecutionMask,
    pub skip_kind: SkipKind,
    pub hits: u64,
    pub misses_as_fallback: u64,
    pub mean_kl: f32,
    pub top1_agree: f32,
    pub mean_executed: u8,
    pub last_generation: u64,
    pub confidence: f32,
}

impl LayerRoute {
    /// k de early-exit si la ruta es cola; None en middle-skip.
    pub fn exit_after(&self) -> Option<usize> {
        match self.skip_kind {
            SkipKind::EarlyExit => exit_after_from_mask(&self.mask),
            SkipKind::MiddleSkip => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerRouteCacheConfig {
    pub min_confidence: f32,
    pub max_kl_promote: f32,
    pub min_overlap: f32,
    pub max_routes: usize,
}

impl Default for LayerRouteCacheConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.55,
            max_kl_promote: 0.15,
            min_overlap: 0.35,
            max_routes: 2_048,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LayerRouteCacheState {
    version: u32,
    next_id: u64,
    config: LayerRouteCacheConfig,
    routes: Vec<LayerRoute>,
}

#[derive(Clone, Debug)]
pub struct LayerRouteCache {
    next_id: u64,
    config: LayerRouteCacheConfig,
    routes: Vec<LayerRoute>,
}

impl Default for LayerRouteCache {
    fn default() -> Self {
        Self::new(LayerRouteCacheConfig::default())
    }
}

impl LayerRouteCache {
    pub fn new(config: LayerRouteCacheConfig) -> Self {
        Self {
            next_id: 1,
            config,
            routes: Vec::new(),
        }
    }

    pub fn load_or_new(path: impl AsRef<Path>, config: LayerRouteCacheConfig) -> Self {
        let path = path.as_ref();
        let Ok(body) = std::fs::read(path) else {
            return Self::new(config);
        };
        let Ok(state) = serde_json::from_slice::<LayerRouteCacheState>(&body) else {
            return Self::new(config);
        };
        if state.version != STATE_VERSION {
            return Self::new(config);
        }
        Self {
            next_id: state.next_id.max(1),
            config: state.config,
            routes: state.routes,
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let state = LayerRouteCacheState {
            version: STATE_VERSION,
            next_id: self.next_id,
            config: self.config.clone(),
            routes: self.routes.clone(),
        };
        let body = serde_json::to_vec_pretty(&state).map_err(|error| error.to_string())?;
        atomic_write(path.as_ref(), &body)
    }

    pub fn config(&self) -> &LayerRouteCacheConfig {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.routes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    pub fn routes(&self) -> &[LayerRoute] {
        &self.routes
    }

    /// Mejor ruta por overlap. No filtra por confianza.
    pub fn lookup(&self, fingerprint: &ActivationFingerprint) -> Option<&LayerRoute> {
        self.best_index(fingerprint)
            .map(|index| &self.routes[index])
    }

    /// Hit usable en vigilia: overlap y confianza.
    pub fn lookup_confident(&self, fingerprint: &ActivationFingerprint) -> Option<&LayerRoute> {
        let route = self.lookup(fingerprint)?;
        (route.confidence + 1.0e-6 >= self.config.min_confidence).then_some(route)
    }

    pub fn observe_turn(
        &mut self,
        fingerprint: &ActivationFingerprint,
        mask: &LayerExecutionMask,
        fallback: bool,
        generation: u64,
    ) {
        let Some(index) = self.best_index(fingerprint) else {
            return;
        };
        if &self.routes[index].mask != mask {
            return;
        }
        if fallback {
            self.routes[index].misses_as_fallback =
                self.routes[index].misses_as_fallback.saturating_add(1);
        } else {
            self.routes[index].hits = self.routes[index].hits.saturating_add(1);
        }
        self.routes[index].last_generation = generation;
        self.recompute_confidence(index);
    }

    /// Promueve una máscara sparse de middle-skip si KL ≤ umbral.
    pub fn promote(
        &mut self,
        fingerprint: ActivationFingerprint,
        mask: LayerExecutionMask,
        kl: f32,
        top1_agree: f32,
        generation: u64,
    ) -> bool {
        self.promote_kind(
            fingerprint,
            mask,
            kl,
            top1_agree,
            generation,
            SkipKind::MiddleSkip,
        )
    }

    /// Promueve una ruta LRC. Early-exit exige máscara prefijo (cola), no agujeros.
    pub fn promote_kind(
        &mut self,
        fingerprint: ActivationFingerprint,
        mask: LayerExecutionMask,
        kl: f32,
        top1_agree: f32,
        generation: u64,
        skip_kind: SkipKind,
    ) -> bool {
        if !kl.is_finite() || kl > self.config.max_kl_promote {
            return false;
        }
        if !is_sparse_mask(&mask) {
            return false;
        }
        if skip_kind == SkipKind::EarlyExit && !is_prefix_mask(&mask) {
            return false;
        }
        if let Some(index) = self.best_index(&fingerprint).filter(|&index| {
            self.routes[index].mask == mask && self.routes[index].skip_kind == skip_kind
        }) {
            let route = &mut self.routes[index];
            route.mean_kl = if route.hits + route.misses_as_fallback == 0 {
                kl
            } else {
                0.7 * route.mean_kl + 0.3 * kl
            };
            route.top1_agree = 0.7 * route.top1_agree + 0.3 * top1_agree.clamp(0.0, 1.0);
            route.mean_executed = mask.executed_count().min(u8::MAX as usize) as u8;
            route.last_generation = generation;
            route.hits = route.hits.saturating_add(1);
            self.recompute_confidence(index);
            return true;
        }
        if self.routes.len() >= self.config.max_routes.max(1) {
            let victim = self
                .routes
                .iter()
                .enumerate()
                .min_by(|left, right| {
                    left.1
                        .confidence
                        .total_cmp(&right.1.confidence)
                        .then(left.1.last_generation.cmp(&right.1.last_generation))
                })
                .map(|(index, _)| index);
            if let Some(index) = victim {
                self.routes.swap_remove(index);
            }
        }
        let executed = mask.executed_count().min(u8::MAX as usize) as u8;
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.routes.push(LayerRoute {
            id,
            fingerprint,
            mask,
            skip_kind,
            hits: 1,
            misses_as_fallback: 0,
            mean_kl: kl,
            top1_agree: top1_agree.clamp(0.0, 1.0),
            mean_executed: executed,
            last_generation: generation,
            confidence: 0.60,
        });
        true
    }

    fn best_index(&self, fingerprint: &ActivationFingerprint) -> Option<usize> {
        let mut best = None::<(usize, f32)>;
        for (index, route) in self.routes.iter().enumerate() {
            let overlap = fingerprint_overlap(fingerprint, &route.fingerprint);
            if overlap + 1.0e-6 < self.config.min_overlap {
                continue;
            }
            if best.map(|(_, score)| overlap > score).unwrap_or(true) {
                best = Some((index, overlap));
            }
        }
        best.map(|(index, _)| index)
    }

    fn recompute_confidence(&mut self, index: usize) {
        let route = &self.routes[index];
        let trials = route.hits + route.misses_as_fallback;
        let posterior = (route.hits as f32 + 1.0) / (trials as f32 + 2.0);
        let kl_bonus = if route.mean_kl <= self.config.max_kl_promote {
            0.10
        } else {
            0.0
        };
        self.routes[index].confidence = (posterior + kl_bonus).clamp(0.0, 1.0);
    }
}

/// Huella de vigilia: cola del prompt (el turno de usuario vive al final
/// de la plantilla de chat). Cero matmul.
pub fn fingerprint_wake(prompt_tokens: &[u32]) -> ActivationFingerprint {
    let hash_tail = &prompt_tokens[prompt_tokens.len().saturating_sub(WAKE_HASH_WINDOW)..];
    let turn = &prompt_tokens[prompt_tokens.len().saturating_sub(WAKE_TURN_WINDOW)..];
    let mut counts = std::collections::HashMap::<u32, usize>::new();
    for token in turn {
        *counts.entry(*token).or_default() += 1;
    }
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
    ranked.truncate(WAKE_TOP_K);
    let denominator = turn.len().max(1) as f32;
    let mut entries = ranked
        .into_iter()
        .map(|(token, count)| (token, count as f32 / denominator))
        .collect::<Vec<_>>();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_tail.hash(&mut hasher);
    turn.len().hash(&mut hasher);
    entries.push(((hasher.finish() & u32::MAX as u64) as u32, 1.0));
    let diversity = turn.iter().copied().collect::<HashSet<_>>().len() as f32 / denominator;
    ActivationFingerprint {
        entries,
        confidence: if turn.is_empty() { 0.0 } else { 0.85 },
        entropy: diversity.clamp(0.0, 1.0),
    }
}

pub fn fingerprint_overlap(query: &ActivationFingerprint, stored: &ActivationFingerprint) -> f32 {
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

pub fn is_sparse_mask(mask: &LayerExecutionMask) -> bool {
    mask.executed_count() < mask.layer_count() && mask.executed_count() > 0
}

/// Prefijo `0..=k` encendido y cola apagada. La máscara densa también es prefijo.
pub fn is_prefix_mask(mask: &LayerExecutionMask) -> bool {
    let mut seen_off = false;
    for layer in 0..mask.layer_count() {
        if mask.executes(layer) {
            if seen_off {
                return false;
            }
        } else {
            seen_off = true;
        }
    }
    true
}

pub fn exit_after_from_mask(mask: &LayerExecutionMask) -> Option<usize> {
    if !is_prefix_mask(mask) || mask.executed_count() == 0 {
        return None;
    }
    Some(mask.executed_count() - 1)
}

/// KL(softmax(dense) || softmax(sparse)) en la última posición.
pub fn logits_kl(dense: &[f32], sparse: &[f32]) -> f32 {
    if dense.len() != sparse.len() || dense.is_empty() {
        return f32::INFINITY;
    }
    let log_p = log_softmax(dense);
    let log_q = log_softmax(sparse);
    let mut kl = 0.0f64;
    for (log_p, log_q) in log_p.iter().zip(&log_q) {
        let p = log_p.exp();
        kl += p * (log_p - log_q);
    }
    kl.max(0.0) as f32
}

pub fn top1_index(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
}

pub fn top1_agree(dense: &[f32], sparse: &[f32]) -> f32 {
    f32::from(top1_index(dense) == top1_index(sparse))
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_z = logits
        .iter()
        .map(|&value| (value as f64 - max).exp())
        .sum::<f64>()
        .max(f64::EPSILON)
        .ln();
    logits
        .iter()
        .map(|&value| value as f64 - max - log_z)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_gemma2::early_exit_mask;

    fn sparse(layers: usize) -> LayerExecutionMask {
        let mut enabled = vec![true; layers];
        if layers > 2 {
            enabled[layers / 2] = false;
        }
        LayerExecutionMask::from_enabled(enabled)
    }

    #[test]
    fn fingerprint_wake_is_deterministic_and_ignores_a_long_prefix() {
        let mut long = vec![9u32; 200];
        long.extend_from_slice(&[1, 2, 3, 4, 5]);
        let mut also = vec![8u32; 200];
        also.extend_from_slice(&[1, 2, 3, 4, 5]);
        let left = fingerprint_wake(&long);
        let right = fingerprint_wake(&long);
        assert_eq!(left.entries, right.entries);
        let overlap = fingerprint_overlap(&fingerprint_wake(&long), &fingerprint_wake(&also));
        assert!(
            overlap >= 0.35,
            "la cola del turno debe dominar la huella: {overlap}"
        );
    }

    #[test]
    fn lookup_requires_overlap_and_confidence() {
        let mut cache = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let fp = fingerprint_wake(&[1, 2, 3, 4, 5, 6]);
        assert!(cache.lookup_confident(&fp).is_none());
        assert!(cache.promote(fp.clone(), sparse(8), 0.05, 1.0, 1));
        assert_eq!(cache.len(), 1);
        let hit = cache.lookup_confident(&fp).expect("ruta promocionada");
        assert_eq!(hit.mask.executed_count(), 7);
        assert!(hit.confidence >= 0.55);
        let other = fingerprint_wake(&[90, 91, 92, 93, 94]);
        assert!(cache.lookup_confident(&other).is_none());
    }

    #[test]
    fn promote_rejects_high_kl_and_dense_masks() {
        let mut cache = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let fp = fingerprint_wake(&[1, 2, 3]);
        assert!(!cache.promote(fp.clone(), sparse(8), 0.40, 1.0, 1));
        assert!(!cache.promote(fp, LayerExecutionMask::all(8), 0.01, 1.0, 1));
        assert!(cache.is_empty());
    }

    #[test]
    fn observe_turn_updates_hits_on_matching_mask() {
        let mut cache = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let fp = fingerprint_wake(&[3, 1, 4, 1, 5]);
        let mask = sparse(8);
        assert!(cache.promote(fp.clone(), mask.clone(), 0.04, 1.0, 1));
        cache.observe_turn(&fp, &mask, false, 2);
        cache.observe_turn(&fp, &mask, false, 3);
        let route = cache.lookup(&fp).unwrap();
        assert_eq!(route.hits, 3);
        cache.observe_turn(&fp, &LayerExecutionMask::all(8), true, 4);
        let route = cache.lookup(&fp).unwrap();
        assert_eq!(route.misses_as_fallback, 0);
    }

    #[test]
    fn roundtrip_json_preserves_a_promoted_route() {
        let dir = std::env::temp_dir().join(format!(
            "lrc-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(LAYER_ROUTES_FILE);
        let mut cache = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let fp = fingerprint_wake(&[7, 8, 9, 10]);
        cache.promote(fp.clone(), sparse(26), 0.02, 1.0, 9);
        cache.save(&path).unwrap();
        let loaded = LayerRouteCache::load_or_new(&path, LayerRouteCacheConfig::default());
        assert_eq!(loaded.len(), 1);
        assert!(loaded.lookup_confident(&fp).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn promote_early_exit_stores_kind_and_rejects_middle_holes() {
        let mut cache = LayerRouteCache::new(LayerRouteCacheConfig::default());
        let fp = fingerprint_wake(&[1, 2, 3, 4]);
        let prefix = early_exit_mask(26, 20);
        assert!(is_prefix_mask(&prefix));
        assert_eq!(exit_after_from_mask(&prefix), Some(20));
        assert!(cache.promote_kind(
            fp.clone(),
            prefix.clone(),
            0.05,
            1.0,
            1,
            SkipKind::EarlyExit
        ));
        let route = cache.lookup_confident(&fp).expect("ruta early-exit");
        assert_eq!(route.skip_kind, SkipKind::EarlyExit);
        assert_eq!(route.exit_after(), Some(20));
        assert_eq!(route.mask.executed_count(), 21);

        let mut hole = vec![true; 26];
        hole[7] = false;
        let middle = LayerExecutionMask::from_enabled(hole);
        assert!(!is_prefix_mask(&middle));
        let mut other = LayerRouteCache::new(LayerRouteCacheConfig::default());
        assert!(
            !other.promote_kind(fp, middle, 0.05, 1.0, 1, SkipKind::EarlyExit),
            "no mezclar middle-skip y early-exit"
        );
        assert!(other.is_empty());
    }

    #[test]
    fn identical_logits_have_zero_kl_and_agree_on_top1() {
        let dense = [1.0f32, 0.2, -0.4, 0.0];
        assert!(logits_kl(&dense, &dense) < 1.0e-6);
        assert_eq!(top1_agree(&dense, &dense), 1.0);
        let shifted = [3.0f32, 2.2, 1.6, 2.0];
        assert!(
            logits_kl(&dense, &shifted) < 1.0e-5,
            "softmax es invariante a un shift"
        );
    }

    #[test]
    fn peaked_versus_uniform_has_positive_kl() {
        let peaked = [5.0f32, 0.0, 0.0, 0.0];
        let flat = [0.0f32, 0.0, 0.0, 0.0];
        let kl = logits_kl(&peaked, &flat);
        assert!(kl > 0.5, "KL={kl}");
        assert_eq!(top1_agree(&peaked, &peaked), 1.0);
        assert_eq!(top1_agree(&peaked, &[0.0, 5.0, 0.0, 0.0]), 0.0);
    }
}
