//! Arquitectura de **dos secciones** (fuente de verdad del POC de inferencia):
//!
//! 1. **Líquido** (`LiquidInfer` / `WavePredictCore`) — **toda** la inferencia
//!    (cue frío y cue caliente). Nunca llama a RQM ni a `NativeThermoCdtSubstrate::step`
//!    en el camino caliente de query.
//! 2. **Thermo CDT** (`CdtConsolidatedMemory`) — memoria dinámica consolidada
//!    **tras el sueño**. Engramas incrementales sin borrar los anteriores.
//!
//! **RQM** (`NativeThermoRqmEprSubstrate`) vive **solo dentro de**
//! `LiquidCdtSystem::sleep_consolidate`: destila la relación cue→label hacia
//! la memoria termodinámica durante el sueño. No es router de inferencia.
//!
//! Ver `docs/arquitectura_liquido_cdt_memoria.md`.

use crate::entanglement::EntanglementConfig;
use crate::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use crate::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use crate::relational_field::ObserverId;
use crate::wave_predict_core::{PredictionReport, WavePredictCore};
use std::collections::HashMap;
use std::f32::consts::TAU;
use std::time::Instant;

const OBSERVER: ObserverId = ObserverId(0x51EE_0001);
const CUE_BASE: usize = 64;
const PILOT_AMP: f32 = 1.2;
const THERMO_STEPS: usize = 8;
const DEFAULT_CAPACITY: usize = 16;
const DEFAULT_SCORE_THRESHOLD: f64 = 0.2;

// ─── Sección 1: inferencia líquida ──────────────────────────────────────────

/// Informe de una query líquida (única vía de inferencia).
#[derive(Clone, Debug)]
pub struct LiquidReport {
    pub observation: usize,
    pub predicted: usize,
    pub score: f64,
    pub candidates_scored: usize,
}

/// Sección 1 — inferencia: solo `WavePredictCore`.
#[derive(Clone, Debug, Default)]
pub struct LiquidInfer {
    pub core: WavePredictCore,
}

impl LiquidInfer {
    pub fn new() -> Self {
        Self {
            core: WavePredictCore::new(),
        }
    }

    /// Inferencia pura por interferencia pasado×futuro. No toca CDT ni RQM.
    pub fn infer(&mut self, observation: usize, candidates: &[usize]) -> LiquidReport {
        let pred: PredictionReport = self.core.predict_from_observation(observation, candidates);
        LiquidReport {
            observation,
            predicted: pred.best_content,
            score: pred.best_score,
            candidates_scored: pred.scores,
        }
    }
}

// ─── Sección 2: memoria CDT consolidada ─────────────────────────────────────

/// Plantilla de engrama: nodos disjuntos + firma amp/fase.
#[derive(Clone, Debug)]
pub struct Engram {
    pub concept: usize,
    pub nodes: Vec<usize>,
    /// Concat(amp, cos φ, sin φ) de los nodos propios.
    pub template: Vec<f64>,
}

/// Sección 2 — memoria dinámica consolidada en `NativeThermoCdtSubstrate`.
///
/// Aprende engramas de forma **incremental**: no reconstruye el sustrato
/// al aprender un concepto nuevo; solo actualiza nodos / entrada del mapa.
#[derive(Clone, Debug)]
pub struct CdtConsolidatedMemory {
    pub substrate: NativeThermoCdtSubstrate,
    pub engrams: HashMap<usize, Engram>,
    /// Capacidad máxima de conceptos (partición disjunta de nodos).
    pub capacity: usize,
    /// Contador de `step()` del sustrato (para tests: infer no debe incrementarlo).
    pub step_count: u64,
}

impl CdtConsolidatedMemory {
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(8);
        let nodes_per_slice = 64usize;
        let config = NativeThermoCdtConfig {
            slices: 2,
            nodes_per_slice,
            spatial_degree: 3,
            temporal_degree: 1,
            temperature: 0.2,
            seed: 0xCD7C_E100,
            ..NativeThermoCdtConfig::default()
        };
        Self {
            substrate: NativeThermoCdtSubstrate::new(config),
            engrams: HashMap::new(),
            capacity,
            step_count: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn engram_count(&self) -> usize {
        self.engrams.len()
    }

    pub fn substrate_tick(&self) -> u64 {
        self.substrate.tick()
    }

    fn concept_nodes(&self, concept: usize) -> Vec<usize> {
        let nc = self.substrate.node_count();
        let n_concepts = self.capacity;
        let per = (nc / n_concepts).max(1);
        let c = concept % n_concepts;
        let start = c * per;
        let end = if c == n_concepts - 1 {
            nc
        } else {
            (start + per).min(nc)
        };
        (start..end).collect()
    }

    fn concept_phase(concept: usize, capacity: usize) -> f32 {
        (concept % capacity) as f32 * (TAU / capacity as f32)
    }

    /// Limpia solo los nodos del concepto (protege el resto del mapa de engramas).
    fn clear_concept_nodes(&mut self, nodes: &[usize]) {
        for &i in nodes {
            if i < self.substrate.node_count() {
                self.substrate.amplitude[i] = 0.5;
                self.substrate.phase[i] = 0.0;
                self.substrate.thermal_state[i] = 0.0;
                self.substrate.pilot_force[i] = 0.0;
                self.substrate.activation[i] = 0.0;
            }
        }
    }

    fn snapshot_nodes(&self, nodes: &[usize]) -> Vec<f64> {
        let mut v = Vec::with_capacity(nodes.len() * 3);
        for &i in nodes {
            let a = self.substrate.amplitude[i] as f64;
            let p = self.substrate.phase[i] as f64;
            v.push(a);
            v.push(p.cos());
            v.push(p.sin());
        }
        v
    }

    fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for i in 0..n {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
        dot / denom
    }

    /// Aprendizaje incremental: inyecta piloto + pasos + snapshot **solo** del
    /// concepto. No borra engramas previos ni reconstruye el sustrato.
    pub fn encode_engram(&mut self, concept: usize) {
        let concept = concept % self.capacity;
        let nodes = self.concept_nodes(concept);
        self.clear_concept_nodes(&nodes);
        let phase = Self::concept_phase(concept, self.capacity);
        self.substrate
            .inject_pilot_pattern(&nodes, PILOT_AMP, phase);
        for _ in 0..THERMO_STEPS {
            let _ = self.substrate.step();
            self.step_count = self.step_count.wrapping_add(1);
        }
        let template = self.snapshot_nodes(&nodes);
        self.engrams.insert(
            concept,
            Engram {
                concept,
                nodes,
                template,
            },
        );
    }

    /// Soft-recall opcional (tests / introspección). **No** es la inferencia primaria.
    /// Compara la firma del cue contra plantillas almacenadas (sin destruir engramas).
    pub fn recall(&mut self, cue: usize) -> Option<(usize, f64)> {
        if self.engrams.is_empty() {
            return None;
        }
        let cue = cue % self.capacity;
        let nodes = self.concept_nodes(cue);
        self.clear_concept_nodes(&nodes);
        let phase = Self::concept_phase(cue, self.capacity);
        self.substrate
            .inject_pilot_pattern(&nodes, PILOT_AMP, phase);
        for _ in 0..THERMO_STEPS {
            let _ = self.substrate.step();
            self.step_count = self.step_count.wrapping_add(1);
        }
        let feat = self.snapshot_nodes(&nodes);
        let mut best_c = None;
        let mut best_s = f64::NEG_INFINITY;
        for (&c, eng) in &self.engrams {
            let s = Self::cosine_sim(&feat, &eng.template);
            if s > best_s {
                best_s = s;
                best_c = Some(c);
            }
        }
        best_c.map(|c| (c, best_s))
    }
}

impl Default for CdtConsolidatedMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Orquestador + sueño ────────────────────────────────────────────────────

/// Origen del episodio en el buffer de vigilia.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum EpisodeKind {
    /// Predicción líquida observada (identidad / score alto).
    #[default]
    Liquid,
    /// Relación arbitraria enseñada (`teach_relation` / índice RQM).
    Relation,
}

#[derive(Clone, Copy, Debug)]
pub struct Episode {
    pub obs: usize,
    pub predicted: usize,
    pub score: f64,
    pub kind: EpisodeKind,
}

#[derive(Clone, Debug)]
pub struct SystemReport {
    pub liquid: LiquidReport,
    /// Siempre false en queries: la inferencia no usa memoria CDT ni RQM.
    pub used_memory: bool,
    pub used_rqm: bool,
}

#[derive(Clone, Debug)]
pub struct SleepReport {
    pub episodes_consolidated: usize,
    pub engrams_before: usize,
    pub engrams_after: usize,
    pub used_rqm: bool,
    pub rqm_relations_trained: usize,
    pub sleep_ms: f64,
}

/// Orquestador: líquido (query) + CDT (memoria) + RQM solo en sueño.
pub struct LiquidCdtSystem {
    pub liquid: LiquidInfer,
    pub memory: CdtConsolidatedMemory,
    /// Pegamento relacional; **solo** se escribe en `sleep_consolidate`.
    rqm: NativeThermoRqmEprSubstrate,
    /// Episodios pendientes de consolidación: (obs, predicted, score).
    pub wake_buffer: Vec<Episode>,
    pub score_threshold: f64,
    pub num_labels: usize,
    /// Contador de llamadas a la API RQM (debe crecer solo en sueño).
    pub rqm_api_calls: u64,
}

impl LiquidCdtSystem {
    pub fn new(num_labels: usize) -> Self {
        let num_labels = num_labels.max(2);
        let capacity = num_labels.max(DEFAULT_CAPACITY);
        let thermal = NativeThermoCdtConfig {
            slices: 2,
            nodes_per_slice: (CUE_BASE + num_labels).max(128),
            seed: 0x51EE_F100,
            ..NativeThermoCdtConfig::default()
        };
        let mut rqm_cfg = NativeThermoRqmConfig::default();
        rqm_cfg.max_candidates = num_labels * 2;
        rqm_cfg.thermal_steps_per_train = 1;
        rqm_cfg.thermal_steps_per_query = 2;
        rqm_cfg.thermal_activation_margin = 0.05;
        let epr = EntanglementConfig {
            create_threshold: 0.4,
            max_syncs_per_step: 64,
            ..EntanglementConfig::default()
        };
        Self {
            liquid: LiquidInfer::new(),
            memory: CdtConsolidatedMemory::with_capacity(capacity),
            rqm: NativeThermoRqmEprSubstrate::new(thermal, rqm_cfg, epr),
            wake_buffer: Vec::new(),
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            num_labels,
            rqm_api_calls: 0,
        }
    }

    /// Inferencia del sistema: **solo líquido**. No toca CDT ni RQM.
    pub fn infer(&mut self, obs: usize, candidates: &[usize]) -> SystemReport {
        let liquid = self.liquid.infer(obs % self.num_labels.max(1), candidates);
        SystemReport {
            liquid,
            used_memory: false,
            used_rqm: false,
        }
    }

    /// Infer + buffer si el score supera el umbral (candidatos a sueño).
    pub fn observe_and_buffer(&mut self, obs: usize, candidates: &[usize]) -> SystemReport {
        let report = self.infer(obs, candidates);
        if report.liquid.score >= self.score_threshold {
            self.wake_buffer.push(Episode {
                obs: report.liquid.observation,
                predicted: report.liquid.predicted,
                score: report.liquid.score,
                kind: EpisodeKind::Liquid,
            });
        }
        report
    }

    /// Sueño: consolida buffer → memoria CDT; RQM distila cue→label **solo aquí**.
    pub fn sleep_consolidate(&mut self) -> SleepReport {
        let t0 = Instant::now();
        let engrams_before = self.memory.engram_count();
        let mut consolidated = 0usize;
        let mut rqm_trained = 0usize;
        let episodes: Vec<Episode> = std::mem::take(&mut self.wake_buffer);

        for ep in episodes {
            if ep.score < self.score_threshold {
                continue;
            }
            // 1) Engrama durable en Thermo CDT (incremental, sin wipe).
            self.memory.encode_engram(ep.predicted);
            // 2) Pegamento relacional RQM — únicamente durante el sueño.
            let cue = CUE_BASE + (ep.obs % self.num_labels);
            let label = ep.predicted % self.num_labels;
            self.rqm
                .train_observed_transition(OBSERVER, 0.0, &[cue], &[label], 0.95);
            self.rqm_api_calls = self.rqm_api_calls.wrapping_add(1);
            rqm_trained += 1;
            consolidated += 1;
        }

        SleepReport {
            episodes_consolidated: consolidated,
            engrams_before,
            engrams_after: self.memory.engram_count(),
            used_rqm: rqm_trained > 0,
            rqm_relations_trained: rqm_trained,
            sleep_ms: t0.elapsed().as_secs_f64() * 1e3,
        }
    }

    /// Contador de llamadas a la API RQM (solo debe crecer en sueño).
    pub fn rqm_api_calls(&self) -> u64 {
        self.rqm_api_calls
    }

    /// Número de engramas consolidados en Thermo CDT.
    pub fn engram_count(&self) -> usize {
        self.memory.engram_count()
    }

    /// Tick del sustrato CDT (inferencia no debe avanzarlo).
    pub fn cdt_tick(&self) -> u64 {
        self.memory.substrate_tick()
    }

    /// Contador de `step()` en memoria CDT (encode/recall).
    pub fn cdt_step_count(&self) -> u64 {
        self.memory.step_count
    }

    /// Episodios pendientes de consolidación.
    pub fn wake_buffer_len(&self) -> usize {
        self.wake_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn liquid_only_for_inference() {
        let mut sys = LiquidCdtSystem::new(8);
        let tick_before = sys.memory.substrate_tick();
        let steps_before = sys.memory.step_count;
        let rqm_before = sys.rqm_api_calls;
        let candidates: Vec<usize> = (0..8).collect();

        let mut ok = 0usize;
        let loops = 64usize;
        let t0 = Instant::now();
        for _ in 0..loops {
            for obs in 0..8 {
                let r = sys.infer(obs, &candidates);
                assert!(!r.used_memory, "infer must not use CDT memory");
                assert!(!r.used_rqm, "infer must not use RQM");
                if r.liquid.predicted == obs {
                    ok += 1;
                }
            }
        }
        let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
        let total = loops * 8;
        let mean_us = elapsed_us / total as f64;

        assert_eq!(
            sys.memory.substrate_tick(),
            tick_before,
            "infer must not advance CDT substrate tick"
        );
        assert_eq!(
            sys.memory.step_count, steps_before,
            "infer must not call memory.step / encode"
        );
        assert_eq!(sys.rqm_api_calls, rqm_before, "infer must not call RQM API");
        let acc = ok as f64 / total as f64;
        println!(
            "liquid_only_for_inference: mean_us={:.4} µs/query acc={:.4} queries={} \
             cdt_tick_delta=0 rqm_api_delta=0",
            mean_us, acc, total
        );
        assert!(acc >= 0.99, "liquid acc={acc}");
    }

    #[test]
    fn sleep_stores_in_cdt_without_forgetting() {
        let mut sys = LiquidCdtSystem::new(8);
        let candidates: Vec<usize> = (0..8).collect();

        // Lote 1: conceptos 0..3 vía observe + sleep
        for obs in 0..4 {
            let _ = sys.observe_and_buffer(obs, &candidates);
        }
        let s1 = sys.sleep_consolidate();
        println!(
            "sleep batch1: consolidated={} engrams {}→{} used_rqm={}",
            s1.episodes_consolidated, s1.engrams_before, s1.engrams_after, s1.used_rqm
        );
        assert!(s1.used_rqm);
        assert_eq!(s1.engrams_after, 4);
        assert_eq!(sys.wake_buffer.len(), 0);

        // Lote 2: conceptos 4..7 — no debe borrar 0..3
        for obs in 4..8 {
            let _ = sys.observe_and_buffer(obs, &candidates);
        }
        let s2 = sys.sleep_consolidate();
        println!(
            "sleep batch2: consolidated={} engrams {}→{} used_rqm={}",
            s2.episodes_consolidated, s2.engrams_before, s2.engrams_after, s2.used_rqm
        );
        assert_eq!(s2.engrams_before, 4);
        assert_eq!(s2.engrams_after, 8);
        assert_eq!(sys.memory.engram_count(), 8);

        // Soft-recall: los 8 engramas siguen recuperables
        let mut recall_ok = 0usize;
        for c in 0..8 {
            let (pred, score) = sys
                .memory
                .recall(c)
                .unwrap_or_else(|| panic!("missing recall for concept {c}"));
            println!("recall cue={c} → pred={pred} score={score:.4}");
            if pred == c {
                recall_ok += 1;
            }
        }
        println!(
            "sleep_stores_in_cdt_without_forgetting: engrams={} recall_ok={}/8",
            sys.memory.engram_count(),
            recall_ok
        );
        assert_eq!(recall_ok, 8, "all 8 engrams must remain matchable");

        // Exactitud líquida intacta (inferencia no depende de CDT)
        let mut ok = 0usize;
        for obs in 0..8 {
            let r = sys.infer(obs, &candidates);
            if r.liquid.predicted == obs {
                ok += 1;
            }
        }
        let liquid_acc = ok as f64 / 8.0;
        println!("liquid accuracy after sleep batches: {liquid_acc:.4}");
        assert!((liquid_acc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rqm_only_during_sleep() {
        let mut sys = LiquidCdtSystem::new(8);
        let candidates: Vec<usize> = (0..8).collect();
        assert_eq!(sys.rqm_api_calls, 0);

        for obs in 0..8 {
            let r = sys.observe_and_buffer(obs, &candidates);
            assert!(!r.used_rqm);
        }
        assert_eq!(
            sys.rqm_api_calls, 0,
            "buffering/infer must not touch RQM; calls={}",
            sys.rqm_api_calls
        );

        let sleep = sys.sleep_consolidate();
        println!(
            "rqm_only_during_sleep: rqm_api_calls={} consolidated={} used_rqm={} relations={}",
            sys.rqm_api_calls,
            sleep.episodes_consolidated,
            sleep.used_rqm,
            sleep.rqm_relations_trained
        );
        assert!(sleep.used_rqm);
        assert_eq!(sys.rqm_api_calls, sleep.rqm_relations_trained as u64);
        assert!(sys.rqm_api_calls >= 8);

        // Más inferencia post-sueño: RQM no debe moverse
        let after_sleep = sys.rqm_api_calls;
        for obs in 0..8 {
            let r = sys.infer(obs, &candidates);
            assert!(!r.used_rqm);
        }
        assert_eq!(sys.rqm_api_calls, after_sleep);
    }

    #[test]
    fn bench_liquid_infer_vs_sleep_cost() {
        let mut sys = LiquidCdtSystem::new(8);
        let candidates: Vec<usize> = (0..8).collect();

        // Warmup líquido
        for _ in 0..16 {
            for obs in 0..8 {
                let _ = sys.infer(obs, &candidates);
            }
        }

        let loops = 80usize;
        let t0 = Instant::now();
        let mut total = 0usize;
        for _ in 0..loops {
            for obs in 0..8 {
                let _ = sys.infer(obs, &candidates);
                total += 1;
            }
        }
        let liquid_us = t0.elapsed().as_secs_f64() * 1e6 / total as f64;

        // Coste de sueño: buffer 8 episodios + consolidate
        for obs in 0..8 {
            let _ = sys.observe_and_buffer(obs, &candidates);
        }
        let t1 = Instant::now();
        let sleep = sys.sleep_consolidate();
        let sleep_ms = t1.elapsed().as_secs_f64() * 1e3;

        println!(
            "bench: liquid_infer={:.4} µs/query (n={}) | sleep_consolidate={:.3} ms \
             (episodes={}, engrams_after={}, used_rqm={})",
            liquid_us,
            total,
            sleep_ms,
            sleep.episodes_consolidated,
            sleep.engrams_after,
            sleep.used_rqm
        );
        assert!(
            liquid_us < 50.0,
            "liquid should stay sub-50µs debug; got {liquid_us}"
        );
        assert!(sleep.used_rqm);
        assert_eq!(sleep.engrams_after, 8);
        // Sueño es órdenes de magnitud más caro que una query líquida (ms vs µs).
        let sleep_per_episode_us = (sleep_ms * 1e3) / sleep.episodes_consolidated.max(1) as f64;
        println!(
            "bench ratio: sleep_per_episode≈{:.1} µs vs liquid≈{:.4} µs (≈{:.0}×)",
            sleep_per_episode_us,
            liquid_us,
            sleep_per_episode_us / liquid_us.max(1e-12)
        );
    }
}
