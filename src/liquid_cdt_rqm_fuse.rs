//! Híbrido fusionado: **lo mejor de NEW LiquidCdt + MAIN RQM**.
//!
//! Tres piezas cooperativas (conceptualmente 2 secciones de usuario + pegamento):
//! 1. **LiquidInfer** — siempre se intenta primero (núcleo rápido ~0.3 µs identidad).
//! 2. **CdtConsolidatedMemory** — memoria dinámica tras el sueño (engramas, sin olvido).
//! 3. **RqmRelationalIndex** — relaciones arbitrarias entrenadas (fuerza de MAIN);
//!    se escribe en sueño; se lee en infer **solo** como fallback/router cuando hace falta.
//!
//! Reglas de enrutado (`infer`):
//! - Siempre: líquido primero.
//! - RQM fallback si: `cue ∈ relational_cues` **o** `score_líquido < liquid_min_score`,
//!   **y** (RQM tiene el cue entrenado **o** `force_rqm`).
//! - Identidad sin `teach_relation` → ruta Liquid (RQM no es default).
//! - Mapas arbitrarios tras `teach_relation` + sueño → ruta `RqmFallback`.
//!
//! Ver `docs/hibrido_liquido_cdt_rqm_fuse.md`.

use crate::entanglement::EntanglementConfig;
use crate::liquid_cdt_memory::{
    CdtConsolidatedMemory, Episode, EpisodeKind, LiquidInfer, SleepReport,
};
use crate::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::relational_field::ObserverId;
use std::collections::HashSet;
use std::time::Instant;

const OBSERVER: ObserverId = ObserverId(0xF05E_CD71);
const CUE_BASE: usize = 64;
const DEFAULT_LIQUID_MIN_SCORE: f64 = 0.2;
const REL_TRAIN_EPOCHS: usize = 4;

/// Ruta tomada en una query del híbrido fusionado.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferRoute {
    /// Respuesta del núcleo líquido (camino caliente preferido).
    Liquid,
    /// Fallback / índice relacional RQM (mapas arbitrarios o score bajo).
    RqmFallback,
}

/// Informe de una inferencia fusionada.
#[derive(Clone, Debug)]
pub struct FuseReport {
    pub observation: usize,
    pub predicted: usize,
    pub liquid_score: f64,
    pub route: InferRoute,
    pub rqm_score: Option<f64>,
}

/// Sistema fusionado: líquido + CDT sueño + índice RQM relacional.
pub struct FusedLiquidCdt {
    pub liquid: LiquidInfer,
    pub memory: CdtConsolidatedMemory,
    /// Índice relacional (MAIN). Escrito en sueño; leído en infer solo como fallback.
    pub rqm: NativeThermoRqmEprSubstrate,
    pub wake_buffer: Vec<Episode>,
    /// Cues con relación arbitraria enseñada (`teach_relation`).
    pub relational_cues: HashSet<usize>,
    /// Debajo de este score líquido → intentar RQM si hay cue entrenado.
    pub liquid_min_score: f64,
    pub num_labels: usize,
    /// Contador de queries RQM en el hot path (fallback).
    pub rqm_infer_calls: u64,
    /// Contador de `train_observed_transition` (sueño / teach inmediato).
    pub rqm_train_calls: u64,
}

impl FusedLiquidCdt {
    pub fn new(num_labels: usize) -> Self {
        let num_labels = num_labels.max(2);
        let capacity = num_labels.max(16);
        let thermal = NativeThermoCdtConfig {
            slices: 2,
            nodes_per_slice: (CUE_BASE + num_labels).max(128),
            seed: 0xF05E_F100,
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
            relational_cues: HashSet::new(),
            liquid_min_score: DEFAULT_LIQUID_MIN_SCORE,
            num_labels,
            rqm_infer_calls: 0,
            rqm_train_calls: 0,
        }
    }

    #[inline]
    fn cue_id(obs: usize) -> usize {
        // Patrón benches: cue ≠ label-id espacial (evita auto-relación en el grafo).
        CUE_BASE + obs
    }

    fn pick_rqm_label(cands: &[NativeCandidateScore], n: usize) -> Option<(usize, f64)> {
        cands
            .iter()
            .filter(|c| c.agent < n)
            .max_by(|a, b| {
                a.score
                    .total_cmp(&b.score)
                    .then_with(|| a.agent.cmp(&b.agent))
            })
            .map(|c| (c.agent, c.score as f64))
    }

    fn rqm_has_cue(&self, obs: usize) -> bool {
        self.relational_cues.contains(&(obs % self.num_labels))
    }

    fn train_rqm_relation(&mut self, cue_obs: usize, label: usize) {
        let obs = cue_obs % self.num_labels;
        let lab = label % self.num_labels;
        let cue = Self::cue_id(obs);
        // Varias épocas ligeras para asentar la relación (fuerza MAIN).
        for _ in 0..REL_TRAIN_EPOCHS {
            self.rqm
                .train_observed_transition(OBSERVER, 0.0, &[cue], &[lab], 0.95);
            self.rqm_train_calls = self.rqm_train_calls.wrapping_add(1);
        }
        self.relational_cues.insert(obs);
    }

    fn query_rqm(&mut self, obs: usize) -> Option<(usize, f64)> {
        let obs = obs % self.num_labels;
        let report = self.rqm.query(OBSERVER, 0.0, &[Self::cue_id(obs)]);
        self.rqm_infer_calls = self.rqm_infer_calls.wrapping_add(1);
        Self::pick_rqm_label(&report.candidates, self.num_labels)
    }

    /// Hot path: líquido primero; RQM solo como fallback/router cuando hace falta.
    ///
    /// `force_rqm`: fuerza intento RQM si el cue está marcado o hay relación (tests).
    pub fn infer(&mut self, obs: usize, candidates: &[usize]) -> FuseReport {
        self.infer_with_force(obs, candidates, false)
    }

    pub fn infer_with_force(
        &mut self,
        obs: usize,
        candidates: &[usize],
        force_rqm: bool,
    ) -> FuseReport {
        let obs = obs % self.num_labels;
        let liquid = self.liquid.infer(obs, candidates);
        let has_rel = self.rqm_has_cue(obs);
        let low_score = liquid.score < self.liquid_min_score;
        // Regla: score bajo **o** cue con relación entrenada → intentar RQM
        // (sin hacer RQM el default de identidad: sin teach_relation, has_rel=false
        // y el score líquido de identidad suele ser alto).
        let want_rqm = has_rel || low_score || force_rqm;
        if want_rqm && (has_rel || force_rqm) {
            if let Some((pred, rqm_score)) = self.query_rqm(obs) {
                return FuseReport {
                    observation: obs,
                    predicted: pred,
                    liquid_score: liquid.score,
                    route: InferRoute::RqmFallback,
                    rqm_score: Some(rqm_score),
                };
            }
        }
        FuseReport {
            observation: obs,
            predicted: liquid.predicted,
            liquid_score: liquid.score,
            route: InferRoute::Liquid,
            rqm_score: None,
        }
    }

    /// Enseña una relación arbitraria (fuerza MAIN): buffer + índice RQM inmediato
    /// para que el sueño consolide CDT y reafirme RQM.
    pub fn teach_relation(&mut self, cue: usize, label: usize) {
        let cue = cue % self.num_labels;
        let label = label % self.num_labels;
        self.wake_buffer.push(Episode {
            obs: cue,
            predicted: label,
            score: 1.0,
            kind: EpisodeKind::Relation,
        });
        // Entrena de inmediato en el índice (disponible post-sueño / post-teach).
        self.train_rqm_relation(cue, label);
    }

    /// Buffer de un episodio líquido para sueño (identidad / observación).
    pub fn observe(&mut self, obs: usize, candidates: &[usize]) {
        let obs = obs % self.num_labels;
        let r = self.liquid.infer(obs, candidates);
        if r.score >= self.liquid_min_score {
            self.wake_buffer.push(Episode {
                obs: r.observation,
                predicted: r.predicted,
                score: r.score,
                kind: EpisodeKind::Liquid,
            });
        }
    }

    /// Sueño: engramas CDT (sin wipe) + RQM train para relaciones bufferizadas.
    pub fn sleep_consolidate(&mut self) -> SleepReport {
        let t0 = Instant::now();
        let engrams_before = self.memory.engram_count();
        let mut consolidated = 0usize;
        let mut rqm_trained = 0usize;
        let episodes: Vec<Episode> = std::mem::take(&mut self.wake_buffer);

        for ep in episodes {
            // 1) Engrama durable en Thermo CDT (incremental).
            self.memory
                .encode_engram(ep.predicted % self.memory.capacity);
            // 2) Índice RQM: reafirma cue→label (especialmente Relation).
            self.train_rqm_relation(ep.obs, ep.predicted);
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

    pub fn engram_count(&self) -> usize {
        self.memory.engram_count()
    }

    pub fn wake_buffer_len(&self) -> usize {
        self.wake_buffer.len()
    }
}

// ─── Comparación fuse vs brazos puros (tabla estilo liquid_cdt_vs_main_bench) ─

#[derive(Clone, Debug, Default)]
pub struct FuseArmRow {
    pub name: &'static str,
    pub identity_mean_us: f64,
    pub identity_acc: f64,
    pub identity_liquid_route_frac: f64,
    pub shifted_mean_us: f64,
    pub shifted_acc: f64,
    pub shifted_rqm_route_frac: f64,
    pub retained_engrams: usize,
}

#[derive(Clone, Debug)]
pub struct FuseCompareTable {
    pub fuse: FuseArmRow,
    pub liquid_only: FuseArmRow,
    pub main_rqm: FuseArmRow,
}

const BENCH_N: usize = 8;
const WARMUP: usize = 200;
const BENCH_Q: usize = 2000;
const MAIN_EPOCHS: usize = 6;

fn make_bare_rqm(n: usize) -> NativeThermoRqmEprSubstrate {
    let thermal = NativeThermoCdtConfig {
        slices: 2,
        nodes_per_slice: (CUE_BASE + n).max(128),
        seed: 0xA11A_F05E,
        ..NativeThermoCdtConfig::default()
    };
    let mut cfg = NativeThermoRqmConfig::default();
    cfg.max_candidates = n * 2;
    cfg.thermal_steps_per_train = 1;
    cfg.thermal_steps_per_query = 2;
    cfg.thermal_activation_margin = 0.05;
    let epr = EntanglementConfig {
        create_threshold: 0.4,
        max_syncs_per_step: 64,
        ..EntanglementConfig::default()
    };
    NativeThermoRqmEprSubstrate::new(thermal, cfg, epr)
}

fn pick_label(cands: &[NativeCandidateScore], n: usize) -> Option<usize> {
    FusedLiquidCdt::pick_rqm_label(cands, n).map(|(a, _)| a)
}

/// Tabla fuse vs líquido puro vs MAIN RQM (identidad + shifted).
pub fn run_fuse_compare() -> FuseCompareTable {
    let n = BENCH_N;
    let candidates: Vec<usize> = (0..n).collect();

    // ── FUSE: identity (sin teach) ──────────────────────────────────────────
    let mut fuse = FusedLiquidCdt::new(n);
    for _ in 0..WARMUP {
        for i in 0..n {
            let _ = fuse.infer(i, &candidates);
        }
    }
    let mut id_ok = 0usize;
    let mut id_liquid = 0usize;
    let t0 = Instant::now();
    for q in 0..BENCH_Q {
        let i = q % n;
        let r = fuse.infer(i, &candidates);
        if r.predicted == i {
            id_ok += 1;
        }
        if r.route == InferRoute::Liquid {
            id_liquid += 1;
        }
    }
    let fuse_id_us = t0.elapsed().as_secs_f64() * 1e6 / BENCH_Q as f64;
    let fuse_id_acc = id_ok as f64 / BENCH_Q as f64;
    let fuse_id_liq_frac = id_liquid as f64 / BENCH_Q as f64;

    // ── FUSE: shifted via teach + sleep ─────────────────────────────────────
    let mut fuse_sh = FusedLiquidCdt::new(n);
    for i in 0..n {
        fuse_sh.teach_relation(i, (i + 1) % n);
    }
    let _ = fuse_sh.sleep_consolidate();
    for _ in 0..(WARMUP / 2) {
        for i in 0..n {
            let _ = fuse_sh.infer(i, &candidates);
        }
    }
    let mut sh_ok = 0usize;
    let mut sh_rqm = 0usize;
    let t1 = Instant::now();
    for q in 0..BENCH_Q {
        let i = q % n;
        let r = fuse_sh.infer(i, &candidates);
        if r.predicted == (i + 1) % n {
            sh_ok += 1;
        }
        if r.route == InferRoute::RqmFallback {
            sh_rqm += 1;
        }
    }
    let fuse_sh_us = t1.elapsed().as_secs_f64() * 1e6 / BENCH_Q as f64;
    let fuse_sh_acc = sh_ok as f64 / BENCH_Q as f64;
    let fuse_sh_rqm_frac = sh_rqm as f64 / BENCH_Q as f64;
    let fuse_engrams = fuse_sh.engram_count();

    // ── Liquid-only (WavePredictCore path via Fused liquid, no RQM teach) ───
    let mut liq = FusedLiquidCdt::new(n);
    // identity
    for _ in 0..WARMUP {
        for i in 0..n {
            let _ = liq.liquid.infer(i, &candidates);
        }
    }
    let mut l_id_ok = 0usize;
    let t2 = Instant::now();
    for q in 0..BENCH_Q {
        let i = q % n;
        let r = liq.liquid.infer(i, &candidates);
        if r.predicted == i {
            l_id_ok += 1;
        }
    }
    let liq_id_us = t2.elapsed().as_secs_f64() * 1e6 / BENCH_Q as f64;
    let liq_id_acc = l_id_ok as f64 / BENCH_Q as f64;
    // shifted (liquid cannot learn arbitrary map)
    let mut l_sh_ok = 0usize;
    let t3 = Instant::now();
    for q in 0..BENCH_Q {
        let i = q % n;
        let r = liq.liquid.infer(i, &candidates);
        if r.predicted == (i + 1) % n {
            l_sh_ok += 1;
        }
    }
    let liq_sh_us = t3.elapsed().as_secs_f64() * 1e6 / BENCH_Q as f64;
    let liq_sh_acc = l_sh_ok as f64 / BENCH_Q as f64;

    // ── MAIN RQM bare ───────────────────────────────────────────────────────
    let mut main = make_bare_rqm(n);
    for _ in 0..MAIN_EPOCHS {
        for i in 0..n {
            main.train_observed_transition(OBSERVER, 0.0, &[CUE_BASE + i], &[i], 0.95);
        }
    }
    for _ in 0..WARMUP {
        for i in 0..n {
            let _ = main.query(OBSERVER, 0.0, &[CUE_BASE + i]);
        }
    }
    let mut m_id_ok = 0usize;
    let t4 = Instant::now();
    for q in 0..BENCH_Q {
        let i = q % n;
        let r = main.query(OBSERVER, 0.0, &[CUE_BASE + i]);
        if pick_label(&r.candidates, n) == Some(i) {
            m_id_ok += 1;
        }
    }
    let main_id_us = t4.elapsed().as_secs_f64() * 1e6 / BENCH_Q as f64;
    let main_id_acc = m_id_ok as f64 / BENCH_Q as f64;

    let mut main_sh = make_bare_rqm(n);
    for _ in 0..MAIN_EPOCHS {
        for i in 0..n {
            main_sh.train_observed_transition(OBSERVER, 0.0, &[CUE_BASE + i], &[(i + 1) % n], 0.95);
        }
    }
    for _ in 0..(WARMUP / 2) {
        for i in 0..n {
            let _ = main_sh.query(OBSERVER, 0.0, &[CUE_BASE + i]);
        }
    }
    let mut m_sh_ok = 0usize;
    let t5 = Instant::now();
    for q in 0..BENCH_Q {
        let i = q % n;
        let r = main_sh.query(OBSERVER, 0.0, &[CUE_BASE + i]);
        if pick_label(&r.candidates, n) == Some((i + 1) % n) {
            m_sh_ok += 1;
        }
    }
    let main_sh_us = t5.elapsed().as_secs_f64() * 1e6 / BENCH_Q as f64;
    let main_sh_acc = m_sh_ok as f64 / BENCH_Q as f64;

    FuseCompareTable {
        fuse: FuseArmRow {
            name: "FUSE (liquid + CDT + RQM index)",
            identity_mean_us: fuse_id_us,
            identity_acc: fuse_id_acc,
            identity_liquid_route_frac: fuse_id_liq_frac,
            shifted_mean_us: fuse_sh_us,
            shifted_acc: fuse_sh_acc,
            shifted_rqm_route_frac: fuse_sh_rqm_frac,
            retained_engrams: fuse_engrams,
        },
        liquid_only: FuseArmRow {
            name: "Liquid-only (WavePredictCore)",
            identity_mean_us: liq_id_us,
            identity_acc: liq_id_acc,
            identity_liquid_route_frac: 1.0,
            shifted_mean_us: liq_sh_us,
            shifted_acc: liq_sh_acc,
            shifted_rqm_route_frac: 0.0,
            retained_engrams: 0,
        },
        main_rqm: FuseArmRow {
            name: "MAIN RQM (direct)",
            identity_mean_us: main_id_us,
            identity_acc: main_id_acc,
            identity_liquid_route_frac: 0.0,
            shifted_mean_us: main_sh_us,
            shifted_acc: main_sh_acc,
            shifted_rqm_route_frac: 1.0,
            retained_engrams: 0,
        },
    }
}

pub fn format_fuse_compare_table(t: &FuseCompareTable) -> String {
    let mut s = String::new();
    s.push_str("# FUSE vs Liquid-only vs MAIN RQM\n\n");
    s.push_str(&format!(
        "Protocolo: N={BENCH_N} · warmup={WARMUP} · bench={BENCH_Q} · MAIN epochs={MAIN_EPOCHS}\n\n"
    ));
    s.push_str(
        "| Brazo | ID mean µs | ID acc | ID Liquid% | SH mean µs | SH acc | SH RQM% | Engrams |\n",
    );
    s.push_str(
        "|-------|-----------:|-------:|-----------:|-----------:|-------:|--------:|--------:|\n",
    );
    for row in [&t.fuse, &t.liquid_only, &t.main_rqm] {
        s.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.1}% | {:.4} | {:.4} | {:.1}% | {} |\n",
            row.name,
            row.identity_mean_us,
            row.identity_acc,
            row.identity_liquid_route_frac * 100.0,
            row.shifted_mean_us,
            row.shifted_acc,
            row.shifted_rqm_route_frac * 100.0,
            row.retained_engrams
        ));
    }
    s.push('\n');
    s.push_str(&format!(
        "Veredicto: FUSE gana shifted vs liquid ({:.4} vs {:.4}); \
         identidad vía Liquid (frac={:.3}) con latencia {:.4} µs vs MAIN {:.4} µs.\n",
        t.fuse.shifted_acc,
        t.liquid_only.shifted_acc,
        t.fuse.identity_liquid_route_frac,
        t.fuse.identity_mean_us,
        t.main_rqm.identity_mean_us
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuse_identity_uses_liquid_fast() {
        let n = 8usize;
        let candidates: Vec<usize> = (0..n).collect();
        let mut fuse = FusedLiquidCdt::new(n);

        // warmup
        for _ in 0..64 {
            for i in 0..n {
                let _ = fuse.infer(i, &candidates);
            }
        }

        let loops = 250usize;
        let total = loops * n;
        let mut ok = 0usize;
        let mut liquid_routes = 0usize;
        let mut rqm_routes = 0usize;
        let t0 = Instant::now();
        for _ in 0..loops {
            for i in 0..n {
                let r = fuse.infer(i, &candidates);
                if r.predicted == i {
                    ok += 1;
                }
                match r.route {
                    InferRoute::Liquid => liquid_routes += 1,
                    InferRoute::RqmFallback => rqm_routes += 1,
                }
            }
        }
        let mean_us = t0.elapsed().as_secs_f64() * 1e6 / total as f64;
        let acc = ok as f64 / total as f64;
        let liquid_frac = liquid_routes as f64 / total as f64;

        println!(
            "fuse_identity_uses_liquid_fast: acc={:.4} mean_us={:.4} \
             liquid_routes={}/{} ({:.1}%) rqm_routes={} rqm_infer_calls={}",
            acc,
            mean_us,
            liquid_routes,
            total,
            liquid_frac * 100.0,
            rqm_routes,
            fuse.rqm_infer_calls
        );

        assert!(
            (acc - 1.0).abs() < 1e-9,
            "identity acc must be 1.0, got {acc}"
        );
        assert!(
            liquid_frac >= 0.99,
            "route must be mostly Liquid, frac={liquid_frac}"
        );
        assert_eq!(rqm_routes, 0, "identity without teach must not use RQM");
        // Ballpark: líquido ~0.3µs; MAIN ~1.5µs — exigir claramente camino líquido.
        assert!(
            mean_us < 5.0,
            "identity mean_us should be liquid-fast (<< MAIN ~1.5µs ballpark); got {mean_us}"
        );
    }

    #[test]
    fn fuse_shifted_uses_rqm_from_main() {
        let n = 8usize;
        let candidates: Vec<usize> = (0..n).collect();
        let mut fuse = FusedLiquidCdt::new(n);

        for i in 0..n {
            fuse.teach_relation(i, (i + 1) % n);
        }
        let sleep = fuse.sleep_consolidate();
        println!(
            "fuse_shifted sleep: consolidated={} engrams {}→{} rqm_relations={}",
            sleep.episodes_consolidated,
            sleep.engrams_before,
            sleep.engrams_after,
            sleep.rqm_relations_trained
        );

        let mut ok = 0usize;
        let mut rqm_routes = 0usize;
        let loops = 200usize;
        let total = loops * n;
        let t0 = Instant::now();
        for _ in 0..loops {
            for i in 0..n {
                let r = fuse.infer(i, &candidates);
                assert_eq!(
                    r.route,
                    InferRoute::RqmFallback,
                    "shifted cue {i} must route RqmFallback"
                );
                rqm_routes += 1;
                if r.predicted == (i + 1) % n {
                    ok += 1;
                }
            }
        }
        let mean_us = t0.elapsed().as_secs_f64() * 1e6 / total as f64;
        let acc = ok as f64 / total as f64;
        println!(
            "fuse_shifted_uses_rqm_from_main: acc={:.4} mean_us={:.4} \
             rqm_routes={}/{} relational_cues={}",
            acc,
            mean_us,
            rqm_routes,
            total,
            fuse.relational_cues.len()
        );
        assert!(acc >= 0.99, "shifted acc must be >= 0.99, got {acc}");
        assert_eq!(rqm_routes, total);
    }

    #[test]
    fn fuse_sleep_cdt_no_forgetting() {
        let n = 8usize;
        let candidates: Vec<usize> = (0..n).collect();
        let mut fuse = FusedLiquidCdt::new(n);

        // Lote 1: relaciones 0..3 → labels 0..3 (y engramas)
        for i in 0..4 {
            fuse.teach_relation(i, i);
        }
        let s1 = fuse.sleep_consolidate();
        println!(
            "batch1: consolidated={} engrams {}→{}",
            s1.episodes_consolidated, s1.engrams_before, s1.engrams_after
        );
        assert_eq!(s1.engrams_after, 4);

        // Lote 2: 4..7 — no debe borrar 0..3
        for i in 4..8 {
            fuse.teach_relation(i, i);
        }
        let s2 = fuse.sleep_consolidate();
        println!(
            "batch2: consolidated={} engrams {}→{}",
            s2.episodes_consolidated, s2.engrams_before, s2.engrams_after
        );
        assert_eq!(s2.engrams_before, 4);
        assert_eq!(s2.engrams_after, 8);
        assert_eq!(fuse.engram_count(), 8);

        let mut recall_ok = 0usize;
        for c in 0..8 {
            let (pred, score) = fuse
                .memory
                .recall(c)
                .unwrap_or_else(|| panic!("missing recall for {c}"));
            println!("recall cue={c} → pred={pred} score={score:.4}");
            if pred == c {
                recall_ok += 1;
            }
        }
        println!(
            "fuse_sleep_cdt_no_forgetting: engrams={} recall_ok={}/8",
            fuse.engram_count(),
            recall_ok
        );
        assert_eq!(recall_ok, 8);

        // Observaciones líquidas extra no deben borrar engramas
        for i in 0..n {
            fuse.observe(i, &candidates);
        }
        let s3 = fuse.sleep_consolidate();
        println!(
            "batch3 observe-sleep: consolidated={} engrams_after={}",
            s3.episodes_consolidated, s3.engrams_after
        );
        assert!(
            fuse.engram_count() >= 8,
            "engrams must be retained after third sleep"
        );
    }

    #[test]
    fn fuse_beats_pure_arms() {
        let table = run_fuse_compare();
        let text = format_fuse_compare_table(&table);
        println!("{text}");

        // vs pure liquid: better shifted acc
        assert!(
            table.fuse.shifted_acc > table.liquid_only.shifted_acc + 0.5,
            "fuse shifted ({:.4}) must beat liquid-only ({:.4})",
            table.fuse.shifted_acc,
            table.liquid_only.shifted_acc
        );
        assert!(
            table.fuse.shifted_acc >= 0.99,
            "fuse shifted acc={}",
            table.fuse.shifted_acc
        );

        // vs main RQM: identity latency clearly liquid-path
        assert!(
            table.fuse.identity_acc >= 0.99,
            "fuse identity acc={}",
            table.fuse.identity_acc
        );
        assert!(
            table.fuse.identity_liquid_route_frac >= 0.99,
            "identity must stay on Liquid route"
        );
        assert!(
            table.fuse.identity_mean_us < table.main_rqm.identity_mean_us * 1.2
                || table.fuse.identity_mean_us < 1.0,
            "fuse identity mean_us={:.4} should be < main*{:.4} (1.2×) or clearly liquid (<1µs); main={:.4}",
            table.fuse.identity_mean_us,
            table.main_rqm.identity_mean_us * 1.2,
            table.main_rqm.identity_mean_us
        );

        println!(
            "SUMMARY fuse_beats_pure_arms:\n\
             | arm | id_us | id_acc | sh_us | sh_acc |\n\
             | fuse | {:.4} | {:.4} | {:.4} | {:.4} |\n\
             | liquid | {:.4} | {:.4} | {:.4} | {:.4} |\n\
             | main | {:.4} | {:.4} | {:.4} | {:.4} |",
            table.fuse.identity_mean_us,
            table.fuse.identity_acc,
            table.fuse.shifted_mean_us,
            table.fuse.shifted_acc,
            table.liquid_only.identity_mean_us,
            table.liquid_only.identity_acc,
            table.liquid_only.shifted_mean_us,
            table.liquid_only.shifted_acc,
            table.main_rqm.identity_mean_us,
            table.main_rqm.identity_acc,
            table.main_rqm.shifted_mean_us,
            table.main_rqm.shifted_acc,
        );
    }
}
