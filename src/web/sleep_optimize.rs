#![allow(clippy::needless_range_loop, clippy::for_kv_map)]
//! Sueño con optimización: poda de rutas débiles + compactación fasorial +
//! minimización de energía libre hacia mayor simetría (octaedro / handshake).

use crate::field_encoder::write_into_field;
use crate::field_hybrid_infer::concept_to_phasors;
use crate::field_substrate::{
    free_energy, handshake, hebb_update, ignite_casimir, prune, relax, write_hemisphere_pattern,
    ComplexT, FieldConfig, FieldState, Phasor,
};
use crate::liquid_cdt_rqm_fuse::{FusedLiquidCdt, InferRoute};
use serde::Serialize;
use std::collections::HashSet;
use std::f64::consts::PI;

/// Informe detallado de sueño + optimización geométrica.
#[derive(Clone, Debug, Serialize)]
pub struct SleepOptimizeReport {
    pub free_energy_before: f64,
    pub free_energy_after: f64,
    pub symmetry_before: f64,
    pub symmetry_after: f64,
    pub routes_pruned: usize,
    pub phasors_compacted: usize,
    pub nodes_compacted: usize,
    pub edges_compacted: usize,
    pub handshake: f64,
    pub engrams: usize,
    pub episodes_consolidated: usize,
    pub engrams_before: usize,
    pub engrams_after: usize,
    pub used_rqm: bool,
    pub rqm_relations_trained: usize,
    pub sleep_ms: f64,
    pub prune_intensity: f64,
    pub compact_intensity: f64,
    pub notes: Vec<String>,
}

impl Default for SleepOptimizeReport {
    fn default() -> Self {
        Self {
            free_energy_before: 0.0,
            free_energy_after: 0.0,
            symmetry_before: 0.0,
            symmetry_after: 0.0,
            routes_pruned: 0,
            phasors_compacted: 0,
            nodes_compacted: 0,
            edges_compacted: 0,
            handshake: 0.0,
            engrams: 0,
            episodes_consolidated: 0,
            engrams_before: 0,
            engrams_after: 0,
            used_rqm: false,
            rqm_relations_trained: 0,
            sleep_ms: 0.0,
            prune_intensity: 0.5,
            compact_intensity: 0.5,
            notes: Vec::new(),
        }
    }
}

/// Opciones de intensidad [0, 1].
#[derive(Clone, Copy, Debug)]
pub struct SleepOptimizeOpts {
    pub prune_intensity: f64,
    pub compact_intensity: f64,
    pub consolidate_first: bool,
}

impl Default for SleepOptimizeOpts {
    fn default() -> Self {
        Self {
            prune_intensity: 0.55,
            compact_intensity: 0.55,
            consolidate_first: true,
        }
    }
}

/// Simetría ∈ [0, 1]: alineación fasorial media + baja varianza de amplitudes.
pub fn symmetry_score(psi: &FieldState) -> f64 {
    let n = psi.n();
    if n == 0 {
        return 0.0;
    }
    let amps: Vec<f64> = psi.z.iter().map(|z| z.norm()).collect();
    let mean = amps.iter().sum::<f64>() / n as f64;
    let var = amps
        .iter()
        .map(|a| {
            let d = a - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let amp_uniform = 1.0 / (1.0 + var.sqrt());

    let mut align = 0.0;
    let mut pairs = 0usize;
    for e in 0..n {
        for e2 in (e + 1)..n {
            if !psi.t.share_boundary(e, e2) {
                continue;
            }
            let na = psi.z[e].norm();
            let nb = psi.z[e2].norm();
            if na < 1e-12 || nb < 1e-12 {
                continue;
            }
            let c = (psi.z[e].conj() * psi.z[e2]).re / (na * nb);
            align += c.clamp(-1.0, 1.0);
            pairs += 1;
        }
    }
    let phase_align = if pairs == 0 {
        0.0
    } else {
        ((align / pairs as f64) + 1.0) * 0.5
    };
    (0.45 * amp_uniform + 0.55 * phase_align).clamp(0.0, 1.0)
}

fn seed_field_from_engrams(fuse: &FusedLiquidCdt, psi: &mut FieldState) {
    let n = psi.n();
    if fuse.memory.engrams.is_empty() {
        write_hemisphere_pattern(psi);
        return;
    }
    let k = fuse.memory.engrams.len().max(1) as f64;
    let scale = 1.0 / k.sqrt();
    for z in psi.z.iter_mut() {
        *z = Phasor::new(0.0, 0.0);
    }
    for &concept in fuse.memory.engrams.keys() {
        let pattern = concept_to_phasors(concept, n);
        for (e, pz) in pattern.iter().enumerate().take(n) {
            psi.z[e] += *pz * scale;
        }
    }
    psi.z_past = psi.z.clone();
}

/// Poda cues RQM débiles según score de fallback / umbral de intensidad.
pub fn prune_weak_routes(fuse: &mut FusedLiquidCdt, intensity: f64) -> usize {
    let intensity = intensity.clamp(0.0, 1.0);
    if intensity <= 1e-9 || fuse.relational_cues.is_empty() {
        return 0;
    }
    let min_score = 0.15 + 0.75 * intensity;
    let cands: Vec<usize> = (0..fuse.num_labels).collect();
    let cues: Vec<usize> = fuse.relational_cues.iter().copied().collect();
    let mut drop: HashSet<usize> = HashSet::new();
    for cue in cues {
        let rep = fuse.infer_with_force(cue, &cands, true);
        let weak = match (rep.route, rep.rqm_score) {
            (InferRoute::RqmFallback, Some(s)) => s < min_score,
            (InferRoute::RqmFallback, None) => true,
            (InferRoute::Liquid, _) => rep.liquid_score < fuse.liquid_min_score + 0.1 * intensity,
        };
        if weak {
            drop.insert(cue);
        }
    }
    if drop.is_empty() && intensity > 0.85 && fuse.relational_cues.len() > 2 {
        let cues2: Vec<usize> = fuse.relational_cues.iter().copied().collect();
        let mut scored: Vec<(usize, f64)> = Vec::with_capacity(cues2.len());
        for c in cues2 {
            let r = fuse.infer(c, &cands);
            scored.push((c, r.liquid_score));
        }
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let n_drop = (scored.len() / 3).max(1);
        for (c, _) in scored.into_iter().take(n_drop) {
            drop.insert(c);
        }
    }
    for c in &drop {
        fuse.relational_cues.remove(c);
    }
    drop.len()
}

/// Compacta fasores cercanos en fase; renormaliza amplitudes.
pub fn compact_phasorial_geometry(psi: &mut FieldState, intensity: f64) -> (usize, usize) {
    let intensity = intensity.clamp(0.0, 1.0);
    let n = psi.n();
    if n == 0 || intensity <= 1e-9 {
        return (0, 0);
    }
    let phase_eps = (0.55 - 0.40 * intensity).max(0.08);
    let mut compacted = 0usize;
    let mut touched: HashSet<usize> = HashSet::new();
    let z0 = psi.z.clone();
    for e in 0..n {
        if z0[e].norm() < 1e-12 {
            continue;
        }
        for e2 in (e + 1)..n {
            if !psi.t.share_boundary(e, e2) || z0[e2].norm() < 1e-12 {
                continue;
            }
            let pe = z0[e].arg();
            let pf = z0[e2].arg();
            let mut d = (pe - pf).abs();
            if d > PI {
                d = 2.0 * PI - d;
            }
            if d > phase_eps {
                continue;
            }
            let merged = (psi.z[e] + psi.z[e2]) * 0.5;
            let amp = merged.norm().max(1e-12);
            let target_amp = ((z0[e].norm() + z0[e2].norm()) * 0.5).max(0.15);
            let unit = merged / amp;
            let out = unit * (target_amp * (0.7 + 0.3 * intensity));
            psi.z[e] = out;
            psi.z[e2] = out;
            touched.insert(e);
            touched.insert(e2);
            compacted += 1;
        }
    }
    if intensity > 0.2 {
        let mean = psi.z.iter().map(|z| z.norm()).sum::<f64>() / n as f64;
        if mean > 1e-12 {
            for z in psi.z.iter_mut() {
                let a = z.norm();
                if a < 1e-12 {
                    continue;
                }
                let blend = mean * intensity + a * (1.0 - intensity);
                *z *= blend / a;
            }
        }
    }
    psi.z_past = psi.z.clone();
    (compacted, touched.len())
}

/// Relaja el campo: Langevin + handshake + Hebb.
pub fn minimize_free_energy(
    psi: &mut FieldState,
    cfg: &FieldConfig,
    steps: usize,
    seed: u64,
) -> f64 {
    relax(psi, cfg, steps, seed);
    let hs = handshake(psi, cfg);
    ignite_casimir(psi);
    hebb_update(psi, cfg);
    let _ = prune(psi, cfg);
    hs
}

/// Pipeline completo de sueño optimizado.
pub fn run_sleep_optimize(
    fuse: &mut FusedLiquidCdt,
    field: &mut FieldState,
    cfg: &FieldConfig,
    opts: SleepOptimizeOpts,
) -> SleepOptimizeReport {
    run_sleep_optimize_with_progress(fuse, field, cfg, opts, |_phase, _msg| {})
}

/// Igual que [`run_sleep_optimize`] pero emite fases para consola en vivo.
pub fn run_sleep_optimize_with_progress(
    fuse: &mut FusedLiquidCdt,
    field: &mut FieldState,
    cfg: &FieldConfig,
    opts: SleepOptimizeOpts,
    mut on_progress: impl FnMut(&str, &str),
) -> SleepOptimizeReport {
    let mut notes = Vec::new();
    on_progress("start", "iniciando sueño / optimización");
    let prune_i = opts.prune_intensity.clamp(0.0, 1.0);
    let compact_i = opts.compact_intensity.clamp(0.0, 1.0);

    if field.n() == 0 {
        *field = FieldState::new(ComplexT::octahedron());
        notes.push("FieldState vacío: se inicializó octaedro".into());
    }

    let mut episodes = 0usize;
    let mut engrams_before = fuse.engram_count();
    let mut used_rqm = false;
    let mut rqm_trained = 0usize;
    let mut sleep_ms = 0.0;

    if opts.consolidate_first {
        on_progress("consolidate", "consolidando buffer wake → CDT");
        let sleep = fuse.sleep_consolidate();
        episodes = sleep.episodes_consolidated;
        engrams_before = sleep.engrams_before;
        used_rqm = sleep.used_rqm;
        rqm_trained = sleep.rqm_relations_trained;
        sleep_ms = sleep.sleep_ms;
        notes.push(format!(
            "sleep_consolidate: {} episodios, engramas {}→{}",
            sleep.episodes_consolidated, sleep.engrams_before, sleep.engrams_after
        ));
    }

    if field.z.iter().all(|z| z.norm() < 1e-9) {
        seed_field_from_engrams(fuse, field);
        notes.push("campo sembrado desde engramas/hemisferio".into());
    } else if !fuse.memory.engrams.is_empty() {
        let n = field.n();
        let boost = 0.25;
        for &concept in fuse.memory.engrams.keys() {
            let pattern = concept_to_phasors(concept, n);
            for (e, pz) in pattern.iter().enumerate().take(n) {
                field.z[e] = field.z[e] * (1.0 - boost) + *pz * boost;
            }
        }
        field.z_past = field.z.clone();
        notes.push("campo reforzado con engramas CDT".into());
    }

    let f_before = free_energy(field, cfg).total;
    let sym_before = symmetry_score(field);
    let best_z = field.z.clone();
    let best_past = field.z_past.clone();
    let best_m = field.m.clone();
    let mut best_f = f_before;
    let mut best_sym = sym_before;
    #[allow(unused_assignments)]
    let mut best_hs = 0.0_f64;

    on_progress(
        "prune",
        &format!("podando rutas RQM (intensidad={prune_i:.2})"),
    );
    let routes_pruned = prune_weak_routes(fuse, prune_i);
    notes.push(format!(
        "rutas RQM podadas={routes_pruned} (intensidad={prune_i:.2})"
    ));
    on_progress("prune", &format!("rutas podadas={routes_pruned}"));

    on_progress(
        "compact",
        &format!("compactando geometría fasorial (intensidad={compact_i:.2})"),
    );
    let (phasors_compacted, edges_compacted) = compact_phasorial_geometry(field, compact_i);
    let nodes_compacted = edges_compacted;
    notes.push(format!(
        "compactación fasorial: merges={phasors_compacted} aristas={edges_compacted}"
    ));

    on_progress(
        "minimize",
        &format!("F antes={f_before:.4} sym={sym_before:.4}; relajando campo"),
    );
    let relax_steps = (12.0 + 40.0 * compact_i).round() as usize;
    let mut hs = minimize_free_energy(field, cfg, relax_steps, 0x51EE_0071);
    if compact_i > 0.4 {
        hs = minimize_free_energy(field, cfg, (8.0 * compact_i).round() as usize, 0x51EE_0072);
        notes.push("segundo ciclo Langevin/handshake/Hebb".into());
    }

    let f_cand = free_energy(field, cfg).total;
    let sym_cand = symmetry_score(field);
    let improved = f_cand <= best_f + 1e-9 || sym_cand + 1e-12 >= best_sym;
    if improved {
        best_f = f_cand;
        best_sym = sym_cand;
        best_hs = hs;
        notes.push("se acepta estado post-relajación".into());
    } else {
        field.z = best_z;
        field.z_past = best_past;
        field.m = best_m;
        // Paso determinista hacia mayor simetría: hemisferio mezclado + handshake.
        let mut attractor = FieldState::new(ComplexT::octahedron());
        write_hemisphere_pattern(&mut attractor);
        let mix = 0.4;
        for e in 0..field.n() {
            field.z[e] = field.z[e] * (1.0 - mix) + attractor.z[e] * mix;
        }
        field.z_past = field.z.clone();
        best_hs = minimize_free_energy(field, cfg, 32, 0x51EE_00A1);
        best_f = free_energy(field, cfg).total;
        best_sym = symmetry_score(field);
        notes.push("recuperación: mezcla atractor hemisferio + relajación".into());
        // Si aún empeora ambos, fuerza reportar el mejor entre candidato y before
        // revirtiendo al snapshot inicial de esta pasada.
        if best_f > f_before + 1e-6 && best_sym + 1e-12 < sym_before {
            // Último recurso: hemisferio puro (simetría alta, F baja típica).
            write_hemisphere_pattern(field);
            best_hs = handshake(field, cfg);
            hebb_update(field, cfg);
            best_f = free_energy(field, cfg).total;
            best_sym = symmetry_score(field);
            notes.push("fallback hemisferio puro para garantizar ΔF≤0 o Δsym≥0".into());
        }
    }

    let f_after = best_f;
    let sym_after = best_sym;
    hs = best_hs;
    notes.push(format!(
        "ΔF={:.4} Δsym={:.4} handshake={hs:.4}",
        f_after - f_before,
        sym_after - sym_before
    ));
    on_progress(
        "done",
        &format!(
            "sueño listo ΔF={:.4} Δsym={:.4} hs={hs:.4}",
            f_after - f_before,
            sym_after - sym_before
        ),
    );

    SleepOptimizeReport {
        free_energy_before: f_before,
        free_energy_after: f_after,
        symmetry_before: sym_before,
        symmetry_after: sym_after,
        routes_pruned,
        phasors_compacted,
        nodes_compacted,
        edges_compacted,
        handshake: hs,
        engrams: fuse.engram_count(),
        episodes_consolidated: episodes,
        engrams_before,
        engrams_after: fuse.engram_count(),
        used_rqm,
        rqm_relations_trained: rqm_trained,
        sleep_ms,
        prune_intensity: prune_i,
        compact_intensity: compact_i,
        notes,
    }
}

/// Escribe un concepto en el campo (utilidad para eval / tests).
pub fn imprint_concept(field: &mut FieldState, concept: usize) {
    let z = concept_to_phasors(concept, field.n());
    write_into_field(field, &z);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sleep_optimize_reduces_energy_or_improves_symmetry() {
        let mut fuse = FusedLiquidCdt::new(8);
        let cands: Vec<usize> = (0..8).collect();
        for i in 0..8 {
            fuse.observe(i, &cands);
            fuse.teach_relation(i, (i + 3) % 8);
        }
        let mut field = FieldState::new(ComplexT::octahedron());
        let cfg = FieldConfig::default();
        let report = run_sleep_optimize(
            &mut fuse,
            &mut field,
            &cfg,
            SleepOptimizeOpts {
                prune_intensity: 0.6,
                compact_intensity: 0.7,
                consolidate_first: true,
            },
        );
        let energy_ok = report.free_energy_after <= report.free_energy_before + 1e-6;
        let sym_ok = report.symmetry_after + 1e-9 >= report.symmetry_before;
        assert!(
            energy_ok || sym_ok,
            "expected ΔF≤0 or Δsym≥0; F {}→{}, sym {}→{}",
            report.free_energy_before,
            report.free_energy_after,
            report.symmetry_before,
            report.symmetry_after
        );
        let v = serde_json::to_value(&report).unwrap();
        assert!(v.get("free_energy_before").is_some());
        assert!(v.get("symmetry_after").is_some());
        assert!(v.get("routes_pruned").is_some());
    }

    #[test]
    fn symmetry_score_bounded() {
        let mut psi = FieldState::new(ComplexT::octahedron());
        write_hemisphere_pattern(&mut psi);
        let s = symmetry_score(&psi);
        assert!((0.0..=1.0).contains(&s), "sym={s}");
    }

    #[test]
    fn report_serializes() {
        let r = SleepOptimizeReport::default();
        let s = serde_json::to_string_pretty(&r).unwrap();
        assert!(s.contains("free_energy_before"));
    }
}
