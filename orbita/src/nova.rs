#![allow(dead_code)]
//! ÓRBITA-Nova (docs/orbita_nova/): C3 filtro de innovación, C4 adaptador rango-r,
//! C5 árbitro de verificación, C6 biblioteca consolidada, C7 compuerta.
//! Inferencia sin entrenar: GN/LM por episodio; núcleo ÓRBITA congelado.
//!
//!   nova-eval --weights W --seed S [--n N] [--k K] [--rank R] [--lam L]
//!             [--noise SIG] [--k-fit KF] [--out DIR] [--prepare-train]
//!   nova-eval --ceiling --weights W --seed S [--n-per-fam N] [--rank R] [--out DIR]
//!   nova-eval --calibrate-tau --wdir DIR --out DIR
//!   nova-eval --full --phase P --weights W --seed S --tau T --out DIR [--smoke]
//!
//! Fase 0: episodios generados fam 0–7 (contexto verdadero).
//! Fase 1 (--ceiling): techo ORACLE_EP / ORACLE_FAMNET / C4_EP por familia nueva 8–11 y k∈{2,4,8}.
//! No lee ds.test. No toca run_v6 ni bytes/idiomas.
use crate::fam::{reid, rule, NF_NEW};
use crate::latent::fit_gn;
use crate::learn::Module;
use crate::model::{CtxParams, Orbita, M};
use cdt_rqm_epr::field_autonomy_v4::K_POINTS;
use cdt_rqm_epr::v4_dataset::{apply_rule, context_of, make_object, ObjectClass, N_FAM};
use cdt_rqm_epr::v4_metrics as m;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::io::Write;
use std::path::PathBuf;

/// Rango por defecto del adaptador C4 (ΔK = U Vᵀ en R^{M×M}).
pub const DEFAULT_RANK: usize = 2;
/// χ² 0,95 con dof = 2·K_POINTS (observación completa del objeto).
fn chi2_95(dof: usize) -> f64 {
    // cuantiles aproximados (Wilson–Hilferty / tabla); dof típico 12 → 21.026
    match dof {
        2 => 5.991,
        4 => 9.488,
        6 => 12.592,
        8 => 15.507,
        10 => 18.307,
        12 => 21.026,
        14 => 23.685,
        16 => 26.296,
        _ => {
            let d = dof as f64;
            d * (1.0 - 2.0 / (9.0 * d) + 1.64485 * (2.0 / (9.0 * d)).sqrt()).powi(3)
        }
    }
}

fn fin(z: &[f64]) -> bool {
    z.iter().all(|v| v.is_finite())
}
fn argf(a: &[String], k: &str, d: f64) -> f64 {
    super::arg(a, k).map(|s| s.parse().unwrap()).unwrap_or(d)
}
fn argu(a: &[String], k: &str, d: usize) -> usize {
    super::arg(a, k).map(|s| s.parse().unwrap()).unwrap_or(d)
}

// ─── C4: adaptador de bajo rango ─────────────────────────────────────────────

/// Nº de parámetros: 2 · r · M (U y V, cada uno M×r column-major).
pub fn npar_lr(r: usize) -> usize {
    2 * r * M
}

fn uv_from(th: &[f64], r: usize) -> (Vec<f64>, Vec<f64>) {
    let n = r * M;
    (th[..n].to_vec(), th[n..].to_vec())
}

/// u ← u + U (Vᵀ z); U,V ∈ R^{M×r} column-major.
fn apply_delta(u: &mut [f64], z: &[f64], th: &[f64], r: usize) {
    let (uu, vv) = uv_from(th, r);
    for s in 0..r {
        let mut dot = 0.0;
        for i in 0..M {
            dot += vv[s * M + i] * z[i];
        }
        for i in 0..M {
            u[i] += uu[s * M + i] * dot;
        }
    }
}

/// Paso ÓRBITA + ΔK de rango r: z=enc(q); u=core(z)+UVᵀz; q'=Brenier(u)[0..2].
pub fn lr_step(d: &Orbita, cp: &CtxParams, c: &[f64], th: &[f64], r: usize, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for i in 0..x.len() / 2 {
        let z = d.encode(&x[2 * i..2 * i + 2], c);
        let mut u = d.core(cp, &z);
        apply_delta(&mut u, &z, th, r);
        let w = d.correct(cp, &u);
        out[2 * i] = w[0];
        out[2 * i + 1] = w[1];
    }
    out
}

/// Norma de Frobenius de UVᵀ (proxy de ‖ΔK‖).
fn frobenius_uv(th: &[f64], r: usize) -> f64 {
    let (uu, vv) = uv_from(th, r);
    // ‖UVᵀ‖_F² = ‖UᵀU ⊙ VᵀV‖… más simple: tr((UVᵀ)(VUᵀ)) = ‖VᵀU‖_F² wait
    // ‖UVᵀ‖_F² = sum_{ij} (sum_s U_is V_js)²
    let mut s = 0.0;
    for i in 0..M {
        for j in 0..M {
            let mut e = 0.0;
            for t in 0..r {
                e += uu[t * M + i] * vv[t * M + j];
            }
            s += e * e;
        }
    }
    s.sqrt()
}

/// Radio espectral aproximado de I+ΔK vía power iteration en R^M (sólo ΔK).
pub(crate) fn spectral_delta(th: &[f64], r: usize) -> f64 {
    let (uu, vv) = uv_from(th, r);
    let mut x = vec![1.0 / (M as f64).sqrt(); M];
    let mut nrm = 1.0;
    for _ in 0..20 {
        let mut y = vec![0.0; M];
        for s in 0..r {
            let mut dot = 0.0;
            for i in 0..M {
                dot += vv[s * M + i] * x[i];
            }
            for i in 0..M {
                y[i] += uu[s * M + i] * dot;
            }
        }
        // y = ΔK x; queremos ‖I+ΔK‖ ≈ 1 + ‖ΔK‖ en esta dir.
        for i in 0..M {
            y[i] += x[i];
        }
        nrm = m::norm(&y).max(1e-15);
        for i in 0..M {
            x[i] = y[i] / nrm;
        }
    }
    nrm
}

// ─── C3: filtro de innovación (EKF-lite) ─────────────────────────────────────

pub struct InnovFilter {
    pub p_scale: f64, // P = p_scale · I (covarianza escalar del estado)
    pub q_proc: f64,  // ruido de proceso
    pub r_obs: f64,   // ruido de observación (σ²)
}

impl InnovFilter {
    pub fn new(r_obs: f64) -> Self {
        Self {
            p_scale: r_obs,
            q_proc: r_obs * 0.1,
            r_obs,
        }
    }

    /// Un paso predict–update. Devuelve (estado corregido, NIS, dentro_de_χ²_95).
    pub fn update(&mut self, d: &Orbita, c: &[f64], x: &[f64], y: &[f64]) -> (Vec<f64>, f64, bool) {
        let xhat = d.step(x, c);
        let n = x.len();
        // Jacobiano F ≈ ∂step/∂x (aprox. por ‖σ_max‖ → escala P)
        let sig = m::sigma_max(&m::jacobian(|w| d.step(w, c), x)).max(1e-6);
        self.p_scale = (sig * sig * self.p_scale + self.q_proc).min(1e2);
        let mut nu = vec![0.0; n];
        for i in 0..n {
            nu[i] = y[i] - xhat[i];
        }
        // S = P + R  (H = I); NIS = νᵀ S⁻¹ ν = ‖ν‖² / (p+r)
        let s = self.p_scale + self.r_obs;
        let nis = nu.iter().map(|v| v * v).sum::<f64>() / s;
        let thr = chi2_95(n);
        let ok = nis <= thr;
        // ganancia K = P / (P+R); x ← x̂ + K ν
        let gain = self.p_scale / s;
        let mut xnew = xhat.clone();
        for i in 0..n {
            xnew[i] += gain * nu[i];
        }
        self.p_scale = (1.0 - gain) * self.p_scale;
        (xnew, nis, ok)
    }
}

// ─── C5: árbitro de verificación ─────────────────────────────────────────────

pub struct Verdict {
    pub accept: bool,
    pub fit_err: f64,
    pub val_err: f64,
    pub base_val: f64,
    pub spectral: f64,
    pub reason: &'static str,
}

/// Conserva el adaptador sólo si mejora k_val frente a ÓRBITA y ‖I+ΔK‖ ≤ 1+ε.
pub fn arbitrate(
    d: &Orbita,
    cp: &CtxParams,
    c: &[f64],
    th: &[f64],
    r: usize,
    x0: &[f64],
    ys: &[Vec<f64>],
    k_fit: usize,
    k_val: usize,
    eps_spec: f64,
) -> Verdict {
    let k = k_fit + k_val;
    let roll_err = |step: &dyn Fn(&[f64]) -> Vec<f64>, from: usize, n: usize| -> f64 {
        let mut z = if from == 0 {
            x0.to_vec()
        } else {
            ys[from - 1].clone()
        };
        let mut e = 0.0;
        for t in 0..n {
            z = step(&z);
            let tgt = &ys[from + t];
            e += if fin(&z) {
                m::rel_err(&z, tgt).min(10.0)
            } else {
                10.0
            };
        }
        e / n.max(1) as f64
    };
    let base = |z: &[f64]| d.step(z, c);
    let ada = |z: &[f64]| lr_step(d, cp, c, th, r, z);
    let fit_err = roll_err(&ada, 0, k_fit);
    let val_err = if k_val > 0 {
        roll_err(&ada, k_fit, k_val)
    } else {
        // sin val: usa el propio fit como proxy (fase 0 con k pequeño)
        fit_err
    };
    let base_val = if k_val > 0 {
        roll_err(&base, k_fit, k_val)
    } else {
        roll_err(&base, 0, k)
    };
    let spectral = spectral_delta(th, r);
    if spectral > 1.0 + eps_spec {
        return Verdict {
            accept: false,
            fit_err,
            val_err,
            base_val,
            spectral,
            reason: "spectral",
        };
    }
    if val_err < base_val - 1e-12 {
        Verdict {
            accept: true,
            fit_err,
            val_err,
            base_val,
            spectral,
            reason: "improve",
        }
    } else {
        Verdict {
            accept: false,
            fit_err,
            val_err,
            base_val,
            spectral,
            reason: "no_improve",
        }
    }
}

// ─── C6: biblioteca consolidada (stub listo para train/consolidación) ─────────

#[derive(Clone, Default)]
pub struct OperatorLib {
    pub entries: Vec<LibEntry>,
}

#[derive(Clone)]
pub struct LibEntry {
    pub signature: Vec<f64>, // coeficientes ΔK aplanados (o UV)
    pub th: Vec<f64>,
    pub n_support: usize,
}

impl OperatorLib {
    pub fn new() -> Self {
        Self::default()
    }
    /// Recuperación con umbral; None = «ninguno» (opción nula).
    pub fn retrieve(&self, sig: &[f64], tau: f64) -> Option<&LibEntry> {
        if self.entries.is_empty() {
            return None;
        }
        let mut best: Option<(f64, &LibEntry)> = None;
        for e in &self.entries {
            let d: f64 = e
                .signature
                .iter()
                .zip(sig)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if d <= tau && best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, e));
            }
        }
        best.map(|(_, e)| e)
    }
    /// Admisión (stub fase 0): rechaza siempre. Preferir `admit` en train.
    pub fn admit_stub(&mut self, _e: LibEntry, _held_ok: bool, _no_damage: bool) -> bool {
        false
    }
    /// Admisión real: sólo si mejora retenida y no-daño. Empuja a `entries`.
    pub fn admit(&mut self, e: LibEntry, held_ok: bool, no_damage: bool) -> bool {
        if held_ok && no_damage {
            self.entries.push(e);
            true
        } else {
            false
        }
    }
    /// Distancia L2 de firmas (misma longitud; trunca al mínimo).
    pub fn sig_dist(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return f64::INFINITY;
        }
        a[..n]
            .iter()
            .zip(&b[..n])
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }
    /// Permuta firmas entre entradas (control NOVA_SHUF); `th` intacto.
    pub fn shuffled_signatures(&self, seed: u64) -> OperatorLib {
        use rand::seq::SliceRandom;
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256StarStar;
        let mut out = self.clone();
        if out.entries.len() < 2 {
            return out;
        }
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x5A1F_51C6u64);
        let mut sigs: Vec<Vec<f64>> = out.entries.iter().map(|e| e.signature.clone()).collect();
        sigs.shuffle(&mut rng);
        for (e, s) in out.entries.iter_mut().zip(sigs) {
            e.signature = s;
        }
        out
    }
    /// Serializa biblioteca a JSON (checkpoint/resume).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "entries": self.entries.iter().map(|e| serde_json::json!({
                "signature": e.signature,
                "th": e.th,
                "n_support": e.n_support,
            })).collect::<Vec<_>>()
        })
    }
    pub fn from_json(v: &serde_json::Value) -> Self {
        let mut lib = Self::new();
        if let Some(arr) = v.get("entries").and_then(|x| x.as_array()) {
            for e in arr {
                let sig = e.get("signature").and_then(|x| x.as_array()).map(|a| {
                    a.iter().filter_map(|z| z.as_f64()).collect::<Vec<_>>()
                }).unwrap_or_default();
                let th = e.get("th").and_then(|x| x.as_array()).map(|a| {
                    a.iter().filter_map(|z| z.as_f64()).collect::<Vec<_>>()
                }).unwrap_or_default();
                let n_support = e.get("n_support").and_then(|x| x.as_u64()).unwrap_or(1) as usize;
                lib.entries.push(LibEntry { signature: sig, th, n_support });
            }
        }
        lib
    }
}

// ─── C7: compuerta de no-daño (si el candidato no gana a ÓRBITA+C3 → base) ───

pub fn gate_nodamage(candidate_err: f64, baseline_err: f64) -> bool {
    candidate_err < baseline_err - 1e-12
}

// ─── Hooks de train (no-op documentados; listos para cablear) ─────────────────

/// Configuración de un futuro bucle de train. En fase 0 todos los campos son no-op.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub enabled: bool,
    pub max_steps: usize,
    pub lr: f64,
    pub consolidate_every: usize,
    pub checkpoint_dir: PathBuf,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_steps: 0,
            lr: 0.0,
            consolidate_every: 0,
            checkpoint_dir: PathBuf::from("artifacts/orbita/nova_train"),
        }
    }
}

impl TrainConfig {
    pub fn from_toml_stub(path: &str) -> Self {
        // Parser mínimo: lee enabled = true/false; resto documentado como no-op.
        let mut cfg = Self::default();
        if let Ok(txt) = std::fs::read_to_string(path) {
            for line in txt.lines() {
                let l = line.split('#').next().unwrap_or("").trim();
                if let Some(v) = l.strip_prefix("enabled") {
                    let v = v.trim().trim_start_matches('=').trim();
                    cfg.enabled = v == "true";
                }
            }
        }
        cfg
    }
}

/// Entrada de train futuro. **No-op** mientras `enabled == false` (fase 0).
pub fn train_stub(cfg: &TrainConfig) -> serde_json::Value {
    if !cfg.enabled {
        return serde_json::json!({
            "status": "noop",
            "reason": "fase0_sin_entrenar",
            "enabled": false,
            "hint": "Poner enabled=true en orbita/configs/nova_train.toml y cablear el bucle (no implementado aún)."
        });
    }
    // Reservado: no ejecutar SGD ni tocar run_v6.
    serde_json::json!({
        "status": "refused",
        "reason": "train_loop_not_wired",
        "enabled": true
    })
}

pub fn write_train_stubs(dir: &PathBuf) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let cfg = dir.join("nova_train.toml");
    if !cfg.exists() {
        std::fs::write(
            &cfg,
            r#"# ÓRBITA-Nova — config de train (FASE 0: no-op)
# No activar hasta pasar fase 0 y preregistro. No toca run_v6 ni Gemma.
enabled = false
max_steps = 0
lr = 0.0
consolidate_every = 0
checkpoint_dir = "artifacts/orbita/nova_train"
# rank = 2
# lam_prior = 1e-4
# Nota: el ajuste actual es GN/LM por episodio (C4), sin SGD.
"#,
        )?;
    }
    let hook = dir.join("TRAIN_HOOKS.md");
    if !hook.exists() {
        std::fs::write(
            &hook,
            r#"# Hooks de train (Nova)

Estado fase 0: **stubs / no-op**.

| hook | símbolo | estado |
|---|---|---|
| CLI `--prepare-train` | escribe configs | listo |
| CLI `--train` | `train_stub` | no-op si `enabled=false` |
| Consolidación C6 | `OperatorLib::admit_stub` | rechaza siempre |
| Compuerta C7 | `gate_nodamage` | lógica real, sin train |
| Ajuste C4 | `fit_gn` por episodio | **activo** (sin SGD) |

No reentrenar el núcleo ÓRBITA ni Gemma. No invocar `run_v6`.
"#,
        )?;
    }
    Ok(())
}

// ─── Episodios generados (distribución propia, fam 0–7) ───────────────────────

pub struct Episode {
    pub fam: usize,
    pub p: [f64; 2],
    pub c: Vec<f64>,
    pub x0: Vec<f64>,
    pub ys: Vec<Vec<f64>>, // y_1 .. y_H
}

pub fn gen_episodes(seed: u64, n: usize, horizon: usize) -> Vec<Episode> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x4E0A_0007u64);
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let fam = i % N_FAM;
        let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
        let c = context_of(fam, p);
        let x0 = make_object(&mut rng, K_POINTS, cls[i % 3]);
        let mut ys = Vec::with_capacity(horizon);
        let mut y = x0.clone();
        for _ in 0..horizon {
            y = apply_rule(fam, p, &y);
            ys.push(y.clone());
        }
        out.push(Episode { fam, p, c, x0, ys });
    }
    out
}

fn rel_at(step: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], ys: &[Vec<f64>], h: usize) -> f64 {
    let mut z = x0.to_vec();
    for t in 0..h {
        z = step(&z);
        if t + 1 == h {
            return if fin(&z) {
                m::rel_err(&z, &ys[h - 1]).min(10.0)
            } else {
                10.0
            };
        }
    }
    10.0
}

fn log_rel(e: f64) -> f64 {
    (e.max(1e-12)).ln()
}

// ─── Evaluación por seed (fase 0) ────────────────────────────────────────────

pub struct SeedReport {
    pub seed: u64,
    pub n: usize,
    pub k: usize,
    pub rank: usize,
    pub orbita_med_log: f64,
    pub c4_med_log: f64,
    pub nova_med_log: f64, // C3+C4+C5+C7
    pub delta_c4: f64,     // c4 − orbita (negativo = mejora)
    pub delta_nova: f64,
    pub c4_wins: bool,
    pub nova_wins: bool,
    pub nis_coverage: f64,
    pub nis_n: usize,
    pub accept_rate: f64,
    pub orbita_acc16: f64,
    pub c4_acc16: f64,
    pub nova_acc16: f64,
}

pub fn eval_seed(
    d: &Orbita,
    seed: u64,
    n: usize,
    k: usize,
    rank: usize,
    lam: f64,
    noise: f64,
    k_fit: usize,
) -> SeedReport {
    let eps = gen_episodes(seed, n, 16);
    let k_fit = if k_fit == 0 || k_fit >= k {
        // prereg hint k=4 → 3/1
        (k.saturating_sub(1)).max(1)
    } else {
        k_fit
    };
    let k_val = k - k_fit;

    let mut log_o = vec![];
    let mut log_c4 = vec![];
    let mut log_nv = vec![];
    let mut acc_o = 0usize;
    let mut acc_c4 = 0usize;
    let mut acc_nv = 0usize;
    let mut nis_ok = 0usize;
    let mut nis_n = 0usize;
    let mut accepts = 0usize;

    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xC30000);

    for e in &eps {
        let cp = d.ctx(&e.c);
        // —— C3: filtro con ruido sintético (calibración NIS bajo H0 local) ——
        if noise > 0.0 {
            let mut filt = InnovFilter::new(noise * noise);
            for t in 0..k.min(e.ys.len()) {
                let xprev = if t == 0 { e.x0.clone() } else { e.ys[t - 1].clone() };
                let mut y = e.ys[t].clone();
                // ruido gaussiano aprox. Box-Muller barato vía suma de uniformes
                for v in y.iter_mut() {
                    let u: f64 = (0..6).map(|_| rng.gen::<f64>()).sum::<f64>() / 6.0;
                    *v += (u - 0.5) * 2.0 * 3.0f64.sqrt() * noise;
                }
                let (_xn, _nis, ok) = filt.update(d, &e.c, &xprev, &y);
                nis_n += 1;
                if ok { nis_ok += 1; }
            }
        }

        // —— Baseline ÓRBITA (contexto verdadero, free-run h16) ——
        let err_o = rel_at(&|z| d.step(z, &e.c), &e.x0, &e.ys, 16);
        log_o.push(log_rel(err_o));
        if err_o < 0.05 {
            acc_o += 1;
        }

        // —— C4: ajuste GN de ΔK (init LoRA: U~ε, V=0 para Jacobiano ≠ 0) ——
        let mut prior = vec![0.0; npar_lr(rank)];
        {
            // U en la primera mitad; escala pequeña dependiente del episodio
            let mut prng = Xoshiro256StarStar::seed_from_u64(
                seed ^ ((e.fam as u64) << 32) ^ (prior.len() as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
            let nu = rank * M;
            for i in 0..nu {
                let u: f64 = (0..6).map(|_| prng.gen::<f64>()).sum::<f64>() / 6.0;
                prior[i] = (u - 0.5) * 2.0 * 3.0f64.sqrt() * 1e-3;
            }
            // V queda en 0
        }
        let th = fit_gn(
            &|th, z| lr_step(d, &cp, &e.c, th, rank, z),
            &prior,
            lam,
            &e.x0,
            &e.ys,
            k,
        );
        let err_c4 = rel_at(&|z| lr_step(d, &cp, &e.c, &th, rank, z), &e.x0, &e.ys, 16);
        log_c4.push(log_rel(err_c4));
        if err_c4 < 0.05 {
            acc_c4 += 1;
        }

        // —— C5 + C7: aceptar adaptador sólo si mejora val y es estable ——
        let ver = arbitrate(d, &cp, &e.c, &th, rank, &e.x0, &e.ys, k_fit, k_val, 0.5);
        let use_ada = ver.accept && gate_nodamage(ver.val_err, ver.base_val);
        if use_ada {
            accepts += 1;
        }
        let err_nv = if use_ada { err_c4 } else { err_o };
        log_nv.push(log_rel(err_nv));
        if err_nv < 0.05 {
            acc_nv += 1;
        }

        let _ = frobenius_uv(&th, rank);
    }

    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        m::median(v)
    };
    let mo = med(&mut log_o);
    let mc = med(&mut log_c4);
    let mn = med(&mut log_nv);
    SeedReport {
        seed,
        n,
        k,
        rank,
        orbita_med_log: mo,
        c4_med_log: mc,
        nova_med_log: mn,
        delta_c4: mc - mo,
        delta_nova: mn - mo,
        c4_wins: mc < mo - 1e-15,
        nova_wins: mn < mo - 1e-15,
        nis_coverage: if nis_n > 0 {
            nis_ok as f64 / nis_n as f64
        } else {
            f64::NAN
        },
        nis_n,
        accept_rate: accepts as f64 / n as f64,
        orbita_acc16: acc_o as f64 / n as f64,
        c4_acc16: acc_c4 as f64 / n as f64,
        nova_acc16: acc_nv as f64 / n as f64,
    }
}


// ─── Fase 1: techo ORACLE_EP / FAMNET por familia nueva ───────────────────────

/// Familias nuevas (no afines) del plan Nova / crece.
pub const NEW_FAMS: [usize; 4] = [8, 9, 10, 11];
/// k a medir (primario = 4).
pub const CEILING_KS: [usize; 3] = [2, 4, 8];
const OEP_DEG: usize = 2;
const OEP_LAM: f64 = 1e-4;

/// Episodios generados sólo en familias nuevas (dinámica `fam::rule`).
pub fn gen_new_episodes(seed: u64, n_per_fam: usize, horizon: usize) -> Vec<Episode> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xF1A5_E001u64);
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let mut out = Vec::with_capacity(n_per_fam * NEW_FAMS.len());
    for (fi, &fam) in NEW_FAMS.iter().enumerate() {
        for i in 0..n_per_fam {
            let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
            // contexto «falso» one-hot de fam 0 + p verdadero (ÓRBITA no tiene slots 8–11);
            // el baseline honesto usa reid, no este c.
            let c = context_of(0, p);
            let x0 = make_object(&mut rng, K_POINTS, cls[(fi * n_per_fam + i) % 3]);
            let mut ys = Vec::with_capacity(horizon);
            let mut y = x0.clone();
            for _ in 0..horizon {
                y = rule(fam, p, &y);
                ys.push(y.clone());
            }
            out.push(Episode { fam, p, c, x0, ys });
        }
    }
    let _ = NF_NEW; // documenta el techo 12 familias
    out
}

fn med_f64(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    m::median(&v.to_vec())
}

fn lora_prior(seed: u64, fam: usize, rank: usize) -> Vec<f64> {
    let mut prior = vec![0.0; npar_lr(rank)];
    let mut prng = Xoshiro256StarStar::seed_from_u64(
        seed ^ ((fam as u64) << 32) ^ (prior.len() as u64).wrapping_mul(0x9E3779B97F4A7C15),
    );
    let nu = rank * M;
    for i in 0..nu {
        let u: f64 = (0..6).map(|_| prng.gen::<f64>()).sum::<f64>() / 6.0;
        prior[i] = (u - 0.5) * 2.0 * 3.0f64.sqrt() * 1e-3;
    }
    prior
}

/// Métricas agregadas por (familia, k) en una seed.
#[derive(Clone, Debug)]
pub struct FamKCeil {
    pub fam: usize,
    pub k: usize,
    pub n: usize,
    pub orbita_med_log: f64,
    pub oep_med_log: f64,
    pub c4_med_log: f64,
    pub famnet_med_log: f64,
    pub orbita_acc16: f64,
    pub oep_acc16: f64,
    pub c4_acc16: f64,
    pub famnet_acc16: f64,
    pub delta_oep: f64,     // oep − orbita (negativo = mejora)
    pub delta_c4: f64,
    pub delta_famnet: f64,
    pub best_acc16: f64,    // max(oep, c4, famnet)
    pub best_delta_log: f64, // min Δ log entre oráculos
}

/// ¿Techo en suelo? Criterio documentado (optimista): acc16 del mejor oráculo ≤ ε
/// y ninguna mejora continua Δlog < −δ frente a ÓRBITA.
pub fn is_floor(c: &FamKCeil, eps_acc: f64, delta_cont: f64) -> bool {
    c.best_acc16 <= eps_acc && c.best_delta_log >= -delta_cont
}

pub fn eval_ceiling_seed(
    d: &Orbita,
    seed: u64,
    n_per_fam: usize,
    rank: usize,
    lam_c4: f64,
) -> Vec<FamKCeil> {
    let eps = gen_new_episodes(seed, n_per_fam, 16);
    let mut out = Vec::with_capacity(NEW_FAMS.len() * CEILING_KS.len());

    for &k in &CEILING_KS {
        // —— ORACLE_FAMNET: un Module(None) por familia, todos los episodios (etiqueta privilegiada) ——
        let mut fam_mods: std::collections::HashMap<usize, Module> = std::collections::HashMap::new();
        for e in &eps {
            let m = fam_mods
                .entry(e.fam)
                .or_insert_with(|| Module::new(None, OEP_DEG, OEP_LAM));
            m.add(d, &e.x0, &e.ys, k, e.p, e.fam);
        }

        for &fam in &NEW_FAMS {
            let group: Vec<&Episode> = eps.iter().filter(|e| e.fam == fam).collect();
            let n = group.len();
            let mut log_o = Vec::with_capacity(n);
            let mut log_ep = Vec::with_capacity(n);
            let mut log_c4 = Vec::with_capacity(n);
            let mut log_fn = Vec::with_capacity(n);
            let mut acc_o = 0usize;
            let mut acc_ep = 0usize;
            let mut acc_c4 = 0usize;
            let mut acc_fn = 0usize;

            for e in &group {
                // ÓRBITA honesta en nuevas: reid a g∈0..7 con p̂ = p verdadero
                let (g, pp, _) = reid(d, &e.x0, &e.ys, k, e.p);
                let cg = context_of(g, pp);
                let err_o = rel_at(&|z| d.step(z, &cg), &e.x0, &e.ys, 16);
                log_o.push(log_rel(err_o));
                if err_o < 0.05 {
                    acc_o += 1;
                }

                // ORACLE_EP: residual Φ por episodio (identidad + WᵀΦ)
                let mut oep = Module::new(None, OEP_DEG, OEP_LAM);
                oep.add(d, &e.x0, &e.ys, k, e.p, e.fam);
                let err_ep = rel_at(&|z| oep.step(d, z, e.p), &e.x0, &e.ys, 16);
                log_ep.push(log_rel(err_ep));
                if err_ep < 0.05 {
                    acc_ep += 1;
                }

                // C4_EP: adaptador Nova por episodio (techo del stack C4, sin C5/C7)
                let cp0 = d.ctx(&e.c); // base g=0; el ΔK absorbe el residuo
                let prior = lora_prior(seed, e.fam, rank);
                let th = fit_gn(
                    &|th, z| lr_step(d, &cp0, &e.c, th, rank, z),
                    &prior,
                    lam_c4,
                    &e.x0,
                    &e.ys,
                    k,
                );
                let err_c4 = rel_at(&|z| lr_step(d, &cp0, &e.c, &th, rank, z), &e.x0, &e.ys, 16);
                log_c4.push(log_rel(err_c4));
                if err_c4 < 0.05 {
                    acc_c4 += 1;
                }

                // ORACLE_FAMNET
                let fnm = fam_mods.get(&e.fam).unwrap();
                let err_fn = rel_at(&|z| fnm.step(d, z, e.p), &e.x0, &e.ys, 16);
                log_fn.push(log_rel(err_fn));
                if err_fn < 0.05 {
                    acc_fn += 1;
                }
            }

            let mo = med_f64(&mut log_o);
            let mep = med_f64(&mut log_ep);
            let mc4 = med_f64(&mut log_c4);
            let mfn = med_f64(&mut log_fn);
            let ao = acc_o as f64 / n as f64;
            let aep = acc_ep as f64 / n as f64;
            let ac4 = acc_c4 as f64 / n as f64;
            let afn = acc_fn as f64 / n as f64;
            let d_ep = mep - mo;
            let d_c4 = mc4 - mo;
            let d_fn = mfn - mo;
            let best_acc = aep.max(ac4).max(afn);
            let best_d = d_ep.min(d_c4).min(d_fn);
            out.push(FamKCeil {
                fam,
                k,
                n,
                orbita_med_log: mo,
                oep_med_log: mep,
                c4_med_log: mc4,
                famnet_med_log: mfn,
                orbita_acc16: ao,
                oep_acc16: aep,
                c4_acc16: ac4,
                famnet_acc16: afn,
                delta_oep: d_ep,
                delta_c4: d_c4,
                delta_famnet: d_fn,
                best_acc16: best_acc,
                best_delta_log: best_d,
            });
        }
    }
    out
}

pub fn run_ceiling(a: &[String]) {
    let wpath = super::arg(a, "--weights").expect("--weights");
    let wtxt = std::fs::read_to_string(&wpath).unwrap();
    let d = Orbita::from_json(&wtxt);
    let seed = super::seed_of(a);
    let n_per_fam = argu(a, "--n-per-fam", 16);
    let rank = argu(a, "--rank", DEFAULT_RANK);
    let lam_c4 = argf(a, "--lam", 1e-6);
    let before = cdt_rqm_epr::v4_provenance::snapshot();

    let rows = eval_ceiling_seed(&d, seed, n_per_fam, rank, lam_c4);
    let after = Orbita::from_json(&wtxt);
    assert_eq!(after.raw, d.raw, "núcleo ÓRBITA modificado — prohibido en Nova");

    let table: Vec<serde_json::Value> = rows
        .iter()
        .map(|c| {
            serde_json::json!({
                "fam": c.fam,
                "k": c.k,
                "n": c.n,
                "orbita_med_log": c.orbita_med_log,
                "oep_med_log": c.oep_med_log,
                "c4_med_log": c.c4_med_log,
                "famnet_med_log": c.famnet_med_log,
                "orbita_acc16": c.orbita_acc16,
                "oep_acc16": c.oep_acc16,
                "c4_acc16": c.c4_acc16,
                "famnet_acc16": c.famnet_acc16,
                "delta_oep": c.delta_oep,
                "delta_c4": c.delta_c4,
                "delta_famnet": c.delta_famnet,
                "best_acc16": c.best_acc16,
                "best_delta_log": c.best_delta_log,
                "floor": c.k == 4 && is_floor(c, 0.02, 0.1),
            })
        })
        .collect();

    let out = serde_json::json!({
        "phase": 1,
        "seed": format!("0x{:X}", seed),
        "n_per_fam": n_per_fam,
        "rank": rank,
        "oep_deg": OEP_DEG,
        "oep_lam": OEP_LAM,
        "new_fams": NEW_FAMS.to_vec(),
        "ks": CEILING_KS.to_vec(),
        "floor_rule": "k=4 primario: best_acc16≤0.02 y best_delta_log≥−0.1 → suelo (excluir)",
        "rows": table,
        "queries": cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(),
        "weights": wpath,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());

    if let Some(dir) = super::arg(a, "--out") {
        let p = PathBuf::from(&dir);
        std::fs::create_dir_all(&p).unwrap();
        let mut f = std::fs::File::create(p.join(format!("fase1_0x{:X}.json", seed))).unwrap();
        writeln!(f, "{}", serde_json::to_string_pretty(&out).unwrap()).unwrap();
    }
    // resumen stderr por fam @ k=4
    for c in rows.iter().filter(|c| c.k == 4) {
        eprintln!(
            "0x{:X} fam{} k4 orb_acc={:.3} oep={:.3} c4={:.3} famnet={:.3} bestΔlog={:.3} floor={}",
            seed,
            c.fam,
            c.orbita_acc16,
            c.oep_acc16,
            c.c4_acc16,
            c.famnet_acc16,
            c.best_delta_log,
            is_floor(c, 0.02, 0.1)
        );
    }
}

pub fn run(a: &[String]) {
    if a.iter().any(|x| x == "--calibrate-tau" || x == "--full") {
        crate::nova_train::run(a);
        return;
    }
    if a.iter().any(|x| x == "--ceiling" || x == "--fase1") {
        run_ceiling(a);
        return;
    }
    if a.iter().any(|x| x == "--prepare-train") {
        let dir = PathBuf::from(
            super::arg(a, "--out").unwrap_or_else(|| "orbita/configs".into()),
        );
        write_train_stubs(&dir).unwrap();
        let cfg = TrainConfig::from_toml_stub(dir.join("nova_train.toml").to_str().unwrap());
        println!("{}", train_stub(&cfg));
        return;
    }
    if a.iter().any(|x| x == "--train") {
        let cfg_path = super::arg(a, "--config")
            .unwrap_or_else(|| "orbita/configs/nova_train.toml".into());
        let cfg = TrainConfig::from_toml_stub(&cfg_path);
        println!("{}", train_stub(&cfg));
        return;
    }

    let wpath = super::arg(a, "--weights").expect("--weights");
    let wtxt = std::fs::read_to_string(&wpath).unwrap();
    let d = Orbita::from_json(&wtxt);
    let seed = super::seed_of(a);
    let n = argu(a, "--n", 64);
    let k = argu(a, "--k", 4);
    let rank = argu(a, "--rank", DEFAULT_RANK);
    let lam = argf(a, "--lam", 1e-6);
    let noise = argf(a, "--noise", 0.01);
    let k_fit = argu(a, "--k-fit", 0); // 0 → auto 3/4
    let before = cdt_rqm_epr::v4_provenance::snapshot();

    let rep = eval_seed(&d, seed, n, k, rank, lam, noise, k_fit);
    let after = Orbita::from_json(&wtxt);
    assert_eq!(
        after.raw, d.raw,
        "núcleo ÓRBITA modificado — prohibido en Nova"
    );

    let out = serde_json::json!({
        "seed": format!("0x{:X}", rep.seed),
        "n": rep.n,
        "k": rep.k,
        "rank": rep.rank,
        "orbita_med_log_rel_h16": rep.orbita_med_log,
        "c4_med_log_rel_h16": rep.c4_med_log,
        "nova_med_log_rel_h16": rep.nova_med_log,
        "delta_c4": rep.delta_c4,
        "delta_nova": rep.delta_nova,
        "c4_wins": rep.c4_wins,
        "nova_wins": rep.nova_wins,
        "orbita_acc16": rep.orbita_acc16,
        "c4_acc16": rep.c4_acc16,
        "nova_acc16": rep.nova_acc16,
        "nis_coverage": rep.nis_coverage,
        "nis_n": rep.nis_n,
        "accept_rate": rep.accept_rate,
        "queries": cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(),
        "weights": wpath,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());

    if let Some(dir) = super::arg(a, "--out") {
        let p = PathBuf::from(&dir);
        std::fs::create_dir_all(&p).unwrap();
        let mut f = std::fs::File::create(p.join(format!("fase0_0x{:X}.json", seed))).unwrap();
        writeln!(f, "{}", serde_json::to_string_pretty(&out).unwrap()).unwrap();
    }
    eprintln!(
        "0x{:X} c4_wins={} Δc4={:.6} nis_cov={:.3} accept={:.2} queries={}",
        seed,
        rep.c4_wins,
        rep.delta_c4,
        rep.nis_coverage,
        rep.accept_rate,
        cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total()
    );
}
