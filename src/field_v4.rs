//! Motor de campo v4 para la aplicación (Chat · Entrenamiento · Sueño · Pruebas).
//!
//! Arquitectura única de `main` (ver `docs/resultados_v4.md`):
//!
//! * periferia/encoder = identidad sobre las coordenadas de los puntos (sin
//!   parámetros); decoder = identidad (no hay lookup posible);
//! * **Dφ** = campo residual local-lineal con compuerta (`Dphi::gated_h`,
//!   H = 136), entrenado con la receta pre-registrada §16: TRAIN real ∪
//!   *replay de confianza* del teacher consolidado desde la CDT (familias con
//!   residuo de ajuste < τ = 1 %);
//! * la CDT (experiencias) y el teacher se **borran** antes de evaluar; toda
//!   evaluación corre bajo el guard de procedencia (`v4_provenance`): cualquier
//!   consulta a memoria durante TEST marca la fila como `LEAKED`.
//!
//! Este módulo no depende de `web`: los tests del gate científico lo usan
//! directamente.

use crate::field_autonomy_v4::{
    collect_experience, consolidate, evaluate, run_e35_core, AuditFlags, Auditor, ConsolidateMode,
    ExpMode, ITERABLE_FAMS, K_POINTS,
};
use crate::v4_controls::{Dphi, Predictor, TrainBudget};
use crate::v4_dataset::{
    apply_rule, context_of, generate_and_seal, make_object, DatasetConfig, Example, ObjectClass,
    Sealed, CTX_DIM, FAMILIES, N_FAM,
};
use crate::v4_distill::{family_fit_residual, train_trusted_on, E_HIDDEN, TRUST_TAU};
use crate::v4_metrics as m;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Horizonte del rollout corto tipo E33 (rotate + reflect, free-run).
pub const ROLLOUT_H: usize = 8;
/// Pasos por defecto de un lote de entrenamiento en la app (la receta §16
/// completa son 256k; la app usa presupuestos cortos e incrementales).
pub const APP_TRAIN_STEPS: usize = 4000;
/// Pasos por defecto de una destilación de sueño.
pub const APP_SLEEP_STEPS: usize = 8000;
/// Tamaño de minibatch / lr de la receta §16.
pub const APP_BATCH: usize = 32;
pub const APP_LR: f64 = 3e-3;

pub fn budget(steps: usize) -> TrainBudget {
    TrainBudget {
        steps: steps.max(1),
        batch: APP_BATCH,
        lr: APP_LR,
    }
}

/// Dφ nuevo con la parametrización §16 (cabeza local-lineal con compuerta).
pub fn new_dphi(seed: u64) -> Dphi {
    Dphi::gated_h(2 * K_POINTS, CTX_DIM, E_HIDDEN, true, seed)
}

/// Métricas de un Dφ sobre un TEST sellado (un paso + rollout corto).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct V4Metrics {
    pub n: usize,
    /// Accuracy a un paso (rel_err < 5 %).
    pub acc: f64,
    /// Error relativo medio a un paso.
    pub rel_err_1step: f64,
    pub disp_cos: f64,
    /// Accuracy por familia (nombre, acc).
    pub fam_acc: Vec<(String, f64)>,
    /// Rollout free-run h = [`ROLLOUT_H`] sobre rotate+reflect.
    pub rollout_h: usize,
    pub rollout_acc: f64,
    pub rollout_rel_err: f64,
    /// Consultas a memoria (CDT/RQM/tablas/NN/…) durante TEST. Debe ser 0.
    pub test_queries: u64,
    /// Predicciones marcadas LEAKED por el auditor. Debe ser 0.
    pub leaked: usize,
}

pub fn eval_model(d: &Dphi, s: &Sealed, flags: &AuditFlags) -> V4Metrics {
    let aud = Auditor::new(s);
    let ev = evaluate(d, &s.ds.test, 1, Some((&aud, flags)), None);
    let iter: Vec<Example> =
        s.ds.test
            .iter()
            .filter(|e| ITERABLE_FAMS.contains(&e.family_id))
            .cloned()
            .collect();
    let ro = evaluate(d, &iter, ROLLOUT_H, Some((&aud, flags)), None);
    V4Metrics {
        n: ev.n,
        acc: ev.acc,
        rel_err_1step: ev.rel_err,
        disp_cos: ev.disp_cos,
        fam_acc: FAMILIES
            .iter()
            .zip(&ev.fam_acc)
            .map(|(f, a)| (f.to_string(), if a.is_nan() { 0.0 } else { *a }))
            .collect(),
        rollout_h: ROLLOUT_H,
        rollout_acc: ro.acc,
        rollout_rel_err: ro.rel_err,
        test_queries: ev.queries.total() + ro.queries.total(),
        leaked: ev.leaked + ro.leaked,
    }
}

pub fn sealed(seed: u64) -> Sealed {
    generate_and_seal(seed, &DatasetConfig::standard(K_POINTS))
}

// ─────────────────────────── entrenamiento ───────────────────────────

/// Resultado de un lote de entrenamiento (dataset → CDT → teacher → Dφ).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainBatchReport {
    pub seed: u64,
    pub dataset_hash: String,
    pub audit_status: String,
    pub n_train_batch: usize,
    /// Experiencias acumuladas en la CDT de vigilia (incluye este lote).
    pub cdt_experiences: usize,
    pub trusted_families: Vec<String>,
    pub fit_residual: Vec<(String, f64)>,
    pub steps: usize,
    pub train_ms: f64,
    /// Métricas en el TEST sellado de este lote (CDT/teacher ya borrados).
    pub metrics: V4Metrics,
    pub replay_hits_test: usize,
    pub params: usize,
    pub weights_hash: String,
}

/// Un lote: genera y sella un dataset v4, lo añade a la CDT de vigilia y
/// ajusta Dφ con la receta §16 sobre toda la experiencia acumulada.
pub fn train_batch(
    dphi: &mut Dphi,
    wake: &mut Vec<Example>,
    seed: u64,
    steps: usize,
) -> TrainBatchReport {
    let s = sealed(seed);
    wake.extend(s.ds.train.iter().cloned());
    let aud = Auditor::new(&s);
    let t0 = Instant::now();
    let (flags, trusted, res) =
        train_trusted_on(dphi, wake, seed, Some(&aud), TRUST_TAU, budget(steps));
    let train_ms = t0.elapsed().as_secs_f64() * 1e3;
    s.verify_test().expect("TEST sellado");
    let metrics = eval_model(dphi, &s, &flags);
    TrainBatchReport {
        seed,
        dataset_hash: s.manifest.manifest_hash(),
        audit_status: s.manifest.audit_status.to_string(),
        n_train_batch: s.ds.train.len(),
        cdt_experiences: wake.len(),
        trusted_families: fam_names(&trusted),
        fit_residual: fam_vals(&res),
        steps,
        train_ms,
        metrics,
        replay_hits_test: flags.equiv_hits.len() + usize::from(flags.any_equiv_global),
        params: dphi.params(),
        weights_hash: dphi.weights_hash(),
    }
}

fn fam_names(mask: &[bool]) -> Vec<String> {
    FAMILIES
        .iter()
        .zip(mask)
        .filter(|(_, t)| **t)
        .map(|(f, _)| f.to_string())
        .collect()
}

fn fam_vals(v: &[f64]) -> Vec<(String, f64)> {
    FAMILIES
        .iter()
        .zip(v)
        .map(|(f, r)| (f.to_string(), if r.is_finite() { *r } else { -1.0 }))
        .collect()
}

// ─────────────────────────── sueño ───────────────────────────

/// Sueño: consolida toda la CDT de vigilia en el teacher, destila Dφ con
/// replay de confianza y **borra** la CDT. Mide antes/después en un TEST
/// sellado nuevo (nunca visto por el teacher).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SleepReport {
    pub experiences_consolidated: usize,
    pub teacher_numbers: usize,
    pub trusted_families: Vec<String>,
    pub fit_residual: Vec<(String, f64)>,
    pub steps: usize,
    pub sleep_ms: f64,
    pub eval_seed: u64,
    pub before: V4Metrics,
    pub after: V4Metrics,
    /// La CDT y el teacher se borraron antes de evaluar.
    pub cdt_dropped: bool,
    pub weights_hash: String,
}

pub fn sleep_consolidate(
    dphi: &mut Dphi,
    wake: &mut Vec<Example>,
    eval_seed: u64,
    steps: usize,
) -> Option<SleepReport> {
    if wake.is_empty() {
        return None;
    }
    let s = sealed(eval_seed);
    let none = AuditFlags::default();
    let before = eval_model(dphi, &s, &none);
    let n = wake.len();
    let teacher_numbers = {
        let cdt = collect_experience(wake, ExpMode::Real, eval_seed);
        let sig = consolidate(&cdt, ConsolidateMode::Full);
        let _ = family_fit_residual(&sig, cdt.read_all());
        sig.n_numbers()
    };
    let aud = Auditor::new(&s);
    let t0 = Instant::now();
    let (flags, trusted, res) = train_trusted_on(
        dphi,
        wake,
        eval_seed ^ 0x51EE,
        Some(&aud),
        TRUST_TAU,
        budget(steps),
    );
    let sleep_ms = t0.elapsed().as_secs_f64() * 1e3;
    // Borrado tras aprender (E37): la experiencia episódica no sobrevive al sueño.
    wake.clear();
    wake.shrink_to_fit();
    let after = eval_model(dphi, &s, &flags);
    Some(SleepReport {
        experiences_consolidated: n,
        teacher_numbers,
        trusted_families: fam_names(&trusted),
        fit_residual: fam_vals(&res),
        steps,
        sleep_ms,
        eval_seed,
        before,
        after,
        cdt_dropped: true,
        weights_hash: dphi.weights_hash(),
    })
}

// ─────────────────────────── pruebas (smoke v4) ───────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmokeRow {
    pub test: String,
    pub seed: String,
    pub value: String,
    pub threshold: String,
    pub verdict: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SmokeReport {
    pub rows: Vec<SmokeRow>,
    pub pass: usize,
    pub fail: usize,
    pub elapsed_ms: f64,
    pub e35_mean_a: f64,
    pub e35_mean_b: f64,
    pub notes: Vec<String>,
}

impl SmokeReport {
    fn push(&mut self, test: &str, seed: u64, value: String, threshold: &str, ok: bool) {
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
        self.rows.push(SmokeRow {
            test: test.into(),
            seed: format!("0x{seed:X}"),
            value,
            threshold: threshold.into(),
            verdict: if ok { "PASS" } else { "FAIL" }.into(),
        });
    }
}

/// Seeds del smoke de la app / CI (subconjunto DEV, sin retune).
pub const SMOKE_SEEDS: [u64; 3] = [0xA400, 0xA401, 0xA402];

/// E35 (A vs B) con auditoría de memoria cero en `seeds`.
pub fn smoke_e35(seeds: &[u64], rep: &mut SmokeReport) {
    let (mut sa, mut sb) = (vec![], vec![]);
    for &seed in seeds {
        let s = sealed(seed);
        let o = run_e35_core(&s, seed, K_POINTS, "E35_SMOKE", false);
        sa.push(o.ev_a.acc);
        sb.push(o.ev_b.acc);
        rep.push(
            "E35 B−A (acc TEST)",
            seed,
            format!(
                "A={:.3} B={:.3} Δ={:+.3}",
                o.ev_a.acc,
                o.ev_b.acc,
                o.ev_b.acc - o.ev_a.acc
            ),
            "Δ > 0",
            o.ev_b.acc > o.ev_a.acc,
        );
        let q = o.ev_a.queries.total() + o.ev_b.queries.total();
        let leaked = o.ev_a.leaked + o.ev_b.leaked;
        rep.push(
            "auditoría memoria cero",
            seed,
            format!("queries={q} leaked={leaked} cdt_borrada={}", o.cdt_dropped),
            "queries=0 ∧ leaked=0",
            q == 0 && leaked == 0 && o.cdt_dropped,
        );
    }
    rep.e35_mean_a = m::mean(&sa);
    rep.e35_mean_b = m::mean(&sb);
}

/// Error a un paso y rollout corto del Dφ dado (o de uno entrenado al vuelo).
pub fn smoke_dphi(d: &Dphi, seed: u64, label: &str, rep: &mut SmokeReport) -> V4Metrics {
    let s = sealed(seed);
    let mt = eval_model(d, &s, &AuditFlags::default());
    rep.push(
        &format!("error a un paso ({label})"),
        seed,
        format!("rel_err₁={:.4} acc={:.3}", mt.rel_err_1step, mt.acc),
        "rel_err₁ < 0.05",
        mt.rel_err_1step < 0.05,
    );
    rep.push(
        &format!("rollout E33 h{} ({label})", ROLLOUT_H),
        seed,
        format!("acc={:.3} rel={:.3}", mt.rollout_acc, mt.rollout_rel_err),
        "informativo (E33 marginal)",
        true,
    );
    rep.push(
        &format!("auditoría memoria cero ({label})"),
        seed,
        format!("queries={} leaked={}", mt.test_queries, mt.leaked),
        "queries=0 ∧ leaked=0",
        mt.test_queries == 0 && mt.leaked == 0,
    );
    mt
}

/// Smoke v4 completo de la pestaña Pruebas.
pub fn run_smoke(app_model: Option<&Dphi>, quick_steps: usize) -> SmokeReport {
    let t0 = Instant::now();
    let mut rep = SmokeReport::default();
    smoke_e35(&SMOKE_SEEDS, &mut rep);
    let eval_seed = 0x5A00;
    match app_model {
        Some(d) => {
            smoke_dphi(d, eval_seed, "Dφ de la app", &mut rep);
        }
        None => {
            rep.notes
                .push("la app aún no tiene Dφ entrenado: se entrena uno al vuelo (§16, presupuesto corto)".into());
            let mut d = new_dphi(eval_seed ^ 0xD0);
            let mut wake = Vec::new();
            train_batch(&mut d, &mut wake, eval_seed ^ 0x77, quick_steps);
            smoke_dphi(&d, eval_seed, "Dφ al vuelo", &mut rep);
        }
    }
    rep.elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
    rep
}

// ─────────────────────────── chat (decoder del campo) ───────────────────────────

/// Estado del campo para un mensaje: la familia/parámetros se infieren del
/// texto (palabras clave + números), el objeto es determinista por mensaje.
#[derive(Clone, Debug, Serialize)]
pub struct ChatField {
    pub family: String,
    pub family_id: usize,
    pub family_from_text: bool,
    pub params: [f64; 2],
    pub input: Vec<f64>,
    pub predicted: Vec<f64>,
    pub expected: Vec<f64>,
    pub rel_err: f64,
    pub correct: bool,
    pub rollout_h: usize,
    pub rollout_rel_err: f64,
    pub trained: bool,
}

const KEYWORDS: [(&[&str], usize); 8] = [
    (&["trasl", "mueve", "despla", "translat", "shift"], 0),
    (&["rota", "gira", "rotat", "giro"], 1),
    (&["refle", "espej", "mirror", "reflect"], 2),
    (&["escal", "agrand", "encog", "scale", "zoom"], 3),
    (&["cizall", "shear", "inclin"], 4),
    (&["afín", "afin", "affine"], 5),
    (&["compu", "compos", "compose"], 6),
    (&["radial", "warp", "deform", "curva"], 7),
];

fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn parse_family(msg: &str) -> Option<usize> {
    let l = msg.to_lowercase();
    KEYWORDS
        .iter()
        .find(|(ks, _)| ks.iter().any(|k| l.contains(k)))
        .map(|(_, f)| *f)
}

fn parse_params(msg: &str) -> Vec<f64> {
    msg.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .filter_map(|t| t.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .take(2)
        .map(|v| {
            if v.abs() > 1.0 {
                (v / 180.0).clamp(-1.0, 1.0)
            } else {
                v
            }
        })
        .collect()
}

pub fn chat_field(d: &Dphi, trained: bool, msg: &str) -> ChatField {
    let h = fnv(msg.trim());
    let parsed = parse_family(msg);
    let fam = parsed.unwrap_or((h % N_FAM as u64) as usize);
    let mut rng = Xoshiro256StarStar::seed_from_u64(h);
    let nums = parse_params(msg);
    let rnd = |i: u64| ((h.rotate_left(17 * i as u32) % 2000) as f64 / 1000.0) - 1.0;
    let p = [
        nums.first().copied().unwrap_or_else(|| rnd(1)),
        nums.get(1).copied().unwrap_or_else(|| rnd(2)),
    ];
    let x = make_object(&mut rng, K_POINTS, ObjectClass::Blob);
    let c = context_of(fam, p);
    let pred = d.step(&x, &c);
    let y = apply_rule(fam, p, &x);
    let rh = if ITERABLE_FAMS.contains(&fam) {
        ROLLOUT_H
    } else {
        1
    };
    let zr = d.rollout(&x, &c, rh);
    let mut yr = x.clone();
    for _ in 0..rh {
        yr = apply_rule(fam, p, &yr);
    }
    ChatField {
        family: FAMILIES[fam].into(),
        family_id: fam,
        family_from_text: parsed.is_some(),
        params: p,
        rel_err: m::rel_err(&pred, &y),
        correct: m::correct(&pred, &y),
        input: x,
        predicted: pred,
        expected: y,
        rollout_h: rh,
        rollout_rel_err: m::rel_err(&zr, &yr),
        trained,
    }
}

impl ChatField {
    /// Resumen textual (entra al prompt del LLM decoder y a la UI).
    pub fn summary(&self) -> String {
        let pts = |v: &[f64]| {
            v.chunks(2)
                .take(3)
                .map(|p| format!("({:.2},{:.2})", p[0], p[1]))
                .collect::<Vec<_>>()
                .join(" ")
        };
        format!(
            "Dφ v4{} · familia {}{} · p=({:.2},{:.2}) · entrada {} … → Dφ {} … · regla {} … · error relativo a un paso {:.1}% ({}) · rollout h{} {:.1}%",
            if self.trained { "" } else { " (sin entrenar)" },
            self.family,
            if self.family_from_text { "" } else { " (no detectada en el texto; elegida por hash)" },
            self.params[0],
            self.params[1],
            pts(&self.input),
            pts(&self.predicted),
            pts(&self.expected),
            100.0 * self.rel_err,
            if self.correct { "correcto <5%" } else { "fuera de tolerancia" },
            self.rollout_h,
            100.0 * self.rollout_rel_err
        )
    }
}

/// Checkpoint bit-exacto del Dφ de la app.
pub fn to_checkpoint(d: &Dphi) -> String {
    d.to_ckpt()
}

pub fn from_checkpoint(s: &str) -> Option<Dphi> {
    Dphi::from_ckpt(s).filter(|d| d.is_gated() && d.zdim == 2 * K_POINTS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_field_parses_family_and_is_deterministic() {
        let d = new_dphi(1);
        let a = chat_field(&d, false, "rota el triángulo 0.3");
        let b = chat_field(&d, false, "rota el triángulo 0.3");
        assert_eq!(a.family, "rotate");
        assert!(a.family_from_text);
        assert!((a.params[0] - 0.3).abs() < 1e-12);
        assert_eq!(a.predicted, b.predicted);
        assert!(a.summary().contains("rotate"));
    }

    #[test]
    fn checkpoint_roundtrip_bit_exact() {
        let d = new_dphi(7);
        let r = from_checkpoint(&to_checkpoint(&d)).unwrap();
        assert_eq!(d.weights_hash(), r.weights_hash());
    }

    #[test]
    fn sleep_requires_experience_and_drops_cdt() {
        let mut d = new_dphi(3);
        let mut wake = Vec::new();
        assert!(sleep_consolidate(&mut d, &mut wake, 9, 10).is_none());
        let rep = train_batch(&mut d, &mut wake, 0x5A11, 50);
        assert_eq!(rep.metrics.test_queries, 0);
        assert!(!wake.is_empty());
        let sl = sleep_consolidate(&mut d, &mut wake, 0x5A12, 50).unwrap();
        assert!(wake.is_empty() && sl.cdt_dropped);
        assert_eq!(sl.after.test_queries, 0);
        assert_eq!(sl.after.leaked, 0);
    }

    /// Gate científico v4 (CI): memoria cero, E35 B>A en seeds DEV, y cota
    /// del error a un paso con la receta §16 (presupuesto reducido).
    #[test]
    fn scientific_v4_gate() {
        let mut rep = SmokeReport::default();
        smoke_e35(&SMOKE_SEEDS, &mut rep);
        for r in &rep.rows {
            eprintln!("{} {} {} {}", r.test, r.seed, r.value, r.verdict);
        }
        assert!(
            rep.rows
                .iter()
                .filter(|r| r.test.starts_with("auditoría"))
                .all(|r| r.verdict == "PASS"),
            "fuga/consultas de memoria en TEST"
        );
        assert!(
            rep.e35_mean_b > rep.e35_mean_a,
            "E35: B ({:.3}) no supera a A ({:.3})",
            rep.e35_mean_b,
            rep.e35_mean_a
        );
        let mut d = new_dphi(0xC0DE);
        let mut wake = Vec::new();
        let tr = train_batch(&mut d, &mut wake, 0xA403, 12_000);
        eprintln!(
            "§16 corto: rel_err1={:.4} acc={:.3} rollout_h8={:.3}",
            tr.metrics.rel_err_1step, tr.metrics.acc, tr.metrics.rollout_acc
        );
        assert_eq!(tr.metrics.test_queries, 0);
        assert_eq!(tr.metrics.leaked, 0);
        assert!(
            tr.metrics.rel_err_1step < 0.05,
            "error a un paso {:.4} ≥ 0.05",
            tr.metrics.rel_err_1step
        );
    }
}
