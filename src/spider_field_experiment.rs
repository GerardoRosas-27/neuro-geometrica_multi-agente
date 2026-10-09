//! Experimento de sustrato para el decoder del campo del Prompt Spider:
//! **(A) sustrato en blanco** vs **(B) capa externa sobre el núcleo de chat
//! congelado** vs **(C) ajuste fino del núcleo compartido**, más ablaciones
//! de (A). Mismo hold-out, varias semillas. Mide exactitud y calibración
//! (ECE/Brier) por pregunta y por ruta, % resuelto vs pendiente, precisión de
//! lo resuelto, latencia e **interferencia** sobre el campo del chat.
//!
//! Resultados y decisión: `docs/prompt_spider_decoder_campo.md`.

#![allow(clippy::needless_range_loop)]

use crate::liquid_cdt_rqm_fuse::FusedLiquidCdt;
use crate::prompt_spider::{Question, Route, DEFAULT_LLM_FLOOR, DEFAULT_THRESHOLD, SAMPLE_PROMPT};
use crate::spider_field::{
    bin_metrics, core_response_table, router_metrics, scorer_examples, synth_examples,
    synth_prompt, text_group, AnalyzedPrompt, BinMetrics, Coupling, FieldConfig, LabelSource,
    RouterMetrics, Sm, SpiderExample, SpiderFieldRouter, HEADS, HEAD_NAMES, H_OK, VOCAB_SHIFT,
    VOCAB_TRAIN,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Conceptos del campo de chat (igual que `web::llm_periphery::NUM_CONCEPTS`).
pub const CHAT_CONCEPTS: usize = 8;

/// Semilla del prompt `k` del hold-out en distribución (H1).
pub fn holdout_seed(k: usize) -> u64 {
    1_000_000 + k as u64
}

/// Semilla del prompt `k` del hold-out con vocabulario desplazado (H2).
pub fn shift_seed(k: usize) -> u64 {
    2_000_000 + k as u64
}

/// Brazos del experimento.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Arm {
    /// (A) blanco: líquido + CDT + RQM propios, solo datos del Spider.
    A,
    /// (B1) núcleo de chat congelado + lectura entrenable (adaptador lineal).
    B1,
    /// (B2) núcleo de chat congelado + capa externa líquido/CDT/RQM.
    B2,
    /// (C) ajuste fino del núcleo compartido (escribe en el campo del chat).
    C,
    /// Ablación de (A) sin CDT.
    ANoCdt,
    /// Ablación de (A) sin RQM.
    ANoRqm,
    /// Sin campo: lectura logística sobre rasgos crudos.
    Raw,
}

impl Arm {
    pub fn label(self) -> &'static str {
        match self {
            Arm::A => "A · blanco (líquido+CDT+RQM)",
            Arm::B1 => "B1 · núcleo congelado + lectura",
            Arm::B2 => "B2 · núcleo congelado + capa líquido/CDT/RQM",
            Arm::C => "C · ajuste fino del núcleo compartido",
            Arm::ANoCdt => "A sin CDT",
            Arm::ANoRqm => "A sin RQM",
            Arm::Raw => "sin campo (logística cruda)",
        }
    }

    pub fn config(self, seed: u64) -> FieldConfig {
        let mut c = FieldConfig::blank(seed);
        match self {
            Arm::A => {}
            Arm::B1 => {
                c.liquid = false;
                c.cdt = false;
                c.rqm = false;
                c.coupling = Coupling::FrozenCore;
            }
            Arm::B2 => c.coupling = Coupling::FrozenCore,
            Arm::C => {
                c.cdt = false;
                c.rqm = false;
                c.coupling = Coupling::SharedCore;
            }
            Arm::ANoCdt => c.cdt = false,
            Arm::ANoRqm => c.rqm = false,
            Arm::Raw => {
                c.liquid = false;
                c.cdt = false;
                c.rqm = false;
            }
        }
        c
    }
}

/// Configuración del experimento.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub seeds: Vec<u64>,
    pub arms: Vec<Arm>,
    /// Prompts sintéticos de entrenamiento por semilla.
    pub train_prompts: usize,
    /// Datasets (vigilia + sueño) en que se reparten.
    pub datasets: usize,
    pub holdout_prompts: usize,
    /// Ruido de etiquetas en entrenamiento.
    pub noise: f64,
    pub threshold: f64,
    pub floor: f64,
    /// Neuronas del reservorio líquido.
    pub n_liquid: usize,
    pub epochs: usize,
}

impl ExperimentConfig {
    /// Configuración completa (la de los resultados documentados).
    pub fn full() -> Self {
        Self {
            seeds: vec![11, 22, 33, 44, 55],
            arms: vec![
                Arm::A,
                Arm::B1,
                Arm::B2,
                Arm::C,
                Arm::ANoCdt,
                Arm::ANoRqm,
                Arm::Raw,
            ],
            train_prompts: 240,
            datasets: 6,
            holdout_prompts: 80,
            noise: 0.05,
            threshold: DEFAULT_THRESHOLD,
            floor: DEFAULT_LLM_FLOOR,
            n_liquid: 48,
            epochs: 120,
        }
    }

    /// Versión pequeña (smoke / gate).
    pub fn small(seeds: Vec<u64>, arms: Vec<Arm>) -> Self {
        Self {
            seeds,
            arms,
            train_prompts: 60,
            datasets: 2,
            holdout_prompts: 24,
            n_liquid: 32,
            epochs: 60,
            ..Self::full()
        }
    }
}

/// Etiqueta de maestro (Gemma) para una palabra.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherItem {
    /// `sample` (prompt de ejemplo) | `holdout` (H1).
    pub set: String,
    pub seed: u64,
    pub index: usize,
    pub word: String,
    pub p_yes_first: f64,
    pub p_no_first: f64,
    pub p: f64,
    pub seconds: f64,
}

/// Archivo de etiquetas de maestro.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TeacherFile {
    pub model: String,
    pub floor: f64,
    pub items: Vec<TeacherItem>,
}

/// Métricas de un brazo con una semilla.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ArmRun {
    pub arm: Option<Arm>,
    pub seed: u64,
    /// Por cabeza en todas las palabras de H1.
    pub heads_h1: Vec<BinMetrics>,
    /// Router en escaladas de H1 / H2.
    pub router_h1: RouterMetrics,
    pub router_h2: RouterMetrics,
    /// Exactitud de `ok` por ruta inicial (code, llm, you) en H1.
    pub ok_by_route_h1: Vec<BinMetrics>,
    /// Escaladas del prompt de ejemplo resueltas por el campo (%).
    pub sample_resolved_pct: f64,
    /// Acuerdo resolver/pendiente con el maestro (prompt de ejemplo + H1).
    pub teacher_agreement: Option<f64>,
    pub teacher_n: usize,
    /// Exactitud vs verdad en las palabras de H1 que también juzgó el maestro.
    pub teacher_subset_field_acc: Option<f64>,
    pub latency_us: f64,
    pub train_s: f64,
    /// Exactitud del campo de chat antes/después y |Δscore| máx.
    pub chat_acc_before: f64,
    pub chat_acc_after: f64,
    pub chat_max_dscore: f64,
}

/// Campo de chat «existente» entrenado con el currículo relacional de la
/// app (`cue → (cue+1) mod 8`, 3 rondas de teach + observe + sueño).
pub fn trained_chat_field() -> FusedLiquidCdt {
    let mut f = FusedLiquidCdt::new(CHAT_CONCEPTS);
    let cands: Vec<usize> = (0..CHAT_CONCEPTS).collect();
    for _ in 0..3 {
        for cue in 0..CHAT_CONCEPTS {
            f.observe(cue, &cands);
            f.teach_relation(cue, (cue + 1) % CHAT_CONCEPTS);
        }
        f.sleep_consolidate();
    }
    f
}

/// Sonda del chat sobre una **copia** (no perturba el campo): exactitud del
/// mapa relacional y puntuaciones por cue.
pub fn chat_probe(f: &FusedLiquidCdt) -> (f64, Vec<f64>) {
    let mut c = f.clone();
    let cands: Vec<usize> = (0..CHAT_CONCEPTS).collect();
    let mut ok = 0;
    let mut scores = Vec::new();
    for cue in 0..CHAT_CONCEPTS {
        let r = c.infer(cue, &cands);
        if r.predicted == (cue + 1) % CHAT_CONCEPTS {
            ok += 1;
        }
        scores.push(r.top1_score);
    }
    (ok as f64 / CHAT_CONCEPTS as f64, scores)
}

fn q_idx(q: Question) -> usize {
    match q {
        Question::Relevant => 0,
        Question::Grounded => 1,
        Question::Ambiguous => 2,
        Question::Approval => 3,
    }
}

/// Hold-out codificado (etiquetas limpias).
pub struct Holdout {
    pub h1: Vec<SpiderExample>,
    pub h2: Vec<SpiderExample>,
    pub sample: Vec<SpiderExample>,
}

pub fn build_holdout(cfg: &ExperimentConfig) -> Holdout {
    let mut rng = Sm(0);
    let h1 = (0..cfg.holdout_prompts)
        .flat_map(|k| {
            synth_examples(
                &synth_prompt(holdout_seed(k), &VOCAB_TRAIN),
                cfg.threshold,
                0.0,
                &mut rng,
            )
        })
        .collect();
    let h2 = (0..cfg.holdout_prompts)
        .flat_map(|k| {
            synth_examples(
                &synth_prompt(shift_seed(k), &VOCAB_SHIFT),
                cfg.threshold,
                0.0,
                &mut rng,
            )
        })
        .collect();
    let ap = AnalyzedPrompt::new(SAMPLE_PROMPT, cfg.threshold);
    let sample = (0..ap.tokens.len())
        .filter_map(|i| {
            ap.encode(
                i,
                [None; HEADS],
                LabelSource::Teacher,
                text_group(SAMPLE_PROMPT),
            )
        })
        .filter(|e| e.first_route == Route::Llm)
        .collect();
    Holdout { h1, h2, sample }
}

/// Datos de entrenamiento de una semilla: prompts sintéticos (con ruido) +
/// etiquetas del scorer en el prompt de ejemplo (solo code/you).
pub fn training_sets(cfg: &ExperimentConfig, seed: u64) -> Vec<Vec<SpiderExample>> {
    let mut rng = Sm(seed ^ 0x7EA1);
    let mut sets: Vec<Vec<SpiderExample>> = vec![Vec::new(); cfg.datasets.max(1)];
    for k in 0..cfg.train_prompts {
        let sp = synth_prompt(seed * 100_000 + k as u64, &VOCAB_TRAIN);
        let d = k % sets.len();
        sets[d].extend(synth_examples(&sp, cfg.threshold, cfg.noise, &mut rng));
    }
    sets[0].extend(scorer_examples(
        SAMPLE_PROMPT,
        cfg.threshold,
        text_group(SAMPLE_PROMPT),
    ));
    sets
}

/// Entrena y evalúa un brazo con una semilla.
pub fn run_arm(
    cfg: &ExperimentConfig,
    arm: Arm,
    seed: u64,
    hold: &Holdout,
    teacher: Option<&TeacherFile>,
) -> ArmRun {
    let mut fcfg = arm.config(seed);
    fcfg.n_liquid = cfg.n_liquid;
    fcfg.epochs = cfg.epochs;
    let mut router = SpiderFieldRouter::new(fcfg);
    let mut chat = trained_chat_field();
    let (chat_acc_before, scores_before) = chat_probe(&chat);
    let coupled = fcfg.coupling != Coupling::Blank;
    if coupled {
        // Congelado: la tabla se lee de una copia; el campo vivo no se toca.
        router.set_core_table(core_response_table(&mut chat.clone()));
    }
    let t0 = Instant::now();
    for set in training_sets(cfg, seed) {
        for ex in set {
            if arm == Arm::C {
                // Ajuste fino: la experiencia del Spider se escribe en el
                // campo compartido (cue léxico → pregunta×ok).
                if let Some(ok) = ex.labels[H_OK] {
                    let label = 2 * q_idx(ex.question) + ok as usize;
                    chat.teach_relation(ex.core_cue, label % CHAT_CONCEPTS);
                }
            }
            router.observe(ex);
        }
        if arm == Arm::C {
            chat.sleep_consolidate();
            router.set_core_table(core_response_table(&mut chat.clone()));
        }
        router.sleep();
    }
    let train_s = t0.elapsed().as_secs_f64();
    let (chat_acc_after, scores_after) = chat_probe(&chat);
    let chat_max_dscore = scores_before
        .iter()
        .zip(scores_after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    // Cabezas en todas las palabras de H1 + ok por ruta.
    let mut per_head: Vec<Vec<(f64, bool)>> = vec![Vec::new(); HEADS];
    let mut by_route: Vec<Vec<(f64, bool)>> = vec![Vec::new(); 3];
    let mut rows_h1 = Vec::new();
    let mut lat = Vec::new();
    for ex in &hold.h1 {
        let p = router.predict(ex);
        for h in 0..HEADS {
            if let Some(y) = ex.labels[h] {
                per_head[h].push((p.probs[h], y));
            }
        }
        let r = match ex.first_route {
            Route::Code => 0,
            Route::Llm => 1,
            Route::You => 2,
        };
        by_route[r].push((p.p_ok(), ex.labels[H_OK].unwrap_or(false)));
        if ex.first_route == Route::Llm {
            let d = router.decide(ex, cfg.floor);
            lat.push(d.micros);
            rows_h1.push((
                d.prediction.p_ok(),
                d.resolve,
                ex.labels[H_OK].unwrap_or(false),
            ));
        }
    }
    let rows_h2: Vec<(f64, bool, bool)> = hold
        .h2
        .iter()
        .filter(|e| e.first_route == Route::Llm)
        .map(|ex| {
            let d = router.decide(ex, cfg.floor);
            (
                d.prediction.p_ok(),
                d.resolve,
                ex.labels[H_OK].unwrap_or(false),
            )
        })
        .collect();
    let sample_dec: Vec<(usize, bool)> = hold
        .sample
        .iter()
        .map(|ex| (ex.index, router.decide(ex, cfg.floor).resolve))
        .collect();
    let sample_resolved_pct = if sample_dec.is_empty() {
        0.0
    } else {
        100.0 * sample_dec.iter().filter(|d| d.1).count() as f64 / sample_dec.len() as f64
    };
    // Acuerdo con el maestro.
    let (mut agree, mut tn, mut sub_ok, mut sub_n) = (0usize, 0usize, 0usize, 0usize);
    if let Some(t) = teacher {
        for it in &t.items {
            let t_res = it.p >= t.floor;
            let field = if it.set == "sample" {
                sample_dec
                    .iter()
                    .find(|d| d.0 == it.index)
                    .map(|d| (d.1, None))
            } else {
                let k = it.seed.wrapping_sub(holdout_seed(0)) as usize;
                if k >= cfg.holdout_prompts {
                    None
                } else {
                    hold.h1
                        .iter()
                        .find(|e| e.group == it.seed && e.index == it.index)
                        .map(|e| (router.decide(e, cfg.floor).resolve, e.labels[H_OK]))
                }
            };
            if let Some((f_res, truth)) = field {
                tn += 1;
                agree += (f_res == t_res) as usize;
                if let Some(y) = truth {
                    sub_n += 1;
                    let p = router
                        .predict(
                            hold.h1
                                .iter()
                                .find(|e| e.group == it.seed && e.index == it.index)
                                .unwrap(),
                        )
                        .p_ok();
                    sub_ok += ((p >= 0.5) == y) as usize;
                }
            }
        }
    }
    lat.sort_by(|a, b| a.total_cmp(b));
    ArmRun {
        arm: Some(arm),
        seed,
        heads_h1: per_head.iter().map(|v| bin_metrics(v)).collect(),
        router_h1: router_metrics(&rows_h1),
        router_h2: router_metrics(&rows_h2),
        ok_by_route_h1: by_route.iter().map(|v| bin_metrics(v)).collect(),
        sample_resolved_pct,
        teacher_agreement: (tn > 0).then(|| agree as f64 / tn as f64),
        teacher_n: tn,
        teacher_subset_field_acc: (sub_n > 0).then(|| sub_ok as f64 / sub_n as f64),
        latency_us: if lat.is_empty() {
            0.0
        } else {
            lat[lat.len() / 2]
        },
        train_s,
        chat_acc_before,
        chat_acc_after,
        chat_max_dscore,
    }
}

/// Baseline heurístico en el mismo hold-out: puntuaciones del scorer como
/// probabilidades; resuelve si `P(ok) heurística ≥ piso`.
pub fn heuristic_baseline(cfg: &ExperimentConfig, hold: &Holdout) -> ArmRun {
    let mut per_head: Vec<Vec<(f64, bool)>> = vec![Vec::new(); HEADS];
    for ex in &hold.h1 {
        for h in 0..4 {
            if let Some(y) = ex.labels[h] {
                per_head[h].push((ex.heur_scores[h], y));
            }
        }
        if let Some(y) = ex.labels[H_OK] {
            let p = if ex.first_route == Route::Code {
                ex.global[4] as f64
            } else {
                ex.heur_ok
            };
            per_head[H_OK].push((p, y));
        }
    }
    let rows = |v: &[SpiderExample]| -> Vec<(f64, bool, bool)> {
        v.iter()
            .filter(|e| e.first_route == Route::Llm)
            .map(|e| {
                (
                    e.heur_ok,
                    e.heur_ok >= cfg.floor,
                    e.labels[H_OK].unwrap_or(false),
                )
            })
            .collect()
    };
    ArmRun {
        arm: None,
        heads_h1: per_head.iter().map(|v| bin_metrics(v)).collect(),
        router_h1: router_metrics(&rows(&hold.h1)),
        router_h2: router_metrics(&rows(&hold.h2)),
        sample_resolved_pct: {
            let n = hold.sample.len().max(1);
            100.0
                * hold
                    .sample
                    .iter()
                    .filter(|e| e.heur_ok >= cfg.floor)
                    .count() as f64
                / n as f64
        },
        ..Default::default()
    }
}

/// Media y desviación típica.
pub fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    (m, v.sqrt())
}

/// Resultados completos.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentResults {
    pub config: ExperimentConfig,
    pub holdout_sizes: [usize; 5],
    pub baseline: ArmRun,
    pub runs: Vec<ArmRun>,
    pub teacher_model: Option<String>,
    pub teacher_alone: Option<serde_json::Value>,
}

pub fn run_experiment(cfg: &ExperimentConfig, teacher: Option<&TeacherFile>) -> ExperimentResults {
    let hold = build_holdout(cfg);
    let esc = |v: &[SpiderExample]| v.iter().filter(|e| e.first_route == Route::Llm).count();
    let baseline = heuristic_baseline(cfg, &hold);
    let mut runs = Vec::new();
    for &arm in &cfg.arms {
        for &seed in &cfg.seeds {
            runs.push(run_arm(cfg, arm, seed, &hold, teacher));
        }
    }
    // Maestro solo: exactitud vs verdad en las palabras H1 que juzgó.
    let teacher_alone = teacher.map(|t| {
        let mut rows = Vec::new();
        let mut sample_res = (0usize, 0usize);
        for it in &t.items {
            if it.set == "sample" {
                sample_res.1 += 1;
                sample_res.0 += (it.p >= t.floor) as usize;
                continue;
            }
            if let Some(e) = hold
                .h1
                .iter()
                .find(|e| e.group == it.seed && e.index == it.index)
            {
                rows.push((it.p, it.p >= t.floor, e.labels[H_OK].unwrap_or(false)));
            }
        }
        let m = router_metrics(&rows);
        let mut secs: Vec<f64> = t.items.iter().map(|i| i.seconds).collect();
        secs.sort_by(|a, b| a.total_cmp(b));
        let median = secs.get(secs.len() / 2).copied().unwrap_or(0.0);
        let sample_pct = if sample_res.1 > 0 {
            100.0 * sample_res.0 as f64 / sample_res.1 as f64
        } else {
            0.0
        };
        serde_json::json!({
            "holdout_words": rows.len(),
            "router": m,
            "sample_words": sample_res.1,
            "sample_resolved_pct": sample_pct,
            "median_seconds_per_word": median,
        })
    });
    ExperimentResults {
        config: cfg.clone(),
        holdout_sizes: [
            hold.h1.len(),
            esc(&hold.h1),
            hold.h2.len(),
            esc(&hold.h2),
            hold.sample.len(),
        ],
        baseline,
        runs,
        teacher_model: teacher.map(|t| t.model.clone()),
        teacher_alone,
    }
}

fn agg(runs: &[&ArmRun], f: impl Fn(&ArmRun) -> f64) -> String {
    let xs: Vec<f64> = runs.iter().map(|r| f(r)).collect();
    let (m, s) = mean_std(&xs);
    format!("{m:.3} ± {s:.3}")
}

fn agg_pct(runs: &[&ArmRun], f: impl Fn(&ArmRun) -> f64) -> String {
    let xs: Vec<f64> = runs.iter().map(|r| f(r)).collect();
    let (m, s) = mean_std(&xs);
    format!("{m:.1} ± {s:.1}")
}

/// Tablas markdown (media ± desviación sobre semillas).
pub fn format_markdown(r: &ExperimentResults) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "Semillas: {:?} · prompts de entrenamiento/semilla: {} en {} datasets · ruido de etiquetas {:.0} % · hold-out H1 {} palabras ({} escaladas), H2 vocabulario desplazado {} ({} escaladas), prompt de ejemplo {} escaladas · piso {:.2} · umbral {:.2}\n\n",
        r.config.seeds, r.config.train_prompts, r.config.datasets, r.config.noise * 100.0,
        r.holdout_sizes[0], r.holdout_sizes[1], r.holdout_sizes[2], r.holdout_sizes[3], r.holdout_sizes[4],
        r.config.floor, r.config.threshold
    ));
    s.push_str("### Router en escaladas (H1, en distribución)\n\n| brazo | exactitud ok | ECE ok | Brier ok | resuelto % | precisión resuelto | resuelto erróneo % | pendiente % | latencia µs (mediana) |\n|---|---|---|---|---|---|---|---|---|\n");
    let b = &r.baseline;
    s.push_str(&format!(
        "| heurística (scorer, piso) | {:.3} | {:.3} | {:.3} | {:.1} | {:.3} | {:.1} | {:.1} | — |\n",
        b.router_h1.ok.accuracy, b.router_h1.ok.ece, b.router_h1.ok.brier, b.router_h1.resolved_pct,
        b.router_h1.resolved_precision, b.router_h1.wrong_resolved_pct, b.router_h1.pending_pct
    ));
    for &arm in &r.config.arms {
        let rs: Vec<&ArmRun> = r.runs.iter().filter(|x| x.arm == Some(arm)).collect();
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            arm.label(),
            agg(&rs, |x| x.router_h1.ok.accuracy),
            agg(&rs, |x| x.router_h1.ok.ece),
            agg(&rs, |x| x.router_h1.ok.brier),
            agg_pct(&rs, |x| x.router_h1.resolved_pct),
            agg(&rs, |x| x.router_h1.resolved_precision),
            agg_pct(&rs, |x| x.router_h1.wrong_resolved_pct),
            agg_pct(&rs, |x| x.router_h1.pending_pct),
            agg_pct(&rs, |x| x.latency_us),
        ));
    }
    s.push_str("\n### Router en escaladas (H2, vocabulario desplazado)\n\n| brazo | exactitud ok | ECE ok | resuelto % | precisión resuelto | resuelto erróneo % |\n|---|---|---|---|---|---|\n");
    s.push_str(&format!(
        "| heurística | {:.3} | {:.3} | {:.1} | {:.3} | {:.1} |\n",
        b.router_h2.ok.accuracy,
        b.router_h2.ok.ece,
        b.router_h2.resolved_pct,
        b.router_h2.resolved_precision,
        b.router_h2.wrong_resolved_pct
    ));
    for &arm in &r.config.arms {
        let rs: Vec<&ArmRun> = r.runs.iter().filter(|x| x.arm == Some(arm)).collect();
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            arm.label(),
            agg(&rs, |x| x.router_h2.ok.accuracy),
            agg(&rs, |x| x.router_h2.ok.ece),
            agg_pct(&rs, |x| x.router_h2.resolved_pct),
            agg(&rs, |x| x.router_h2.resolved_precision),
            agg_pct(&rs, |x| x.router_h2.wrong_resolved_pct),
        ));
    }
    s.push_str(
        "\n### Por pregunta (todas las palabras de H1): exactitud / ECE / Brier\n\n| brazo |",
    );
    for h in HEAD_NAMES {
        s.push_str(&format!(" {h} |"));
    }
    s.push_str("\n|---|---|---|---|---|---|\n| heurística |");
    for m in &b.heads_h1 {
        s.push_str(&format!(
            " {:.3} / {:.3} / {:.3} |",
            m.accuracy, m.ece, m.brier
        ));
    }
    s.push('\n');
    for &arm in &r.config.arms {
        let rs: Vec<&ArmRun> = r.runs.iter().filter(|x| x.arm == Some(arm)).collect();
        s.push_str(&format!("| {} |", arm.label()));
        for h in 0..HEADS {
            let (a, _) = mean_std(
                &rs.iter()
                    .map(|x| x.heads_h1[h].accuracy)
                    .collect::<Vec<_>>(),
            );
            let (e, _) = mean_std(&rs.iter().map(|x| x.heads_h1[h].ece).collect::<Vec<_>>());
            let (br, _) = mean_std(&rs.iter().map(|x| x.heads_h1[h].brier).collect::<Vec<_>>());
            s.push_str(&format!(" {a:.3} / {e:.3} / {br:.3} |"));
        }
        s.push('\n');
    }
    s.push_str("\n### Exactitud de `ok` por ruta inicial (H1)\n\n| brazo | code | llm (escaladas) | tú |\n|---|---|---|---|\n");
    for &arm in &r.config.arms {
        let rs: Vec<&ArmRun> = r.runs.iter().filter(|x| x.arm == Some(arm)).collect();
        s.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            arm.label(),
            agg(&rs, |x| x.ok_by_route_h1[0].accuracy),
            agg(&rs, |x| x.ok_by_route_h1[1].accuracy),
            agg(&rs, |x| x.ok_by_route_h1[2].accuracy),
        ));
    }
    s.push_str("\n### Interferencia sobre el campo del chat, prompt real y maestro\n\n| brazo | chat exactitud antes → después | máx. |Δ score| chat | escaladas del ejemplo resueltas % | acuerdo con maestro | exactitud campo en subset del maestro | entrenamiento s |\n|---|---|---|---|---|---|---|\n");
    for &arm in &r.config.arms {
        let rs: Vec<&ArmRun> = r.runs.iter().filter(|x| x.arm == Some(arm)).collect();
        let ta: Vec<f64> = rs.iter().filter_map(|x| x.teacher_agreement).collect();
        let tsub: Vec<f64> = rs
            .iter()
            .filter_map(|x| x.teacher_subset_field_acc)
            .collect();
        s.push_str(&format!(
            "| {} | {} → {} | {} | {} | {} | {} | {} |\n",
            arm.label(),
            agg(&rs, |x| x.chat_acc_before),
            agg(&rs, |x| x.chat_acc_after),
            agg(&rs, |x| x.chat_max_dscore),
            agg_pct(&rs, |x| x.sample_resolved_pct),
            if ta.is_empty() {
                "—".into()
            } else {
                let (m, sd) = mean_std(&ta);
                format!("{m:.3} ± {sd:.3} (n={})", rs[0].teacher_n)
            },
            if tsub.is_empty() {
                "—".into()
            } else {
                let (m, sd) = mean_std(&tsub);
                format!("{m:.3} ± {sd:.3}")
            },
            agg_pct(&rs, |x| x.train_s),
        ));
    }
    if let Some(t) = &r.teacher_alone {
        s.push_str(&format!(
            "\nMaestro ({}) solo: {}\n",
            r.teacher_model.clone().unwrap_or_default(),
            t
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke (CI): A aprende por encima de la heurística en escaladas y no
    /// toca el campo del chat; B congelado tampoco.
    #[test]
    fn spider_field_smoke_experiment() {
        let cfg = ExperimentConfig::small(vec![1], vec![Arm::A, Arm::B1]);
        let r = run_experiment(&cfg, None);
        let a = r.runs.iter().find(|x| x.arm == Some(Arm::A)).unwrap();
        let b1 = r.runs.iter().find(|x| x.arm == Some(Arm::B1)).unwrap();
        assert!(
            a.router_h1.ok.accuracy > r.baseline.router_h1.ok.accuracy + 0.1,
            "A {:?} vs heurística {:?}",
            a.router_h1.ok,
            r.baseline.router_h1.ok
        );
        assert_eq!(a.chat_acc_before, a.chat_acc_after);
        assert_eq!(a.chat_max_dscore, 0.0);
        assert_eq!(b1.chat_max_dscore, 0.0);
    }

    /// Experimento completo (≈ minutos). `SPIDER_TEACHER=<json>` añade el
    /// maestro; `SPIDER_FIELD_OUT=<prefijo>` escribe `.json` y `.md`.
    #[test]
    #[ignore]
    fn spider_field_substrate_experiment() {
        let mut cfg = ExperimentConfig::full();
        if let Ok(s) = std::env::var("SPIDER_FIELD_SEEDS") {
            cfg.seeds = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        }
        let teacher: Option<TeacherFile> = std::env::var("SPIDER_TEACHER")
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|t| serde_json::from_str(&t).ok());
        let r = run_experiment(&cfg, teacher.as_ref());
        let md = format_markdown(&r);
        println!("{md}");
        if let Ok(prefix) = std::env::var("SPIDER_FIELD_OUT") {
            std::fs::write(
                format!("{prefix}.json"),
                serde_json::to_string_pretty(&r).unwrap(),
            )
            .unwrap();
            std::fs::write(format!("{prefix}.md"), md).unwrap();
        }
    }
}
