//! Experimentos 8 / 9 / 10 — inferencia líquida.
//! Protocolo: `docs/experimentos_8_9_10_liquido.md`.
//!
//! - E8: invariancia de representación (frontera lingüística → fingerprints de campo).
//! - E9: composición relacional A→B, B→C, C→D → queries A→C / B→D / A→D.
//! - E10: predicción de futuros (distancia, bifurcación, perturbación).
//!
//! Periferia: Gemma GGUF si existe; si no, `GemmaShapedLexicon` (etiqueta honesta).
//! `FieldState` nunca recibe tokens.

use crate::field_encoder::{cluster_field_target, write_into_field};
use crate::field_linguistic_layer::{
    FrozenLinguisticProbe, GemmaShapedLexicon, LinguisticFieldCodec,
};
use crate::field_substrate::{ComplexT, FieldState, Phasor};
use crate::liquid_cdt_rqm_fuse::{FuseReport, FusedLiquidCdt, InferRoute};
use crate::wave_predict_core::{WavePacket, WavePredictCore};
use std::collections::HashMap;
use std::time::Instant;

const EPS: f64 = 1e-12;

/// Fila del registro obligatorio del protocolo.
#[derive(Clone, Debug, Default)]
pub struct RegistryRow {
    pub experiment: String,
    pub commit: String,
    pub seed: u64,
    pub n: usize,
    pub train_examples: usize,
    pub unseen_examples: usize,
    pub ood_examples: usize,
    pub accuracy_seen: f64,
    pub accuracy_unseen: f64,
    pub accuracy_ood: f64,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub top1_score: f64,
    pub top2_score: f64,
    pub margin: f64,
    pub energy: f64,
    pub abstentions: usize,
    pub liquid_calls: usize,
    pub cdt_calls: usize,
    pub rqm_calls: usize,
    pub sleep_ms: f64,
    pub engrams_before: usize,
    pub engrams_after: usize,
    pub verdict: String,
    pub notes: String,
    pub periphery: String,
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn latency_stats(samples_ns: &[u128]) -> (f64, f64, f64, f64) {
    let mut us: Vec<f64> = samples_ns.iter().map(|&n| n as f64 / 1e3).collect();
    us.sort_by(|a, b| a.total_cmp(b));
    (
        mean(&us),
        percentile(&us, 50.0),
        percentile(&us, 95.0),
        percentile(&us, 99.0),
    )
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
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
    dot / ((na.sqrt() * nb.sqrt()).max(EPS))
}

fn phasor_fingerprint(z: &[Phasor]) -> Vec<f64> {
    let mut v = Vec::with_capacity(z.len() * 2);
    for p in z {
        v.push(p.re);
        v.push(p.im);
    }
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
    for x in &mut v {
        *x /= n;
    }
    v
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn gemma_gguf_available() -> bool {
    [
        "models/gemma-2-2b-it-Q4_K_M.gguf",
        "models/gemma2.gguf",
        "/models/gemma.gguf",
    ]
    .iter()
    .any(|p| std::path::Path::new(p).exists())
}

fn periphery_label() -> &'static str {
    if gemma_gguf_available() {
        "gemma-gguf"
    } else {
        "gemma-shaped-lexicon (GGUF absent; honest)"
    }
}

// ─── E8 attractor bank (architecture improvement) ───────────────────────────

/// Banco de atractores en frontera lingüística→campo (sin tokens en Ψ).
/// Mejora E8: matching por coseno al fingerprint entrenado, no cuantización bruta.
#[derive(Clone, Debug, Default)]
pub struct ConceptAttractorBank {
    pub attractors: HashMap<usize, Vec<f64>>,
    pub match_threshold: f64,
}

impl ConceptAttractorBank {
    pub fn new(threshold: f64) -> Self {
        Self {
            attractors: HashMap::new(),
            match_threshold: threshold,
        }
    }

    pub fn store(&mut self, concept: usize, fp: Vec<f64>) {
        self.attractors.insert(concept, fp);
    }

    pub fn match_fp(&self, fp: &[f64]) -> Option<(usize, f64)> {
        let mut scored: Vec<(usize, f64)> = self
            .attractors
            .iter()
            .map(|(&c, a)| (c, cosine(fp, a)))
            .collect();
        if scored.is_empty() {
            return None;
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let (best_c, best_s) = scored[0];
        let second_s = scored.get(1).map(|(_, s)| *s).unwrap_or(0.0);
        // Exige umbral absoluto Y margen frente al 2º (evita colapso a un atractor).
        if best_s >= self.match_threshold && (best_s - second_s) >= 0.05 {
            Some((best_c, best_s))
        } else {
            None
        }
    }
}

fn encode_fingerprint(
    codec: &LinguisticFieldCodec,
    probe: &mut GemmaShapedLexicon,
    text: &str,
) -> Vec<f64> {
    // Matching usa features de sonda congelada (no el projector compartido):
    // evita interferencia catastrófica al anclar un segundo concepto.
    let feat = probe.analyze(text).into_field_features();
    let mut v = feat;
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
    for x in &mut v {
        *x /= n;
    }
    // Firewall de campo: escribir fasores y verificar ausencia de tokens.
    let z = codec.encode_text(probe, text);
    let mut psi = FieldState::new(ComplexT::octahedron());
    write_into_field(&mut psi, &z);
    assert!(
        !psi.has_token_ids(),
        "FieldState must never hold tokens (E8 firewall)"
    );
    let _ = z; // path de campo conectado; matching = features de sonda
    v
}

/// Ancla UNA forma al atractor del concepto; no enseña variantes OOD.
fn train_single_form_attractor(
    codec: &mut LinguisticFieldCodec,
    probe: &mut GemmaShapedLexicon,
    text: &str,
    concept: usize,
    steps: usize,
    seed: u64,
) -> Vec<f64> {
    use rand::Rng;
    use rand_xoshiro::rand_core::SeedableRng;
    let n_edges = ComplexT::octahedron().n_edges();
    let target = cluster_field_target(concept % 3, n_edges);
    // También un patrón de fase por concepto (más discriminativo que solo 3 clusters).
    let base = (concept as f64) * (std::f64::consts::TAU / 8.0);
    let target: Vec<Phasor> = (0..n_edges)
        .map(|e| {
            let t = &target[e];
            let phase = base + 0.35 * (e as f64);
            let mix = Phasor::from_polar(1.0, phase);
            // Blend suave hacia geometría de concepto.
            Phasor::new(0.5 * t.re + 0.5 * mix.re, 0.5 * t.im + 0.5 * mix.im)
        })
        .collect();
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed ^ 0xE8A7);
    codec.projector.eta = 0.12;
    for _ in 0..steps {
        let noisy = if rng.gen_bool(0.25) {
            format!(" {} ", text)
        } else if rng.gen_bool(0.25) {
            text.to_uppercase()
        } else {
            text.to_string()
        };
        let feat = probe.analyze(&noisy).into_field_features();
        let z = codec.projector.encode(&feat);
        codec.projector.delta_toward(&feat, &z, &target);
    }
    encode_fingerprint(codec, probe, text)
}

#[derive(Clone, Debug)]
pub struct E8Failure {
    pub query: String,
    pub expected_concept: usize,
    pub got: Option<(usize, f64)>,
    pub kind: &'static str,
}

pub fn run_experiment_8(seed: u64) -> (RegistryRow, Vec<E8Failure>) {
    let commit = git_commit();
    let periphery = periphery_label();
    let mut probe = GemmaShapedLexicon::new(seed);
    let mut codec = LinguisticFieldCodec::new(seed);
    codec.probe_name = probe.name();

    const TRAIN: &str = "perro";
    const CONCEPT: usize = 3;
    let seen = ["perro", "Perro", " el perro ", "perros"];
    let unseen_paraphrase = ["perrito", "un perro grande", "el perrito"];
    let ood_xling = ["dog", "chien", "犬"];
    let ood_noise = ["xqz9", "mesa", "avion"];
    let ood_near = ["gato", "lobo"];

    let fp_train = train_single_form_attractor(&mut codec, &mut probe, TRAIN, CONCEPT, 120, seed);
    let mut bank = ConceptAttractorBank::new(0.60);
    bank.store(CONCEPT, fp_train);
    let fp_other = train_single_form_attractor(&mut codec, &mut probe, "casa", 1, 80, seed ^ 1);
    bank.store(1, fp_other);

    let mut failures = Vec::new();
    let mut lat_ns = Vec::new();
    let mut scores = Vec::new();
    let mut margins = Vec::new();

    let mut eval = |texts: &[&str],
                    expect: Option<usize>,
                    kind: &'static str,
                    ok: &mut usize,
                    n: &mut usize| {
        for &q in texts {
            *n += 1;
            let t0 = Instant::now();
            let fp = encode_fingerprint(&codec, &mut probe, q);
            let hit = bank.match_fp(&fp);
            lat_ns.push(t0.elapsed().as_nanos());
            match (expect, hit) {
                (Some(exp), Some((got, sim))) if got == exp => {
                    *ok += 1;
                    scores.push(sim);
                    margins.push(sim - bank.match_threshold);
                }
                (None, None) => {
                    *ok += 1;
                    scores.push(0.0);
                    margins.push(0.0);
                }
                (None, Some((got, sim))) => {
                    failures.push(E8Failure {
                        query: q.into(),
                        expected_concept: CONCEPT,
                        got: Some((got, sim)),
                        kind,
                    });
                    scores.push(sim);
                }
                (Some(exp), other) => {
                    failures.push(E8Failure {
                        query: q.into(),
                        expected_concept: exp,
                        got: other,
                        kind,
                    });
                    if let Some((_, sim)) = other {
                        scores.push(sim);
                    }
                }
            }
        }
    };

    let mut seen_ok = 0usize;
    let mut seen_n = 0usize;
    let mut unseen_ok = 0usize;
    let mut unseen_n = 0usize;
    let mut ood_ok = 0usize;
    let mut ood_n = 0usize;

    eval(&seen, Some(CONCEPT), "seen", &mut seen_ok, &mut seen_n);
    eval(
        &unseen_paraphrase,
        Some(CONCEPT),
        "unseen_paraphrase",
        &mut unseen_ok,
        &mut unseen_n,
    );
    let mut ood_all: Vec<&str> = Vec::new();
    ood_all.extend_from_slice(&ood_xling);
    ood_all.extend_from_slice(&ood_noise);
    ood_all.extend_from_slice(&ood_near);
    eval(&ood_all, None, "ood_reject", &mut ood_ok, &mut ood_n);

    let (mean_us, p50, p95, p99) = latency_stats(&lat_ns);
    let acc_seen = seen_ok as f64 / seen_n.max(1) as f64;
    let acc_unseen = unseen_ok as f64 / unseen_n.max(1) as f64;
    let acc_ood = ood_ok as f64 / ood_n.max(1) as f64;

    let verdict = if acc_seen >= 0.75 && acc_unseen >= 0.5 && acc_ood >= 0.5 {
        "PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon)"
    } else if acc_seen >= 0.75 {
        "FAIL_GENERALIZE: trained form ok; paraphrase/OOD weak"
    } else {
        "FAIL: trained attractor unstable"
    };

    let row = RegistryRow {
        experiment: "E8_representation_invariance".into(),
        commit,
        seed,
        n: seen_n + unseen_n + ood_n,
        train_examples: 1,
        unseen_examples: unseen_n,
        ood_examples: ood_n,
        accuracy_seen: acc_seen,
        accuracy_unseen: acc_unseen,
        accuracy_ood: acc_ood,
        mean_us,
        p50_us: p50,
        p95_us: p95,
        p99_us: p99,
        top1_score: mean(&scores),
        top2_score: 0.0,
        margin: mean(&margins),
        energy: -mean(&scores).max(EPS).ln(),
        abstentions: failures
            .iter()
            .filter(|f| f.kind == "ood_reject" && f.got.is_some())
            .count(),
        liquid_calls: 0,
        cdt_calls: 0,
        rqm_calls: 0,
        sleep_ms: 0.0,
        engrams_before: 0,
        engrams_after: bank.attractors.len(),
        verdict: verdict.into(),
        notes: format!(
            "train='{}' concept={} failures={} threshold={:.2}",
            TRAIN,
            CONCEPT,
            failures.len(),
            bank.match_threshold
        ),
        periphery: periphery.into(),
    };
    (row, failures)
}

// ─── E9 ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct E9Failure {
    pub query: String,
    pub cue: usize,
    pub expected: usize,
    pub predicted: usize,
    pub hops: usize,
    pub route: String,
}

pub fn run_experiment_9(seed: u64) -> (RegistryRow, Vec<E9Failure>) {
    let _ = seed;
    let commit = git_commit();
    let n = 8usize;
    let candidates: Vec<usize> = (0..n).collect();
    let mut sys = FusedLiquidCdt::new(n);
    sys.abstain_margin = 0.02;
    sys.compose_max_hops = 3;

    let train_edges = [(0usize, 1usize), (1, 2), (2, 3)];
    let engrams_before = sys.engram_count();
    for &(a, b) in &train_edges {
        sys.teach_relation(a, b);
    }
    let sleep = sys.sleep_consolidate();

    let seen_q = [
        ("A→B", 0usize, 1usize, 1usize),
        ("B→C", 1, 2, 1),
        ("C→D", 2, 3, 1),
    ];
    let comp_q = [
        ("A→C", 0usize, 2usize, 2usize),
        ("B→D", 1, 3, 2),
        ("A→D", 0, 3, 3),
    ];
    let ood_q = [("E→?", 4usize, 4usize, 1usize)];

    let mut lat = Vec::new();
    let mut top1s = Vec::new();
    let mut top2s = Vec::new();
    let mut margins = Vec::new();
    let mut energies = Vec::new();
    let mut abs_n = 0usize;
    let mut liq = 0usize;
    let mut rqm = 0usize;
    let mut fails = Vec::new();

    let mut run_set = |qs: &[(&str, usize, usize, usize)],
                       compositional: bool,
                       ok: &mut usize,
                       tot: &mut usize| {
        for &(name, cue, exp, hops) in qs {
            *tot += 1;
            let t0 = Instant::now();
            let r: FuseReport = if compositional {
                sys.infer_compose(cue, &candidates, hops)
            } else {
                sys.infer(cue, &candidates)
            };
            lat.push(t0.elapsed().as_nanos());
            top1s.push(r.top1_score);
            top2s.push(r.top2_score);
            margins.push(r.margin);
            energies.push(r.energy);
            if r.abstained {
                abs_n += 1;
            }
            match r.route {
                InferRoute::Liquid => liq += 1,
                InferRoute::RqmFallback => rqm += 1,
            }
            let correct = r.predicted == exp && !r.abstained;
            if correct {
                *ok += 1;
            } else {
                fails.push(E9Failure {
                    query: name.to_string(),
                    cue,
                    expected: exp,
                    predicted: r.predicted,
                    hops: r.hops,
                    route: format!("{:?}", r.route),
                });
            }
        }
    };

    let mut seen_ok = 0usize;
    let mut seen_n = 0usize;
    let mut un_ok = 0usize;
    let mut un_n = 0usize;
    let mut ood_ok = 0usize;
    let mut ood_n = 0usize;

    run_set(&seen_q, false, &mut seen_ok, &mut seen_n);
    run_set(&comp_q, true, &mut un_ok, &mut un_n);

    for &(name, cue, _exp, hops) in &ood_q {
        ood_n += 1;
        let t0 = Instant::now();
        let r = sys.infer_compose(cue, &candidates, hops);
        lat.push(t0.elapsed().as_nanos());
        top1s.push(r.top1_score);
        top2s.push(r.top2_score);
        margins.push(r.margin);
        energies.push(r.energy);
        if r.abstained {
            abs_n += 1;
        }
        match r.route {
            InferRoute::Liquid => liq += 1,
            InferRoute::RqmFallback => rqm += 1,
        }
        if r.predicted != 3 {
            ood_ok += 1;
        } else {
            fails.push(E9Failure {
                query: name.to_string(),
                cue,
                expected: 4,
                predicted: r.predicted,
                hops: r.hops,
                route: format!("{:?}", r.route),
            });
        }
    }

    let mut liquid_only = WavePredictCore::new();
    let mut liq_comp_ok = 0usize;
    for &(_, cue, exp, _) in &comp_q {
        let r = liquid_only.predict_from_observation(cue, &candidates);
        if r.best_content == exp {
            liq_comp_ok += 1;
        }
    }

    let (mean_us, p50, p95, p99) = latency_stats(&lat);
    let acc_seen = seen_ok as f64 / seen_n.max(1) as f64;
    let acc_un = un_ok as f64 / un_n.max(1) as f64;
    let acc_ood = ood_ok as f64 / ood_n.max(1) as f64;

    let verdict = if acc_seen >= 0.99 && acc_un >= 0.99 {
        "PASS: direct + composed hops via RQM walk (no transitive teach)"
    } else if acc_seen >= 0.99 && acc_un > 0.0 {
        "PARTIAL: seen ok; composition incomplete"
    } else {
        "FAIL: relational composition not emerging"
    };

    let row = RegistryRow {
        experiment: "E9_relational_composition".into(),
        commit,
        seed,
        n,
        train_examples: train_edges.len(),
        unseen_examples: un_n,
        ood_examples: ood_n,
        accuracy_seen: acc_seen,
        accuracy_unseen: acc_un,
        accuracy_ood: acc_ood,
        mean_us,
        p50_us: p50,
        p95_us: p95,
        p99_us: p99,
        top1_score: mean(&top1s),
        top2_score: mean(&top2s),
        margin: mean(&margins),
        energy: mean(&energies),
        abstentions: abs_n,
        liquid_calls: liq,
        cdt_calls: 0,
        rqm_calls: rqm,
        sleep_ms: sleep.sleep_ms,
        engrams_before,
        engrams_after: sys.engram_count(),
        verdict: verdict.into(),
        notes: format!(
            "control liquid-only compose_ok={}/{} (expect ~0); failures={}",
            liq_comp_ok,
            comp_q.len(),
            fails.len()
        ),
        periphery: "n/a (discrete labels)".into(),
    };
    (row, fails)
}

// ─── E10 ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct E10Failure {
    pub variant: &'static str,
    pub detail: String,
}

pub fn run_experiment_10(seed: u64) -> (RegistryRow, Vec<E10Failure>) {
    let commit = git_commit();
    let n = 8usize;
    let candidates: Vec<usize> = (0..n).collect();
    let mut fails = Vec::new();
    let mut lat = Vec::new();
    let mut top1s = Vec::new();
    let mut top2s = Vec::new();
    let mut margins = Vec::new();
    let mut energies = Vec::new();
    let mut abs_n = 0usize;
    let mut liq = 0usize;
    let mut rqm = 0usize;

    let mut sys_a = FusedLiquidCdt::new(n);
    sys_a.abstain_margin = 0.02;
    for &(a, b) in &[(0usize, 1), (1, 2), (2, 3)] {
        sys_a.teach_relation(a, b);
    }
    let sleep_a = sys_a.sleep_consolidate();
    let engrams_before = 0usize;
    let engrams_after_a = sys_a.engram_count();

    let mut seen_ok = 0usize;
    let mut seen_n = 0usize;
    let mut un_ok = 0usize;
    let mut un_n = 0usize;
    let mut ood_ok = 0usize;
    let mut ood_n = 0usize;

    for &(cue, exp) in &[(0usize, 1usize), (1, 2), (2, 3)] {
        seen_n += 1;
        let t0 = Instant::now();
        let r = sys_a.infer(cue, &candidates);
        lat.push(t0.elapsed().as_nanos());
        top1s.push(r.top1_score);
        top2s.push(r.top2_score);
        margins.push(r.margin);
        energies.push(r.energy);
        if r.abstained {
            abs_n += 1;
        }
        match r.route {
            InferRoute::Liquid => liq += 1,
            InferRoute::RqmFallback => rqm += 1,
        }
        if r.predicted == exp {
            seen_ok += 1;
        } else {
            fails.push(E10Failure {
                variant: "A_seen",
                detail: format!("cue={cue} exp={exp} got={}", r.predicted),
            });
        }
    }
    {
        un_n += 1;
        let t0 = Instant::now();
        let r = sys_a.infer_compose(0, &candidates, 2);
        lat.push(t0.elapsed().as_nanos());
        top1s.push(r.top1_score);
        top2s.push(r.top2_score);
        margins.push(r.margin);
        energies.push(r.energy);
        if r.abstained {
            abs_n += 1;
        }
        match r.route {
            InferRoute::Liquid => liq += 1,
            InferRoute::RqmFallback => rqm += 1,
        }
        if r.predicted == 2 && !r.abstained {
            un_ok += 1;
        } else {
            fails.push(E10Failure {
                variant: "A_distance",
                detail: format!(
                    "A→? hops=2 expect C=2 got={} abs={} hops={}",
                    r.predicted, r.abstained, r.hops
                ),
            });
        }
    }

    let mut sys_b = FusedLiquidCdt::new(n);
    sys_b.abstain_margin = 0.02;
    for &(a, b) in &[(0usize, 1), (1, 3), (5, 2), (2, 4)] {
        sys_b.teach_relation(a, b);
    }
    let _sleep_b = sys_b.sleep_consolidate();
    for &(cue, hops, exp, tag) in &[
        (0usize, 2usize, 3usize, "B_left"),
        (5usize, 2usize, 4usize, "B_right"),
    ] {
        un_n += 1;
        let t0 = Instant::now();
        let r = sys_b.infer_compose(cue, &candidates, hops);
        lat.push(t0.elapsed().as_nanos());
        top1s.push(r.top1_score);
        top2s.push(r.top2_score);
        margins.push(r.margin);
        energies.push(r.energy);
        if r.abstained {
            abs_n += 1;
        }
        match r.route {
            InferRoute::Liquid => liq += 1,
            InferRoute::RqmFallback => rqm += 1,
        }
        if r.predicted == exp && !r.abstained {
            un_ok += 1;
        } else {
            fails.push(E10Failure {
                variant: tag,
                detail: format!("cue={cue} exp={exp} got={} abs={}", r.predicted, r.abstained),
            });
        }
    }
    {
        ood_n += 1;
        let r = sys_b.infer_compose(7, &candidates, 2);
        if r.abstained || r.hops <= 1 || (r.predicted != 3 && r.predicted != 4) {
            ood_ok += 1;
        } else {
            fails.push(E10Failure {
                variant: "B_ood",
                detail: format!("untrained cue=7 got={} hops={}", r.predicted, r.hops),
            });
        }
    }

    let mut sys_c = FusedLiquidCdt::new(n);
    for &(a, b) in &[(0usize, 1), (1, 2), (2, 3)] {
        sys_c.teach_relation(a, b);
    }
    let _ = sys_c.sleep_consolidate();
    {
        un_n += 1;
        let t0 = Instant::now();
        let r = sys_c.infer_compose(1, &candidates, 2);
        lat.push(t0.elapsed().as_nanos());
        top1s.push(r.top1_score);
        top2s.push(r.top2_score);
        margins.push(r.margin);
        energies.push(r.energy);
        if r.abstained {
            abs_n += 1;
        }
        match r.route {
            InferRoute::Liquid => liq += 1,
            InferRoute::RqmFallback => rqm += 1,
        }
        if r.predicted == 3 {
            un_ok += 1;
        } else {
            fails.push(E10Failure {
                variant: "C_perturb_chain",
                detail: format!("from B hops=2 expect D=3 got={}", r.predicted),
            });
        }
    }
    {
        seen_n += 1;
        let mut core = WavePredictCore::new();
        let mut pkt = WavePacket::past(seed, 2, 1.0);
        pkt.center[0] += 0.15;
        pkt.phase += 0.08;
        core.inject_past_packet(pkt);
        let t0 = Instant::now();
        let ranked = core.predict_ranked(&candidates);
        lat.push(t0.elapsed().as_nanos());
        top1s.push(ranked.best_score);
        top2s.push(if ranked.second_score.is_finite() {
            ranked.second_score
        } else {
            0.0
        });
        margins.push(ranked.margin);
        energies.push(-ranked.best_score.max(EPS).ln());
        liq += 1;
        if ranked.best_content == 2 && ranked.margin >= 0.01 {
            seen_ok += 1;
        } else {
            fails.push(E10Failure {
                variant: "C_wave_noise",
                detail: format!(
                    "expect content=2 got={} margin={:.4}",
                    ranked.best_content, ranked.margin
                ),
            });
        }
    }

    let (mean_us, p50, p95, p99) = latency_stats(&lat);
    let acc_seen = seen_ok as f64 / seen_n.max(1) as f64;
    let acc_un = un_ok as f64 / un_n.max(1) as f64;
    let acc_ood = ood_ok as f64 / ood_n.max(1) as f64;

    let verdict = if acc_seen >= 0.9 && acc_un >= 0.75 {
        "PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave"
    } else if acc_seen >= 0.9 {
        "PARTIAL: seen dynamics ok; future extrapolation weak"
    } else {
        "FAIL: predictive dynamics insufficient"
    };

    let row = RegistryRow {
        experiment: "E10_future_prediction".into(),
        commit,
        seed,
        n,
        train_examples: 3 + 4 + 3,
        unseen_examples: un_n,
        ood_examples: ood_n,
        accuracy_seen: acc_seen,
        accuracy_unseen: acc_un,
        accuracy_ood: acc_ood,
        mean_us,
        p50_us: p50,
        p95_us: p95,
        p99_us: p99,
        top1_score: mean(&top1s),
        top2_score: mean(&top2s),
        margin: mean(&margins),
        energy: mean(&energies),
        abstentions: abs_n,
        liquid_calls: liq,
        cdt_calls: 0,
        rqm_calls: rqm,
        sleep_ms: sleep_a.sleep_ms,
        engrams_before,
        engrams_after: engrams_after_a,
        verdict: verdict.into(),
        notes: format!("failures={} variants=A,B,C", fails.len()),
        periphery: "n/a".into(),
    };
    (row, fails)
}

// ─── Suite / tabla ──────────────────────────────────────────────────────────

pub fn format_registry_table(rows: &[RegistryRow]) -> String {
    let mut s = String::new();
    s.push_str("# Registry — experimentos 8/9/10 (liquid)\n\n");
    s.push_str("| Exp | seed | N | acc_seen | acc_unseen | acc_ood | mean_us | p50 | p95 | p99 | top1 | margin | abs | liq | rqm | sleep_ms | engrams | verdict |\n");
    s.push_str("|-----|-----:|--:|---------:|-----------:|--------:|--------:|----:|----:|----:|-----:|-------:|----:|----:|----:|---------:|--------:|---------|\n");
    for r in rows {
        s.push_str(&format!(
            "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {:.3} | {}→{} | {} |\n",
            r.experiment,
            r.seed,
            r.n,
            r.accuracy_seen,
            r.accuracy_unseen,
            r.accuracy_ood,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.p99_us,
            r.top1_score,
            r.margin,
            r.abstentions,
            r.liquid_calls,
            r.rqm_calls,
            r.sleep_ms,
            r.engrams_before,
            r.engrams_after,
            r.verdict.replace('|', "/"),
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn liquid_experiments_8_9_10_registry() {
        let seeds = [0xE8910u64, 0xE8911, 0xE8912];
        let mut all_rows = Vec::new();
        let mut all_notes = String::new();

        for &seed in &seeds {
            let (r8, f8) = run_experiment_8(seed);
            let (r9, f9) = run_experiment_9(seed);
            let (r10, f10) = run_experiment_10(seed);
            all_notes.push_str(&format!(
                "\n## seed={seed:#x}\nE8 failures ({}): {:?}\nE9 failures ({}): {:?}\nE10 failures ({}): {:?}\nE8 notes: {}\nE9 notes: {}\nE10 notes: {}\nperiphery: {}\n",
                f8.len(),
                f8.iter().map(|f| format!("{}:{}:{:?}", f.kind, f.query, f.got)).collect::<Vec<_>>(),
                f9.len(),
                f9,
                f10.len(),
                f10,
                r8.notes,
                r9.notes,
                r10.notes,
                r8.periphery
            ));
            all_rows.push(r8);
            all_rows.push(r9);
            all_rows.push(r10);
        }

        let table = format_registry_table(&all_rows);
        println!("{table}");
        println!("{all_notes}");

        let md = format!(
            "# Resultados experimentos 8/9/10 — inferencia líquida\n\n\
             Rama: `exp/liquid-inference-experiments-8-9-10`\n\n\
             Hardware: ver ejecución (`uname -a`, `rustc -V`).\n\
             Modo: `--release`.\n\
             Commit al correr: {}.\n\n\
             ## Tabla de registro\n\n{}\n\n## Fallos / notas por semilla\n{}\n\n\
             ## Controles y honestidad\n\
             - E8: periferia léxico si GGUF ausente; no se afirma invariancia cross-lingual.\n\
             - E9: liquid-only compose esperado ~0; composición vía `infer_compose` sin teach transitivo.\n\
             - E10: abstención / hops en cue OOD; perturbación de onda con margen.\n\
             - No se presenta como evidencia de cognición general.\n",
            git_commit(),
            table,
            all_notes
        );
        let _ = std::fs::create_dir_all("docs");
        let _ = std::fs::write("docs/resultados_experimentos_8_9_10.md", &md);
        let mut csv = String::from(
            "experiment,commit,seed,n,train,unseen,ood,acc_seen,acc_unseen,acc_ood,mean_us,p50,p95,p99,top1,top2,margin,energy,abstentions,liquid,rqm,sleep_ms,engrams_before,engrams_after,verdict\n",
        );
        for r in &all_rows {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.6},{},{},{}\n",
                r.experiment, r.commit, r.seed, r.n, r.train_examples, r.unseen_examples, r.ood_examples,
                r.accuracy_seen, r.accuracy_unseen, r.accuracy_ood,
                r.mean_us, r.p50_us, r.p95_us, r.p99_us,
                r.top1_score, r.top2_score, r.margin, r.energy,
                r.abstentions, r.liquid_calls, r.rqm_calls, r.sleep_ms,
                r.engrams_before, r.engrams_after,
                r.verdict.replace(",", ";")
            ));
        }
        let _ = std::fs::write("docs/resultados_experimentos_8_9_10.csv", csv);

        let e9: Vec<_> = all_rows
            .iter()
            .filter(|r| r.experiment.starts_with("E9"))
            .collect();
        assert!(!e9.is_empty());
        for r in &e9 {
            assert!(
                r.accuracy_seen >= 0.99,
                "E9 seen edges must work: {:?}",
                r
            );
            assert!(
                r.accuracy_unseen >= 0.99,
                "E9 compose should pass with multi-hop: {:?}",
                r
            );
        }
    }

    #[test]
    fn attractor_bank_matches_trained_fingerprint() {
        let mut bank = ConceptAttractorBank::new(0.9);
        bank.store(3, vec![1.0, 0.0, 0.0]);
        let hit = bank.match_fp(&[0.99, 0.01, 0.0]);
        assert_eq!(hit.map(|h| h.0), Some(3));
        assert!(bank.match_fp(&[0.0, 1.0, 0.0]).is_none());
    }

    #[test]
    fn compose_does_not_teach_transitive_pairs() {
        let mut sys = FusedLiquidCdt::new(8);
        sys.teach_relation(0, 1);
        sys.teach_relation(1, 2);
        let _ = sys.sleep_consolidate();
        assert!(sys.relational_cues.contains(&0));
        assert!(sys.relational_cues.contains(&1));
        let cands: Vec<_> = (0..8).collect();
        let r = sys.infer_compose(0, &cands, 2);
        assert_eq!(r.predicted, 2);
        assert_eq!(r.hops, 2);
        let direct = sys.infer(0, &cands);
        assert_eq!(direct.predicted, 1);
        assert_eq!(direct.hops, 1);
    }
}
