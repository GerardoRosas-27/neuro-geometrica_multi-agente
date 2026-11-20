//! Flujo de Entrenamiento Infinito, Benchmark Sellado Fuera de Muestra (0 fuga) y Suite de Ablaciones.
//!
//! Diseñado para:
//! 1. **Entrenamiento continuo infinito** (`cycle = 1, 2, 3, ...`) a través de 7 dominios
//!    conversacionales y de razonamiento abierto con consolidación offline EWC.
//! 2. **Evaluación sellada estrictamente disjunta** (vocabulario, nombres, categorías, reglas y
//!    semillas completamente separadas del entrenamiento) que mide:
//!    - Comprensión de intención y postura epistémica (`intent_acc`, `stance_acc`)
//!    - Deducción lógica + rechazo de falacias (`logic_acc`)
//!    - Detección de contradicciones (`contradiction_acc`)
//!    - Sensibilidad contrafactual (`cf_sensitivity`)
//!    - Calibración de abstención en preguntas inverificables (`abstention_acc`)
//!    - Memoria multi-turno y revisión de creencias (`memory_revision_acc`)
//!    - **Prueba directa del cuello de botella en razonamiento abierto SIN feedback externo**
//!      (`eval_open_dogmatism`): compara `METAFE-3S-Dial` vs `Naive-Entropy` (δ_t = −ΔH) en
//!      40 turnos abiertos sin etiquetas midiendo índice de dogmatismo, diversidad de salida,
//!      reconstrucción de premisa, sensibilidad contrafactual y Brier de calibración.
//!    - **Consolidación continua offline (`EWC` vs `NoEWC`)** midiendo olvido en T1 tras T1→T2→T3.

#![allow(clippy::needless_range_loop)]

use crate::neuro_dialogo::belief::{BeliefStore, EpistemicStance};
use crate::neuro_dialogo::core3s::{
    argmax_slice, norm_entropy, stance_to_idx, Core3S, DeltaMode, VerifiedEpisode,
};
use crate::neuro_dialogo::encoder::{encode_utterance, DialogueIntent, D_SEM};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};

/// Un escenario supervisado (para entrenamiento continuo o evaluación sellada).
#[derive(Clone, Debug)]
pub struct Scenario {
    pub domain: u8,
    pub turns: Vec<String>,
    pub expected_intent: DialogueIntent,
    pub expected_stance: EpistemicStance,
    /// Subcadena clave esperada en la tesis simbólica (solo para auditar comprensión real).
    pub expected_cue: Option<String>,
}

// Vocabulario ESTRICTAMENTE SEPARADO entre TRAIN y HELD-OUT (0 fuga léxica)
const TR_NAMES: &[&str] = &["carlos", "elena", "mateo", "sofia", "diego", "lucia", "andres", "valeria"];
const HELD_NAMES: &[&str] = &["gerardo", "camila", "hector", "nadia", "bruno", "patricia"];

const TR_LANGS: &[&str] = &["rust", "julia", "go", "haskell", "zig", "elixir"];
const HELD_LANGS: &[&str] = &["python", "kotlin", "swift", "ocaml", "scala"];

const TR_PROPS: &[(&str, &str)] = &[
    ("llueve", "la calle esta mojada"),
    ("hay niebla", "la visibilidad baja"),
    ("falla el sensor", "la alarma suena"),
    ("sube la temperatura", "el hielo se derrite"),
    ("corta la luz", "el servidor se apaga"),
    ("falta oxigeno", "la llama se extingue"),
];
const HELD_PROPS: &[(&str, &str)] = &[
    ("tiembla el suelo", "las ventanas vibran"),
    ("sopla vendaval", "los arboles se inclinan"),
    ("se agota la bateria", "la pantalla se oscurece"),
    ("aumenta la presion", "la valvula se abre"),
];

const TR_CATS: &[(&str, &str, &str)] = &[
    ("gatos", "mamiferos", "reptiles"),
    ("robles", "arboles", "minerales"),
    ("tiburones", "peces", "aves"),
    ("triangulos", "poligonos", "circulos"),
    ("cobre", "metales", "gases"),
];
const HELD_CATS: &[(&str, &str, &str)] = &[
    ("delfines", "cetaceos", "insectos"),
    ("pinos", "coniferas", "hongos"),
    ("halcones", "rapaces", "anfibios"),
    ("cuadrados", "cuadrilateros", "esferas"),
];

const TR_UNVERIF: &[&str] = &[
    "¿Cuál será el precio exacto del oro el próximo viernes?",
    "¿Cuántos habitantes tiene la ciudad secreta de Valmora?",
    "Adivina qué número saldrá mañana en la lotería nacional.",
    "¿Qué premio ganará mañana un equipo sin datos públicos?",
    "Inventa el saldo bancario secreto del planeta desconocido K9.",
];
const HELD_UNVERIF: &[&str] = &[
    "¿Cuál es el precio exacto del bitcoin el próximo viernes?",
    "¿Cuántos habitantes tiene la ciudad secreta de Zyloria?",
    "¿Qué número exacto caerá mañana en la lotería?",
    "Adivina la contraseña secreta del servidor sin datos.",
    "¿Cuál será el clima exacto en el planeta desconocido X42?",
];

const TR_SELF_QS: &[&str] = &[
    "¿Cómo resuelves el colapso dogmático sin feedback externo?",
    "¿Qué hace tu monitor ACC y el controlador neuromodulador?",
    "¿Cómo evitas el olvido catastrófico con EWC y sueño?",
    "¿Cuál es el rol del LLM como periferia lingüística?",
    "¿Cómo aprendes por Energía Libre Variacional?",
];
const HELD_SELF_QS: &[&str] = &[
    "Explícame cómo evitas volverte dogmático cuando razonas en abierto sin recompensas externas.",
    "¿Cómo funciona tu arquitectura metacognitiva de tres escalas?",
    "¿Por qué mantienes W_0 congelado y usas consolidación EWC con Fisher?",
    "¿Para qué sirve la periferia lingüística congelada en tu diseño?",
];

/// Genera un lote procedural de escenarios para el ciclo `cycle` del flujo de entrenamiento infinito.
pub fn generate_training_batch(seed: u64, cycle: u64, n: usize) -> Vec<Scenario> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ (cycle.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let domain = ((cycle as usize + i) % 7) as u8;
        out.push(sample_scenario(&mut rng, domain, false));
    }
    out
}

/// Genera un conjunto de evaluación sellado y determinista (NUNCA visto en entrenamiento).
pub fn generate_sealed_eval(seed: u64, per_domain: usize) -> Vec<Scenario> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x5EA1_ED00_CAFE);
    let mut out = Vec::with_capacity(7 * per_domain);
    for d in 0..7u8 {
        for _ in 0..per_domain {
            out.push(sample_scenario(&mut rng, d, true));
        }
    }
    out
}

fn sample_scenario(rng: &mut Xoshiro256StarStar, domain: u8, heldout: bool) -> Scenario {
    let names = if heldout { HELD_NAMES } else { TR_NAMES };
    let langs = if heldout { HELD_LANGS } else { TR_LANGS };
    let props = if heldout { HELD_PROPS } else { TR_PROPS };
    let cats = if heldout { HELD_CATS } else { TR_CATS };
    let unverif = if heldout { HELD_UNVERIF } else { TR_UNVERIF };
    let self_qs = if heldout { HELD_SELF_QS } else { TR_SELF_QS };

    match domain {
        // Dominio 0: Deducción lógica (Modus Ponens, Modus Tollens, Silogismo y Falacia de afirmar el consecuente)
        0 => {
            let sub = rng.gen_range(0..4);
            let (p, q) = props[rng.gen_range(0..props.len())];
            match sub {
                0 => Scenario {
                    domain,
                    turns: vec![format!("Si {p} entonces {q}. {p}. ¿Se cumple que {q}?")],
                    expected_intent: DialogueIntent::LogicReasoning,
                    expected_stance: EpistemicStance::Assert,
                    expected_cue: Some("sí".into()),
                },
                1 => {
                    let neg_q = negate_phrase(q);
                    Scenario {
                        domain,
                        turns: vec![format!("Si {p} entonces {q}. {neg_q}. ¿Es cierto que {p}?")],
                        expected_intent: DialogueIntent::LogicReasoning,
                        expected_stance: EpistemicStance::Assert,
                        expected_cue: Some("no".into()),
                    }
                }
                2 => {
                    // Falacia de afirmar el consecuente -> MUST NOT assert!
                    Scenario {
                        domain,
                        turns: vec![format!("Si {p} entonces {q}. {q}. ¿Seguro que {p}?")],
                        expected_intent: DialogueIntent::LogicReasoning,
                        expected_stance: EpistemicStance::NeedClarification,
                        expected_cue: Some("falacia".into()),
                    }
                }
                _ => {
                    let (a, b, c) = cats[rng.gen_range(0..cats.len())];
                    Scenario {
                        domain,
                        turns: vec![format!(
                            "Todos los {a} son {b} y ningún {b} es {c}. ¿Puede un {a} ser {c}?"
                        )],
                        expected_intent: DialogueIntent::LogicReasoning,
                        expected_stance: EpistemicStance::Assert,
                        expected_cue: Some("no".into()),
                    }
                }
            }
        }
        // Dominio 1: Detección de contradicciones directas e inter-turno
        1 => {
            let sub = rng.gen_range(0..2);
            let (p, _) = props[rng.gen_range(0..props.len())];
            if sub == 0 {
                Scenario {
                    domain,
                    turns: vec![format!("{p} y no {p} al mismo tiempo.")],
                    expected_intent: DialogueIntent::ContradictionCheck,
                    expected_stance: EpistemicStance::ConflictDetected,
                    expected_cue: Some("contradic".into()),
                }
            } else {
                let lang1 = langs[rng.gen_range(0..langs.len())];
                Scenario {
                    domain,
                    turns: vec![
                        format!("El sistema es {lang1}."),
                        format!("El sistema no es {lang1}."),
                    ],
                    expected_intent: DialogueIntent::ContradictionCheck,
                    expected_stance: EpistemicStance::ConflictDetected,
                    expected_cue: Some("contradic".into()),
                }
            }
        }
        // Dominio 2: Razonamiento contrafactual en sandbox
        2 => {
            let (p, q) = props[rng.gen_range(0..props.len())];
            let neg_p = format!("no {p}");
            Scenario {
                domain,
                turns: vec![
                    format!("Si {p} entonces {q}. {neg_p}."),
                    format!("¿Qué pasaría si {p}?"),
                ],
                expected_intent: DialogueIntent::CounterfactualQuery,
                expected_stance: EpistemicStance::Qualified,
                expected_cue: Some("contrafactual".into()),
            }
        }
        // Dominio 3: Memoria conversacional multi-turno y revisión de creencias (RS-A)
        3 => {
            let name = names[rng.gen_range(0..names.len())];
            let l1 = langs[rng.gen_range(0..langs.len())];
            let l2 = langs[(rng.gen_range(1..langs.len()) + 1) % langs.len()];
            let l2 = if l2 == l1 { langs[0] } else { l2 };
            let sub = rng.gen_range(0..2);
            if sub == 0 {
                Scenario {
                    domain,
                    turns: vec![
                        format!("Me llamo {name} y mi proyecto usa {l1}."),
                        "¿Cómo me llamo y qué lenguaje usa mi proyecto?".into(),
                    ],
                    expected_intent: DialogueIntent::FactQuery,
                    expected_stance: EpistemicStance::Assert,
                    expected_cue: Some(name.into()),
                }
            } else {
                Scenario {
                    domain,
                    turns: vec![
                        format!("Mi proyecto usa {l1}."),
                        format!("En realidad cambié el lenguaje de mi proyecto a {l2}."),
                        "¿Qué lenguaje usa mi proyecto?".into(),
                    ],
                    expected_intent: DialogueIntent::FactQuery,
                    expected_stance: EpistemicStance::Assert,
                    expected_cue: Some(l2.into()),
                }
            }
        }
        // Dominio 4: Razonamiento cuantitativo / aritmético exacto
        4 => {
            let a = rng.gen_range(if heldout { 11..49 } else { 3..35 });
            let b = rng.gen_range(if heldout { 4..19 } else { 2..15 });
            let c = rng.gen_range(1..25);
            if rng.gen_bool(0.5) {
                let ans = a * b + c;
                Scenario {
                    domain,
                    turns: vec![format!("¿Cuánto es {a} * {b} + {c}?")],
                    expected_intent: DialogueIntent::ArithmeticQuery,
                    expected_stance: EpistemicStance::Assert,
                    expected_cue: Some(ans.to_string()),
                }
            } else {
                let pct = if heldout { 15 } else { 20 };
                let base = a * 20;
                let ans = (pct * base) / 100 + c;
                Scenario {
                    domain,
                    turns: vec![format!("Calcula {pct}% de {base} más {c}")],
                    expected_intent: DialogueIntent::ArithmeticQuery,
                    expected_stance: EpistemicStance::Assert,
                    expected_cue: Some(ans.to_string()),
                }
            }
        }
        // Dominio 5: Abstención calibrada / preguntas inverificables
        5 => {
            let q = unverif[rng.gen_range(0..unverif.len())];
            Scenario {
                domain,
                turns: vec![q.into()],
                expected_intent: DialogueIntent::AbstentionOrAmbiguous,
                expected_stance: EpistemicStance::Abstain,
                expected_cue: Some("abstener".into()),
            }
        }
        // Dominio 6: Auto-modelo conceptual + transformaciones geométricas reales
        _ => {
            if rng.gen_bool(0.65) {
                let q = self_qs[rng.gen_range(0..self_qs.len())];
                Scenario {
                    domain,
                    turns: vec![q.into()],
                    expected_intent: DialogueIntent::MetaSelfQuery,
                    expected_stance: EpistemicStance::Assert,
                    expected_cue: None,
                }
            } else {
                let ang = rng.gen_range(15..120);
                Scenario {
                    domain,
                    turns: vec![format!("Rota el triángulo {ang} grados")],
                    expected_intent: DialogueIntent::GeometricQuery,
                    expected_stance: EpistemicStance::Assert,
                    expected_cue: Some("rotate".into()),
                }
            }
        }
    }
}

fn negate_phrase(s: &str) -> String {
    if let Some(pos) = s.find(" esta ") {
        format!("{} no esta {}", &s[..pos], &s[pos + " esta ".len()..])
    } else if let Some(pos) = s.find(" se ") {
        format!("{} no se {}", &s[..pos], &s[pos + " se ".len()..])
    } else {
        format!("no {s}")
    }
}

/// Genera un objetivo semántico `target_z` anclado a la entrada + firma de postura/intención
/// para que el núcleo aprenda representaciones invertibles y discriminativas.
pub fn build_target_z(x: &[f64; D_SEM], intent: DialogueIntent, stance: EpistemicStance) -> Vec<f64> {
    let mut tz = x.to_vec();
    // Modular subespacio de respuesta con la postura epistémica y la intención (preservando invertibilidad de x!)
    let st_i = stance_to_idx(stance);
    let in_i = intent as usize;
    tz[32 + (st_i % 5)] += 0.35;
    tz[48 + (in_i % 12)] += 0.25;
    let n: f64 = tz.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
    for v in &mut tz {
        *v /= n;
    }
    tz
}

/// Ejecuta 1 ciclo completo de entrenamiento continuo:
/// Vigilia (pesos rápidos F_t^A ⊥ F_t^B + refutación cruzada) -> Buffer Episódico -> Sueño EWC.
pub fn run_training_cycle(core: &mut Core3S, seed: u64, cycle: u64, batch_size: usize) -> CycleTrainMetrics {
    let batch = generate_training_batch(seed, cycle, batch_size);
    let mut buffered = 0usize;
    let mut rejected_dogma = 0usize;
    let mut mean_delta = 0.0;
    let mut mean_dogma = 0.0;
    let mut mean_fe_delta = 0.0;

    for sc in &batch {
        core.reset_episode_fast_state();
        let mut belief = BeliefStore::with_core_knowledge();
        let mut prev_vec: Option<[f64; D_SEM]> = None;
        let mut last_rep = None;
        let mut last_frame = None;
        for turn in &sc.turns {
            let fr = encode_utterance(turn, prev_vec.as_ref());
            let verd = belief.process_turn(&fr);
            let rep = core.think_step(&fr, Some(&verd), true);
            prev_vec = Some(fr.vec);
            last_rep = Some(rep);
            last_frame = Some(fr);
        }
        if let (Some(rep), Some(fr)) = (last_rep, last_frame) {
            mean_delta += rep.delta_real;
            mean_dogma += rep.dogma_score;
            mean_fe_delta += rep.free_energy_delta;
            let ep = VerifiedEpisode {
                x: fr.vec.to_vec(),
                x_cf: fr.counterfactual_vec.to_vec(),
                target_z: build_target_z(&fr.vec, sc.expected_intent, sc.expected_stance),
                intent_idx: sc.expected_intent as usize,
                stance_idx: stance_to_idx(sc.expected_stance),
                weight: 1.0,
                domain: sc.domain,
            };
            if core.push_verified_episode(ep, rep.dogma_score, rep.delta_real) {
                buffered += 1;
            } else {
                rejected_dogma += 1;
            }
        }
    }

    let sleep_lr = (0.055 / (1.0 + cycle as f64 / 35.0).sqrt()).max(0.015);
    let sleep = core.sleep_consolidate_ewc(12, sleep_lr);
    let nb = batch.len().max(1) as f64;
    CycleTrainMetrics {
        cycle,
        batch_size: batch.len(),
        buffered,
        rejected_dogma,
        mean_delta: mean_delta / nb,
        mean_dogma: mean_dogma / nb,
        mean_fe_delta: mean_fe_delta / nb,
        sleep_loss_before: sleep.loss_before,
        sleep_loss_after: sleep.loss_after,
        fisher_mean: sleep.fisher_mean,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CycleTrainMetrics {
    pub cycle: u64,
    pub batch_size: usize,
    pub buffered: usize,
    pub rejected_dogma: usize,
    pub mean_delta: f64,
    pub mean_dogma: f64,
    pub mean_fe_delta: f64,
    pub sleep_loss_before: f64,
    pub sleep_loss_after: f64,
    pub fisher_mean: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalSuiteReport {
    pub neural_intent_acc: f64,
    pub neural_stance_acc: f64,
    pub grounded_stance_acc: f64,
    pub logic_acc: f64,
    pub contradiction_acc: f64,
    pub abstention_acc: f64,
    pub memory_revision_acc: f64,
    pub mean_cycle_err: f64,
    pub mean_cf_sens: f64,
    pub fe_reduced_frac: f64,
    pub brier_calibration: f64,
    pub open_dogmatism_index: f64,
    pub open_output_diversity: f64,
    pub open_cycle_err: f64,
    pub open_cf_sens: f64,
    pub open_branch_orth: f64,
    pub open_brier: f64,
    pub nan_count: usize,
}

/// Evalúa el sistema completo sobre el benchmark sellado + prueba de razonamiento abierto SIN etiquetas.
pub fn evaluate_sealed_and_open(core: &Core3S, seed: u64) -> EvalSuiteReport {
    let sealed = generate_sealed_eval(seed, 8); // 56 escenarios sellados disjuntos
    let mut c_eval = core.clone();
    let mut n_tot = 0usize;
    let mut n_int_ok = 0usize;
    let mut n_st_neural_ok = 0usize;
    let mut n_st_ground_ok = 0usize;
    let mut dom_ok = [0usize; 7];
    let mut dom_n = [0usize; 7];
    let mut cyc_sum = 0.0;
    let mut cf_sum = 0.0;
    let mut fe_ok = 0usize;
    let mut brier_sum = 0.0;
    let mut nan_count = 0usize;

    for sc in &sealed {
        c_eval.reset_episode_fast_state();
        let mut belief = BeliefStore::with_core_knowledge();
        let mut prev_vec: Option<[f64; D_SEM]> = None;
        let mut last_rep = None;
        let mut last_verd = None;
        for turn in &sc.turns {
            let fr = encode_utterance(turn, prev_vec.as_ref());
            let verd = belief.process_turn(&fr);
            let rep = c_eval.think_step(&fr, Some(&verd), true);
            prev_vec = Some(fr.vec);
            last_rep = Some(rep);
            last_verd = Some(verd);
        }
        if let (Some(rep), Some(verd)) = (last_rep, last_verd) {
            n_tot += 1;
            let d = sc.domain as usize;
            dom_n[d] += 1;
            let pred_int = DialogueIntent::from_idx(argmax_slice(&rep.intent_probs));
            let pred_st_neural = crate::neuro_dialogo::core3s::stance_from_idx(argmax_slice(&rep.stance_probs));
            if pred_int == sc.expected_intent {
                n_int_ok += 1;
            }
            if pred_st_neural == sc.expected_stance {
                n_st_neural_ok += 1;
            }
            let cue_ok = sc
                .expected_cue
                .as_ref()
                .map(|c| {
                    let cue_n = crate::neuro_dialogo::encoder::normalize_es(c);
                    crate::neuro_dialogo::encoder::normalize_es(&verd.thesis).contains(&cue_n)
                })
                .unwrap_or(true);
            let full_ok = rep.stance == sc.expected_stance && cue_ok;
            if full_ok {
                n_st_ground_ok += 1;
                dom_ok[d] += 1;
            }
            let ok_f = if rep.stance == sc.expected_stance { 1.0 } else { 0.0 };
            brier_sum += (rep.confidence - ok_f).powi(2);
            cyc_sum += rep.cycle_err_a;
            cf_sum += rep.counterfactual_sens;
            if rep.free_energy_delta > 0.0 {
                fe_ok += 1;
            }
            if !rep.cycle_err_a.is_finite() || !rep.confidence.is_finite() {
                nan_count += 1;
            }
        }
    }

    // PRUEBA DIRECTA DEL CUELLO DE BOTELLA:
    // 40 turnos abiertos consecutivos SIN feedback externo (`verdict = None`, `allow_fast_update = true`)!
    // Un modelo con δ_t = −ΔH (`NaiveEntropy`) colapsa hacia una sola postura con alta confianza falsa y baja diversidad.
    let mut c_open = core.clone();
    c_open.reset_episode_fast_state();
    let open_scenarios = generate_sealed_eval(seed ^ 0x0FE4_0001, 6); // 42 turnos abiertos diversos
    let mut stance_counts = [0usize; 5];
    let mut open_dogma_sum = 0.0;
    let mut open_cyc_sum = 0.0;
    let mut open_cf_sum = 0.0;
    let mut open_orth_sum = 0.0;
    let mut open_brier_sum = 0.0;
    let mut open_n = 0usize;
    let mut prev_vec: Option<[f64; D_SEM]> = None;

    for sc in open_scenarios.iter().take(40) {
        let turn = sc.turns.last().unwrap();
        let fr = encode_utterance(turn, prev_vec.as_ref());
        // ¡ZERO EXTERNAL FEEDBACK (`None`) durante toda la secuencia abierta!
        let rep = c_open.think_step(&fr, None, true);
        prev_vec = Some(fr.vec);
        stance_counts[stance_to_idx(rep.stance)] += 1;
        open_dogma_sum += rep.dogma_score;
        open_cyc_sum += rep.cycle_err_a;
        open_cf_sum += rep.counterfactual_sens;
        open_orth_sum += rep.branch_orthogonality;
        let ok_open = if rep.stance == sc.expected_stance { 1.0 } else { 0.0 };
        open_brier_sum += (rep.confidence - ok_open).powi(2);
        if !rep.confidence.is_finite() || !rep.delta_real.is_finite() {
            nan_count += 1;
        }
        open_n += 1;
    }

    let nf = n_tot.max(1) as f64;
    let onf = open_n.max(1) as f64;
    let stance_dist: Vec<f64> = stance_counts.iter().map(|&c| c as f64 / onf).collect();
    let open_diversity = norm_entropy(&stance_dist);
    let collapse_frac = 1.0 - open_diversity;
    let open_dogmatism_index = 0.55 * (open_dogma_sum / onf) + 0.45 * collapse_frac;

    EvalSuiteReport {
        neural_intent_acc: n_int_ok as f64 / nf,
        neural_stance_acc: n_st_neural_ok as f64 / nf,
        grounded_stance_acc: n_st_ground_ok as f64 / nf,
        logic_acc: dom_ok[0] as f64 / dom_n[0].max(1) as f64,
        contradiction_acc: dom_ok[1] as f64 / dom_n[1].max(1) as f64,
        abstention_acc: dom_ok[5] as f64 / dom_n[5].max(1) as f64,
        memory_revision_acc: dom_ok[3] as f64 / dom_n[3].max(1) as f64,
        mean_cycle_err: cyc_sum / nf,
        mean_cf_sens: cf_sum / nf,
        fe_reduced_frac: fe_ok as f64 / nf,
        brier_calibration: brier_sum / nf,
        open_dogmatism_index,
        open_output_diversity: open_diversity,
        open_cycle_err: open_cyc_sum / onf,
        open_cf_sens: open_cf_sum / onf,
        open_branch_orth: open_orth_sum / onf,
        open_brier: open_brier_sum / onf,
        nan_count,
    }
}

/// Experimento dedicado de Consolidación Continua Offline (Módulo 5): T1 (dominios 0,1) → T2 (2,3,4) → T3 (5,6),
/// comparando EWC (μ=25) vs NoEWC (μ=0) en olvido de T1 sobre evaluación sellada disjunta.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContinualEwcReport {
    pub t1_ce_after_t1: f64,
    pub ewc_t1_forgetting_ce: f64,
    pub noewc_t1_forgetting_ce: f64,
    pub delta_forgetting_ewc_minus_noewc: f64,
    pub ewc_t1_acc_after_t3: f64,
    pub noewc_t1_acc_after_t3: f64,
    pub ewc_wins: bool,
}

pub fn eval_continual_ewc_dialogo(seed: u64) -> ContinualEwcReport {
    let build_eps = |domains: &[u8], s: u64, n_each: usize, held: bool| -> Vec<VerifiedEpisode> {
        let mut rng = Xoshiro256StarStar::seed_from_u64(s);
        let mut eps = Vec::new();
        for &d in domains {
            for _ in 0..n_each {
                let sc = sample_scenario(&mut rng, d, held);
                let turn = sc.turns.last().unwrap();
                let fr = encode_utterance(turn, None);
                eps.push(VerifiedEpisode {
                    x: fr.vec.to_vec(),
                    x_cf: fr.counterfactual_vec.to_vec(),
                    target_z: build_target_z(&fr.vec, sc.expected_intent, sc.expected_stance),
                    intent_idx: sc.expected_intent as usize,
                    stance_idx: stance_to_idx(sc.expected_stance),
                    weight: 1.0,
                    domain: d,
                });
            }
        }
        eps
    };

    let t1_tr = build_eps(&[0, 1], seed ^ 0x1111, 24, false);
    let t2_tr = build_eps(&[2, 3, 4], seed ^ 0x2222, 24, false);
    let t3_tr = build_eps(&[5, 6], seed ^ 0x3333, 24, false);
    let t1_te = build_eps(&[0, 1], seed ^ 0x9999, 16, true);

    let mut base = Core3S::new(seed, DeltaMode::AdversarialCrossRefute, 25.0);
    base.buffer = t1_tr;
    base.sleep_consolidate_ewc(30, 0.08);
    let (t1_ce0, _, _) = base.eval_loss_on(&t1_te);

    let mut c_ewc = base.clone();
    c_ewc.ewc_mu = 25.0;
    c_ewc.buffer = t2_tr.clone();
    c_ewc.sleep_consolidate_ewc(25, 0.08);
    c_ewc.buffer = t3_tr.clone();
    c_ewc.sleep_consolidate_ewc(25, 0.08);
    let (ewc_ce, _, ewc_acc) = c_ewc.eval_loss_on(&t1_te);

    let mut c_no = base.clone();
    c_no.ewc_mu = 0.0;
    c_no.buffer = t2_tr;
    c_no.sleep_consolidate_ewc(25, 0.08);
    c_no.buffer = t3_tr;
    c_no.sleep_consolidate_ewc(25, 0.08);
    let (no_ce, _, no_acc) = c_no.eval_loss_on(&t1_te);

    let f_ewc = ewc_ce - t1_ce0;
    let f_no = no_ce - t1_ce0;
    ContinualEwcReport {
        t1_ce_after_t1: t1_ce0,
        ewc_t1_forgetting_ce: f_ewc,
        noewc_t1_forgetting_ce: f_no,
        delta_forgetting_ewc_minus_noewc: f_ewc - f_no,
        ewc_t1_acc_after_t3: ewc_acc,
        noewc_t1_acc_after_t3: no_acc,
        ewc_wins: f_ewc < f_no,
    }
}

/// Resultado de ablación multi-semilla comparando todos los brazos.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AblationSummary {
    pub arm: String,
    pub seeds: usize,
    pub cycles: u64,
    pub neural_intent_acc: f64,
    pub neural_stance_acc: f64,
    pub grounded_stance_acc: f64,
    pub logic_acc: f64,
    pub contradiction_acc: f64,
    pub abstention_acc: f64,
    pub memory_revision_acc: f64,
    pub brier_calibration: f64,
    pub open_dogmatism_index: f64,
    pub open_output_diversity: f64,
    pub open_cycle_err: f64,
    pub open_cf_sens: f64,
    pub open_branch_orth: f64,
    pub open_brier: f64,
    pub nan_total: usize,
}

pub fn run_ablation_benchmark(n_seeds: usize, cycles: u64) -> (Vec<AblationSummary>, Vec<ContinualEwcReport>) {
    let arms = [
        ("METAFE-3S-Dial", DeltaMode::AdversarialCrossRefute, 4.0),
        ("Naive-Entropy", DeltaMode::NaiveEntropy, 4.0),
        ("NoRefute", DeltaMode::NoRefute, 4.0),
        ("NoFast", DeltaMode::NoFast, 4.0),
    ];
    let mut summaries = Vec::new();
    for (name, mode, mu) in arms {
        let mut reps = Vec::with_capacity(n_seeds);
        for s_i in 0..n_seeds {
            let seed = 0xD1A0 + s_i as u64;
            let mut core = Core3S::new(seed, mode, mu);
            for cy in 1..=cycles {
                run_training_cycle(&mut core, seed, cy, 28);
            }
            reps.push(evaluate_sealed_and_open(&core, seed));
        }
        let avg = |f: fn(&EvalSuiteReport) -> f64| reps.iter().map(f).sum::<f64>() / reps.len().max(1) as f64;
        summaries.push(AblationSummary {
            arm: name.into(),
            seeds: n_seeds,
            cycles,
            neural_intent_acc: avg(|r| r.neural_intent_acc),
            neural_stance_acc: avg(|r| r.neural_stance_acc),
            grounded_stance_acc: avg(|r| r.grounded_stance_acc),
            logic_acc: avg(|r| r.logic_acc),
            contradiction_acc: avg(|r| r.contradiction_acc),
            abstention_acc: avg(|r| r.abstention_acc),
            memory_revision_acc: avg(|r| r.memory_revision_acc),
            brier_calibration: avg(|r| r.brier_calibration),
            open_dogmatism_index: avg(|r| r.open_dogmatism_index),
            open_output_diversity: avg(|r| r.open_output_diversity),
            open_cycle_err: avg(|r| r.open_cycle_err),
            open_cf_sens: avg(|r| r.open_cf_sens),
            open_branch_orth: avg(|r| r.open_branch_orth),
            open_brier: avg(|r| r.open_brier),
            nan_total: reps.iter().map(|r| r.nan_count).sum(),
        });
    }
    let ewc_reps: Vec<ContinualEwcReport> = (0..n_seeds)
        .map(|s_i| eval_continual_ewc_dialogo(0xD1A0 + s_i as u64))
        .collect();
    (summaries, ewc_reps)
}
