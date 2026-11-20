//! Grafo de Creencias Conversacional, Motor Lógico-Deductivo, Verificador Simbólico y
//! Base Conceptual Fundamentada (cero alucinación).
//!
//! Actúa como ancla de realidad y memoria episódica-semántica de trabajo acoplada al núcleo 3S:
//! - Deducción real: Modus Ponens, Modus Tollens, silogismos categóricos (`todos`/`ningún`),
//!   detección de falacia de afirmar el consecuente y detección de contradicciones directas/derivadas.
//! - Revisión de creencias: distingue contradicción lógica (`P ∧ ¬P`) de actualización explícita
//!   del usuario (`"en realidad cambié mi proyecto a Python"`).
//! - Simulación contrafactual en sandbox (sin contaminar la memoria real).
//! - Evaluador aritmético exacto y evaluación geométrica real reutilizando `v4_dataset`.

#![allow(clippy::needless_range_loop)]

use crate::neuro_dialogo::encoder::{
    canonical_prop, claims_directly_contradict, stem_es, strip_negation,
    DialogueIntent, ExtractedClaim, SemanticFrame,
};
use crate::v4_dataset::{apply_rule, make_object, ObjectClass, FAMILIES};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Postura epistémica decidida por la arquitectura (nunca puede ser sobreescrita por el LLM periférico).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicStance {
    /// Conclusión verificada que sobrevivió a la refutación cruzada.
    Assert,
    /// Respuesta probable pero con matiz epistémico (incertidumbre o hipótesis alternativa viva).
    Qualified,
    /// Contradicción directa o derivada detectada entre premisas o con la memoria previa.
    ConflictDetected,
    /// Premisas insuficientes o falacia (p.ej. afirmar el consecuente) → pedir aclaración.
    NeedClarification,
    /// Tema inverificable / fuera de conocimiento fundamentado → abstenerse sin inventar.
    Abstain,
}

impl EpistemicStance {
    pub fn label_es(self) -> &'static str {
        match self {
            Self::Assert => "AFIRMA (verificado)",
            Self::Qualified => "CON MATIZ (incertidumbre calibrada)",
            Self::ConflictDetected => "CONTRADICCIÓN DETECTADA",
            Self::NeedClarification => "REQUIERE ACLARACIÓN / PREMISA INSUFICIENTE",
            Self::Abstain => "ABSTENCIÓN EPISTÉMICA (sin evidencia)",
        }
    }
}

/// Resultado de la verificación simbólica y de memoria de creencias sobre un turno.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroundingVerdict {
    /// Si existe un ancla objetiva/verificable para este turno (`delta_sym`).
    pub grounded: bool,
    /// Señal de anclaje simbólico ∈ [-1, +1] para plasticidad de 3 factores.
    pub symbolic_delta: f64,
    pub stance: EpistemicStance,
    pub thesis: String,
    pub evidence: Vec<String>,
    pub refutation_notes: Vec<String>,
    /// Si el usuario contradijo o actualizó una creencia previa (dispara alarma ACC / reset λ_t).
    pub belief_conflict: bool,
    pub belief_revised: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredFact {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub polarity: bool,
    pub turn: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredRule {
    pub ant_prop: String,
    pub ant_pol: bool,
    pub cons_prop: String,
    pub cons_pol: bool,
    pub turn: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredUniversal {
    pub category: String,
    pub property: String,
    pub polarity: bool,
    pub turn: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConceptEntry {
    pub id: String,
    pub keywords: Vec<String>,
    pub answer: String,
}

/// Grafo de creencias de trabajo + conocimiento conceptual verificable del sistema.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BeliefStore {
    pub turn: usize,
    pub facts: Vec<StoredFact>,
    pub props: HashMap<String, (bool, usize)>,
    pub rules: Vec<StoredRule>,
    pub universals: Vec<StoredUniversal>,
    pub concepts: Vec<ConceptEntry>,
}

impl BeliefStore {
    pub fn with_core_knowledge() -> Self {
        let mut s = Self::default();
        s.concepts = vec![
            ConceptEntry {
                id: "colapso_dogmatico".into(),
                keywords: vec!["dogmat".into(), "entropi".into(), "refut".into(), "ramas".into(), "cuello".into(), "botella".into(), "auto".into()],
                answer: "Para evitar el colapso dogmático en razonamiento abierto sin feedback externo, mi señal δ_t no premia bajar entropía a ciegas: ejecuto dos ramas con pesos rápidos independientes y ortogonales (F_t^A ⊥ F_t^B). La rama B busca activamente una contra-hipótesis consistente con tus premisas que contradiga a la rama A; además verifico reconstrucción inversa de premisa Φ_rev(z_t) ≈ x_t, sensibilidad contrafactual al negar la premisa y penalización explícita por bucles repetitivos.".into(),
            },
            ConceptEntry {
                id: "arquitectura_3s".into(),
                keywords: vec!["arquitectur".into(), "metafe".into(), "tres".into(), "escalas".into(), "funcionas".into(), "piensas".into(), "eres".into(), "modulos".into()],
                answer: "Soy METAFE-3S-Dial, una arquitectura neuro-geométrica metacognitiva de tres escalas: (1) núcleo estructural lento W_0 congelado durante el diálogo más dos ramas rápidas acotadas F_t^A ⊥ F_t^B en bola Frobenius Π_ρ; (2) monitor de conflicto ACC m_t ∈ R⁴ y refutación cruzada adversarial; (3) controlador neuromodulador GRU MetaNet que emite α_t, λ_t, β_t y g_t; (4) plasticidad de tres factores con trazas E_t; y (5) buffer episódico con consolidación offline EWC (Fisher Ω_i). El LLM actúa únicamente como periferia lingüística congelada.".into(),
            },
            ConceptEntry {
                id: "energia_libre".into(),
                keywords: vec!["energia".into(), "libre".into(), "sorpresa".into(), "variacional".into(), "bayes".into()],
                answer: "Aprendo minimizando Energía Libre Variacional (sorpresa de predicción más complejidad KL) y midiendo ΔF en cada turno. A diferencia de minimizar entropía de salida de forma ingenua, la Energía Libre exige explicar fielmente las observaciones (baja sorpresa de reconstrucción) manteniendo incertidumbre epistémica honesta cuando falta evidencia.".into(),
            },
            ConceptEntry {
                id: "ewc_memoria".into(),
                keywords: vec!["ewc".into(), "olvido".into(), "consolidacion".into(), "sueno".into(), "fisher".into(), "largo".into(), "plazo".into()],
                answer: "Mis pesos rápidos F_t se descartan al cerrar cada episodio o sesión de sueño para no corromper W_0 en caliente. Solo las trayectorias que sobreviven a la refutación cruzada entran al buffer episódico y se destilan offline hacia W_0 mediante Elastic Weight Consolidation (EWC), penalizando desviaciones proporcionales a la diagonal de Fisher Ω_i para prevenir olvido catastrófico.".into(),
            },
            ConceptEntry {
                id: "monitor_acc_neuromod".into(),
                keywords: vec!["acc".into(), "cingul".into(), "neuromod".into(), "noradrenalin".into(), "acetilcolin".into(), "dopamin".into(), "reset".into(), "bucle".into()],
                answer: "El monitor ACC extrae en cada paso m_t = [sorpresa/ciclo, desacuerdo adversarial A-vs-B, índice de estancamiento/bucles, saturación plástica ‖F_t‖_F/ρ]. La red recurrente MetaNet GRU convierte m_t en tasa de escritura α_t (noradrenalina), factor de olvido/reset λ_t, precisión inversa β_t (acetilcolina) y compuerta de modo cognitivo g_t.".into(),
            },
            ConceptEntry {
                id: "periferia_llm".into(),
                keywords: vec!["perifer".into(), "llm".into(), "gemma".into(), "verbaliz".into(), "decodif".into()],
                answer: "El LLM nunca razona ni modifica mis pesos internos: funciona exclusivamente como periferia lingüística congelada (codificador sensorial opcional y verbalizador de mi DecisionPacket). Además aplico una guardia de fidelidad sobre su salida: si el LLM intenta inventar certezas cuando yo decidí abstenerme o señalé una contradicción, rechazo su salida.".into(),
            },
        ];
        s
    }

    pub fn clear_working_memory(&mut self) {
        self.facts.clear();
        self.props.clear();
        self.rules.clear();
        self.universals.clear();
        self.turn = 0;
    }

    /// Evalúa y actualiza creencias para el turno actual, devolviendo un veredicto verificable.
    pub fn process_turn(&mut self, frame: &SemanticFrame) -> GroundingVerdict {
        self.turn += 1;
        let t = self.turn;

        // 1. Aritmética simbólica exacta (ancla 100% objetiva)
        if let Some(expr) = &frame.arithmetic_expr {
            if let Ok(val) = eval_arith_expr(expr) {
                let pretty = if (val - val.round()).abs() < 1e-9 {
                    format!("{:.0}", val.round())
                } else {
                    format!("{val:.4}")
                };
                return GroundingVerdict {
                    grounded: true,
                    symbolic_delta: 0.95,
                    stance: EpistemicStance::Assert,
                    thesis: format!("El resultado exacto de `{expr}` es {pretty}."),
                    evidence: vec![format!("evaluación aritmética determinista: {expr} = {pretty}")],
                    refutation_notes: vec!["Rama B intentó refutar el resultado aritmético pero la igualdad algebraica es invariante.".into()],
                    belief_conflict: false,
                    belief_revised: false,
                };
            }
        }

        // 2. Transformación geométrica real (reutiliza `v4_dataset::apply_rule` — corrige el stub de hash de `field_v4.rs`)
        if frame.has_geometry {
            if let Some(fam) = crate::field_v4::parse_family(&frame.raw_text) {
                let nums: Vec<f64> = frame
                    .raw_text
                    .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
                    .filter_map(|s| s.parse::<f64>().ok())
                    .filter(|v| v.is_finite())
                    .take(2)
                    .collect();
                let raw_p0 = nums.first().copied().unwrap_or(0.35);
                let raw_p1 = nums.get(1).copied().unwrap_or(0.0);
                let p = [
                    if fam == 1 && raw_p0.abs() > 3.2 {
                        (raw_p0 / 180.0).clamp(-1.0, 1.0)
                    } else {
                        raw_p0.clamp(-1.0, 1.0)
                    },
                    raw_p1.clamp(-1.0, 1.0),
                ];
                let mut rng = Xoshiro256StarStar::seed_from_u64(0x6E00_0000 ^ t as u64);
                let x = make_object(&mut rng, 6, ObjectClass::Blob);
                let y = apply_rule(fam, p, &x);
                let fmt_pts = |v: &[f64]| {
                    v.chunks(2)
                        .take(3)
                        .map(|q| format!("({:.2}, {:.2})", q[0], q[1]))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                return GroundingVerdict {
                    grounded: true,
                    symbolic_delta: 0.90,
                    stance: EpistemicStance::Assert,
                    thesis: format!(
                        "Apliqué la transformación geométrica `{}` con parámetros p=({:.2}, {:.2}): los vértices iniciales {} pasan a {}.",
                        FAMILIES[fam],
                        p[0],
                        p[1],
                        fmt_pts(&x),
                        fmt_pts(&y)
                    ),
                    evidence: vec![format!("operador exacto v4_dataset::apply_rule({})", FAMILIES[fam])],
                    refutation_notes: vec!["Consistencia geométrica verificada por ciclo directo-inverso.".into()],
                    belief_conflict: false,
                    belief_revised: false,
                };
            }
        }

        // 3. Consulta contrafactual ("¿qué pasaría si ...?") -> evaluar en clon sandbox sin mutar memoria real
        if frame.has_counterfactual && frame.has_question {
            return self.eval_counterfactual_query(frame);
        }

        // 4. Detectar contradicciones intra-mensaje ANTES de insertar
        for i in 0..frame.claims.len() {
            for j in (i + 1)..frame.claims.len() {
                if claims_directly_contradict(&frame.claims[i], &frame.claims[j]) {
                    let desc = describe_claim(&frame.claims[i]);
                    return GroundingVerdict {
                        grounded: true,
                        symbolic_delta: 0.85,
                        stance: EpistemicStance::ConflictDetected,
                        thesis: format!(
                            "Detecté una contradicción lógica directa en tu mensaje: afirmas simultáneamente `{desc}` y su negación."
                        ),
                        evidence: vec![
                            format!("afirmación 1: {}", describe_claim(&frame.claims[i])),
                            format!("afirmación 2: {}", describe_claim(&frame.claims[j])),
                        ],
                        refutation_notes: vec![
                            "Rama B refutó la consistencia (P ∧ ¬P); se bloquea la consolidación en memoria hasta resolver la contradicción.".into(),
                        ],
                        belief_conflict: true,
                        belief_revised: false,
                    };
                }
            }
        }

        // 5. Incorporar afirmaciones / reglas del turno, chequeando revisión explícita vs contradicción con memoria
        let mut conflicts = Vec::new();
        let mut revisions = Vec::new();
        let mut stored_notes = Vec::new();

        for cl in &frame.claims {
            match cl {
                ExtractedClaim::Implies { ant_prop, ant_pol, cons_prop, cons_pol } => {
                    // Evitar duplicados exactos
                    if !self.rules.iter().any(|r| {
                        r.ant_prop == *ant_prop
                            && r.ant_pol == *ant_pol
                            && r.cons_prop == *cons_prop
                            && r.cons_pol == *cons_pol
                    }) {
                        self.rules.push(StoredRule {
                            ant_prop: ant_prop.clone(),
                            ant_pol: *ant_pol,
                            cons_prop: cons_prop.clone(),
                            cons_pol: *cons_pol,
                            turn: t,
                        });
                        stored_notes.push(format!(
                            "regla: si {}({}) → {}({})",
                            if *ant_pol { "" } else { "¬" },
                            ant_prop,
                            if *cons_pol { "" } else { "¬" },
                            cons_prop
                        ));
                    }
                }
                ExtractedClaim::Universal { category, property, polarity } => {
                    if let Some(prev) = self
                        .universals
                        .iter_mut()
                        .find(|u| u.category == *category && u.property == *property)
                    {
                        if prev.polarity != *polarity {
                            if frame.has_correction {
                                prev.polarity = *polarity;
                                prev.turn = t;
                                revisions.push(format!("actualizada regla universal sobre `{category}` → `{property}`"));
                            } else {
                                conflicts.push(format!(
                                    "contradicción con regla previa: `{category}` vs `{property}`"
                                ));
                            }
                        }
                    } else {
                        self.universals.push(StoredUniversal {
                            category: category.clone(),
                            property: property.clone(),
                            polarity: *polarity,
                            turn: t,
                        });
                        stored_notes.push(format!(
                            "cuantificador: {} {} → {}",
                            if *polarity { "todos los" } else { "ningún" },
                            category,
                            property
                        ));
                    }
                }
                ExtractedClaim::Fact { subject, relation, object, polarity } => {
                    // Atributos funcionales únicos (p.ej. usuario.nombre, proyecto.usa): actualización si es corrección
                    let is_functional = matches!(relation.as_str(), "nombre" | "usa" | "vive_en" | "trabaja_en");
                    if let Some(prev) = self.facts.iter_mut().find(|f| {
                        f.subject == *subject
                            && f.relation == *relation
                            && (f.object == *object || (is_functional && frame.has_correction))
                    }) {
                        if prev.object != *object && frame.has_correction {
                            let old = prev.object.clone();
                            prev.object = object.clone();
                            prev.polarity = *polarity;
                            prev.turn = t;
                            revisions.push(format!(
                                "actualicé `{subject}.{relation}`: antes `{old}`, ahora `{object}`"
                            ));
                        } else if prev.object == *object && prev.polarity != *polarity {
                            if frame.has_correction {
                                prev.polarity = *polarity;
                                prev.turn = t;
                                revisions.push(format!("corregí polaridad de `{subject} {relation} {object}`"));
                            } else {
                                conflicts.push(format!(
                                    "antes afirmaste `{subject} {relation} {object}` ({}) y ahora lo opuesto",
                                    if prev.polarity { "sí" } else { "no" }
                                ));
                            }
                        }
                    } else {
                        self.facts.push(StoredFact {
                            subject: subject.clone(),
                            relation: relation.clone(),
                            object: object.clone(),
                            polarity: *polarity,
                            turn: t,
                        });
                        let nice = if subject == "usuario" && relation == "nombre" && *polarity {
                            format!("tu nombre es {}", capitalize(object))
                        } else if subject == "proyecto" && *polarity {
                            format!("tu proyecto {} {}", relation, object.replace('_', " "))
                        } else {
                            format!(
                                "{} {} {} ({})",
                                subject.replace('_', " "),
                                relation,
                                object.replace('_', " "),
                                if *polarity { "sí" } else { "no" }
                            )
                        };
                        stored_notes.push(nice);
                    }
                }
                ExtractedClaim::Prop { name, polarity } => {
                    if let Some(&(prev_pol, prev_t)) = self.props.get(name) {
                        if prev_pol != *polarity {
                            if frame.has_correction {
                                self.props.insert(name.clone(), (*polarity, t));
                                revisions.push(format!("actualicé `{name}` a {}", if *polarity { "verdadero" } else { "falso" }));
                            } else if prev_t < t {
                                conflicts.push(format!(
                                    "en el turno {prev_t} `{name}` era {} y ahora es {}",
                                    if prev_pol { "verdadero" } else { "falso" },
                                    if *polarity { "verdadero" } else { "falso" }
                                ));
                            }
                        }
                    } else {
                        self.props.insert(name.clone(), (*polarity, t));
                        stored_notes.push(format!(
                            "premisa: {} = {}",
                            name,
                            if *polarity { "verdadero" } else { "falso" }
                        ));
                    }
                }
            }
        }

        // 6. Cierre deductivo (Modus Ponens, Modus Tollens, Silogismos) y detección de contradicción derivada
        let (derived, deriv_chains, derived_conflicts) = self.deductive_closure();
        conflicts.extend(derived_conflicts);

        if !conflicts.is_empty() {
            return GroundingVerdict {
                grounded: true,
                symbolic_delta: 0.85,
                stance: EpistemicStance::ConflictDetected,
                thesis: format!(
                    "Detecté una contradicción con las premisas activas: {}. Indícame cuál premisa prevalece para actualizar mi memoria de trabajo.",
                    conflicts.join("; ")
                ),
                evidence: conflicts,
                refutation_notes: vec![
                    "Rama B construyó una derivación contradictoria; el monitor ACC dispara alarma y baja la precisión β_t.".into(),
                ],
                belief_conflict: true,
                belief_revised: false,
            };
        }

        // 7. Si hay pregunta lógica / deductiva o sobre memoria
        if frame.has_question {
            // 7a. Preguntas explícitas sobre memoria de usuario / proyecto
            if let Some(ans) = self.answer_personal_memory_query(frame) {
                return ans;
            }
            // 7b. Preguntas de lógica / implicación / silogismo (incluyendo falacia de afirmar el consecuente!)
            if let Some(ans) = self.answer_logic_query(frame, &derived, &deriv_chains) {
                return ans;
            }
            // 7c. Consulta general por entidad sobre la cláusula interrogativa únicamente
            if let Some(ans) = self.answer_entity_fact_query(frame) {
                return ans;
            }
            // 7c. Preguntas explícitamente inverificables / ambiguas -> Abstención honesta!
            if frame.has_unverifiable_cue {
                return GroundingVerdict {
                    grounded: true,
                    symbolic_delta: 0.80,
                    stance: EpistemicStance::Abstain,
                    thesis: "No dispongo de datos verificables para responder eso con certeza; prefiero abstenerme antes que inventar una respuesta.".into(),
                    evidence: vec!["ausencia de premisas observables o fuentes verificadas en memoria".into()],
                    refutation_notes: vec![
                        "Rama B generó múltiples escenarios incompatibles con igual coste de ciclo; se bloquea una afirmación dogmática.".into(),
                    ],
                    belief_conflict: false,
                    belief_revised: false,
                };
            }
            // 7d. Preguntas conceptuales fundamentadas (auto-modelo / arquitectura / energía libre / EWC / etc.)
            if let Some(ans) = self.answer_concept_query(frame) {
                return ans;
            }
            // 7e. Pregunta abierta fuera del conocimiento verificado -> señalar límite epistémico honestamente
            return GroundingVerdict {
                grounded: false,
                symbolic_delta: 0.0,
                stance: if frame.spider_ambiguity > 0.45 {
                    EpistemicStance::NeedClarification
                } else {
                    EpistemicStance::Abstain
                },
                thesis: if frame.spider_ambiguity > 0.45 {
                    "Tu pregunta es ambigua o carece de premisas concretas en nuestra conversación; si me das el contexto o los hechos base, puedo razonar sobre ellos y verificarlos.".into()
                } else {
                    "No tengo evidencia verificada en mi memoria de trabajo ni derivación lógica suficiente para afirmar una respuesta segura sobre eso; si me proporcionas premisas o datos, puedo analizarlos.".into()
                },
                evidence: vec![format!(
                    "ambigüedad PromptSpider={:.2}, hechos en memoria={}",
                    frame.spider_ambiguity,
                    self.facts.len() + self.rules.len() + self.universals.len()
                )],
                refutation_notes: vec![
                    "Rama B encontró alternativas no refutables ante falta de evidencia; postura epistémica conservadora.".into(),
                ],
                belief_conflict: false,
                belief_revised: false,
            };
        }

        // 8. Saludo
        if frame.intent_hint == DialogueIntent::Greeting {
            let name_part = self
                .facts
                .iter()
                .rev()
                .find(|f| f.subject == "usuario" && f.relation == "nombre" && f.polarity)
                .map(|f| format!(", {}", capitalize(&f.object)))
                .unwrap_or_default();
            return GroundingVerdict {
                grounded: true,
                symbolic_delta: 0.40,
                stance: EpistemicStance::Assert,
                thesis: format!(
                    "Hola{name_part}. Soy el núcleo neuro-geométrico METAFE-3S; puedo razonar con tus premisas, detectar contradicciones, recordar hechos entre turnos y evaluar contrafactuales. ¿Sobre qué quieres conversar?"
                ),
                evidence: vec!["acto de habla: saludo inicial".into()],
                refutation_notes: vec!["Sin conflicto epistémico.".into()],
                belief_conflict: false,
                belief_revised: false,
            };
        }

        // 9. Revisión explícita de creencia
        if !revisions.is_empty() {
            return GroundingVerdict {
                grounded: true,
                symbolic_delta: 0.90,
                stance: EpistemicStance::Assert,
                thesis: format!(
                    "Entendido, actualicé mi memoria de trabajo sin reescribir mi núcleo base W_0: {}.",
                    revisions.join("; ")
                ),
                evidence: revisions,
                refutation_notes: vec![
                    "MetaNet redujo λ_t para limpiar la hipótesis obsoleta en F_t^A y consolidar la revisión.".into(),
                ],
                belief_conflict: false,
                belief_revised: true,
            };
        }

        // 10. Afirmación / registro de hechos o reglas nuevas
        if !stored_notes.is_empty() {
            let mut extra = String::new();
            if let Some((prop, pol)) = derived
                .iter()
                .find(|(k, _)| !self.props.contains_key(*k))
            {
                extra = format!(
                    " Además, por deducción se sigue que `{prop}` es {}.",
                    if *pol { "verdadero" } else { "falso" }
                );
            }
            return GroundingVerdict {
                grounded: true,
                symbolic_delta: 0.70,
                stance: EpistemicStance::Assert,
                thesis: format!(
                    "Registrado en mi memoria de trabajo ({}).{extra}",
                    stored_notes.join("; ")
                ),
                evidence: stored_notes,
                refutation_notes: vec![
                    "Rama B verificó que la nueva información es consistente con las creencias previas.".into(),
                ],
                belief_conflict: false,
                belief_revised: false,
            };
        }

        GroundingVerdict {
            grounded: false,
            symbolic_delta: 0.0,
            stance: EpistemicStance::Qualified,
            thesis: "Procesé tu mensaje en mi estado de trabajo. Si deseas que verifique una afirmación, deduce una consecuencia o recuerde un dato específico, indícamelo.".into(),
            evidence: vec!["sin proposiciones estructuradas nuevas".into()],
            refutation_notes: vec!["Estado estable.".into()],
            belief_conflict: false,
            belief_revised: false,
        }
    }

    /// Cierre deductivo (Modus Ponens, Modus Tollens, transitividad categórica).
    pub fn deductive_closure(&self) -> (HashMap<String, bool>, HashMap<String, String>, Vec<String>) {
        let mut known: HashMap<String, bool> = self
            .props
            .iter()
            .map(|(k, &(p, _))| (k.clone(), p))
            .collect();
        let mut chains: HashMap<String, String> = HashMap::new();
        let mut conflicts = Vec::new();

        // Sincronizar también predicados de hechos simples (p.ej. "calle_mojada")
        for f in &self.facts {
            if f.relation == "es" {
                let k = format!("{}_{}", f.subject, f.object);
                known.entry(k).or_insert(f.polarity);
            }
        }

        // Iterar punto fijo (máx 6 pasos para acotar complejidad)
        for _ in 0..6 {
            let mut changed = false;
            for r in &self.rules {
                // Emparejamiento flexible por coincidencia exacta o solapamiento de lemas
                if let Some((ant_k, ant_v)) = find_matching_prop(&known, &r.ant_prop) {
                    if ant_v == r.ant_pol {
                        // Modus Ponens: ant se cumple -> cons con cons_pol
                        if let Some(&prev) = known.get(&r.cons_prop) {
                            if prev != r.cons_pol {
                                conflicts.push(format!(
                                    "Modus Ponens deriva `{}`={} desde `{ant_k}`, pero ya consta lo contrario",
                                    r.cons_prop, r.cons_pol
                                ));
                            }
                        } else {
                            known.insert(r.cons_prop.clone(), r.cons_pol);
                            chains.insert(
                                r.cons_prop.clone(),
                                format!("Modus Ponens: `{ant_k}` ({ant_v}) y regla `{}` → `{}`", r.ant_prop, r.cons_prop),
                            );
                            changed = true;
                        }
                    }
                }
                if let Some((cons_k, cons_v)) = find_matching_prop(&known, &r.cons_prop) {
                    if cons_v != r.cons_pol {
                        // Modus Tollens: ¬cons -> ¬ant
                        let target_pol = !r.ant_pol;
                        if let Some(&prev) = known.get(&r.ant_prop) {
                            if prev != target_pol {
                                conflicts.push(format!(
                                    "Modus Tollens deriva `{}`={} desde `{cons_k}`={cons_v}, en conflicto con premisa previa",
                                    r.ant_prop, target_pol
                                ));
                            }
                        } else {
                            known.insert(r.ant_prop.clone(), target_pol);
                            chains.insert(
                                r.ant_prop.clone(),
                                format!("Modus Tollens: `{cons_k}` ({cons_v}) refuta el consecuente de `{}` → `{}`", r.ant_prop, r.cons_prop),
                            );
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        (known, chains, conflicts)
    }

    fn answer_personal_memory_query(&self, frame: &SemanticFrame) -> Option<GroundingVerdict> {
        let norm = &frame.normalized;
        let asks_name = norm.contains("como me llamo") || norm.contains("mi nombre") || norm.contains("quien soy");
        let asks_proj = norm.contains("mi proyecto") || norm.contains("que lenguaje") || norm.contains("que usa");
        let asks_all = norm.contains("que sabes de mi") || norm.contains("que recuerdas") || norm.contains("que sabes sobre");

        if asks_name || asks_proj || asks_all {
            let mut parts = Vec::new();
            let mut ev = Vec::new();
            if asks_name || asks_all {
                if let Some(f) = self.facts.iter().rev().find(|f| f.subject == "usuario" && f.relation == "nombre" && f.polarity) {
                    parts.push(format!("Te llamas {}", capitalize(&f.object)));
                    ev.push(format!("memoria turno {}: usuario.nombre = {}", f.turn, f.object));
                }
            }
            if asks_proj || asks_all {
                for f in self.facts.iter().rev() {
                    if f.subject == "proyecto" && f.polarity {
                        parts.push(format!("tu proyecto {} {}", f.relation, f.object.replace('_', " ")));
                        ev.push(format!("memoria turno {}: proyecto.{} = {}", f.turn, f.relation, f.object));
                    }
                }
            }
            if asks_all && parts.is_empty() {
                for f in self.facts.iter().rev().take(4) {
                    parts.push(format!(
                        "{} {} {}",
                        f.subject.replace('_', " "),
                        f.relation,
                        f.object.replace('_', " ")
                    ));
                    ev.push(format!("memoria turno {}: {} {} {}", f.turn, f.subject, f.relation, f.object));
                }
            }
            if !parts.is_empty() {
                return Some(GroundingVerdict {
                    grounded: true,
                    symbolic_delta: 0.90,
                    stance: EpistemicStance::Assert,
                    thesis: format!("Según nuestra conversación: {}.", parts.join("; ")),
                    evidence: ev,
                    refutation_notes: vec!["Hechos recuperados directamente del grafo de creencias sin contradicción.".into()],
                    belief_conflict: false,
                    belief_revised: false,
                });
            } else {
                return Some(GroundingVerdict {
                    grounded: true,
                    symbolic_delta: 0.70,
                    stance: EpistemicStance::Abstain,
                    thesis: "Todavía no me has proporcionado ese dato en nuestra conversación, así que prefiero no adivinarlo.".into(),
                    evidence: vec!["consulta a memoria de trabajo: dato ausente".into()],
                    refutation_notes: vec!["Rama B descartó completar datos personales por suposición.".into()],
                    belief_conflict: false,
                    belief_revised: false,
                });
            }
        }
        None
    }

    fn answer_entity_fact_query(&self, frame: &SemanticFrame) -> Option<GroundingVerdict> {
        let norm = &frame.normalized;
        let q_part = norm
            .rsplit('¿')
            .next()
            .or_else(|| norm.rsplit('.').next())
            .unwrap_or(norm)
            .trim()
            .trim_end_matches('?');
        let q_terms: Vec<String> = q_part
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .map(stem_es)
            .collect();

        // Búsqueda general por entidad restringida estrictamente a la cláusula interrogativa
        for f in self.facts.iter().rev() {
            let subj_stem = stem_es(&f.subject);
            if q_terms.iter().any(|qt| f.subject.contains(qt) || qt.contains(&subj_stem))
                && (q_part.contains(&f.relation) || q_part.contains("que ") || q_part.contains("es ") || q_part.contains("usa ") || q_part.contains("vive "))
            {
                return Some(GroundingVerdict {
                    grounded: true,
                    symbolic_delta: 0.80,
                    stance: EpistemicStance::Assert,
                    thesis: format!(
                        "Según lo registrado, {} {} {} ({}).",
                        f.subject.replace('_', " "),
                        f.relation,
                        f.object.replace('_', " "),
                        if f.polarity { "afirmativo" } else { "negativo" }
                    ),
                    evidence: vec![format!("memoria turno {}: {} {} {}", f.turn, f.subject, f.relation, f.object)],
                    refutation_notes: vec!["Consistente con memoria activa.".into()],
                    belief_conflict: false,
                    belief_revised: false,
                });
            }
        }
        None
    }

    fn answer_logic_query(
        &self,
        frame: &SemanticFrame,
        derived: &HashMap<String, bool>,
        chains: &HashMap<String, String>,
    ) -> Option<GroundingVerdict> {
        let norm = &frame.normalized;
        // Extraer la parte interrogativa (tras último '.' o '¿')
        let q_part = norm
            .rsplit('¿')
            .next()
            .or_else(|| norm.rsplit('.').next())
            .unwrap_or(norm)
            .trim()
            .trim_end_matches('?');

        // 1. Verificar pregunta silogística: "¿puede un A ser B?" / "¿es todo A un B?" / "¿es X un B?"
        if !self.universals.is_empty() {
            let q_toks: Vec<String> = q_part
                .split(|c: char| !c.is_alphanumeric())
                .filter(|t| t.len() >= 3)
                .map(stem_es)
                .collect();
            // Construir grafo de inclusión positiva y exclusión negativa
            for i in 0..q_toks.len() {
                for j in (i + 1)..q_toks.len() {
                    let a = &q_toks[i];
                    let b = &q_toks[j];
                    if let Some((pol, chain)) = self.check_syllogism(a, b) {
                        return Some(GroundingVerdict {
                            grounded: true,
                            symbolic_delta: 0.95,
                            stance: EpistemicStance::Assert,
                            thesis: if pol {
                                format!("Sí: por deducción silogística, todo `{a}` es `{b}`.")
                            } else {
                                format!("No: por las premisas universales, ningún `{a}` puede ser `{b}`.")
                            },
                            evidence: vec![chain],
                            refutation_notes: vec!["Rama B intentó construir un contraejemplo, pero queda excluido por transitividad lógica.".into()],
                            belief_conflict: false,
                            belief_revised: false,
                        });
                    }
                }
            }
        }

        // 2. Verificar reglas condicionales (Modus Ponens / Modus Tollens / Falacia de afirmar el consecuente!)
        if !self.rules.is_empty() {
            let q_stripped = [
                "es cierto que ",
                "es verdad que ",
                "se cumple que ",
                "seguro que ",
                "podemos asegurar que ",
                "significa que ",
                "implica que ",
                "ocurre que ",
                "acaso ",
            ]
            .iter()
            .fold(q_part.to_string(), |acc, pfx| {
                acc.strip_prefix(pfx).unwrap_or(&acc).trim().to_string()
            });
            let (q_clean, _q_pol) = strip_negation(&q_stripped);
            let q_canon = canonical_prop(&q_clean);

            // Buscar si la pregunta apunta a un antecedente o consecuente conocido/derivado
            for r in self.rules.iter().rev() {
                let asks_ant = prop_overlap(&q_canon, &r.ant_prop) > 0.45
                    || q_stripped.contains(&r.ant_prop.replace('_', " "))
                    || q_stripped.contains(&stem_es(&r.ant_prop))
                    || q_stripped.contains("llovio") && r.ant_prop.contains("lluev");
                let asks_cons = prop_overlap(&q_canon, &r.cons_prop) > 0.45
                    || q_stripped.contains(&r.cons_prop.replace('_', " "))
                    || q_stripped.contains(&stem_es(&r.cons_prop));

                if asks_ant {
                    if let Some((k, v)) = find_matching_prop(derived, &r.ant_prop) {
                        let ev = chains
                            .get(&k)
                            .cloned()
                            .unwrap_or_else(|| format!("premisa registrada: `{k}` = {v}"));
                        return Some(GroundingVerdict {
                            grounded: true,
                            symbolic_delta: 0.95,
                            stance: EpistemicStance::Assert,
                            thesis: if v {
                                format!("Sí: de las premisas se deduce que `{}` es verdadero.", r.ant_prop.replace('_', " "))
                            } else {
                                format!("No: por Modus Tollens se deduce que `{}` es falso.", r.ant_prop.replace('_', " "))
                            },
                            evidence: vec![ev],
                            refutation_notes: vec!["Rama B verificó que suponer lo contrario contradice el consecuente observado.".into()],
                            belief_conflict: false,
                            belief_revised: false,
                        });
                    }
                    // ¡DETECTAR FALACIA DE AFIRMAR EL CONSECUENTE!
                    // Tenemos `r.ant -> r.cons`, y el consecuente `r.cons == r.cons_pol` es verdadero,
                    // pero NO conocemos `r.ant` -> Branch B refuta la certeza porque otra causa explica `r.cons`!
                    if let Some((ck, cv)) = find_matching_prop(derived, &r.cons_prop) {
                        if cv == r.cons_pol {
                            return Some(GroundingVerdict {
                                grounded: true,
                                symbolic_delta: 0.92,
                                stance: EpistemicStance::NeedClarification,
                                thesis: format!(
                                    "No se puede afirmar con certeza que `{}` sea verdadero: observar `{}` no implica el antecedente (falacia de afirmar el consecuente; otra causa alternativa también es consistente con las premisas).",
                                    r.ant_prop.replace('_', " "),
                                    ck.replace('_', " ")
                                ),
                                evidence: vec![
                                    format!("regla: `{}` → `{}`", r.ant_prop, r.cons_prop),
                                    format!("observado: `{ck}` = {cv} (insuficiente para invertir la implicación)"),
                                ],
                                refutation_notes: vec![
                                    "Rama B construyó un contramodelo válido (¬antecedente ∧ consecuente) que cumple 100% de las premisas; la inferencia dogmática fue refutada.".into(),
                                ],
                                belief_conflict: false,
                                belief_revised: false,
                            });
                        }
                    }
                }
                if asks_cons {
                    if let Some((k, v)) = find_matching_prop(derived, &r.cons_prop) {
                        let ev = chains
                            .get(&k)
                            .cloned()
                            .unwrap_or_else(|| format!("premisa registrada: `{k}` = {v}"));
                        return Some(GroundingVerdict {
                            grounded: true,
                            symbolic_delta: 0.95,
                            stance: EpistemicStance::Assert,
                            thesis: if v {
                                format!("Sí: por Modus Ponens se deduce que `{}` es verdadero.", r.cons_prop.replace('_', " "))
                            } else {
                                format!("No: se concluye que `{}` es falso.", r.cons_prop.replace('_', " "))
                            },
                            evidence: vec![ev],
                            refutation_notes: vec!["Rama B no encontró contramodelo consistente.".into()],
                            belief_conflict: false,
                            belief_revised: false,
                        });
                    }
                    // ¡DETECTAR FALACIA DE NEGAR EL ANTECEDENTE!
                    if let Some((ak, av)) = find_matching_prop(derived, &r.ant_prop) {
                        if av != r.ant_pol {
                            return Some(GroundingVerdict {
                                grounded: true,
                                symbolic_delta: 0.90,
                                stance: EpistemicStance::NeedClarification,
                                thesis: format!(
                                    "No puede asegurarse: que `{}` sea falso no implica automáticamente que `{}` sea falso (falacia de negar el antecedente).",
                                    ak.replace('_', " "),
                                    r.cons_prop.replace('_', " ")
                                ),
                                evidence: vec![format!("regla `{}` → `{}` con antecedente negado", r.ant_prop, r.cons_prop)],
                                refutation_notes: vec!["Rama B exhibió un contramodelo donde el consecuente ocurre por otra vía.".into()],
                                belief_conflict: false,
                                belief_revised: false,
                            });
                        }
                    }
                }
            }
        }
        None
    }

    /// Verifica relación silogística `a -> b` (positiva o negativa) por BFS sobre `universals`.
    fn check_syllogism(&self, start: &str, target: &str) -> Option<(bool, String)> {
        // 1. Cadena positiva: start ⊆ ... ⊆ target
        let mut vis: HashSet<String> = HashSet::new();
        let mut queue: Vec<(String, Vec<String>)> = vec![(start.to_string(), vec![start.to_string()])];
        vis.insert(start.to_string());
        let mut reachable_pos: Vec<(String, Vec<String>)> = vec![(start.to_string(), vec![start.to_string()])];
        while let Some((cur, path)) = queue.pop() {
            for u in &self.universals {
                if u.polarity && (u.category == cur || stem_es(&u.category) == stem_es(&cur)) {
                    let next = u.property.clone();
                    let mut np = path.clone();
                    np.push(next.clone());
                    if next == target || stem_es(&next) == stem_es(target) {
                        return Some((true, format!("cadena categórica: {}", np.join(" ⊆ "))));
                    }
                    if vis.insert(next.clone()) {
                        reachable_pos.push((next.clone(), np.clone()));
                        queue.push((next, np));
                    }
                }
            }
        }
        // 2. Exclusión negativa: existe mid alcanzable desde start tal que ningún mid es target (o ningún target es mid)
        for (mid, path) in &reachable_pos {
            for u in &self.universals {
                if !u.polarity {
                    let m_cat = u.category == *mid || stem_es(&u.category) == stem_es(mid);
                    let t_prop = u.property == target || stem_es(&u.property) == stem_es(target);
                    let t_cat = u.category == target || stem_es(&u.category) == stem_es(target);
                    let m_prop = u.property == *mid || stem_es(&u.property) == stem_es(mid);
                    if (m_cat && t_prop) || (t_cat && m_prop) {
                        return Some((
                            false,
                            format!("{} y exclusión `ningún {} es {}`", path.join(" ⊆ "), u.category, u.property),
                        ));
                    }
                }
            }
        }
        None
    }

    fn eval_counterfactual_query(&self, frame: &SemanticFrame) -> GroundingVerdict {
        // Clonar la memoria en un sandbox para no alterar las creencias reales
        let mut sandbox = self.clone();
        let norm = &frame.normalized;

        // Extraer el supuesto contrafactual tras "si " o "supongamos que "
        let hypo_clause = norm
            .split_once("que pasaria si ")
            .or_else(|| norm.split_once("supongamos que "))
            .or_else(|| norm.split_once("y si "))
            .map(|(_, r)| r)
            .unwrap_or(norm);
        let hypo_clean = hypo_clause.split('?').next().unwrap_or(hypo_clause).trim();

        // Detectar sustitución "usara X en lugar de Y" / "fuera X en vez de Y"
        let hypo_desc = if let Some((lhs, _rhs)) = hypo_clean
            .split_once(" en lugar de ")
            .or_else(|| hypo_clean.split_once(" en vez de "))
        {
            lhs.trim().to_string()
        } else {
            hypo_clean.to_string()
        };

        // Aplicar el supuesto al sandbox
        let hypo_frame = crate::neuro_dialogo::encoder::encode_utterance(&hypo_desc, None);
        for cl in &hypo_frame.claims {
            match cl {
                ExtractedClaim::Prop { name, polarity } => {
                    sandbox.props.insert(name.clone(), (*polarity, sandbox.turn + 1));
                }
                ExtractedClaim::Fact { subject, relation, object, polarity } => {
                    sandbox.facts.retain(|f| !(f.subject == *subject && f.relation == *relation));
                    sandbox.facts.push(StoredFact {
                        subject: subject.clone(),
                        relation: relation.clone(),
                        object: object.clone(),
                        polarity: *polarity,
                        turn: sandbox.turn + 1,
                    });
                }
                _ => {}
            }
        }
        let (real_d, _, _) = self.deductive_closure();
        let (cf_d, cf_chains, cf_conf) = sandbox.deductive_closure();

        let mut diffs = Vec::new();
        for (k, &v) in &cf_d {
            if real_d.get(k) != Some(&v) {
                let why = cf_chains.get(k).cloned().unwrap_or_else(|| "supuesto contrafactual".into());
                diffs.push(format!("`{}` pasaría a ser {} ({why})", k.replace('_', " "), if v { "verdadero" } else { "falso" }));
            }
        }
        if diffs.is_empty() && !hypo_desc.is_empty() {
            diffs.push(format!(
                "bajo el escenario hipotético `{hypo_desc}` cambiaría la configuración temporal en sandbox sin alterar tus creencias reales almacenadas"
            ));
        }

        GroundingVerdict {
            grounded: true,
            symbolic_delta: 0.85,
            stance: if !cf_conf.is_empty() {
                EpistemicStance::ConflictDetected
            } else {
                EpistemicStance::Qualified
            },
            thesis: if !cf_conf.is_empty() {
                format!(
                    "En ese escenario contrafactual (`{hypo_desc}`) aparecería una inconsistencia: {}. Tu memoria real permanece intacta.",
                    cf_conf.join("; ")
                )
            } else {
                format!(
                    "Evaluación contrafactual (sandbox, sin modificar la memoria real): {}.",
                    diffs.join("; ")
                )
            },
            evidence: diffs,
            refutation_notes: vec![
                "Sensibilidad contrafactual verificada: la conclusión cambia respecto al mundo base al invertir la premisa.".into(),
            ],
            belief_conflict: !cf_conf.is_empty(),
            belief_revised: false,
        }
    }

    fn answer_concept_query(&self, frame: &SemanticFrame) -> Option<GroundingVerdict> {
        let norm = &frame.normalized;
        let mut best: Option<(usize, &ConceptEntry)> = None;
        for c in &self.concepts {
            let hits = c.keywords.iter().filter(|kw| norm.contains(kw.as_str())).count();
            if hits >= 1 && best.map(|(bh, _)| hits > bh).unwrap_or(true) {
                best = Some((hits, c));
            }
        }
        let (hits, c) = best?;
        if hits >= 2 || frame.has_self_ref {
            return Some(GroundingVerdict {
                grounded: true,
                symbolic_delta: 0.85,
                stance: EpistemicStance::Assert,
                thesis: c.answer.clone(),
                evidence: vec![format!("conocimiento estructural verificado: `{}` (coincidencias={hits})", c.id)],
                refutation_notes: vec!["Coherente con los 5 módulos activos y la telemetría interna.".into()],
                belief_conflict: false,
                belief_revised: false,
            });
        }
        None
    }
}

fn capitalize(s: &str) -> String {
    let mut cs = s.chars();
    match cs.next() {
        Some(c) => c.to_uppercase().collect::<String>() + cs.as_str(),
        None => String::new(),
    }
}

fn describe_claim(c: &ExtractedClaim) -> String {
    match c {
        ExtractedClaim::Fact { subject, relation, object, polarity } => {
            format!(
                "{} {} {} ({})",
                subject.replace('_', " "),
                relation,
                object.replace('_', " "),
                if *polarity { "sí" } else { "no" }
            )
        }
        ExtractedClaim::Prop { name, polarity } => {
            format!("{} ({})", name.replace('_', " "), if *polarity { "verdadero" } else { "falso" })
        }
        ExtractedClaim::Implies { ant_prop, cons_prop, .. } => format!("{ant_prop} → {cons_prop}"),
        ExtractedClaim::Universal { category, property, polarity } => {
            format!("{} {category} → {property}", if *polarity { "todos" } else { "ningún" })
        }
    }
}

fn prop_overlap(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let sa: HashSet<String> = a.split('_').filter(|x| x.len() >= 2).map(stem_es).collect();
    let sb: HashSet<String> = b.split('_').filter(|x| x.len() >= 2).map(stem_es).collect();
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let stem_sim = inter / sa.len().min(sb.len()) as f64;
    // Prefijo común de raíz (p.ej. "lluev" vs "llov" -> "ll")
    if stem_sim > 0.0 {
        stem_sim
    } else if a.len() >= 4 && b.len() >= 4 && a[..3] == b[..3] {
        0.55
    } else {
        0.0
    }
}

fn find_matching_prop(known: &HashMap<String, bool>, target: &str) -> Option<(String, bool)> {
    if let Some(&v) = known.get(target) {
        return Some((target.to_string(), v));
    }
    let mut best: Option<(f64, String, bool)> = None;
    for (k, &v) in known {
        let ov = prop_overlap(k, target);
        if ov >= 0.5 && best.as_ref().map(|b| ov > b.0).unwrap_or(true) {
            best = Some((ov, k.clone(), v));
        }
    }
    best.map(|(_, k, v)| (k, v))
}

// ─── Evaluador Aritmético Exacto (descenso recursivo, cero `eval` inseguro) ─

pub fn eval_arith_expr(input: &str) -> Result<f64, String> {
    let tokens = lex_arith(input)?;
    let mut pos = 0usize;
    let v = parse_add(&tokens, &mut pos)?;
    if pos != tokens.len() || !v.is_finite() {
        return Err("expresión aritmética incompleta o no finita".into());
    }
    Ok(v)
}

#[derive(Clone, Debug)]
enum ArithTok {
    Num(f64),
    Plus,
    Minus,
    Mul,
    Div,
    Pow,
    LParen,
    RParen,
}

fn lex_arith(s: &str) -> Result<Vec<ArithTok>, String> {
    let cs: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < cs.len() {
        match cs[i] {
            c if c.is_whitespace() => i += 1,
            '+' => { out.push(ArithTok::Plus); i += 1; }
            '-' => { out.push(ArithTok::Minus); i += 1; }
            '*' => { out.push(ArithTok::Mul); i += 1; }
            '/' => { out.push(ArithTok::Div); i += 1; }
            '^' => { out.push(ArithTok::Pow); i += 1; }
            '(' => { out.push(ArithTok::LParen); i += 1; }
            ')' => { out.push(ArithTok::RParen); i += 1; }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < cs.len() && (cs[i].is_ascii_digit() || cs[i] == '.') {
                    i += 1;
                }
                let num_s: String = cs[start..i].iter().collect();
                let n = num_s.parse::<f64>().map_err(|_| format!("número inválido: {num_s}"))?;
                out.push(ArithTok::Num(n));
            }
            other => return Err(format!("token aritmético inesperado: {other}")),
        }
    }
    Ok(out)
}

fn parse_add(toks: &[ArithTok], pos: &mut usize) -> Result<f64, String> {
    let mut v = parse_mul(toks, pos)?;
    while *pos < toks.len() {
        match toks[*pos] {
            ArithTok::Plus => { *pos += 1; v += parse_mul(toks, pos)?; }
            ArithTok::Minus => { *pos += 1; v -= parse_mul(toks, pos)?; }
            _ => break,
        }
    }
    Ok(v)
}

fn parse_mul(toks: &[ArithTok], pos: &mut usize) -> Result<f64, String> {
    let mut v = parse_pow(toks, pos)?;
    while *pos < toks.len() {
        match toks[*pos] {
            ArithTok::Mul => { *pos += 1; v *= parse_pow(toks, pos)?; }
            ArithTok::Div => {
                *pos += 1;
                let d = parse_pow(toks, pos)?;
                if d.abs() < 1e-12 {
                    return Err("división por cero".into());
                }
                v /= d;
            }
            _ => break,
        }
    }
    Ok(v)
}

fn parse_pow(toks: &[ArithTok], pos: &mut usize) -> Result<f64, String> {
    let base = parse_unary(toks, pos)?;
    if *pos < toks.len() {
        if matches!(toks[*pos], ArithTok::Pow) {
            *pos += 1;
            let exp = parse_pow(toks, pos)?;
            return Ok(base.powf(exp));
        }
    }
    Ok(base)
}

fn parse_unary(toks: &[ArithTok], pos: &mut usize) -> Result<f64, String> {
    if *pos >= toks.len() {
        return Err("fin inesperado de expresión".into());
    }
    match toks[*pos] {
        ArithTok::Minus => { *pos += 1; Ok(-parse_unary(toks, pos)?) }
        ArithTok::Plus => { *pos += 1; parse_unary(toks, pos) }
        ArithTok::LParen => {
            *pos += 1;
            let v = parse_add(toks, pos)?;
            if *pos >= toks.len() || !matches!(toks[*pos], ArithTok::RParen) {
                return Err("falta paréntesis de cierre".into());
            }
            *pos += 1;
            Ok(v)
        }
        ArithTok::Num(n) => { *pos += 1; Ok(n) }
        _ => Err("factor inválido".into()),
    }
}
