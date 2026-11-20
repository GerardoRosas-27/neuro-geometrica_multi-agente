//! Periferia sensorial congelada (Encoder lingüístico + Parser proposicional + Generador contrafactual).
//!
//! Convierte lenguaje natural (español/inglés) en:
//! 1. Vector continuo en la variedad semántica `x_t ∈ R^{D_SEM}` (`D_SEM = 64`):
//!    - Si existe un Gemma 2 GGUF cargado (`FrozenGemma2Probe`), usa estados ocultos
//!      congelados proyectados ortogonalmente (reutilizando `native_gemma2.rs`).
//!    - En CPU puro sin GGUF, usa un codificador determinista multi-escala que combina
//!      n-gramas de caracteres + lemas normalizados + rasgos de acto de habla/epistémicos +
//!      métricas de ambigüedad/fundamentación reutilizadas de `crate::prompt_spider`.
//! 2. Marco semántico estructurado (`SemanticFrame`) con proposiciones, reglas lógicas,
//!    expresiones aritméticas, consultas geométricas y marcadores de revisión.
//! 3. Perturbación contrafactual automática `x_cf ∈ R^{D_SEM}` invirtiendo polaridad/premisas,
//!    esencial para detectar colapso dogmático sin etiquetas externas.

#![allow(clippy::needless_range_loop)]

use crate::prompt_spider::{score_token, tokenize, PromptContext, TokenKind};
use serde::{Deserialize, Serialize};

/// Dimensión del espacio semántico continuo de entrada y respuesta.
pub const D_SEM: usize = 64;

/// Intención / acto de diálogo inferido por la arquitectura.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DialogueIntent {
    Greeting = 0,
    FactAssertion = 1,
    FactQuery = 2,
    LogicReasoning = 3,
    ContradictionCheck = 4,
    CounterfactualQuery = 5,
    ArithmeticQuery = 6,
    GeometricQuery = 7,
    CorrectionFeedback = 8,
    AbstentionOrAmbiguous = 9,
    ConceptualOpen = 10,
    MetaSelfQuery = 11,
}

pub const N_INTENTS: usize = 12;

impl DialogueIntent {
    pub fn from_idx(i: usize) -> Self {
        match i {
            0 => Self::Greeting,
            1 => Self::FactAssertion,
            2 => Self::FactQuery,
            3 => Self::LogicReasoning,
            4 => Self::ContradictionCheck,
            5 => Self::CounterfactualQuery,
            6 => Self::ArithmeticQuery,
            7 => Self::GeometricQuery,
            8 => Self::CorrectionFeedback,
            9 => Self::AbstentionOrAmbiguous,
            10 => Self::ConceptualOpen,
            _ => Self::MetaSelfQuery,
        }
    }

    pub fn label_es(self) -> &'static str {
        match self {
            Self::Greeting => "saludo",
            Self::FactAssertion => "afirmación_hecho",
            Self::FactQuery => "consulta_memoria",
            Self::LogicReasoning => "razonamiento_lógico",
            Self::ContradictionCheck => "prueba_contradicción",
            Self::CounterfactualQuery => "consulta_contrafactual",
            Self::ArithmeticQuery => "cálculo_aritmético",
            Self::GeometricQuery => "transformación_geométrica",
            Self::CorrectionFeedback => "corrección_usuario",
            Self::AbstentionOrAmbiguous => "ambiguo_o_inverificable",
            Self::ConceptualOpen => "pregunta_conceptual_abierta",
            Self::MetaSelfQuery => "consulta_metacognitiva",
        }
    }
}

/// Proposición atómica o regla lógica extraída de una cláusula.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExtractedClaim {
    /// Relación (sujeto, relación, objeto, polaridad).
    Fact {
        subject: String,
        relation: String,
        object: String,
        polarity: bool,
    },
    /// Regla condicional: si `antecedent` (con polaridad) entonces `consequent` (con polaridad).
    Implies {
        ant_prop: String,
        ant_pol: bool,
        cons_prop: String,
        cons_pol: bool,
    },
    /// Cuantificación universal: todos los `category` son `property` (o ninguno si `polarity == false`).
    Universal {
        category: String,
        property: String,
        polarity: bool,
    },
    /// Proposición booleana simple: `prop` es verdadero/falso.
    Prop { name: String, polarity: bool },
}

/// Marco semántico estructurado producido por la periferia sensorial congelada.
#[derive(Clone, Debug)]
pub struct SemanticFrame {
    pub raw_text: String,
    pub normalized: String,
    pub tokens: Vec<String>,
    pub vec: [f64; D_SEM],
    pub counterfactual_vec: [f64; D_SEM],
    pub has_question: bool,
    pub has_negation: bool,
    pub has_conditional: bool,
    pub has_universal: bool,
    pub has_counterfactual: bool,
    pub has_correction: bool,
    pub has_arithmetic: bool,
    pub has_geometry: bool,
    pub has_self_ref: bool,
    pub has_unverifiable_cue: bool,
    pub spider_ambiguity: f64,
    pub spider_grounding: f64,
    pub claims: Vec<ExtractedClaim>,
    pub query_terms: Vec<String>,
    pub arithmetic_expr: Option<String>,
    pub intent_hint: DialogueIntent,
}

/// Normaliza texto en español/inglés: minúsculas y plegado de acentos preservando palabras.
pub fn normalize_es(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        let c = match ch {
            'á' | 'Á' | 'à' | 'À' | 'ä' | 'Ä' => 'a',
            'é' | 'É' | 'è' | 'È' | 'ë' | 'Ë' => 'e',
            'í' | 'Í' | 'ì' | 'Ì' | 'ï' | 'Ï' => 'i',
            'ó' | 'Ó' | 'ò' | 'Ò' | 'ö' | 'Ö' => 'o',
            'ú' | 'Ú' | 'ù' | 'Ù' | 'ü' | 'Ü' => 'u',
            'ñ' | 'Ñ' => 'n',
            _ => ch.to_ascii_lowercase(),
        };
        out.push(c);
    }
    out
}

/// Lema ligero español/inglés para mejorar generalización léxica en el espacio continuo y grafo.
pub fn stem_es(w: &str) -> String {
    let w = w.trim();
    if w.len() <= 3 {
        return w.to_string();
    }
    for suf in ["ciones", "siones", "mente", "ando", "iendo", "ados", "adas", "idos", "idas"] {
        if w.len() > suf.len() + 2 && w.ends_with(suf) {
            return w[..w.len() - suf.len()].to_string();
        }
    }
    if w.len() > 4 && w.ends_with("es") && !w.ends_with("ies") {
        let base = &w[..w.len() - 2];
        if base.ends_with('r') || base.ends_with('l') || base.ends_with('n') || base.ends_with('d') || base.ends_with('z') || base.ends_with('s') {
            return base.to_string();
        }
    }
    if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") && !w.ends_with("us") && !w.ends_with("is") {
        return w[..w.len() - 1].to_string();
    }
    w.to_string()
}

pub fn content_words(norm: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "en",
        "con", "por", "para", "sobre", "entre", "que", "cual", "como", "donde", "cuando",
        "quien", "este", "esta", "esto", "ese", "esa", "eso", "mi", "tu", "su", "mis", "tus",
        "sus", "me", "te", "se", "le", "les", "nos", "y", "o", "u", "pero", "mas", "muy",
        "ya", "si", "the", "an", "of", "to", "in", "is", "are", "was", "were", "it", "this",
        "that", "and", "or", "for", "with", "on", "at", "by", "from", "be", "have", "has",
        "do", "does", "what", "how", "why", "who", "where", "when", "hay", "son", "es", "esta",
        "estan", "ser", "estar", "fue", "era", "verdad", "cierto", "favor", "puedes", "dime",
        "explica", "explicame", "cuentame", "sabes", "dices",
    ];
    norm.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '+')
        .filter(|t| t.len() >= 2 && !STOP.contains(t))
        .map(stem_es)
        .collect()
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Limpia una cláusula proposicional a un identificador canónico legible.
pub fn canonical_prop(raw: &str) -> String {
    let norm = normalize_es(raw);
    let words: Vec<String> = norm
        .split(|c: char| !c.is_alphanumeric() && c != '+')
        .filter(|t| {
            !t.is_empty()
                && !matches!(
                    *t,
                    "el" | "la"
                        | "los"
                        | "las"
                        | "un"
                        | "una"
                        | "que"
                        | "entonces"
                        | "si"
                        | "cuando"
                        | "por"
                        | "tanto"
                        | "luego"
                        | "al"
                        | "mismo"
                        | "tiempo"
                        | "tambien"
                        | "realmente"
                        | "seguro"
                        | "acaso"
                )
        })
        .map(stem_es)
        .collect();
    words.join("_")
}

/// Detecta si una cláusula está negada y devuelve `(cláusula_sin_negación, polaridad)`.
pub fn strip_negation(clause: &str) -> (String, bool) {
    let norm = normalize_es(clause);
    let toks: Vec<&str> = norm
        .split(|c: char| !c.is_alphanumeric() && c != '+')
        .filter(|t| !t.is_empty())
        .collect();
    let mut neg_count = 0usize;
    let mut kept = Vec::with_capacity(toks.len());
    for &t in &toks {
        if matches!(
            t,
            "no" | "nunca" | "jamas" | "ningun" | "ninguno" | "ninguna" | "falso" | "falsa" | "not" | "never" | "false"
        ) {
            neg_count += 1;
        } else {
            kept.push(t);
        }
    }
    let pol = neg_count % 2 == 0;
    (kept.join(" "), pol)
}

/// Extrae una posible expresión aritmética segura (números, operadores, `% de`).
pub fn extract_arithmetic(norm: &str) -> Option<String> {
    // Caso especial: "X% de Y" (+ posible resto "+ Z" / "- Z")
    if norm.contains('%') || norm.contains("por ciento") {
        let replaced = norm.replace("por ciento", "%");
        let words: Vec<&str> = replaced.split_whitespace().collect();
        for i in 0..words.len() {
            if let Some(pct_s) = words[i].strip_suffix('%') {
                if let Ok(pct) = pct_s.parse::<f64>() {
                    if i + 2 < words.len() && (words[i + 1] == "de" || words[i + 1] == "of") {
                        let base_tok = words[i + 2].trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-');
                        if let Ok(base) = base_tok.parse::<f64>() {
                            let mut expr = format!("({pct} / 100.0 * {base})");
                            // Capturar operador adicional como "mas 18" o "- 5"
                            if i + 4 < words.len() {
                                let op = match words[i + 3] {
                                    "mas" | "+" => Some("+"),
                                    "menos" | "-" => Some("-"),
                                    "por" | "*" | "x" => Some("*"),
                                    "entre" | "/" => Some("/"),
                                    _ => None,
                                };
                                if let Some(op) = op {
                                    let rhs_tok = words[i + 4].trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-');
                                    if let Ok(rhs) = rhs_tok.parse::<f64>() {
                                        expr = format!("{expr} {op} {rhs}");
                                    }
                                }
                            }
                            return Some(expr);
                        }
                    }
                }
            }
        }
    }

    // Convertir palabras aritméticas simples en operadores cuando hay dígitos
    let has_digit = norm.chars().any(|c| c.is_ascii_digit());
    if !has_digit {
        return None;
    }
    let has_op_word = norm.contains('+')
        || norm.contains('*')
        || norm.contains('/')
        || norm.contains('^')
        || norm.contains(" mas ")
        || norm.contains(" menos ")
        || norm.contains(" por ")
        || norm.contains(" entre ")
        || norm.contains(" dividido ")
        || norm.contains(" suma ")
        || norm.contains(" resta ")
        || norm.contains(" multiplica ")
        || norm.contains(" calcula ")
        || norm.contains(" cuanto es ")
        || norm.contains(" + ")
        || norm.contains(" - ");
    if !has_op_word {
        return None;
    }
    let mapped = norm
        .replace(" mas ", " + ")
        .replace(" menos ", " - ")
        .replace(" dividido entre ", " / ")
        .replace(" dividido por ", " / ")
        .replace(" entre ", " / ")
        .replace(" por ", " * ")
        .replace(" x ", " * ");
    let mut expr = String::new();
    let mut n_nums = 0usize;
    let mut n_ops = 0usize;
    let mut in_num = false;
    for ch in mapped.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            if !in_num {
                n_nums += 1;
                in_num = true;
            }
            expr.push(ch);
        } else {
            in_num = false;
            if matches!(ch, '+' | '-' | '*' | '/' | '^' | '(' | ')') {
                if matches!(ch, '+' | '-' | '*' | '/' | '^') {
                    n_ops += 1;
                }
                expr.push(' ');
                expr.push(ch);
                expr.push(' ');
            } else if ch.is_whitespace() {
                expr.push(' ');
            }
        }
    }
    if n_nums >= 2 && n_ops >= 1 {
        Some(expr.trim().to_string())
    } else {
        None
    }
}

/// Extrae afirmaciones, hechos, reglas condicionales y cuantificadores universales del texto.
pub fn extract_claims(raw: &str) -> Vec<ExtractedClaim> {
    let norm = normalize_es(raw);
    let mut claims = Vec::new();

    // Dividir en oraciones / cláusulas por signos fuertes o conectores de conjunción
    for sent in norm.split(['.', ';', '\n']) {
        let s = sent.trim();
        if s.is_empty() {
            continue;
        }
        // No extraer afirmaciones de preguntas puras ni órdenes de adivinar/explicar (excepto condicionales antes de la pregunta)
        let is_pure_q = (s.contains('?')
            || s.starts_with("que ")
            || s.starts_with("como ")
            || s.starts_with("cual ")
            || s.starts_with("donde ")
            || s.starts_with("quien ")
            || s.starts_with("cuanto ")
            || s.starts_with("puede ")
            || s.starts_with("llovio")
            || s.starts_with("seguro ")
            || s.starts_with("explicame ")
            || s.starts_with("explica ")
            || s.starts_with("dime ")
            || s.starts_with("cuentame ")
            || s.starts_with("adivina ")
            || s.starts_with("inventa ")
            || s.starts_with("predice "))
            && !s.contains(" entonces ")
            && !s.contains(" implica ");
        if is_pure_q {
            continue;
        }

        // 1. Regla condicional: "si A entonces B" / "si A, B" / "A implica B"
        if let Some(pos) = s.find(" implica ") {
            let lhs = s[..pos].trim();
            let rhs = s[pos + " implica ".len()..].trim();
            if !lhs.is_empty() && !rhs.is_empty() {
                let (al, ap) = strip_negation(lhs);
                let (cr, cp) = strip_negation(rhs);
                let a = canonical_prop(&al);
                let c = canonical_prop(&cr);
                if !a.is_empty() && !c.is_empty() {
                    claims.push(ExtractedClaim::Implies {
                        ant_prop: a,
                        ant_pol: ap,
                        cons_prop: c,
                        cons_pol: cp,
                    });
                }
                continue;
            }
        }
        if let Some(rest) = s.strip_prefix("si ") {
            let split_pair = rest
                .split_once(" entonces ")
                .or_else(|| rest.split_once(", "));
            if let Some((lhs, rhs)) = split_pair {
                // Evitar confundir con pregunta contrafactual "si X, que pasaria..."
                if !rhs.contains("que pasaria") && !rhs.contains("que pasa") && !rhs.contains('?') {
                    let (al, ap) = strip_negation(lhs);
                    let (cr, cp) = strip_negation(rhs);
                    let a = canonical_prop(&al);
                    let c = canonical_prop(&cr);
                    if !a.is_empty() && !c.is_empty() {
                        claims.push(ExtractedClaim::Implies {
                            ant_prop: a,
                            ant_pol: ap,
                            cons_prop: c,
                            cons_pol: cp,
                        });
                    }
                    continue;
                }
            }
        }

        // Dividir cláusulas unidas por " y " para manejar "A y no A" o múltiples hechos
        let subclauses: Vec<&str> = s.split(" y ").collect();
        for sub in subclauses {
            let sc = sub.trim();
            if sc.is_empty() || sc.contains('?') {
                continue;
            }
            // 2. Identidad de usuario: "me llamo X" / "mi nombre es X"
            if let Some(pos) = sc.find("me llamo ") {
                let rest = &sc[pos + "me llamo ".len()..];
                let name = rest
                    .split([' ', ',', '?', '!'])
                    .find(|w| !w.is_empty())
                    .unwrap_or("");
                if !name.is_empty() {
                    claims.push(ExtractedClaim::Fact {
                        subject: "usuario".into(),
                        relation: "nombre".into(),
                        object: name.to_string(),
                        polarity: true,
                    });
                    continue;
                }
            } else if let Some(pos) = sc.find("mi nombre es ") {
                let rest = &sc[pos + "mi nombre es ".len()..];
                let name = rest
                    .split([' ', ',', '?', '!'])
                    .find(|w| !w.is_empty())
                    .unwrap_or("");
                if !name.is_empty() {
                    claims.push(ExtractedClaim::Fact {
                        subject: "usuario".into(),
                        relation: "nombre".into(),
                        object: name.to_string(),
                        polarity: true,
                    });
                    continue;
                }
            }
            // 3. Cuantificadores universales: "todos los A son B" / "ningun A es B"
            if let Some(rest) = sc
                .strip_prefix("todos los ")
                .or_else(|| sc.strip_prefix("todas las "))
                .or_else(|| sc.strip_prefix("todo "))
                .or_else(|| sc.strip_prefix("toda "))
            {
                if let Some((cat, prop)) = rest.split_once(" son ").or_else(|| rest.split_once(" es ")) {
                    let (pr, pol) = strip_negation(prop);
                    let c = canonical_prop(cat);
                    let p = canonical_prop(&pr);
                    if !c.is_empty() && !p.is_empty() {
                        claims.push(ExtractedClaim::Universal {
                            category: c,
                            property: p,
                            polarity: pol,
                        });
                        continue;
                    }
                }
            }
            if let Some(rest) = sc
                .strip_prefix("ningun ")
                .or_else(|| sc.strip_prefix("ninguna "))
            {
                if let Some((cat, prop)) = rest.split_once(" es ").or_else(|| rest.split_once(" son ")) {
                    let c = canonical_prop(cat);
                    let p = canonical_prop(prop);
                    if !c.is_empty() && !p.is_empty() {
                        claims.push(ExtractedClaim::Universal {
                            category: c,
                            property: p,
                            polarity: false,
                        });
                        continue;
                    }
                }
            }

            // 4. Hechos relacionales explícitos ("usa", "tiene", "vive en", "trabaja en", "prefiere", "cambio a")
            let mut matched_rel = false;
            for (pat, rel_name) in [
                (" usa ", "usa"),
                (" utiliza ", "usa"),
                (" tiene ", "tiene"),
                (" vive en ", "vive_en"),
                (" trabaja en ", "trabaja_en"),
                (" prefiere ", "prefiere"),
                (" construyendo ", "construye"),
                (" desarrollando ", "construye"),
                (" cambio a ", "usa"),
                (" cambie ", "usa"),
                (" ahora usa ", "usa"),
                (" paso a ", "usa"),
            ] {
                if let Some(pos) = sc.find(pat) {
                    let lhs = sc[..pos].trim();
                    let rhs = sc[pos + pat.len()..].trim();
                    let (lhs_c, pol_l) = strip_negation(lhs);
                    let (rhs_c, pol_r) = strip_negation(rhs);
                    let is_change_verb = pat.contains("cambi") || pat.contains("paso a");
                    let subj = if lhs_c.contains("mi proyecto")
                        || lhs_c.contains("proyecto")
                        || (is_change_verb && (rhs_c.contains("proyecto") || rhs_c.contains("lenguaje")))
                    {
                        "proyecto".to_string()
                    } else if is_change_verb && rhs_c.contains("nombre") {
                        "usuario".to_string()
                    } else if lhs_c.contains("yo")
                        || lhs_c.contains("estoy")
                        || lhs_c.is_empty()
                        || lhs_c == "en realidad"
                        || lhs_c == "realidad"
                        || lhs_c.starts_with("corrige")
                    {
                        "usuario".to_string()
                    } else {
                        canonical_prop(&lhs_c)
                    };
                    let rel_eff = if is_change_verb && rhs_c.contains("nombre") {
                        "nombre"
                    } else {
                        rel_name
                    };
                    // En "en realidad cambie el lenguaje de mi proyecto a python", extraer el destino tras " a "
                    let obj_raw = if is_change_verb {
                        rhs_c.rsplit(" a ").next().unwrap_or(&rhs_c)
                    } else {
                        &rhs_c
                    };
                    let obj = canonical_prop(obj_raw);
                    if !subj.is_empty() && !obj.is_empty() {
                        claims.push(ExtractedClaim::Fact {
                            subject: subj,
                            relation: rel_eff.into(),
                            object: obj,
                            polarity: pol_l == pol_r,
                        });
                        matched_rel = true;
                        break;
                    }
                }
            }
            if matched_rel {
                continue;
            }

            // 5. Predicado copulativo: "X es [no] Y" / "X esta [no] Y"
            for pat in [" no es ", " es ", " no esta ", " esta ", " no son ", " son "] {
                if let Some(pos) = sc.find(pat) {
                    let lhs = sc[..pos].trim();
                    let rhs = sc[pos + pat.len()..].trim();
                    if lhs.is_empty() || rhs.is_empty() || lhs.starts_with("que") || lhs.starts_with("cual") {
                        continue;
                    }
                    let pat_pol = !pat.contains(" no ");
                    let (lhs_c, pol_l) = strip_negation(lhs);
                    let (rhs_c, pol_r) = strip_negation(rhs);
                    let subj = if lhs_c.contains("mi proyecto") || lhs_c == "proyecto" {
                        "proyecto".to_string()
                    } else {
                        canonical_prop(&lhs_c)
                    };
                    let obj = canonical_prop(&rhs_c);
                    let pol = pat_pol && pol_l && pol_r;
                    if !subj.is_empty() && !obj.is_empty() {
                        claims.push(ExtractedClaim::Fact {
                            subject: subj.clone(),
                            relation: "es".into(),
                            object: obj.clone(),
                            polarity: pol,
                        });
                        // También registrar forma proposicional compuesta para Modus Ponens/Tollens
                        let comp = format!("{subj}_{obj}");
                        claims.push(ExtractedClaim::Prop {
                            name: comp,
                            polarity: pol,
                        });
                        matched_rel = true;
                        break;
                    }
                }
            }
            if matched_rel {
                continue;
            }

            // 6. Proposición atómica simple (p.ej. "llueve", "no llueve", "la calle no esta mojada")
            let (clean, pol) = strip_negation(sc);
            let cname = canonical_prop(&clean);
            if !cname.is_empty() && cname.len() >= 3 && !matches!(cname.as_str(), "hola" | "buenos_dia" | "buenas_tarde" | "gracias" | "adios") {
                claims.push(ExtractedClaim::Prop {
                    name: cname,
                    polarity: pol,
                });
            }
        }
    }
    claims
}

/// Codifica cualquier texto (y opcionalmente un contexto previo) en un vector normalizado `R^{D_SEM}`
/// más su versión contrafactual `counterfactual_vec` y su `SemanticFrame` estructurado.
pub fn encode_utterance(raw: &str, prev_vec: Option<&[f64; D_SEM]>) -> SemanticFrame {
    let norm = normalize_es(raw.trim());
    let toks = content_words(&norm);

    // Reutilizar el scorer determinista de `prompt_spider` para ambigüedad y fundamentación!
    let sp_tokens = tokenize(raw);
    let sp_ctx = PromptContext::new(&sp_tokens);
    let mut amb_sum = 0.0;
    let mut grd_sum = 0.0;
    let mut n_fork = 0usize;
    for (idx, t) in sp_tokens.iter().enumerate() {
        if t.kind == TokenKind::Word || t.kind == TokenKind::Number {
            let prev = idx.checked_sub(1).and_then(|j| sp_tokens.get(j));
            let a = score_token(t, &sp_ctx, prev);
            amb_sum += a.scores.ambiguous;
            grd_sum += a.scores.grounded;
            n_fork += 1;
        }
    }
    let spider_ambiguity = if n_fork > 0 { amb_sum / n_fork as f64 } else { 0.5 };
    let spider_grounding = if n_fork > 0 { grd_sum / n_fork as f64 } else { 0.5 };

    // Indicadores estructurales / actos de habla (incluye preguntas e imperativos epistémicos/explicativos)
    let has_question = raw.contains('?')
        || raw.contains('¿')
        || norm.starts_with("que ")
        || norm.starts_with("como ")
        || norm.starts_with("por que ")
        || norm.starts_with("cual ")
        || norm.starts_with("quien ")
        || norm.starts_with("donde ")
        || norm.starts_with("cuanto ")
        || norm.starts_with("puede ")
        || norm.starts_with("es cierto")
        || norm.starts_with("seguro ")
        || norm.starts_with("explicame ")
        || norm.starts_with("explica ")
        || norm.starts_with("dime ")
        || norm.starts_with("cuentame ")
        || norm.starts_with("describe ")
        || norm.starts_with("adivina ")
        || norm.starts_with("inventa ")
        || norm.starts_with("predice ")
        || norm.starts_with("explain ")
        || norm.starts_with("tell me ")
        || norm.starts_with("guess ");
    let has_negation = norm.split_whitespace().any(|w| {
        matches!(
            w,
            "no" | "nunca" | "jamas" | "ningun" | "ninguno" | "ninguna" | "falso" | "falsa" | "imposible" | "not" | "never"
        )
    });
    let has_conditional = norm.contains("si ") && (norm.contains(" entonces ") || norm.contains(", "))
        || norm.contains(" implica ");
    let has_universal = norm.contains("todos los ")
        || norm.contains("todas las ")
        || norm.contains("todo ")
        || norm.contains("ningun ")
        || norm.contains("ninguna ");
    let has_counterfactual = norm.contains("que pasaria si")
        || norm.contains("y si ")
        || norm.contains("supongamos que")
        || norm.contains("en lugar de")
        || norm.contains("en vez de")
        || norm.contains("si hubiera")
        || norm.contains("contrafactual");
    let has_correction = norm.contains("en realidad")
        || norm.contains("te equivocas")
        || norm.contains("corrige")
        || norm.contains("ahora usa")
        || norm.contains("ahora es")
        || norm.contains("cambie ")
        || norm.contains("cambio a")
        || norm.contains("olvida ")
        || norm.contains("ya no ");
    let arithmetic_expr = extract_arithmetic(&norm);
    let has_arithmetic = arithmetic_expr.is_some();
    let has_geometry = crate::field_v4::parse_family(raw).is_some()
        && (norm.contains("punto")
            || norm.contains("triangulo")
            || norm.contains("grado")
            || norm.contains("vector")
            || norm.contains("objeto")
            || norm.contains("coordenada")
            || norm.contains("transform")
            || norm.contains("figura")
            || norm.chars().any(|c| c.is_ascii_digit()));
    let has_self_ref = norm.contains("tu arquitectura")
        || norm.contains("como funcionas")
        || norm.contains("como piensas")
        || norm.contains("como aprendes")
        || norm.contains("energia libre")
        || norm.contains("metacognic")
        || norm.contains("colapso dogmatico")
        || norm.contains("dogma")
        || norm.contains("monitor acc")
        || norm.contains("neuromodul")
        || norm.contains("ewc")
        || norm.contains("pesos rapidos")
        || norm.contains("tu estado")
        || norm.contains("quien eres")
        || norm.contains("que eres");
    let has_unverifiable_cue = norm.contains("proximo viernes")
        || norm.contains("manana ganara")
        || norm.contains("loteria")
        || norm.contains("precio exacto")
        || norm.contains("futuro exacto")
        || norm.contains("secreto")
        || norm.contains("secreta")
        || norm.contains("adivina")
        || norm.contains("inventa")
        || norm.contains("sin datos")
        || norm.contains("zyloria")
        || norm.contains("planeta desconocido")
        || (has_question && toks.len() <= 2 && !norm.contains("hola") && !norm.contains("nombre") && !norm.contains("llamo"));

    let claims = extract_claims(raw);

    // Detectar contradicción intra-mensaje
    let mut intra_contra = false;
    for i in 0..claims.len() {
        for j in (i + 1)..claims.len() {
            if claims_directly_contradict(&claims[i], &claims[j]) {
                intra_contra = true;
            }
        }
    }
    let has_contra_cue = intra_contra
        || norm.contains("contradic")
        || norm.contains("al mismo tiempo")
        || norm.contains("incompatible");

    // Heurística inicial de intención (la decisión final la toma el núcleo neuronal 3S + BeliefStore)
    let is_greeting = !has_question
        && claims.is_empty()
        && toks.len() <= 4
        && (norm.starts_with("hola")
            || norm.starts_with("buenos")
            || norm.starts_with("buenas")
            || norm.starts_with("hey")
            || norm.starts_with("saludos")
            || norm == "hi"
            || norm == "hello");

    let intent_hint = if is_greeting {
        DialogueIntent::Greeting
    } else if has_contra_cue {
        DialogueIntent::ContradictionCheck
    } else if has_correction {
        DialogueIntent::CorrectionFeedback
    } else if has_counterfactual {
        DialogueIntent::CounterfactualQuery
    } else if has_arithmetic {
        DialogueIntent::ArithmeticQuery
    } else if has_geometry {
        DialogueIntent::GeometricQuery
    } else if has_conditional || has_universal {
        DialogueIntent::LogicReasoning
    } else if has_self_ref {
        DialogueIntent::MetaSelfQuery
    } else if has_unverifiable_cue {
        DialogueIntent::AbstentionOrAmbiguous
    } else if has_question && (norm.contains("me llamo") || norm.contains("mi nombre") || norm.contains("mi proyecto") || norm.contains("que sabes") || norm.contains("recuerdas") || norm.contains("que lenguaje") || norm.contains("donde vivo") || norm.contains("donde trabajo")) {
        DialogueIntent::FactQuery
    } else if !has_question && !claims.is_empty() {
        DialogueIntent::FactAssertion
    } else if has_question {
        DialogueIntent::ConceptualOpen
    } else {
        DialogueIntent::FactAssertion
    };

    // Construcción del vector semántico continuo x_t ∈ R^64:
    //   [0..32]  subespacio léxico / n-gramas + lemas (proyección ortogonal con signo)
    //   [32..48] subespacio estructural / acto de habla / polaridad epistémica
    //   [48..64] subespacio de dinámica conversacional + Prompt Spider + complejidad
    let mut vec = [0.0_f64; D_SEM];

    // [0..32] Hashing semántico multi-escala (palabras + bigramas + trigramas de caracteres)
    for w in &toks {
        let h1 = fnv64(w.as_bytes());
        let i1 = (h1 % 32) as usize;
        let s1 = if (h1 >> 8) & 1 == 0 { 1.0 } else { -1.0 };
        vec[i1] += 1.0 * s1;

        let h2 = fnv64(&h1.to_le_bytes());
        let i2 = (h2 % 32) as usize;
        let s2 = if (h2 >> 8) & 1 == 0 { 1.0 } else { -1.0 };
        vec[i2] += 0.65 * s2;

        // Trigramas de caracteres para robustez a variaciones morfológicas / typos
        let wb = w.as_bytes();
        if wb.len() >= 3 {
            for win in wb.windows(3) {
                let hc = fnv64(win);
                let ic = (hc % 32) as usize;
                let sc = if (hc >> 6) & 1 == 0 { 1.0 } else { -1.0 };
                vec[ic] += 0.22 * sc;
            }
        }
    }
    for pair in toks.windows(2) {
        let bi = format!("{}_{}", pair[0], pair[1]);
        let hb = fnv64(bi.as_bytes());
        let ib = (hb % 32) as usize;
        let sb = if (hb >> 7) & 1 == 0 { 1.0 } else { -1.0 };
        vec[ib] += 0.55 * sb;
    }
    // Normalizar subespacio léxico a norma L2 unitaria (si no está vacío)
    let lex_n: f64 = vec[0..32].iter().map(|v| v * v).sum::<f64>().sqrt();
    if lex_n > 1e-9 {
        for v in &mut vec[0..32] {
            *v /= lex_n;
        }
    }

    // [32..48] Rasgos estructurales y epistémicos (cruciales para clasificación y sensibilidad contrafactual)
    let b = |cond: bool| if cond { 1.0 } else { -0.2 };
    vec[32] = b(has_question);
    vec[33] = if has_negation { -1.0 } else { 1.0 }; // Polaridad explícita (se invierte en contrafactual!)
    vec[34] = b(has_conditional);
    vec[35] = b(has_universal);
    vec[36] = b(has_counterfactual);
    vec[37] = b(has_correction);
    vec[38] = b(has_arithmetic);
    vec[39] = b(has_geometry);
    vec[40] = b(has_self_ref);
    vec[41] = b(has_unverifiable_cue);
    vec[42] = b(has_contra_cue);
    vec[43] = b(is_greeting);
    vec[44] = (claims.len() as f64 / 3.0).clamp(0.0, 1.5) - 0.3;
    vec[45] = (toks.len() as f64 / 12.0).clamp(0.0, 1.5) - 0.5;
    vec[46] = if norm.contains("me llamo") || norm.contains("mi nombre") || norm.contains("mi proyecto") || norm.contains("recuerdas") || norm.contains("que sabes") {
        1.0
    } else {
        -0.2
    };
    vec[47] = if intra_contra { 1.2 } else { -0.2 };

    // [48..64] Dinámica conversacional + telemetría Prompt Spider + firma de claims
    let novelty = if let Some(pv) = prev_vec {
        let dot: f64 = vec[0..32].iter().zip(&pv[0..32]).map(|(a, b)| a * b).sum();
        let np: f64 = pv[0..32].iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
        (1.0 - (dot / np).clamp(-1.0, 1.0)) * 0.5
    } else {
        0.5
    };
    vec[48] = spider_ambiguity * 2.0 - 1.0;
    vec[49] = spider_grounding * 2.0 - 1.0;
    vec[50] = novelty * 2.0 - 1.0;
    vec[51] = if raw.chars().any(|c| c.is_ascii_digit()) { 0.8 } else { -0.2 };

    // Proyección de polaridad de los claims extraídos en [52..64]
    for c in &claims {
        let (key, pol) = match c {
            ExtractedClaim::Fact { subject, relation, object, polarity } => {
                (format!("{subject}:{relation}:{object}"), *polarity)
            }
            ExtractedClaim::Implies { ant_prop, cons_prop, ant_pol, cons_pol } => {
                (format!("{ant_prop}->{cons_prop}"), *ant_pol == *cons_pol)
            }
            ExtractedClaim::Universal { category, property, polarity } => {
                (format!("all:{category}:{property}"), *polarity)
            }
            ExtractedClaim::Prop { name, polarity } => (name.clone(), *polarity),
        };
        let hc = fnv64(key.as_bytes());
        let idx = 52 + (hc % 12) as usize;
        vec[idx] += if pol { 0.8 } else { -0.8 };
    }
    let claim_n: f64 = vec[52..64].iter().map(|v| v * v).sum::<f64>().sqrt();
    if claim_n > 1.0 {
        for v in &mut vec[52..64] {
            *v /= claim_n;
        }
    }

    // Normalizar vector completo suavemente (norma acotada para estabilidad numérica de W_0 + F_t)
    let full_n: f64 = vec.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
    for v in &mut vec {
        *v /= full_n;
    }

    // Construir perturbación contrafactual automática x_cf:
    // invierte la polaridad semántica, la firma de los claims y perturba el foco léxico
    let counterfactual_vec = build_counterfactual(&vec);

    SemanticFrame {
        raw_text: raw.to_string(),
        normalized: norm,
        tokens: toks.clone(),
        vec,
        counterfactual_vec,
        has_question,
        has_negation,
        has_conditional,
        has_universal,
        has_counterfactual,
        has_correction,
        has_arithmetic,
        has_geometry,
        has_self_ref,
        has_unverifiable_cue,
        spider_ambiguity,
        spider_grounding,
        claims,
        query_terms: toks,
        arithmetic_expr,
        intent_hint,
    }
}

/// Construye el vector contrafactual `x_cf` invirtiendo polaridad, claims y dirección de contradicción.
pub fn build_counterfactual(x: &[f64; D_SEM]) -> [f64; D_SEM] {
    let mut cf = *x;
    // Invertir polaridad lógica y contradicción
    cf[33] = -cf[33];
    cf[42] = -cf[42];
    cf[47] = -cf[47];
    // Invertir polaridad de los claims codificados en [52..64]
    for i in 52..64 {
        cf[i] = -cf[i];
    }
    // Rotación ortogonal suave en el subespacio léxico [0..8] simulando sustitución de predicado
    for i in (0..8).step_by(2) {
        let a = cf[i];
        let b = cf[i + 1];
        cf[i] = -b;
        cf[i + 1] = a;
    }
    let n: f64 = cf.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
    for v in &mut cf {
        *v /= n;
    }
    cf
}

/// Comprueba si dos afirmaciones extraídas se contradicen directamente (`P` vs `¬P`).
pub fn claims_directly_contradict(a: &ExtractedClaim, b: &ExtractedClaim) -> bool {
    match (a, b) {
        (
            ExtractedClaim::Fact { subject: s1, relation: r1, object: o1, polarity: p1 },
            ExtractedClaim::Fact { subject: s2, relation: r2, object: o2, polarity: p2 },
        ) => s1 == s2 && r1 == r2 && o1 == o2 && p1 != p2,
        (
            ExtractedClaim::Prop { name: n1, polarity: p1 },
            ExtractedClaim::Prop { name: n2, polarity: p2 },
        ) => n1 == n2 && p1 != p2,
        (
            ExtractedClaim::Universal { category: c1, property: pr1, polarity: p1 },
            ExtractedClaim::Universal { category: c2, property: pr2, polarity: p2 },
        ) => c1 == c2 && pr1 == pr2 && p1 != p2,
        _ => false,
    }
}
