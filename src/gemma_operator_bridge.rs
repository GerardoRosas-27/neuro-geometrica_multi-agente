//! Puente estructurado Gemma 2 → `OperatorRecipe`.
//!
//! Gemma describe variables y factores lógicos. La validación y la compilación
//! numérica permanecen en Rust.

use crate::native_cognitive_closed_loop::{memory_context, summarize_solution};
use crate::native_gemma2::{Gemma2Tokenizer, QuantizedGemma2};
use crate::native_multi_operator_core::{CognitiveEpisode, OperatorRecipe, SolvedRecipe};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct GemmaRecipeGenerationConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

impl Default for GemmaRecipeGenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 768,
            temperature: 0.05,
            top_p: 0.90,
            seed: 0x0FEB_A70A_5EED,
        }
    }
}

#[derive(Debug)]
pub enum GemmaOperatorBridgeError {
    Model(candle_core::Error),
    MissingJsonObject,
    InvalidRecipeJson(String),
    InvalidRecipe(String),
}

impl fmt::Display for GemmaOperatorBridgeError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(output, "error Gemma/Candle: {error}"),
            Self::MissingJsonObject => write!(output, "Gemma no produjo un objeto JSON completo"),
            Self::InvalidRecipeJson(error) => write!(output, "JSON de receta inválido: {error}"),
            Self::InvalidRecipe(error) => write!(output, "receta rechazada: {error}"),
        }
    }
}

impl std::error::Error for GemmaOperatorBridgeError {}

impl From<candle_core::Error> for GemmaOperatorBridgeError {
    fn from(error: candle_core::Error) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecipeOrigin {
    GemmaStructured,
    DeterministicQuboFallback,
}

#[derive(Clone, Debug)]
pub struct GeneratedOperatorRecipe {
    pub recipe: OperatorRecipe,
    pub raw_model_output: String,
    pub origin: RecipeOrigin,
}

pub fn operator_recipe_prompt(problem: &str) -> String {
    format!(
        r#"Transforma la tarea a DSL. Responde solo la DSL terminada en END.

Ejemplo tarea: Minimiza -a -b + 2ab para bits a,b.
operator=qubo
name=ejemplo_bits
variables=a:binary,b:binary
unary=a:-1:0,b:-1:0
pairs=a:b:2:0
faces=
flows=
max_working_set=8192
ridge=0.001
END

Reglas: qubo usa binary; l0 usa phasor; l1 usa complex, pairs, faces y flows.
No copies el ejemplo. Genera la DSL de esta tarea:
{problem}
DSL:"#
    )
}

pub fn parse_operator_recipe(text: &str) -> Result<OperatorRecipe, GemmaOperatorBridgeError> {
    let recipe = if let Some(json) = extract_first_json_object(text) {
        serde_json::from_str::<OperatorRecipe>(json)
            .map_err(|error| GemmaOperatorBridgeError::InvalidRecipeJson(error.to_string()))?
    } else {
        parse_compact_dsl(text)?
    };
    recipe
        .validate()
        .map_err(|error| GemmaOperatorBridgeError::InvalidRecipe(error.to_string()))?;
    Ok(recipe)
}

pub fn generate_operator_recipe(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    problem: &str,
    device: &Device,
    config: GemmaRecipeGenerationConfig,
) -> Result<GeneratedOperatorRecipe, GemmaOperatorBridgeError> {
    generate_operator_recipe_with_memory(model, tokenizer, problem, &[], device, config)
}

pub fn generate_operator_recipe_with_memory(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    problem: &str,
    memories: &[CognitiveEpisode],
    device: &Device,
    config: GemmaRecipeGenerationConfig,
) -> Result<GeneratedOperatorRecipe, GemmaOperatorBridgeError> {
    if let Some(recipe) = compile_simple_qubo_expression(problem) {
        return Ok(GeneratedOperatorRecipe {
            recipe,
            raw_model_output: String::new(),
            origin: RecipeOrigin::DeterministicQuboFallback,
        });
    }
    let instruction = if memories.is_empty() {
        operator_recipe_prompt(problem)
    } else {
        format!(
            "Memoria externa verificada. Úsala sólo como precedente; la tarea actual manda:\n{}\n\n{}",
            memory_context(memories),
            operator_recipe_prompt(problem)
        )
    };
    let rendered =
        generate_gemma_text(model, tokenizer, &instruction, device, config, Some("END"))?;
    if let Ok(recipe) = parse_operator_recipe(&rendered) {
        return Ok(GeneratedOperatorRecipe {
            recipe,
            raw_model_output: rendered,
            origin: RecipeOrigin::GemmaStructured,
        });
    }
    if let Some(recipe) = compile_simple_qubo_expression(problem) {
        return Ok(GeneratedOperatorRecipe {
            recipe,
            raw_model_output: rendered,
            origin: RecipeOrigin::DeterministicQuboFallback,
        });
    }
    Err(GemmaOperatorBridgeError::InvalidRecipeJson(format!(
        "salida Gemma no compilable y sin fallback determinista: {rendered}"
    )))
}

// Generación guiada: los parámetros son facetas del prompt que los callers
// pasan de forma posicional; un struct de opciones no mejoraría la legibilidad.
#[allow(clippy::too_many_arguments)]
pub fn generate_solution_explanation(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    problem: &str,
    recipe: &OperatorRecipe,
    solved: &SolvedRecipe,
    memories: &[CognitiveEpisode],
    device: &Device,
    config: GemmaRecipeGenerationConfig,
) -> Result<String, GemmaOperatorBridgeError> {
    let variable_preview = recipe
        .variables
        .iter()
        .take(12)
        .map(|variable| variable.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let summary = summarize_solution(solved);
    let instruction = format!(
        "Explica en español el resultado verificado del solver nativo. No recalcules, no cambies \
         cifras y no afirmes optimalidad global salvo que los datos indiquen exactitud. La primera \
         línea debe ser exactamente `EVIDENCIA: {summary}`. Después interpreta sin introducir \
         ningún número nuevo. Termina en END.\n\
         Tarea: {problem}\n\
         Receta: nombre={} operador={:?} variables={} [{}] pares={} caras={} flujos={}\n\
         Resultado del solver: {}\n\
         Memoria recuperada:\n{}",
        recipe.name,
        solved.operator,
        recipe.variables.len(),
        variable_preview,
        recipe.pair_factors.len(),
        recipe.oriented_faces.len(),
        recipe.flow_demands.len(),
        summary,
        memory_context(memories)
    );
    let rendered =
        generate_gemma_text(model, tokenizer, &instruction, device, config, Some("END"))?;
    let answer = rendered
        .split_once("END")
        .map_or(rendered.as_str(), |(answer, _)| answer)
        .trim()
        .to_string();
    if explanation_is_grounded(&answer, &summary, solved) {
        Ok(answer)
    } else {
        Err(GemmaOperatorBridgeError::InvalidRecipe(
            "explicación Gemma no quedó anclada exactamente al resultado del solver".to_string(),
        ))
    }
}

fn explanation_is_grounded(answer: &str, summary: &str, solved: &SolvedRecipe) -> bool {
    if !answer.lines().next().is_some_and(|line| {
        line.trim()
            .strip_prefix("EVIDENCIA:")
            .is_some_and(|evidence| evidence.trim() == summary)
    }) {
        return false;
    }
    let allowed_numbers = numeric_tokens(summary);
    if numeric_tokens(answer)
        .iter()
        .any(|number| !allowed_numbers.contains(number))
    {
        return false;
    }
    let lower = answer.to_lowercase();
    let exact = matches!(
        &solved.solution,
        crate::native_multi_operator_core::OperatorSolution::Qubo(solution) if solution.exact
    );
    exact || (!lower.contains("óptimo global") && !lower.contains("optimo global"))
}

fn numeric_tokens(text: &str) -> BTreeSet<String> {
    text.split(|character: char| {
        !(character.is_ascii_digit() || matches!(character, '.' | '-' | '+' | 'e' | 'E' | '/'))
    })
    .filter(|token| token.chars().any(|character| character.is_ascii_digit()))
    .map(str::to_string)
    .collect()
}

fn generate_gemma_text(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    instruction: &str,
    device: &Device,
    config: GemmaRecipeGenerationConfig,
    stop_marker: Option<&str>,
) -> Result<String, GemmaOperatorBridgeError> {
    let chat = format!("<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n");
    let mut prompt_tokens = vec![tokenizer.bos_id];
    prompt_tokens.extend(tokenizer.encode(&chat)?);
    if prompt_tokens.len() >= model.max_context() {
        return Err(GemmaOperatorBridgeError::InvalidRecipe(
            "prompt excede contexto Gemma".to_string(),
        ));
    }
    model.clear_kv_cache();
    let input = Tensor::new(prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
    let mut logits = model.forward(&input, 0)?.squeeze(0)?;
    let mut sampler = LogitsProcessor::new(
        config.seed,
        Some(config.temperature.max(f64::EPSILON)),
        Some(config.top_p.clamp(f64::EPSILON, 1.0)),
    );
    let mut generated = Vec::new();
    for _ in 0..config.max_tokens.max(1) {
        let token = sampler.sample(&logits)?;
        if token == tokenizer.eos_id || Some(token) == tokenizer.end_of_turn_id {
            break;
        }
        generated.push(token);
        if generated.len() % 4 == 0 {
            let partial = tokenizer.decode(&generated, true)?;
            if stop_marker.is_some_and(|marker| partial.contains(marker)) {
                break;
            }
        }
        if prompt_tokens.len() + generated.len() >= model.max_context() {
            break;
        }
        let next = Tensor::new(&[token], device)?.unsqueeze(0)?;
        logits = model
            .forward(&next, prompt_tokens.len() + generated.len() - 1)?
            .squeeze(0)?;
    }
    tokenizer.decode(&generated, true).map_err(Into::into)
}

pub fn compile_simple_qubo_expression(problem: &str) -> Option<OperatorRecipe> {
    use crate::native_multi_operator_core::{
        PairFactor, RequestedOperator, UnaryFactor, VariableDomain, VariableSpec,
    };

    let lower = problem.to_ascii_lowercase().replace('−', "-");
    let expression = lower
        .split_once("minimiza")
        .or_else(|| lower.split_once("minimize"))
        .map(|(_, expression)| expression)?
        .split(" para ")
        .next()?
        .split('.')
        .next()?
        .replace([' ', '*'], "");
    let mut normalized = expression;
    if !normalized.starts_with('+') && !normalized.starts_with('-') {
        normalized.insert(0, '+');
    }
    let mut terms = Vec::new();
    let bytes = normalized.as_bytes();
    let mut start = 0usize;
    for index in 1..bytes.len() {
        if matches!(bytes[index], b'+' | b'-') {
            terms.push(&normalized[start..index]);
            start = index;
        }
    }
    terms.push(&normalized[start..]);

    let mut unary = BTreeMap::<String, f32>::new();
    let mut pairs = BTreeMap::<(String, String), f32>::new();
    let mut variables = BTreeSet::new();
    for term in terms {
        let sign = if term.starts_with('-') { -1.0 } else { 1.0 };
        let body = &term[1..];
        let coefficient_end = body
            .char_indices()
            .find(|(_, character)| character.is_ascii_alphabetic())
            .map(|(index, _)| index)?;
        let coefficient = if coefficient_end == 0 {
            1.0
        } else {
            body[..coefficient_end].parse::<f32>().ok()?
        };
        let names = body[coefficient_end..]
            .chars()
            .filter(|character| character.is_ascii_alphabetic())
            .map(|character| character.to_string())
            .collect::<Vec<_>>();
        match names.as_slice() {
            [variable] => {
                variables.insert(variable.clone());
                *unary.entry(variable.clone()).or_default() += sign * coefficient;
            }
            [a, b] if a != b => {
                variables.insert(a.clone());
                variables.insert(b.clone());
                let key = if a <= b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                };
                *pairs.entry(key).or_default() += sign * coefficient;
            }
            _ => return None,
        }
    }
    if variables.is_empty() {
        return None;
    }
    let recipe = OperatorRecipe {
        name: format!("qubo_{:016x}", stable_text_hash(problem.as_bytes())),
        requested_operator: RequestedOperator::Qubo,
        variables: variables
            .into_iter()
            .map(|name| VariableSpec {
                name,
                domain: VariableDomain::Binary,
            })
            .collect(),
        unary_factors: unary
            .into_iter()
            .filter(|(_, weight)| weight.abs() > f32::EPSILON)
            .map(|(variable, weight)| UnaryFactor {
                variable,
                weight,
                phase: 0.0,
            })
            .collect(),
        pair_factors: pairs
            .into_iter()
            .filter(|(_, weight)| weight.abs() > f32::EPSILON)
            .map(|((a, b), weight)| PairFactor {
                a,
                b,
                weight,
                phase: 0.0,
            })
            .collect(),
        oriented_faces: Vec::new(),
        flow_demands: Vec::new(),
        max_working_set: 512,
        ridge: 1.0e-3,
    };
    recipe.validate().ok()?;
    Some(recipe)
}

fn stable_text_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn parse_compact_dsl(text: &str) -> Result<OperatorRecipe, GemmaOperatorBridgeError> {
    use crate::native_multi_operator_core::{
        FlowDemand, OrientedFace, PairFactor, RequestedOperator, UnaryFactor, VariableDomain,
        VariableSpec,
    };

    let mut normalized = text.replace('\r', "\n");
    for key in [
        "operator",
        "name",
        "variables",
        "unary",
        "pairs",
        "faces",
        "flows",
        "max_working_set",
        "ridge",
    ] {
        normalized = normalized.replace(&format!(" {key}="), &format!("\n{key}="));
    }
    let normalized = normalized.split("END").next().unwrap_or(&normalized);
    let mut fields = std::collections::BTreeMap::<String, String>::new();
    for line in normalized.lines() {
        let line = line.trim().trim_matches('`');
        if line.is_empty() || line == "END" {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            fields.insert(key.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    let required = |key: &str| {
        fields
            .get(key)
            .cloned()
            .ok_or_else(|| GemmaOperatorBridgeError::InvalidRecipeJson(format!("falta {key}")))
    };
    let requested_operator = match required("operator")?.to_ascii_lowercase().as_str() {
        "auto" => RequestedOperator::Auto,
        "l0" => RequestedOperator::L0,
        "qubo" => RequestedOperator::Qubo,
        "l1" => RequestedOperator::L1,
        other => {
            return Err(GemmaOperatorBridgeError::InvalidRecipeJson(format!(
                "operator desconocido: {other}"
            )))
        }
    };
    let variables = parse_items(fields.get("variables"), 2, |parts| {
        let domain = match parts[1].to_ascii_lowercase().as_str() {
            "binary" => VariableDomain::Binary,
            "phasor" => VariableDomain::Phasor,
            "complex" => VariableDomain::Complex,
            other => return Err(format!("dominio desconocido: {other}")),
        };
        Ok(VariableSpec {
            name: parts[0].to_string(),
            domain,
        })
    })?;
    let unary_factors = parse_items(fields.get("unary"), 2, |parts| {
        Ok(UnaryFactor {
            variable: parts[0].to_string(),
            weight: parse_f32(parts[1], "peso unary")?,
            phase: parts
                .get(2)
                .map_or(Ok(0.0), |value| parse_f32(value, "fase unary"))?,
        })
    })?;
    let pair_factors = parse_items(fields.get("pairs"), 3, |parts| {
        Ok(PairFactor {
            a: parts[0].to_string(),
            b: parts[1].to_string(),
            weight: parse_f32(parts[2], "peso pair")?,
            phase: parts
                .get(3)
                .map_or(Ok(0.0), |value| parse_f32(value, "fase pair"))?,
        })
    })?;
    let oriented_faces = parse_items(fields.get("faces"), 3, |parts| {
        Ok(OrientedFace {
            vertices: [
                parts[0].to_string(),
                parts[1].to_string(),
                parts[2].to_string(),
            ],
        })
    })?;
    let flow_demands = parse_items(fields.get("flows"), 3, |parts| {
        Ok(FlowDemand {
            from: parts[0].to_string(),
            to: parts[1].to_string(),
            real: parse_f32(parts[2], "flujo real")?,
            imag: parts
                .get(3)
                .map_or(Ok(0.0), |value| parse_f32(value, "flujo imaginario"))?,
        })
    })?;
    let max_working_set = fields
        .get("max_working_set")
        .map_or(Ok(8_192), |value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("max_working_set: {error}"))
        })
        .map_err(GemmaOperatorBridgeError::InvalidRecipeJson)?;
    let ridge = fields
        .get("ridge")
        .map_or(Ok(1.0e-3), |value| parse_f32(value, "ridge"))
        .map_err(GemmaOperatorBridgeError::InvalidRecipeJson)?;
    Ok(OperatorRecipe {
        name: required("name")?,
        requested_operator,
        variables,
        unary_factors,
        pair_factors,
        oriented_faces,
        flow_demands,
        max_working_set,
        ridge,
    })
}

fn parse_items<T>(
    value: Option<&String>,
    minimum_parts: usize,
    mut parse: impl FnMut(&[&str]) -> Result<T, String>,
) -> Result<Vec<T>, GemmaOperatorBridgeError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let value = value.trim();
    if value.is_empty() || matches!(value, "[]" | "none" | "null") {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .map(|item| {
            let parts = item.trim().split(':').map(str::trim).collect::<Vec<_>>();
            if parts.len() < minimum_parts {
                return Err(GemmaOperatorBridgeError::InvalidRecipeJson(format!(
                    "elemento DSL incompleto: {item}"
                )));
            }
            parse(&parts).map_err(GemmaOperatorBridgeError::InvalidRecipeJson)
        })
        .collect()
}

fn parse_f32(value: &str, field: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|error| format!("{field}: {error}"))
}

fn extract_first_json_object(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|byte| *byte == b'{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, byte) in bytes[start..].iter().copied().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(&text[start..start + offset + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_multi_operator_core::RequestedOperator;

    #[test]
    fn extracts_recipe_from_surrounding_text_and_braces_inside_strings() {
        let text = r#"prefacio
```json
{"name":"memoria_{a}","requested_operator":"l0","variables":[{"name":"a","domain":"phasor"},{"name":"b","domain":"phasor"}],"unary_factors":[],"pair_factors":[{"a":"a","b":"b","weight":1.0,"phase":0.0}],"oriented_faces":[],"flow_demands":[],"max_working_set":8,"ridge":0.001}
```
epílogo"#;
        let recipe = parse_operator_recipe(text).unwrap();
        assert_eq!(recipe.name, "memoria_{a}");
        assert_eq!(recipe.selected_operator().unwrap(), RequestedOperator::L0);
    }

    #[test]
    fn rejects_incomplete_json() {
        let error = parse_operator_recipe(r#"{"name":"incompleta""#).unwrap_err();
        assert!(matches!(
            error,
            GemmaOperatorBridgeError::InvalidRecipeJson(_)
        ));
    }

    #[test]
    fn parses_compact_qubo_dsl() {
        let recipe = parse_operator_recipe(
            "operator=qubo\nname=dos_bits\nvariables=x:binary,y:binary\n\
             unary=x:-1:0,y:-1:0\npairs=x:y:2:0\nfaces=\nflows=\n\
             max_working_set=32\nridge=0.001\nEND",
        )
        .unwrap();
        assert_eq!(recipe.selected_operator().unwrap(), RequestedOperator::Qubo);
        assert_eq!(recipe.variables.len(), 2);
        assert_eq!(recipe.pair_factors.len(), 1);
    }

    #[test]
    fn deterministic_fallback_compiles_simple_qubo_expression() {
        let recipe = compile_simple_qubo_expression(
            "Minimiza -x -y + 2xy para dos variables binarias x e y.",
        )
        .unwrap();
        assert_eq!(recipe.selected_operator().unwrap(), RequestedOperator::Qubo);
        assert_eq!(recipe.variables.len(), 2);
        assert_eq!(recipe.unary_factors.len(), 2);
        assert_eq!(recipe.pair_factors[0].weight, 2.0);
    }
}
