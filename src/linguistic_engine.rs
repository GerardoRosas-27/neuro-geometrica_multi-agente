use crate::simplicial::ConceptProjection;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct LinguisticContext {
    pub user_prompt: String,
    pub inferred_intent: String,
    pub geometric_projection: ConceptProjection,
    pub memory_summary: String,
}

#[derive(Clone, Debug)]
pub struct LinguisticResponse {
    pub text: String,
    pub engine: String,
}

pub trait LinguisticEngine {
    fn generate(&self, context: &LinguisticContext) -> Result<LinguisticResponse, String>;
}

#[derive(Clone, Debug)]
pub struct OllamaGemmaEngine {
    pub host: String,
    pub model: String,
}

impl Default for OllamaGemmaEngine {
    fn default() -> Self {
        Self {
            host: "127.0.0.1:11434".to_string(),
            model: "gemma2:2b".to_string(),
        }
    }
}

impl LinguisticEngine for OllamaGemmaEngine {
    fn generate(&self, context: &LinguisticContext) -> Result<LinguisticResponse, String> {
        let prompt = build_gemma_prompt(context);
        let body = format!(
            "{{\"model\":\"{}\",\"prompt\":\"{}\",\"stream\":false}}",
            escape_json(&self.model),
            escape_json(&prompt)
        );
        let request = format!(
            "POST /api/generate HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.host,
            body.as_bytes().len(),
            body
        );

        let mut stream = TcpStream::connect(&self.host)
            .map_err(|err| format!("no se pudo conectar a Ollama en {}: {err}", self.host))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(45)))
            .map_err(|err| format!("no se pudo configurar timeout de lectura: {err}"))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(10)))
            .map_err(|err| format!("no se pudo configurar timeout de escritura: {err}"))?;
        stream
            .write_all(request.as_bytes())
            .map_err(|err| format!("fallo enviando prompt a Ollama: {err}"))?;

        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .map_err(|err| format!("fallo leyendo respuesta de Ollama: {err}"))?;
        let body = response
            .split("\r\n\r\n")
            .nth(1)
            .ok_or_else(|| "respuesta HTTP invalida de Ollama".to_string())?;
        let text = extract_json_string_field(body, "response")
            .ok_or_else(|| format!("Ollama no devolvio campo response: {body}"))?;

        Ok(LinguisticResponse {
            text: text.trim().to_string(),
            engine: format!("ollama/{}", self.model),
        })
    }
}

pub fn fallback_response(context: &LinguisticContext) -> LinguisticResponse {
    let active = context
        .geometric_projection
        .top_agents
        .iter()
        .map(|(idx, value)| format!("{idx}:{value:.2}"))
        .collect::<Vec<_>>()
        .join(", ");
    LinguisticResponse {
        text: format!(
            "SNGA detecta intencion '{}' con memoria geometrica activa [{}]. {}",
            context.inferred_intent, active, context.memory_summary
        ),
        engine: "snga-symbolic-fallback".to_string(),
    }
}

fn build_gemma_prompt(context: &LinguisticContext) -> String {
    let projection = context
        .geometric_projection
        .top_agents
        .iter()
        .map(|(idx, value)| format!("agente {idx} sorpresa {value:.3}"))
        .collect::<Vec<_>>()
        .join("; ");

    format!(
        "Eres el renderizador linguistico periferico de SNGA.\n\
        No inventes memoria nueva. Usa el estado geometrico como contexto.\n\
        Responde en espanol, breve y claro.\n\n\
        Prompt del usuario: {}\n\
        Intencion inferida por SNGA: {}\n\
        Proyeccion geometrica activa: {}\n\
        Memoria/resumen SNGA: {}\n\n\
        Respuesta:",
        context.user_prompt, context.inferred_intent, projection, context.memory_summary
    )
}

fn escape_json(value: &str) -> String {
    let mut escaped = String::new();
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn extract_json_string_field(json: &str, field: &str) -> Option<String> {
    let needle = format!("\"{field}\":\"");
    let start = json.find(&needle)? + needle.len();
    let mut out = String::new();
    let mut escaped = false;
    for ch in json[start..].chars() {
        if escaped {
            match ch {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                _ => out.push(ch),
            }
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => return Some(out),
            _ => out.push(ch),
        }
    }
    None
}
