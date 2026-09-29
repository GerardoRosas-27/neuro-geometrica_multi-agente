//! Proveedores LLM de la app: **Gemma local** (GGUF, por defecto) o APIs
//! externas compatibles con OpenAI (p. ej. docker-llm, claves `obk1.…`).
//!
//! - [`ProviderStore`]: lista persistida (JSON en el dir de datos) + proveedor
//!   activo. Bootstrap opcional por env `LLM_API_BASE` / `LLM_API_KEY` /
//!   `LLM_API_MODEL` (`LLM_API_NAME`, `LLM_API_ACTIVE=1`).
//! - [`parse_curl`]: convierte un `curl` pegado (continuaciones, comillas,
//!   `export VAR=…`, `$VAR`/`${VAR}`, `-H`/`--header`, `-d`/`--data[-raw]`) en
//!   URL base (hasta `/v1`), API key (Bearer) y modelo.
//! - [`OpenAiClient`]: cliente HTTP bloqueante (llamar en `spawn_blocking`).
//!
//! La API key **nunca** se registra en logs y en respuestas GET solo sale
//! enmascarada ([`mask_key`]).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Id reservado del LLM local (Gemma 2 GGUF / léxico).
pub const LOCAL_ID: &str = "gemma_local";
/// Id del proveedor definido por variables de entorno.
pub const ENV_ID: &str = "env";

/// Proveedor externo compatible con OpenAI.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProviderConfig {
    pub id: String,
    pub name: String,
    /// Base hasta `/v1` incluido, sin barra final (p. ej. `https://x/v1`).
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub created_ms: u64,
    /// `ui` (guardado desde la app) | `env` (variables de entorno).
    #[serde(default = "default_source")]
    pub source: String,
}

fn default_source() -> String {
    "ui".into()
}

impl ProviderConfig {
    /// Etiqueta corta para la UI/respuestas: `API · nombre (modelo)`.
    pub fn label(&self) -> String {
        if self.model.is_empty() || self.name.contains(&self.model) {
            format!("API · {}", self.name)
        } else {
            format!("API · {} ({})", self.name, self.model)
        }
    }

    pub fn public(&self) -> ProviderPublic {
        ProviderPublic {
            id: self.id.clone(),
            name: self.name.clone(),
            kind: "openai_compatible".into(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            api_key_masked: mask_key(&self.api_key),
            has_key: !self.api_key.is_empty(),
            source: self.source.clone(),
            created_ms: self.created_ms,
        }
    }
}

/// Vista pública (sin clave en claro).
#[derive(Clone, Debug, Serialize)]
pub struct ProviderPublic {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub base_url: String,
    pub model: String,
    pub api_key_masked: String,
    pub has_key: bool,
    pub source: String,
    pub created_ms: u64,
}

/// Enmascara una clave: prefijo + `…` + 4 últimos. Nunca devuelve la clave.
pub fn mask_key(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.is_empty() {
        return String::new();
    }
    if chars.len() <= 12 {
        return "••••".into();
    }
    let prefix_len = if key.starts_with("obk1.") { 7 } else { 4 };
    let prefix: String = chars[..prefix_len].iter().collect();
    let suffix: String = chars[chars.len() - 4..].iter().collect();
    format!("{prefix}…{suffix}")
}

/// Normaliza la URL base: quita barras finales y rutas de endpoint conocidas;
/// si hay `/v1` se corta ahí. Sin `/v1` se deja la ruta tal cual y se añade
/// `/v1` solo si la ruta está vacía.
pub fn normalize_base_url(raw: &str) -> Result<String, String> {
    let s = raw.trim().trim_end_matches('/');
    if s.is_empty() {
        return Err("falta la URL base".into());
    }
    let lower = s.to_ascii_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
        return Err(format!("URL base inválida (usa http:// o https://): «{s}»"));
    }
    let scheme_end = s.find("://").map(|i| i + 3).unwrap_or(0);
    let path_start = s[scheme_end..]
        .find('/')
        .map(|i| i + scheme_end)
        .unwrap_or(s.len());
    let host = &s[scheme_end..path_start];
    if host.is_empty() {
        return Err(format!("URL base sin host: «{s}»"));
    }
    let path = &s[path_start..];
    // Cortar query/fragment.
    let path = path.split(['?', '#']).next().unwrap_or("");
    if let Some(i) = find_v1(path) {
        return Ok(format!("{}{}", &s[..path_start], &path[..i + 3]));
    }
    let mut p = path.to_string();
    for suffix in ["/chat/completions", "/completions", "/models"] {
        if p.ends_with(suffix) {
            p.truncate(p.len() - suffix.len());
            break;
        }
    }
    let p = p.trim_end_matches('/');
    if p.is_empty() {
        Ok(format!("{}/v1", &s[..path_start]))
    } else {
        Ok(format!("{}{}", &s[..path_start], p))
    }
}

/// Índice de un segmento `/v1` completo en la ruta.
fn find_v1(path: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(i) = path[from..].find("/v1") {
        let at = from + i;
        let end = at + 3;
        if end == path.len() || path.as_bytes()[end] == b'/' {
            return Some(at);
        }
        from = end;
    }
    None
}

// ------------------------------------------------------------------ curl

/// Resultado de interpretar un `curl`.
#[derive(Clone, Debug, Serialize, Default, PartialEq)]
pub struct ParsedCurl {
    pub url: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    /// `chat_completions` | `completions` | `models` | `other`.
    pub endpoint: String,
    pub stream: bool,
    pub method: String,
    pub name: String,
    pub warnings: Vec<String>,
}

/// Divide texto tipo shell en comandos (por salto de línea, `;`, `&&`, `||`, `|`)
/// y cada comando en palabras, expandiendo variables conocidas.
fn shell_commands(text: &str, vars: &mut HashMap<String, String>) -> Vec<Vec<String>> {
    // Unir continuaciones `\` + salto de línea (fuera de comillas simples también
    // es válido en bash; dentro de comillas simples el `\` es literal, pero en
    // la práctica los curl pegados no lo usan así).
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut text = String::with_capacity(normalized.len());
    for line in normalized.lines() {
        // `\` final (con o sin espacios detrás) = continuación de línea.
        let t = line.trim_end();
        match t.strip_suffix('\\') {
            Some(body) => {
                text.push_str(body);
                text.push(' ');
            }
            None => {
                text.push_str(line);
                text.push('\n');
            }
        }
    }
    let chars: Vec<char> = text.chars().collect();
    let mut cmds: Vec<Vec<String>> = Vec::new();
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut has_word = false;
    let mut i = 0;
    let flush_word = |cur: &mut String, has: &mut bool, words: &mut Vec<String>| {
        if *has {
            words.push(std::mem::take(cur));
            *has = false;
        }
    };
    // Las asignaciones se aplican al terminar cada comando (export A=… ; curl $A).
    let finish_cmd = |words: &mut Vec<String>,
                      cmds: &mut Vec<Vec<String>>,
                      vars: &mut HashMap<String, String>| {
        if words.is_empty() {
            return;
        }
        let w = std::mem::take(words);
        let mut idx = 0;
        if w[0] == "export" || w[0] == "set" || w[0] == "declare" {
            idx = 1;
        }
        let assigns: Vec<&String> = w[idx..].iter().collect();
        let all_assign = !assigns.is_empty()
            && assigns.iter().all(|a| {
                a.split_once('=')
                    .map(|(k, _)| is_var_name(k))
                    .unwrap_or(false)
            });
        if all_assign {
            for a in assigns {
                if let Some((k, v)) = a.split_once('=') {
                    vars.insert(k.to_string(), v.to_string());
                }
            }
        } else {
            cmds.push(w);
        }
    };
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\'' => {
                has_word = true;
                i += 1;
                while i < chars.len() && chars[i] != '\'' {
                    cur.push(chars[i]);
                    i += 1;
                }
                i += 1;
            }
            '"' => {
                has_word = true;
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        let n = chars[i + 1];
                        if matches!(n, '"' | '\\' | '$' | '`') {
                            cur.push(n);
                            i += 2;
                            continue;
                        }
                        if n == '\n' {
                            i += 2;
                            continue;
                        }
                    }
                    if chars[i] == '$' {
                        let (val, used) = expand_var(&chars[i..], vars);
                        if used > 0 {
                            cur.push_str(&val);
                            i += used;
                            continue;
                        }
                    }
                    cur.push(chars[i]);
                    i += 1;
                }
                i += 1;
            }
            '\\' => {
                if i + 1 < chars.len() {
                    if chars[i + 1] != '\n' {
                        cur.push(chars[i + 1]);
                        has_word = true;
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            '$' => {
                let (val, used) = expand_var(&chars[i..], vars);
                if used > 0 {
                    cur.push_str(&val);
                    has_word = true;
                    i += used;
                } else {
                    cur.push('$');
                    has_word = true;
                    i += 1;
                }
            }
            '#' if !has_word => {
                // Comentario hasta fin de línea.
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '\n' | ';' => {
                flush_word(&mut cur, &mut has_word, &mut words);
                finish_cmd(&mut words, &mut cmds, vars);
                i += 1;
            }
            '&' | '|' => {
                flush_word(&mut cur, &mut has_word, &mut words);
                finish_cmd(&mut words, &mut cmds, vars);
                i += 1;
                if i < chars.len() && (chars[i] == '&' || chars[i] == '|') {
                    i += 1;
                }
            }
            c if c.is_whitespace() => {
                flush_word(&mut cur, &mut has_word, &mut words);
                i += 1;
            }
            _ => {
                cur.push(c);
                has_word = true;
                i += 1;
            }
        }
    }
    flush_word(&mut cur, &mut has_word, &mut words);
    finish_cmd(&mut words, &mut cmds, vars);
    cmds
}

fn is_var_name(k: &str) -> bool {
    !k.is_empty()
        && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !k.chars().next().unwrap().is_ascii_digit()
}

/// Expande `$VAR` / `${VAR}` al inicio de `s`. Devuelve (valor, chars usados).
/// Variables desconocidas se dejan literales (`$VAR`) para avisar después.
fn expand_var(s: &[char], vars: &HashMap<String, String>) -> (String, usize) {
    if s.len() < 2 || s[0] != '$' {
        return (String::new(), 0);
    }
    if s[1] == '{' {
        if let Some(end) = s.iter().position(|&c| c == '}') {
            let inner: String = s[2..end].iter().collect();
            let name: String = inner.split([':', '-']).next().unwrap_or("").to_string();
            if is_var_name(&name) {
                let val = vars
                    .get(&name)
                    .cloned()
                    .unwrap_or_else(|| format!("${{{name}}}"));
                return (val, end + 1);
            }
        }
        return (String::new(), 0);
    }
    let mut n = 1;
    while n < s.len() && (s[n].is_ascii_alphanumeric() || s[n] == '_') {
        n += 1;
    }
    if n == 1 {
        return (String::new(), 0);
    }
    let name: String = s[1..n].iter().collect();
    let val = vars
        .get(&name)
        .cloned()
        .unwrap_or_else(|| format!("${name}"));
    (val, n)
}

/// Interpreta un `curl` (docker-llm u otra API compatible con OpenAI).
pub fn parse_curl(text: &str) -> Result<ParsedCurl, String> {
    let mut vars: HashMap<String, String> = HashMap::new();
    let cmds = shell_commands(text, &mut vars);
    let curl = cmds
        .iter()
        .rev()
        .find(|w| {
            w.first()
                .map(|c| c == "curl" || c.ends_with("/curl") || c == "curl.exe")
                .unwrap_or(false)
        })
        .ok_or_else(|| "no se encontró ningún comando «curl» en el texto pegado".to_string())?;

    let mut out = ParsedCurl {
        method: "GET".into(),
        ..Default::default()
    };
    let mut headers: Vec<String> = Vec::new();
    let mut data: Option<String> = None;
    let mut url: Option<String> = None;
    let mut explicit_method: Option<String> = None;
    let mut i = 1;
    // Opciones sin argumento más comunes (se ignoran).
    let takes_arg = [
        "-o",
        "--output",
        "-u",
        "--user",
        "-A",
        "--user-agent",
        "-e",
        "--referer",
        "-m",
        "--max-time",
        "--connect-timeout",
        "-w",
        "--write-out",
        "--retry",
        "-b",
        "--cookie",
        "-c",
        "--cookie-jar",
        "-x",
        "--proxy",
        "--cacert",
        "--cert",
        "--key",
        "-F",
        "--form",
        "--resolve",
        "--limit-rate",
    ];
    while i < curl.len() {
        let a = &curl[i];
        let next = curl.get(i + 1).cloned();
        let mut take = |flag: &str, value: &mut Option<String>| -> bool {
            if a == flag {
                *value = next.clone();
                i += 1;
                true
            } else if let Some(rest) = a.strip_prefix(&format!("{flag}=")) {
                if flag.starts_with("--") {
                    *value = Some(rest.to_string());
                    true
                } else {
                    false
                }
            } else if flag.len() == 2
                && !flag.starts_with("--")
                && a.starts_with(flag)
                && a.len() > 2
            {
                *value = Some(a[2..].to_string());
                true
            } else {
                false
            }
        };
        let mut v: Option<String> = None;
        if take("-H", &mut v) || take("--header", &mut v) {
            if let Some(h) = v {
                headers.push(h);
            }
        } else if take("-d", &mut v)
            || take("--data", &mut v)
            || take("--data-raw", &mut v)
            || take("--data-binary", &mut v)
            || take("--data-ascii", &mut v)
            || take("--json", &mut v)
        {
            if let Some(d) = v {
                data = Some(match data.take() {
                    Some(prev) => format!("{prev}&{d}"),
                    None => d,
                });
            }
        } else if take("-X", &mut v) || take("--request", &mut v) {
            explicit_method = v;
        } else if take("--url", &mut v) {
            url = v;
        } else if a == "-N" || a == "--no-buffer" {
            out.stream = true;
        } else if takes_arg.contains(&a.as_str()) {
            i += 1;
        } else if a.starts_with('-') {
            // Banderas combinadas (-sS, -sN…): detectar N.
            if !a.starts_with("--") && a.contains('N') {
                out.stream = true;
            }
        } else if url.is_none() {
            url = Some(a.clone());
        }
        i += 1;
    }

    let url = url.ok_or_else(|| "el curl no incluye ninguna URL".to_string())?;
    if url.contains('$') {
        out.warnings.push(format!(
            "la URL usa una variable sin definir ({url}); añade la línea «export BASE=…» o edita la URL base"
        ));
    }
    out.url = url.clone();

    for h in &headers {
        let (name, value) = match h.split_once(':') {
            Some((n, v)) => (n.trim().to_ascii_lowercase(), v.trim().to_string()),
            None => continue,
        };
        match name.as_str() {
            "authorization" => {
                let v = value.trim();
                let key = v
                    .strip_prefix("Bearer ")
                    .or_else(|| v.strip_prefix("bearer "))
                    .or_else(|| v.strip_prefix("BEARER "))
                    .unwrap_or(v)
                    .trim();
                out.api_key = key.to_string();
            }
            "x-api-key" | "api-key" if out.api_key.is_empty() => {
                out.api_key = value;
            }
            _ => {}
        }
    }
    if out.api_key.contains('$') {
        out.warnings.push(
            "la API key es una variable sin definir; pega también la línea «export API_KEY=…» o escríbela a mano"
                .into(),
        );
    } else if out.api_key.is_empty() {
        out.warnings
            .push("el curl no trae cabecera «Authorization: Bearer …»; escribe la API key".into());
    }

    if let Some(d) = &data {
        out.method = "POST".into();
        match serde_json::from_str::<Value>(d) {
            Ok(body) => {
                if let Some(m) = body.get("model").and_then(Value::as_str) {
                    out.model = m.to_string();
                }
                if body.get("stream").and_then(Value::as_bool) == Some(true) {
                    out.stream = true;
                }
            }
            Err(_) => out
                .warnings
                .push("el cuerpo (-d) no es JSON válido; se ignoró".into()),
        }
    }
    if let Some(m) = explicit_method {
        out.method = m.to_ascii_uppercase();
    }

    // Ruta: endpoint y modelo en la ruta (/v1/models/<id>/…).
    let scheme_end = url.find("://").map(|i| i + 3).unwrap_or(0);
    let path = url[scheme_end..]
        .find('/')
        .map(|i| &url[scheme_end + i..])
        .unwrap_or("");
    let path = path.split(['?', '#']).next().unwrap_or("");
    let after_v1 = find_v1(path).map(|i| &path[i + 3..]).unwrap_or(path);
    out.endpoint = if after_v1.ends_with("/chat/completions") {
        "chat_completions"
    } else if after_v1.ends_with("/completions") {
        "completions"
    } else if after_v1.starts_with("/models") {
        "models"
    } else {
        "other"
    }
    .into();
    if let Some(rest) = after_v1.strip_prefix("/models/") {
        let id = rest.split('/').next().unwrap_or("");
        if !id.is_empty() && out.model.is_empty() {
            out.model = id.to_string();
        }
        if !id.is_empty() && rest.contains('/') && out.endpoint == "models" {
            out.endpoint = "other".into();
        }
    }
    if !url.contains('$') {
        out.base_url = normalize_base_url(&url)?;
    }
    if out.model.is_empty() {
        out.warnings.push(
            "el curl no indica modelo; pulsa «Probar conexión» para listar modelos o escríbelo"
                .into(),
        );
    }
    out.name = suggest_name(&out.base_url, &out.model);
    Ok(out)
}

/// Nombre sugerido: primer label del host (sin `-production`) + modelo.
pub fn suggest_name(base_url: &str, model: &str) -> String {
    let host = base_url
        .split("://")
        .nth(1)
        .unwrap_or(base_url)
        .split(['/', ':'])
        .next()
        .unwrap_or("");
    let label = host.split('.').next().unwrap_or(host);
    let label = label.strip_suffix("-production").unwrap_or(label);
    let label = if label.is_empty() { "api" } else { label };
    if model.is_empty() {
        label.to_string()
    } else {
        format!("{label} · {model}")
    }
}

// ------------------------------------------------------------------ store

/// Persistencia de proveedores + proveedor activo.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProviderStore {
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    /// `gemma_local` o id de proveedor.
    #[serde(default = "local_id")]
    pub active: String,
    /// Proveedor desde env (no se persiste: la clave vive en la variable).
    #[serde(skip)]
    pub env_provider: Option<ProviderConfig>,
    #[serde(skip)]
    pub path: Option<PathBuf>,
}

fn local_id() -> String {
    LOCAL_ID.into()
}

/// Ruta del JSON de proveedores. Env `LLM_PROVIDERS_FILE`; si no,
/// `data/llm_providers.json` (en Docker `/app/data/…`).
pub fn default_store_path() -> PathBuf {
    if let Ok(p) = std::env::var("LLM_PROVIDERS_FILE") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    PathBuf::from("data/llm_providers.json")
}

impl ProviderStore {
    /// Store en memoria (tests): sin archivo, sin env.
    pub fn in_memory() -> Self {
        Self {
            active: LOCAL_ID.into(),
            ..Default::default()
        }
    }

    /// Carga desde archivo (si existe) + bootstrap env.
    pub fn load(path: &Path) -> Self {
        let mut s = match std::fs::read_to_string(path) {
            Ok(txt) => match serde_json::from_str::<ProviderStore>(&txt) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, path = %path.display(), "llm_providers.json inválido; se ignora");
                    Self::in_memory()
                }
            },
            Err(_) => Self::in_memory(),
        };
        s.path = Some(path.to_path_buf());
        s.env_provider = provider_from_env();
        if let Some(env) = &s.env_provider {
            let force = std::env::var("LLM_API_ACTIVE")
                .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
                .unwrap_or(false);
            if force {
                s.active = env.id.clone();
            }
        }
        if s.active.is_empty() || (s.active != LOCAL_ID && s.get(&s.active).is_none()) {
            s.active = LOCAL_ID.into();
        }
        tracing::info!(
            providers = s.providers.len(),
            env = s.env_provider.is_some(),
            active = %s.active,
            "proveedores LLM cargados"
        );
        s
    }

    pub fn load_default() -> Self {
        Self::load(&default_store_path())
    }

    /// Todos (env primero).
    pub fn all(&self) -> Vec<&ProviderConfig> {
        self.env_provider
            .iter()
            .chain(self.providers.iter())
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<&ProviderConfig> {
        self.all().into_iter().find(|p| p.id == id)
    }

    /// Proveedor externo activo (None = Gemma local).
    pub fn active_external(&self) -> Option<ProviderConfig> {
        if self.active == LOCAL_ID {
            return None;
        }
        self.get(&self.active).cloned()
    }

    pub fn set_active(&mut self, id: &str) -> Result<(), String> {
        let id = id.trim();
        if id.is_empty() || id == LOCAL_ID {
            self.active = LOCAL_ID.into();
        } else if self.get(id).is_some() {
            self.active = id.to_string();
        } else {
            return Err(format!("no existe el proveedor «{id}»"));
        }
        self.save()
    }

    /// Crea o actualiza. Si `api_key` viene vacía y el id existe, conserva la clave.
    pub fn upsert(&mut self, input: ProviderInput) -> Result<ProviderConfig, String> {
        let base_url = normalize_base_url(&input.base_url)?;
        let model = input.model.trim().to_string();
        let id = input
            .id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        if id.as_deref() == Some(LOCAL_ID) || id.as_deref() == Some(ENV_ID) {
            return Err("ese proveedor no se puede editar desde la app".into());
        }
        let existing = id
            .as_deref()
            .and_then(|i| self.providers.iter().position(|p| p.id == i));
        let key_in = input.api_key.trim().to_string();
        if key_in.contains('…') || key_in.contains('•') {
            return Err("la API key parece enmascarada; pega la clave completa".into());
        }
        let name = {
            let n = input.name.trim();
            if n.is_empty() {
                suggest_name(&base_url, &model)
            } else {
                n.chars().take(80).collect()
            }
        };
        match existing {
            Some(pos) => {
                let p = &mut self.providers[pos];
                p.name = name;
                p.base_url = base_url;
                p.model = model;
                if !key_in.is_empty() {
                    p.api_key = key_in;
                }
                let out = p.clone();
                self.save()?;
                Ok(out)
            }
            None => {
                if key_in.is_empty() {
                    return Err("falta la API key".into());
                }
                let p = ProviderConfig {
                    id: id.unwrap_or_else(new_id),
                    name,
                    base_url,
                    api_key: key_in,
                    model,
                    created_ms: now_ms(),
                    source: "ui".into(),
                };
                self.providers.push(p.clone());
                self.save()?;
                Ok(p)
            }
        }
    }

    pub fn remove(&mut self, id: &str) -> Result<bool, String> {
        if id == LOCAL_ID {
            return Err("Gemma local no se puede borrar".into());
        }
        if self.env_provider.as_ref().map(|p| p.id.as_str()) == Some(id) {
            return Err(
                "este proveedor viene de variables de entorno (LLM_API_*); quítalo allí".into(),
            );
        }
        let before = self.providers.len();
        self.providers.retain(|p| p.id != id);
        let removed = self.providers.len() != before;
        if removed && self.active == id {
            self.active = LOCAL_ID.into();
        }
        self.save()?;
        Ok(removed)
    }

    /// Escribe el JSON (0600 en unix). Sin ruta (tests) no hace nada.
    pub fn save(&self) -> Result<(), String> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)
                    .map_err(|e| format!("no se pudo crear {}: {e}", dir.display()))?;
            }
        }
        let txt = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, txt).map_err(|e| format!("no se pudo guardar: {e}"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
        }
        std::fs::rename(&tmp, path).map_err(|e| format!("no se pudo guardar: {e}"))?;
        Ok(())
    }

    /// Lista pública (claves enmascaradas).
    pub fn public_list(&self) -> Vec<ProviderPublic> {
        self.all().into_iter().map(ProviderConfig::public).collect()
    }
}

/// Entrada de creación/edición (POST /api/llm/providers).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ProviderInput {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub model: String,
}

fn provider_from_env() -> Option<ProviderConfig> {
    let base = std::env::var("LLM_API_BASE").ok()?;
    let base = normalize_base_url(&base).ok()?;
    let key = std::env::var("LLM_API_KEY").unwrap_or_default();
    let model = std::env::var("LLM_API_MODEL").unwrap_or_default();
    let name = std::env::var("LLM_API_NAME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("{} (env)", suggest_name(&base, &model)));
    Some(ProviderConfig {
        id: ENV_ID.into(),
        name,
        base_url: base,
        api_key: key.trim().to_string(),
        model: model.trim().to_string(),
        created_ms: 0,
        source: "env".into(),
    })
}

fn new_id() -> String {
    let u = uuid::Uuid::new_v4().simple().to_string();
    format!("api_{}", &u[..10])
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ------------------------------------------------------------------ cliente

/// Respuesta de chat de la API externa.
#[derive(Clone, Debug, Serialize)]
pub struct ExternalReply {
    pub text: String,
    pub model: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub seconds: f64,
}

/// Mensaje OpenAI (`system` | `user` | `assistant`).
#[derive(Clone, Debug, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Parámetros de generación externos.
#[derive(Clone, Copy, Debug)]
pub struct ExternalGenConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

/// Cliente bloqueante OpenAI-compatible (ureq). Llamar desde `spawn_blocking`.
pub struct OpenAiClient {
    cfg: ProviderConfig,
    agent: ureq::Agent,
}

impl OpenAiClient {
    pub fn new(cfg: &ProviderConfig, timeout: Duration) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(timeout.min(Duration::from_secs(15)))
            .timeout(timeout)
            .user_agent("neuro-geometrica/agentic_web")
            .build();
        Self {
            cfg: cfg.clone(),
            agent,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.cfg.base_url.trim_end_matches('/'), path)
    }

    fn auth(&self, req: ureq::Request) -> ureq::Request {
        if self.cfg.api_key.is_empty() {
            req
        } else {
            req.set("Authorization", &format!("Bearer {}", self.cfg.api_key))
        }
    }

    /// `GET /v1/models` → ids.
    pub fn list_models(&self) -> Result<Vec<String>, String> {
        let req = self.auth(self.agent.get(&self.url("/models")));
        let resp = req.call().map_err(|e| map_ureq_error(e, &self.cfg))?;
        let v: Value = resp
            .into_json()
            .map_err(|e| format!("respuesta de /v1/models no es JSON: {e}"))?;
        let ids = v
            .get("data")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        Ok(ids)
    }

    /// `POST /v1/chat/completions` (sin streaming).
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        gen: ExternalGenConfig,
    ) -> Result<ExternalReply, String> {
        let t0 = Instant::now();
        let mut model = self.cfg.model.clone();
        if model.is_empty() {
            model = self
                .list_models()?
                .into_iter()
                .next()
                .ok_or_else(|| "la API no lista ningún modelo; indica el modelo".to_string())?;
        }
        let body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": gen.max_tokens,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "stream": false,
        });
        let req = self
            .auth(self.agent.post(&self.url("/chat/completions")))
            .set("Content-Type", "application/json");
        let resp = req
            .send_json(body)
            .map_err(|e| map_ureq_error(e, &self.cfg))?;
        let v: Value = resp
            .into_json()
            .map_err(|e| format!("respuesta de chat no es JSON: {e}"))?;
        let msg = v.pointer("/choices/0/message").ok_or_else(|| {
            format!(
                "respuesta sin choices[0].message: {}",
                snippet(&v.to_string())
            )
        })?;
        let mut text = msg
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        text = strip_think(&text);
        if text.trim().is_empty() {
            if let Some(r) = msg.get("reasoning_content").and_then(Value::as_str) {
                text = r.trim().to_string();
            }
        }
        Ok(ExternalReply {
            text: text.trim().to_string(),
            model: v
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(&model)
                .to_string(),
            prompt_tokens: v
                .pointer("/usage/prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            completion_tokens: v
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            seconds: t0.elapsed().as_secs_f64(),
        })
    }
}

/// Quita bloques `<think>…</think>` (modelos razonadores).
pub fn strip_think(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        match rest.find("<think>") {
            Some(i) => {
                out.push_str(&rest[..i]);
                match rest[i..].find("</think>") {
                    Some(j) => rest = &rest[i + j + "</think>".len()..],
                    None => break,
                }
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
    }
    out.trim().to_string()
}

fn snippet(s: &str) -> String {
    let t: String = s.chars().take(240).collect();
    if s.chars().count() > 240 {
        format!("{t}…")
    } else {
        t
    }
}

/// Error legible (sin la clave). 401 → mensaje explícito.
fn map_ureq_error(e: ureq::Error, cfg: &ProviderConfig) -> String {
    let redact = |s: String| {
        if cfg.api_key.len() >= 6 {
            s.replace(&cfg.api_key, "***")
        } else {
            s
        }
    };
    match e {
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            let detail = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| {
                    let m = v
                        .pointer("/error/message")
                        .or_else(|| v.pointer("/detail"))
                        .or_else(|| v.get("error"))
                        .cloned()?;
                    Some(match m {
                        Value::String(s) => s,
                        other => other.to_string(),
                    })
                })
                .unwrap_or_else(|| snippet(&body));
            let detail = redact(detail);
            match code {
                401 => format!(
                    "401 no autorizado: la API key no es válida, está revocada o caducó ({detail})"
                ),
                403 => format!("403 prohibido: la API key no tiene acceso ({detail})"),
                404 => format!("404 no encontrado: revisa la URL base y el modelo ({detail})"),
                _ => format!("HTTP {code}: {detail}"),
            }
        }
        ureq::Error::Transport(t) => {
            let msg = redact(t.to_string());
            let lower = msg.to_ascii_lowercase();
            if lower.contains("timed out") || lower.contains("timeout") {
                format!(
                    "tiempo de espera agotado hablando con {} ({msg})",
                    cfg.base_url
                )
            } else {
                format!("no se pudo conectar con {}: {msg}", cfg.base_url)
            }
        }
    }
}

/// Resultado de «Probar conexión».
#[derive(Clone, Debug, Serialize)]
pub struct ProviderTestReport {
    pub ok: bool,
    pub base_url: String,
    pub model: String,
    pub models_ok: bool,
    pub models: Vec<String>,
    pub model_listed: Option<bool>,
    pub models_ms: u64,
    pub chat_ok: bool,
    pub chat_ms: u64,
    pub reply_preview: Option<String>,
    pub error: Option<String>,
}

/// Prueba: `GET /models` + chat mínimo. Bloqueante.
pub fn test_provider(cfg: &ProviderConfig, timeout: Duration) -> ProviderTestReport {
    let client = OpenAiClient::new(cfg, timeout);
    let mut rep = ProviderTestReport {
        ok: false,
        base_url: cfg.base_url.clone(),
        model: cfg.model.clone(),
        models_ok: false,
        models: Vec::new(),
        model_listed: None,
        models_ms: 0,
        chat_ok: false,
        chat_ms: 0,
        reply_preview: None,
        error: None,
    };
    let t0 = Instant::now();
    match client.list_models() {
        Ok(ids) => {
            rep.models_ok = true;
            if !cfg.model.is_empty() {
                rep.model_listed = Some(ids.iter().any(|m| m == &cfg.model));
            }
            rep.models = ids;
        }
        Err(e) => {
            rep.models_ms = t0.elapsed().as_millis() as u64;
            rep.error = Some(format!("GET /v1/models: {e}"));
            // 401 en /models ya es concluyente: no intentar chat.
            return rep;
        }
    }
    rep.models_ms = t0.elapsed().as_millis() as u64;
    let mut chat_cfg = cfg.clone();
    if chat_cfg.model.is_empty() {
        if let Some(first) = rep.models.first() {
            chat_cfg.model = first.clone();
            rep.model = first.clone();
        }
    }
    let t1 = Instant::now();
    let r = OpenAiClient::new(&chat_cfg, timeout).chat(
        &[ChatMessage::new("user", "Responde solo: ok")],
        ExternalGenConfig {
            max_tokens: 8,
            temperature: 0.0,
            top_p: 1.0,
        },
    );
    rep.chat_ms = t1.elapsed().as_millis() as u64;
    match r {
        Ok(reply) => {
            rep.chat_ok = true;
            rep.reply_preview = Some(snippet(&reply.text));
            rep.ok = true;
        }
        Err(e) => rep.error = Some(format!("POST /v1/chat/completions: {e}")),
    }
    rep
}

/// Config de generación externa para chat crudo (mismas env que Gemma).
pub fn external_raw_config() -> ExternalGenConfig {
    let c = crate::web::llm_periphery::raw_chat_config(0);
    ExternalGenConfig {
        max_tokens: env_usize("EXTERNAL_CHAT_MAX_TOKENS", c.max_tokens.max(256)),
        temperature: c.temperature,
        top_p: c.top_p,
    }
}

/// Config externa del decoder del campo (respuestas cortas).
pub fn external_decoder_config() -> ExternalGenConfig {
    let c = crate::web::llm_periphery::field_decoder_config(0);
    ExternalGenConfig {
        max_tokens: env_usize("EXTERNAL_DECODER_MAX_TOKENS", c.max_tokens.max(160)),
        temperature: c.temperature,
        top_p: c.top_p,
    }
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
        .clamp(1, 4096)
}

/// Mensajes del **decoder del campo** para API externa: el campo ya decidió el
/// estado; el LLM solo lo verbaliza.
pub fn field_decoder_messages(user_msg: &str, field_state: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage::new(
            "system",
            "Eres el decoder de un modelo de campo (líquido/CDT/RQM). No inventes el estado: \
             verbaliza en español, en 1-3 frases, la respuesta al usuario y cierra mencionando \
             el estado del campo que se te da.",
        ),
        ChatMessage::new(
            "user",
            format!("{user_msg}\n\n(Estado del campo: {field_state})"),
        ),
    ]
}

/// Mensajes de chat crudo con historial (pares user/assistant).
pub fn raw_chat_messages(history: &[(String, String)], input: &str) -> Vec<ChatMessage> {
    let mut v = Vec::with_capacity(history.len() * 2 + 1);
    for (u, a) in history {
        v.push(ChatMessage::new("user", u.clone()));
        v.push(ChatMessage::new("assistant", a.clone()));
    }
    v.push(ChatMessage::new("user", input));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "obk1.AbCdEf0123456789.xyzXYZ987";

    #[test]
    fn parse_readme_chat_curl() {
        let c = format!(
            "curl https://docker-llm-production.up.railway.app/v1/chat/completions \\\n  -H \"Authorization: Bearer {KEY}\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{{\"model\":\"qwen-qwen3.5-2b-q4-k-m\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hola\"}}]}}'"
        );
        let p = parse_curl(&c).unwrap();
        assert_eq!(
            p.base_url,
            "https://docker-llm-production.up.railway.app/v1"
        );
        assert_eq!(p.api_key, KEY);
        assert_eq!(p.model, "qwen-qwen3.5-2b-q4-k-m");
        assert_eq!(p.endpoint, "chat_completions");
        assert!(!p.stream);
        assert_eq!(p.method, "POST");
        assert_eq!(p.name, "docker-llm · qwen-qwen3.5-2b-q4-k-m");
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn parse_stream_curl_with_n() {
        let c = format!(
            "curl -N https://docker-llm-production.up.railway.app/v1/chat/completions \\\n  -H \"Authorization: Bearer {KEY}\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{{\"model\":\"m1\",\"stream\":true,\"messages\":[{{\"role\":\"user\",\"content\":\"Hola\"}}]}}'"
        );
        let p = parse_curl(&c).unwrap();
        assert!(p.stream);
        assert_eq!(p.model, "m1");
        assert_eq!(p.api_key, KEY);
        assert_eq!(
            p.base_url,
            "https://docker-llm-production.up.railway.app/v1"
        );
    }

    #[test]
    fn parse_per_model_route_with_exports() {
        let c = format!(
            "export BASE=https://docker-llm-production.up.railway.app   # o http://localhost:8080\nexport API_KEY=\"{KEY}\"   # la que te dio Generar API key\n\n# Ruta propia del modelo\ncurl $BASE/v1/models/qwen-qwen3.5-2b-q4-k-m/chat/completions \\\n  -H \"Authorization: Bearer $API_KEY\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hola\"}}]}}'"
        );
        let p = parse_curl(&c).unwrap();
        assert_eq!(
            p.base_url,
            "https://docker-llm-production.up.railway.app/v1"
        );
        assert_eq!(p.api_key, KEY);
        assert_eq!(p.model, "qwen-qwen3.5-2b-q4-k-m");
        assert_eq!(p.endpoint, "chat_completions");
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn parse_models_listing_braced_vars_and_long_flags() {
        let c = format!(
            "BASE='http://localhost:8080'; API_KEY={KEY}\ncurl --silent \"${{BASE}}/v1/models\" --header \"Authorization: Bearer ${{API_KEY}}\""
        );
        let p = parse_curl(&c).unwrap();
        assert_eq!(p.base_url, "http://localhost:8080/v1");
        assert_eq!(p.api_key, KEY);
        assert_eq!(p.endpoint, "models");
        assert_eq!(p.method, "GET");
        assert!(p.model.is_empty());
        assert!(p.warnings.iter().any(|w| w.contains("modelo")));
    }

    #[test]
    fn parse_models_by_id_route() {
        let p = parse_curl(&format!(
            "curl https://h.example.com/v1/models/foo-7b -H 'Authorization: Bearer {KEY}'"
        ))
        .unwrap();
        assert_eq!(p.model, "foo-7b");
        assert_eq!(p.endpoint, "models");
    }

    #[test]
    fn parse_completions_data_raw_and_equals_forms() {
        let c = format!(
            "curl --url=https://api.example.com/v1/completions --header=\"Authorization: Bearer {KEY}\" --data-raw '{{\"model\":\"llama\",\"prompt\":\"Érase una vez\",\"max_tokens\":40}}'"
        );
        let p = parse_curl(&c).unwrap();
        assert_eq!(p.endpoint, "completions");
        assert_eq!(p.model, "llama");
        assert_eq!(p.base_url, "https://api.example.com/v1");
        assert_eq!(p.api_key, KEY);
    }

    #[test]
    fn parse_data_long_flag_double_quoted_json() {
        let c = format!(
            "curl -s -X POST \"https://x.io/v1/chat/completions\" -H \"authorization: bearer {KEY}\" --data \"{{\\\"model\\\":\\\"q\\\",\\\"messages\\\":[]}}\""
        );
        let p = parse_curl(&c).unwrap();
        assert_eq!(p.model, "q");
        assert_eq!(p.api_key, KEY);
        assert_eq!(p.method, "POST");
    }

    #[test]
    fn parse_windows_crlf_and_trailing_spaces_after_backslash() {
        let c = format!(
            "curl https://x.io/v1/chat/completions \\  \r\n  -H \"Authorization: Bearer {KEY}\" \\\r\n  -d '{{\"model\":\"z\"}}'"
        );
        let p = parse_curl(&c).unwrap();
        assert_eq!(p.model, "z");
        assert_eq!(p.api_key, KEY);
    }

    #[test]
    fn parse_undefined_vars_warn() {
        let p = parse_curl(
            "curl $BASE/v1/chat/completions -H \"Authorization: Bearer $API_KEY\" -d '{\"model\":\"m\"}'",
        )
        .unwrap();
        assert!(p.base_url.is_empty());
        assert!(p.warnings.iter().any(|w| w.contains("BASE")));
        assert!(p.warnings.iter().any(|w| w.contains("API_KEY")));
        assert_eq!(p.model, "m");
    }

    #[test]
    fn parse_rejects_non_curl() {
        assert!(parse_curl("hola mundo").is_err());
        assert!(parse_curl("curl -H 'a: b'").is_err());
    }

    #[test]
    fn normalize_base_variants() {
        assert_eq!(
            normalize_base_url("https://a.b/v1/").unwrap(),
            "https://a.b/v1"
        );
        assert_eq!(normalize_base_url("https://a.b").unwrap(), "https://a.b/v1");
        assert_eq!(
            normalize_base_url("http://localhost:9/v1/models/x/chat/completions").unwrap(),
            "http://localhost:9/v1"
        );
        assert_eq!(
            normalize_base_url("https://a.b/openai/chat/completions").unwrap(),
            "https://a.b/openai"
        );
        assert_eq!(
            normalize_base_url("https://a.b/v10/x").unwrap(),
            "https://a.b/v10/x"
        );
        assert!(normalize_base_url("ftp://a").is_err());
        assert!(normalize_base_url("").is_err());
    }

    #[test]
    fn mask_never_reveals_key() {
        let m = mask_key(KEY);
        assert!(m.starts_with("obk1.Ab"));
        assert!(m.ends_with("Z987"));
        assert!(!m.contains("0123456789"));
        assert_eq!(mask_key("short"), "••••");
        assert_eq!(mask_key(""), "");
    }

    #[test]
    fn strip_think_blocks() {
        assert_eq!(strip_think("<think>x</think>\nHola"), "Hola");
        assert_eq!(strip_think("a<think>b</think>c"), "ac");
        assert_eq!(strip_think("sin think"), "sin think");
    }

    fn input(name: &str, key: &str) -> ProviderInput {
        ProviderInput {
            id: None,
            name: name.into(),
            base_url: "https://x.io/v1/".into(),
            api_key: key.into(),
            model: "m".into(),
        }
    }

    #[test]
    fn store_upsert_select_remove_and_persist() {
        let dir = std::env::temp_dir().join(format!("llm_store_{}", now_ms()));
        let path = dir.join("p.json");
        let mut s = ProviderStore::load(&path);
        s.env_provider = None; // aislar de env
        assert_eq!(s.active, LOCAL_ID);
        assert!(s.active_external().is_none());
        let p = s.upsert(input("uno", KEY)).unwrap();
        assert_eq!(p.base_url, "https://x.io/v1");
        s.set_active(&p.id).unwrap();
        assert_eq!(s.active_external().unwrap().id, p.id);
        assert!(s.set_active("nope").is_err());
        // Editar sin clave conserva la clave.
        let mut edit = input("uno-bis", "");
        edit.id = Some(p.id.clone());
        let e = s.upsert(edit).unwrap();
        assert_eq!(e.api_key, KEY);
        assert_eq!(e.name, "uno-bis");
        // Clave enmascarada rechazada.
        let mut bad = input("x", "obk1.Ab…Z987");
        bad.id = Some(p.id.clone());
        assert!(s.upsert(bad).is_err());
        // Persistencia (0600) y recarga.
        let txt = std::fs::read_to_string(&path).unwrap();
        assert!(txt.contains(&p.id));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let s2 = ProviderStore::load(&path);
        assert_eq!(s2.active, p.id);
        assert_eq!(s2.providers.len(), 1);
        // Público enmascarado.
        let pub_json = serde_json::to_string(&s2.public_list()).unwrap();
        assert!(!pub_json.contains(KEY));
        // Borrar el activo → vuelve a Gemma local.
        assert!(s.remove(&p.id).unwrap());
        assert_eq!(s.active, LOCAL_ID);
        assert!(s.remove(LOCAL_ID).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_new_requires_key() {
        let mut s = ProviderStore::in_memory();
        assert!(s.upsert(input("x", "")).is_err());
        assert!(s.upsert(input("x", KEY)).is_ok());
    }

    #[test]
    fn messages_shapes() {
        let m = raw_chat_messages(&[("a".into(), "b".into())], "c");
        assert_eq!(m.len(), 3);
        assert_eq!(m[1].role, "assistant");
        let d = field_decoder_messages("hola", "concepto 1→2");
        assert_eq!(d[0].role, "system");
        assert!(d[1].content.contains("concepto 1→2"));
    }
}
