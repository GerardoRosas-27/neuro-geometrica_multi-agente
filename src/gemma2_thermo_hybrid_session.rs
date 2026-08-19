//! Persistencia de sesiones del chat híbrido Gemma 2 + motor CTP.

use crate::gemma2_thermo_hybrid_llm::{
    Gemma2ThermoHybridConfig, Gemma2ThermoHybridLearnedState, Gemma2ThermoHybridLlm,
};
use crate::native_checkpoint::atomic_write;
use crate::native_gemma2::QuantizedGemma2;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DEFAULT_CHAT_ROOT: &str = "data/native_gemma2_thermo_chats";
const SESSION_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermoHybridChatSession {
    pub version: u32,
    pub name: String,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
    pub turns: u64,
    pub history: Vec<(String, String)>,
    pub learned: Gemma2ThermoHybridLearnedState,
}

#[derive(Clone, Debug)]
pub struct ChatSessionLoadReport {
    pub name: String,
    pub path: PathBuf,
    pub resumed: bool,
    pub turns: u64,
    pub attractors: usize,
}

pub fn sanitize_chat_name(name: &str) -> Result<String, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("el nombre del chat no puede estar vacío".to_string());
    }
    if trimmed.len() > 64 {
        return Err("el nombre del chat debe tener como máximo 64 caracteres".to_string());
    }
    if !trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err(
            "el nombre del chat sólo puede contener letras, números, '-' o '_'".to_string(),
        );
    }
    Ok(trimmed.to_string())
}

pub fn chat_session_path(root: impl AsRef<Path>, name: &str) -> PathBuf {
    root.as_ref().join(format!("{name}.cdt"))
}

pub fn save_chat_session(
    path: &Path,
    name: &str,
    created_at_unix: u64,
    turns: u64,
    history: &[(String, String)],
    hybrid: &Gemma2ThermoHybridLlm,
) -> Result<(), String> {
    let session = ThermoHybridChatSession {
        version: SESSION_VERSION,
        name: name.to_string(),
        created_at_unix,
        updated_at_unix: unix_now(),
        turns,
        history: history.to_vec(),
        learned: hybrid.export_learned_state(),
    };
    let body = serde_json::to_vec_pretty(&session).map_err(|error| error.to_string())?;
    atomic_write(path, &body)
}

pub fn load_chat_session(path: &Path) -> Result<ThermoHybridChatSession, String> {
    let body = fs::read(path).map_err(|error| error.to_string())?;
    let session: ThermoHybridChatSession =
        serde_json::from_slice(&body).map_err(|error| error.to_string())?;
    if session.version != SESSION_VERSION {
        return Err(format!(
            "versión de sesión incompatible: {} (esperada {SESSION_VERSION})",
            session.version
        ));
    }
    Ok(session)
}

pub fn restore_hybrid_from_session(
    model: &QuantizedGemma2,
    config: Gemma2ThermoHybridConfig,
    session: &ThermoHybridChatSession,
) -> Result<Gemma2ThermoHybridLlm, String> {
    if session.learned.config.cdt_nodes != config.cdt_nodes
        || session.learned.config.rff_features_cap != config.rff_features_cap
        || session.learned.config.seed != config.seed
    {
        return Err(
            "la sesión guardada fue creada con otra configuración termodinámica (nodos/RFF/semilla)"
                .to_string(),
        );
    }
    let mut hybrid =
        Gemma2ThermoHybridLlm::for_gemma(model, config).map_err(|error| error.to_string())?;
    hybrid
        .apply_learned_state(&session.learned)
        .map_err(|error| error.to_string())?;
    Ok(hybrid)
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_rejects_invalid_names() {
        assert!(sanitize_chat_name("").is_err());
        assert!(sanitize_chat_name("../escape").is_err());
        assert_eq!(sanitize_chat_name("mi-chat_1").unwrap(), "mi-chat_1");
    }
}
