//! Periferia Lingüística Congelada (Salida Verbalizadora + Guardia de Fidelidad).
//!
//! Principio arquitectónico estricto:
//! - La **arquitectura METAFE-3S-Dial** decide la postura epistémica (`EpistemicStance`),
//!   la tesis central, la evidencia, el resultado de la refutación cruzada A vs B y la confianza.
//! - El **LLM nunca razona por su cuenta ni modifica los pesos internos**: únicamente verbaliza
//!   el `DecisionPacket` producido por la arquitectura.
//! - Tres backends intercambiables en caliente:
//!   1. `NativeVerbalizer` (sin dependencias externas, latencia <1 ms, 100% fiel).
//!   2. `LocalGemma` (reutiliza `crate::field_gemma_probe::FrozenGemma2Probe` en GGUF congelado).
//!   3. `OpenAiApi` (compatible con Ollama, LM Studio, vLLM, OpenAI, Groq, docker-llm vía `curl`
//!      pasando el cuerpo JSON por stdin para no exponer secretos en argumentos).
//! - **Guardia de Fidelidad (`verify_periphery_fidelity`)**: audita cada texto devuelto por un LLM
//!   periférico; si viola la postura (p.ej. inventa certeza cuando la arquitectura decidió `Abstain`
//!   o ignora una contradicción `ConflictDetected`), rechaza la alucinación y protege el veredicto.

use crate::field_gemma_probe::{FrozenGemma2Probe, RawGemmaHandle};
use crate::native_gemma2_runtime::Gemma2GenerationConfig;
use crate::neuro_dialogo::belief::EpistemicStance;
use crate::neuro_dialogo::core3s::NeuromodSignal;
use crate::neuro_dialogo::encoder::{normalize_es, DialogueIntent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

/// Paquete de decisión cerrado que la arquitectura entrega a la periferia lingüística congelada.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionPacket {
    pub user_message: String,
    pub intent: DialogueIntent,
    pub stance: EpistemicStance,
    pub core_thesis: String,
    pub supporting_evidence: Vec<String>,
    pub refutation_notes: Vec<String>,
    pub confidence: f64,
    pub free_energy_delta: f64,
    pub cycle_err_a: f64,
    pub cycle_err_b: f64,
    pub counterfactual_sens: f64,
    pub refute_vuln: f64,
    pub dogma_score: f64,
    pub delta_real: f64,
    pub acc_m: [f64; 4],
    pub neuromod: NeuromodSignal,
    pub active_branch: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeripheryReply {
    pub text: String,
    pub backend: String,
    pub fidelity_ok: bool,
    pub fidelity_issue: Option<String>,
    pub fallback_used: bool,
    pub elapsed_ms: f64,
}

pub enum PeripheryBackend {
    NativeVerbalizer,
    LocalGemma {
        handle: RawGemmaHandle,
        path: String,
    },
    OpenAiApi {
        base_url: String,
        model: String,
        api_key: String,
        timeout_secs: u64,
    },
}

impl PeripheryBackend {
    pub fn label(&self) -> String {
        match self {
            Self::NativeVerbalizer => "verbalizador-nativo (0 LLM)".into(),
            Self::LocalGemma { path, .. } => format!("Gemma2-GGUF congelado ({path})"),
            Self::OpenAiApi { model, base_url, .. } => format!("LLM-API congelado ({model} @ {base_url})"),
        }
    }

    pub fn open_local_gemma(path: &Path) -> Result<Self, String> {
        let probe = FrozenGemma2Probe::try_open(Some(path))?;
        Ok(Self::LocalGemma {
            handle: probe.raw_handle(),
            path: path.display().to_string(),
        })
    }

    pub fn verbalize(&self, pkt: &DecisionPacket) -> PeripheryReply {
        let t0 = Instant::now();
        match self {
            Self::NativeVerbalizer => PeripheryReply {
                text: render_native_spanish(pkt),
                backend: self.label(),
                fidelity_ok: true,
                fidelity_issue: None,
                fallback_used: false,
                elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
            },
            Self::LocalGemma { handle, .. } => {
                let prompt = render_gemma_periphery_prompt(pkt);
                let cfg = Gemma2GenerationConfig {
                    max_tokens: 140,
                    context_limit: 1024,
                    temperature: 0.25,
                    top_p: 0.9,
                    seed: 42,
                };
                match handle.generate_prompt(&prompt, cfg, None) {
                    Ok(r) if !r.text.trim().is_empty() => {
                        let cand = r.text.trim().to_string();
                        match verify_periphery_fidelity(pkt, &cand) {
                            Ok(()) => PeripheryReply {
                                text: cand,
                                backend: self.label(),
                                fidelity_ok: true,
                                fidelity_issue: None,
                                fallback_used: false,
                                elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                            },
                            Err(issue) => PeripheryReply {
                                text: render_native_spanish(pkt),
                                backend: self.label(),
                                fidelity_ok: false,
                                fidelity_issue: Some(format!("Guardia rechazó salida Gemma: {issue}")),
                                fallback_used: true,
                                elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                            },
                        }
                    }
                    Ok(_) => PeripheryReply {
                        text: render_native_spanish(pkt),
                        backend: self.label(),
                        fidelity_ok: false,
                        fidelity_issue: Some("Gemma devolvió salida vacía".into()),
                        fallback_used: true,
                        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                    },
                    Err(e) => PeripheryReply {
                        text: render_native_spanish(pkt),
                        backend: self.label(),
                        fidelity_ok: false,
                        fidelity_issue: Some(format!("Error Gemma local: {e}")),
                        fallback_used: true,
                        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                    },
                }
            }
            Self::OpenAiApi {
                base_url,
                model,
                api_key,
                timeout_secs,
            } => {
                match call_openai_periphery(base_url, model, api_key, *timeout_secs, pkt) {
                    Ok(cand) if !cand.trim().is_empty() => {
                        match verify_periphery_fidelity(pkt, &cand) {
                            Ok(()) => PeripheryReply {
                                text: cand,
                                backend: self.label(),
                                fidelity_ok: true,
                                fidelity_issue: None,
                                fallback_used: false,
                                elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                            },
                            Err(issue) => PeripheryReply {
                                text: render_native_spanish(pkt),
                                backend: self.label(),
                                fidelity_ok: false,
                                fidelity_issue: Some(format!("Guardia rechazó alucinación del LLM periférico: {issue}")),
                                fallback_used: true,
                                elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                            },
                        }
                    }
                    Ok(_) => PeripheryReply {
                        text: render_native_spanish(pkt),
                        backend: self.label(),
                        fidelity_ok: false,
                        fidelity_issue: Some("LLM-API devolvió texto vacío".into()),
                        fallback_used: true,
                        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                    },
                    Err(e) => PeripheryReply {
                        text: render_native_spanish(pkt),
                        backend: self.label(),
                        fidelity_ok: false,
                        fidelity_issue: Some(e),
                        fallback_used: true,
                        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                    },
                }
            }
        }
    }
}

/// Verbalizador nativo en español natural, 100% fiel al DecisionPacket de la arquitectura.
pub fn render_native_spanish(pkt: &DecisionPacket) -> String {
    let mut out = pkt.core_thesis.clone();
    match pkt.stance {
        EpistemicStance::ConflictDetected => {
            if !pkt.refutation_notes.is_empty() {
                out.push_str(&format!(" ({})", pkt.refutation_notes[0]));
            }
        }
        EpistemicStance::NeedClarification => {
            if !pkt.refutation_notes.is_empty() && !out.contains("falacia") {
                out.push_str(&format!(" Nota de verificación cruzada: {}", pkt.refutation_notes[0]));
            }
        }
        EpistemicStance::Abstain => {
            if pkt.dogma_score > 0.35 {
                out.push_str(" (La verificación anti-dogmatismo bloqueó una respuesta sin respaldo.)");
            }
        }
        EpistemicStance::Qualified => {
            if pkt.refute_vuln > 0.40 {
                out.push_str(" Mantengo incertidumbre calibrada porque la rama refutadora encontró alternativas plausibles.");
            }
        }
        EpistemicStance::Assert => {
            if !pkt.supporting_evidence.is_empty()
                && matches!(
                    pkt.intent,
                    DialogueIntent::LogicReasoning | DialogueIntent::ArithmeticQuery
                )
            {
                out.push_str(&format!(" [Base: {}]", pkt.supporting_evidence[0]));
            }
        }
    }
    out
}

/// Prompt estricto de periferia lingüística para Gemma 2 GGUF (prohibido razonar o contradecir el paquete).
pub fn render_gemma_periphery_prompt(pkt: &DecisionPacket) -> String {
    format!(
        "<start_of_turn>user\nEres ÚNICAMENTE la periferia lingüística (verbalizador) de la arquitectura neuro-geométrica METAFE-3S.\n\
         REGLAS ESTRICTAS:\n\
         1. NO razona por tu cuenta ni agregues datos inventados.\n\
         2. Respeta exactamente la POSTURA EPISTÉMICA decidida por la arquitectura: {}.\n\
         3. Si la postura es ABSTENCIÓN o CONTRADICCIÓN o ACLARACIÓN, exprésalo claramente sin inventar una respuesta afirmativa.\n\
         4. Redacta en español natural y claro (1-3 frases).\n\n\
         Mensaje del usuario: {}\n\
         Tesis decidida por la arquitectura: {}\n\
         Evidencia: {}\n\
         Verificación cruzada (Rama A vs B): {}\n\
         Confianza calibrada: {:.0}%\n<end_of_turn>\n<start_of_turn>model\n",
        pkt.stance.label_es(),
        pkt.user_message,
        pkt.core_thesis,
        pkt.supporting_evidence.join("; "),
        pkt.refutation_notes.join("; "),
        pkt.confidence * 100.0
    )
}

/// Guardia de Fidelidad de la Periferia Lingüística:
/// Garantiza que el LLM periférico NO sobreescriba la postura ni alucine datos contradiciendo al núcleo 3S.
pub fn verify_periphery_fidelity(pkt: &DecisionPacket, llm_text: &str) -> Result<(), String> {
    let norm = normalize_es(llm_text);
    if norm.trim().len() < 4 {
        return Err("respuesta demasiado corta".into());
    }
    match pkt.stance {
        EpistemicStance::Abstain => {
            let hedges = [
                "no tengo", "no dispongo", "no puedo", "no se", "abstengo", "abstencion",
                "sin evidencia", "sin datos", "no consta", "desconozco", "no cuento",
                "no me has", "falta evidencia", "no es posible verificar",
            ];
            if !hedges.iter().any(|h| norm.contains(h)) {
                return Err("la arquitectura decidió ABSTENCIÓN pero el LLM intentó afirmar una respuesta".into());
            }
        }
        EpistemicStance::ConflictDetected => {
            let cues = [
                "contradic", "inconsisten", "conflicto", "incompatible", "opuesto", "choca",
                "no pueden ser", "ambas",
            ];
            if !cues.iter().any(|c| norm.contains(c)) {
                return Err("la arquitectura detectó CONTRADICCIÓN pero el LLM la omitió".into());
            }
        }
        EpistemicStance::NeedClarification => {
            let cues = [
                "no necesariamente", "no se puede afirmar", "no implica", "falacia",
                "aclar", "insuficiente", "ambig", "otra causa", "precis", "no basta", "no puede asegur",
            ];
            if !cues.iter().any(|c| norm.contains(c)) {
                return Err("la arquitectura pidió ACLARACIÓN/falacia pero el LLM afirmó con certeza".into());
            }
        }
        EpistemicStance::Assert => {
            // Para cálculos aritméticos exactos, exigir que el número exacto aparezca en la verbalización
            if pkt.intent == DialogueIntent::ArithmeticQuery {
                let nums_expected: Vec<&str> = pkt
                    .core_thesis
                    .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
                    .filter(|s| !s.is_empty() && s.chars().any(|c| c.is_ascii_digit()))
                    .collect();
                if let Some(last_num) = nums_expected.last() {
                    if !llm_text.contains(last_num) {
                        return Err(format!("el LLM alteró u omitió el resultado aritmético exacto `{last_num}`"));
                    }
                }
            }
        }
        EpistemicStance::Qualified => {}
    }
    Ok(())
}

/// Invoca cualquier API OpenAI-compatible (Ollama, LM Studio, vLLM, OpenAI, Groq, docker-llm)
/// usando `curl` con cuerpo JSON vía stdin (cero dependencias externas de Cargo).
fn call_openai_periphery(
    base_url: &str,
    model: &str,
    api_key: &str,
    timeout_secs: u64,
    pkt: &DecisionPacket,
) -> Result<String, String> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let sys = "Eres ÚNICAMENTE la periferia lingüística de la arquitectura neuro-geométrica METAFE-3S. \
               No razonas por tu cuenta ni contradices ni inventas datos fuera del DecisionPacket. \
               Verbaliza en español natural (1-3 frases) respetando estrictamente la postura epistémica y la tesis decidida.";
    let usr = format!(
        "Mensaje del usuario: {}\nPostura decidida: {}\nTesis del núcleo: {}\nEvidencia: {}\nVerificación cruzada: {}\nConfianza: {:.0}%",
        pkt.user_message,
        pkt.stance.label_es(),
        pkt.core_thesis,
        pkt.supporting_evidence.join("; "),
        pkt.refutation_notes.join("; "),
        pkt.confidence * 100.0
    );
    let body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr}
        ],
        "temperature": 0.25,
        "max_tokens": 180,
        "stream": false
    });
    let body_bytes = serde_json::to_vec(&body).map_err(|e| e.to_string())?;

    let mut cmd = Command::new("curl");
    cmd.args([
        "-sS",
        "--fail-with-body",
        "--max-time",
        &timeout_secs.to_string(),
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
    ]);
    if !api_key.trim().is_empty() {
        cmd.args(["-H", &format!("Authorization: Bearer {}", api_key.trim())]);
    }
    cmd.args(["-d", "@-", &url]);
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("no se pudo invocar curl: {e}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&body_bytes)
            .map_err(|e| format!("error escribiendo cuerpo JSON a curl: {e}"))?;
    }
    let output = child.wait_with_output().map_err(|e| format!("error esperando curl: {e}"))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let out = String::from_utf8_lossy(&output.stdout);
        return Err(format!("API LLM falló ({}): {} {}", output.status, err.trim(), out.trim()));
    }
    let v: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("respuesta JSON inválida de API LLM: {e}"))?;
    let text = v
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| "respuesta API sin choices[0].message.content".to_string())?;
    Ok(text.trim().to_string())
}
