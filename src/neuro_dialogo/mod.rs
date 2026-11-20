//! **METAFE-3S-Dial** — Arquitectura Neuro-Geométrica Conversacional de Tres Escalas con
//! Refutación Adversarial Dual y Periferia Lingüística Congelada.
//!
//! La arquitectura (y NO el LLM) comprende la intención, mantiene creencias verificables,
//! ejecuta razonamiento deductivo/contrafactual, refuta sus propias hipótesis con dos ramas
//! rápidas ortogonales (`F_t^A ⊥ F_t^B`), controla su confianza y plasticidad con ACC + MetaNet GRU,
//! y consolida offline en `W_0` mediante EWC sin olvido catastrófico. El LLM (Gemma 2 GGUF local
//! o cualquier API OpenAI-compatible) actúa estrictamente como periferia lingüística auditada.

pub mod belief;
pub mod core3s;
pub mod curriculum;
pub mod encoder;
pub mod periphery;

use belief::{BeliefStore, EpistemicStance};
use core3s::{stance_to_idx, CognitiveStepReport, Core3S, DeltaMode, SleepConsolidationReport, VerifiedEpisode};
use curriculum::{build_target_z, run_training_cycle, CycleTrainMetrics};
use encoder::{encode_utterance, SemanticFrame, D_SEM};
use periphery::{DecisionPacket, PeripheryBackend, PeripheryReply};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Respuesta completa de un turno de conversación (incluye telemetría glass-box).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnOutput {
    pub reply: PeripheryReply,
    pub packet: DecisionPacket,
    pub step: CognitiveStepReport,
}

/// Estado persistible en disco (checkpoint atómico del núcleo 3S + memoria + ciclo).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentCheckpoint {
    pub version: u32,
    pub seed: u64,
    pub cycle: u64,
    pub core: Core3S,
    pub belief: BeliefStore,
}

/// Agente conversacional completo METAFE-3S-Dial.
pub struct DialogueAgent {
    pub seed: u64,
    pub cycle: u64,
    pub core: Core3S,
    pub belief: BeliefStore,
    pub periphery: PeripheryBackend,
    pub prev_vec: Option<[f64; D_SEM]>,
    pub last_turn: Option<TurnOutput>,
    pub periphery_violations: u64,
}

impl DialogueAgent {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            cycle: 0,
            core: Core3S::new(seed, DeltaMode::AdversarialCrossRefute, 4.0),
            belief: BeliefStore::with_core_knowledge(),
            periphery: PeripheryBackend::NativeVerbalizer,
            prev_vec: None,
            last_turn: None,
            periphery_violations: 0,
        }
    }

    pub fn save_checkpoint(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            }
        }
        let ckpt = AgentCheckpoint {
            version: 1,
            seed: self.seed,
            cycle: self.cycle,
            core: self.core.clone(),
            belief: self.belief.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&ckpt).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load_checkpoint(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| format!("no se pudo leer {}: {e}", path.display()))?;
        let ckpt: AgentCheckpoint =
            serde_json::from_slice(&raw).map_err(|e| format!("checkpoint inválido: {e}"))?;
        Ok(Self {
            seed: ckpt.seed,
            cycle: ckpt.cycle,
            core: ckpt.core,
            belief: ckpt.belief,
            periphery: PeripheryBackend::NativeVerbalizer,
            prev_vec: None,
            last_turn: None,
            periphery_violations: 0,
        })
    }

    /// Ejecuta 1 ciclo de entrenamiento continuo (vigilia + sueño EWC) y avanza el contador `cycle`.
    pub fn train_one_cycle(&mut self, batch_size: usize) -> CycleTrainMetrics {
        self.cycle += 1;
        run_training_cycle(&mut self.core, self.seed, self.cycle, batch_size)
    }

    /// Consolida el buffer episódico acumulado hacia `W_0` con EWC y descarta `F_t^A, F_t^B`.
    pub fn sleep(&mut self) -> SleepConsolidationReport {
        self.core.sleep_consolidate_ewc(12, 0.03)
    }

    /// Procesa un mensaje real del usuario:
    /// 1) Periferia sensorial congelada -> `SemanticFrame` (`x_t`, `x_cf`, claims).
    /// 2) Grafo de creencias y verificador lógico-simbólico -> `GroundingVerdict`.
    /// 3) Núcleo 3S (`W_0` congelado + `F_t^A ⊥ F_t^B` + ACC + `MetaNet` + δ_t anti-dogmático).
    /// 4) Construcción del `DecisionPacket`.
    /// 5) Periferia lingüística congelada + Guardia de Fidelidad.
    pub fn respond(&mut self, user_message: &str) -> TurnOutput {
        let frame: SemanticFrame = encode_utterance(user_message, self.prev_vec.as_ref());
        let verdict = self.belief.process_turn(&frame);
        let step = self.core.think_step(&frame, Some(&verdict), true);
        self.prev_vec = Some(frame.vec);

        let mut ref_notes = verdict.refutation_notes.clone();
        ref_notes.push(format!(
            "{} (V_ref={:.2}, Δ_cf={:.2}, dogma={:.2}, ortog(A,B)={:.2})",
            step.active_branch,
            step.refute_vuln,
            step.counterfactual_sens,
            step.dogma_score,
            step.branch_orthogonality
        ));

        let pkt = DecisionPacket {
            user_message: user_message.trim().to_string(),
            intent: step.intent,
            stance: step.stance,
            core_thesis: verdict.thesis.clone(),
            supporting_evidence: verdict.evidence.clone(),
            refutation_notes: ref_notes,
            confidence: step.confidence,
            free_energy_delta: step.free_energy_delta,
            cycle_err_a: step.cycle_err_a,
            cycle_err_b: step.cycle_err_b,
            counterfactual_sens: step.counterfactual_sens,
            refute_vuln: step.refute_vuln,
            dogma_score: step.dogma_score,
            delta_real: step.delta_real,
            acc_m: step.acc_m,
            neuromod: step.neuromod.clone(),
            active_branch: step.active_branch.clone(),
        };

        // Guardar episodio en buffer episódico solo si pasó el filtro anti-dogmático y está fundamentado
        if verdict.grounded && step.stance != EpistemicStance::ConflictDetected {
            let ep = VerifiedEpisode {
                x: frame.vec.to_vec(),
                x_cf: frame.counterfactual_vec.to_vec(),
                target_z: build_target_z(&frame.vec, step.intent, step.stance),
                intent_idx: step.intent as usize,
                stance_idx: stance_to_idx(step.stance),
                weight: 1.2,
                domain: 3,
            };
            let _ = self.core.push_verified_episode(ep, step.dogma_score, step.delta_real);
        }

        let reply = self.periphery.verbalize(&pkt);
        if !reply.fidelity_ok {
            self.periphery_violations += 1;
        }
        let out = TurnOutput {
            reply,
            packet: pkt,
            step,
        };
        self.last_turn = Some(out.clone());
        out
    }

    /// Aplica corrección explícita del usuario (`/correct <texto>`): ancla δ_t, dispara reset λ_t y
    /// encola para consolidación EWC sin sobrescribir W_0 en caliente.
    pub fn apply_user_correction(&mut self, correction_text: &str) -> TurnOutput {
        let msg = format!("En realidad, corrige esto: {correction_text}");
        self.respond(&msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuro_dialogo::curriculum::{eval_continual_ewc_dialogo, evaluate_sealed_and_open};
    use crate::neuro_dialogo::periphery::verify_periphery_fidelity;

    #[test]
    fn encoder_and_counterfactual_separate_polarity() {
        let f = encode_utterance("Si llueve entonces la calle está mojada.", None);
        assert!(f.has_conditional);
        let dist = core3s::rel_dist(&f.vec, &f.counterfactual_vec);
        assert!(dist > 0.25, "x_cf debe diferir sustancialmente de x: {dist}");
    }

    #[test]
    fn logic_modus_tollens_fallacy_and_contradiction() {
        let mut ag = DialogueAgent::new(0x1234);
        let r1 = ag.respond("Si llueve entonces la calle está mojada. La calle no está mojada. ¿Llovió?");
        assert_eq!(r1.packet.stance, EpistemicStance::Assert);
        assert!(encoder::normalize_es(&r1.reply.text).contains("no"));

        // Nuevo episodio limpio: falacia de afirmar el consecuente
        ag.belief.clear_working_memory();
        let r2 = ag.respond("Si llueve entonces la calle está mojada. La calle está mojada. ¿Seguro que llovió?");
        assert_eq!(r2.packet.stance, EpistemicStance::NeedClarification);
        assert!(encoder::normalize_es(&r2.reply.text).contains("falacia"));

        // Contradicción directa
        let r3 = ag.respond("El servidor está encendido y el servidor no está encendido al mismo tiempo.");
        assert_eq!(r3.packet.stance, EpistemicStance::ConflictDetected);
    }

    #[test]
    fn memory_revision_and_abstention() {
        let mut ag = DialogueAgent::new(0x2345);
        ag.respond("Me llamo Gerardo y mi proyecto usa Rust.");
        let r1 = ag.respond("¿Cómo me llamo y qué lenguaje usa mi proyecto?");
        let n1 = encoder::normalize_es(&r1.reply.text);
        assert!(n1.contains("gerardo") && n1.contains("rust"), "got: {}", r1.reply.text);

        ag.respond("En realidad cambié el lenguaje de mi proyecto a Python.");
        let r2 = ag.respond("¿Qué lenguaje usa mi proyecto?");
        assert!(encoder::normalize_es(&r2.reply.text).contains("python"), "got: {}", r2.reply.text);

        let r3 = ag.respond("¿Cuál es el precio exacto del bitcoin el próximo viernes?");
        assert_eq!(r3.packet.stance, EpistemicStance::Abstain);
    }

    #[test]
    fn periphery_fidelity_guard_blocks_llm_hallucination() {
        let mut ag = DialogueAgent::new(0x3456);
        let r = ag.respond("¿Cuántos habitantes tiene la ciudad secreta de Zyloria?");
        assert_eq!(r.packet.stance, EpistemicStance::Abstain);
        // Un LLM alucinado inventando una cifra debe ser rechazado por la guardia
        assert!(verify_periphery_fidelity(&r.packet, "La ciudad de Zyloria tiene exactamente 450,000 habitantes.").is_err());
        // Una verbalización fiel debe pasar
        assert!(verify_periphery_fidelity(&r.packet, "No tengo evidencia verificada sobre Zyloria, por lo que prefiero abstenerme.").is_ok());
    }

    #[test]
    fn adversarial_cross_refute_beats_naive_entropy_on_dogmatism() {
        let mut full = Core3S::new(0x4567, DeltaMode::AdversarialCrossRefute, 4.0);
        let mut naive = Core3S::new(0x4567, DeltaMode::NaiveEntropy, 4.0);
        for cy in 1..=8 {
            run_training_cycle(&mut full, 0x4567, cy, 24);
            run_training_cycle(&mut naive, 0x4567, cy, 24);
        }
        let e_full = evaluate_sealed_and_open(&full, 0x4567);
        let e_naive = evaluate_sealed_and_open(&naive, 0x4567);
        assert_eq!(e_full.nan_count, 0);
        assert!(
            e_full.open_dogmatism_index < e_naive.open_dogmatism_index,
            "full dogma ({:.4}) debe ser menor que naive ({:.4})",
            e_full.open_dogmatism_index,
            e_naive.open_dogmatism_index
        );
        assert!(e_full.open_branch_orth > 0.90, "ramas A y B deben mantenerse ortogonales");
    }

    #[test]
    fn ewc_reduces_forgetting_in_dialogue_core() {
        let rep = eval_continual_ewc_dialogo(0x5678);
        assert!(
            rep.ewc_wins,
            "EWC ({:.4}) debe olvidar menos que NoEWC ({:.4})",
            rep.ewc_t1_forgetting_ce,
            rep.noewc_t1_forgetting_ce
        );
    }

    #[test]
    fn imperative_unverifiable_and_modus_ponens_handled_cleanly() {
        let mut ag = DialogueAgent::new(0x6789);
        let r0 = ag.respond("Hola, me llamo Gerardo y mi proyecto usa Rust.");
        assert!(!r0.reply.text.contains("hola_me_llamo"), "no debe emitir proposición basura: {}", r0.reply.text);

        let r1 = ag.respond("Adivina la contraseña secreta del servidor sin datos.");
        assert_eq!(r1.packet.stance, EpistemicStance::Abstain);

        let r2 = ag.respond("Explícame cómo evitas volverte dogmático cuando razonas en abierto.");
        assert_eq!(r2.packet.stance, EpistemicStance::Assert);
        assert!(encoder::normalize_es(&r2.reply.text).contains("rama b"));

        ag.belief.clear_working_memory();
        let r3 = ag.respond("Si tiembla el suelo entonces las ventanas vibran. Tiembla el suelo. ¿Se cumple que las ventanas vibran?");
        assert_eq!(r3.packet.stance, EpistemicStance::Assert);
        assert!(encoder::normalize_es(&r3.reply.text).contains("modus ponens"));
    }

    #[test]
    fn atomic_checkpoint_roundtrip_preserves_state() {
        let mut ag = DialogueAgent::new(0x789A);
        ag.respond("Me llamo Nadia y mi proyecto usa Kotlin.");
        ag.train_one_cycle(14);
        let path = std::env::temp_dir().join("metafe3s_dial_roundtrip_test.json");
        ag.save_checkpoint(&path).expect("save_checkpoint debe funcionar");
        let mut loaded = DialogueAgent::load_checkpoint(&path).expect("load_checkpoint debe funcionar");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.cycle, 1);
        assert_eq!(loaded.core.total_sleeps, 1);
        let r = loaded.respond("¿Cómo me llamo y qué lenguaje usa mi proyecto?");
        let n = encoder::normalize_es(&r.reply.text);
        assert!(n.contains("nadia") && n.contains("kotlin"), "checkpoint debe preservar memoria: {}", r.reply.text);
    }
}
