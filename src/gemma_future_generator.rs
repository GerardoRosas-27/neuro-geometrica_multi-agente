//! Adaptador de Gemma 2 al generador de fronteras futuras.
//!
//! Gemma no decide qué se consolida. Su única autoridad es proponer una lista
//! de estados parciales futuros en un formato restringido. El entrenador
//! fasorial vuelve a calcular F para cada propuesta y el CDT conserva sólo lo
//! que supera los gates de wake, sleep y `ΔF_store`.

use crate::future_guided_training::{FutureProposal, FutureProposalGenerator};
use crate::native_gemma2::{resolve_gemma2_device, Gemma2Tokenizer, QuantizedGemma2};
use crate::native_gemma2_runtime::{
    chat_tokens, Gemma2GenerationConfig, Gemma2GenerationMetrics, Gemma2Session,
};
use crate::native_hybrid_phasor_cdt_engine::NativePhasorCue;
use candle_core::quantized::gguf_file;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::path::Path;

pub struct GemmaFutureGenerator {
    model: QuantizedGemma2,
    tokenizer: Gemma2Tokenizer,
    session: Gemma2Session,
    generation: Gemma2GenerationConfig,
    pub last_text: String,
    pub last_metrics: Gemma2GenerationMetrics,
    pub last_error: Option<String>,
}

impl GemmaFutureGenerator {
    pub fn from_gguf(
        path: &Path,
        device: &str,
        generation: Gemma2GenerationConfig,
    ) -> Result<Self, Box<dyn Error>> {
        let device = resolve_gemma2_device(device)?;
        let mut file = File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
        let model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
        Ok(Self {
            model,
            tokenizer,
            session: Gemma2Session::new(),
            generation,
            last_text: String::new(),
            last_metrics: Gemma2GenerationMetrics::default(),
            last_error: None,
        })
    }

    fn prompt(&self, cue: &[NativePhasorCue], node_count: usize, count: usize) -> String {
        let observed_nodes = cue.iter().map(|item| item.node).collect::<HashSet<_>>();
        let allowed = (0..node_count)
            .filter(|node| !observed_nodes.contains(node))
            .map(|node| node.to_string())
            .collect::<Vec<_>>();
        let evidence = cue
            .iter()
            .map(|item| {
                format!(
                    "{}:{}",
                    item.node,
                    if item.phase.cos() >= 0.0 { "+" } else { "-" }
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let example_nodes = allowed.iter().take(3).cloned().collect::<Vec<_>>();
        let example = if example_nodes.len() == 3 {
            format!(
                "{}:+,{}:-,{}:+",
                example_nodes[0], example_nodes[1], example_nodes[2]
            )
        } else {
            "1:+,2:-,4:+".to_string()
        };
        let minimum_states = ((allowed.len() as f32 * 0.30).round() as usize).clamp(3, 24);
        format!(
            "You generate future boundary candidates for a binary phasor system with \
             {node_count} nodes. The only present evidence is [{evidence}]. Valid unobserved \
             numeric node IDs are [{}]. Produce {count} distinct plausible futures using only \
             IDs from that list. Every future must contain at least {minimum_states} different \
             node IDs. Output exactly one candidate per line in this format:\n\
             FUTURE|0.80|{example}\n\
             Replace the example with numeric IDs and + or -. Never output words such as node, \
             no, yes, nodo, or placeholders. No explanation, no JSON, no markdown.",
            allowed.join(",")
        )
    }
}

impl FutureProposalGenerator for GemmaFutureGenerator {
    fn propose(
        &mut self,
        cue: &[NativePhasorCue],
        node_count: usize,
        count: usize,
        seed: u64,
    ) -> Vec<FutureProposal> {
        self.last_error = None;
        self.last_text.clear();
        self.session.reset(&mut self.model);
        let prompt = self.prompt(cue, node_count, count);
        let prompt_limit = self
            .generation
            .context_limit
            .saturating_sub(self.generation.max_tokens)
            .max(32);
        let prompt_tokens = match chat_tokens(&self.tokenizer, &[], &prompt, prompt_limit) {
            Ok(tokens) => tokens,
            Err(error) => {
                self.last_error = Some(error.to_string());
                return Vec::new();
            }
        };
        let generation = Gemma2GenerationConfig {
            seed,
            ..self.generation
        };
        match self.session.generate(
            &mut self.model,
            &self.tokenizer,
            &prompt_tokens,
            None,
            generation,
            |_| {},
        ) {
            Ok(output) => {
                self.last_text = output.text;
                self.last_metrics = output.metrics;
                parse_gemma_future_proposals(&self.last_text, cue, node_count, count)
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
                Vec::new()
            }
        }
    }
}

pub fn parse_gemma_future_proposals(
    text: &str,
    cue: &[NativePhasorCue],
    node_count: usize,
    maximum: usize,
) -> Vec<FutureProposal> {
    let observed = cue.iter().map(|item| item.node).collect::<HashSet<_>>();
    let mut proposals = Vec::new();
    let starts = text
        .match_indices("FUTURE")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    for (position, start) in starts.iter().copied().enumerate() {
        let end = starts.get(position + 1).copied().unwrap_or(text.len());
        let segment = text[start + "FUTURE".len()..end]
            .trim()
            .trim_matches('`')
            .trim_start_matches(|character: char| {
                character.is_ascii_digit() || character.is_whitespace()
            })
            .trim_start_matches('|');
        let (confidence, states) = match segment.split_once('|') {
            Some((value, states)) if value.trim().parse::<f32>().is_ok() => (
                value.trim().parse::<f32>().unwrap_or(0.5).clamp(0.0, 1.0),
                states,
            ),
            // Algunos modelos omiten confianza y empiezan directamente con
            // `nodo:signo`. Se acepta con confianza conservadora, pero todas
            // las validaciones geométricas posteriores permanecen intactas.
            _ => (0.5, segment),
        };
        let mut used = HashSet::new();
        let mut goal = Vec::new();
        for state in states.split(',') {
            let Some((node, sign)) = state.trim().split_once(':') else {
                continue;
            };
            let Ok(node) = node.trim().parse::<usize>() else {
                continue;
            };
            if node >= node_count || observed.contains(&node) || !used.insert(node) {
                continue;
            }
            let sign = sign.trim();
            let phase = if sign.starts_with('-') || sign == "0" {
                std::f32::consts::PI
            } else if sign.starts_with('+') || sign == "1" {
                0.0
            } else {
                continue;
            };
            goal.push(NativePhasorCue {
                node,
                amplitude: confidence.max(0.25),
                phase,
            });
        }
        if !goal.is_empty() {
            proposals.push(FutureProposal {
                goal,
                confidence,
                latent_id: proposals.len(),
            });
            if proposals.len() >= maximum.max(1) {
                break;
            }
        }
    }
    proposals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_rejects_observed_duplicate_and_out_of_range_nodes() {
        let cue = [NativePhasorCue {
            node: 2,
            amplitude: 1.0,
            phase: 0.0,
        }];
        let text = "explicación descartada\n\
                    FUTURE1|0.9|2:+,3:-,3:+,99:+,4:+ FUTURE2|5:-,6:+";
        let proposals = parse_gemma_future_proposals(text, &cue, 8, 4);
        assert_eq!(proposals.len(), 2);
        assert_eq!(proposals[0].goal.len(), 2);
        assert_eq!(proposals[0].goal[0].node, 3);
        assert_eq!(proposals[0].goal[1].node, 4);
        assert_eq!(proposals[1].confidence, 0.5);
        assert_eq!(proposals[1].goal[0].node, 5);
    }
}
