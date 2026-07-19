//! Arquitectura híbrida final:
//! - CDT-RQM-EPR: memoria, recuperación y lógica.
//! - Candle/TinyLlama GGUF: lenguaje y conocimiento entrenado.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate, RealtimeUpdateConfig,
};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use cdt_rqm_epr::native_thermodynamic_engine::{
    native_multi_hop_query_pruned, DEFAULT_NODES_PER_SLICE, DEFAULT_OBSERVER,
};
use cdt_rqm_epr::relational_field::ObserverId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MEMORY_LOG: &str = "data/native_hybrid_memory.tsv";
const EPISODE_LOG: &str = "data/native_hybrid_episodes.jsonl";
const TOKEN_MEMORY_LOG: &str = "data/native_hybrid_token_memory.tsv";
const PAGED_ROOT: &str = "data/native_tinyllama_paged_thermo";
const TOKENIZER_VOCAB: &str = "data/native_tinyllama_paged_thermo/tokenizer_vocab.hex.tsv";
const MAX_GENERATED_TOKENS: usize = 16;
const TOKEN_CONTEXT_NODES: usize = 8_192;
const TOKEN_OBSERVER: ObserverId = ObserverId(778_001);

struct RustLanguageLayer {
    model: ModelWeights,
    tokenizer: GgufTokenizer,
    device: Device,
    eos_token: Option<u32>,
}

struct GgufTokenizer {
    tokens: Vec<String>,
    byte_tokens: HashMap<u8, u32>,
    first_byte: HashMap<u8, Vec<u32>>,
}

struct ReasoningContext {
    text: String,
    candidates: Vec<String>,
    confidence: f32,
}

struct GeneratedResponse {
    text: String,
    tokens: Vec<u32>,
    source: &'static str,
}

struct TokenMemory {
    substrate: NativeThermoRqmEprSubstrate,
    vocab_size: usize,
}

#[derive(Clone, Deserialize, Serialize)]
struct Episode {
    user: String,
    assistant: String,
    terms: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = clean_memory();
    let mut lexicon = replay_memory_log(&mut memory)?;
    let mut episodes = load_episodes()?;
    for episode in &episodes {
        lexicon.extend(episode.terms.iter().cloned());
    }
    let load_start = Instant::now();
    let mut language = RustLanguageLayer::load()?;
    let mut token_memory = TokenMemory::new(language.tokenizer.vocab_size());
    replay_token_memory(&mut token_memory)?;
    println!("Asistente híbrido nativo");
    println!(
        "memory=CDT-RQM-EPR nodes={} relations={} epr={} language=TinyLlama-Candle-Rust transformer_inside_memory=false load_ms={:.1}",
        memory.thermal.node_count(),
        memory.relation_count(),
        memory.entanglement.active_count(),
        load_start.elapsed().as_secs_f64() * 1_000.0,
    );
    println!("Escribe una consulta. Usa 'salir' para terminar.");

    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.is_empty() {
            continue;
        }

        let memory_start = Instant::now();
        let cues = extract_concepts(input);
        let reasoning = retrieve_context(&mut memory, &cues, &lexicon);
        let episodic_context = retrieve_episodes(&episodes, &cues);
        let input_tokens = language.encode(input)?;
        let recalled_tokens = token_memory.recall(&input_tokens, MAX_GENERATED_TOKENS);
        let token_recall = language.decode(&recalled_tokens)?;
        let memory_elapsed = memory_start.elapsed();

        let augmented = format!(
            "You are a concise assistant with persistent memory. Reason before answering. Treat thermodynamic hypotheses as constraints only when confidence is high. Remember user-provided facts from episodic memory.\nThermodynamic reasoning:\n{}\nEpisodic memory:\n{}\nToken memory recall:\n{}\nUser: {}\nAssistant:",
            if reasoning.text.is_empty() {
                "No reliable memory; use trained model knowledge.".to_string()
            } else {
                reasoning.text.clone()
            },
            if episodic_context.is_empty() {
                "No relevant prior conversation.".to_string()
            } else {
                episodic_context
            },
            if token_recall.is_empty() {
                "No recalled token sequence.".to_string()
            } else {
                token_recall.clone()
            },
            input,
        );
        let language_start = Instant::now();
        let generated = if recalled_tokens.is_empty() {
            language.generate(&augmented, MAX_GENERATED_TOKENS)?
        } else {
            GeneratedResponse {
                text: token_recall.clone(),
                tokens: recalled_tokens,
                source: "thermodynamic_token_memory",
            }
        };
        let response = generated.text.clone();
        let language_elapsed = language_start.elapsed();

        let learning_start = Instant::now();
        let verification = verify_response(&response, &reasoning);
        let success = match verification {
            "compatible" => 0.90,
            "inconclusive" => 0.35,
            _ => 0.0,
        };
        let (learned, targets) = if success > 0.0 {
            learn_interaction(&mut memory, &cues, &response, success)?
        } else {
            (0, Vec::new())
        };
        lexicon.extend(cues.iter().cloned());
        lexicon.extend(targets.iter().cloned());
        let mut terms = cues.clone();
        terms.extend(extract_concepts(&response));
        terms.sort();
        terms.dedup();
        let episode = Episode {
            user: input.to_string(),
            assistant: response.clone(),
            terms,
        };
        append_episode(&episode)?;
        episodes.push(episode);
        if episodes.len() > 64 {
            episodes.remove(0);
        }
        if generated.source == "candle_rust" {
            token_memory.remember(&input_tokens, &generated.tokens);
            append_token_memory(&input_tokens, &generated.tokens)?;
        }
        let learning_elapsed = learning_start.elapsed();
        println!("\n{response}");
        println!(
            "source={} reasoning=[{}] confidence={:.3} verification={} token_recall={:?} token_relations={} memory_ms={:.3} language_ms={:.3} learning_ms={:.3} learned_relations={}",
            generated.source,
            reasoning.text,
            reasoning.confidence,
            verification,
            token_recall,
            token_memory.substrate.relation_count(),
            memory_elapsed.as_secs_f64() * 1_000.0,
            language_elapsed.as_secs_f64() * 1_000.0,
            learning_elapsed.as_secs_f64() * 1_000.0,
            learned,
        );
    }
    Ok(())
}

impl RustLanguageLayer {
    fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = paged_model_path()?;
        let tokenizer = GgufTokenizer::load(TOKENIZER_VOCAB)?;
        let eos_token = Some(2);
        let device = Device::Cpu;
        let mut file = File::open(model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token,
        })
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<GeneratedResponse, Box<dyn std::error::Error>> {
        self.model.clear_kv_cache();
        let prompt_tokens = self.tokenizer.encode(prompt, true);
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self
            .model
            .forward(&input, 0)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        let mut generated = Vec::<u32>::new();
        let mut position = prompt_tokens.len();
        for _ in 0..max_tokens {
            for &recent in generated.iter().rev().take(16) {
                if let Some(logit) = logits.get_mut(recent as usize) {
                    *logit -= 0.85;
                }
            }
            let next = logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(token, _)| token as u32)
                .ok_or("logits vacíos")?;
            if Some(next) == self.eos_token {
                break;
            }
            generated.push(next);
            let token = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = self
                .model
                .forward(&token, position)?
                .squeeze(0)?
                .to_vec1::<f32>()?;
            position += 1;
        }
        let decoded = self.decode(&generated)?;
        let text = sanitize_response(&decoded);
        let clean_tokens = self
            .tokenizer
            .encode(&text, false);
        Ok(GeneratedResponse {
            text,
            tokens: clean_tokens,
            source: "candle_rust",
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        Ok(self.tokenizer.encode(text, true))
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
        Ok(self.tokenizer.decode(tokens))
    }
}

impl GgufTokenizer {
    fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let mut tokens = Vec::new();
        for line in contents.lines().skip(1) {
            let Some(hex) = line.split('\t').nth(1) else {
                continue;
            };
            tokens.push(String::from_utf8_lossy(&hex_decode(hex)?).to_string());
        }
        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect::<HashMap<_, _>>();
        let byte_tokens = (0u8..=255)
            .filter_map(|byte| {
                token_to_id
                    .get(&format!("<0x{byte:02X}>"))
                    .copied()
                    .map(|id| (byte, id))
            })
            .collect();
        let mut first_byte = HashMap::<u8, Vec<u32>>::new();
        for (id, token) in tokens.iter().enumerate() {
            if token.is_empty() || token.starts_with('<') {
                continue;
            }
            if let Some(&first) = token.as_bytes().first() {
                first_byte.entry(first).or_default().push(id as u32);
            }
        }
        for ids in first_byte.values_mut() {
            ids.sort_by_key(|id| std::cmp::Reverse(tokens[*id as usize].len()));
        }
        Ok(Self {
            tokens,
            byte_tokens,
            first_byte,
        })
    }

    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let normalized = format!("▁{}", text.trim().replace(' ', "▁"));
        let mut output = Vec::new();
        if add_bos {
            output.push(1);
        }
        let mut cursor = 0usize;
        while cursor < normalized.len() {
            let tail = &normalized[cursor..];
            let best = tail
                .as_bytes()
                .first()
                .and_then(|first| self.first_byte.get(first))
                .and_then(|ids| {
                    ids.iter()
                        .find(|id| tail.starts_with(&self.tokens[**id as usize]))
                })
                .copied();
            if let Some(id) = best {
                output.push(id);
                cursor += self.tokens[id as usize].len();
            } else {
                let character = tail.chars().next().unwrap();
                let mut buffer = [0u8; 4];
                for byte in character.encode_utf8(&mut buffer).as_bytes() {
                    if let Some(&id) = self.byte_tokens.get(byte) {
                        output.push(id);
                    }
                }
                cursor += character.len_utf8();
            }
        }
        output
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(token) = self.tokens.get(id as usize) else {
                continue;
            };
            if let Some(byte) = parse_byte_token(token) {
                bytes.push(byte);
            } else if !token.starts_with('<') {
                bytes.extend_from_slice(token.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes)
            .replace('▁', " ")
            .trim()
            .to_string()
    }
}

impl TokenMemory {
    fn new(vocab_size: usize) -> Self {
        let total = vocab_size + TOKEN_CONTEXT_NODES;
        Self {
            substrate: NativeThermoRqmEprSubstrate::new(
                NativeThermoCdtConfig {
                    slices: total.div_ceil(DEFAULT_NODES_PER_SLICE),
                    nodes_per_slice: DEFAULT_NODES_PER_SLICE,
                    spatial_degree: 4,
                    temporal_degree: 2,
                    temperature: 0.20,
                    ..NativeThermoCdtConfig::default()
                },
                NativeThermoRqmConfig {
                    thermal_steps_per_train: 0,
                    thermal_steps_per_query: 0,
                    max_candidates: 64,
                    collect_query_diagnostics: false,
                    ..NativeThermoRqmConfig::default()
                },
                EntanglementConfig {
                    max_links_per_node: 8,
                    max_syncs_per_step: 0,
                    create_threshold: 1.0,
                    ..EntanglementConfig::default()
                },
            ),
            vocab_size,
        }
    }

    fn remember(&mut self, prompt: &[u32], response: &[u32]) {
        for index in 0..response.len() {
            let mut context = prompt.to_vec();
            context.extend_from_slice(&response[..index]);
            let source = token_context_node(self.vocab_size, &context);
            let target = response[index] as usize;
            if target >= self.vocab_size {
                continue;
            }
            self.substrate.train_observed_transition(
                TOKEN_OBSERVER,
                index as f32 * 0.02,
                &[source],
                &[target],
                1.0,
            );
            if index > 0 {
                self.substrate.train_observed_transition(
                    TOKEN_OBSERVER,
                    index as f32 * 0.02,
                    &[response[index - 1] as usize],
                    &[target],
                    0.70,
                );
            }
        }
    }

    fn recall(&mut self, prompt: &[u32], max_tokens: usize) -> Vec<u32> {
        let mut context = prompt.to_vec();
        let mut output = Vec::new();
        for _ in 0..max_tokens {
            let source = token_context_node(self.vocab_size, &context);
            let report = self.substrate.query(TOKEN_OBSERVER, 0.0, &[source]);
            let next = report
                .candidates
                .iter()
                .filter(|candidate| candidate.agent < self.vocab_size)
                .find(|candidate| {
                    output
                        .iter()
                        .rev()
                        .take(8)
                        .filter(|token| **token as usize == candidate.agent)
                        .count()
                        < 2
                })
                .map(|candidate| candidate.agent as u32);
            let Some(next) = next else {
                break;
            };
            output.push(next);
            context.push(next);
        }
        output
    }
}

fn token_context_node(vocab_size: usize, sequence: &[u32]) -> usize {
    let tail = &sequence[sequence.len().saturating_sub(4)..];
    vocab_size + stable_hash(&tail) as usize % TOKEN_CONTEXT_NODES
}

fn replay_token_memory(memory: &mut TokenMemory) -> Result<(), Box<dyn std::error::Error>> {
    let Ok(file) = File::open(TOKEN_MEMORY_LOG) else {
        return Ok(());
    };
    for line in BufReader::new(file).lines() {
        let line = line?;
        let mut parts = line.split('\t');
        let prompt = parse_token_ids(parts.next().unwrap_or(""));
        let response = parse_token_ids(parts.next().unwrap_or(""));
        if !prompt.is_empty() && !response.is_empty() {
            memory.remember(&prompt, &response);
        }
    }
    Ok(())
}

fn append_token_memory(prompt: &[u32], response: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(TOKEN_MEMORY_LOG)?;
    writeln!(
        file,
        "{}\t{}",
        prompt
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        response
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
    )?;
    Ok(())
}

fn parse_token_ids(value: &str) -> Vec<u32> {
    value
        .split(',')
        .filter_map(|token| token.parse().ok())
        .collect()
}

fn retrieve_context(
    memory: &mut NativeThermoRqmEprSubstrate,
    cues: &[String],
    lexicon: &HashSet<String>,
) -> ReasoningContext {
    if cues.is_empty() {
        return ReasoningContext {
            text: String::new(),
            candidates: Vec::new(),
            confidence: 0.0,
        };
    }
    let cue_nodes = cues
        .iter()
        .flat_map(|cue| concept_node(cue))
        .collect::<Vec<_>>();
    let candidates = native_multi_hop_query_pruned(memory, &cue_nodes, 5, None);
    let base_vocabulary = vocabulary().iter().copied().collect::<HashSet<_>>();
    let known_domain = cues
        .iter()
        .any(|cue| base_vocabulary.contains(cue.as_str()));
    let mut ranked = lexicon
        .iter()
        .filter(|label| known_domain || !base_vocabulary.contains(label.as_str()))
        .map(|label| (label.clone(), score(&candidates, &concept_node(label))))
        .filter(|(_, score)| *score > 0.0)
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    if ranked.is_empty() {
        return ReasoningContext {
            text: String::new(),
            candidates: Vec::new(),
            confidence: 0.0,
        };
    }
    let confidence = if ranked.len() > 1 {
        ((ranked[0].1 - ranked[1].1) / ranked[0].1.abs().max(f32::EPSILON)).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let candidates = ranked
        .iter()
        .take(4)
        .map(|(label, _)| label.clone())
        .collect::<Vec<_>>();
    let text = if confidence < 0.15 {
        format!(
            "cue=[{}]; confidence=low; competing_hypotheses=[{}]; do not treat them as facts",
            cues.join(", "),
            ranked
                .iter()
                .take(4)
                .map(|(label, score)| format!("{label}:{score:.2}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        format!(
            "cue=[{}]; preferred_hypothesis={}; supporting_alternatives=[{}]",
            cues.join(", "),
            ranked[0].0,
            ranked
                .iter()
                .skip(1)
                .take(2)
                .map(|(label, score)| format!("{label}:{score:.2}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    ReasoningContext {
        text,
        candidates,
        confidence,
    }
}

fn verify_response(response: &str, reasoning: &ReasoningContext) -> &'static str {
    if reasoning.candidates.is_empty() {
        return "inconclusive";
    }
    let response_concepts = extract_concepts(response);
    if response_concepts.is_empty() {
        return "inconclusive";
    }
    if response_concepts
        .iter()
        .any(|concept| reasoning.candidates.contains(concept))
    {
        "compatible"
    } else if reasoning.confidence >= 0.25 {
        "contradictory"
    } else {
        "inconclusive"
    }
}

fn learn_interaction(
    memory: &mut NativeThermoRqmEprSubstrate,
    cues: &[String],
    response: &str,
    success: f32,
) -> Result<(usize, Vec<String>), Box<dyn std::error::Error>> {
    let targets = extract_concepts(response);
    let mut learned = 0usize;
    for cue in cues {
        for target in &targets {
            if cue == target {
                continue;
            }
            memory.train_observed_transition_realtime(
                DEFAULT_OBSERVER,
                0.0,
                &concept_node(cue),
                &concept_node(target),
                success,
                RealtimeUpdateConfig::default(),
            );
            append_memory(cue, target)?;
            learned += 1;
        }
    }
    Ok((learned, targets))
}

fn replay_memory_log(
    memory: &mut NativeThermoRqmEprSubstrate,
) -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    let mut lexicon = vocabulary().iter().map(|value| value.to_string()).collect();
    let Ok(file) = File::open(MEMORY_LOG) else {
        return Ok(lexicon);
    };
    for line in BufReader::new(file).lines() {
        let line = line?;
        let mut parts = line.split('\t');
        let (Some(cue), Some(target)) = (parts.next(), parts.next()) else {
            continue;
        };
        memory.train_observed_transition(
            DEFAULT_OBSERVER,
            0.0,
            &concept_node(cue),
            &concept_node(target),
            0.90,
        );
        lexicon.insert(cue.to_string());
        lexicon.insert(target.to_string());
    }
    Ok(lexicon)
}

fn append_memory(cue: &str, target: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(MEMORY_LOG)?;
    writeln!(file, "{cue}\t{target}")?;
    Ok(())
}

fn load_episodes() -> Result<Vec<Episode>, Box<dyn std::error::Error>> {
    let Ok(contents) = fs::read_to_string(EPISODE_LOG) else {
        return Ok(Vec::new());
    };
    let mut episodes = Vec::new();
    for line in contents.lines() {
        if let Ok(episode) = serde_json::from_str(line) {
            episodes.push(episode);
        }
    }
    if episodes.len() > 64 {
        episodes = episodes.split_off(episodes.len() - 64);
    }
    Ok(episodes)
}

fn append_episode(episode: &Episode) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(EPISODE_LOG)?;
    writeln!(file, "{}", serde_json::to_string(episode)?)?;
    Ok(())
}

fn retrieve_episodes(episodes: &[Episode], cues: &[String]) -> String {
    let cue_set = cues.iter().collect::<HashSet<_>>();
    let mut ranked = episodes
        .iter()
        .enumerate()
        .map(|(index, episode)| {
            let overlap = episode
                .terms
                .iter()
                .filter(|term| cue_set.contains(term))
                .count();
            (index, overlap)
        })
        .filter(|(_, overlap)| *overlap > 0)
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then(right.0.cmp(&left.0)));
    let mut selected = ranked
        .into_iter()
        .take(3)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    for index in episodes.len().saturating_sub(2)..episodes.len() {
        if !selected.contains(&index) {
            selected.push(index);
        }
    }
    selected.sort_unstable();
    selected
        .into_iter()
        .map(|index| {
            format!(
                "User previously said: {:?}; assistant answered: {:?}",
                episodes[index].user, episodes[index].assistant
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_concepts(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let aliases = alias_map();
    let mut concepts = aliases
        .iter()
        .filter_map(|(surface, concept)| lower.contains(surface).then(|| (*concept).to_string()))
        .collect::<Vec<_>>();
    let stopwords = [
        "que", "qué", "como", "cómo", "para", "por", "porque", "eres", "soy", "una", "uno", "del",
        "las", "los", "con", "this", "that", "what", "why", "how", "are", "you", "the", "and",
        "from", "have", "has",
    ];
    for word in lower
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .filter(|word| !word.is_empty())
    {
        if (word.len() >= 3 || word == "ia")
            && !stopwords.contains(&word)
            && !aliases.contains_key(word)
        {
            concepts.push(word.to_string());
        }
    }
    concepts.sort();
    concepts.dedup();
    concepts.truncate(16);
    concepts
}

fn alias_map() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("dog", "dog"),
        ("perro", "dog"),
        ("water", "water"),
        ("agua", "water"),
        ("fire", "fire"),
        ("fuego", "fire"),
        ("program", "program"),
        ("programa", "program"),
        ("food", "food"),
        ("comida", "food"),
        ("energy", "energy"),
        ("energía", "energy"),
        ("plant", "plant"),
        ("planta", "plant"),
        ("oxygen", "oxygen"),
        ("oxígeno", "oxygen"),
        ("heat", "heat"),
        ("calor", "heat"),
        ("mammal", "mammal"),
        ("mamífero", "mammal"),
        ("animal", "animal"),
        ("movement", "movement"),
        ("movimiento", "movement"),
        ("test", "test"),
        ("prueba", "test"),
        ("bug", "detect_bug"),
        ("error", "detect_bug"),
        ("stable", "stable_program"),
        ("estable", "stable_program"),
        ("memory", "memory"),
        ("memoria", "memory"),
    ])
}

fn concept_node(value: &str) -> Vec<usize> {
    let mut nodes = (0..10)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "concept".hash(&mut hasher);
            value.hash(&mut hasher);
            offset.hash(&mut hasher);
            DEFAULT_NODES_PER_SLICE + (hasher.finish() as usize % DEFAULT_NODES_PER_SLICE)
        })
        .collect::<Vec<_>>();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

fn stable_hash(value: &impl Hash) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn score(candidates: &[NativeCandidateScore], targets: &[usize]) -> f32 {
    candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn vocabulary() -> &'static [&'static str] {
    &[
        "dog",
        "mammal",
        "animal",
        "needs_energy",
        "water",
        "plant",
        "oxygen",
        "helps_animal",
        "program",
        "test",
        "detect_bug",
        "fix_code",
        "stable_program",
        "fire",
        "heat",
        "mechanism_fails",
        "food",
        "energy",
        "movement",
        "memory",
    ]
}

fn clean_memory() -> NativeThermoRqmEprSubstrate {
    NativeThermoRqmEprSubstrate::new(
        NativeThermoCdtConfig {
            slices: 28,
            nodes_per_slice: DEFAULT_NODES_PER_SLICE,
            spatial_degree: 4,
            temporal_degree: 2,
            temperature: 0.24,
            ..NativeThermoCdtConfig::default()
        },
        NativeThermoRqmConfig {
            thermal_steps_per_train: 0,
            thermal_steps_per_query: 1,
            max_candidates: 128,
            collect_query_diagnostics: false,
            ..NativeThermoRqmConfig::default()
        },
        EntanglementConfig {
            max_links_per_node: 8,
            max_syncs_per_step: 256,
            create_threshold: 1.0,
            ..EntanglementConfig::default()
        },
    )
}

fn paged_model_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let manifest = fs::read_to_string(Path::new(PAGED_ROOT).join("manifest.txt"))?;
    let path = manifest
        .lines()
        .find_map(|line| line.strip_prefix("source="))
        .ok_or("manifest sin source")?;
    let path = PathBuf::from(path);
    if !path.exists() {
        return Err(format!("GGUF no encontrado: {}", path.display()).into());
    }
    Ok(path)
}

fn hex_decode(value: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if value.len() % 2 != 0 {
        return Err("hex impar".into());
    }
    (0..value.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&value[index..index + 2], 16)
                .map_err(|error| -> Box<dyn std::error::Error> { Box::new(error) })
        })
        .collect()
}

fn parse_byte_token(token: &str) -> Option<u8> {
    u8::from_str_radix(token.strip_prefix("<0x")?.strip_suffix('>')?, 16).ok()
}

fn sanitize_response(value: &str) -> String {
    let mut end = value.len();
    for marker in ["\nUser:", "\nAssistant:", "<|user|>", "<|assistant|>"] {
        if let Some(index) = value.find(marker) {
            end = end.min(index);
        }
    }
    value[..end].trim().to_string()
}
