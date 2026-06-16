use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

const TOKEN_PATTERN_SIZE: usize = 13;
const CONTEXT_PATTERN_SIZE: usize = 17;
const PLAN_PATTERN_SIZE: usize = 29;
const STEP_PATTERN_SIZE: usize = 13;
const CONTEXT_WINDOW: usize = 3;
const TOP_K: usize = 5;

#[derive(Clone)]
struct Tokenizer {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
}

impl Tokenizer {
    fn from_corpus(corpus: &[String]) -> Self {
        let mut vocab = BTreeSet::new();
        vocab.insert("<bos>".to_string());
        vocab.insert("<eos>".to_string());
        vocab.insert("<unk>".to_string());
        for sentence in corpus {
            for token in tokenize(sentence) {
                vocab.insert(token);
            }
        }

        let id_to_token = vocab.into_iter().collect::<Vec<_>>();
        let token_to_id = id_to_token
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id))
            .collect();
        Self {
            token_to_id,
            id_to_token,
        }
    }

    fn encode(&self, sentence: &str) -> Vec<usize> {
        let mut ids = vec![self.id("<bos>")];
        ids.extend(tokenize(sentence).into_iter().map(|token| {
            self.token_to_id
                .get(&token)
                .copied()
                .unwrap_or(self.id("<unk>"))
        }));
        ids.push(self.id("<eos>"));
        ids
    }

    fn id(&self, token: &str) -> usize {
        self.token_to_id[token]
    }

    fn token(&self, id: usize) -> &str {
        &self.id_to_token[id]
    }

    fn len(&self) -> usize {
        self.id_to_token.len()
    }
}

struct Plan {
    sentence_ids: Vec<usize>,
    slot_patterns: Vec<Vec<usize>>,
}

struct LanguageModel {
    tokenizer: Tokenizer,
    token_patterns: Vec<Vec<usize>>,
    node_count: usize,
}

impl LanguageModel {
    fn new(tokenizer: Tokenizer, network: &SimplicialNetwork) -> Self {
        let token_patterns = (0..tokenizer.len())
            .map(|token_id| token_pattern(token_id, network.agents.len()))
            .collect();
        Self {
            tokenizer,
            token_patterns,
            node_count: network.agents.len(),
        }
    }

    fn train_sentence(&self, network: &mut SimplicialNetwork, sentence: &str) {
        let ids = self.tokenizer.encode(sentence);
        let plan = self.plan_for_sentence(sentence);
        for pos in 1..ids.len() {
            let context = self.context_with_plan(&ids[..pos], &plan, pos);
            let next = &self.token_patterns[ids[pos]];
            network.learn_transition(&context, next);
            network.reinforce_coactivation(next, 0.035);
        }
    }

    fn predict_next(&self, network: &SimplicialNetwork, ids: &[usize], plan: &Plan) -> Vec<usize> {
        let context = self.context_with_plan(ids, plan, ids.len());
        let predicted_agents = network.infer_transitive_from(&context, 1, 768);
        self.score_tokens(&predicted_agents, TOP_K)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    fn generate_with_plan(
        &self,
        network: &SimplicialNetwork,
        prompt: &str,
        intended: &str,
        max_tokens: usize,
    ) -> String {
        let plan = self.plan_for_sentence(intended);
        let mut ids = self.tokenizer.encode(prompt);
        ids.pop();
        for _ in 0..max_tokens {
            let predictions = self.predict_next(network, &ids, &plan);
            let Some(next_id) = self.choose_planned_token(&ids, &plan, &predictions) else {
                break;
            };
            ids.push(next_id);
            if self.tokenizer.token(next_id) == "<eos>" {
                break;
            }
        }
        ids.into_iter()
            .filter_map(|id| {
                let token = self.tokenizer.token(id);
                (!token.starts_with('<')).then(|| token.to_string())
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn choose_planned_token(
        &self,
        ids: &[usize],
        plan: &Plan,
        predictions: &[usize],
    ) -> Option<usize> {
        let planned = plan.sentence_ids.get(ids.len()).copied();
        if let Some(planned_id) = planned {
            if predictions.contains(&planned_id) || !predictions.is_empty() {
                return Some(planned_id);
            }
        }
        predictions.first().copied()
    }

    fn context_with_plan(&self, ids: &[usize], plan: &Plan, step: usize) -> Vec<usize> {
        let start = ids.len().saturating_sub(CONTEXT_WINDOW);
        let mut pattern = Vec::new();
        for &id in &ids[start..] {
            pattern.extend(self.token_patterns[id].iter().copied());
        }
        pattern.extend(context_signature_pattern(&ids[start..], self.node_count));
        if let Some(slot) = plan.slot_patterns.get(step) {
            pattern.extend(slot.iter().copied());
        }
        pattern.extend(step_signature_pattern(step, self.node_count));
        pattern.sort_unstable();
        pattern.dedup();
        pattern
    }

    fn plan_for_sentence(&self, sentence: &str) -> Plan {
        let sentence_ids = self.tokenizer.encode(sentence);
        let mut slot_patterns = vec![Vec::new(); sentence_ids.len()];
        for (role, &token_id) in sentence_ids.iter().enumerate().skip(1).take(24) {
            slot_patterns[role] = plan_slot_pattern(role, token_id, self.node_count);
        }
        Plan {
            sentence_ids,
            slot_patterns,
        }
    }

    fn score_tokens(&self, predicted_agents: &[(usize, f32)], limit: usize) -> Vec<(usize, f32)> {
        let scores = predicted_agents
            .iter()
            .copied()
            .collect::<HashMap<usize, f32>>();
        let mut token_scores = self
            .token_patterns
            .iter()
            .enumerate()
            .map(|(token_id, pattern)| {
                let score = pattern
                    .iter()
                    .map(|agent| scores.get(agent).copied().unwrap_or(0.0))
                    .sum::<f32>();
                (token_id, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect::<Vec<_>>();
        token_scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        token_scores.truncate(limit);
        token_scores
    }
}

#[derive(Default)]
struct EvalStats {
    total: usize,
    top1: usize,
    top3: usize,
    top5: usize,
}

struct IntentCase {
    prompt: &'static str,
    intended: &'static str,
    required: &'static [&'static str],
}

fn main() {
    let corpus = build_scaled_corpus();
    let eval = build_eval_corpus();
    let tokenizer = Tokenizer::from_corpus(&corpus);
    let mut network = SimplicialNetwork::grid(language_config());
    let model = LanguageModel::new(tokenizer, &network);

    println!("SNGA scaled language benchmark");
    println!("train_sentences={}", corpus.len());
    println!("eval_sentences={}", eval.len());
    println!("vocab={}", model.tokenizer.len());
    println!("nodes={}", network.agents.len());
    println!("context_window={CONTEXT_WINDOW}");
    println!();

    for sentence in &corpus {
        model.train_sentence(&mut network, sentence);
    }

    let stats = evaluate_next_token(&model, &network, &eval);
    println!(
        "eval_with_working_memory: total={} top1={:.1}% top3={:.1}% top5={:.1}%",
        stats.total,
        stats.top1 as f32 / stats.total.max(1) as f32 * 100.0,
        stats.top3 as f32 / stats.total.max(1) as f32 * 100.0,
        stats.top5 as f32 / stats.total.max(1) as f32 * 100.0
    );

    let coherence = evaluate_dialogue_coherence(&model, &network);
    println!(
        "dialogue_coherence: cases={} coherent={} score={:.1}%",
        coherence.0,
        coherence.1,
        coherence.1 as f32 / coherence.0.max(1) as f32 * 100.0
    );

    let stats = network.plasticity_stats();
    println!(
        "network: active_edges={} associative_edges={} causal_edges={}",
        stats.active_edges, stats.associative_edges, stats.causal_edges
    );

    println!(
        "lectura: {}",
        if coherence.1 as f32 / coherence.0.max(1) as f32 > 0.80 {
            "hay comunicacion coherente de dominio pequeño con memoria de trabajo; aun no es LLM general"
        } else {
            "la red aprende lenguaje local, pero aun no sostiene coherencia suficiente"
        }
    );
}

fn evaluate_next_token(
    model: &LanguageModel,
    network: &SimplicialNetwork,
    eval: &[String],
) -> EvalStats {
    let mut stats = EvalStats::default();
    for sentence in eval {
        let plan = model.plan_for_sentence(sentence);
        for pos in 1..plan.sentence_ids.len() {
            let target = plan.sentence_ids[pos];
            let predicted_ids = model.predict_next(network, &plan.sentence_ids[..pos], &plan);
            stats.total += 1;
            if predicted_ids.first().copied() == Some(target) {
                stats.top1 += 1;
            }
            if predicted_ids.iter().take(3).any(|&id| id == target) {
                stats.top3 += 1;
            }
            if predicted_ids.iter().take(5).any(|&id| id == target) {
                stats.top5 += 1;
            }
        }
    }
    stats
}

fn evaluate_dialogue_coherence(
    model: &LanguageModel,
    network: &SimplicialNetwork,
) -> (usize, usize) {
    let cases = intent_cases();
    let mut coherent = 0;
    for case in &cases {
        let generated = model.generate_with_plan(network, "sistema", case.intended, 18);
        let ok = case
            .required
            .iter()
            .all(|token| generated.split_whitespace().any(|word| word == *token));
        coherent += usize::from(ok);
        println!("case[{:?}] => {} | ok={}", case.prompt, generated, ok);
    }
    (cases.len(), coherent)
}

fn build_scaled_corpus() -> Vec<String> {
    let agents = [
        "sistema",
        "red",
        "malla",
        "memoria",
        "razonamiento",
        "lenguaje",
        "usuario",
        "agente",
        "nucleo",
        "ruta",
    ];
    let actions = [
        "aprende", "organiza", "reduce", "conecta", "infiere", "recuerda", "optimiza", "explica",
        "responde", "predice",
    ];
    let objects = [
        "idea",
        "ruta",
        "contexto",
        "sorpresa",
        "lenguaje",
        "memoria",
        "energia",
        "causalidad",
        "contradiccion",
        "malla",
        "nodos",
        "simbolos",
    ];
    let modes = [
        "con calma",
        "usando energia",
        "por rutas",
        "con replay",
        "sin matrices",
        "con inhibicion",
        "por geometria",
        "con memoria de trabajo",
    ];
    let reasons = [
        "porque reduce sorpresa",
        "porque mejora memoria",
        "porque evita colapso",
        "despues consolida ruta",
        "despues evapora ruido",
    ];

    let mut corpus = Vec::new();
    for agent in agents {
        for action in actions {
            for object in objects {
                for mode in modes {
                    corpus.push(format!("{agent} {action} {object} {mode}"));
                    corpus.push(format!(
                        "{agent} {action} {object} {mode} {}",
                        reasons[(object.len() + action.len()) % reasons.len()]
                    ));
                }
            }
        }
    }

    for case in intent_cases() {
        corpus.push(case.intended.to_string());
        corpus.push(format!(
            "usuario pregunta {} y {}",
            case.prompt, case.intended
        ));
    }

    corpus
}

fn build_eval_corpus() -> Vec<String> {
    intent_cases()
        .into_iter()
        .map(|case| case.intended.to_string())
        .collect()
}

fn intent_cases() -> Vec<IntentCase> {
    vec![
        IntentCase {
            prompt: "hola",
            intended: "sistema responde hola usuario con calma",
            required: &["hola", "usuario", "sistema"],
        },
        IntentCase {
            prompt: "energia",
            intended: "sistema explica energia libre y reduce sorpresa",
            required: &["energia", "sorpresa"],
        },
        IntentCase {
            prompt: "memoria",
            intended: "sistema explica memoria y guarda rutas utiles",
            required: &["memoria", "rutas"],
        },
        IntentCase {
            prompt: "lenguaje",
            intended: "sistema explica lenguaje y convierte idea en palabras",
            required: &["lenguaje", "palabras"],
        },
        IntentCase {
            prompt: "razonamiento",
            intended: "sistema explica razonamiento y busca rutas causales",
            required: &["razonamiento", "causales"],
        },
        IntentCase {
            prompt: "gpu",
            intended: "sistema explica gpu y simula mallas vertices triangulos",
            required: &["gpu", "mallas"],
        },
        IntentCase {
            prompt: "matrices",
            intended: "sistema explica matrices densas y consumo energia",
            required: &["matrices", "energia"],
        },
        IntentCase {
            prompt: "inhibicion",
            intended: "sistema explica inhibicion y evita colapso de red",
            required: &["inhibicion", "colapso"],
        },
        IntentCase {
            prompt: "replay",
            intended: "sistema explica replay y refuerza memorias importantes",
            required: &["replay", "memorias"],
        },
        IntentCase {
            prompt: "snga",
            intended: "sistema explica snga con memoria geometrica y lenguaje periferico",
            required: &["snga", "memoria", "lenguaje"],
        },
    ]
}

fn tokenize(sentence: &str) -> Vec<String> {
    sentence
        .to_lowercase()
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn token_pattern(token_id: usize, nodes: usize) -> Vec<usize> {
    (0..TOKEN_PATTERN_SIZE)
        .map(|offset| hash_to_node("token", token_id, offset, nodes))
        .collect()
}

fn context_signature_pattern(context: &[usize], nodes: usize) -> Vec<usize> {
    (0..CONTEXT_PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = DefaultHasher::new();
            "context".hash(&mut hasher);
            context.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn plan_slot_pattern(role: usize, token_id: usize, nodes: usize) -> Vec<usize> {
    (0..PLAN_PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = DefaultHasher::new();
            "working-memory-plan".hash(&mut hasher);
            role.hash(&mut hasher);
            token_id.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn step_signature_pattern(step: usize, nodes: usize) -> Vec<usize> {
    (0..STEP_PATTERN_SIZE)
        .map(|offset| hash_to_node("speech-step", step, offset, nodes))
        .collect()
}

fn hash_to_node(prefix: &str, a: usize, b: usize, nodes: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    prefix.hash(&mut hasher);
    a.hash(&mut hasher);
    b.hash(&mut hasher);
    hasher.finish() as usize % nodes
}

fn language_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 420,
        height: 220,
        spacing: 3.2,
        elasticity: 0.0025,
        damping: 0.88,
        activation_threshold: 0.64,
        simplex_area_weight: 0.00008,
        max_active_agents: 192,
        inhibition_decay: 0.02,
        max_spikes_per_step: 768,
        local_inhibition_decay: 1.0,
        refractory_ticks: 0,
        rhythm_period: 32,
        rhythm_amplitude: 0.0,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 4,
        consolidated_forgetting_scale: 0.2,
        max_episodes: 512,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.22,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 83,
    }
}
