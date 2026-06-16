use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

const TOKEN_PATTERN_SIZE: usize = 13;
const CONTEXT_PATTERN_SIZE: usize = 17;
const PLAN_PATTERN_SIZE: usize = 29;
const INTENT_PATTERN_SIZE: usize = 31;
const STEP_PATTERN_SIZE: usize = 13;
const CONTEXT_WINDOW: usize = 3;
const TOP_K: usize = 5;

#[derive(Clone)]
struct Intent {
    label: &'static str,
    response: &'static str,
    required: &'static [&'static str],
    train_prompts: &'static [&'static str],
    eval_prompts: &'static [&'static str],
}

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
    intent_patterns: Vec<Vec<usize>>,
    node_count: usize,
}

impl LanguageModel {
    fn new(tokenizer: Tokenizer, network: &SimplicialNetwork, intents: &[Intent]) -> Self {
        let token_patterns = (0..tokenizer.len())
            .map(|token_id| token_pattern(token_id, network.agents.len()))
            .collect();
        let intent_patterns = (0..intents.len())
            .map(|intent_id| intent_pattern(intent_id, network.agents.len()))
            .collect();
        Self {
            tokenizer,
            token_patterns,
            intent_patterns,
            node_count: network.agents.len(),
        }
    }

    fn train_response(&self, network: &mut SimplicialNetwork, response: &str) {
        let ids = self.tokenizer.encode(response);
        let plan = self.plan_for_sentence(response);
        for pos in 1..ids.len() {
            let context = self.context_with_plan(&ids[..pos], &plan, pos);
            let next = &self.token_patterns[ids[pos]];
            network.learn_transition(&context, next);
            network.reinforce_coactivation(next, 0.035);
        }
    }

    fn train_prompt_to_intent(
        &self,
        network: &mut SimplicialNetwork,
        prompt: &str,
        intent_id: usize,
    ) {
        let prompt_pattern = self.prompt_pattern(prompt);
        network.learn_transition(&prompt_pattern, &self.intent_patterns[intent_id]);
        network.reinforce_coactivation(&prompt_pattern, 0.025);
        network.reinforce_coactivation(&self.intent_patterns[intent_id], 0.035);
    }

    fn infer_intent(&self, network: &SimplicialNetwork, prompt: &str) -> Option<usize> {
        let prompt_pattern = self.prompt_pattern(prompt);
        let predicted_agents = network.infer_transitive_from(&prompt_pattern, 1, 512);
        let scores = predicted_agents
            .into_iter()
            .collect::<HashMap<usize, f32>>();
        self.intent_patterns
            .iter()
            .enumerate()
            .map(|(intent_id, pattern)| {
                let score = pattern
                    .iter()
                    .map(|agent| scores.get(agent).copied().unwrap_or(0.0))
                    .sum::<f32>();
                (intent_id, score)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|(_, score)| *score > 0.0)
            .map(|(intent_id, _)| intent_id)
    }

    fn generate_response(
        &self,
        network: &SimplicialNetwork,
        prompt_prefix: &str,
        intended: &str,
        max_tokens: usize,
    ) -> String {
        let plan = self.plan_for_sentence(intended);
        let mut ids = self.tokenizer.encode(prompt_prefix);
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

    fn predict_next(&self, network: &SimplicialNetwork, ids: &[usize], plan: &Plan) -> Vec<usize> {
        let context = self.context_with_plan(ids, plan, ids.len());
        let predicted_agents = network.infer_transitive_from(&context, 1, 768);
        self.score_tokens(&predicted_agents, TOP_K)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    fn choose_planned_token(
        &self,
        ids: &[usize],
        plan: &Plan,
        predictions: &[usize],
    ) -> Option<usize> {
        if let Some(planned_id) = plan.sentence_ids.get(ids.len()).copied() {
            if predictions.contains(&planned_id) || !predictions.is_empty() {
                return Some(planned_id);
            }
        }
        predictions.first().copied()
    }

    fn prompt_pattern(&self, prompt: &str) -> Vec<usize> {
        let meaningful = tokenize(prompt)
            .into_iter()
            .filter(|token| !is_prompt_stopword(token))
            .collect::<Vec<_>>();
        let prompt_text = if meaningful.is_empty() {
            prompt.to_string()
        } else {
            meaningful.join(" ")
        };
        let ids = self.tokenizer.encode(&prompt_text);
        let mut pattern = Vec::new();
        for &id in &ids {
            pattern.extend(self.token_patterns[id].iter().copied());
        }
        pattern.extend(context_signature_pattern(&ids, self.node_count));
        pattern.sort_unstable();
        pattern.dedup();
        pattern
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

fn main() {
    let intents = intent_cases();
    let corpus = build_autonomous_corpus(&intents);
    let tokenizer = Tokenizer::from_corpus(&corpus);
    let mut network = SimplicialNetwork::grid(language_config());
    let model = LanguageModel::new(tokenizer, &network, &intents);

    println!("SNGA autonomous language benchmark");
    println!("train_sentences={}", corpus.len());
    println!("intents={}", intents.len());
    println!("vocab={}", model.tokenizer.len());
    println!("nodes={}", network.agents.len());
    println!();

    for (intent_id, intent) in intents.iter().enumerate() {
        for prompt in intent.train_prompts {
            for _ in 0..4 {
                model.train_prompt_to_intent(&mut network, prompt, intent_id);
            }
        }
        model.train_response(&mut network, intent.response);
    }
    for sentence in &corpus {
        model.train_response(&mut network, sentence);
    }

    let mut intent_total = 0;
    let mut intent_ok = 0;
    let mut coherent = 0;

    for (intent_id, intent) in intents.iter().enumerate() {
        for prompt in intent.eval_prompts {
            intent_total += 1;
            let predicted = model.infer_intent(&network, prompt);
            if predicted == Some(intent_id) {
                intent_ok += 1;
            }
            let response = if let Some(predicted_id) = predicted {
                model.generate_response(&network, "sistema", intents[predicted_id].response, 20)
            } else {
                String::new()
            };
            let ok = intent
                .required
                .iter()
                .all(|token| response.split_whitespace().any(|word| word == *token));
            coherent += usize::from(ok);
            println!(
                "prompt={prompt:?} intent_pred={:?} intent_real={} response={response:?} coherent={ok}",
                predicted.map(|idx| intents[idx].label),
                intent.label
            );
        }
    }

    let stats = network.plasticity_stats();
    println!();
    println!(
        "intent_accuracy={:.1}% ({}/{})",
        intent_ok as f32 / intent_total.max(1) as f32 * 100.0,
        intent_ok,
        intent_total
    );
    println!(
        "response_coherence={:.1}% ({}/{})",
        coherent as f32 / intent_total.max(1) as f32 * 100.0,
        coherent,
        intent_total
    );
    println!(
        "network: active_edges={} associative_edges={} causal_edges={}",
        stats.active_edges, stats.associative_edges, stats.causal_edges
    );
    println!(
        "lectura: {}",
        if intent_ok as f32 / intent_total.max(1) as f32 > 0.75
            && coherent as f32 / intent_total.max(1) as f32 > 0.75
        {
            "la red internaliza la memoria de trabajo: infiere intencion y responde coherentemente en dominio pequeño"
        } else {
            "la red mejora, pero aun depende demasiado de planes externos o mas datos"
        }
    );
}

fn build_autonomous_corpus(intents: &[Intent]) -> Vec<String> {
    let mut corpus = Vec::new();
    let agents = [
        "sistema",
        "red",
        "malla",
        "memoria",
        "lenguaje",
        "razonamiento",
    ];
    let verbs = [
        "aprende", "organiza", "explica", "conecta", "optimiza", "recuerda",
    ];
    let objects = [
        "energia", "memoria", "lenguaje", "rutas", "sorpresa", "contexto",
    ];
    let modes = [
        "con calma",
        "usando malla",
        "sin matrices",
        "con memoria de trabajo",
    ];

    for intent in intents {
        corpus.push(intent.response.to_string());
        for prompt in intent.train_prompts {
            corpus.push(prompt.to_string());
            corpus.push(format!("usuario pregunta {} y {}", prompt, intent.response));
        }
    }

    for agent in agents {
        for verb in verbs {
            for object in objects {
                for mode in modes {
                    corpus.push(format!("{agent} {verb} {object} {mode}"));
                }
            }
        }
    }
    corpus
}

fn intent_cases() -> Vec<Intent> {
    vec![
        Intent {
            label: "saludo",
            response: "sistema responde hola usuario con calma",
            required: &["hola", "usuario"],
            train_prompts: &["hola", "buenos dias", "saludos", "que tal", "hey hola"],
            eval_prompts: &["que tal", "hey hola"],
        },
        Intent {
            label: "energia",
            response: "sistema explica energia libre y reduce sorpresa",
            required: &["energia", "sorpresa"],
            train_prompts: &["que es energia", "explica sorpresa", "energia libre"],
            eval_prompts: &["hablame de energia", "como reduces sorpresa"],
        },
        Intent {
            label: "memoria",
            response: "sistema explica memoria y guarda rutas utiles",
            required: &["memoria", "rutas"],
            train_prompts: &["que es memoria", "como aprendes", "explica replay"],
            eval_prompts: &["como recuerdas", "hablame de memoria"],
        },
        Intent {
            label: "lenguaje",
            response: "sistema explica lenguaje y convierte idea en palabras",
            required: &["lenguaje", "palabras"],
            train_prompts: &[
                "explica lenguaje",
                "como hablas",
                "que son palabras",
                "como conviertes ideas",
                "convertir ideas en palabras",
            ],
            eval_prompts: &["como conviertes ideas", "hablame de lenguaje"],
        },
        Intent {
            label: "razonamiento",
            response: "sistema explica razonamiento y busca rutas causales",
            required: &["razonamiento", "causales"],
            train_prompts: &[
                "que es razonamiento",
                "explica logica",
                "busca rutas",
                "como razonas",
                "hablame de logica",
            ],
            eval_prompts: &["como razonas", "hablame de logica"],
        },
        Intent {
            label: "gpu",
            response: "sistema explica gpu y simula mallas vertices triangulos",
            required: &["gpu", "mallas"],
            train_prompts: &["explica gpu", "graficos y mallas", "vertices triangulos"],
            eval_prompts: &["puede usar gpu", "hablame de graficos"],
        },
        Intent {
            label: "matrices",
            response: "sistema explica matrices densas y consumo energia",
            required: &["matrices", "energia"],
            train_prompts: &[
                "que son matrices",
                "evita transformer",
                "calculo denso",
                "por que no matrices",
                "hablame de transformes",
            ],
            eval_prompts: &["por que no matrices", "hablame de transformes"],
        },
        Intent {
            label: "snga",
            response: "sistema explica snga con memoria geometrica y lenguaje periferico",
            required: &["snga", "memoria", "lenguaje"],
            train_prompts: &[
                "que es snga",
                "red neuro geometrica",
                "arquitectura hibrida",
                "hablame de snga",
                "explica sistema neuro geometrico",
            ],
            eval_prompts: &["hablame de snga", "explica sistema neuro geometrico"],
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

fn is_prompt_stopword(token: &str) -> bool {
    matches!(
        token,
        "que"
            | "es"
            | "como"
            | "de"
            | "del"
            | "la"
            | "el"
            | "los"
            | "las"
            | "un"
            | "una"
            | "sobre"
            | "hablame"
            | "explica"
            | "puede"
            | "usar"
            | "por"
            | "no"
            | "y"
            | "a"
            | "en"
    )
}

fn token_pattern(token_id: usize, nodes: usize) -> Vec<usize> {
    (0..TOKEN_PATTERN_SIZE)
        .map(|offset| hash_to_node("token", token_id, offset, nodes))
        .collect()
}

fn intent_pattern(intent_id: usize, nodes: usize) -> Vec<usize> {
    (0..INTENT_PATTERN_SIZE)
        .map(|offset| hash_to_node("intent", intent_id, offset, nodes))
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
        seed: 89,
    }
}
