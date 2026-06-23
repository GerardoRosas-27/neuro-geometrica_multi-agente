use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

const TOKEN_PATTERN_SIZE: usize = 11;
const CONTEXT_PATTERN_SIZE: usize = 13;
const PLAN_PATTERN_SIZE: usize = 23;
const STEP_PATTERN_SIZE: usize = 11;
const CONTEXT_WINDOW: usize = 2;
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

struct LanguageModel {
    tokenizer: Tokenizer,
    token_patterns: Vec<Vec<usize>>,
    node_count: usize,
}

#[derive(Clone)]
struct WorkingMemoryPlan {
    sentence_ids: Vec<usize>,
    slot_patterns: Vec<Vec<usize>>,
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
        for pos in 1..ids.len() {
            let context = self.context_pattern(&ids[..pos]);
            let next = &self.token_patterns[ids[pos]];
            network.learn_transition(&context, next);
            network.reinforce_coactivation(next, 0.04);
        }
    }

    fn train_sentence_with_plan(&self, network: &mut SimplicialNetwork, sentence: &str) {
        let plan = self.plan_for_sentence(sentence);
        for pos in 1..plan.sentence_ids.len() {
            let context = self.context_pattern_with_plan(&plan.sentence_ids[..pos], &plan, pos);
            let next = &self.token_patterns[plan.sentence_ids[pos]];
            network.learn_transition(&context, next);
            network.reinforce_coactivation(next, 0.04);
        }
    }

    fn predict_next(
        &self,
        network: &SimplicialNetwork,
        context_ids: &[usize],
    ) -> Vec<(usize, f32)> {
        let context = self.context_pattern(context_ids);
        let predicted_agents = network.infer_transitive_from(&context, 1, 256);
        self.score_tokens(&predicted_agents, TOP_K)
    }

    fn predict_next_with_plan(
        &self,
        network: &SimplicialNetwork,
        context_ids: &[usize],
        plan: &WorkingMemoryPlan,
    ) -> Vec<(usize, f32)> {
        let context = self.context_pattern_with_plan(context_ids, plan, context_ids.len());
        let predicted_agents = network.infer_transitive_from(&context, 1, 384);
        self.score_tokens(&predicted_agents, TOP_K)
    }

    fn context_pattern(&self, ids: &[usize]) -> Vec<usize> {
        let start = ids.len().saturating_sub(CONTEXT_WINDOW);
        let mut pattern = Vec::new();
        for &id in &ids[start..] {
            pattern.extend(self.token_patterns[id].iter().copied());
        }
        pattern.extend(context_signature_pattern(&ids[start..], self.node_count));
        pattern.sort_unstable();
        pattern.dedup();
        pattern
    }

    fn context_pattern_with_plan(
        &self,
        ids: &[usize],
        plan: &WorkingMemoryPlan,
        step: usize,
    ) -> Vec<usize> {
        let mut pattern = self.context_pattern(ids);
        if let Some(slot) = plan.slot_patterns.get(step) {
            pattern.extend(slot.iter().copied());
        }
        pattern.extend(step_signature_pattern(step, self.node_count));
        pattern.sort_unstable();
        pattern.dedup();
        pattern
    }

    fn plan_for_sentence(&self, sentence: &str) -> WorkingMemoryPlan {
        let sentence_ids = self.tokenizer.encode(sentence);
        let mut slot_patterns = vec![Vec::new(); sentence_ids.len()];
        for (role, &token_id) in sentence_ids.iter().enumerate().skip(1).take(16) {
            slot_patterns[role] = plan_slot_pattern(role, token_id, self.node_count);
        }
        WorkingMemoryPlan {
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

    fn generate(
        &self,
        network: &SimplicialNetwork,
        prompt: &str,
        max_tokens: usize,
    ) -> Vec<String> {
        let mut ids = self.tokenizer.encode(prompt);
        ids.pop(); // remove <eos> while generating

        for _ in 0..max_tokens {
            let predictions = self.predict_next(network, &ids);
            let Some((next_id, _)) = predictions.first().copied() else {
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
                if token.starts_with('<') {
                    None
                } else {
                    Some(token.to_string())
                }
            })
            .collect()
    }

    fn generate_with_plan(
        &self,
        network: &SimplicialNetwork,
        prompt: &str,
        intended_sentence: &str,
        max_tokens: usize,
    ) -> Vec<String> {
        let plan = self.plan_for_sentence(intended_sentence);
        let mut ids = self.tokenizer.encode(prompt);
        ids.pop();

        for _ in 0..max_tokens {
            let predictions = self.predict_next_with_plan(network, &ids, &plan);
            let Some((next_id, _)) = predictions.first().copied() else {
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
                if token.starts_with('<') {
                    None
                } else {
                    Some(token.to_string())
                }
            })
            .collect()
    }
}

#[derive(Default)]
struct EvalStats {
    total: usize,
    top1: usize,
    top3: usize,
    top5: usize,
}

fn main() {
    let train = build_train_corpus();
    let eval = build_eval_corpus();
    let tokenizer = Tokenizer::from_corpus(&train);
    let mut network = SimplicialNetwork::grid(language_config());
    let model = LanguageModel::new(tokenizer, &network);

    println!("SNGA language tokenizer experiment");
    println!("train_sentences={}", train.len());
    println!("eval_sentences={}", eval.len());
    println!("vocab={}", model.tokenizer.len());
    println!("nodes={}", network.agents.len());
    println!("context_window={CONTEXT_WINDOW}");
    println!();

    for sentence in &train {
        model.train_sentence(&mut network, sentence);
        model.train_sentence_with_plan(&mut network, sentence);
    }

    let train_sample = train.iter().take(32).cloned().collect::<Vec<_>>();
    let train_stats = evaluate_language(&model, &network, &train_sample);
    let stats = evaluate_language(&model, &network, &eval);
    let plan_stats = evaluate_language_with_plan(&model, &network, &eval);
    println!(
        "train_sample_next_token: total={} top1={:.1}% top3={:.1}% top5={:.1}%",
        train_stats.total,
        train_stats.top1 as f32 / train_stats.total.max(1) as f32 * 100.0,
        train_stats.top3 as f32 / train_stats.total.max(1) as f32 * 100.0,
        train_stats.top5 as f32 / train_stats.total.max(1) as f32 * 100.0
    );
    println!(
        "eval_next_token: total={} top1={:.1}% top3={:.1}% top5={:.1}%",
        stats.total,
        stats.top1 as f32 / stats.total.max(1) as f32 * 100.0,
        stats.top3 as f32 / stats.total.max(1) as f32 * 100.0,
        stats.top5 as f32 / stats.total.max(1) as f32 * 100.0
    );
    println!(
        "eval_with_working_memory: total={} top1={:.1}% top3={:.1}% top5={:.1}%",
        plan_stats.total,
        plan_stats.top1 as f32 / plan_stats.total.max(1) as f32 * 100.0,
        plan_stats.top3 as f32 / plan_stats.total.max(1) as f32 * 100.0,
        plan_stats.top5 as f32 / plan_stats.total.max(1) as f32 * 100.0
    );

    for prompt in ["el perro", "la niña", "un robot", "una maquina"] {
        let generated = model.generate(&network, prompt, 6).join(" ");
        println!("generacion[{prompt:?}] => {generated}");
    }

    for (prompt, plan) in [
        ("el perro", "el perro mira pelota en cocina"),
        ("la niña", "la niña encuentra luz en jardin"),
        ("un robot", "un robot empuja piedra en taller"),
    ] {
        let generated = model
            .generate_with_plan(&network, prompt, plan, 6)
            .join(" ");
        println!("generacion_con_plan[{prompt:?} | idea={plan:?}] => {generated}");
    }

    println!(
        "lectura: {}",
        if plan_stats.top1 as f32 / plan_stats.total.max(1) as f32 > 0.80 {
            "la memoria de trabajo mejora la verbalizacion: la red ordena una idea abstracta antes de hablar"
        } else {
            "aprende transiciones, pero la memoria de trabajo aun no organiza suficiente la salida"
        }
    );
}

fn evaluate_language(
    model: &LanguageModel,
    network: &SimplicialNetwork,
    eval: &[String],
) -> EvalStats {
    let mut stats = EvalStats::default();
    for sentence in eval {
        let ids = model.tokenizer.encode(sentence);
        for pos in 1..ids.len() {
            let target = ids[pos];
            let predictions = model.predict_next(network, &ids[..pos]);
            let predicted_ids = predictions
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>();
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

fn evaluate_language_with_plan(
    model: &LanguageModel,
    network: &SimplicialNetwork,
    eval: &[String],
) -> EvalStats {
    let mut stats = EvalStats::default();
    for sentence in eval {
        let plan = model.plan_for_sentence(sentence);
        for pos in 1..plan.sentence_ids.len() {
            let target = plan.sentence_ids[pos];
            let predictions =
                model.predict_next_with_plan(network, &plan.sentence_ids[..pos], &plan);
            let predicted_ids = predictions
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>();
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

fn build_train_corpus() -> Vec<String> {
    let subjects = [
        "perro",
        "gato",
        "robot",
        "niño",
        "niña",
        "ave",
        "pez",
        "maquina",
        "maestro",
        "doctora",
        "campesino",
        "artista",
    ];
    let determiners = ["el", "la", "un", "una"];
    let adjectives = ["rojo", "azul", "rapido", "lento", "pequeño", "grande"];
    let verbs = [
        "come",
        "mira",
        "toca",
        "sigue",
        "empuja",
        "encuentra",
        "levanta",
        "observa",
    ];
    let objects = [
        "manzana", "pelota", "piedra", "casa", "ruta", "luz", "agua", "fuego", "libro", "puerta",
    ];
    let places = [
        "parque", "cocina", "calle", "jardin", "taller", "rio", "escuela", "mercado",
    ];
    let adverbs = ["despacio", "rapido", "cuidadosamente", "silenciosamente"];
    let outcomes = [
        "aprende", "descansa", "avanza", "sonrie", "espera", "responde", "cambia",
    ];

    let mut corpus = Vec::new();
    for (i, subject) in subjects.iter().enumerate() {
        for (j, verb) in verbs.iter().enumerate() {
            for (k, object) in objects.iter().enumerate() {
                let det = determiners[(i + k) % determiners.len()];
                let adj = adjectives[(i + j + k) % adjectives.len()];
                let place = places[(i + j + k) % places.len()];
                let adv = adverbs[(i + 2 * j + k) % adverbs.len()];
                let outcome = outcomes[(i + j + 2 * k) % outcomes.len()];
                corpus.push(format!("{det} {subject} {verb} {object} en {place}"));
                corpus.push(format!("{det} {subject} {verb} {object} {adj} en {place}"));
                corpus.push(format!("{det} {subject} {verb} {object} {adv} en {place}"));
                if (i + j + k) % 2 == 0 {
                    corpus.push(format!(
                        "{det} {subject} {verb} {object} en {place} porque {subject} {outcome}"
                    ));
                } else {
                    corpus.push(format!(
                        "{det} {subject} {verb} {object} en {place} despues {subject} {outcome}"
                    ));
                }
            }
        }
    }
    corpus
}

fn build_eval_corpus() -> Vec<String> {
    vec![
        "el perro come manzana rojo en parque".to_string(),
        "la niña mira pelota cuidadosamente en cocina".to_string(),
        "un robot empuja piedra en taller porque robot responde".to_string(),
        "una niña encuentra luz en jardin despues niña sonrie".to_string(),
        "el ave sigue ruta rapido en calle".to_string(),
        "un pez toca agua en rio porque pez avanza".to_string(),
        "la doctora observa libro azul en escuela".to_string(),
        "un campesino levanta puerta despacio en mercado".to_string(),
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
        .map(|offset| {
            let mut hasher = DefaultHasher::new();
            "token".hash(&mut hasher);
            token_id.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn context_signature_pattern(context: &[usize], nodes: usize) -> Vec<usize> {
    // La firma contextual actua como tokenizador temporal n-grama: distingue
    // "perro come" de "perro mira" sin usar matrices ni embeddings densos.
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
        .map(|offset| {
            let mut hasher = DefaultHasher::new();
            "speech-step".hash(&mut hasher);
            step.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn language_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 260,
        height: 140,
        spacing: 4.0,
        elasticity: 0.003,
        damping: 0.88,
        activation_threshold: 0.64,
        simplex_area_weight: 0.00008,
        max_active_agents: 96,
        inhibition_decay: 0.02,
        max_spikes_per_step: 384,
        local_inhibition_decay: 1.0,
        refractory_ticks: 0,
        rhythm_period: 32,
        rhythm_amplitude: 0.0,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 4,
        consolidated_forgetting_scale: 0.2,
        max_episodes: 256,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.22,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 61,
    }
}
