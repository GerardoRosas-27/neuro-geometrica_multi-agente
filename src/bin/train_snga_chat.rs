use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::hash::{Hash, Hasher};

const TOKEN_PATTERN_SIZE: usize = 11;
const CONTEXT_PATTERN_SIZE: usize = 13;
const PLAN_PATTERN_SIZE: usize = 23;
const STEP_PATTERN_SIZE: usize = 11;
const CONTEXT_WINDOW: usize = 2;
const CHAT_MEMORY_PATH: &str = "data/chat_memory.txt";

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

    fn len(&self) -> usize {
        self.id_to_token.len()
    }
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
        let mut slot_patterns = vec![Vec::new(); ids.len()];
        for (role, &token_id) in ids.iter().enumerate().skip(1).take(16) {
            slot_patterns[role] = plan_slot_pattern(role, token_id, self.node_count);
        }

        for pos in 1..ids.len() {
            let mut context = self.context_pattern(&ids[..pos]);
            if let Some(slot) = slot_patterns.get(pos) {
                context.extend(slot.iter().copied());
            }
            context.extend(step_signature_pattern(pos, self.node_count));
            context.sort_unstable();
            context.dedup();
            let next = &self.token_patterns[ids[pos]];
            network.learn_transition(&context, next);
            network.reinforce_coactivation(next, 0.04);
        }
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
}

fn main() {
    let corpus = expanded_chat_corpus();
    fs::create_dir_all("data").expect("crear data/");
    fs::write(CHAT_MEMORY_PATH, corpus.join("\n")).expect("guardar memoria del chat");

    let tokenizer = Tokenizer::from_corpus(&corpus);
    let mut network = SimplicialNetwork::grid(chat_config());
    let model = LanguageModel::new(tokenizer, &network);

    for sentence in &corpus {
        model.train_sentence(&mut network, sentence);
    }

    let stats = network.plasticity_stats();
    println!("SNGA chat training complete");
    println!("memory_path={CHAT_MEMORY_PATH}");
    println!("sentences={}", corpus.len());
    println!("vocab={}", model.tokenizer.len());
    println!("nodes={}", network.agents.len());
    println!(
        "active_edges={} associative_edges={} causal_edges={} episodes={}",
        stats.active_edges, stats.associative_edges, stats.causal_edges, stats.episodes
    );
}

fn expanded_chat_corpus() -> Vec<String> {
    let mut corpus = vec![
        "sistema responde saludo con calma",
        "sistema reduce sorpresa y busca ruta estable",
        "sistema aprende memoria con rutas y replay",
        "sistema organiza idea abstracta antes de hablar",
        "sistema usa rutas causales para inferir ideas",
        "sistema evita matrices densas y usa malla espacial",
        "sistema recibe idea nueva y aprende contexto",
        "hola usuario soy sistema neuro geometrico",
        "energia libre significa reducir sorpresa",
        "memoria significa guardar rutas utiles",
        "lenguaje significa convertir idea en palabras",
        "razonamiento significa buscar rutas causales",
        "gpu puede simular mallas vertices y triangulos",
        "matrices densas consumen mas energia",
        "malla espacial usa nodos aristas y simplices",
        "inhibicion evita colapso de la red",
        "replay refuerza memorias importantes",
        "contradiccion aumenta energia libre",
        "ruta optima se refuerza y ruta debil se evapora",
        "snga usa memoria geometrica y lenguaje periferico",
        "red binaria aprende con coactivacion",
        "idea abstracta se organiza antes de hablar",
        "sistema no usa transformer en este experimento",
        "sistema responde con frases simples aprendidas",
        "si preguntas energia sistema explica sorpresa",
        "si preguntas memoria sistema explica rutas",
        "si preguntas lenguaje sistema explica palabras",
        "si preguntas razonamiento sistema explica causalidad",
        "si preguntas gpu sistema explica mallas",
        "si preguntas matrices sistema explica costo",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();

    let agents = [
        "sistema",
        "red",
        "malla",
        "memoria",
        "razonamiento",
        "lenguaje",
    ];
    let actions = [
        "aprende", "organiza", "reduce", "conecta", "infiere", "recuerda", "optimiza",
    ];
    let objects = [
        "idea", "ruta", "contexto", "sorpresa", "lenguaje", "memoria", "energia",
    ];
    let modes = [
        "con calma",
        "usando energia",
        "por rutas",
        "con replay",
        "sin matrices",
    ];
    let topics = [
        "energia",
        "memoria",
        "lenguaje",
        "razonamiento",
        "matrices",
        "rutas",
    ];

    for agent in agents {
        for action in actions {
            for object in objects {
                for mode in modes {
                    corpus.push(format!("{agent} {action} {object} {mode}"));
                    corpus.push(format!(
                        "usuario pregunta sobre {object} y {agent} {action} {object}"
                    ));
                }
            }
        }
    }

    for topic in topics {
        corpus.push(format!("usuario pregunta sobre {topic}"));
        corpus.push(format!("sistema responde sobre {topic} con idea clara"));
        corpus.push(format!("sistema explica {topic} usando malla espacial"));
    }

    corpus
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

fn chat_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 260,
        height: 140,
        spacing: 4.0,
        elasticity: 0.003,
        damping: 0.88,
        activation_threshold: 0.64,
        simplex_area_weight: 0.00008,
        max_active_agents: 128,
        inhibition_decay: 0.02,
        max_spikes_per_step: 512,
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
        seed: 73,
    }
}
