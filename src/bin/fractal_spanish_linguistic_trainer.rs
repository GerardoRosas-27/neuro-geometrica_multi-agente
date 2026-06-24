use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::hash::{Hash, Hasher};

const DEFAULT_BASE_STATE: &str = "data/snga_scaled_gemma_language_fractal_compressed.snga";
const DEFAULT_OUTPUT_STATE: &str = "data/snga_spanish_fractal_linguistic.snga";
const AGENT_COUNT: usize = 5_760;
const LETTER_PATTERN_SIZE: usize = 7;
const WORD_PATTERN_SIZE: usize = 11;
const UNION_PATTERN_SIZE: usize = 13;
const SENTENCE_PATTERN_SIZE: usize = 17;
const ROLE_PATTERN_SIZE: usize = 9;

#[derive(Clone)]
struct SpanishTokenizer {
    letters: Vec<char>,
    word_to_id: HashMap<String, usize>,
}

impl SpanishTokenizer {
    fn from_corpus(corpus: &[String]) -> Self {
        let mut letters = BTreeSet::new();
        let mut words = BTreeSet::new();
        letters.insert(' ');
        for sentence in corpus {
            for ch in normalized_chars(sentence) {
                letters.insert(ch);
            }
            for word in tokenize_words(sentence) {
                words.insert(word);
            }
        }
        let word_to_id = words
            .into_iter()
            .enumerate()
            .map(|(idx, word)| (word, idx))
            .collect();
        Self {
            letters: letters.into_iter().collect(),
            word_to_id,
        }
    }

    fn letter_id(&self, ch: char) -> usize {
        self.letters
            .binary_search(&ch)
            .unwrap_or_else(|_| self.letters.binary_search(&' ').unwrap_or(0))
    }

    fn word_id(&self, word: &str) -> usize {
        self.word_to_id.get(word).copied().unwrap_or(0)
    }

    fn words(&self, sentence: &str) -> Vec<String> {
        tokenize_words(sentence)
    }
}

fn main() {
    let epochs = arg_value("--epochs")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    let base_state = arg_value("--base").unwrap_or_else(|| DEFAULT_BASE_STATE.to_string());
    let output_state = arg_value("--output").unwrap_or_else(|| DEFAULT_OUTPUT_STATE.to_string());

    let corpus = spanish_curriculum();
    let tokenizer = SpanishTokenizer::from_corpus(&corpus);
    let mut network = SimplicialNetwork::fractal_3d(fractal_config(), fractal_mesh_config());
    let loaded = network.load_persistent_state(&base_state).is_ok();
    network.enable_neural_oscillations();

    println!("SNGA fractal Spanish linguistic trainer");
    println!(
        "base_loaded={} base={} output={} epochs={} nodes={} letters={} words={} sentences={}",
        loaded,
        base_state,
        output_state,
        epochs,
        network.agents.len(),
        tokenizer.letters.len(),
        tokenizer.word_to_id.len(),
        corpus.len()
    );

    for epoch in 0..epochs {
        train_letters(&mut network, &tokenizer);
        train_words(&mut network, &tokenizer, &corpus);
        train_word_unions(&mut network, &tokenizer, &corpus);
        train_sentences(&mut network, &tokenizer, &corpus);
        let adjusted = network.anneal_active_edge_rest_lengths(0.08 / (epoch as f32 + 1.0), 1.05);
        let probe = probe(&network, &tokenizer);
        let stats = network.plasticity_stats();
        println!(
            "epoch={} probe={}/{} conf={:.3} edges={} assoc={} causal={} adjusted={} energy={:.1}",
            epoch + 1,
            probe.nonzero,
            probe.total,
            probe.confidence,
            stats.active_edges,
            stats.associative_edges,
            stats.causal_edges,
            adjusted,
            network.total_free_energy()
        );
    }

    let stabilized = network.anneal_active_edge_rest_lengths(1.0, 0.0);
    let final_probe = probe(&network, &tokenizer);
    println!(
        "stabilized_edges={} final_probe={}/{} final_conf={:.3} final_energy={:.1}",
        stabilized,
        final_probe.nonzero,
        final_probe.total,
        final_probe.confidence,
        network.total_free_energy()
    );

    match network.save_persistent_state(&output_state) {
        Ok(report) => println!(
            "saved=true path={} agents={} edges={} causal_edges={}",
            output_state, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => println!("saved=false error={err}"),
    }
}

fn train_letters(network: &mut SimplicialNetwork, tokenizer: &SpanishTokenizer) {
    for (idx, &letter) in tokenizer.letters.iter().enumerate() {
        let letter_pattern = letter_pattern(idx, network.agents.len());
        let name_pattern = concept_pattern("letra", &letter.to_string(), network.agents.len());
        network.learn_transition(&name_pattern, &letter_pattern);
        network.reinforce_coactivation_if_useful(&letter_pattern, 0.035, 0.9);
    }
}

fn train_words(network: &mut SimplicialNetwork, tokenizer: &SpanishTokenizer, corpus: &[String]) {
    for sentence in corpus {
        for word in tokenizer.words(sentence) {
            let word_pattern = word_pattern(tokenizer.word_id(&word), network.agents.len());
            let chars = normalized_chars(&word);
            let mut previous = concept_pattern("inicio_palabra", &word, network.agents.len());
            for (pos, ch) in chars.iter().copied().enumerate() {
                let letter = letter_pattern(tokenizer.letter_id(ch), network.agents.len());
                let role = role_pattern("letra_pos", pos, network.agents.len());
                let mut context = previous.clone();
                context.extend(role);
                context.sort_unstable();
                context.dedup();
                network.learn_transition(&context, &letter);
                previous = letter;
            }
            network.learn_transition(&previous, &word_pattern);

            let mut fused = word_pattern.clone();
            for ch in chars {
                fused.extend(letter_pattern(
                    tokenizer.letter_id(ch),
                    network.agents.len(),
                ));
            }
            fused.sort_unstable();
            fused.dedup();
            network.reinforce_coactivation_if_useful(&fused, 0.04, 0.9);
        }
    }
}

fn train_word_unions(
    network: &mut SimplicialNetwork,
    tokenizer: &SpanishTokenizer,
    corpus: &[String],
) {
    for sentence in corpus {
        let words = tokenizer.words(sentence);
        for pair in words.windows(2) {
            let left = word_pattern(tokenizer.word_id(&pair[0]), network.agents.len());
            let right = word_pattern(tokenizer.word_id(&pair[1]), network.agents.len());
            let union = union_pattern(&pair[0], &pair[1], network.agents.len());
            network.learn_transition(&left, &union);
            network.learn_transition(&union, &right);
            network.learn_transition(&left, &right);
            let mut fused = left;
            fused.extend(right.iter().copied());
            fused.extend(union);
            fused.sort_unstable();
            fused.dedup();
            network.reinforce_coactivation_if_useful(&fused, 0.045, 0.9);
        }
    }
}

fn train_sentences(
    network: &mut SimplicialNetwork,
    tokenizer: &SpanishTokenizer,
    corpus: &[String],
) {
    for sentence in corpus {
        let words = tokenizer.words(sentence);
        if words.is_empty() {
            continue;
        }
        let sentence_pattern = sentence_pattern(sentence, network.agents.len());
        let mut context = concept_pattern("inicio_oracion", sentence, network.agents.len());
        context.extend(sentence_pattern.iter().copied());
        context.sort_unstable();
        context.dedup();

        for (pos, word) in words.iter().enumerate() {
            let next = word_pattern(tokenizer.word_id(word), network.agents.len());
            let role = role_pattern(sentence_role(pos, words.len()), pos, network.agents.len());
            context.extend(role);
            context.sort_unstable();
            context.dedup();
            network.learn_transition(&context, &next);

            if pos + 1 < words.len() {
                let next_union = union_pattern(word, &words[pos + 1], network.agents.len());
                network.learn_transition(&next, &next_union);
            }
            context = next;
        }

        network.learn_transition(&context, &sentence_pattern);
        network.reinforce_coactivation_if_useful(&sentence_pattern, 0.05, 0.9);
        network.clear_activity();
        network.inject_pattern(&sentence_pattern, 0.8, 2);
        for _ in 0..3 {
            network.step();
        }
        network.clear_activity();
    }
}

struct ProbeStats {
    total: usize,
    nonzero: usize,
    confidence: f32,
}

fn probe(network: &SimplicialNetwork, tokenizer: &SpanishTokenizer) -> ProbeStats {
    let cases = [
        ("palabra", "una palabra une letras y significado"),
        ("oracion", "una oracion organiza sujeto verbo y objeto"),
        ("causa", "una causa explica porque ocurre un efecto"),
        ("tiempo", "el pasado presente y futuro ordenan una accion"),
        ("pregunta", "una pregunta busca una respuesta clara"),
    ];
    let mut nonzero = 0;
    let mut confidence = 0.0;
    for (topic, sentence) in cases {
        let cue = concept_pattern("tema", topic, network.agents.len());
        let predicted = network.predict_next_pattern(&cue, 1, 32);
        if !predicted.is_empty() {
            nonzero += 1;
        }
        confidence += predicted.iter().map(|(_, score)| *score).sum::<f32>() / 32.0;

        let words = tokenizer.words(sentence);
        if let Some(first) = words.first() {
            let word = word_pattern(tokenizer.word_id(first), network.agents.len());
            let sentence_sig = sentence_pattern(sentence, network.agents.len());
            let mut context = concept_pattern("inicio_oracion", sentence, network.agents.len());
            context.extend(sentence_sig);
            let predicted_word = network.predict_next_pattern(&context, 1, 32);
            if predicted_word.iter().any(|(idx, _)| word.contains(idx)) {
                nonzero += 1;
            }
            confidence += predicted_word.iter().map(|(_, score)| *score).sum::<f32>() / 32.0;
        }
    }
    ProbeStats {
        total: cases.len() * 2,
        nonzero,
        confidence: confidence / (cases.len() * 2).max(1) as f32,
    }
}

fn spanish_curriculum() -> Vec<String> {
    [
        "a e i o u son vocales",
        "b c d f g son consonantes",
        "la letra forma silaba",
        "la silaba forma palabra",
        "la palabra nombra una idea",
        "dos palabras pueden formar una frase",
        "una frase expresa una relacion",
        "una oracion tiene sujeto verbo y objeto",
        "el sujeto indica quien actua",
        "el verbo indica accion estado o cambio",
        "el objeto recibe la accion",
        "el adjetivo describe un nombre",
        "el adverbio modifica una accion",
        "el articulo acompana al nombre",
        "el singular habla de uno",
        "el plural habla de varios",
        "el pasado indica accion anterior",
        "el presente indica accion actual",
        "el futuro indica accion posible despues",
        "una pregunta busca informacion",
        "una respuesta entrega informacion clara",
        "una causa explica porque ocurre un efecto",
        "un efecto aparece despues de una causa",
        "una condicion indica cuando algo puede ocurrir",
        "comparar muestra semejanzas y diferencias",
        "explicar conecta una idea con otra",
        "definir dice que significa una palabra",
        "resumir reduce una oracion a su idea central",
        "el lenguaje organiza sonidos letras palabras y oraciones",
        "la memoria conserva rutas utiles entre conceptos",
        "la red aprende cuando coactivan letras palabras y frases",
        "la malla fractal poda ruido y conserva rutas utiles",
        "el conocimiento linguistico empieza en letras y crece a oraciones",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn normalized_chars(text: &str) -> Vec<char> {
    text.to_lowercase()
        .chars()
        .map(normalize_char)
        .filter(|ch| ch.is_ascii_alphabetic() || *ch == ' ')
        .collect()
}

fn tokenize_words(text: &str) -> Vec<String> {
    normalized_chars(text)
        .into_iter()
        .collect::<String>()
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn normalize_char(ch: char) -> char {
    match ch {
        'á' | 'à' | 'ä' | 'â' => 'a',
        'é' | 'è' | 'ë' | 'ê' => 'e',
        'í' | 'ì' | 'ï' | 'î' => 'i',
        'ó' | 'ò' | 'ö' | 'ô' => 'o',
        'ú' | 'ù' | 'ü' | 'û' => 'u',
        'ñ' => 'n',
        other => other,
    }
}

fn sentence_role(pos: usize, len: usize) -> &'static str {
    if pos == 0 {
        "sujeto"
    } else if pos + 1 == len {
        "cierre"
    } else if pos == 1 {
        "verbo"
    } else {
        "complemento"
    }
}

fn letter_pattern(letter_id: usize, nodes: usize) -> Vec<usize> {
    hashed_pattern("letter", letter_id, LETTER_PATTERN_SIZE, nodes)
}

fn word_pattern(word_id: usize, nodes: usize) -> Vec<usize> {
    hashed_pattern("word", word_id, WORD_PATTERN_SIZE, nodes)
}

fn union_pattern(left: &str, right: &str, nodes: usize) -> Vec<usize> {
    string_pattern(
        "union",
        &format!("{left}_{right}"),
        UNION_PATTERN_SIZE,
        nodes,
    )
}

fn sentence_pattern(sentence: &str, nodes: usize) -> Vec<usize> {
    string_pattern("sentence", sentence, SENTENCE_PATTERN_SIZE, nodes)
}

fn concept_pattern(kind: &str, value: &str, nodes: usize) -> Vec<usize> {
    string_pattern(kind, value, UNION_PATTERN_SIZE, nodes)
}

fn role_pattern(role: &str, pos: usize, nodes: usize) -> Vec<usize> {
    let mut pattern = string_pattern(role, &pos.to_string(), ROLE_PATTERN_SIZE, nodes);
    pattern.extend(string_pattern("role", role, 3, nodes));
    pattern.sort_unstable();
    pattern.dedup();
    pattern
}

fn hashed_pattern(prefix: &str, id: usize, size: usize, nodes: usize) -> Vec<usize> {
    (0..size)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prefix.hash(&mut hasher);
            id.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn string_pattern(prefix: &str, value: &str, size: usize, nodes: usize) -> Vec<usize> {
    (0..size)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prefix.hash(&mut hasher);
            value.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn fractal_mesh_config() -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 7,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes: AGENT_COUNT,
        base_radius: 0.0,
        lateral_link_weight: 0.35,
        parent_link_weight: 1.0,
    }
}

fn fractal_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 72,
        height: 40,
        spacing: 6.5,
        elasticity: 0.005,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.00012,
        max_active_agents: 384,
        inhibition_decay: 0.035,
        max_spikes_per_step: 1024,
        local_inhibition_decay: 0.78,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 2048,
        replay_interval: 8,
        replay_batch: 12,
        replay_learning_rate: 0.05,
        causal_learning_rate: 0.18,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.00008,
        hyperbolic_curvature: 0.0,
        seed: 401,
    }
}

fn arg_value(name: &str) -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            return args.next();
        }
    }
    None
}
