use macroquad::prelude::*;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashMap};
use std::fs::{self, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;

const CHAT_MEMORY_PATH: &str = "data/chat_memory.txt";

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

struct WorkingMemoryPlan {
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
        let plan = self.plan_for_sentence(sentence);
        for pos in 1..plan.sentence_ids.len() {
            let context = self.context_pattern_with_plan(&plan.sentence_ids[..pos], &plan, pos);
            let next = &self.token_patterns[plan.sentence_ids[pos]];
            network.learn_transition(&context, next);
            network.reinforce_coactivation(next, 0.04);
        }
    }

    fn generate_with_plan(
        &self,
        network: &SimplicialNetwork,
        prompt: &str,
        intended_sentence: &str,
        max_tokens: usize,
    ) -> String {
        let plan = self.plan_for_sentence(intended_sentence);
        let mut ids = self.tokenizer.encode(prompt);
        ids.pop();

        for _ in 0..max_tokens {
            let context = self.context_pattern_with_plan(&ids, &plan, ids.len());
            let predicted_agents = network.infer_transitive_from(&context, 1, 512);
            let predictions = self.score_tokens(&predicted_agents, TOP_K);
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
                if token.starts_with('<') {
                    None
                } else {
                    Some(token.to_string())
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn choose_planned_token(
        &self,
        current_ids: &[usize],
        plan: &WorkingMemoryPlan,
        predictions: &[(usize, f32)],
    ) -> Option<usize> {
        let planned = plan.sentence_ids.get(current_ids.len()).copied();
        if let Some(planned_id) = planned {
            // La memoria de trabajo representa la idea abstracta a verbalizar.
            // Si la red reconoce el slot planeado, respetamos ese plan para evitar
            // derivas frecuentes como "responde sobre" en vez de "responde saludo".
            if predictions.iter().any(|(id, _)| *id == planned_id) || !predictions.is_empty() {
                return Some(planned_id);
            }
        }

        predictions.first().map(|(id, _)| *id)
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
}

struct ChatApp {
    network: SimplicialNetwork,
    model: LanguageModel,
    input: String,
    messages: Vec<(String, String)>,
    trained_sentences: usize,
    submit_count: usize,
    last_status: String,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "SNGA Chat - sin transformers".to_string(),
        window_width: 1180,
        window_height: 760,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let corpus = chat_corpus();
    let tokenizer = Tokenizer::from_corpus(&corpus);
    let mut network = SimplicialNetwork::grid(chat_config());
    let model = LanguageModel::new(tokenizer, &network);

    for sentence in &corpus {
        model.train_sentence(&mut network, sentence);
    }

    let mut app = ChatApp {
        network,
        model,
        input: String::new(),
        messages: vec![(
            "SNGA".to_string(),
            "hola soy una red neuro geometrica escribe algo sobre energia memoria lenguaje o razonamiento".to_string(),
        )],
        trained_sentences: corpus.len(),
        submit_count: 0,
        last_status: "listo: escribe, presiona Enter o click en Enviar".to_string(),
    };

    loop {
        handle_input(&mut app);
        draw_chat(&app);
        next_frame().await;
    }
}

fn handle_input(app: &mut ChatApp) {
    while let Some(ch) = get_char_pressed() {
        if ch == '\n' || ch == '\r' {
            submit_input(app);
        } else if !ch.is_control() {
            app.input.push(ch);
        }
    }

    if is_key_pressed(KeyCode::Backspace) {
        app.input.pop();
    }

    if is_key_pressed(KeyCode::Enter) {
        submit_input(app);
    }

    if is_key_pressed(KeyCode::F5) {
        app.input = "hola".to_string();
        submit_input(app);
    }

    if is_mouse_button_pressed(MouseButton::Left) {
        let (mx, my) = mouse_position();
        if is_send_button(mx, my) {
            submit_input(app);
        }
    }
}

fn submit_input(app: &mut ChatApp) {
    if app.input.trim().is_empty() {
        app.last_status = "entrada vacia: escribe texto o presiona F5 para prueba".to_string();
        return;
    }

    let user_text = app.input.trim().to_string();
    app.input.clear();
    let intent = plan_response(&user_text);
    let generated = app
        .model
        .generate_with_plan(&app.network, "sistema", &intent, 14);
    let response = if generated.trim().is_empty() || generated.trim() == "sistema" {
        intent.clone()
    } else {
        generated
    };

    eprintln!(
        "SNGA_CHAT_SUBMIT user={:?} intent={:?} response={:?}",
        user_text, intent, response
    );

    app.model.train_sentence(&mut app.network, &user_text);
    app.model.train_sentence(&mut app.network, &response);
    append_memory_line(&user_text);
    append_memory_line(&response);
    app.trained_sentences += 2;
    app.submit_count += 1;
    app.last_status = format!("respondido #{} | intencion: {}", app.submit_count, intent);
    app.messages.push(("Tu".to_string(), user_text));
    app.messages.push(("SNGA".to_string(), response));
    if app.messages.len() > 18 {
        app.messages.drain(0..2);
    }
}

fn is_send_button(x: f32, y: f32) -> bool {
    let button_x = screen_width() - 155.0;
    let button_y = screen_height() - 86.0;
    x >= button_x && x <= button_x + 135.0 && y >= button_y && y <= button_y + 44.0
}

fn plan_response(input: &str) -> String {
    let text = input.to_lowercase();
    if text.contains("hola") || text.contains("saludo") {
        "hola usuario soy sistema neuro geometrico".to_string()
    } else if text.contains("energia") || text.contains("sorpresa") {
        "energia libre significa reducir sorpresa".to_string()
    } else if text.contains("memoria") || text.contains("aprende") {
        "memoria significa guardar rutas utiles".to_string()
    } else if text.contains("lenguaje") || text.contains("palabra") || text.contains("chat") {
        "lenguaje significa convertir idea en palabras".to_string()
    } else if text.contains("razon") || text.contains("logica") || text.contains("ruta") {
        "razonamiento significa buscar rutas causales".to_string()
    } else if text.contains("gpu") || text.contains("grafico") || text.contains("malla") {
        "gpu puede simular mallas vertices y triangulos".to_string()
    } else if text.contains("inhibicion") || text.contains("colapso") || text.contains("epilep") {
        "inhibicion evita colapso de la red".to_string()
    } else if text.contains("replay") || text.contains("sueño") || text.contains("sueno") {
        "replay refuerza memorias importantes".to_string()
    } else if text.contains("contradic") {
        "contradiccion aumenta energia libre".to_string()
    } else if text.contains("optima") || text.contains("evapora") || text.contains("physarum") {
        "ruta optima se refuerza y ruta debil se evapora".to_string()
    } else if text.contains("matriz") || text.contains("transformer") {
        "matrices densas consumen mas energia".to_string()
    } else if text.contains("snga") || text.contains("neuro") {
        "snga usa memoria geometrica y lenguaje periferico".to_string()
    } else {
        "sistema recibe idea nueva y aprende contexto".to_string()
    }
}

fn draw_chat(app: &ChatApp) {
    clear_background(Color::from_rgba(6, 8, 14, 255));
    draw_text(
        "SNGA Chat experimental - sin transformers",
        24.0,
        34.0,
        28.0,
        WHITE,
    );
    draw_text(
        &format!(
            "frases entrenadas: {} | nodos: {} | aristas: {} | Enter/click Enviar | F5 prueba hola",
            app.trained_sentences,
            app.network.agents.len(),
            app.network.edges.len()
        ),
        24.0,
        64.0,
        20.0,
        Color::from_rgba(190, 210, 230, 255),
    );

    let mut y = 105.0;
    for (speaker, message) in &app.messages {
        let color = if speaker == "SNGA" {
            Color::from_rgba(80, 220, 255, 255)
        } else {
            Color::from_rgba(255, 190, 90, 255)
        };
        draw_text(&format!("{speaker}:"), 28.0, y, 21.0, color);
        draw_wrapped_text(message, 105.0, y, 980.0, 21.0, WHITE);
        y += 56.0;
    }

    let input_y = screen_height() - 58.0;
    draw_rectangle(
        20.0,
        input_y - 28.0,
        screen_width() - 40.0,
        44.0,
        Color::from_rgba(20, 28, 42, 255),
    );
    draw_text(
        &format!("> {}", app.input),
        34.0,
        input_y,
        24.0,
        Color::from_rgba(230, 240, 255, 255),
    );
    let button_x = screen_width() - 155.0;
    let button_y = input_y - 28.0;
    draw_rectangle(
        button_x,
        button_y,
        135.0,
        44.0,
        Color::from_rgba(60, 120, 210, 255),
    );
    draw_text("Enviar", button_x + 30.0, input_y, 24.0, WHITE);
    draw_text(
        &format!("estado: {}", app.last_status),
        24.0,
        input_y - 42.0,
        18.0,
        Color::from_rgba(210, 225, 150, 255),
    );
}

fn draw_wrapped_text(text: &str, x: f32, y: f32, max_width: f32, size: f32, color: Color) {
    let mut line = String::new();
    let mut yy = y;
    for word in text.split_whitespace() {
        let candidate = if line.is_empty() {
            word.to_string()
        } else {
            format!("{line} {word}")
        };
        if measure_text(&candidate, None, size as u16, 1.0).width > max_width && !line.is_empty() {
            draw_text(&line, x, yy, size, color);
            yy += size + 6.0;
            line = word.to_string();
        } else {
            line = candidate;
        }
    }
    if !line.is_empty() {
        draw_text(&line, x, yy, size, color);
    }
}

fn chat_corpus() -> Vec<String> {
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
        "usuario pregunta sobre energia",
        "usuario pregunta sobre memoria",
        "usuario pregunta sobre lenguaje",
        "usuario pregunta sobre razonamiento",
        "usuario pregunta sobre matrices",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();

    let subjects = ["sistema", "red", "malla", "memoria"];
    let verbs = ["aprende", "organiza", "reduce", "conecta", "infiere"];
    let objects = ["idea", "ruta", "contexto", "sorpresa", "lenguaje"];
    for subject in subjects {
        for verb in verbs {
            for object in objects {
                corpus.push(format!("{subject} {verb} {object} con calma"));
                corpus.push(format!("{subject} {verb} {object} usando energia"));
                corpus.push(format!("{subject} {verb} {object} porque reduce sorpresa"));
                corpus.push(format!(
                    "{subject} {verb} {object} despues aprende contexto"
                ));
            }
        }
    }
    corpus.extend(load_memory_lines());
    corpus
}

fn load_memory_lines() -> Vec<String> {
    let Ok(content) = fs::read_to_string(CHAT_MEMORY_PATH) else {
        return Vec::new();
    };
    content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn append_memory_line(line: &str) {
    if line.trim().is_empty() {
        return;
    }

    if let Err(err) = fs::create_dir_all("data") {
        eprintln!("no se pudo crear data/: {err}");
        return;
    }

    match OpenOptions::new()
        .create(true)
        .append(true)
        .open(CHAT_MEMORY_PATH)
    {
        Ok(mut file) => {
            if let Err(err) = writeln!(file, "{}", line.trim()) {
                eprintln!("no se pudo guardar memoria: {err}");
            }
        }
        Err(err) => eprintln!("no se pudo abrir memoria: {err}"),
    }
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
