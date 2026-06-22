use macroquad::prelude::*;
use snga::geometry::Vec2;
use snga::linguistic_engine::{
    fallback_response, LinguisticContext, LinguisticEngine, OllamaGemmaEngine,
};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;

const STATE_PATH: &str = "data/snga_gemma_visual_chat.snga";
const MAX_MESSAGES: usize = 64;
const MAX_DRAW_EDGES: usize = 2_400;
const MAX_DRAW_TRIANGLES: usize = 1_000;
const CHAT_PANEL_WIDTH: f32 = 430.0;
const CHAT_TOP: f32 = 190.0;
const INPUT_HEIGHT: f32 = 72.0;

struct VisualChat {
    network: SimplicialNetwork,
    engine: OllamaGemmaEngine,
    messages: Vec<(String, String)>,
    input: String,
    status: String,
    camera: Vec2,
    zoom: f32,
    show_mesh: bool,
    show_resting_edges: bool,
    frame: u64,
    chat_scroll: f32,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "SNGA + Gemma - Chat visual con malla 3D".to_string(),
        window_width: 1420,
        window_height: 880,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut app = VisualChat::new();
    loop {
        app.handle_input();
        for _ in 0..2 {
            app.network.step();
        }
        app.frame += 1;
        app.draw();
        next_frame().await;
        if is_key_pressed(KeyCode::Escape) {
            app.save_state("guardado al salir");
            break;
        }
    }
}

impl VisualChat {
    fn new() -> Self {
        let mut network = SimplicialNetwork::grid_3d(visual_chat_config(), 2);
        let load_status = match network.load_persistent_state(STATE_PATH) {
            Ok(report) => format!(
                "sustrato cargado: agentes={} aristas={} causales={}",
                report.agents, report.edges, report.causal_edges
            ),
            Err(_) => "sustrato limpio: aun no hay memoria guardada".to_string(),
        };
        network.enable_neural_oscillations();

        Self {
            network,
            engine: OllamaGemmaEngine {
                host: env::var("SNGA_OLLAMA_HOST")
                    .unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
                model: env::var("SNGA_GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string()),
            },
            messages: vec![(
                "sistema".to_string(),
                "Chat visual listo. SNGA aprende en la malla; Gemma solo verbaliza.".to_string(),
            )],
            input: String::new(),
            status: load_status,
            camera: Vec2::new(40.0, 155.0),
            zoom: 0.58,
            show_mesh: true,
            show_resting_edges: true,
            frame: 0,
            chat_scroll: 0.0,
        }
    }

    fn handle_input(&mut self) {
        let panel_x = screen_width() - CHAT_PANEL_WIDTH;
        let (mx, my) = mouse_position();
        let mouse_in_chat = mx >= panel_x;
        if mouse_in_chat {
            let (_wheel_x, wheel_y) = mouse_wheel();
            if wheel_y.abs() > 0.0 {
                self.chat_scroll = (self.chat_scroll + wheel_y * 34.0).max(0.0);
            }
        }

        while let Some(ch) = get_char_pressed() {
            if !ch.is_control() {
                self.input.push(ch);
            }
        }
        if is_key_pressed(KeyCode::Backspace) {
            self.input.pop();
        }
        if is_key_pressed(KeyCode::Enter) {
            self.submit_message();
        }
        if is_mouse_button_pressed(MouseButton::Left)
            && self
                .send_button_rect()
                .contains(macroquad::prelude::Vec2::new(mx, my))
        {
            self.submit_message();
        }
        if is_key_pressed(KeyCode::F2) {
            self.save_state("guardado manual");
        }
        if is_key_pressed(KeyCode::F5) {
            self.network = SimplicialNetwork::grid_3d(visual_chat_config(), 2);
            self.network.enable_neural_oscillations();
            self.messages.clear();
            self.messages.push((
                "sistema".to_string(),
                "Sustrato reiniciado solo en memoria. Usa F2 para sobrescribir el guardado."
                    .to_string(),
            ));
            self.status = "reinicio temporal: el archivo guardado sigue intacto".to_string();
        }
        if is_key_pressed(KeyCode::F3) {
            self.show_mesh = !self.show_mesh;
        }
        if is_key_pressed(KeyCode::F4) {
            self.show_resting_edges = !self.show_resting_edges;
        }
        if is_key_pressed(KeyCode::Equal) {
            self.zoom = (self.zoom * 1.12).min(2.8);
        }
        if is_key_pressed(KeyCode::Minus) {
            self.zoom = (self.zoom / 1.12).max(0.22);
        }
        let speed = 12.0 / self.zoom.max(0.3);
        if is_key_down(KeyCode::Left) {
            self.camera.x += speed;
        }
        if is_key_down(KeyCode::Right) {
            self.camera.x -= speed;
        }
        if is_key_down(KeyCode::Up) {
            self.camera.y += speed;
        }
        if is_key_down(KeyCode::Down) {
            self.camera.y -= speed;
        }
    }

    fn submit_message(&mut self) {
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        self.input.clear();
        self.messages.push(("usuario".to_string(), prompt.clone()));
        truncate_messages(&mut self.messages);

        self.learn_from_prompt(&prompt);
        let context = LinguisticContext {
            user_prompt: prompt.clone(),
            inferred_intent: infer_intent(&prompt),
            geometric_projection: self.network.project_active_state(12),
            memory_summary: format!(
                "energia={:.2}; aristas={}; episodios={}; la memoria principal vive en el sustrato geometrico persistente",
                self.network.total_free_energy(),
                self.network.edges.len(),
                self.network.plasticity_stats().episodes,
            ),
        };

        self.status = "Gemma verbalizando estado SNGA...".to_string();
        let response = self.engine.generate(&context).unwrap_or_else(|err| {
            self.status = format!("Gemma no disponible, fallback SNGA: {err}");
            fallback_response(&context)
        });
        if response.engine.starts_with("ollama/") {
            self.status = format!("respuesta periferica via {}", response.engine);
        }
        self.messages.push(("gemma".to_string(), response.text));
        truncate_messages(&mut self.messages);
        self.chat_scroll = 0.0;
        self.save_state("aprendizaje guardado tras mensaje");
    }

    fn learn_from_prompt(&mut self, prompt: &str) {
        let prompt_pattern = text_pattern(prompt, self.network.agents.len());
        let intent = infer_intent(prompt);
        let intent_pattern = text_pattern(&intent, self.network.agents.len());
        let mut fused = prompt_pattern.clone();
        fused.extend(intent_pattern.iter().copied());
        fused.sort_unstable();
        fused.dedup();

        self.network.clear_activity();
        self.network.set_attention_goal(&intent_pattern);
        self.network.inject_pattern(&prompt_pattern, 1.2, 2);
        self.network.inject_pattern(&intent_pattern, 0.9, 1);
        self.network
            .reinforce_coactivation_if_useful(&fused, 0.075, 0.9);
        self.network
            .learn_transition(&prompt_pattern, &intent_pattern);
        for _ in 0..8 {
            self.network.step();
        }
        self.network.clear_attention_goal();
    }

    fn save_state(&mut self, prefix: &str) {
        match self.network.save_persistent_state(STATE_PATH) {
            Ok(report) => {
                self.status = format!(
                    "{prefix}: agentes={} aristas={} causales={}",
                    report.agents, report.edges, report.causal_edges
                );
            }
            Err(err) => {
                self.status = format!("fallo guardando sustrato: {err}");
            }
        }
    }

    fn draw(&self) {
        clear_background(Color::from_rgba(5, 7, 13, 255));
        if self.show_mesh {
            self.draw_triangles();
        }
        self.draw_edges();
        self.draw_agents();
        self.draw_chat_panel();
    }

    fn draw_triangles(&self) {
        let skip = (self.network.simplices.len() / MAX_DRAW_TRIANGLES).max(1);
        for simplex in self.network.simplices.iter().step_by(skip) {
            let a = self.project_agent(simplex.a);
            let b = self.project_agent(simplex.b);
            let c = self.project_agent(simplex.c);
            if a.x < screen_width() - CHAT_PANEL_WIDTH {
                draw_triangle_lines(a, b, c, 0.5, Color::from_rgba(40, 65, 88, 60));
            }
        }
    }

    fn draw_edges(&self) {
        let skip = if self.show_resting_edges {
            (self.network.edges.len() / MAX_DRAW_EDGES).max(1)
        } else {
            1
        };
        for edge in self.network.edges.iter().step_by(skip) {
            let a = &self.network.agents[edge.a];
            let b = &self.network.agents[edge.b];
            let active = a.surprise > 0.05 || b.surprise > 0.05;
            let associative = edge.weight > 1.05;
            if !self.show_resting_edges && !active && !associative {
                continue;
            }
            let pa = self.project_agent(edge.a);
            let pb = self.project_agent(edge.b);
            if pa.x > screen_width() - CHAT_PANEL_WIDTH || pb.x > screen_width() - CHAT_PANEL_WIDTH
            {
                continue;
            }
            let color = if active {
                Color::from_rgba(80, 230, 255, 230)
            } else if associative {
                Color::from_rgba(255, 180, 70, 165)
            } else {
                Color::from_rgba(62, 78, 105, 75)
            };
            let thickness = if active {
                2.3
            } else if associative {
                1.35
            } else {
                0.7
            };
            draw_line(pa.x, pa.y, pb.x, pb.y, thickness, color);
        }
    }

    fn draw_agents(&self) {
        for agent in &self.network.agents {
            let p = self.project_agent(agent.id);
            if p.x > screen_width() - CHAT_PANEL_WIDTH {
                continue;
            }
            let radius = (1.6 + agent.surprise * 4.5) * self.zoom.sqrt().clamp(0.55, 1.25);
            let color = if agent.surprise > 0.2 {
                Color::from_rgba(255, 166, 55, 255)
            } else if agent.activation {
                Color::from_rgba(255, 220, 90, 235)
            } else {
                Color::from_rgba(190, 215, 230, 185)
            };
            draw_circle(p.x, p.y, radius, color);
        }
    }

    fn draw_chat_panel(&self) {
        let x = screen_width() - CHAT_PANEL_WIDTH;
        draw_rectangle(
            x,
            0.0,
            CHAT_PANEL_WIDTH,
            screen_height(),
            Color::from_rgba(8, 11, 18, 240),
        );
        draw_text("SNGA + Gemma", x + 18.0, 32.0, 28.0, WHITE);
        draw_text(
            "LLM periferico; SNGA aprende",
            x + 18.0,
            58.0,
            18.0,
            SKYBLUE,
        );

        let stats = self.network.stats();
        let plasticity = self.network.plasticity_stats();
        let osc = self.network.oscillation_stats();
        let info = [
            format!(
                "energia {:.1} activos {} spikes {}",
                stats.total_free_energy, stats.active_agents, stats.active_spikes
            ),
            format!(
                "aristas {} asociativas {} causales {}",
                self.network.edges.len(),
                plasticity.associative_edges,
                plasticity.causal_edges
            ),
            format!(
                "modo {:?} regiones beta/gamma {}/{}",
                osc.mode, osc.beta_regions, osc.gamma_regions
            ),
            self.status.clone(),
        ];
        let mut y = 90.0;
        for line in info {
            draw_wrapped(
                &line,
                x + 18.0,
                y,
                CHAT_PANEL_WIDTH - 34.0,
                17.0,
                Color::from_rgba(210, 225, 240, 255),
            );
            y += 38.0;
        }

        let input_y = screen_height() - INPUT_HEIGHT + 28.0;
        let message_bottom = input_y - 54.0;
        self.draw_message_area(x, CHAT_TOP, message_bottom);

        let input_rect = Rect::new(x + 14.0, input_y - 28.0, CHAT_PANEL_WIDTH - 126.0, 38.0);
        draw_rectangle(
            input_rect.x,
            input_rect.y,
            input_rect.w,
            input_rect.h,
            Color::from_rgba(25, 34, 50, 255),
        );
        draw_wrapped(
            &format!("> {}", self.input),
            input_rect.x + 10.0,
            input_y - 4.0,
            input_rect.w - 16.0,
            18.0,
            WHITE,
        );
        let send = self.send_button_rect();
        let (mx, my) = mouse_position();
        let hovering = send.contains(macroquad::prelude::Vec2::new(mx, my));
        draw_rectangle(
            send.x,
            send.y,
            send.w,
            send.h,
            if hovering {
                Color::from_rgba(75, 145, 235, 255)
            } else {
                Color::from_rgba(55, 115, 205, 255)
            },
        );
        draw_text("Enviar", send.x + 18.0, send.y + 25.0, 22.0, WHITE);
        draw_text(
            "Scroll historial | Enter/Enviar | F2 guardar | F5 reinicio temp | Esc guardar/salir",
            x + 18.0,
            screen_height() - 14.0,
            15.0,
            GRAY,
        );
    }

    fn draw_message_area(&self, panel_x: f32, top: f32, bottom: f32) {
        let area_h = (bottom - top).max(40.0);
        draw_rectangle(
            panel_x + 12.0,
            top - 8.0,
            CHAT_PANEL_WIDTH - 24.0,
            area_h + 14.0,
            Color::from_rgba(10, 15, 24, 180),
        );

        let blocks = self.message_blocks(CHAT_PANEL_WIDTH - 52.0);
        let total_h = blocks.iter().map(|block| block.height).sum::<f32>();
        let max_scroll = (total_h - area_h).max(0.0);
        let scroll = self.chat_scroll.min(max_scroll);
        let mut y = bottom - total_h + scroll;

        for block in blocks {
            let block_bottom = y + block.height;
            if block_bottom >= top && y <= bottom {
                let speaker_color = if block.speaker == "usuario" {
                    YELLOW
                } else if block.speaker == "gemma" {
                    GREEN
                } else {
                    SKYBLUE
                };
                if y + 20.0 >= top && y <= bottom {
                    draw_text(
                        &format!("{}:", block.speaker),
                        panel_x + 20.0,
                        y + 18.0,
                        19.0,
                        speaker_color,
                    );
                }
                let mut line_y = y + 42.0;
                for line in &block.lines {
                    if line_y >= top && line_y <= bottom {
                        draw_text(line, panel_x + 28.0, line_y, 18.0, WHITE);
                    }
                    line_y += 22.0;
                }
            }
            y += block.height;
        }

        if total_h > area_h {
            let track_x = panel_x + CHAT_PANEL_WIDTH - 12.0;
            draw_rectangle(track_x, top, 4.0, area_h, Color::from_rgba(50, 60, 80, 210));
            let thumb_h = (area_h * area_h / total_h).clamp(24.0, area_h);
            let thumb_y = top + (area_h - thumb_h) * (1.0 - scroll / max_scroll.max(1.0));
            draw_rectangle(
                track_x - 1.0,
                thumb_y,
                6.0,
                thumb_h,
                Color::from_rgba(110, 150, 210, 240),
            );
        }
    }

    fn message_blocks(&self, max_width: f32) -> Vec<MessageBlock> {
        self.messages
            .iter()
            .map(|(speaker, message)| {
                let lines = wrap_lines(message, max_width, 18.0);
                let height = 42.0 + lines.len().max(1) as f32 * 22.0 + 14.0;
                MessageBlock {
                    speaker: speaker.clone(),
                    lines,
                    height,
                }
            })
            .collect()
    }

    fn send_button_rect(&self) -> Rect {
        let x = screen_width() - CHAT_PANEL_WIDTH;
        let input_y = screen_height() - INPUT_HEIGHT + 28.0;
        Rect::new(x + CHAT_PANEL_WIDTH - 104.0, input_y - 28.0, 90.0, 38.0)
    }

    fn project_agent(&self, idx: usize) -> macroquad::prelude::Vec2 {
        let agent = &self.network.agents[idx];
        macroquad::prelude::Vec2::new(
            (agent.position.x + agent.depth * 0.36) * self.zoom + self.camera.x,
            (agent.position.y - agent.depth * 0.24) * self.zoom + self.camera.y,
        )
    }
}

struct MessageBlock {
    speaker: String,
    lines: Vec<String>,
    height: f32,
}

fn truncate_messages(messages: &mut Vec<(String, String)>) {
    while messages.len() > MAX_MESSAGES {
        messages.remove(0);
    }
}

fn draw_wrapped(text: &str, x: f32, y: f32, max_width: f32, size: f32, color: Color) -> f32 {
    let lines = wrap_lines(text, max_width, size);
    let mut yy = y;
    for line in lines {
        draw_text(&line, x, yy, size, color);
        yy += size + 5.0;
    }
    yy
}

fn wrap_lines(text: &str, max_width: f32, size: f32) -> Vec<String> {
    let mut line = String::new();
    let mut lines = Vec::new();
    for word in text.split_whitespace() {
        let candidate = if line.is_empty() {
            word.to_string()
        } else {
            format!("{line} {word}")
        };
        if measure_text(&candidate, None, size as u16, 1.0).width > max_width && !line.is_empty() {
            lines.push(line);
            line = word.to_string();
        } else {
            line = candidate;
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn infer_intent(prompt: &str) -> String {
    let lower = prompt.to_lowercase();
    if lower.contains("aprende") || lower.contains("entrena") {
        "aprendizaje_geometrico".to_string()
    } else if lower.contains("memoria") || lower.contains("recuerda") {
        "memoria_episodica".to_string()
    } else if lower.contains("lenguaje") || lower.contains("palabra") {
        "renderizado_linguistico".to_string()
    } else if lower.contains("plan") || lower.contains("razona") {
        "planificacion_causal".to_string()
    } else {
        "consulta_general_snga".to_string()
    }
}

fn text_pattern(text: &str, nodes: usize) -> Vec<usize> {
    text.bytes()
        .enumerate()
        .map(|(i, byte)| ((byte as usize * 37) + i * 53 + text.len() * 11) % nodes)
        .collect()
}

fn visual_chat_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 34,
        height: 22,
        spacing: 15.0,
        elasticity: 0.008,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.00025,
        max_active_agents: 128,
        inhibition_decay: 0.06,
        max_spikes_per_step: 256,
        local_inhibition_decay: 0.76,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0008,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.08,
        max_episodes: 256,
        replay_interval: 8,
        replay_batch: 6,
        replay_learning_rate: 0.06,
        causal_learning_rate: 0.18,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.00015,
        hyperbolic_curvature: 0.0,
        seed: 229,
    }
}
