use macroquad::prelude::*;
use snga::geometry::Vec2;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

const TRAIN_EVERY_FRAMES: u64 = 18;
const SIM_STEPS_PER_FRAME: usize = 2;
const MAX_DRAW_EDGES: usize = 2_800;
const MAX_DRAW_TRIANGLES: usize = 1_200;
const SAVE_EVERY_FRAMES: u64 = 180;
const STATE_PATH: &str = "data/mesh_training_state.snga";

#[derive(Clone)]
struct Concept {
    label: &'static str,
    cue: Vec<usize>,
    body: Vec<usize>,
    next: Vec<usize>,
}

struct Viewer {
    network: SimplicialNetwork,
    concepts: Vec<Concept>,
    frame: u64,
    train_step: usize,
    paused: bool,
    show_mesh: bool,
    show_resting_edges: bool,
    camera: Vec2,
    zoom: f32,
    stats_message: String,
    persistent_buffer: String,
    dirty: bool,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "SNGA - Motor 3D y red binaria en entrenamiento".to_string(),
        window_width: 1280,
        window_height: 860,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut viewer = Viewer::new();

    loop {
        viewer.handle_input();
        if !viewer.paused {
            viewer.train_tick();
            for _ in 0..SIM_STEPS_PER_FRAME {
                viewer.network.step();
            }
        }
        viewer.draw();
        next_frame().await;
        if viewer.should_close() {
            break;
        }
    }
}

impl Viewer {
    fn new() -> Self {
        let mut network = SimplicialNetwork::grid_3d(viewer_config(), 2);
        let load_message = match network.load_persistent_state(STATE_PATH) {
            Ok(report) => format!(
                "sustrato cargado: agentes={} aristas={} causales={}",
                report.agents, report.edges, report.causal_edges
            ),
            Err(_) => "sin estado previo; creando sustrato nuevo".to_string(),
        };
        network.enable_neural_oscillations();
        let concepts = concepts(network.agents.len());
        let mut viewer = Self {
            network,
            concepts,
            frame: 0,
            train_step: 0,
            paused: false,
            show_mesh: true,
            show_resting_edges: true,
            camera: Vec2::new(45.0, 155.0),
            zoom: 0.62,
            stats_message: load_message,
            persistent_buffer: String::new(),
            dirty: false,
        };
        viewer.prime_network();
        viewer.refresh_persistent_buffer();
        viewer
    }

    fn prime_network(&mut self) {
        for concept in &self.concepts {
            let mut fused = concept.cue.clone();
            fused.extend(concept.body.iter().copied());
            self.network
                .reinforce_coactivation_if_useful(&fused, 0.08, 0.9);
            self.network.learn_transition(&concept.cue, &concept.body);
            self.network.learn_transition(&concept.body, &concept.next);
        }
        self.network.clear_activity();
    }

    fn handle_input(&mut self) {
        let speed = 12.0 / self.zoom.max(0.3);
        if is_key_pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        if is_key_pressed(KeyCode::M) {
            self.show_mesh = !self.show_mesh;
        }
        if is_key_pressed(KeyCode::E) {
            self.show_resting_edges = !self.show_resting_edges;
        }
        if is_key_pressed(KeyCode::R) {
            *self = Self::new();
        }
        if is_key_pressed(KeyCode::S) {
            self.save_now();
        }
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
        if is_key_pressed(KeyCode::Equal) {
            self.zoom = (self.zoom * 1.12).min(2.8);
        }
        if is_key_pressed(KeyCode::Minus) {
            self.zoom = (self.zoom / 1.12).max(0.22);
        }
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            self.excite_nearest(mx, my);
        }
    }

    fn train_tick(&mut self) {
        self.frame += 1;
        if self.frame % TRAIN_EVERY_FRAMES != 0 {
            return;
        }

        let idx = self.train_step % self.concepts.len();
        let concept = self.concepts[idx].clone();
        let mut fused = concept.cue.clone();
        fused.extend(concept.body.iter().copied());

        self.network.clear_activity();
        self.network.set_attention_goal(&concept.body);
        self.network.inject_pattern(&concept.cue, 1.2, 2);
        self.network.inject_pattern(&concept.body, 0.9, 1);
        self.network
            .reinforce_coactivation_if_useful(&fused, 0.055, 0.85);
        self.network.learn_transition(&concept.cue, &concept.body);
        self.network.learn_transition(&concept.body, &concept.next);
        self.dirty = true;
        self.stats_message = format!(
            "train {}: cue={} cuerpo={} siguiente={}",
            concept.label,
            concept.cue.len(),
            concept.body.len(),
            concept.next.len()
        );
        self.train_step += 1;
        if self.frame % SAVE_EVERY_FRAMES == 0 {
            self.autosave();
        }
    }

    fn refresh_persistent_buffer(&mut self) {
        self.persistent_buffer = self.network.serialize_persistent_state();
    }

    fn autosave(&mut self) {
        if !self.dirty {
            return;
        }
        self.refresh_persistent_buffer();
        match self.network.save_persistent_state(STATE_PATH) {
            Ok(report) => {
                self.stats_message = format!(
                    "autosave: agentes={} aristas={} causales={}",
                    report.agents, report.edges, report.causal_edges
                );
                self.dirty = false;
            }
            Err(err) => {
                self.stats_message = format!("autosave fallo: {err}");
            }
        }
    }

    fn save_now(&mut self) {
        self.refresh_persistent_buffer();
        match self.network.save_persistent_state(STATE_PATH) {
            Ok(report) => {
                self.stats_message = format!(
                    "guardado: agentes={} aristas={} causales={}",
                    report.agents, report.edges, report.causal_edges
                );
                self.dirty = false;
            }
            Err(err) => {
                self.stats_message = format!("guardado fallo: {err}");
            }
        }
    }

    fn should_close(&mut self) -> bool {
        if is_key_pressed(KeyCode::Escape) {
            self.save_now();
            return true;
        }
        false
    }

    fn draw(&self) {
        clear_background(Color::from_rgba(5, 7, 13, 255));
        if self.show_mesh {
            self.draw_triangles();
        }
        self.draw_edges();
        self.draw_agents();
        self.draw_hud();
    }

    fn draw_triangles(&self) {
        let skip = (self.network.simplices.len() / MAX_DRAW_TRIANGLES).max(1);
        for simplex in self.network.simplices.iter().step_by(skip) {
            let a = self.project_agent(simplex.a);
            let b = self.project_agent(simplex.b);
            let c = self.project_agent(simplex.c);
            draw_triangle_lines(a, b, c, 0.55, Color::from_rgba(40, 65, 88, 65));
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
            let color = if active {
                Color::from_rgba(80, 230, 255, 230)
            } else if associative {
                Color::from_rgba(255, 180, 70, 160)
            } else {
                Color::from_rgba(62, 78, 105, 80)
            };
            let thickness = if active {
                2.3
            } else if associative {
                1.45
            } else {
                0.75
            };
            draw_line(pa.x, pa.y, pb.x, pb.y, thickness, color);
        }
    }

    fn draw_agents(&self) {
        for agent in &self.network.agents {
            let p = self.project_agent(agent.id);
            let depth_alpha = (180.0 + agent.depth * 0.8).clamp(90.0, 255.0) as u8;
            let radius = (1.65 + agent.surprise * 4.4) * self.zoom.sqrt().clamp(0.55, 1.25);
            let color = if agent.surprise > 0.2 {
                Color::from_rgba(255, 166, 55, 255)
            } else if agent.activation {
                Color::from_rgba(255, 220, 90, 235)
            } else {
                Color::from_rgba(190, 215, 230, depth_alpha)
            };
            draw_circle(p.x, p.y, radius, color);
        }
    }

    fn draw_hud(&self) {
        let stats = self.network.stats();
        let plasticity = self.network.plasticity_stats();
        let osc = self.network.oscillation_stats();
        let panel_h = 150.0;
        draw_rectangle(
            0.0,
            0.0,
            screen_width(),
            panel_h,
            Color::from_rgba(4, 7, 12, 225),
        );
        let lines = [
            "SNGA visor opcional - motor geometrico 3D + red binaria".to_string(),
            format!(
                "estado={} | energia={:.2} | activos={} | spikes={} | frame={}",
                if self.paused { "pausa" } else { "entrenando" },
                stats.total_free_energy,
                stats.active_agents,
                stats.active_spikes,
                self.frame
            ),
            format!(
                "nodos={} aristas={} triangulos={} tetraedros={} asociativas={} consolidadas={}",
                self.network.agents.len(),
                self.network.edges.len(),
                self.network.simplices.len(),
                self.network.tetrahedra.len(),
                plasticity.associative_edges,
                plasticity.consolidated_edges
            ),
            format!(
                "ondas={} modo={:?} regiones={} delta={} theta={} alpha={} beta={} gamma={}",
                osc.enabled,
                osc.mode,
                osc.regions,
                osc.delta_regions,
                osc.theta_regions,
                osc.alpha_regions,
                osc.beta_regions,
                osc.gamma_regions
            ),
            self.stats_message.clone(),
            format!(
                "buffer={} KB | controles: Espacio pausa | S guardar | Esc guardar/salir | M malla | E aristas | R reset",
                self.persistent_buffer.len() / 1024
            ),
        ];
        for (i, line) in lines.iter().enumerate() {
            draw_text(line, 18.0, 24.0 + i as f32 * 22.0, 20.0, WHITE);
        }
    }

    fn excite_nearest(&mut self, x: f32, y: f32) {
        let screen = macroquad::prelude::Vec2::new(x, y);
        let Some((idx, _)) = self
            .network
            .agents
            .iter()
            .map(|agent| {
                let p = self.project_agent(agent.id);
                (agent.id, p.distance(screen))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
        else {
            return;
        };
        self.network.inject_pattern(&[idx], 1.4, 3);
    }

    fn project_agent(&self, idx: usize) -> macroquad::prelude::Vec2 {
        let agent = &self.network.agents[idx];
        let x = (agent.position.x + agent.depth * 0.36) * self.zoom + self.camera.x;
        let y = (agent.position.y - agent.depth * 0.24) * self.zoom + self.camera.y;
        macroquad::prelude::Vec2::new(x, y)
    }
}

fn concepts(nodes: usize) -> Vec<Concept> {
    vec![
        Concept {
            label: "energia",
            cue: pattern(nodes, 11),
            body: pattern(nodes, 211),
            next: pattern(nodes, 411),
        },
        Concept {
            label: "memoria",
            cue: pattern(nodes, 71),
            body: pattern(nodes, 271),
            next: pattern(nodes, 471),
        },
        Concept {
            label: "razonamiento",
            cue: pattern(nodes, 131),
            body: pattern(nodes, 331),
            next: pattern(nodes, 531),
        },
        Concept {
            label: "lenguaje",
            cue: pattern(nodes, 191),
            body: pattern(nodes, 391),
            next: pattern(nodes, 591),
        },
    ]
}

fn pattern(nodes: usize, start: usize) -> Vec<usize> {
    (0..8)
        .map(|offset| (start + offset * 13 + offset * offset * 7) % nodes)
        .collect()
}

fn viewer_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 34,
        height: 22,
        spacing: 15.0,
        elasticity: 0.007,
        damping: 0.88,
        activation_threshold: 0.64,
        simplex_area_weight: 0.00025,
        max_active_agents: 96,
        inhibition_decay: 0.06,
        max_spikes_per_step: 256,
        local_inhibition_decay: 0.78,
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
        seed: 177,
    }
}
