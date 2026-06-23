use macroquad::prelude::*;
use snga::geometry::Vec2;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::fs;
use std::time::SystemTime;

const STATE_PATH: &str = "data/snga_scaled_gemma_language.snga";
const RELOAD_EVERY_FRAMES: u64 = 180;
const MAX_DRAW_EDGES: usize = 4_000;
const MAX_DRAW_TRIANGLES: usize = 1_600;

struct Viewer {
    network: SimplicialNetwork,
    camera: Vec2,
    zoom: f32,
    frame: u64,
    last_modified: Option<SystemTime>,
    status: String,
    show_mesh: bool,
    show_resting_edges: bool,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "SNGA - Visor de entrenamiento escalado".to_string(),
        window_width: 1320,
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
        viewer.frame += 1;
        if viewer.frame % RELOAD_EVERY_FRAMES == 0 {
            viewer.reload_if_changed();
        }
        viewer.draw();
        next_frame().await;
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
    }
}

impl Viewer {
    fn new() -> Self {
        let mut viewer = Self {
            network: SimplicialNetwork::grid_3d(scaled_config(), 2),
            camera: Vec2::new(38.0, 145.0),
            zoom: 0.42,
            frame: 0,
            last_modified: None,
            status: "iniciando visor de solo lectura".to_string(),
            show_mesh: true,
            show_resting_edges: true,
        };
        viewer.reload_force();
        viewer
    }

    fn handle_input(&mut self) {
        let (_wheel_x, wheel_y) = mouse_wheel();
        if wheel_y > 0.0 {
            self.zoom = (self.zoom * (1.0 + wheel_y * 0.10)).min(2.5);
        } else if wheel_y < 0.0 {
            self.zoom = (self.zoom / (1.0 + wheel_y.abs() * 0.10)).max(0.12);
        }
        if is_key_pressed(KeyCode::L) {
            self.reload_force();
        }
        if is_key_pressed(KeyCode::M) {
            self.show_mesh = !self.show_mesh;
        }
        if is_key_pressed(KeyCode::E) {
            self.show_resting_edges = !self.show_resting_edges;
        }
        if is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::Z) {
            self.zoom = (self.zoom * 1.12).min(2.5);
        }
        if is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::X) {
            self.zoom = (self.zoom / 1.12).max(0.12);
        }
        let speed = 14.0 / self.zoom.max(0.2);
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

    fn reload_if_changed(&mut self) {
        let Ok(meta) = fs::metadata(STATE_PATH) else {
            self.status = format!("sin archivo: {STATE_PATH}");
            return;
        };
        let modified = meta.modified().ok();
        if modified.is_some() && modified == self.last_modified {
            return;
        }
        self.reload_force();
    }

    fn reload_force(&mut self) {
        let mut candidate = SimplicialNetwork::grid_3d(scaled_config(), 2);
        match candidate.load_persistent_state(STATE_PATH) {
            Ok(report) => {
                self.last_modified = fs::metadata(STATE_PATH).and_then(|m| m.modified()).ok();
                self.network = candidate;
                self.status = format!(
                    "checkpoint cargado: agentes={} aristas={} causales={}",
                    report.agents, report.edges, report.causal_edges
                );
            }
            Err(err) => {
                self.status = format!("no se pudo cargar checkpoint: {err}");
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
        self.draw_hud();
    }

    fn draw_triangles(&self) {
        let skip = (self.network.simplices.len() / MAX_DRAW_TRIANGLES).max(1);
        for simplex in self.network.simplices.iter().step_by(skip) {
            let a = self.project(simplex.a);
            let b = self.project(simplex.b);
            let c = self.project(simplex.c);
            draw_triangle_lines(a, b, c, 0.45, Color::from_rgba(38, 62, 86, 55));
        }
    }

    fn draw_edges(&self) {
        let skip = if self.show_resting_edges {
            (self.network.edges.len() / MAX_DRAW_EDGES).max(1)
        } else {
            1
        };
        for edge in self.network.edges.iter().step_by(skip) {
            let associative = edge.weight > 1.05;
            if !self.show_resting_edges && !associative {
                continue;
            }
            let a = &self.network.agents[edge.a];
            let b = &self.network.agents[edge.b];
            let active = a.surprise > 0.05 || b.surprise > 0.05;
            let color = if active {
                Color::from_rgba(80, 230, 255, 230)
            } else if associative {
                Color::from_rgba(255, 180, 70, 150)
            } else {
                Color::from_rgba(62, 78, 105, 62)
            };
            let thickness = if active {
                2.0
            } else if associative {
                1.2
            } else {
                0.6
            };
            let pa = self.project(edge.a);
            let pb = self.project(edge.b);
            draw_line(pa.x, pa.y, pb.x, pb.y, thickness, color);
        }
    }

    fn draw_agents(&self) {
        let skip = (self.network.agents.len() / 4_500).max(1);
        for agent in self.network.agents.iter().step_by(skip) {
            let p = self.project(agent.id);
            let radius = (1.25 + agent.surprise * 4.0) * self.zoom.sqrt().clamp(0.45, 1.15);
            let color = if agent.surprise > 0.2 {
                Color::from_rgba(255, 166, 55, 255)
            } else if agent.activation {
                Color::from_rgba(255, 220, 90, 235)
            } else {
                Color::from_rgba(190, 215, 230, 170)
            };
            draw_circle(p.x, p.y, radius, color);
        }
    }

    fn draw_hud(&self) {
        let stats = self.network.stats();
        let plasticity = self.network.plasticity_stats();
        draw_rectangle(
            0.0,
            0.0,
            screen_width(),
            142.0,
            Color::from_rgba(4, 7, 12, 225),
        );
        let lines = [
            "SNGA visor solo lectura - entrenamiento escalado por lotes".to_string(),
            format!(
                "energia={:.1} activos={} spikes={} frame={} zoom={:.2}",
                stats.total_free_energy, stats.active_agents, stats.active_spikes, self.frame, self.zoom
            ),
            format!(
                "nodos={} aristas={} asociativas={} consolidadas={} causales={} tetraedros={}",
                self.network.agents.len(),
                self.network.edges.len(),
                plasticity.associative_edges,
                plasticity.consolidated_edges,
                plasticity.causal_edges,
                self.network.tetrahedra.len()
            ),
            self.status.clone(),
            "controles: L recargar | M malla | E aristas | rueda/Z/X zoom | flechas mover | Esc cerrar".to_string(),
        ];
        for (i, line) in lines.iter().enumerate() {
            draw_text(line, 18.0, 26.0 + i as f32 * 24.0, 21.0, WHITE);
        }
    }

    fn project(&self, idx: usize) -> macroquad::prelude::Vec2 {
        let agent = &self.network.agents[idx];
        macroquad::prelude::Vec2::new(
            (agent.position.x + agent.depth * 0.34) * self.zoom + self.camera.x,
            (agent.position.y - agent.depth * 0.22) * self.zoom + self.camera.y,
        )
    }
}

fn scaled_config() -> SimplicialConfig {
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
