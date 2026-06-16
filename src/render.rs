use crate::geometry::Vec2;
use crate::multimodal::DemoTrace;
use crate::simplicial::{EnergyStats, SimplicialNetwork};
use macroquad::prelude::*;

pub struct Renderer {
    camera_offset: Vec2,
    scale: f32,
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            camera_offset: Vec2::new(70.0, 95.0),
            scale: 1.0,
        }
    }

    pub fn handle_input(&mut self) {
        let speed = 8.0;
        if is_key_down(KeyCode::Left) {
            self.camera_offset.x += speed;
        }
        if is_key_down(KeyCode::Right) {
            self.camera_offset.x -= speed;
        }
        if is_key_down(KeyCode::Up) {
            self.camera_offset.y += speed;
        }
        if is_key_down(KeyCode::Down) {
            self.camera_offset.y -= speed;
        }
        if is_key_pressed(KeyCode::Equal) {
            self.scale = (self.scale * 1.1).min(2.4);
        }
        if is_key_pressed(KeyCode::Minus) {
            self.scale = (self.scale / 1.1).max(0.35);
        }
    }

    pub fn draw(
        &self,
        network: &SimplicialNetwork,
        stats: &EnergyStats,
        paused: bool,
        trace: &DemoTrace,
    ) {
        clear_background(Color::from_rgba(8, 10, 16, 255));
        self.draw_simplices(network);
        self.draw_edges(network);
        self.draw_agents(network);
        self.draw_hud(network, stats, paused, trace);
    }

    fn draw_simplices(&self, network: &SimplicialNetwork) {
        for simplex in &network.simplices {
            let a = self.to_screen(network.agents[simplex.a].position);
            let b = self.to_screen(network.agents[simplex.b].position);
            let c = self.to_screen(network.agents[simplex.c].position);
            draw_triangle_lines(a, b, c, 0.8, Color::from_rgba(30, 70, 95, 90));
        }
    }

    fn draw_edges(&self, network: &SimplicialNetwork) {
        for edge in &network.edges {
            let a = &network.agents[edge.a];
            let b = &network.agents[edge.b];
            let active = a.activation || b.activation;
            let color = if active {
                Color::from_rgba(80, 220, 255, 210)
            } else {
                Color::from_rgba(65, 84, 110, 115)
            };
            let thickness = if active { 2.2 } else { 1.0 };
            let pa = self.to_screen(a.position);
            let pb = self.to_screen(b.position);
            draw_line(pa.x, pa.y, pb.x, pb.y, thickness, color);
        }
    }

    fn draw_agents(&self, network: &SimplicialNetwork) {
        for agent in &network.agents {
            let p = self.to_screen(agent.position);
            let radius = 3.0 + agent.surprise * 5.0;
            let color = if agent.activation {
                Color::from_rgba(255, 172, 70, 255)
            } else {
                Color::from_rgba(190, 214, 230, 220)
            };
            draw_circle(p.x, p.y, radius, color);
            draw_circle_lines(
                p.x,
                p.y,
                radius + 1.5,
                0.7,
                Color::from_rgba(255, 255, 255, 70),
            );
        }
    }

    fn draw_hud(
        &self,
        network: &SimplicialNetwork,
        stats: &EnergyStats,
        paused: bool,
        trace: &DemoTrace,
    ) {
        let status = if paused { "pausado" } else { "corriendo" };
        let projected = trace
            .projection
            .top_agents
            .iter()
            .map(|(id, value)| format!("{id}:{value:.2}"))
            .collect::<Vec<_>>()
            .join(", ");
        let lines = [
            "SNGA - Sistema Neuro-Geometrico de Agentes".to_string(),
            format!(
                "energia libre: {:.2} | agentes activos: {} | spikes: {} | {}",
                stats.total_free_energy, stats.active_agents, stats.active_spikes, status
            ),
            format!(
                "vertices: {} | aristas: {} | simplices 2D: {}",
                network.agents.len(),
                network.edges.len(),
                network.simplices.len()
            ),
            trace.message.clone(),
            format!("proyeccion activa: [{projected}]"),
            "controles: espacio=pausa, click=estimulo, M=train, L=manzana, O=roca, T=texto, R=reset".to_string(),
        ];

        let panel_height = 132.0;
        draw_rectangle(
            0.0,
            0.0,
            screen_width(),
            panel_height,
            Color::from_rgba(5, 8, 13, 220),
        );
        for (i, line) in lines.iter().enumerate() {
            draw_text(line, 18.0, 24.0 + i as f32 * 20.0, 20.0, WHITE);
        }
    }

    pub fn screen_to_world(&self, x: f32, y: f32) -> Vec2 {
        Vec2::new(
            (x - self.camera_offset.x) / self.scale,
            (y - self.camera_offset.y) / self.scale,
        )
    }

    fn to_screen(&self, p: Vec2) -> macroquad::prelude::Vec2 {
        macroquad::prelude::Vec2::new(
            p.x * self.scale + self.camera_offset.x,
            p.y * self.scale + self.camera_offset.y,
        )
    }
}
