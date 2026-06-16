mod geometry;
mod multimodal;
mod render;
mod simplicial;

use geometry::Vec2;
use macroquad::prelude::*;
use multimodal::MultimodalDemo;
use render::Renderer;
use simplicial::{SimplicialConfig, SimplicialNetwork, Spike};

fn window_conf() -> Conf {
    Conf {
        window_title: "SNGA - Red Neuro-Geometrica Binaria".to_string(),
        window_width: 1100,
        window_height: 760,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let config = SimplicialConfig::default();
    let mut network = SimplicialNetwork::grid(config.clone());
    let mut renderer = Renderer::new();
    let mut demo = MultimodalDemo::new(&network);
    let mut paused = false;
    let mut stats = network.stats();

    network.inject_text_pattern("arquitectura neuro geometrica multi agente");

    loop {
        renderer.handle_input();

        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::R) {
            network = SimplicialNetwork::grid(config.clone());
            demo = MultimodalDemo::new(&network);
            network.inject_text_pattern("reset topologico");
            stats = network.stats();
        }
        if is_key_pressed(KeyCode::M) {
            demo.train_all(&mut network);
        }
        if is_key_pressed(KeyCode::L) {
            demo.recall_language(&mut network, "manzana");
        }
        if is_key_pressed(KeyCode::O) {
            demo.recall_language(&mut network, "roca");
        }
        if is_key_pressed(KeyCode::T) {
            network.inject_text_pattern("energia libre simplicial lenguaje periferico");
        }
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            let world = renderer.screen_to_world(mx, my);
            excite_nearest(&mut network, world);
        }

        if !paused {
            for _ in 0..2 {
                stats = network.step();
            }
        }

        demo.refresh_projection(&network);
        renderer.draw(&network, &stats, paused, demo.trace());
        next_frame().await;
    }
}

fn excite_nearest(network: &mut SimplicialNetwork, point: Vec2) {
    let Some((idx, _)) = network
        .agents
        .iter()
        .map(|agent| (agent.id, agent.position.distance(point)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
    else {
        return;
    };

    network.agents[idx].activation = true;
    network.agents[idx].surprise = 1.4;

    for neighbor in network.neighbor_ids(idx) {
        network.spikes.push_back(Spike {
            source: idx,
            target: neighbor,
            ttl: 3,
        });
    }
}
