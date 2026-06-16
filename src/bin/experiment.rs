#[path = "../geometry.rs"]
mod geometry;
#[path = "../multimodal.rs"]
mod multimodal;
#[path = "../simplicial.rs"]
mod simplicial;

use multimodal::{MultimodalDemo, RecallReport};
use simplicial::{SimplicialConfig, SimplicialNetwork};

fn main() {
    let labels = {
        let network = SimplicialNetwork::grid(SimplicialConfig::default());
        MultimodalDemo::new(&network).concept_labels()
    };

    println!("SNGA headless experiment");
    println!("dataset sintetico: lenguaje + vision + audio");
    println!("steps_por_evocacion=12");
    println!();

    for label in labels {
        let before = evaluate(label, false);
        let after = evaluate(label, true);
        print_comparison(&before, &after);
    }
}

fn evaluate(label: &str, trained: bool) -> RecallReport {
    let mut network = SimplicialNetwork::grid(SimplicialConfig::default());
    let mut demo = MultimodalDemo::new(&network);

    if trained {
        demo.train_all(&mut network);
        for _ in 0..18 {
            network.step();
        }
    }

    demo.evaluate_recall(&mut network, label, 12)
        .expect("known concept label")
}

fn print_comparison(before: &RecallReport, after: &RecallReport) {
    let before_ratio = before.active_target_agents as f32 / before.target_agents.max(1) as f32;
    let after_ratio = after.active_target_agents as f32 / after.target_agents.max(1) as f32;
    let ratio_gain = after_ratio - before_ratio;
    let surprise_gain = after.mean_target_surprise - before.mean_target_surprise;

    println!("concepto={}", after.label);
    println!(
        "  antes:  activos_objetivo={}/{} ({:.1}%) sorpresa_media={:.3} energia={:.2}",
        before.active_target_agents,
        before.target_agents,
        before_ratio * 100.0,
        before.mean_target_surprise,
        before.total_free_energy
    );
    println!(
        "  despues: activos_objetivo={}/{} ({:.1}%) sorpresa_media={:.3} energia={:.2}",
        after.active_target_agents,
        after.target_agents,
        after_ratio * 100.0,
        after.mean_target_surprise,
        after.total_free_energy
    );
    println!(
        "  ganancia: cobertura={:+.1}% sorpresa_media={:+.3} sorpresa_total_despues={:.3}",
        ratio_gain * 100.0,
        surprise_gain,
        after.target_surprise
    );
    println!();
}
