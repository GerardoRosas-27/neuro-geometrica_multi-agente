use snga::multimodal::{MultimodalDemo, RecallReport};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

const EPOCHS: usize = 6;
const RECALL_STEPS: usize = 4;

fn main() {
    let labels = {
        let network = SimplicialNetwork::grid(experiment_config());
        MultimodalDemo::new(&network).concept_labels()
    };

    println!("SNGA headless experiment");
    println!("dataset sintetico: lenguaje + vision + audio");
    println!("conceptos={}", labels.len());
    println!("epocas_entrenamiento={EPOCHS}");
    println!("steps_por_evocacion={RECALL_STEPS}");
    println!();

    let mut before_reports = Vec::new();
    let mut after_reports = Vec::new();

    for label in labels {
        let before = evaluate(label, false);
        let after = evaluate(label, true);
        print_comparison(&before, &after);
        before_reports.push(before);
        after_reports.push(after);
    }

    print_summary("antes", &before_reports);
    print_summary("despues", &after_reports);
    print_viability_note(&before_reports, &after_reports);
}

fn evaluate(label: &str, trained: bool) -> RecallReport {
    let mut network = SimplicialNetwork::grid(experiment_config());
    let mut demo = MultimodalDemo::new(&network);

    if trained {
        demo.train_epochs(&mut network, EPOCHS);
        network.clear_activity();
    }

    demo.evaluate_recall(&mut network, label, RECALL_STEPS)
        .expect("known concept label")
}

fn print_comparison(before: &RecallReport, after: &RecallReport) {
    let ratio_gain = after.recall - before.recall;
    let surprise_gain = after.mean_target_surprise - before.mean_target_surprise;

    println!("concepto={}", after.label);
    println!(
        "  antes:  recall={:.1}% precision={:.1}% fuga={:.1}% objetivo={}/{} distractores={}/{} sorpresa_obj={:.3} energia={:.2}",
        before.recall * 100.0,
        before.precision * 100.0,
        before.leakage * 100.0,
        before.active_target_agents,
        before.target_agents,
        before.active_distractor_agents,
        before.distractor_agents,
        before.mean_target_surprise,
        before.total_free_energy
    );
    println!(
        "  despues: recall={:.1}% precision={:.1}% fuga={:.1}% objetivo={}/{} distractores={}/{} sorpresa_obj={:.3} energia={:.2}",
        after.recall * 100.0,
        after.precision * 100.0,
        after.leakage * 100.0,
        after.active_target_agents,
        after.target_agents,
        after.active_distractor_agents,
        after.distractor_agents,
        after.mean_target_surprise,
        after.total_free_energy
    );
    println!(
        "  ganancia: recall={:+.1}% sorpresa_media={:+.3} sorpresa_total_obj={:.3} sorpresa_total_fuga={:.3}",
        ratio_gain * 100.0,
        surprise_gain,
        after.target_surprise,
        after.distractor_surprise
    );
    println!();
}

fn print_summary(label: &str, reports: &[RecallReport]) {
    let n = reports.len().max(1) as f32;
    let recall = reports.iter().map(|report| report.recall).sum::<f32>() / n;
    let precision = reports.iter().map(|report| report.precision).sum::<f32>() / n;
    let leakage = reports.iter().map(|report| report.leakage).sum::<f32>() / n;
    let target_surprise = reports
        .iter()
        .map(|report| report.mean_target_surprise)
        .sum::<f32>()
        / n;
    let distractor_surprise = reports
        .iter()
        .map(|report| report.mean_distractor_surprise)
        .sum::<f32>()
        / n;

    println!(
        "resumen_{label}: recall_medio={:.1}% precision_media={:.1}% fuga_media={:.1}% sorpresa_obj={:.3} sorpresa_fuga={:.3}",
        recall * 100.0,
        precision * 100.0,
        leakage * 100.0,
        target_surprise,
        distractor_surprise
    );
}

fn print_viability_note(before: &[RecallReport], after: &[RecallReport]) {
    let n = before.len().max(1) as f32;
    let before_recall = before.iter().map(|report| report.recall).sum::<f32>() / n;
    let after_recall = after.iter().map(|report| report.recall).sum::<f32>() / n;
    let after_leakage = after.iter().map(|report| report.leakage).sum::<f32>() / n;

    println!();
    if after_recall > before_recall + 0.25 && after_leakage < after_recall {
        println!("lectura: viable como memoria asociativa topologica inicial; aun no demuestra razonamiento general.");
    } else {
        println!("lectura: la senal de aprendizaje no es suficiente; hay que ajustar dinamica o codificacion.");
    }
}

fn experiment_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 30,
        height: 18,
        spacing: 28.0,
        elasticity: 0.018,
        damping: 0.86,
        activation_threshold: 0.68,
        simplex_area_weight: 0.0008,
        max_active_agents: 96,
        inhibition_decay: 0.18,
        max_spikes_per_step: 512,
        seed: 11,
    }
}
