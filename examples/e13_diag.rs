use cdt_rqm_epr::field_gemma_probe::FrozenGemma2Probe;
use cdt_rqm_epr::field_linguistic_layer::FrozenLinguisticProbe;
use cdt_rqm_epr::liquid_experiments_11_17::{gemma_gguf_available, run_experiment_13};

fn main() {
    assert!(gemma_gguf_available(), "need GGUF");
    let mut p = FrozenGemma2Probe::try_open(None).expect("open gguf");
    let words = ["perro", "gato", "lobo", "águila", "mesa", "animal"];
    let feats: Vec<Vec<f64>> = words
        .iter()
        .map(|w| {
            let packet = p.analyze(w);
            let mut f = packet.hidden.clone();
            let n = f.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            for x in &mut f {
                *x /= n;
            }
            f
        })
        .collect();
    let cos = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    println!("GGUF hidden cosines:");
    for (i, wi) in words.iter().enumerate() {
        for (j, wj) in words.iter().enumerate() {
            if j <= i {
                continue;
            }
            println!("  cos({wi},{wj})={:.3}", cos(&feats[i], &feats[j]));
        }
    }
    let r = run_experiment_13(0xE1100);
    println!("\nnotes={}", r.notes);
    println!("verdict={} unseen={:.3}", r.verdict, r.accuracy_unseen);
}
