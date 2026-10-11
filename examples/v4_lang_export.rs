//! §18: export pairs, anchors, §11 splits and the frozen random encoder
//! embeddings for the JAX Ψ64 pipeline (training/v4_psi64/).
use cdt_rqm_epr::v4_language::*;
fn main() {
    let pairs: Vec<serde_json::Value> = all_pairs()
        .iter()
        .map(|p| serde_json::json!({"concept": p.concept, "rel": p.rel, "lang": p.lang, "src": p.src, "dst": p.dst}))
        .collect();
    let mut splits = serde_json::Map::new();
    for s in (0xA400u64..0xA410).chain(0xB400..0xB410).chain([0x5A00]) {
        splits.insert(format!("0x{s:X}"), serde_json::json!(train_concepts(s)));
    }
    let rand: std::collections::BTreeMap<String, Vec<f64>> = all_texts()
        .into_iter()
        .map(|t| {
            let v = random_encoder(&t, 512);
            (t, v)
        })
        .collect();
    let out = serde_json::json!({"langs": LANGS, "pairs": pairs, "anchors": ANCHORS.to_vec(), "train_concepts": splits});
    std::fs::create_dir_all("artifacts/v4/psi64").unwrap();
    std::fs::write("artifacts/v4/psi64/lang_task.json", out.to_string()).unwrap();
    std::fs::write(
        "artifacts/v4/psi64/emb_random512.json",
        serde_json::to_string(&rand).unwrap(),
    )
    .unwrap();
    println!("ok pairs={} texts={}", all_pairs().len(), rand.len());
}
