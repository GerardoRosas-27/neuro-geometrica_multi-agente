//! E43/E44: embed every task/anchor word once with frozen LLMs and cache.
//! gemma: gemma-2-2b-it-Q3_K_L.gguf (dense, mean of final sequence hidden)
//! qwen: qwen2.5-1.5b-instruct-q4_k_m.gguf (vendored loader, mean final normed hidden)
use candle_core::{Device, Tensor};
use cdt_rqm_epr::native_gemma2::{Gemma2Tokenizer, QuantizedGemma2};
use cdt_rqm_epr::v4_language::all_texts;
use cdt_rqm_epr::v4_qwen2_hidden::ModelWeights;
use std::collections::BTreeMap;
use std::fs::File;
fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "gemma".into());
    let dev = Device::Cpu;
    let texts = all_texts();
    let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let t0 = std::time::Instant::now();
    if which == "gemma" {
        let path = "/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q3_K_L.gguf";
        let mut f = File::open(path).unwrap();
        let content = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
        let tok = Gemma2Tokenizer::from_gguf(&content).unwrap();
        let mut model = QuantizedGemma2::from_gguf(content, &mut f, &dev).unwrap();
        for t in &texts {
            let mut ids = vec![tok.bos_id];
            ids.extend(tok.encode(t).unwrap());
            let x = Tensor::new(ids.as_slice(), &dev)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            model.clear_kv_cache();
            let o = model
                .forward_with_mask(&x, 0, None, None, false, true)
                .unwrap();
            let h = o
                .sequence_hidden
                .unwrap()
                .mean(1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(candle_core::DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            out.insert(t.clone(), h.iter().map(|v| *v as f64).collect());
        }
    } else {
        let path = "/workspace/models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
        let mut f = File::open(path).unwrap();
        let content = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
        let mut model = ModelWeights::from_gguf(content, &mut f, &dev).unwrap();
        let tok =
            tokenizers::Tokenizer::from_file("/workspace/models/qwen25tok/tokenizer.json").unwrap();
        for t in &texts {
            let ids = tok.encode(t.as_str(), false).unwrap().get_ids().to_vec();
            let x = Tensor::new(ids.as_slice(), &dev)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            model.clear_kv_cache();
            let h = model.forward_hidden(&x).unwrap();
            out.insert(t.clone(), h.iter().map(|v| *v as f64).collect());
        }
    }
    std::fs::create_dir_all("artifacts/v4/lang").unwrap();
    std::fs::write(
        format!("artifacts/v4/lang/emb_{which}.json"),
        serde_json::to_string(&out).unwrap(),
    )
    .unwrap();
    eprintln!(
        "{which}: {} texts in {:.1}s",
        texts.len(),
        t0.elapsed().as_secs_f64()
    );
}
