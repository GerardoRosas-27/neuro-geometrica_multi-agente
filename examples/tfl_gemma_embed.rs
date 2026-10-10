//! tfl-gen-v2: hidden final medio (mean-pool) de Gemma-2-2B-it Q3_K_L para
//! una lista de textos (JSON array). Salida: f32 little-endian [n, 2304].
//! Uso: cargo run --release --example tfl_gemma_embed -- <texts.json> <out.f32> [start] [end]
//! Mismo procedimiento que `examples/v4_llm_embed.rs` de v4 (infra genérica).
use candle_core::{Device, Tensor};
use cdt_rqm_epr::native_gemma2::{Gemma2Tokenizer, QuantizedGemma2};
use std::fs::File;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let texts: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let start: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let end: usize = a
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(texts.len())
        .min(texts.len());
    let path = "/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q3_K_L.gguf";
    let dev = Device::Cpu;
    let mut f = File::open(path).unwrap();
    let content = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
    let tok = Gemma2Tokenizer::from_gguf(&content).unwrap();
    let mut model = QuantizedGemma2::from_gguf(content, &mut f, &dev).unwrap();
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&a[2])
        .unwrap();
    let t0 = std::time::Instant::now();
    for (i, t) in texts[start..end].iter().enumerate() {
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
        let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
        out.write_all(&bytes).unwrap();
        if (i + 1) % 100 == 0 {
            eprintln!(
                "{} textos, {:.1}s",
                start + i + 1,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!(
        "hecho {}..{} en {:.1}s",
        start,
        end,
        t0.elapsed().as_secs_f64()
    );
}
