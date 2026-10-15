//! Encoder lingüístico congelado: Gemma 2 2B GGUF (sin generación).
//! Por texto: media sobre tokens (sin BOS) del estado oculto final (pre-norm) → f32[d].
use cdt_rqm_epr::native_gemma2::{Gemma2Tokenizer, QuantizedGemma2};
use candle_core::{Device, IndexOp, Tensor};
use std::io::{BufRead, Write};
use std::time::Instant;

pub fn run(a: &[String], arg: impl Fn(&str) -> Option<String>) {
    let _ = a;
    let model = arg("--model").expect("--model");
    let inp = arg("--in").expect("--in");
    let out = arg("--out").expect("--out");
    let exit_after: Option<usize> = arg("--exit-after").map(|v| v.parse().unwrap());
    let t0 = Instant::now();
    let dev = Device::Cpu;
    let mut f = std::fs::File::open(&model).unwrap();
    let content = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
    let tok = Gemma2Tokenizer::from_gguf(&content).unwrap();
    let mut m = QuantizedGemma2::from_gguf(content, &mut f, &dev).unwrap();
    let load_s = t0.elapsed().as_secs_f64();
    let texts: Vec<String> = std::io::BufReader::new(std::fs::File::open(&inp).unwrap()).lines().map(|l| l.unwrap()).collect();
    // reanudable: se agregan filas al final
    let done = std::fs::metadata(&out).map(|x| x.len()).unwrap_or(0);
    let mut o = std::fs::OpenOptions::new().create(true).append(true).open(&out).unwrap();
    let mut dim = 0usize; let mut start = 0usize; let mut ntok = 0usize;
    let t1 = Instant::now();
    for (i, t) in texts.iter().enumerate() {
        let mut ids = vec![2u32]; ids.extend(tok.encode(t).unwrap());
        if dim > 0 && i < start { continue; }
        m.clear_kv_cache();
        let x = Tensor::new(ids.as_slice(), &dev).unwrap().unsqueeze(0).unwrap();
        let h = m.forward_with_mask(&x, 0, None, exit_after, false, true).unwrap().sequence_hidden.unwrap();
        let h = h.i((0, 1.., ..)).unwrap().mean(0).unwrap().to_dtype(candle_core::DType::F32).unwrap().to_vec1::<f32>().unwrap();
        if dim == 0 {
            dim = h.len(); start = (done as usize) / (4 * dim);
            if i < start { continue; }
        }
        for v in &h { o.write_all(&v.to_le_bytes()).unwrap(); }
        ntok += ids.len();
        if (i + 1) % 50 == 0 { eprintln!("{} / {} textos, {:.3} s/texto", i + 1, texts.len(), t1.elapsed().as_secs_f64() / (i + 1 - start) as f64); }
    }
    let el = t1.elapsed().as_secs_f64();
    let n = texts.len().saturating_sub(start);
    println!("{{\"model\":\"{model}\",\"exit_after\":\"{exit_after:?}\",\"dim\":{dim},\"n_texts\":{n},\"resumed_from\":{start},\"tokens\":{ntok},\"load_s\":{load_s:.2},\"encode_s\":{el:.2},\"s_per_text\":{:.4},\"tok_per_s\":{:.2}}}", el / n.max(1) as f64, ntok as f64 / el.max(1e-9));
}
