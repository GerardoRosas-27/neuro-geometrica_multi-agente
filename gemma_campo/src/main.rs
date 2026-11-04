use gemma_campo::{cpu::{self, Model, ModelMeta, Rng}, data::{self, Split}, features::FeatCfg, gpu::Gpu};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

const USAGE: &str = "uso:
  gemma_campo validate-wgsl
  gemma_campo check-gpu [--n 96 --d 8 --ctx 2 --steps 5]   (pérdida GPU == referencia CPU)
  gemma_campo train --data DIR --out DIR [--device gpu|cpu] [--ctx 4] [--quad 1] [--epochs 5] [--batch 1024]
                    [--lr 0.05] [--mom 0.9] [--ridge-max 50000] [--init ridge|zero] [--threads 4] [--seed 12345]
                    [--init-model DIR] [--val-data DIR] [--replay DIR1,DIR2]   (modo continuo)
  gemma_campo predict --data DIR --model DIR --split test [--kmax 8] [--stride 4] [--threads 4]";

fn args() -> (String, HashMap<String, String>) {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let cmd = a.first().cloned().unwrap_or_default();
    let mut m = HashMap::new();
    let mut i = 1;
    while i < a.len() { if let Some(k) = a[i].strip_prefix("--") { m.insert(k.to_string(), a.get(i + 1).cloned().unwrap_or_default()); i += 2; } else { i += 1; } }
    (cmd, m)
}
fn g<T: std::str::FromStr>(m: &HashMap<String, String>, k: &str, def: T) -> T { m.get(k).and_then(|v| v.parse().ok()).unwrap_or(def) }

fn main() {
    let (cmd, m) = args();
    match cmd.as_str() {
        "validate-wgsl" => match gemma_campo::validate_wgsl() { Ok(()) => println!("WGSL OK (naga)"), Err(e) => { eprintln!("{e}"); std::process::exit(1) } },
        "check-gpu" => check_gpu(&m),
        "train" => train(&m),
        "predict" => predict(&m),
        _ => { eprintln!("{USAGE}"); std::process::exit(2) }
    }
}

fn check_gpu(m: &HashMap<String, String>) {
    let (n, d, ctx, steps) = (g(m, "n", 96usize), g(m, "d", 8usize), g(m, "ctx", 2usize), g(m, "steps", 5usize));
    let cfg = FeatCfg { d, ctx, quad: true };
    let f = cfg.dim();
    let mut r = Rng(12345);
    let phi: Vec<f32> = (0..n * f).map(|i| if i % f == 0 { 1.0 } else { r.normal() as f32 }).collect();
    let y: Vec<f32> = (0..n * d).map(|_| r.normal() as f32).collect();
    let lam: Vec<f32> = cpu::lam_per_feature(&cfg, [0.0, 1e-3, 1e-2]).iter().map(|x| *x as f32).collect();
    let w0: Vec<f32> = (0..f * d).map(|_| 0.05 * r.normal() as f32).collect();
    let gpu = match Gpu::new(n, f, d, &lam, &w0) { Ok(g) => g, Err(e) => { eprintln!("GPU no disponible: {e}"); std::process::exit(3) } };
    println!("adaptador: {}", gpu.adapter_name);
    let (mut w, mut v) = (w0.clone(), vec![0f32; f * d]);
    let mut worst = 0f64;
    for s in 0..steps {
        let lc = cpu::sgd_step_ref(&phi, &y, &mut w, &mut v, &lam, n, f, d, 0.01, 0.9);
        let lg = gpu.step(&phi, &y, n, 0.01, 0.9, true);
        let rel = (lg - lc).abs() / lc.abs().max(1e-12);
        worst = worst.max(rel);
        println!("paso {s}: loss_cpu={lc:.8} loss_gpu={lg:.8} rel={rel:.2e}");
    }
    let wg = gpu.weights();
    let dw = w.iter().zip(&wg).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    // determinismo: repetir el forward da exactamente el mismo valor
    let l1 = gpu.step(&phi, &y, n, 0.0, 0.0, false); let l2 = gpu.step(&phi, &y, n, 0.0, 0.0, false);
    println!("max|W_cpu-W_gpu|={dw:.3e}  determinista={}", l1 == l2);
    let ok = worst < 1e-4 && dw < 1e-4 && l1 == l2;
    println!("{}", if ok { "CHECK-GPU OK" } else { "CHECK-GPU FALLA" });
    if !ok { std::process::exit(1) }
}

fn log(f: &mut std::fs::File, s: &str) { println!("{s}"); let _ = writeln!(f, "{s}"); }

fn save_model(dir: &Path, name: &str, meta: &ModelMeta, w: &[f32]) {
    data::write_f32(&dir.join(format!("{name}.bin")), w).unwrap();
    std::fs::write(dir.join(format!("{name}.json")), serde_json::to_string_pretty(meta).unwrap()).unwrap();
}

fn train(m: &HashMap<String, String>) {
    let dd = PathBuf::from(m.get("data").expect("--data"));
    let out = PathBuf::from(m.get("out").expect("--out"));
    std::fs::create_dir_all(&out).unwrap();
    let mut lf = std::fs::File::create(out.join("train_log.txt")).unwrap();
    let mut tr = Split::load(&dd, "train").unwrap();
    // --replay DIR1,DIR2: concatena el train de shards anteriores (buffer de repetición contra olvido)
    if let Some(rp) = m.get("replay") { for r in rp.split(',').filter(|x| !x.is_empty()) {
        let o = Split::load(Path::new(r), "train").unwrap(); assert_eq!(o.d, tr.d);
        let mut z = tr.z.clone(); z.extend_from_slice(&o.z); let mut l = tr.lengths.clone(); l.extend_from_slice(&o.lengths);
        tr = Split::new(tr.d, z, l); } }
    // --val-data DIR: val fijo (held-out) en otro directorio (p.ej. el del shard 0)
    let vd = m.get("val-data").map(PathBuf::from).unwrap_or(dd.clone());
    let va = Split::load(&vd, "val").unwrap();
    // --init-model DIR: continúa desde un model.bin/json (sin rejilla ridge; conserva cfg y lambdas)
    let init_meta: Option<(ModelMeta, Vec<f32>)> = m.get("init-model").map(|d| { let d = PathBuf::from(d);
        (serde_json::from_slice(&std::fs::read(d.join("model.json")).unwrap()).unwrap(), data::read_f32(&d.join("model.bin")).unwrap()) });
    let cfg = match &init_meta { Some((mm, _)) => mm.cfg, None => FeatCfg { d: tr.d, ctx: g(m, "ctx", 4usize), quad: g(m, "quad", 1u8) == 1 } };
    assert_eq!(cfg.d, tr.d, "d del modelo inicial != d de los datos");
    let (f, d) = (cfg.dim(), cfg.d);
    let threads = g(m, "threads", 4usize);
    let seed = g(m, "seed", 12345u64);
    let device = m.get("device").cloned().unwrap_or("gpu".into());
    log(&mut lf, &format!("datos: train {} conv/{} tok, val {} conv/{} tok; d={d} F={f} ctx={} quad={}", tr.lengths.len(), tr.n_tokens(), va.lengths.len(), va.n_tokens(), cfg.ctx, cfg.quad));
    let mut rng = Rng(seed);
    let mut tr_idx = cpu::samples(&tr);
    let mut va_idx = cpu::samples(&va);
    rng.shuffle(&mut tr_idx); rng.shuffle(&mut va_idx);
    // ---- M7: ridge por grado, lambda elegido en val ----
    let rmax = g(m, "ridge-max", 50000usize);
    let st_tr = cpu::gram(&cfg, &tr, &tr_idx[..rmax.min(tr_idx.len())], threads);
    let st_va = cpu::gram(&cfg, &va, &va_idx[..rmax.min(va_idx.len())], threads);
    let grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
    let grid2: Vec<f64> = if cfg.quad { grid.to_vec() } else { vec![0.0] };
    let mut best = (f64::INFINITY, [0.0; 3], Vec::new());
    if let Some((mm, w)) = &init_meta { best = (cpu::mse_from_stats(&st_va, &w.iter().map(|x| *x as f64).collect::<Vec<_>>()), mm.lam_deg, w.iter().map(|x| *x as f64).collect());
        log(&mut lf, &format!("init-model: sin rejilla ridge; l1={:e} l2={:e} val_mse(dz) inicial={:.6}", mm.lam_deg[1], mm.lam_deg[2], best.0)); }
    for &l1 in if init_meta.is_some() { &grid[..0] } else { &grid[..] } { for &l2 in &grid2 {
        let lam = [0.0, l1, l2];
        if let Some(w) = cpu::ridge_solve(&st_tr, &cpu::lam_per_feature(&cfg, lam)) {
            let vm = cpu::mse_from_stats(&st_va, &w);
            log(&mut lf, &format!("ridge l1={l1:e} l2={l2:e}: train_mse={:.6} val_mse={vm:.6}", cpu::mse_from_stats(&st_tr, &w)));
            if vm < best.0 { best = (vm, lam, w); }
        }
    } }
    let lam_deg = best.1;
    log(&mut lf, &format!("ridge elegido (val): l1={:e} l2={:e} val_mse(dz)={:.6}  [ridge con {} muestras train]", lam_deg[1], lam_deg[2], best.0, st_tr.n));
    let lam: Vec<f32> = cpu::lam_per_feature(&cfg, lam_deg).iter().map(|x| *x as f32).collect();
    let w0: Vec<f32> = if m.get("init").map(|s| s.as_str()) == Some("zero") { vec![0.0; f * d] } else { best.2.iter().map(|x| *x as f32).collect() };
    let mut meta = ModelMeta { cfg, lam_deg, fast_eta: 0.0, fast_gamma: 1.0, note: String::new() };
    save_model(&out, "ridge", &meta, &w0);
    // ---- SGD-momentum sobre todo train (GPU o gemelo CPU) ----
    let (epochs, bs, lr, mom) = (g(m, "epochs", 5usize), g(m, "batch", 1024usize), g(m, "lr", 0.05f32), g(m, "mom", 0.9f32));
    let gpu = if device == "gpu" { match Gpu::new(bs, f, d, &lam, &w0) { Ok(g) => { log(&mut lf, &format!("GPU: {}", g.adapter_name)); Some(g) }, Err(e) => { eprintln!("GPU no disponible ({e}); usa --device cpu"); std::process::exit(3) } } } else { None };
    let (mut wc, mut vc) = (w0.clone(), vec![0f32; f * d]);
    let mut phi = vec![0f32; bs * f]; let mut y = vec![0f32; bs * d];
    let val_loss = |w: &[f32]| -> f64 { let mut wd = vec![0f64; f * d]; for i in 0..f * d { wd[i] = w[i] as f64; } cpu::mse_from_stats(&st_va, &wd) };
    let mut best_val = val_loss(&w0);
    log(&mut lf, &format!("epoch 0 (init): val_mse(dz)={best_val:.6}"));
    let mut best_w = w0.clone();
    for ep in 1..=epochs {
        rng.shuffle(&mut tr_idx);
        let (mut s, mut cnt) = (0.0, 0usize);
        for ch in tr_idx.chunks(bs) {
            cpu::build_batch(&cfg, &tr, ch, &mut phi, &mut y);
            let l = match &gpu { Some(g) => g.step(&phi, &y, ch.len(), lr, mom, true), None => cpu::sgd_step_ref(&phi, &y, &mut wc, &mut vc, &lam, ch.len(), f, d, lr, mom) };
            if !l.is_finite() { log(&mut lf, "pérdida no finita: abortando"); std::process::exit(4) }
            s += l * ch.len() as f64; cnt += ch.len();
        }
        let w = match &gpu { Some(g) => g.weights(), None => wc.clone() };
        let vl = val_loss(&w);
        log(&mut lf, &format!("epoch {ep}: train_mse(dz, media batches pre-update)={:.6} val_mse(dz)={vl:.6}", s / cnt as f64));
        save_model(&out, &format!("ckpt_epoch{ep}"), &meta, &w);
        if vl < best_val { best_val = vl; best_w = w; }
    }
    // ---- campo rápido: eta/gamma en val ----
    let mut model = Model { meta: meta.clone(), w: best_w.clone() };
    let mut bf = (f64::INFINITY, 0.0, 1.0);
    for &eta in &[0.0, 0.01, 0.03, 0.1, 0.3] { for &gm in &[1.0, 0.99, 0.95] {
        if eta == 0.0 && gm != 1.0 { continue; }
        let (fl, sl, rp) = cpu::eval_split_1step(&model, &va, eta, gm, threads);
        log(&mut lf, &format!("rapido eta={eta} gamma={gm}: val z-MSE campo={fl:.6} lenta={sl:.6} repeat={rp:.6}"));
        if fl < bf.0 { bf = (fl, eta, gm); }
    } }
    meta.fast_eta = bf.1; meta.fast_gamma = bf.2;
    meta.note = format!("best val dz-mse SGD={best_val:.6}; fast elegido eta={} gamma={} val z-mse={:.6}; device={device}", bf.1, bf.2, bf.0);
    model.meta = meta.clone();
    log(&mut lf, &meta.note);
    save_model(&out, "model", &meta, &best_w);
}

fn predict(m: &HashMap<String, String>) {
    let dd = PathBuf::from(m.get("data").expect("--data"));
    let md = PathBuf::from(m.get("model").expect("--model"));
    let split = m.get("split").cloned().unwrap_or("test".into());
    let (kmax, stride, threads) = (g(m, "kmax", 8usize), g(m, "stride", 4usize), g(m, "threads", 4usize));
    let s = Split::load(&dd, &split).unwrap();
    let meta: ModelMeta = serde_json::from_slice(&std::fs::read(md.join("model.json")).unwrap()).unwrap();
    let w = data::read_f32(&md.join("model.bin")).unwrap();
    assert_eq!(w.len(), meta.cfg.dim() * meta.cfg.d);
    assert_eq!(meta.cfg.d, s.d);
    let model = Model { meta: meta.clone(), w };
    let d = s.d;
    let mut pred = vec![0f32; s.z.len()];
    let convs: Vec<usize> = (0..s.lengths.len()).collect();
    let mut slices: Vec<&mut [f32]> = Vec::new();
    { let mut rest = pred.as_mut_slice(); for &l in &s.lengths { let (a, b) = rest.split_at_mut(l * d); slices.push(a); rest = b; } }
    let per = convs.len().div_ceil(threads.max(1)).max(1);
    let mut evals: Vec<cpu::ConvEval> = Vec::new();
    std::thread::scope(|sc| {
        let mut hs = Vec::new();
        let mut it = slices.into_iter().enumerate().peekable();
        while it.peek().is_some() {
            let chunk: Vec<(usize, &mut [f32])> = it.by_ref().take(per).collect();
            let (model, s) = (&model, &s);
            hs.push(sc.spawn(move || chunk.into_iter().map(|(ci, p)| { let mut e = cpu::eval_conv(model, s.conv(ci), s.lengths[ci], kmax, stride, model.meta.fast_eta, model.meta.fast_gamma, Some(p)); e.conv = ci; e }).collect::<Vec<_>>()));
        }
        for h in hs { evals.extend(h.join().unwrap()); }
    });
    data::write_f32(&md.join(format!("{split}_pred1.f32")), &pred).unwrap();
    std::fs::write(md.join(format!("{split}_zmse.json")), serde_json::to_string(&serde_json::json!({"kmax": kmax, "stride": stride, "meta": meta, "convs": evals})).unwrap()).unwrap();
    let n: f64 = evals.iter().map(|e| e.n1 as f64).sum();
    let avg = |f: &dyn Fn(&cpu::ConvEval) -> f64| evals.iter().map(|e| f(e) * e.n1 as f64).sum::<f64>() / n;
    println!("[{split}] z-MSE 1 paso: campo={:.6} lenta={:.6} repeat={:.6}  (ponderado por token; bootstrap en python/eval.py)", avg(&|e| e.field1), avg(&|e| e.slow1), avg(&|e| e.repeat1));
    println!("escrito {}/{split}_pred1.f32 y {split}_zmse.json", md.display());
}
