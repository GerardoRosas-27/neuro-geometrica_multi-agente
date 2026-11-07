//! Pérdida GPU == referencia CPU en un batch pequeño. Si no hay adaptador wgpu, el test se omite (lo dice).
use gemma_campo::{cpu::{self, Rng}, features::FeatCfg, gpu::Gpu};

#[test]
fn gpu_igual_a_cpu() {
    let cfg = FeatCfg { d: 6, ctx: 2, quad: true, bq10: true };
    let (n, f, d) = (3000, cfg.dim(), 6); // n*d=18000 > 256*NWG=16384: varios elementos por hilo en la reducción
    let mut r = Rng(12345);
    let phi: Vec<f32> = (0..n * f).map(|i| if i % f == 0 { 1.0 } else { r.normal() as f32 }).collect();
    let y: Vec<f32> = (0..n * d).map(|_| r.normal() as f32).collect();
    let lam: Vec<f32> = cpu::lam_per_feature(&cfg, [0.0, 1e-3, 1e-2]).iter().map(|x| *x as f32).collect();
    let w0: Vec<f32> = (0..f * d).map(|_| 0.05 * r.normal() as f32).collect();
    let gpu = match Gpu::new(n, f, d, &lam, &w0) { Ok(g) => g, Err(e) => { eprintln!("OMITIDO: sin GPU ({e})"); return; } };
    let (mut w, mut v) = (w0.clone(), vec![0f32; f * d]);
    for _ in 0..4 {
        let lc = cpu::sgd_step_ref(&phi, &y, &mut w, &mut v, &lam, n, f, d, 0.01, 0.9);
        let lg = gpu.step(&phi, &y, n, 0.01, 0.9, true);
        assert!((lc - lg).abs() < 1e-5 * lc, "cpu {lc} gpu {lg}");
    }
    let wg = gpu.weights();
    for (a, b) in w.iter().zip(&wg) { assert!((a - b).abs() < 1e-5); }
    // batch parcial (n < cap) también coincide
    let lc = cpu::loss_ref(&phi, &y, &wg, 37, f, d);
    let lg = gpu.step(&phi, &y, 37, 0.0, 0.0, false);
    assert!((lc - lg).abs() < 1e-5 * lc);
}
