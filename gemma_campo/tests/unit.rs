use gemma_campo::{cpu::{self, Rng}, data::Split, features::FeatCfg};

#[test]
fn wgsl_valida() { gemma_campo::validate_wgsl().unwrap(); }

#[test]
fn dim_features() {
    let c = FeatCfg { d: 64, ctx: 4, quad: true };
    assert_eq!(c.dim(), 1 + 256 + 2080);
    assert_eq!(c.degree().len(), c.dim());
    let seq: Vec<f32> = (0..3 * 2).map(|x| x as f32).collect(); // 3 pasos, d=2
    let c2 = FeatCfg { d: 2, ctx: 3, quad: true };
    let mut o = vec![0f32; c2.dim()];
    c2.build(&seq, 1, &mut o);
    assert_eq!(o, vec![1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 4.0, 6.0, 9.0]); // padding con z_0
}

fn synth(n_conv: usize, len: usize, d: usize, seed: u64) -> (Split, Vec<f32>) {
    // dinámica lineal: rotaciones de ángulo 0.3 en pares de dims (+ decaimiento 0.9 si d impar) + ruido 0.01
    let mut r = Rng(seed);
    let (c, s_) = (0.3f32.cos(), 0.3f32.sin());
    let mut m = vec![0f32; d * d];
    let mut i = 0; while i + 1 < d { m[i * d + i] = c; m[i * d + i + 1] = -s_; m[(i + 1) * d + i] = s_; m[(i + 1) * d + i + 1] = c; i += 2; }
    if d % 2 == 1 { m[(d - 1) * d + d - 1] = 0.9; }
    let mut z = Vec::new();
    for _ in 0..n_conv { let mut x: Vec<f32> = (0..d).map(|_| r.normal() as f32).collect();
        for _ in 0..len { z.extend_from_slice(&x);
            x = (0..d).map(|i| (0..d).map(|j| m[i * d + j] * x[j]).sum::<f32>() + 0.01 * r.normal() as f32).collect(); } }
    (Split::new(d, z, vec![len; n_conv]), m)
}

#[test]
fn ridge_ajusta_y_baja_de_repeat() {
    let (s, _) = synth(20, 30, 4, 1);
    let cfg = FeatCfg { d: 4, ctx: 2, quad: false };
    let idx = cpu::samples(&s);
    let st = cpu::gram(&cfg, &s, &idx, 2);
    let w = cpu::ridge_solve(&st, &cpu::lam_per_feature(&cfg, [0.0, 1e-6, 0.0])).unwrap();
    // mse por estadísticos == mse directo
    let (f, d) = (cfg.dim(), 4);
    let mut phi = vec![0f32; idx.len() * f]; let mut y = vec![0f32; idx.len() * d];
    cpu::build_batch(&cfg, &s, &idx, &mut phi, &mut y);
    let wf: Vec<f32> = w.iter().map(|x| *x as f32).collect();
    let direct = cpu::loss_ref(&phi, &y, &wf, idx.len(), f, d);
    assert!((direct - cpu::mse_from_stats(&st, &w)).abs() < 1e-5 * (1.0 + direct));
    let zero = cpu::loss_ref(&phi, &y, &vec![0.0; f * d], idx.len(), f, d); // = repeat-last
    assert!(direct < 0.5 * zero, "ridge {direct} vs repeat {zero}");
}

#[test]
fn gradiente_sgd_coincide_con_diferencias_finitas() {
    let mut r = Rng(7);
    let (n, f, d) = (5, 6, 3);
    let phi: Vec<f32> = (0..n * f).map(|_| r.normal() as f32).collect();
    let y: Vec<f32> = (0..n * d).map(|_| r.normal() as f32).collect();
    let lam = vec![0.01f32; f];
    let w0: Vec<f32> = (0..f * d).map(|_| r.normal() as f32).collect();
    let obj = |w: &[f32]| cpu::loss_ref(&phi, &y, w, n, f, d) + w.iter().enumerate().map(|(i, x)| lam[i / d] as f64 * (*x as f64).powi(2)).sum::<f64>();
    let (mut w, mut v) = (w0.clone(), vec![0f32; f * d]);
    cpu::sgd_step_ref(&phi, &y, &mut w, &mut v, &lam, n, f, d, 1.0, 0.0); // v = grad
    for i in [0, 4, 11, 17] {
        let (mut a, mut b) = (w0.clone(), w0.clone()); a[i] += 1e-3; b[i] -= 1e-3;
        let fd = (obj(&a) - obj(&b)) / 2e-3;
        assert!((fd - v[i] as f64).abs() < 1e-3 * (1.0 + fd.abs()), "i={i} fd={fd} g={}", v[i]);
    }
}

#[test]
fn sgd_baja_la_perdida() {
    let (s, _) = synth(10, 20, 4, 3);
    let cfg = FeatCfg { d: 4, ctx: 1, quad: true };
    let idx = cpu::samples(&s); let (f, d) = (cfg.dim(), 4);
    let mut phi = vec![0f32; idx.len() * f]; let mut y = vec![0f32; idx.len() * d];
    cpu::build_batch(&cfg, &s, &idx, &mut phi, &mut y);
    let lam = vec![0f32; f]; let (mut w, mut v) = (vec![0f32; f * d], vec![0f32; f * d]);
    let l0 = cpu::sgd_step_ref(&phi, &y, &mut w, &mut v, &lam, idx.len(), f, d, 0.02, 0.9);
    let mut l = l0; for _ in 0..200 { l = cpu::sgd_step_ref(&phi, &y, &mut w, &mut v, &lam, idx.len(), f, d, 0.02, 0.9); }
    assert!(l < 0.5 * l0, "{l0} -> {l}");
}

#[test]
fn repeat_baseline_y_rapido_causal() {
    let (s, _) = synth(3, 15, 4, 5);
    let cfg = FeatCfg { d: 4, ctx: 1, quad: false };
    let m = cpu::Model { meta: cpu::ModelMeta { cfg, lam_deg: [0.0; 3], fast_eta: 0.0, fast_gamma: 1.0, note: String::new() }, w: vec![0.0; cfg.dim() * 4] };
    // W=0 y eta=0 -> campo == lenta == repeat exactamente
    let e = cpu::eval_conv(&m, s.conv(0), 15, 3, 1, 0.0, 1.0, None);
    assert!((e.field1 - e.repeat1).abs() < 1e-6 * e.repeat1 && (e.slow1 - e.repeat1).abs() < 1e-6 * e.repeat1);
    for k in 0..3 { assert!((e.fieldk[k] - e.repeatk[k]).abs() < 1e-6 * e.repeatk[k]); }
    // el rápido aprende la dinámica lineal dentro de la conversación (en conv larga)
    let (s2, _) = synth(1, 400, 4, 9);
    let e2 = cpu::eval_conv(&m, s2.conv(0), 400, 1, 1, 0.3, 1.0, None);
    assert!(e2.field1 < e2.repeat1, "rapido {} repeat {}", e2.field1, e2.repeat1);
}
