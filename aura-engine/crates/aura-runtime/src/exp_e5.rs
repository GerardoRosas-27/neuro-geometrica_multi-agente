//! Verificación de Ingeniería E5:
//! - Exactitud algebraica bit a bit de `TernaryMatrix::project_zero_mul` frente a matvec `i32`.
//! - Densidad de memoria empaquetada (`2 bits/peso` vs `32 bits/peso f32` = 16x).
//! - Microbenchmark de latencia/throughput en CPU para matrices `256x256` y `1024x1024`
//!   (30% densidad ternaria típica) vs producto denso `f32`.

use aura_math::{TernaryMatrix, Xorshift64};

#[derive(Clone, Debug)]
pub struct E5Report {
    pub max_abs_diff_zero_mul: i32,
    pub memory_ratio_f32_over_packed: usize,
    pub dim256_zero_mul_us: f64,
    pub dim256_f32_dense_us: f64,
    pub dim1024_zero_mul_us: f64,
    pub dim1024_f32_dense_us: f64,
    pub dim1024_packed_bytes: usize,
    pub dim1024_f32_bytes: usize,
}

fn dense_f32_matvec(mat: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for (r, slot) in out.iter_mut().enumerate().take(rows) {
        let row = &mat[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        *slot = acc;
    }
}

pub fn run_e5_verification(seed: u64, iters_256: usize, iters_1024: usize) -> E5Report {
    let mut rng = Xorshift64::from_seed(seed).split_stream(0xE500_BE4C);

    // 1. Verificación algebraica exacta `256 x 256` (30% densidad).
    let m256 = TernaryMatrix::new_random_sparse(256, 256, 30, &mut rng);
    let mut x256_i8 = [0i8; 256];
    let mut x256_f32 = [0.0f32; 256];
    for i in 0..256 {
        let v = rng.next_signed_byte().clamp(-127, 127) as i8;
        x256_i8[i] = v;
        x256_f32[i] = v as f32;
    }
    let mut out256_zm = [0i32; 256];
    m256.project_zero_mul(&x256_i8, &mut out256_zm);

    let mut dense256 = vec![0.0f32; 256 * 256];
    let mut max_diff = 0i32;
    for r in 0..256 {
        let mut expected = 0i32;
        for c in 0..256 {
            let w = m256.weight_at(r, c);
            dense256[r * 256 + c] = w as f32;
            expected += (w as i32) * (x256_i8[c] as i32);
        }
        max_diff = max_diff.max((out256_zm[r] - expected).abs());
    }

    // 2. Microbenchmark 256x256
    let mut out256_f32 = [0.0f32; 256];
    let t0 = std::time::Instant::now();
    let mut sink_i: i64 = 0;
    for _ in 0..iters_256 {
        m256.project_zero_mul(&x256_i8, &mut out256_zm);
        sink_i = sink_i.wrapping_add(out256_zm[0] as i64);
    }
    let zm256_us = (t0.elapsed().as_secs_f64() * 1e6) / (iters_256.max(1) as f64);

    let t1 = std::time::Instant::now();
    let mut sink_f: f32 = 0.0;
    for _ in 0..iters_256 {
        dense_f32_matvec(&dense256, &x256_f32, &mut out256_f32, 256, 256);
        sink_f += out256_f32[0];
    }
    let f32_256_us = (t1.elapsed().as_secs_f64() * 1e6) / (iters_256.max(1) as f64);

    // 3. Microbenchmark 1024x1024 (30% densidad ternaria típica -> 256 KB en bit-planes vs 4 MB f32).
    let m1024 = TernaryMatrix::new_random_sparse(1024, 1024, 30, &mut rng);
    let mut x1024_i8 = vec![0i8; 1024];
    let mut x1024_f32 = vec![0.0f32; 1024];
    for i in 0..1024 {
        let v = rng.next_signed_byte().clamp(-127, 127) as i8;
        x1024_i8[i] = v;
        x1024_f32[i] = v as f32;
    }
    let mut dense1024 = vec![0.0f32; 1024 * 1024];
    for r in 0..1024 {
        for c in 0..1024 {
            dense1024[r * 1024 + c] = m1024.weight_at(r, c) as f32;
        }
    }
    let mut out1024_zm = vec![0i32; 1024];
    let mut out1024_f32 = vec![0.0f32; 1024];

    let t2 = std::time::Instant::now();
    for _ in 0..iters_1024 {
        m1024.project_zero_mul(&x1024_i8, &mut out1024_zm);
        sink_i = sink_i.wrapping_add(out1024_zm[0] as i64);
    }
    let zm1024_us = (t2.elapsed().as_secs_f64() * 1e6) / (iters_1024.max(1) as f64);

    let t3 = std::time::Instant::now();
    for _ in 0..iters_1024 {
        dense_f32_matvec(&dense1024, &x1024_f32, &mut out1024_f32, 1024, 1024);
        sink_f += out1024_f32[0];
    }
    let f32_1024_us = (t3.elapsed().as_secs_f64() * 1e6) / (iters_1024.max(1) as f64);

    std::hint::black_box((sink_i, sink_f));

    E5Report {
        max_abs_diff_zero_mul: max_diff,
        memory_ratio_f32_over_packed: m1024.memory_bytes_f32_equiv() / m1024.memory_bytes_packed(),
        dim256_zero_mul_us: zm256_us,
        dim256_f32_dense_us: f32_256_us,
        dim1024_zero_mul_us: zm1024_us,
        dim1024_f32_dense_us: f32_1024_us,
        dim1024_packed_bytes: m1024.memory_bytes_packed(),
        dim1024_f32_bytes: m1024.memory_bytes_f32_equiv(),
    }
}
