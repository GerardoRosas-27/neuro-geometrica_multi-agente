//! B1 — Bottleneck 1: Silicon Mismatch & Memory-Bound Scatter Benchmark.
//! Compares:
//! 1. Proposal's naive scalar bit-scan (`while pos != 0 { trailing_zeros }`) vs
//!    dense f32 GEMV/GEMM vs Bottleneck 1 Fix (`lut4_avx2_tiled` zero-mul AVX2 vpshufb + L1 tiling)
//!    across densities 30% & 67% (BitNet b1.58), shapes 256x256 & 1024x1024, B=1 and B=16.
//! 2. Random-scatter CSR vs Minicolumn-tiled CSR in `CorticalFabric` (2,048 neurons).
//! 3. Proposal's heap-allocating unnormalized `FastWeightMemory` vs Zero-Alloc AVX2 NLMS.

use aura_cortex::{CorticalFabric, FabricConfig};
use aura_math::{DenseF32Matrix, FastWeightMemory, NaiveBitplaneMatrix, TernaryMatrix, Xorshift64};
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct B1Report {
    pub seed: u64,
    pub max_abs_diff: i32,
    pub mem_ratio_1024: f64,
    pub packed_bytes_1024: usize,
    pub dense_bytes_1024: usize,

    // 256x256 B=1 (67% BitNet density)
    pub us_256_d67_naive: f64,
    pub us_256_d67_f32: f64,
    pub us_256_d67_lut4_avx2: f64,

    // 1024x1024 B=1 (30% sparse & 67% BitNet density)
    pub us_1024_d30_naive: f64,
    pub us_1024_d30_f32: f64,
    pub us_1024_d30_lut4_avx2: f64,
    pub us_1024_d67_naive: f64,
    pub us_1024_d67_f32: f64,
    pub us_1024_d67_lut4_avx2: f64,

    // 1024x1024 Batched B=16 (67% BitNet training mini-batch!) -> us per batch
    pub us_1024_b16_naive: f64,
    pub us_1024_b16_f32: f64,
    pub us_1024_b16_lut4_tiled: f64,
    pub speedup_b16_vs_naive: f64,
    pub speedup_b16_vs_f32: f64,

    // CorticalFabric 2,048-neuron random scatter vs minicolumn-tiled CSR throughput
    pub fabric_scatter_tps: f64,
    pub fabric_tiled_tps: f64,
    pub fabric_tiled_speedup: f64,

    // FastWeightMemory naive alloc+unnormalized vs zero-alloc AVX2 NLMS
    pub fw_naive_ups: f64,
    pub fw_nlms_ups: f64,
    pub fw_nlms_speedup: f64,
    pub fw_naive_large_err: i64,
    pub fw_nlms_large_err: i64,
}

fn bench_single(
    rows: usize,
    cols: usize,
    density: f32,
    iters: usize,
    rng: &mut Xorshift64,
) -> (i32, f64, f64, f64, usize, usize) {
    let naive = NaiveBitplaneMatrix::from_random_density(rows, cols, density, rng);
    let tiled = TernaryMatrix::from_naive(&naive);
    let dense = DenseF32Matrix::from_ternary(&tiled);

    let x_i8: Vec<i8> = (0..cols)
        .map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8)
        .collect();
    let x_f32: Vec<f32> = x_i8.iter().map(|&v| v as f32).collect();

    let mut out_ref = vec![0i32; rows];
    let mut out_naive = vec![0i32; rows];
    let mut out_simd = vec![0i32; rows];
    let mut out_f32 = vec![0.0f32; rows];

    naive.project_reference_i32(&x_i8, &mut out_ref);
    naive.project_naive_bitscan(&x_i8, &mut out_naive);
    tiled.project_lut4_simd(&x_i8, &mut out_simd);

    let mut max_diff = 0i32;
    for r in 0..rows {
        max_diff = max_diff.max((out_ref[r] - out_naive[r]).abs());
        max_diff = max_diff.max((out_ref[r] - out_simd[r]).abs());
    }

    // Warmup
    for _ in 0..4 {
        naive.project_naive_bitscan(black_box(&x_i8), black_box(&mut out_naive));
        dense.project_f32(black_box(&x_f32), black_box(&mut out_f32));
        tiled.project_lut4_simd(black_box(&x_i8), black_box(&mut out_simd));
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        naive.project_naive_bitscan(black_box(&x_i8), black_box(&mut out_naive));
    }
    let us_naive = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

    let t1 = Instant::now();
    for _ in 0..iters {
        dense.project_f32(black_box(&x_f32), black_box(&mut out_f32));
    }
    let us_f32 = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

    let t2 = Instant::now();
    for _ in 0..iters {
        tiled.project_lut4_simd(black_box(&x_i8), black_box(&mut out_simd));
    }
    let us_simd = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);

    (max_diff, us_naive, us_f32, us_simd, tiled.packed_bytes(), dense.bytes())
}

fn bench_batched_1024(density: f32, batch: usize, iters: usize, rng: &mut Xorshift64) -> (i32, f64, f64, f64) {
    let rows = 1024;
    let cols = 1024;
    let naive = NaiveBitplaneMatrix::from_random_density(rows, cols, density, rng);
    let tiled = TernaryMatrix::from_naive(&naive);
    let dense = DenseF32Matrix::from_ternary(&tiled);

    let xs_i8: Vec<Vec<i8>> = (0..batch)
        .map(|_| (0..cols).map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8).collect())
        .collect();
    let xs_f32: Vec<Vec<f32>> = xs_i8
        .iter()
        .map(|v| v.iter().map(|&x| x as f32).collect())
        .collect();
    let x_refs: Vec<&[i8]> = xs_i8.iter().map(|v| v.as_slice()).collect();

    let mut outs_naive: Vec<Vec<i32>> = (0..batch).map(|_| vec![0i32; rows]).collect();
    let mut outs_tiled: Vec<Vec<i32>> = (0..batch).map(|_| vec![0i32; rows]).collect();
    let mut outs_f32: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; rows]).collect();

    // Parity check across entire batch
    for b in 0..batch {
        naive.project_naive_bitscan(&xs_i8[b], &mut outs_naive[b]);
    }
    {
        let mut out_refs: Vec<&mut [i32]> = outs_tiled.iter_mut().map(|v| v.as_mut_slice()).collect();
        tiled.project_batched_tiled(&x_refs, &mut out_refs);
    }
    let mut max_diff = 0i32;
    for b in 0..batch {
        for r in 0..rows {
            max_diff = max_diff.max((outs_naive[b][r] - outs_tiled[b][r]).abs());
        }
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        for b in 0..batch {
            naive.project_naive_bitscan(black_box(&xs_i8[b]), black_box(&mut outs_naive[b]));
        }
    }
    let us_naive = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

    let t1 = Instant::now();
    for _ in 0..iters {
        for b in 0..batch {
            dense.project_f32(black_box(&xs_f32[b]), black_box(&mut outs_f32[b]));
        }
    }
    let us_f32 = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

    let t2 = Instant::now();
    for _ in 0..iters {
        let mut out_refs: Vec<&mut [i32]> = outs_tiled.iter_mut().map(|v| v.as_mut_slice()).collect();
        tiled.project_batched_tiled(black_box(&x_refs), black_box(&mut out_refs));
    }
    let us_tiled = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);

    (max_diff, us_naive, us_f32, us_tiled)
}

pub fn run_b1(seed: u64) -> B1Report {
    let mut rng = Xorshift64::new(seed ^ 0xB100_0001);

    let (d1, u256_n, u256_f, u256_s, _, _) = bench_single(256, 256, 0.67, 250, &mut rng);
    let (d2, u1024_30_n, u1024_30_f, u1024_30_s, _, _) = bench_single(1024, 1024, 0.30, 40, &mut rng);
    let (d3, u1024_67_n, u1024_67_f, u1024_67_s, p_bytes, d_bytes) =
        bench_single(1024, 1024, 0.67, 40, &mut rng);
    let (d4, b16_n, b16_f, b16_s) = bench_batched_1024(0.67, 16, 10, &mut rng);

    let max_abs_diff = d1.max(d2).max(d3).max(d4);

    // 2. CorticalFabric 2,048 neurons: random scatter CSR vs minicolumn-tiled CSR
    let n_fab = 2048;
    let ticks = 240;
    let mut fab_scatter = CorticalFabric::build_cortical_sheet(
        n_fab,
        seed,
        FabricConfig {
            cache_tiled_csr: false,
            ..Default::default()
        },
    );
    let mut fab_tiled = CorticalFabric::build_cortical_sheet(
        n_fab,
        seed,
        FabricConfig {
            cache_tiled_csr: true,
            ..Default::default()
        },
    );
    let bytes: Vec<u8> = (0..ticks).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();

    let ts0 = Instant::now();
    let mut prev = 0u8;
    for &b in &bytes {
        fab_scatter.inject_raw_byte(prev, b);
        black_box(fab_scatter.step());
        prev = b;
    }
    let scatter_tps = (ticks as f64) / ts0.elapsed().as_secs_f64().max(1e-9);

    let ts1 = Instant::now();
    let mut prev2 = 0u8;
    for &b in &bytes {
        fab_tiled.inject_raw_byte(prev2, b);
        black_box(fab_tiled.step());
        prev2 = b;
    }
    let tiled_tps = (ticks as f64) / ts1.elapsed().as_secs_f64().max(1e-9);

    // 3. FastWeightMemory: naive alloc + unnormalized vs zero-alloc AVX2 NLMS
    let dim = 256;
    let mut fw_naive = FastWeightMemory::new(dim, 8);
    let mut fw_nlms = FastWeightMemory::new(dim, 8);
    let key_large: Vec<i8> = (0..dim)
        .map(|i| if (i + (seed as usize)) % 2 == 0 { 95 } else { -95 })
        .collect();
    let target: Vec<i32> = (0..dim).map(|i| (((i as i32) % 17) - 8) * 120).collect();

    let fw_iters = 80;
    let tf0 = Instant::now();
    for _ in 0..fw_iters {
        fw_naive.update_online_naive_proposal(black_box(&key_large), black_box(&target), 6);
    }
    let fw_naive_ups = (fw_iters as f64) / tf0.elapsed().as_secs_f64().max(1e-9);

    let tf1 = Instant::now();
    for _ in 0..fw_iters {
        fw_nlms.update_online_nlms(black_box(&key_large), black_box(&target), 1);
    }
    let fw_nlms_ups = (fw_iters as f64) / tf1.elapsed().as_secs_f64().max(1e-9);

    let mut pred_n = vec![0i32; dim];
    let mut pred_v = vec![0i32; dim];
    fw_naive.query(&key_large, &mut pred_n);
    fw_nlms.query(&key_large, &mut pred_v);
    let fw_naive_large_err: i64 = target.iter().zip(&pred_n).map(|(t, p)| (t - p).abs() as i64).sum();
    let fw_nlms_large_err: i64 = target.iter().zip(&pred_v).map(|(t, p)| (t - p).abs() as i64).sum();

    B1Report {
        seed,
        max_abs_diff,
        mem_ratio_1024: (d_bytes as f64) / (p_bytes as f64),
        packed_bytes_1024: p_bytes,
        dense_bytes_1024: d_bytes,
        us_256_d67_naive: u256_n,
        us_256_d67_f32: u256_f,
        us_256_d67_lut4_avx2: u256_s,
        us_1024_d30_naive: u1024_30_n,
        us_1024_d30_f32: u1024_30_f,
        us_1024_d30_lut4_avx2: u1024_30_s,
        us_1024_d67_naive: u1024_67_n,
        us_1024_d67_f32: u1024_67_f,
        us_1024_d67_lut4_avx2: u1024_67_s,
        us_1024_b16_naive: b16_n,
        us_1024_b16_f32: b16_f,
        us_1024_b16_lut4_tiled: b16_s,
        speedup_b16_vs_naive: b16_n / b16_s.max(1e-9),
        speedup_b16_vs_f32: b16_f / b16_s.max(1e-9),
        fabric_scatter_tps: scatter_tps,
        fabric_tiled_tps: tiled_tps,
        fabric_tiled_speedup: tiled_tps / scatter_tps.max(1e-9),
        fw_naive_ups,
        fw_nlms_ups,
        fw_nlms_speedup: fw_nlms_ups / fw_naive_ups.max(1e-9),
        fw_naive_large_err,
        fw_nlms_large_err,
    }
}
