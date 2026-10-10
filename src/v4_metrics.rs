//! v4 metrics (protocolo §2 `v4_metrics`).

#![allow(clippy::needless_range_loop)]

use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

/// Pre-registered correctness threshold: relative L2 error < 5 %.
pub const REL_ERR_OK: f64 = 0.05;

pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn rel_err(pred: &[f64], tgt: &[f64]) -> f64 {
    let d: f64 = pred
        .iter()
        .zip(tgt)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let r = d / norm(tgt).max(1e-9);
    if r.is_finite() {
        r
    } else {
        1e9
    }
}

pub fn correct(pred: &[f64], tgt: &[f64]) -> bool {
    rel_err(pred, tgt) < REL_ERR_OK
}

pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let n = norm(a) * norm(b);
    if n < 1e-12 {
        0.0
    } else {
        d / n
    }
}

/// Cosine between predicted and true displacement (static = undefined → 0).
pub fn disp_cosine(x: &[f64], pred: &[f64], tgt: &[f64]) -> f64 {
    let dp: Vec<f64> = pred.iter().zip(x).map(|(a, b)| a - b).collect();
    let dt: Vec<f64> = tgt.iter().zip(x).map(|(a, b)| a - b).collect();
    cosine(&dp, &dt)
}

/// Field "energy" = ½‖z‖² (reported relative to the target's energy).
pub fn energy(z: &[f64]) -> f64 {
    0.5 * z.iter().map(|x| x * x).sum::<f64>()
}

/// Distance of point-set `z` to the similarity-orbit of `x0`
/// (best rotation+uniform scale+translation, Procrustes), normalised by ‖z‖.
/// Rigid/similarity rules keep the true trajectory at 0.
pub fn manifold_distance(x0: &[f64], z: &[f64]) -> f64 {
    let k = x0.len() / 2;
    let (mut ax, mut ay, mut bx, mut by) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..k {
        ax += x0[2 * i];
        ay += x0[2 * i + 1];
        bx += z[2 * i];
        by += z[2 * i + 1];
    }
    let kf = k as f64;
    let (ax, ay, bx, by) = (ax / kf, ay / kf, bx / kf, by / kf);
    // complex least squares: b ≈ s·a with s complex
    let (mut num_re, mut num_im, mut den) = (0.0, 0.0, 0.0);
    for i in 0..k {
        let (pr, pi) = (x0[2 * i] - ax, x0[2 * i + 1] - ay);
        let (qr, qi) = (z[2 * i] - bx, z[2 * i + 1] - by);
        num_re += qr * pr + qi * pi;
        num_im += qi * pr - qr * pi;
        den += pr * pr + pi * pi;
    }
    let (sr, si) = (num_re / den.max(1e-12), num_im / den.max(1e-12));
    let mut res = 0.0;
    let mut zn = 0.0;
    for i in 0..k {
        let (pr, pi) = (x0[2 * i] - ax, x0[2 * i + 1] - ay);
        let (qr, qi) = (z[2 * i] - bx, z[2 * i + 1] - by);
        let (er, ei) = (qr - (sr * pr - si * pi), qi - (sr * pi + si * pr));
        res += er * er + ei * ei;
        zn += qr * qr + qi * qi;
    }
    (res / zn.max(1e-12)).sqrt()
}

/// Jacobian of `f` at `z` by central finite differences (h = 1e-5).
pub fn jacobian<F: Fn(&[f64]) -> Vec<f64>>(f: F, z: &[f64]) -> Vec<Vec<f64>> {
    let n = z.len();
    let h = 1e-5;
    let mut cols = Vec::with_capacity(n);
    for j in 0..n {
        let mut zp = z.to_vec();
        let mut zm = z.to_vec();
        zp[j] += h;
        zm[j] -= h;
        let (fp, fm) = (f(&zp), f(&zm));
        cols.push(
            fp.iter()
                .zip(&fm)
                .map(|(a, b)| (a - b) / (2.0 * h))
                .collect::<Vec<f64>>(),
        );
    }
    // J[i][j] = d f_i / d z_j
    let m = cols[0].len();
    (0..m)
        .map(|i| (0..n).map(|j| cols[j][i]).collect())
        .collect()
}

/// σ_max(J) by power iteration on JᵀJ (60 iterations, deterministic start).
pub fn sigma_max(j: &[Vec<f64>]) -> f64 {
    let n = j[0].len();
    let mut v: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
    let mut s = 0.0;
    for _ in 0..60 {
        let jv: Vec<f64> = j
            .iter()
            .map(|r| r.iter().zip(&v).map(|(a, b)| a * b).sum())
            .collect();
        let mut jtjv = vec![0.0; n];
        for (i, r) in j.iter().enumerate() {
            for k in 0..n {
                jtjv[k] += r[k] * jv[i];
            }
        }
        let nn = norm(&jtjv);
        if nn < 1e-15 {
            return 0.0;
        }
        s = nn.sqrt();
        v = jtjv.iter().map(|x| x / nn).collect();
    }
    s
}

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

pub fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

pub fn std(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
}

/// Paired effect size d_z = mean(diff)/sd(diff).
pub fn cohen_dz(diff: &[f64]) -> f64 {
    let s = std(diff);
    if s < 1e-12 {
        if mean(diff).abs() < 1e-12 {
            0.0
        } else {
            mean(diff).signum() * 99.0
        }
    } else {
        mean(diff) / s
    }
}

/// Percentile bootstrap 95 % CI of the mean (2000 resamples, fixed seed).
pub fn bootstrap_ci(x: &[f64]) -> (f64, f64) {
    if x.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = Xoshiro256StarStar::seed_from_u64(0xB007);
    let mut ms: Vec<f64> = (0..2000)
        .map(|_| {
            mean(
                &(0..x.len())
                    .map(|_| x[rng.gen_range(0..x.len())])
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    ms.sort_by(|a, b| a.total_cmp(b));
    (ms[50], ms[1949])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_sigma_of_rotation_is_one() {
        let th: f64 = 0.4;
        let f = |z: &[f64]| {
            vec![
                th.cos() * z[0] - th.sin() * z[1],
                th.sin() * z[0] + th.cos() * z[1],
            ]
        };
        let s = sigma_max(&jacobian(f, &[0.3, 0.2]));
        assert!((s - 1.0).abs() < 1e-6);
        let x0 = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let z: Vec<f64> = x0.chunks(2).flat_map(f).collect();
        assert!(manifold_distance(&x0, &z) < 1e-9);
    }
}
