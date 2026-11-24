//! Zero-dependency linear algebra: Householder QR, One-Sided Jacobi SVD, OLS & Ridge solvers,
//! exact 2-norm condition number, complex Hermitian Jacobi eigensolver (for GUE), and online RLS.

use crate::riemann::C64;

/// Row-major dense real matrix `m x n`.
#[derive(Clone, Debug)]
pub struct Mat {
    pub m: usize,
    pub n: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(m: usize, n: usize) -> Self {
        Self {
            m,
            n,
            data: vec![0.0; m * n],
        }
    }

    pub fn eye(n: usize) -> Self {
        let mut a = Self::zeros(n, n);
        for i in 0..n {
            a.set(i, i, 1.0);
        }
        a
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.n + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.n + c] = v;
    }

    #[inline]
    pub fn add_at(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.n + c] += v;
    }

    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(self.n, x.len());
        let mut out = vec![0.0; self.m];
        for r in 0..self.m {
            let row = &self.data[r * self.n..(r + 1) * self.n];
            let mut s = 0.0;
            for c in 0..self.n {
                s += row[c] * x[c];
            }
            out[r] = s;
        }
        out
    }
}

/// Thin SVD decomposition of `A` (`m >= n`) via Householder QR followed by One-Sided Jacobi SVD on `R`.
/// Stores `b_cols` (columns of `R * V = U_R * Sigma`), `v_cols` (columns of `V`), and `sigma`.
#[derive(Clone, Debug)]
pub struct ThinSvd {
    pub n: usize,
    /// Columns of `B = R * V` (`n` columns each of length `n`), mutually orthogonal with norm `sigma[k]`.
    pub b_cols: Vec<Vec<f64>>,
    /// Columns of orthogonal `V` (`n` columns each of length `n`).
    pub v_cols: Vec<Vec<f64>>,
    /// Singular values (unsorted or sorted descending).
    pub sigma: Vec<f64>,
    /// Stored Householder vectors and betas so we can apply `Q^T` to arbitrary vectors `y`.
    h_vecs: Vec<Vec<f64>>,
    h_betas: Vec<f64>,
    pub m: usize,
}

impl ThinSvd {
    pub fn compute(a: &Mat) -> Self {
        assert!(a.m >= a.n, "ThinSvd requires m >= n");
        let m = a.m;
        let n = a.n;
        let mut work = a.clone();
        let mut h_vecs = Vec::with_capacity(n);
        let mut h_betas = Vec::with_capacity(n);

        // Step 1: Householder QR -> upper-triangular R in top n x n of work
        for k in 0..n {
            let mut norm_sq = 0.0;
            for r in k..m {
                let v = work.get(r, k);
                norm_sq += v * v;
            }
            let norm = norm_sq.sqrt();
            if norm == 0.0 {
                h_vecs.push(vec![0.0; m - k]);
                h_betas.push(0.0);
                continue;
            }
            let x0 = work.get(k, k);
            let alpha = if x0 >= 0.0 { -norm } else { norm };
            let mut v = vec![0.0; m - k];
            v[0] = x0 - alpha;
            for r in (k + 1)..m {
                v[r - k] = work.get(r, k);
            }
            let v_norm_sq: f64 = v.iter().map(|&x| x * x).sum();
            let beta = if v_norm_sq > 0.0 { 2.0 / v_norm_sq } else { 0.0 };

            // Apply H = I - beta * v * v^T to columns k..n
            for c in k..n {
                let mut dot = 0.0;
                for i in 0..(m - k) {
                    dot += v[i] * work.get(k + i, c);
                }
                let scale = beta * dot;
                for i in 0..(m - k) {
                    work.add_at(k + i, c, -scale * v[i]);
                }
            }
            // Enforce exact triangular zeros below diagonal
            work.set(k, k, alpha);
            for r in (k + 1)..m {
                work.set(r, k, 0.0);
            }

            h_vecs.push(v);
            h_betas.push(beta);
        }

        // Extract columns of upper-triangular R (n x n)
        let mut b_cols: Vec<Vec<f64>> = (0..n)
            .map(|c| {
                let mut col = vec![0.0; n];
                for r in 0..=c {
                    col[r] = work.get(r, c);
                }
                col
            })
            .collect();

        // Initialize V = I_n in column-major form
        let mut v_cols: Vec<Vec<f64>> = (0..n)
            .map(|c| {
                let mut col = vec![0.0; n];
                col[c] = 1.0;
                col
            })
            .collect();

        // Step 2: One-Sided Jacobi SVD on R (columns of B orthogonalized via Givens rotations)
        let max_sweeps = 60;
        for _sweep in 0..max_sweeps {
            let mut max_rel = 0.0_f64;
            for p in 0..n {
                for q in (p + 1)..n {
                    let mut alpha = 0.0;
                    let mut beta_v = 0.0;
                    let mut gamma = 0.0;
                    for i in 0..n {
                        let bp = b_cols[p][i];
                        let bq = b_cols[q][i];
                        alpha += bp * bp;
                        beta_v += bq * bq;
                        gamma += bp * bq;
                    }
                    let denom = (alpha * beta_v).sqrt();
                    if denom <= 1e-30 {
                        continue;
                    }
                    let rel = gamma.abs() / denom;
                    if rel > max_rel {
                        max_rel = rel;
                    }
                    if rel <= 1e-14 {
                        continue;
                    }
                    let zeta = (beta_v - alpha) / (2.0 * gamma);
                    let t = if zeta >= 0.0 {
                        1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                    } else {
                        -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = c * t;

                    // b'_p = c*b_p - s*b_q, b'_q = s*b_p + c*b_q
                    for i in 0..n {
                        let bp = b_cols[p][i];
                        let bq = b_cols[q][i];
                        b_cols[p][i] = c * bp - s * bq;
                        b_cols[q][i] = s * bp + c * bq;

                        let vp = v_cols[p][i];
                        let vq = v_cols[q][i];
                        v_cols[p][i] = c * vp - s * vq;
                        v_cols[q][i] = s * vp + c * vq;
                    }
                }
            }
            if max_rel <= 1e-13 {
                break;
            }
        }

        let mut pairs: Vec<(f64, Vec<f64>, Vec<f64>)> = b_cols
            .into_iter()
            .zip(v_cols)
            .map(|(bc, vc)| {
                let sig = bc.iter().map(|&x| x * x).sum::<f64>().sqrt();
                (sig, bc, vc)
            })
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut sigma = Vec::with_capacity(n);
        let mut b_sorted = Vec::with_capacity(n);
        let mut v_sorted = Vec::with_capacity(n);
        for (sig, bc, vc) in pairs {
            sigma.push(sig);
            b_sorted.push(bc);
            v_sorted.push(vc);
        }

        Self {
            n,
            b_cols: b_sorted,
            v_cols: v_sorted,
            sigma,
            h_vecs,
            h_betas,
            m,
        }
    }

    /// Applies `Q^T y` and returns the top `n` coordinates `c = Q_1^T y`.
    pub fn apply_qt(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(y.len(), self.m);
        let mut w = y.to_vec();
        for k in 0..self.n {
            let beta = self.h_betas[k];
            if beta == 0.0 {
                continue;
            }
            let v = &self.h_vecs[k];
            let mut dot = 0.0;
            for i in 0..v.len() {
                dot += v[i] * w[k + i];
            }
            let scale = beta * dot;
            for i in 0..v.len() {
                w[k + i] -= scale * v[i];
            }
        }
        w.truncate(self.n);
        w
    }

    /// Exact 2-norm condition number `sigma_max / max(sigma_min, 1e-15)`.
    pub fn cond(&self) -> f64 {
        let smax = self.sigma[0];
        let smin = self.sigma[self.n - 1];
        smax / smin.max(1e-15)
    }

    /// Effective degrees of freedom `sum_k sigma_k^2 / (sigma_k^2 + lambda)`.
    pub fn effective_df(&self, lambda: f64) -> f64 {
        self.sigma
            .iter()
            .map(|&s| {
                let s2 = s * s;
                if s2 + lambda > 0.0 {
                    s2 / (s2 + lambda)
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Minimum-norm OLS solver (`rcond = 1e-12`, matching numpy `lstsq` truncation).
    pub fn solve_ols(&self, y: &[f64]) -> Vec<f64> {
        let c = self.apply_qt(y);
        let cutoff = 1e-12 * self.sigma[0].max(1e-30);
        let mut x = vec![0.0; self.n];
        for k in 0..self.n {
            let sig = self.sigma[k];
            if sig <= cutoff {
                continue;
            }
            let mut btc = 0.0;
            for i in 0..self.n {
                btc += self.b_cols[k][i] * c[i];
            }
            let coeff = btc / (sig * sig);
            for j in 0..self.n {
                x[j] += coeff * self.v_cols[k][j];
            }
        }
        x
    }

    /// Tikhonov / Ridge solver `(A^T A + lambda I)^{-1} A^T y`, stable even at zero singular values.
    pub fn solve_ridge(&self, y: &[f64], lambda: f64) -> Vec<f64> {
        let c = self.apply_qt(y);
        let mut x = vec![0.0; self.n];
        for k in 0..self.n {
            let s2 = self.sigma[k] * self.sigma[k];
            let denom = s2 + lambda;
            if denom <= 1e-30 {
                continue;
            }
            let mut btc = 0.0;
            for i in 0..self.n {
                btc += self.b_cols[k][i] * c[i];
            }
            let coeff = btc / denom;
            for j in 0..self.n {
                x[j] += coeff * self.v_cols[k][j];
            }
        }
        x
    }
}

/// Complex Hermitian Jacobi eigensolver returning sorted real eigenvalues (ascending)
/// and optional unitary eigenvectors (`n` columns).
pub fn hermitian_eigenvalues(h_in: &[C64], n: usize) -> (Vec<f64>, Vec<C64>) {
    assert_eq!(h_in.len(), n * n);
    let mut h = h_in.to_vec();
    let mut v = vec![C64::new(0.0, 0.0); n * n];
    for i in 0..n {
        v[i * n + i] = C64::new(1.0, 0.0);
    }

    let max_sweeps = 50;
    for _sweep in 0..max_sweeps {
        let mut off_max = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let hpq = h[p * n + q];
                let r = hpq.norm();
                if r > off_max {
                    off_max = r;
                }
                if r <= 1e-13 {
                    continue;
                }
                let e_iphi = hpq.scale(1.0 / r);
                let e_neg_iphi = e_iphi.conj();
                let app = h[p * n + p].re;
                let aqq = h[q * n + q].re;
                let tau = (aqq - app) / (2.0 * r);
                let t = if tau >= 0.0 {
                    -1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                // U = [[c, -s*e^{i*phi}], [s*e^{-i*phi}, c]]
                // Update rows/cols k != p, q: H' = U^H H U
                let s_pos = e_iphi.scale(s); // s * e^{i*phi}
                let s_neg = e_neg_iphi.scale(s); // s * e^{-i*phi}

                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let hkp = h[k * n + p];
                    let hkq = h[k * n + q];
                    // Column update: H'_kp = c*hkp + s*e^{-i*phi}*hkq; H'_kq = -s*e^{i*phi}*hkp + c*hkq
                    let nkp = hkp.scale(c).add(hkq.mul(s_neg));
                    let nkq = hkq.scale(c).sub(hkp.mul(s_pos));
                    h[k * n + p] = nkp;
                    h[p * n + k] = nkp.conj();
                    h[k * n + q] = nkq;
                    h[q * n + k] = nkq.conj();
                }

                // Diagonal update (exact, real)
                let new_pp = c * c * app + 2.0 * c * s * r + s * s * aqq;
                let new_qq = s * s * app - 2.0 * c * s * r + c * c * aqq;
                h[p * n + p] = C64::new(new_pp, 0.0);
                h[q * n + q] = C64::new(new_qq, 0.0);
                h[p * n + q] = C64::new(0.0, 0.0);
                h[q * n + p] = C64::new(0.0, 0.0);

                // Accumulate eigenvectors V <- V * U
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = vkp.scale(c).add(vkq.mul(s_neg));
                    v[k * n + q] = vkq.scale(c).sub(vkp.mul(s_pos));
                }
            }
        }
        if off_max <= 1e-12 {
            break;
        }
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        h[i * n + i]
            .re
            .partial_cmp(&h[j * n + j].re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let evals: Vec<f64> = idx.iter().map(|&i| h[i * n + i].re).collect();
    let mut v_sorted = vec![C64::new(0.0, 0.0); n * n];
    for (new_c, &old_c) in idx.iter().enumerate() {
        for r in 0..n {
            v_sorted[r * n + new_c] = v[r * n + old_c];
        }
    }
    (evals, v_sorted)
}

/// Online Recursive Least Squares (RLS) state with Sherman-Morrison rank-1 covariance update.
#[derive(Clone, Debug)]
pub struct RlsState {
    pub p: usize,
    pub w: Vec<f64>,
    pub cov: Mat,
}

impl RlsState {
    pub fn new(p: usize, init_alpha: f64) -> Self {
        let mut cov = Mat::zeros(p, p);
        for i in 0..p {
            cov.set(i, i, init_alpha);
        }
        Self {
            p,
            w: vec![0.0; p],
            cov,
        }
    }

    /// Single-observation RLS update `(phi, y)` with forgetting factor `lambda = 1.0`.
    pub fn update(&mut self, phi: &[f64], y: f64) {
        assert_eq!(phi.len(), self.p);
        let p_phi = self.cov.mul_vec(phi);
        let mut denom = 1.0;
        let mut pred = 0.0;
        for i in 0..self.p {
            denom += phi[i] * p_phi[i];
            pred += phi[i] * self.w[i];
        }
        let err = y - pred;
        let inv_d = 1.0 / denom.max(1e-15);
        for i in 0..self.p {
            self.w[i] += p_phi[i] * inv_d * err;
        }
        // Symmetrized rank-1 covariance downdate
        for r in 0..self.p {
            let kr = p_phi[r] * inv_d;
            let row_off = r * self.p;
            for c in 0..self.p {
                self.cov.data[row_off + c] -= kr * p_phi[c];
            }
        }
    }
}

pub fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

pub fn rmse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        / a.len().max(1) as f64;
    mse.sqrt()
}
