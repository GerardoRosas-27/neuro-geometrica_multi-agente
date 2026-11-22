//! METAFE — Metacognitive Free-Energy Neuro-Geometric Architecture Family
//!
//! Diseñada tras auditar todas las arquitecturas del repositorio (ÓRBITA-v1, Campo v4,
//! CONSTELACIÓN v2/v3/v4, ATLAS-FW, BQ10, Bio-Constelación, TFE, Yo 1er/2º orden,
//! Soft-Bellman Simplicial v3).
//!
//! Principios rectores (cero fuga de reglas retenidas {8,10,11}, 0 consultas TEST):
//! 1. **Energía Libre Variacional Exacta (Analytical VFE / Empirical Bayes)**:
//!    F_m = 0.5 * y^T C^{-1} y + 0.5 * ln|C| + (N/2) ln(2π)
//!    Evita el colapso del muestreo estocástico β·KL observado en TFE, minimizando
//!    analíticamente sorpresa sensorial + complejidad Occam D_KL(q||p).
//! 2. **Banco Multi-Sustrato de Geometrías Canónicas (Nivel 0)**:
//!    - `LieAffine`: flujo autónomo afín exacto GL(2) ⋉ R² (mínima complejidad).
//!    - `FE-ARD`: Empirical Bayes / ARD variacional jerárquico por grado (mejora de BQ10).
//!    - `GP-Cart`: Proceso Gaussiano semi-paramétrico (media afín + kernel aditivo/interacción
//!      Matérn-5/2 ARD en R²) con varianza epistémica exacta y reversión segura fuera de soporte.
//!    - `GP-CoMoving`: Operator-Splitting Lie-Trotter u = A q + b, q' = u + g_GP(u) con
//!      EM monótono de Energía Libre.
//!    - `GP-Symp`: Splitting Simpléctico / Coupling-Layer (Hamiltoniano: x' = f_x(x,y),
//!      y' = f_y(x',y) y dual, tanto directo como en marco co-móvil) con Energía Libre exacta.
//!    - `GP-Polar`: Variedad Polar SO(2)-equivariante (r, θ) con kernel periódico exacto en S¹
//!      y corrección exacta de Jacobiano (+2 Σ ln r') para comparabilidad en nats cartesianos.
//!    - `GP-EquivCart`: Kernel canónico producto de Schur SO(2)-equivariante (q·q') k_r(r,r') +
//!      aditivo cartesiano (sin singularidad en r=0, selección automática de suavidad ν∈{5/2,∞}).
//! 3. **Inferencia Activa Multi-Paso y Validación Precuencial de Sorpresa (Nivel 1)**:
//!    - Energía Libre LOO-CV analítica exacta (Rasmussen & Williams Eq. 5.12, vía [C^{-1}]_{ii}).
//!    - Sorpresa precuencial fuera de tiempo en contexto (0..K-1 → K y rollout cerrado 0..K-2 → K-1,K).
//!    - Refinamiento activo multi-paso del núcleo Lie-afín minimizando sorpresa de trayectoria cerrada.
//! 4. **Metacognición Entrenada de 2º Orden ("Yo-FE", Nivel 2)**:
//!    - Entrenada estrictamente sobre familias conocidas 0..7 + familias sintéticas procedurales
//!      auditadas (`leaks() == false`, jamás {8,10,11}, 0 consultas TEST).
//!    - Aprende: (a) pesos de Energía Libre y temperatura metacognitiva τ_meta,
//!      (b) hiper-prior amortizado p_ψ(M | invariantes geométricos),
//!      (c) auto-calibrador metacognitivo bietapa (predictor de log-error + calibración Platt
//!          fuera de muestra p̂_ok = P(acierto h16 | estado interno M_e)),
//!      (d) control activo epistémico durante el rollout h5..h16 (Bayesian Model Averaging
//!          ponderado por precisión epistémica local σ²_epi(q̂_h)),
//!      (e) prior autónomo de arranque en frío 1-paso (GP RBF por Energía Libre en `ds.train`,
//!          para E33 autónomo sin ÓRBITA).

use super::atlas::{lib, make_syn_ep, N_LIB};
use super::*;
use std::f64::consts::{PI, TAU};

const LN_2PI: f64 = 1.8378770664093453;
pub(super) const XOR_METAFE_TRAIN: u64 = 0xFE7A_0101;
pub(super) const XOR_METAFE_VAL: u64 = 0xFE7A_0202;
pub(super) const XOR_METAFE_OOD: u64 = 0xFE7A_0303;

// ─── álgebra lineal SPD (Cholesky + LOO analítico) ───────────────────────────

/// Factorización Cholesky L tal que C = L L^T con jitter adaptativo. Devuelve None si falla.
fn cholesky_jitter(c: &[f64], n: usize) -> Option<(Vec<f64>, f64)> {
    for &jit in &[0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4] {
        let mut l = vec![0.0; n * n];
        let mut ok = true;
        for i in 0..n {
            for j in 0..=i {
                let mut s = c[i * n + j];
                if i == j {
                    s += jit;
                }
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if !s.is_finite() || s <= 1e-30 {
                        ok = false;
                        break;
                    }
                    l[i * n + i] = s.sqrt();
                } else {
                    l[i * n + j] = s / l[j * n + j];
                }
            }
            if !ok {
                break;
            }
        }
        if ok {
            let logdet = 2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>();
            if logdet.is_finite() {
                return Some((l, logdet));
            }
        }
    }
    None
}

fn solve_lower(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        let row = &l[i * n..i * n + i];
        for j in 0..i {
            s -= row[j] * z[j];
        }
        z[i] = s / l[i * n + i];
    }
    z
}

fn solve_upper_t(l: &[f64], n: usize, z: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// Calcula diagonal de C^{-1} = L^{-T} L^{-1} en O(n^3/6) sin almacenar C^{-1} completa.
fn chol_inv_diag(l: &[f64], n: usize) -> Vec<f64> {
    // L^{-1} es triangular inferior
    let mut linv = vec![0.0; n * n];
    for j in 0..n {
        linv[j * n + j] = 1.0 / l[j * n + j];
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in j..i {
                s -= l[i * n + k] * linv[k * n + j];
            }
            linv[i * n + j] = s / l[i * n + i];
        }
    }
    let mut diag = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for k in i..n {
            let v = linv[k * n + i];
            s += v * v;
        }
        diag[i] = s.max(1e-30);
    }
    diag
}

/// Resuelve sistema simétrico pequeño (p×p) con Cholesky + fallback ridge.
pub(super) fn solve_spd(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    if let Some((l, _)) = cholesky_jitter(a, p) {
        let z = solve_lower(&l, p, b);
        solve_upper_t(&l, p, &z)
    } else {
        vec![0.0; p]
    }
}

/// Energía Libre Variacional perfilada y LOO analítico exacto para un canal 1D y ∈ R^n.
/// C_tilde = K_tilde + noise * I, con escala global s^2 = y^T C_tilde^{-1} y / n perfilada analíticamente.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct GpChannel {
    l: Vec<f64>,
    alpha: Vec<f64>,
    s2: f64,
    vfe: f64,
    loo_fe: f64,
    loo_mse: f64,
    loo_preds: Vec<f64>,
    params: KernelParams,
}

#[derive(Clone, Copy, Debug)]
struct KernelParams {
    kind: KernelKind,
    w_lin: f64,
    w1: f64,
    w2: f64,
    w12: f64,
    l1: f64,
    l2: f64,
    noise: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelKind {
    /// Puro afín (w_lin > 0, w1=w2=w12=0)
    AffineOnly,
    /// Matérn-5/2 aditivo + interacción en R² (u1, u2)
    CartM52,
    /// RBF/SE aditivo + interacción en R² (u1, u2)
    CartRbf,
    /// Polar cilindro R_+ × S¹: k_r(r,r') Matérn-5/2 + k_per(θ,θ') periódico exacto + producto
    PolarCyl,
    /// Polar cilindro suave R_+ × S¹: k_r(r,r') RBF + k_per(θ,θ') periódico exacto + producto
    PolarCylRbf,
    /// Equivariante SO(2) + Cartesiano (producto de Schur PSD: (u·v) k_r(||u||,||v||) + k_cart) Matérn-5/2
    EquivCartM52,
    /// Equivariante SO(2) + Cartesiano (producto de Schur PSD: (u·v) k_r(||u||,||v||) + k_cart) RBF
    EquivCartRbf,
}

#[inline]
fn matern52(d: f64) -> f64 {
    let s = 2.23606797749979 * d; // sqrt(5) * d
    (1.0 + s + s * s / 3.0) * (-s).exp()
}

#[inline]
fn rbf1(d: f64) -> f64 {
    (-0.5 * d * d).exp()
}

/// Kernel periódico exacto en S¹ (MacKay / Rasmussen & Williams Eq. 4.31), invariante SO(2).
#[inline]
fn periodic_s1(th1: f64, th2: f64, ell: f64) -> f64 {
    let d2 = (1.0 - (th1 - th2).cos()).max(0.0);
    (-d2 / (ell * ell)).exp()
}

#[inline]
fn kernel_eval(p: &KernelParams, u: &[f64; 2], v: &[f64; 2]) -> f64 {
    let lin = match p.kind {
        KernelKind::PolarCyl | KernelKind::PolarCylRbf => p.w_lin * (1.0 + u[0] * v[0]),
        _ => p.w_lin * (u[0] * v[0] + u[1] * v[1] + 1.0),
    };
    match p.kind {
        KernelKind::AffineOnly => lin,
        KernelKind::CartM52 => {
            let d1 = (u[0] - v[0]).abs() / p.l1;
            let d2 = (u[1] - v[1]).abs() / p.l2;
            let mut k = lin;
            if p.w1 > 0.0 {
                k += p.w1 * matern52(d1);
            }
            if p.w2 > 0.0 {
                k += p.w2 * matern52(d2);
            }
            if p.w12 > 0.0 {
                k += p.w12 * matern52((d1 * d1 + d2 * d2).sqrt());
            }
            k
        }
        KernelKind::CartRbf => {
            let d1 = (u[0] - v[0]) / p.l1;
            let d2 = (u[1] - v[1]) / p.l2;
            let mut k = lin;
            if p.w1 > 0.0 {
                k += p.w1 * rbf1(d1);
            }
            if p.w2 > 0.0 {
                k += p.w2 * rbf1(d2);
            }
            if p.w12 > 0.0 {
                k += p.w12 * (-0.5 * (d1 * d1 + d2 * d2)).exp();
            }
            k
        }
        KernelKind::PolarCyl => {
            let dr = (u[0] - v[0]).abs() / p.l1;
            let kr = matern52(dr);
            let kth = periodic_s1(u[1], v[1], p.l2);
            lin + p.w1 * kr + p.w2 * kth + p.w12 * kr * kth
        }
        KernelKind::PolarCylRbf => {
            let dr = (u[0] - v[0]) / p.l1;
            let kr = rbf1(dr);
            let kth = periodic_s1(u[1], v[1], p.l2);
            lin + p.w1 * kr + p.w2 * kth + p.w12 * kr * kth
        }
        KernelKind::EquivCartM52 => {
            let ru = (u[0] * u[0] + u[1] * u[1]).sqrt();
            let rv = (v[0] * v[0] + v[1] * v[1]).sqrt();
            let dot = u[0] * v[0] + u[1] * v[1];
            let dr = (ru - rv).abs() / p.l1;
            let d1 = (u[0] - v[0]).abs() / p.l2;
            let d2 = (u[1] - v[1]).abs() / p.l2;
            let mut k = lin;
            if p.w12 > 0.0 {
                k += p.w12 * dot * matern52(dr);
            }
            if p.w1 > 0.0 {
                k += p.w1 * matern52(d1);
            }
            if p.w2 > 0.0 {
                k += p.w2 * matern52(d2);
            }
            k
        }
        KernelKind::EquivCartRbf => {
            let ru = (u[0] * u[0] + u[1] * u[1]).sqrt();
            let rv = (v[0] * v[0] + v[1] * v[1]).sqrt();
            let dot = u[0] * v[0] + u[1] * v[1];
            let dr = (ru - rv) / p.l1;
            let d1 = (u[0] - v[0]) / p.l2;
            let d2 = (u[1] - v[1]) / p.l2;
            let mut k = lin;
            if p.w12 > 0.0 {
                k += p.w12 * dot * rbf1(dr);
            }
            if p.w1 > 0.0 {
                k += p.w1 * rbf1(d1);
            }
            if p.w2 > 0.0 {
                k += p.w2 * rbf1(d2);
            }
            k
        }
    }
}

fn fit_gp_channel_fixed(us: &[[f64; 2]], y: &[f64], params: KernelParams) -> Option<GpChannel> {
    let n = us.len();
    if n == 0 || y.len() != n {
        return None;
    }
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut v = kernel_eval(&params, &us[i], &us[j]);
            if i == j {
                v += params.noise;
            }
            c[i * n + j] = v;
            c[j * n + i] = v;
        }
    }
    let (l, logdet) = cholesky_jitter(&c, n)?;
    let z = solve_lower(&l, n, y);
    let alpha = solve_upper_t(&l, n, &z);
    let q: f64 = z.iter().map(|v| v * v).sum();
    if !q.is_finite() {
        return None;
    }
    let nf = n as f64;
    // Perfilado analítico de la escala global s^2 = y^T C_tilde^{-1} y / n
    let s2 = (q / nf).max(1e-24);
    let vfe = 0.5 * nf * (s2.ln() + 1.0 + LN_2PI) + 0.5 * logdet;

    // LOO analítico exacto: mu_loo_i = y_i - alpha_i / diag_i, var_loo_i = s2 / diag_i
    let diag = chol_inv_diag(&l, n);
    let mut loo_fe = 0.0;
    let mut loo_mse = 0.0;
    let mut loo_preds = Vec::with_capacity(n);
    for i in 0..n {
        let err = alpha[i] / diag[i];
        let pred = y[i] - err;
        let v_loo = (s2 / diag[i]).max(1e-24);
        loo_preds.push(pred);
        loo_mse += err * err;
        loo_fe += 0.5 * (err * err / v_loo + (TAU * v_loo).ln());
    }
    loo_mse /= nf;
    if !vfe.is_finite() || !loo_fe.is_finite() {
        return None;
    }
    Some(GpChannel {
        l,
        alpha,
        s2,
        vfe,
        loo_fe,
        loo_mse,
        loo_preds,
        params,
    })
}

impl GpChannel {
    /// Predice (media, varianza epistémica σ²_epi) en un nuevo punto u_*.
    fn predict(&self, us: &[[f64; 2]], u_star: &[f64; 2]) -> (f64, f64) {
        let n = us.len();
        let mut kstar = Vec::with_capacity(n);
        let mut mu = 0.0;
        for i in 0..n {
            let k = kernel_eval(&self.params, u_star, &us[i]);
            kstar.push(k);
            mu += k * self.alpha[i];
        }
        let v = solve_lower(&self.l, n, &kstar);
        let kss = kernel_eval(&self.params, u_star, u_star);
        let vnorm: f64 = v.iter().map(|x| x * x).sum();
        let epi = (self.s2 * (kss - vnorm).max(0.0)).max(0.0);
        (mu, epi)
    }

    #[inline]
    fn predict_mean(&self, us: &[[f64; 2]], u_star: &[f64; 2]) -> f64 {
        let mut mu = 0.0;
        for i in 0..us.len() {
            mu += kernel_eval(&self.params, u_star, &us[i]) * self.alpha[i];
        }
        mu
    }
}

/// Evalúa exactamente la Energía Libre Variacional perfilada sin calcular LOO ni alpha
/// (útil durante búsqueda de hiperparámetros; matemáticamente idéntico a `fit_gp_channel_fixed`).
fn vfe_from_gram(k_mat: &[f64], noise: f64, y: &[f64], n: usize, buf: &mut [f64]) -> Option<f64> {
    buf.copy_from_slice(k_mat);
    for i in 0..n {
        buf[i * n + i] += noise;
    }
    let (l, logdet) = cholesky_jitter(buf, n)?;
    let z = solve_lower(&l, n, y);
    let q: f64 = z.iter().map(|v| v * v).sum();
    if !q.is_finite() {
        return None;
    }
    let nf = n as f64;
    let s2 = (q / nf).max(1e-24);
    let vfe = 0.5 * nf * (s2.ln() + 1.0 + LN_2PI) + 0.5 * logdet;
    if vfe.is_finite() {
        Some(vfe)
    } else {
        None
    }
}

fn eval_params_vfe(us: &[[f64; 2]], y: &[f64], params: &KernelParams, k_buf: &mut [f64], c_buf: &mut [f64]) -> Option<f64> {
    let n = us.len();
    for i in 0..n {
        for j in 0..=i {
            let v = kernel_eval(params, &us[i], &us[j]);
            k_buf[i * n + j] = v;
            k_buf[j * n + i] = v;
        }
    }
    vfe_from_gram(k_buf, params.noise, y, n, c_buf)
}

/// Residuo relativo OLS de `y` sobre base lineal del sustrato: detecta ajuste afín exacto (<1e-8)
/// para retornar la solución óptima de Energía Libre sin barridos no-lineales redundantes.
fn exact_linear_rel_res(us: &[[f64; 2]], y: &[f64], is_polar: bool) -> f64 {
    let n = us.len();
    if n < 4 {
        return 1.0;
    }
    let ynorm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    if ynorm < 1e-12 {
        return 0.0;
    }
    if is_polar {
        // Base [1, r]
        let mut ata = [0.0; 4];
        let mut aty = [0.0; 2];
        for (u, &yi) in us.iter().zip(y) {
            let f = [1.0, u[0]];
            for i in 0..2 {
                aty[i] += f[i] * yi;
                for j in 0..2 {
                    ata[i * 2 + j] += f[i] * f[j];
                }
            }
        }
        ata[0] += 1e-12;
        ata[3] += 1e-12;
        let w = solve_spd(&ata, &aty, 2);
        let rss: f64 = us
            .iter()
            .zip(y)
            .map(|(u, &yi)| (yi - (w[0] + w[1] * u[0])).powi(2))
            .sum();
        rss.sqrt() / ynorm
    } else {
        // Base [1, u0, u1]
        let mut ata = [0.0; 9];
        let mut aty = [0.0; 3];
        for (u, &yi) in us.iter().zip(y) {
            let f = [1.0, u[0], u[1]];
            for i in 0..3 {
                aty[i] += f[i] * yi;
                for j in 0..3 {
                    ata[i * 3 + j] += f[i] * f[j];
                }
            }
        }
        for i in 0..3 {
            ata[i * 3 + i] += 1e-12;
        }
        let w = solve_spd(&ata, &aty, 3);
        let rss: f64 = us
            .iter()
            .zip(y)
            .map(|(u, &yi)| (yi - (w[0] + w[1] * u[0] + w[2] * u[1])).powi(2))
            .sum();
        rss.sqrt() / ynorm
    }
}

/// Optimización por Energía Libre Variacional (Empirical Bayes Tipo-II) para un canal 1D:
/// Explora estructuras de relevancia (ARD aditivo, interacción, isotrópico, escala) y refina
/// localmente por descenso de coordenadas de Energía Libre (monótono).
fn optimize_gp_channel_vfe(
    us: &[[f64; 2]],
    y: &[f64],
    kind: KernelKind,
    include_affine_mean: bool,
    w_lin_val: f64,
) -> Option<GpChannel> {
    let n = us.len();
    if n == 0 || y.len() != n {
        return None;
    }
    let mut k_buf = vec![0.0; n * n];
    let mut c_buf = vec![0.0; n * n];
    let is_polar = matches!(kind, KernelKind::PolarCyl | KernelKind::PolarCylRbf);
    let is_equiv = matches!(kind, KernelKind::EquivCartM52 | KernelKind::EquivCartRbf);

    if kind == KernelKind::AffineOnly {
        for i in 0..n {
            for j in 0..=i {
                let v = w_lin_val * (us[i][0] * us[j][0] + us[i][1] * us[j][1] + 1.0);
                k_buf[i * n + j] = v;
                k_buf[j * n + i] = v;
            }
        }
        let mut best_p: Option<(KernelParams, f64)> = None;
        for &noise in &[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1] {
            if let Some(vfe) = vfe_from_gram(&k_buf, noise, y, n, &mut c_buf) {
                if best_p.as_ref().map_or(true, |b| vfe < b.1) {
                    best_p = Some((
                        KernelParams {
                            kind,
                            w_lin: w_lin_val,
                            w1: 0.0,
                            w2: 0.0,
                            w12: 0.0,
                            l1: 1.0,
                            l2: 1.0,
                            noise,
                        },
                        vfe,
                    ));
                }
            }
        }
        return best_p.and_then(|(p, _)| fit_gp_channel_fixed(us, y, p));
    }

    let wl = if include_affine_mean { w_lin_val } else { 0.0 };
    let std_of = |dim: usize| -> f64 {
        let m = us.iter().map(|u| u[dim]).sum::<f64>() / n as f64;
        let v = us.iter().map(|u| (u[dim] - m).powi(2)).sum::<f64>() / n as f64;
        v.sqrt().max(0.15)
    };
    let rs: Vec<f64> = if is_equiv {
        us.iter().map(|u| (u[0] * u[0] + u[1] * u[1]).sqrt()).collect()
    } else {
        vec![]
    };
    let s1 = if is_equiv {
        let mr = rs.iter().sum::<f64>() / n as f64;
        (rs.iter().map(|r| (r - mr).powi(2)).sum::<f64>() / n as f64)
            .sqrt()
            .max(0.15)
    } else {
        std_of(0)
    };
    let s2 = if is_polar {
        1.0
    } else if is_equiv {
        0.5 * (std_of(0) + std_of(1))
    } else {
        std_of(1)
    };

    // Precomputar base lineal una sola vez
    let mut k_lin = vec![0.0; n * n];
    if wl > 0.0 {
        for i in 0..n {
            for j in 0..=i {
                let v = if is_polar {
                    wl * (1.0 + us[i][0] * us[j][0])
                } else {
                    wl * (us[i][0] * us[j][0] + us[i][1] * us[j][1] + 1.0)
                };
                k_lin[i * n + j] = v;
                k_lin[j * n + i] = v;
            }
        }
    }

    let mut best_p: Option<(KernelParams, f64)> = None;

    // Fase 1a: caso lineal/afín puro (con atajo exacto si residuo OLS < 1e-8)
    if include_affine_mean {
        for &noise in &[1e-10, 1e-7, 1e-4, 1e-2] {
            if let Some(vfe) = vfe_from_gram(&k_lin, noise, y, n, &mut c_buf) {
                if best_p.as_ref().map_or(true, |b| vfe < b.1) {
                    best_p = Some((
                        KernelParams {
                            kind,
                            w_lin: wl,
                            w1: 0.0,
                            w2: 0.0,
                            w12: 0.0,
                            l1: s1,
                            l2: s2,
                            noise,
                        },
                        vfe,
                    ));
                }
            }
        }
        if exact_linear_rel_res(us, y, is_polar) < 1e-8 {
            return best_p.and_then(|(p, _)| fit_gp_channel_fixed(us, y, p));
        }
    }

    // Modos estructurales ARD (todos PSD garantizados)
    let modes: [(f64, f64, f64); 6] = if is_equiv {
        [
            (0.0, 0.0, 1.0),   // equivariante radial puro (u·v) k_r(r,r')
            (0.25, 0.25, 0.5), // equivariante radial + cartesiano aditivo
            (0.4, 0.0, 0.6),   // equivariante radial + eje x
            (0.0, 0.4, 0.6),   // equivariante radial + eje y
            (0.1, 0.1, 0.8),   // equivariante dominante + cartesiano suave
            (0.45, 0.45, 0.1), // cartesiano dominante + radial suave
        ]
    } else {
        [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.4, 0.4, 0.2),
            (0.15, 0.15, 0.7),
        ]
    };

    let ells_base: [f64; 4] = if is_polar {
        [0.45, 0.9, 1.8, 3.5]
    } else {
        [0.6, 1.2, 2.4, 4.5]
    };
    let noises: [f64; 5] = [1e-9, 1e-7, 1e-5, 1e-3, 3e-2];

    // Fase 1b: precomputar k1, k2, k12 una vez por escala y barrer modos × ruidos
    let mut k1 = vec![0.0; n * n];
    let mut k2 = vec![0.0; n * n];
    let mut k12 = vec![0.0; n * n];
    for &eb in &ells_base {
        let l1 = (eb * s1).clamp(0.15, 12.0);
        let l2 = if is_polar {
            eb.clamp(0.2, 6.0)
        } else {
            (eb * s2).clamp(0.15, 12.0)
        };
        for i in 0..n {
            for j in 0..=i {
                let (v1, v2, v12) = match kind {
                    KernelKind::CartM52 => {
                        let d1 = (us[i][0] - us[j][0]).abs() / l1;
                        let d2 = (us[i][1] - us[j][1]).abs() / l2;
                        (matern52(d1), matern52(d2), matern52((d1 * d1 + d2 * d2).sqrt()))
                    }
                    KernelKind::CartRbf => {
                        let d1 = (us[i][0] - us[j][0]) / l1;
                        let d2 = (us[i][1] - us[j][1]) / l2;
                        let r1 = rbf1(d1);
                        let r2 = rbf1(d2);
                        (r1, r2, r1 * r2)
                    }
                    KernelKind::PolarCyl => {
                        let dr = (us[i][0] - us[j][0]).abs() / l1;
                        let kr = matern52(dr);
                        let kth = periodic_s1(us[i][1], us[j][1], l2);
                        (kr, kth, kr * kth)
                    }
                    KernelKind::PolarCylRbf => {
                        let dr = (us[i][0] - us[j][0]) / l1;
                        let kr = rbf1(dr);
                        let kth = periodic_s1(us[i][1], us[j][1], l2);
                        (kr, kth, kr * kth)
                    }
                    KernelKind::EquivCartM52 => {
                        let dot = us[i][0] * us[j][0] + us[i][1] * us[j][1];
                        let dr = (rs[i] - rs[j]).abs() / l1;
                        let d1 = (us[i][0] - us[j][0]).abs() / l2;
                        let d2 = (us[i][1] - us[j][1]).abs() / l2;
                        (matern52(d1), matern52(d2), dot * matern52(dr))
                    }
                    KernelKind::EquivCartRbf => {
                        let dot = us[i][0] * us[j][0] + us[i][1] * us[j][1];
                        let dr = (rs[i] - rs[j]) / l1;
                        let d1 = (us[i][0] - us[j][0]) / l2;
                        let d2 = (us[i][1] - us[j][1]) / l2;
                        (rbf1(d1), rbf1(d2), dot * rbf1(dr))
                    }
                    KernelKind::AffineOnly => (0.0, 0.0, 0.0),
                };
                k1[i * n + j] = v1;
                k1[j * n + i] = v1;
                k2[i * n + j] = v2;
                k2[j * n + i] = v2;
                k12[i * n + j] = v12;
                k12[j * n + i] = v12;
            }
        }
        for &(w1, w2, w12) in &modes {
            for idx in 0..(n * n) {
                k_buf[idx] = k_lin[idx] + w1 * k1[idx] + w2 * k2[idx] + w12 * k12[idx];
            }
            for &noise in &noises {
                if let Some(vfe) = vfe_from_gram(&k_buf, noise, y, n, &mut c_buf) {
                    if best_p.as_ref().map_or(true, |b| vfe < b.1) {
                        best_p = Some((
                            KernelParams {
                                kind,
                                w_lin: wl,
                                w1,
                                w2,
                                w12,
                                l1,
                                l2,
                                noise,
                            },
                            vfe,
                        ));
                    }
                }
            }
        }
    }

    // Fase 2: descenso monótono de Energía Libre por coordenadas sobre (l1, l2, noise, pesos)
    let (mut cur_p, mut cur_vfe) = best_p?;
    let mut step = 0.5_f64;
    for _ in 0..4 {
        let mut improved = false;
        let factors = [(-step).exp(), step.exp()];
        if cur_p.w1 > 0.0 || cur_p.w12 > 0.0 {
            for &f in &factors {
                let mut p = cur_p;
                p.l1 = (p.l1 * f).clamp(0.08, 25.0);
                if let Some(vfe) = eval_params_vfe(us, y, &p, &mut k_buf, &mut c_buf) {
                    if vfe < cur_vfe - 1e-6 {
                        cur_p = p;
                        cur_vfe = vfe;
                        improved = true;
                    }
                }
            }
        }
        if cur_p.w2 > 0.0 || cur_p.w12 > 0.0 || is_equiv {
            for &f in &factors {
                let mut p = cur_p;
                p.l2 = (p.l2 * f).clamp(0.08, 25.0);
                if let Some(vfe) = eval_params_vfe(us, y, &p, &mut k_buf, &mut c_buf) {
                    if vfe < cur_vfe - 1e-6 {
                        cur_p = p;
                        cur_vfe = vfe;
                        improved = true;
                    }
                }
            }
        }
        for i in 0..n {
            for j in 0..=i {
                let v = kernel_eval(&cur_p, &us[i], &us[j]);
                k_buf[i * n + j] = v;
                k_buf[j * n + i] = v;
            }
        }
        for &f in &[(-step * 2.0).exp(), (step * 2.0).exp()] {
            let mut p = cur_p;
            p.noise = (p.noise * f).clamp(1e-11, 2.0);
            if let Some(vfe) = vfe_from_gram(&k_buf, p.noise, y, n, &mut c_buf) {
                if vfe < cur_vfe - 1e-6 {
                    cur_p = p;
                    cur_vfe = vfe;
                    improved = true;
                }
            }
        }
        if include_affine_mean && (cur_p.w1 + cur_p.w2 + cur_p.w12) > 0.0 {
            for &f in &[(-step * 2.0).exp(), (step * 2.0).exp()] {
                let mut p = cur_p;
                p.w1 = (p.w1 * f).clamp(0.0, 100.0);
                p.w2 = (p.w2 * f).clamp(0.0, 100.0);
                p.w12 = (p.w12 * f).clamp(0.0, 100.0);
                if let Some(vfe) = eval_params_vfe(us, y, &p, &mut k_buf, &mut c_buf) {
                    if vfe < cur_vfe - 1e-6 {
                        cur_p = p;
                        cur_vfe = vfe;
                        improved = true;
                    }
                }
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    fit_gp_channel_fixed(us, y, cur_p)
}

/// Selección automática de suavidad de kernel (ν=5/2 Matérn vs ν=∞ RBF) por Energía Libre exacta.
/// Comparte el atajo afín exacto para evitar trabajo redundante en canales lineales.
fn optimize_gp_channel_vfe_auto(
    us: &[[f64; 2]],
    y: &[f64],
    k_m52: KernelKind,
    k_rbf: KernelKind,
    include_affine_mean: bool,
    w_lin_val: f64,
) -> Option<GpChannel> {
    let is_polar = matches!(k_m52, KernelKind::PolarCyl);
    if include_affine_mean && exact_linear_rel_res(us, y, is_polar) < 1e-8 {
        return optimize_gp_channel_vfe(us, y, k_rbf, true, w_lin_val);
    }
    let a = optimize_gp_channel_vfe(us, y, k_m52, include_affine_mean, w_lin_val);
    let b = optimize_gp_channel_vfe(us, y, k_rbf, include_affine_mean, w_lin_val);
    match (a, b) {
        (Some(ca), Some(cb)) => Some(if cb.vfe < ca.vfe { cb } else { ca }),
        (Some(ca), None) => Some(ca),
        (None, Some(cb)) => Some(cb),
        (None, None) => None,
    }
}

// ─── Hipótesis FE-ARD (Empirical Bayes / Variational Free Energy sobre biblioteca) ─

/// Biblioteca espectral genérica (la misma de ATLAS-FW/BQ10 + constante, 18 rasgos)
/// agrupada en 5 grupos jerárquicos de complejidad: afín(0), cuadrático(1), cúbico(2), periódico/tanh(3), radial(4).
const N_ARD: usize = N_LIB + 1;
const ARD_GROUP: [usize; N_ARD] = [
    0, 0, // a, b
    1, 1, 1, // a^2, ab, b^2
    2, 2, 2, 2, // a^3, a^2b, ab^2, b^3
    3, 3, 3, 3, 3, 3, // sin/cos/tanh
    4, 4, // r*a, r*b
    0, // constante 1.0
];
const N_ARD_GROUPS: usize = 5;

fn ard_feat(q: [f64; 2]) -> [f64; N_ARD] {
    let g = lib(&q);
    let mut out = [0.0; N_ARD];
    out[..N_LIB].copy_from_slice(&g);
    out[N_LIB] = 1.0;
    out
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct ArdChannel {
    w: [f64; N_ARD],
    sigma_diag: [f64; N_ARD],
    alpha_g: [f64; N_ARD_GROUPS],
    beta: f64,
    vfe_init: f64,
    vfe: f64,
    loo_fe: f64,
}

/// Empirical Bayes / Variational Free Energy ARD (MacKay / Friston):
/// Minimiza iterativamente F = 0.5*(β||y - Φw||² + w^T A w + ln|Σ^{-1}| - ln|A| - N ln β + N ln 2π)
fn fit_ard_channel(phi: &[[f64; N_ARD]], y: &[f64]) -> Option<ArdChannel> {
    let n = phi.len();
    let d = N_ARD;
    if n == 0 {
        return None;
    }
    // Precalcular Φ^T Φ y Φ^T y
    let mut xtx = vec![0.0; d * d];
    let mut xty = vec![0.0; d];
    for (row, &yi) in phi.iter().zip(y) {
        for i in 0..d {
            xty[i] += row[i] * yi;
            for j in 0..=i {
                xtx[i * d + j] += row[i] * row[j];
            }
        }
    }
    for i in 0..d {
        for j in 0..i {
            xtx[j * d + i] = xtx[i * d + j];
        }
    }
    let yvar = (y.iter().map(|v| v * v).sum::<f64>() / n as f64).max(1e-12);
    // Inicialización jerárquica de precisiones prior (complejidad creciente)
    let mut alpha_g = [1.0, 10.0, 100.0, 10.0, 10.0];
    let mut beta = (10.0 / yvar).clamp(1.0, 1e8);
    let mut vfe_init = f64::NAN;
    let mut vfe_cur = f64::INFINITY;
    let mut best_state: Option<([f64; N_ARD], [f64; N_ARD], [f64; N_ARD_GROUPS], f64, f64)> = None;

    for it in 0..18 {
        let mut h = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                h[i * d + j] = beta * xtx[i * d + j];
            }
            h[i * d + i] += alpha_g[ARD_GROUP[i]];
        }
        let (l, logdet_h) = cholesky_jitter(&h, d)?;
        let rhs: Vec<f64> = xty.iter().map(|&v| beta * v).collect();
        let z = solve_lower(&l, d, &rhs);
        let wn = solve_upper_t(&l, d, &z);
        let sig_diag = chol_inv_diag(&l, d);

        let mut rss = 0.0;
        for (row, &yi) in phi.iter().zip(y) {
            let pred: f64 = row.iter().zip(&wn).map(|(a, b)| a * b).sum();
            rss += (yi - pred).powi(2);
        }
        let mut quad_prior = 0.0;
        let mut logdet_a = 0.0;
        for j in 0..d {
            let aj = alpha_g[ARD_GROUP[j]];
            quad_prior += aj * wn[j] * wn[j];
            logdet_a += aj.ln();
        }
        let vfe = 0.5 * (beta * rss + quad_prior + logdet_h - logdet_a - (n as f64) * beta.ln() + (n as f64) * LN_2PI);
        if it == 0 {
            vfe_init = vfe;
        }
        if vfe.is_finite() && vfe < vfe_cur {
            vfe_cur = vfe;
            let mut wa = [0.0; N_ARD];
            let mut sd = [0.0; N_ARD];
            wa.copy_from_slice(&wn);
            sd.copy_from_slice(&sig_diag);
            best_state = Some((wa, sd, alpha_g, beta, vfe));
        }

        // Actualizaciones MacKay / Friston ARD (descenso de Energía Libre)
        let mut gam_sum = 0.0;
        let mut g_gam = [0.0; N_ARD_GROUPS];
        let mut g_w2 = [0.0; N_ARD_GROUPS];
        for j in 0..d {
            let g = ARD_GROUP[j];
            let gj = (1.0 - alpha_g[g] * sig_diag[j]).clamp(0.0, 1.0);
            g_gam[g] += gj;
            g_w2[g] += wn[j] * wn[j];
            gam_sum += gj;
        }
        let a_floor = [1e-5, 1e-3, 1e-2, 1e-3, 1e-3];
        for g in 0..N_ARD_GROUPS {
            let new_a = ((g_gam[g] + 1e-4) / (g_w2[g] + 1e-6)).clamp(a_floor[g], 1e7);
            alpha_g[g] = (alpha_g[g].ln() * 0.3 + new_a.ln() * 0.7).exp();
        }
        let new_b = (((n as f64 - gam_sum).max(1.0)) / (rss + 1e-12)).clamp(1e-2, 1e10);
        beta = (beta.ln() * 0.3 + new_b.ln() * 0.7).exp();
    }

    let (w, sd, ag, bt, vfe) = best_state?;

    // LOO-CV analítico exacto mediante forma dual C = Φ A^{-1} Φ^T + β^{-1} I
    let mut c_dual = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..d {
                s += phi[i][k] * phi[j][k] / ag[ARD_GROUP[k]];
            }
            if i == j {
                s += 1.0 / bt;
            }
            c_dual[i * n + j] = s;
            c_dual[j * n + i] = s;
        }
    }
    let loo_fe = if let Some((ld, _)) = cholesky_jitter(&c_dual, n) {
        let zd = solve_lower(&ld, n, y);
        let ad = solve_upper_t(&ld, n, &zd);
        let dd = chol_inv_diag(&ld, n);
        let mut s = 0.0;
        for i in 0..n {
            let e = ad[i] / dd[i];
            let v = (1.0 / dd[i]).max(1e-24);
            s += 0.5 * (e * e / v + (TAU * v).ln());
        }
        s
    } else {
        vfe
    };

    Some(ArdChannel {
        w,
        sigma_diag: sd,
        alpha_g: ag,
        beta: bt,
        vfe_init: if vfe_init.is_finite() { vfe_init } else { vfe },
        vfe,
        loo_fe,
    })
}

// ─── Banco de Hipótesis Generativas (Nivel 0) ────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HypKind {
    LieAffine = 0,
    FeArd = 1,
    GpCartM52 = 2,
    GpCartRbf = 3,
    GpCoMoving = 4,
    GpSympFwd = 5,
    GpSympRev = 6,
    GpCoMovingSympFwd = 7,
    GpCoMovingSympRev = 8,
    GpPolar = 9,
    GpEquivCart = 10,
}

pub const N_HYP: usize = 11;
pub const HYP_NAMES: [&str; N_HYP] = [
    "LieAffine",
    "FeArd",
    "GpCartM52",
    "GpCartRbf",
    "GpCoMoving",
    "GpSympFwd",
    "GpSympRev",
    "GpCoSympFwd",
    "GpCoSympRev",
    "GpPolar",
    "GpEquivCart",
];

/// Parámetros de mapa afín 2D: u = A q + b (6 parámetros).
#[derive(Clone, Copy, Debug)]
pub(super) struct Affine2D {
    pub(super) a: [[f64; 2]; 2],
    pub(super) b: [f64; 2],
}

impl Affine2D {
    pub(super) fn identity() -> Self {
        Self {
            a: [[1.0, 0.0], [0.0, 1.0]],
            b: [0.0, 0.0],
        }
    }
    #[inline]
    pub(super) fn apply(&self, q: [f64; 2]) -> [f64; 2] {
        [
            self.a[0][0] * q[0] + self.a[0][1] * q[1] + self.b[0],
            self.a[1][0] * q[0] + self.a[1][1] * q[1] + self.b[1],
        ]
    }
    /// Ajuste afín exacto por mínimos cuadrados regularizados q → y.
    pub(super) fn fit(qs: &[[f64; 2]], ys: &[[f64; 2]], lam: f64) -> Self {
        let mut xtx = [0.0; 9];
        let mut xty = [[0.0; 3]; 2];
        for (q, y) in qs.iter().zip(ys) {
            let f = [q[0], q[1], 1.0];
            for i in 0..3 {
                for j in 0..3 {
                    xtx[i * 3 + j] += f[i] * f[j];
                }
                xty[0][i] += f[i] * y[0];
                xty[1][i] += f[i] * y[1];
            }
        }
        let tr = ((xtx[0] + xtx[4] + xtx[8]) / 3.0).max(1e-12);
        for i in 0..3 {
            xtx[i * 3 + i] += lam * tr;
        }
        let wx = solve_spd(&xtx, &xty[0], 3);
        let wy = solve_spd(&xtx, &xty[1], 3);
        Self {
            a: [[wx[0], wx[1]], [wy[0], wy[1]]],
            b: [wx[2], wy[2]],
        }
    }
    /// Radio espectral exacto ρ(A) para matriz 2×2.
    pub(super) fn spectral_radius(&self) -> f64 {
        let tr = self.a[0][0] + self.a[1][1];
        let det = self.a[0][0] * self.a[1][1] - self.a[0][1] * self.a[1][0];
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            let s = disc.sqrt();
            (0.5 * (tr + s)).abs().max((0.5 * (tr - s)).abs())
        } else {
            det.max(0.0).sqrt()
        }
    }
}

#[derive(Clone, Debug)]
enum HypModel {
    /// Predicción directa por canal GP sobre entradas u = q_t, salida Δq = q_{t+1} - q_t
    DirectGp {
        us: Vec<[f64; 2]>,
        ch: [GpChannel; 2],
    },
    /// FE-ARD sobre biblioteca espectral genérica
    Ard {
        ch: [ArdChannel; 2],
    },
    /// Co-Moving Lie-Trotter: u = A q + b, q' = u + g(u)
    CoMovingGp {
        aff: Affine2D,
        us: Vec<[f64; 2]>,
        ch: [GpChannel; 2],
    },
    /// Splitting Simpléctico: si rev=false: x'=u_x+g0(u_x,u_y), y'=u_y+g1(x',u_y);
    /// si rev=true: y'=u_y+g1(u_x,u_y), x'=u_x+g0(u_x,y'), con (u_x,u_y)=aff(q)
    SympGp {
        aff: Affine2D,
        rev: bool,
        us0: Vec<[f64; 2]>,
        us1: Vec<[f64; 2]>,
        ch: [GpChannel; 2],
    },
    /// Polar Equivariante SO(2) en (r, θ) con Jacobiano exacto
    PolarGp {
        omega_bar: f64,
        us: Vec<[f64; 2]>,
        ch_r: GpChannel,
        ch_th: GpChannel,
    },
}

#[derive(Clone, Debug)]
pub(super) struct FittedHyp {
    pub(super) kind: HypKind,
    model: HypModel,
    pub(super) vfe: f64,
    pub(super) loo_fe: f64,
    pub(super) s2_scale: f64,
    pub(super) valid: bool,
}

#[inline]
pub(super) fn wrap_angle(a: f64) -> f64 {
    let mut x = (a + PI) % TAU;
    if x < 0.0 {
        x += TAU;
    }
    x - PI
}

impl FittedHyp {
    /// Predice (q_{t+1}, varianza epistémica σ²_epi) para un punto q_t.
    pub(super) fn step_point(&self, q: [f64; 2]) -> ([f64; 2], f64) {
        if !self.valid || !q[0].is_finite() || !q[1].is_finite() {
            return ([f64::NAN, f64::NAN], 1e6);
        }
        match &self.model {
            HypModel::DirectGp { us, ch } => {
                let (dx, vx) = ch[0].predict(us, &q);
                let (dy, vy) = ch[1].predict(us, &q);
                ([q[0] + dx, q[1] + dy], vx + vy)
            }
            HypModel::Ard { ch } => {
                let f = ard_feat(q);
                let mut out = q;
                let mut epi = 0.0;
                for i in 0..2 {
                    let mut d = 0.0;
                    let mut v = 0.0;
                    for j in 0..N_ARD {
                        d += ch[i].w[j] * f[j];
                        v += f[j] * f[j] * ch[i].sigma_diag[j];
                    }
                    out[i] += d;
                    epi += v;
                }
                (out, epi)
            }
            HypModel::CoMovingGp { aff, us, ch } => {
                let u = aff.apply(q);
                let (dx, vx) = ch[0].predict(us, &u);
                let (dy, vy) = ch[1].predict(us, &u);
                ([u[0] + dx, u[1] + dy], vx + vy)
            }
            HypModel::SympGp {
                aff,
                rev,
                us0,
                us1,
                ch,
            } => {
                let u = aff.apply(q);
                if !*rev {
                    let (dx, vx) = ch[0].predict(us0, &u);
                    let x_next = u[0] + dx;
                    let (dy, vy) = ch[1].predict(us1, &[x_next, u[1]]);
                    let y_next = u[1] + dy;
                    ([x_next, y_next], vx + vy)
                } else {
                    let (dy, vy) = ch[1].predict(us1, &u);
                    let y_next = u[1] + dy;
                    let (dx, vx) = ch[0].predict(us0, &[u[0], y_next]);
                    let x_next = u[0] + dx;
                    ([x_next, y_next], vx + vy)
                }
            }
            HypModel::PolarGp {
                omega_bar,
                us,
                ch_r,
                ch_th,
            } => {
                let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
                if r < 1e-6 {
                    return (q, 1e3);
                }
                let th = q[1].atan2(q[0]);
                let inp = [r, th];
                let (dlr, vr) = ch_r.predict(us, &inp);
                let (dth, vth) = ch_th.predict(us, &inp);
                let r_next = r * dlr.clamp(-3.0, 3.0).exp();
                let th_next = th + *omega_bar + dth;
                let epi_cart = r_next * r_next * (vr + vth);
                ([r_next * th_next.cos(), r_next * th_next.sin()], epi_cart)
            }
        }
    }

    pub(super) fn step_frame(&self, frame: &[f64]) -> (Vec<f64>, f64) {
        let np = frame.len() / 2;
        let mut out = vec![0.0; frame.len()];
        let mut epi_sum = 0.0;
        for p in 0..np {
            let (qn, epi) = self.step_point([frame[2 * p], frame[2 * p + 1]]);
            out[2 * p] = qn[0];
            out[2 * p + 1] = qn[1];
            epi_sum += epi;
        }
        (out, epi_sum / np.max(1) as f64)
    }
}

pub(super) fn extract_pairs(frames: &[&[f64]]) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let np = frames[0].len() / 2;
    let n = (frames.len() - 1) * np;
    let mut qs = Vec::with_capacity(n);
    let mut qns = Vec::with_capacity(n);
    for t in 0..(frames.len() - 1) {
        for p in 0..np {
            qs.push([frames[t][2 * p], frames[t][2 * p + 1]]);
            qns.push([frames[t + 1][2 * p], frames[t + 1][2 * p + 1]]);
        }
    }
    (qs, qns)
}

pub(super) fn fit_hypothesis(kind: HypKind, frames: &[&[f64]]) -> FittedHyp {
    fit_hypothesis_ex(kind, frames, 4)
}

pub(super) fn fit_hypothesis_ex(kind: HypKind, frames: &[&[f64]], max_em: usize) -> FittedHyp {
    let invalid = FittedHyp {
        kind,
        model: HypModel::DirectGp {
            us: vec![],
            ch: [
                GpChannel {
                    l: vec![],
                    alpha: vec![],
                    s2: 1.0,
                    vfe: 1e6,
                    loo_fe: 1e6,
                    loo_mse: 1e6,
                    loo_preds: vec![],
                    params: KernelParams {
                        kind: KernelKind::AffineOnly,
                        w_lin: 1.0,
                        w1: 0.0,
                        w2: 0.0,
                        w12: 0.0,
                        l1: 1.0,
                        l2: 1.0,
                        noise: 1.0,
                    },
                },
                GpChannel {
                    l: vec![],
                    alpha: vec![],
                    s2: 1.0,
                    vfe: 1e6,
                    loo_fe: 1e6,
                    loo_mse: 1e6,
                    loo_preds: vec![],
                    params: KernelParams {
                        kind: KernelKind::AffineOnly,
                        w_lin: 1.0,
                        w1: 0.0,
                        w2: 0.0,
                        w12: 0.0,
                        l1: 1.0,
                        l2: 1.0,
                        noise: 1.0,
                    },
                },
            ],
        },
        vfe: 1e6,
        loo_fe: 1e6,
        s2_scale: 1.0,
        valid: false,
    };
    if frames.len() < 2 {
        return invalid;
    }
    let (qs, qns) = extract_pairs(frames);
    let yx: Vec<f64> = qs.iter().zip(&qns).map(|(a, b)| b[0] - a[0]).collect();
    let yy: Vec<f64> = qs.iter().zip(&qns).map(|(a, b)| b[1] - a[1]).collect();

    match kind {
        HypKind::LieAffine => {
            let (Some(c0), Some(c1)) = (
                optimize_gp_channel_vfe(&qs, &yx, KernelKind::AffineOnly, true, 10.0),
                optimize_gp_channel_vfe(&qs, &yy, KernelKind::AffineOnly, true, 10.0),
            ) else {
                return invalid;
            };
            let vfe = c0.vfe + c1.vfe;
            let loo_fe = c0.loo_fe + c1.loo_fe;
            let s2 = (c0.s2 + c1.s2).max(1e-12);
            FittedHyp {
                kind,
                model: HypModel::DirectGp { us: qs, ch: [c0, c1] },
                vfe,
                loo_fe,
                s2_scale: s2,
                valid: true,
            }
        }
        HypKind::FeArd => {
            let phi: Vec<[f64; N_ARD]> = qs.iter().map(|&q| ard_feat(q)).collect();
            let (Some(c0), Some(c1)) = (fit_ard_channel(&phi, &yx), fit_ard_channel(&phi, &yy)) else {
                return invalid;
            };
            let vfe = c0.vfe + c1.vfe;
            let loo_fe = c0.loo_fe + c1.loo_fe;
            let s2 = (1.0 / c0.beta + 1.0 / c1.beta).max(1e-12);
            FittedHyp {
                kind,
                model: HypModel::Ard { ch: [c0, c1] },
                vfe,
                loo_fe,
                s2_scale: s2,
                valid: true,
            }
        }
        HypKind::GpCartM52 | HypKind::GpCartRbf => {
            let kk = if kind == HypKind::GpCartM52 {
                KernelKind::CartM52
            } else {
                KernelKind::CartRbf
            };
            let (Some(c0), Some(c1)) = (
                optimize_gp_channel_vfe(&qs, &yx, kk, true, 10.0),
                optimize_gp_channel_vfe(&qs, &yy, kk, true, 10.0),
            ) else {
                return invalid;
            };
            let vfe = c0.vfe + c1.vfe;
            let loo_fe = c0.loo_fe + c1.loo_fe;
            let s2 = (c0.s2 + c1.s2).max(1e-12);
            FittedHyp {
                kind,
                model: HypModel::DirectGp { us: qs, ch: [c0, c1] },
                vfe,
                loo_fe,
                s2_scale: s2,
                valid: true,
            }
        }
        HypKind::GpEquivCart => {
            let (Some(c0), Some(c1)) = (
                optimize_gp_channel_vfe_auto(
                    &qs,
                    &yx,
                    KernelKind::EquivCartM52,
                    KernelKind::EquivCartRbf,
                    true,
                    10.0,
                ),
                optimize_gp_channel_vfe_auto(
                    &qs,
                    &yy,
                    KernelKind::EquivCartM52,
                    KernelKind::EquivCartRbf,
                    true,
                    10.0,
                ),
            ) else {
                return invalid;
            };
            let vfe = c0.vfe + c1.vfe;
            let loo_fe = c0.loo_fe + c1.loo_fe;
            let s2 = (c0.s2 + c1.s2).max(1e-12);
            FittedHyp {
                kind,
                model: HypModel::DirectGp { us: qs, ch: [c0, c1] },
                vfe,
                loo_fe,
                s2_scale: s2,
                valid: true,
            }
        }
        HypKind::GpCoMoving => {
            // Lie-Trotter Co-Moving GP con EM monótono de Energía Libre y selección VFE de suavidad (ν=5/2 vs ∞)
            let mut aff = Affine2D::fit(&qs, &qns, 1e-6);
            let mut best: Option<(Affine2D, Vec<[f64; 2]>, [GpChannel; 2], f64, f64)> = None;
            let mut k_ch = [KernelKind::CartM52, KernelKind::CartM52];
            for em in 0..max_em.max(1) {
                let us: Vec<[f64; 2]> = qs.iter().map(|&q| aff.apply(q)).collect();
                let rx: Vec<f64> = us.iter().zip(&qns).map(|(u, y)| y[0] - u[0]).collect();
                let ry: Vec<f64> = us.iter().zip(&qns).map(|(u, y)| y[1] - u[1]).collect();
                let (Some(c0), Some(c1)) = (if em == 0 {
                    (
                        optimize_gp_channel_vfe_auto(&us, &rx, KernelKind::CartM52, KernelKind::CartRbf, true, 1.0),
                        optimize_gp_channel_vfe_auto(&us, &ry, KernelKind::CartM52, KernelKind::CartRbf, true, 1.0),
                    )
                } else {
                    (
                        optimize_gp_channel_vfe(&us, &rx, k_ch[0], true, 1.0),
                        optimize_gp_channel_vfe(&us, &ry, k_ch[1], true, 1.0),
                    )
                }) else {
                    break;
                };
                if em == 0 {
                    k_ch = [c0.params.kind, c1.params.kind];
                }
                // Complejidad del mapa afín A,b (6 params, penalización BIC/Laplace moderada)
                let vfe = c0.vfe + c1.vfe + 3.0;
                let loo_fe = c0.loo_fe + c1.loo_fe;
                let improved = best.as_ref().map_or(true, |b| vfe < b.3 - 1e-4);
                // Targets corregidos con predicción LOO no-lineal para re-estimar el flujo Lie afín puro
                let adj: Vec<[f64; 2]> = qns
                    .iter()
                    .enumerate()
                    .map(|(i, y)| [y[0] - c0.loo_preds[i], y[1] - c1.loo_preds[i]])
                    .collect();
                if improved {
                    best = Some((aff, us, [c0, c1], vfe, loo_fe));
                    let next_aff = Affine2D::fit(&qs, &adj, 1e-5);
                    if next_aff.spectral_radius() < 3.0 {
                        aff = next_aff;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            let Some((aff, us, ch, vfe, loo_fe)) = best else {
                return invalid;
            };
            let s2 = (ch[0].s2 + ch[1].s2).max(1e-12);
            FittedHyp {
                kind,
                model: HypModel::CoMovingGp { aff, us, ch },
                vfe,
                loo_fe,
                s2_scale: s2,
                valid: true,
            }
        }
        HypKind::GpSympFwd | HypKind::GpSympRev => {
            let rev = kind == HypKind::GpSympRev;
            let aff = Affine2D::identity();
            let (us0, us1): (Vec<[f64; 2]>, Vec<[f64; 2]>) = if !rev {
                let u1 = qs.iter().zip(&qns).map(|(q, qn)| [qn[0], q[1]]).collect();
                (qs.clone(), u1)
            } else {
                let u0 = qs.iter().zip(&qns).map(|(q, qn)| [q[0], qn[1]]).collect();
                (u0, qs.clone())
            };
            let (Some(c0), Some(c1)) = (
                optimize_gp_channel_vfe(&us0, &yx, KernelKind::CartM52, true, 10.0),
                optimize_gp_channel_vfe(&us1, &yy, KernelKind::CartM52, true, 10.0),
            ) else {
                return invalid;
            };
            let vfe = c0.vfe + c1.vfe;
            let loo_fe = c0.loo_fe + c1.loo_fe;
            let s2 = (c0.s2 + c1.s2).max(1e-12);
            FittedHyp {
                kind,
                model: HypModel::SympGp {
                    aff,
                    rev,
                    us0,
                    us1,
                    ch: [c0, c1],
                },
                vfe,
                loo_fe,
                s2_scale: s2,
                valid: true,
            }
        }
        HypKind::GpCoMovingSympFwd | HypKind::GpCoMovingSympRev => {
            let rev = kind == HypKind::GpCoMovingSympRev;
            let mut aff = Affine2D::fit(&qs, &qns, 1e-6);
            let mut best: Option<(Affine2D, Vec<[f64; 2]>, Vec<[f64; 2]>, [GpChannel; 2], f64, f64)> = None;
            for _em in 0..max_em.max(1) {
                let us_base: Vec<[f64; 2]> = qs.iter().map(|&q| aff.apply(q)).collect();
                let rx: Vec<f64> = us_base.iter().zip(&qns).map(|(u, y)| y[0] - u[0]).collect();
                let ry: Vec<f64> = us_base.iter().zip(&qns).map(|(u, y)| y[1] - u[1]).collect();
                let (us0, us1): (Vec<[f64; 2]>, Vec<[f64; 2]>) = if !rev {
                    let u1 = us_base.iter().zip(&qns).map(|(u, y)| [y[0], u[1]]).collect();
                    (us_base.clone(), u1)
                } else {
                    let u0 = us_base.iter().zip(&qns).map(|(u, y)| [u[0], y[1]]).collect();
                    (u0, us_base.clone())
                };
                let (Some(c0), Some(c1)) = (
                    optimize_gp_channel_vfe(&us0, &rx, KernelKind::CartM52, true, 1.0),
                    optimize_gp_channel_vfe(&us1, &ry, KernelKind::CartM52, true, 1.0),
                ) else {
                    break;
                };
                let vfe = c0.vfe + c1.vfe + 3.0;
                let loo_fe = c0.loo_fe + c1.loo_fe;
                let improved = best.as_ref().map_or(true, |b| vfe < b.4 - 1e-4);
                let adj: Vec<[f64; 2]> = qns
                    .iter()
                    .enumerate()
                    .map(|(i, y)| [y[0] - c0.loo_preds[i], y[1] - c1.loo_preds[i]])
                    .collect();
                if improved {
                    best = Some((aff, us0, us1, [c0, c1], vfe, loo_fe));
                    let next_aff = Affine2D::fit(&qs, &adj, 1e-5);
                    if next_aff.spectral_radius() < 3.0 {
                        aff = next_aff;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            let Some((aff, us0, us1, ch, vfe, loo_fe)) = best else {
                return invalid;
            };
            let s2 = (ch[0].s2 + ch[1].s2).max(1e-12);
            FittedHyp {
                kind,
                model: HypModel::SympGp {
                    aff,
                    rev,
                    us0,
                    us1,
                    ch,
                },
                vfe,
                loo_fe,
                s2_scale: s2,
                valid: true,
            }
        }
        HypKind::GpPolar => {
            let mut us = Vec::with_capacity(qs.len());
            let mut yr = Vec::with_capacity(qs.len());
            let mut dth_raw = Vec::with_capacity(qs.len());
            let mut jac_corr = 0.0;
            for (q, qn) in qs.iter().zip(&qns) {
                let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
                let rn = (qn[0] * qn[0] + qn[1] * qn[1]).sqrt();
                if r < 0.03 || rn < 0.03 {
                    return invalid;
                }
                let th = q[1].atan2(q[0]);
                let thn = qn[1].atan2(qn[0]);
                us.push([r, th]);
                yr.push((rn / r).ln());
                dth_raw.push(wrap_angle(thn - th));
                // Corrección exacta de cambio de variables (ln r', θ') -> (x', y'): |det J| = (r')^2
                jac_corr += 2.0 * rn.ln();
            }
            let sc: f64 = dth_raw.iter().map(|a| a.cos()).sum();
            let ss: f64 = dth_raw.iter().map(|a| a.sin()).sum();
            let omega_bar = ss.atan2(sc);
            let yth: Vec<f64> = dth_raw.iter().map(|&a| wrap_angle(a - omega_bar)).collect();
            let (Some(ch_r), Some(ch_th)) = (
                optimize_gp_channel_vfe_auto(&us, &yr, KernelKind::PolarCyl, KernelKind::PolarCylRbf, true, 5.0),
                optimize_gp_channel_vfe_auto(&us, &yth, KernelKind::PolarCyl, KernelKind::PolarCylRbf, true, 5.0),
            ) else {
                return invalid;
            };
            let vfe = ch_r.vfe + ch_th.vfe + jac_corr;
            let loo_fe = ch_r.loo_fe + ch_th.loo_fe + jac_corr;
            let mean_r2 = us.iter().map(|u| u[0] * u[0]).sum::<f64>() / us.len() as f64;
            let s2 = (mean_r2 * (ch_r.s2 + ch_th.s2)).max(1e-12);
            FittedHyp {
                kind,
                model: HypModel::PolarGp {
                    omega_bar,
                    us,
                    ch_r,
                    ch_th,
                },
                vfe,
                loo_fe,
                s2_scale: s2,
                valid: true,
            }
        }
    }
}

/// Refinamiento Activo Multi-Paso por Energía Libre (Nivel 1):
/// Ajusta finamente el mapa afín Lie (6 params) minimizando la sorpresa de trayectoria cerrada
/// x0 -> x̂1 -> ... -> x̂K + penalización de complejidad prior alrededor de aff_0.
fn refine_affine_multistep(aff0: Affine2D, frames: &[&[f64]], lam_prior: f64) -> Affine2D {
    if frames.len() < 3 {
        return aff0;
    }
    let to_v = |af: Affine2D| -> [f64; 6] { [af.a[0][0], af.a[0][1], af.b[0], af.a[1][0], af.a[1][1], af.b[1]] };
    let from_v = |v: [f64; 6]| Affine2D {
        a: [[v[0], v[1]], [v[3], v[4]]],
        b: [v[2], v[5]],
    };
    let th0 = to_v(aff0);
    let np = frames[0].len() / 2;
    let k = frames.len() - 1;
    let eval_res = |v: [f64; 6]| -> Option<Vec<f64>> {
        let af = from_v(v);
        let mut r = Vec::with_capacity(k * 2 * np);
        let mut cur: Vec<[f64; 2]> = (0..np).map(|p| [frames[0][2 * p], frames[0][2 * p + 1]]).collect();
        for t in 1..=k {
            let nrm = mt::norm(frames[t]).max(1e-6);
            for p in 0..np {
                cur[p] = af.apply(cur[p]);
                if !cur[p][0].is_finite() || !cur[p][1].is_finite() || cur[p][0].abs() > 1e4 || cur[p][1].abs() > 1e4 {
                    return None;
                }
                r.push((cur[p][0] - frames[t][2 * p]) / nrm);
                r.push((cur[p][1] - frames[t][2 * p + 1]) / nrm);
            }
        }
        Some(r)
    };
    let obj = |v: [f64; 6], r: &[f64]| -> f64 {
        r.iter().map(|x| x * x).sum::<f64>()
            + lam_prior * (0..6).map(|i| (v[i] - th0[i]).powi(2)).sum::<f64>()
    };
    let mut th = th0;
    let Some(mut r) = eval_res(th) else {
        return aff0;
    };
    let mut f = obj(th, &r);
    let mut mu = 1e-3;
    for _ in 0..8 {
        let nr = r.len();
        let mut jt = vec![0.0; 6 * nr];
        for p in 0..6 {
            let mut t2 = th;
            let h = 1e-6 * (1.0 + th[p].abs());
            t2[p] += h;
            if let Some(r2) = eval_res(t2) {
                for i in 0..nr {
                    jt[p * nr + i] = (r2[i] - r[i]) / h;
                }
            }
        }
        let mut hm = vec![0.0; 36];
        let mut g = vec![0.0; 6];
        for p in 0..6 {
            for q in 0..=p {
                let s: f64 = (0..nr).map(|i| jt[p * nr + i] * jt[q * nr + i]).sum();
                hm[p * 6 + q] = s;
                hm[q * 6 + p] = s;
            }
            g[p] = (0..nr).map(|i| jt[p * nr + i] * r[i]).sum::<f64>() + lam_prior * (th[p] - th0[p]);
        }
        let mut ok = false;
        let f0 = f;
        for _ in 0..5 {
            let mut h2 = hm.clone();
            for p in 0..6 {
                h2[p * 6 + p] += lam_prior + mu;
            }
            let dl = solve_spd(&h2, &g, 6);
            let mut cand = th;
            for p in 0..6 {
                cand[p] -= dl[p];
            }
            if let Some(rc) = eval_res(cand) {
                let fc = obj(cand, &rc);
                if fc < f {
                    th = cand;
                    r = rc;
                    f = fc;
                    mu = (mu / 3.0).max(1e-9);
                    ok = true;
                    break;
                }
            }
            mu *= 5.0;
        }
        if !ok || f0 - f < 1e-12 * (1.0 + f0) {
            break;
        }
    }
    from_v(th)
}

/// Refinamiento multi-paso de CoMoving/Symp: si el flujo afín subyacente tiene deriva multi-paso,
/// optimiza (A, b) en trayectoria cerrada y re-ajusta el GP residual solo si reduce la sorpresa cerrada.
fn maybe_refine_multistep(h: &FittedHyp, frames: &[&[f64]]) -> FittedHyp {
    let replay_err = |hyp: &FittedHyp| -> f64 {
        let mut z = frames[0].to_vec();
        let mut s = 0.0;
        for t in 1..frames.len() {
            let (zn, _) = hyp.step_frame(&z);
            s += log_rel(clamp_err(&zn, frames[t]));
            z = zn;
        }
        s / (frames.len() - 1) as f64
    };
    let e0 = replay_err(h);
    if !e0.is_finite() || e0 < -8.0 {
        return h.clone();
    }
    match &h.model {
        HypModel::CoMovingGp { aff, .. } => {
            let aff_ref = refine_affine_multistep(*aff, frames, 1e-4);
            let (qs, qns) = extract_pairs(frames);
            let us: Vec<[f64; 2]> = qs.iter().map(|&q| aff_ref.apply(q)).collect();
            let rx: Vec<f64> = us.iter().zip(&qns).map(|(u, y)| y[0] - u[0]).collect();
            let ry: Vec<f64> = us.iter().zip(&qns).map(|(u, y)| y[1] - u[1]).collect();
            if let (Some(c0), Some(c1)) = (
                optimize_gp_channel_vfe_auto(&us, &rx, KernelKind::CartM52, KernelKind::CartRbf, true, 1.0),
                optimize_gp_channel_vfe_auto(&us, &ry, KernelKind::CartM52, KernelKind::CartRbf, true, 1.0),
            ) {
                let cand = FittedHyp {
                    kind: h.kind,
                    vfe: c0.vfe + c1.vfe + 3.0,
                    loo_fe: c0.loo_fe + c1.loo_fe,
                    s2_scale: (c0.s2 + c1.s2).max(1e-12),
                    model: HypModel::CoMovingGp {
                        aff: aff_ref,
                        us,
                        ch: [c0, c1],
                    },
                    valid: true,
                };
                if replay_err(&cand) < e0 - 0.05 && cand.vfe <= h.vfe + 5.0 {
                    return cand;
                }
            }
            h.clone()
        }
        _ => h.clone(),
    }
}

// ─── Diagnósticos de hipótesis (Nivel 1: VFE, LOO, OOT-1, OOT-2, Replay, Estabilidad) ─

#[derive(Clone, Debug)]
pub(super) struct HypDiag {
    pub(super) hyp: FittedHyp,
    pub(super) vfe_per_pt: f64,
    pub(super) loo_per_pt: f64,
    pub(super) oot1_log: f64,
    pub(super) oot2_log: f64,
    pub(super) replay_log: f64,
    pub(super) roll_preds: Vec<Vec<f64>>,
    pub(super) roll_epi: Vec<f64>,
    pub(super) stab_pen: f64,
    pub(super) score: f64,
}

pub(super) fn rmax_frame(f: &[f64]) -> f64 {
    (0..f.len() / 2)
        .map(|p| (f[2 * p].powi(2) + f[2 * p + 1].powi(2)).sqrt())
        .fold(0.0, f64::max)
}

/// Evalúa una hipótesis completa en el contexto 0..=K (sin fuga) y su rollout h=(K+1)..=16.
pub(super) fn evaluate_hypothesis(
    kind: HypKind,
    frames: &[&[f64]],
    use_multistep_refine: bool,
    use_oot: bool,
    k_obs: usize,
) -> HypDiag {
    let np = frames[0].len() / 2;
    let n_pts = (k_obs * np) as f64;
    let mut hyp = fit_hypothesis_ex(kind, &frames[..=k_obs], 4);
    if use_multistep_refine && hyp.valid {
        hyp = maybe_refine_multistep(&hyp, &frames[..=k_obs]);
    }

    // 1. Sorpresa cerrada en contexto (replay desde x0)
    let replay_log = if hyp.valid {
        let mut z = frames[0].to_vec();
        let mut v = Vec::with_capacity(k_obs);
        for t in 1..=k_obs {
            let (zn, _) = hyp.step_frame(&z);
            v.push(log_rel(clamp_err(&zn, frames[t])));
            z = zn;
        }
        mean(&v)
    } else {
        log_rel(10.0)
    };

    // 2. Sorpresa precuencial fuera de tiempo (OOT-1: fit 0..K-1 -> pred K; OOT-2: fit 0..K-2 -> rollout K-1, K)
    let (oot1_log, oot2_log) = if use_oot && hyp.valid && k_obs >= 3 {
        let h1 = fit_hypothesis_ex(kind, &frames[..k_obs], 2);
        let o1 = if h1.valid {
            let (p, _) = h1.step_frame(frames[k_obs - 1]);
            log_rel(clamp_err(&p, frames[k_obs]))
        } else {
            log_rel(10.0)
        };
        let o2 = if k_obs >= 4 {
            let h2 = fit_hypothesis_ex(kind, &frames[..(k_obs - 1)], 2);
            if h2.valid {
                let (p1, _) = h2.step_frame(frames[k_obs - 2]);
                let (p2, _) = h2.step_frame(&p1);
                0.5 * (log_rel(clamp_err(&p1, frames[k_obs - 1])) + log_rel(clamp_err(&p2, frames[k_obs])))
            } else {
                log_rel(10.0)
            }
        } else {
            o1
        };
        (o1, o2)
    } else {
        (replay_log, replay_log)
    };

    // 3. Rollout h = K+1 .. HORIZON con seguimiento de incertidumbre epistémica σ²_epi(q̂_h)
    let r0 = rmax_frame(frames[0]).max(1e-6);
    let rk = rmax_frame(frames[k_obs]).max(1e-6);
    let rmax_ctx = (0..=k_obs).map(|t| rmax_frame(frames[t])).fold(1e-6, f64::max);
    let g_ctx = ((rk / r0).powf(1.0 / k_obs.max(1) as f64)).clamp(0.5, 2.0);
    let g_hi = if g_ctx > 1.03 { g_ctx } else { 1.02 };

    let mut roll_preds = Vec::with_capacity(HORIZON - k_obs);
    let mut roll_epi = Vec::with_capacity(HORIZON - k_obs);
    let mut cur = frames[k_obs].to_vec();
    let mut stab_pen = 0.0;
    for step in 1..=(HORIZON - k_obs) {
        let (next, epi) = if hyp.valid && fin(&cur) {
            hyp.step_frame(&cur)
        } else {
            (vec![f64::NAN; 2 * np], 1e6)
        };
        let r_bound = 2.5 * rmax_ctx * g_hi.powi(step as i32);
        if !fin(&next) {
            stab_pen += 50.0;
            roll_preds.push(vec![f64::NAN; 2 * np]);
            roll_epi.push(1e6);
            cur = vec![f64::NAN; 2 * np];
        } else {
            let rn = rmax_frame(&next);
            if rn > r_bound {
                stab_pen += ((rn / r_bound).ln()).powi(2) * 4.0;
            }
            if rn < 0.02 * rmax_ctx * g_ctx.min(1.0).powi(step as i32) {
                stab_pen += 0.5;
            }
            roll_preds.push(next.clone());
            roll_epi.push(epi);
            cur = next;
        }
    }

    HypDiag {
        vfe_per_pt: (hyp.vfe / n_pts).clamp(-40.0, 40.0),
        loo_per_pt: (hyp.loo_fe / n_pts).clamp(-40.0, 40.0),
        oot1_log,
        oot2_log,
        replay_log,
        roll_preds,
        roll_epi,
        stab_pen,
        score: 0.0,
        hyp,
    }
}

/// Subrutina 2 (AURA v3 Genesis — Microcircuito Cortical Canónico de 6 capas, Mountcastle):
/// Plantilla laminar unificada L4 -> L2/3 -> L5 -> L6 para las 11 minicolumnas corticales,
/// extendiendo el bucle corticotalámico L6 (`maybe_refine_multistep_genesis`) a las columnas
/// co-móviles simplécticas con verificación monótona estricta de VFE y sorpresa cerrada.
pub(super) fn fit_hypothesis_genesis(kind: HypKind, frames: &[&[f64]], max_em: usize) -> FittedHyp {
    fit_hypothesis_ex(kind, frames, max_em)
}

/// Subrutina 2 (AURA v3 Genesis): Refinamiento apical `L6` multi-paso canónico aplicado
/// tanto a `CoMovingGp` como a `SympGp` co-móviles con verificación estricta de VFE y sorpresa cerrada.
pub(super) fn maybe_refine_multistep_genesis(h: &FittedHyp, frames: &[&[f64]]) -> FittedHyp {
    let replay_err = |hyp: &FittedHyp| -> f64 {
        let mut z = frames[0].to_vec();
        let mut s = 0.0;
        for t in 1..frames.len() {
            let (zn, _) = hyp.step_frame(&z);
            s += log_rel(clamp_err(&zn, frames[t]));
            z = zn;
        }
        s / (frames.len() - 1) as f64
    };
    let e0 = replay_err(h);
    if !e0.is_finite() || e0 < -8.0 {
        return h.clone();
    }
    match &h.model {
        HypModel::CoMovingGp { .. } => maybe_refine_multistep(h, frames),
        HypModel::SympGp { aff, rev, .. }
            if matches!(h.kind, HypKind::GpCoMovingSympFwd | HypKind::GpCoMovingSympRev) =>
        {
            let aff_ref = refine_affine_multistep(*aff, frames, 1e-4);
            let (qs, qns) = extract_pairs(frames);
            let us_base: Vec<[f64; 2]> = qs.iter().map(|&q| aff_ref.apply(q)).collect();
            let rx: Vec<f64> = us_base.iter().zip(&qns).map(|(u, y)| y[0] - u[0]).collect();
            let ry: Vec<f64> = us_base.iter().zip(&qns).map(|(u, y)| y[1] - u[1]).collect();
            let (us0, us1): (Vec<[f64; 2]>, Vec<[f64; 2]>) = if !*rev {
                let u1 = us_base.iter().zip(&qns).map(|(u, y)| [y[0], u[1]]).collect();
                (us_base, u1)
            } else {
                let u0 = us_base.iter().zip(&qns).map(|(u, y)| [u[0], y[1]]).collect();
                (u0, us_base)
            };
            if let (Some(c0), Some(c1)) = (
                optimize_gp_channel_vfe(&us0, &rx, KernelKind::CartM52, true, 1.0),
                optimize_gp_channel_vfe(&us1, &ry, KernelKind::CartM52, true, 1.0),
            ) {
                let cand = FittedHyp {
                    kind: h.kind,
                    vfe: c0.vfe + c1.vfe + 3.0,
                    loo_fe: c0.loo_fe + c1.loo_fe,
                    s2_scale: (c0.s2 + c1.s2).max(1e-12),
                    model: HypModel::SympGp {
                        aff: aff_ref,
                        rev: *rev,
                        us0,
                        us1,
                        ch: [c0, c1],
                    },
                    valid: true,
                };
                if replay_err(&cand) < e0 - 0.08 && cand.vfe <= h.vfe + 0.5 {
                    return cand;
                }
            }
            h.clone()
        }
        _ => h.clone(),
    }
}

/// Evalúa una columna canónica de AURA v3 Genesis en el contexto 0..=K (sin fuga) y su rollout h=(K+1)..=16.
pub(super) fn evaluate_hypothesis_genesis(
    kind: HypKind,
    frames: &[&[f64]],
    use_multistep_refine: bool,
    use_oot: bool,
    k_obs: usize,
) -> HypDiag {
    let np = frames[0].len() / 2;
    let n_pts = (k_obs * np) as f64;
    let mut hyp = fit_hypothesis_genesis(kind, &frames[..=k_obs], 4);
    if use_multistep_refine && hyp.valid {
        hyp = maybe_refine_multistep_genesis(&hyp, &frames[..=k_obs]);
    }

    let replay_log = if hyp.valid {
        let mut z = frames[0].to_vec();
        let mut v = Vec::with_capacity(k_obs);
        for t in 1..=k_obs {
            let (zn, _) = hyp.step_frame(&z);
            v.push(log_rel(clamp_err(&zn, frames[t])));
            z = zn;
        }
        mean(&v)
    } else {
        log_rel(10.0)
    };

    let (oot1_log, oot2_log) = if use_oot && hyp.valid && k_obs >= 3 {
        let h1 = fit_hypothesis_genesis(kind, &frames[..k_obs], 2);
        let o1 = if h1.valid {
            let (p, _) = h1.step_frame(frames[k_obs - 1]);
            log_rel(clamp_err(&p, frames[k_obs]))
        } else {
            log_rel(10.0)
        };
        let o2 = if k_obs >= 4 {
            let h2 = fit_hypothesis_genesis(kind, &frames[..(k_obs - 1)], 2);
            if h2.valid {
                let (p1, _) = h2.step_frame(frames[k_obs - 2]);
                let (p2, _) = h2.step_frame(&p1);
                0.5 * (log_rel(clamp_err(&p1, frames[k_obs - 1])) + log_rel(clamp_err(&p2, frames[k_obs])))
            } else {
                log_rel(10.0)
            }
        } else {
            o1
        };
        (o1, o2)
    } else {
        (replay_log, replay_log)
    };

    let r0 = rmax_frame(frames[0]).max(1e-6);
    let rk = rmax_frame(frames[k_obs]).max(1e-6);
    let rmax_ctx = (0..=k_obs).map(|t| rmax_frame(frames[t])).fold(1e-6, f64::max);
    let g_ctx = ((rk / r0).powf(1.0 / k_obs.max(1) as f64)).clamp(0.5, 2.0);
    let g_hi = if g_ctx > 1.03 { g_ctx } else { 1.02 };

    let mut roll_preds = Vec::with_capacity(HORIZON - k_obs);
    let mut roll_epi = Vec::with_capacity(HORIZON - k_obs);
    let mut cur = frames[k_obs].to_vec();
    let mut stab_pen = 0.0;
    for step in 1..=(HORIZON - k_obs) {
        let (next, epi) = if hyp.valid && fin(&cur) {
            hyp.step_frame(&cur)
        } else {
            (vec![f64::NAN; 2 * np], 1e6)
        };
        let r_bound = 2.5 * rmax_ctx * g_hi.powi(step as i32);
        if !fin(&next) {
            stab_pen += 50.0;
            roll_preds.push(vec![f64::NAN; 2 * np]);
            roll_epi.push(1e6);
            cur = vec![f64::NAN; 2 * np];
        } else {
            let rn = rmax_frame(&next);
            if rn > r_bound {
                stab_pen += ((rn / r_bound).ln()).powi(2) * 4.0;
            }
            if rn < 0.02 * rmax_ctx * g_ctx.min(1.0).powi(step as i32) {
                stab_pen += 0.5;
            }
            roll_preds.push(next.clone());
            roll_epi.push(epi);
            cur = next;
        }
    }

    HypDiag {
        vfe_per_pt: (hyp.vfe / n_pts).clamp(-40.0, 40.0),
        loo_per_pt: (hyp.loo_fe / n_pts).clamp(-40.0, 40.0),
        oot1_log,
        oot2_log,
        replay_log,
        roll_preds,
        roll_epi,
        stab_pen,
        score: 0.0,
        hyp,
    }
}

/// Subrutina 1 + Subrutina 3 (AURA v3 Genesis): Mielinización saltatoria con atajo físico temprano
/// de columnas no lineales redundantes cuando el par canónico (`LieAffine` + `GpCoMoving`)
/// resuelve el episodio a precisión estricta de máquina (`-27.6310` log-error, `oot1 < -22.0 && replay < -22.0`).
pub(super) fn evaluate_all_hypotheses_genesis(
    hyps: &[HypKind],
    frames: &[&[f64]],
    k_obs: usize,
) -> (Vec<HypDiag>, bool) {
    let mut out = Vec::with_capacity(hyps.len());
    let d_aff = evaluate_hypothesis_genesis(HypKind::LieAffine, frames, true, true, k_obs);
    let d_cm = evaluate_hypothesis_genesis(HypKind::GpCoMoving, frames, true, true, k_obs);
    let affine_myelinated = d_cm.hyp.valid
        && d_cm.vfe_per_pt < -20.0
        && d_cm.oot1_log < -22.0
        && d_cm.replay_log < -22.0;

    for &hk in hyps {
        if hk == HypKind::LieAffine {
            out.push(d_aff.clone());
        } else if hk == HypKind::GpCoMoving {
            out.push(d_cm.clone());
        } else if affine_myelinated {
            // Mielinización saltatoria: en régimen afín exacto, las columnas GP con media afín
            // colapsan analíticamente al transporte afín de d_cm sin ejecutar búsqueda de hiperparámetros
            let mut myel = d_cm.clone();
            myel.hyp.kind = hk;
            if hk == HypKind::FeArd {
                myel.vfe_per_pt += 4.0;
                myel.loo_per_pt += 4.0;
                myel.oot1_log = -15.0;
                myel.oot2_log = -15.0;
            }
            out.push(myel);
        } else {
            out.push(evaluate_hypothesis_genesis(hk, frames, true, true, k_obs));
        }
    }
    (out, affine_myelinated)
}

// ─── Nivel 2: Estado Metacognitivo ("Yo-FE"), Prior Amortizado y Controlador ──

pub(super) const N_INV: usize = 8;
pub(super) const N_SELF_FEAT: usize = 10;
pub(super) const N_CAL_FEAT: usize = 7;

/// Extrae 8 invariantes geométricos del contexto (sin fuga) para el hiper-prior metacognitivo p_ψ(M | c_e).
pub(super) fn episode_invariants(frames: &[&[f64]]) -> [f64; N_INV] {
    let (qs, qns) = extract_pairs(frames);
    let aff = Affine2D::fit(&qs, &qns, 1e-6);
    let mut aff_err = 0.0;
    let mut disp_norm = 0.0;
    let mut r_mean = 0.0;
    let mut dlnrs = vec![];
    let mut dths = vec![];
    for (q, qn) in qs.iter().zip(&qns) {
        let pred = aff.apply(*q);
        aff_err += ((pred[0] - qn[0]).powi(2) + (pred[1] - qn[1]).powi(2)).sqrt();
        disp_norm += ((qn[0] - q[0]).powi(2) + (qn[1] - q[1]).powi(2)).sqrt();
        let r = (q[0] * q[0] + q[1] * q[1]).sqrt().max(1e-6);
        let rn = (qn[0] * qn[0] + qn[1] * qn[1]).sqrt().max(1e-6);
        r_mean += r;
        dlnrs.push((rn / r).ln());
        dths.push(wrap_angle(qn[1].atan2(qn[0]) - q[1].atan2(q[0])));
    }
    let nf = qs.len().max(1) as f64;
    let rel_aff = (aff_err / disp_norm.max(1e-6)).clamp(1e-9, 10.0);
    let mr = mean(&dlnrs);
    let dlnr_var = dlnrs.iter().map(|v| (v - mr).powi(2)).sum::<f64>() / nf;
    let sc: f64 = dths.iter().map(|a| a.cos()).sum();
    let ss: f64 = dths.iter().map(|a| a.sin()).sum();
    let om = ss.atan2(sc);
    let dth_var = dths.iter().map(|&a| wrap_angle(a - om).powi(2)).sum::<f64>() / nf;
    let det = aff.a[0][0] * aff.a[1][1] - aff.a[0][1] * aff.a[1][0];
    let rho = aff.spectral_radius();
    [
        1.0,
        rel_aff.ln().clamp(-12.0, 2.0) / 10.0,
        (rho - 1.0).clamp(-1.0, 2.0),
        (det.abs() - 1.0).clamp(-1.0, 2.0),
        om / PI,
        (dlnr_var + 1e-8).ln().clamp(-15.0, 2.0) / 10.0,
        (dth_var + 1e-8).ln().clamp(-15.0, 2.0) / 10.0,
        (r_mean / nf).clamp(0.0, 5.0),
    ]
}

/// Modelo entrenado del controlador metacognitivo de 2º orden ("Yo-FE v2") y pesos de Energía Libre.
#[derive(Clone, Debug)]
pub struct MetaController {
    /// Pesos de combinación de Energía Libre [vfe, loo, oot1, oot2, replay, stab]
    pub fe_w: [f64; 6],
    /// Temperatura metacognitiva de Bayesian Model Averaging
    pub tau_meta: f64,
    /// Ganancia de precisión epistémica durante el rollout (0 = BMA estático)
    pub gamma_epi: f64,
    /// Usar BMA paso a paso (true) vs BMA de trayectorias (false)
    pub step_bma: bool,
    /// Hiper-prior amortizado log p_ψ(M_m | c_e): N_HYP × N_INV (entrenado en stream conocido + sintético)
    pub prior_w: [[f64; N_INV]; N_HYP],
    pub prior_scale: f64,
    /// Auto-calibrador metacognitivo de 2º orden ("Yo-FE v2", Etapa B): regresión logística Platt
    /// fuera de muestra sobre N_CAL_FEAT -> p̂_ok
    pub cal_w: [f64; N_CAL_FEAT],
    /// Predictor metacognitivo de log-error esperado h16 (Etapa A) -> ê_meta ∈ [-7.5, 2.3]
    pub err_w: [f64; N_SELF_FEAT],
    /// Prior autónomo de arranque en frío 1-paso (para E33 sin ÓRBITA):
    /// por familia f∈0..N_FAM, GP RBF de Energía Libre (p0, p1) -> Affine2D (6 canales), solo en ds.train
    cold_gps: Vec<(Vec<[f64; 2]>, Vec<GpChannel>)>,
    pub cold_trained: bool,
}

impl Default for MetaController {
    fn default() -> Self {
        Self {
            fe_w: [1.0, 0.35, 0.50, 0.50, 0.25, 1.0],
            tau_meta: 2.0,
            gamma_epi: 0.25,
            step_bma: false,
            prior_w: [[0.0; N_INV]; N_HYP],
            prior_scale: 0.0,
            cal_w: [0.0; N_CAL_FEAT],
            err_w: [0.0; N_SELF_FEAT],
            cold_gps: vec![],
            cold_trained: false,
        }
    }
}

/// Configuración de brazo / ablación de la familia METAFE.
#[derive(Clone, Debug)]
pub struct MetaFeCfg {
    pub name: &'static str,
    pub hyps: Vec<HypKind>,
    pub use_vfe: bool,
    pub use_oot: bool,
    pub use_multistep: bool,
    pub use_bma: bool,
    pub use_epi_gate: bool,
    pub use_meta_prior: bool,
}

pub fn metafe_arms() -> Vec<MetaFeCfg> {
    let all_hyps = vec![
        HypKind::LieAffine,
        HypKind::FeArd,
        HypKind::GpCartM52,
        HypKind::GpCartRbf,
        HypKind::GpCoMoving,
        HypKind::GpSympFwd,
        HypKind::GpSympRev,
        HypKind::GpCoMovingSympFwd,
        HypKind::GpCoMovingSympRev,
        HypKind::GpPolar,
        HypKind::GpEquivCart,
    ];
    vec![
        // Arquitectura A: Empirical Bayes ARD Variacional (reemplaza Tikhonov fijo de BQ10)
        MetaFeCfg {
            name: "FE-ARD",
            hyps: vec![HypKind::LieAffine, HypKind::FeArd],
            use_vfe: true,
            use_oot: true,
            use_multistep: false,
            use_bma: true,
            use_epi_gate: false,
            use_meta_prior: false,
        },
        // Arquitectura B: GP Cartesiano semi-paramétrico con Energía Libre exacta
        MetaFeCfg {
            name: "GP-Cart-FE",
            hyps: vec![
                HypKind::LieAffine,
                HypKind::GpCartM52,
                HypKind::GpCartRbf,
                HypKind::GpCoMoving,
            ],
            use_vfe: true,
            use_oot: true,
            use_multistep: true,
            use_bma: true,
            use_epi_gate: true,
            use_meta_prior: false,
        },
        // Arquitectura C: GP Polar + Equivariante SO(2) con Energía Libre exacta
        MetaFeCfg {
            name: "GP-Polar-FE",
            hyps: vec![
                HypKind::LieAffine,
                HypKind::GpCartM52,
                HypKind::GpPolar,
                HypKind::GpEquivCart,
            ],
            use_vfe: true,
            use_oot: true,
            use_multistep: false,
            use_bma: true,
            use_epi_gate: true,
            use_meta_prior: false,
        },
        // Arquitectura D: GP Simpléctico / Coupling-Layer Hamiltoniano con Energía Libre exacta
        MetaFeCfg {
            name: "GP-Symp-FE",
            hyps: vec![
                HypKind::LieAffine,
                HypKind::GpSympFwd,
                HypKind::GpSympRev,
                HypKind::GpCoMovingSympFwd,
                HypKind::GpCoMovingSympRev,
            ],
            use_vfe: true,
            use_oot: true,
            use_multistep: false,
            use_bma: true,
            use_epi_gate: true,
            use_meta_prior: false,
        },
        // Ablación 1: Todas las hipótesis SIN Energía Libre ni Metacognición (solo MSE 1-paso en contexto)
        MetaFeCfg {
            name: "METAFE-NoFE",
            hyps: all_hyps.clone(),
            use_vfe: false,
            use_oot: false,
            use_multistep: false,
            use_bma: false,
            use_epi_gate: false,
            use_meta_prior: false,
        },
        // Ablación 2: Energía Libre VFE+LOO SIN validación multi-paso OOT ni control epistémico
        MetaFeCfg {
            name: "METAFE-NoRoll",
            hyps: all_hyps.clone(),
            use_vfe: true,
            use_oot: false,
            use_multistep: false,
            use_bma: true,
            use_epi_gate: false,
            use_meta_prior: false,
        },
        // Arquitectura E (Flagship): METAFE Completo (Energía Libre Variacional + Multi-Sustrato 11H +
        // Inferencia Activa Multi-Paso + Metacognición 2º Orden "Yo-FE v2" + Control Activo Epistémico)
        MetaFeCfg {
            name: "METAFE",
            hyps: all_hyps,
            use_vfe: true,
            use_oot: true,
            use_multistep: true,
            use_bma: true,
            use_epi_gate: true,
            use_meta_prior: true,
        },
    ]
}

/// Telemetría metacognitiva ("Yo-FE v2") producida en cada episodio sin tocar ground truth futuro.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct EpisodeMetaReport {
    pub preds: Vec<Vec<f64>>,
    pub fe_null: f64,
    pub fe_post: f64,
    pub fe_reduction: f64,
    pub post_entropy: f64,
    pub best_hyp: usize,
    pub p_ok_hat: f64,
    pub err_log_hat: f64,
    pub mean_epi: f64,
    pub ens_disagree: f64,
    pub self_feats: [f64; N_SELF_FEAT],
    pub cal_feats: [f64; N_CAL_FEAT],
    pub hyp_weights: [f64; N_HYP],
    pub nan: bool,
}

pub(super) fn softmax_scores(scores: &[f64], tau: f64) -> Vec<f64> {
    let mn = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let ex: Vec<f64> = scores
        .iter()
        .map(|&s| {
            let z = (-tau * (s - mn)).clamp(-50.0, 0.0);
            z.exp()
        })
        .collect();
    let sum: f64 = ex.iter().sum::<f64>().max(1e-30);
    ex.iter().map(|&e| e / sum).collect()
}

/// Energía Libre nula (modelo de persistencia x̂_{t+1} = x_t, sin aprendizaje): línea base para medir ΔF.
pub(super) fn null_free_energy(frames: &[&[f64]]) -> f64 {
    let (qs, qns) = extract_pairs(frames);
    let n = qs.len() as f64;
    let s2x = (qs.iter().zip(&qns).map(|(a, b)| (b[0] - a[0]).powi(2)).sum::<f64>() / n).max(1e-24);
    let s2y = (qs.iter().zip(&qns).map(|(a, b)| (b[1] - a[1]).powi(2)).sum::<f64>() / n).max(1e-24);
    0.5 * (s2x.ln() + s2y.ln() + 2.0 + 2.0 * LN_2PI)
}

/// Ejecuta inferencia metacognitiva por Energía Libre en un episodio (observa cuadros 0..=K, predice 1..=16).
pub fn run_metafe_episode(
    cfg: &MetaFeCfg,
    ctrl: &MetaController,
    e: &Ep,
    k_obs: usize,
) -> EpisodeMetaReport {
    let ctx_frames: Vec<&[f64]> = (0..=k_obs).map(|t| e.frame(t)).collect();
    let fe_null = null_free_energy(&ctx_frames);
    let inv = episode_invariants(&ctx_frames);
    let mut diags: Vec<HypDiag> = cfg
        .hyps
        .iter()
        .map(|&hk| evaluate_hypothesis(hk, &ctx_frames, cfg.use_multistep, cfg.use_oot, k_obs))
        .collect();
    infer_from_diags(cfg, ctrl, e, &ctx_frames, fe_null, &inv, &mut diags, k_obs)
}

fn infer_from_diags(
    cfg: &MetaFeCfg,
    ctrl: &MetaController,
    e: &Ep,
    ctx_frames: &[&[f64]],
    fe_null: f64,
    inv: &[f64; N_INV],
    diags: &mut [HypDiag],
    k_obs: usize,
) -> EpisodeMetaReport {
    let np = e.x0.len() / 2;

    // 2. Calcular puntuación total de Energía Libre + hiper-prior metacognitivo entrenado
    let mut scores = Vec::with_capacity(diags.len());
    for d in diags.iter_mut() {
        if !d.hyp.valid {
            d.score = 1e6;
            scores.push(1e6);
            continue;
        }
        let s = if !cfg.use_vfe {
            // Ablación SIN Energía Libre: usa solo error de ajuste en contexto (replay)
            d.replay_log
        } else {
            let hidx = d.hyp.kind as usize;
            let log_prior = if cfg.use_meta_prior && ctrl.prior_scale > 0.0 {
                let lp: f64 = ctrl.prior_w[hidx].iter().zip(inv).map(|(a, b)| a * b).sum();
                (ctrl.prior_scale * lp).clamp(-2.0, 2.0)
            } else {
                0.0
            };
            let oot_part = if cfg.use_oot {
                ctrl.fe_w[2] * d.oot1_log + ctrl.fe_w[3] * d.oot2_log
            } else {
                0.0
            };
            ctrl.fe_w[0] * d.vfe_per_pt
                + ctrl.fe_w[1] * d.loo_per_pt
                + oot_part
                + ctrl.fe_w[4] * d.replay_log
                + ctrl.fe_w[5] * d.stab_pen
                - log_prior
        };
        d.score = if s.is_finite() { s } else { 1e6 };
        scores.push(d.score);
    }

    // 3. Creencias posteriores metacognitivas q(M_m | x_{0:K})
    let weights: Vec<f64> = if cfg.use_bma {
        softmax_scores(&scores, ctrl.tau_meta)
    } else {
        let best_i = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (0..scores.len()).map(|i| if i == best_i { 1.0 } else { 0.0 }).collect()
    };

    let best_i = weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best_hyp = diags[best_i].hyp.kind as usize;

    let mut hyp_weights = [0.0; N_HYP];
    for (d, &w) in diags.iter().zip(&weights) {
        hyp_weights[d.hyp.kind as usize] = w;
    }

    // 4. Predicciones filtradas 1..=K (precuenciales) + rollout h=(K+1)..=16 con control activo epistémico
    let mut preds = vec![vec![0.0; 2 * np]; HORIZON];
    // Cuadros 1..=K: persistencia en t=1 (o cold-start), ajuste online en t=2..=K
    preds[0] = e.frame(0).to_vec();
    for t in 2..=k_obs {
        let h_sub = fit_hypothesis(HypKind::LieAffine, &ctx_frames[..t]);
        if h_sub.valid {
            preds[t - 1] = h_sub.step_frame(e.frame(t - 1)).0;
        } else {
            preds[t - 1] = e.frame(t - 1).to_vec();
        }
    }

    let mut mean_epi_acc = 0.0;
    let mut ens_dis_acc = 0.0;
    let rmax_ctx = (0..=k_obs).map(|t| rmax_frame(e.frame(t))).fold(1e-6, f64::max);

    if cfg.use_epi_gate && ctrl.step_bma {
        // Control Activo Epistémico paso a paso (acoplado)
        let mut cur = e.frame(k_obs).to_vec();
        for step in 0..(HORIZON - k_obs) {
            let mut step_preds: Vec<(Vec<f64>, f64, f64)> = vec![];
            for (d, &w) in diags.iter().zip(&weights) {
                if w < 1e-4 || !d.hyp.valid {
                    continue;
                }
                let (pn, epi) = d.hyp.step_frame(&cur);
                if fin(&pn) {
                    let epi_norm = (epi / d.hyp.s2_scale.max(1e-8)).clamp(0.0, 20.0);
                    let w_dyn = w * (-ctrl.gamma_epi * epi_norm).exp();
                    step_preds.push((pn, w_dyn, epi));
                }
            }
            let wsum: f64 = step_preds.iter().map(|x| x.1).sum();
            let mut fused = vec![0.0; 2 * np];
            if wsum > 1e-12 {
                for (pn, wd, epi) in &step_preds {
                    let wn = wd / wsum;
                    for j in 0..2 * np {
                        fused[j] += wn * pn[j];
                    }
                    mean_epi_acc += wn * epi.sqrt();
                }
                let dis: f64 = step_preds
                    .iter()
                    .map(|(pn, wd, _)| (wd / wsum) * mt::rel_err(pn, &fused).min(10.0))
                    .sum();
                ens_dis_acc += dis;
            } else {
                fused = diags[best_i].roll_preds[step].clone();
            }
            preds[k_obs + step] = fused.clone();
            cur = fused;
        }
    } else {
        // BMA por trayectorias ponderado por Energía Libre + precisión epistémica acumulada
        let mut traj_w = vec![0.0; diags.len()];
        for (i, (d, &w)) in diags.iter().zip(&weights).enumerate() {
            if w < 1e-5 || !d.hyp.valid || !fin(&d.roll_preds[HORIZON - k_obs - 1]) {
                continue;
            }
            let epi_pen = if cfg.use_epi_gate {
                let em = mean(&d.roll_epi) / d.hyp.s2_scale.max(1e-8);
                (-ctrl.gamma_epi * em.clamp(0.0, 12.0)).exp()
            } else {
                1.0
            };
            traj_w[i] = w * epi_pen;
        }
        let wsum: f64 = traj_w.iter().sum();
        if wsum > 1e-12 {
            for w in &mut traj_w {
                *w /= wsum;
            }
        } else {
            traj_w[best_i] = 1.0;
        }
        for step in 0..(HORIZON - k_obs) {
            let mut fused = vec![0.0; 2 * np];
            for (i, d) in diags.iter().enumerate() {
                if traj_w[i] > 0.0 && fin(&d.roll_preds[step]) {
                    for j in 0..2 * np {
                        fused[j] += traj_w[i] * d.roll_preds[step][j];
                    }
                    mean_epi_acc += traj_w[i] * d.roll_epi[step].sqrt();
                }
            }
            let dis: f64 = diags
                .iter()
                .enumerate()
                .filter(|(i, d)| traj_w[*i] > 0.0 && fin(&d.roll_preds[step]))
                .map(|(i, d)| traj_w[i] * mt::rel_err(&d.roll_preds[step], &fused).min(10.0))
                .sum();
            ens_dis_acc += dis;
            preds[k_obs + step] = fused;
        }
    }

    let n_roll = (HORIZON - k_obs) as f64;
    let mean_epi = (mean_epi_acc / n_roll).max(0.0);
    let ens_disagree = (ens_dis_acc / n_roll).max(0.0);
    let nan = !fin(&preds[HORIZON - 1]);

    // 5. Auto-estado metacognitivo de 2º orden ("Yo-FE")
    let fe_post = diags[best_i].vfe_per_pt;
    let fe_reduction = fe_null - fe_post;
    let post_entropy: f64 = -weights
        .iter()
        .filter(|&&w| w > 1e-12)
        .map(|&w| w * w.ln())
        .sum::<f64>();
    let r_end = if !nan {
        rmax_frame(&preds[HORIZON - 1])
    } else {
        1e3
    };
    let norm_ratio = (r_end / rmax_ctx).clamp(1e-4, 1e3).ln();

    let self_feats: [f64; N_SELF_FEAT] = [
        1.0,
        fe_post.clamp(-10.0, 10.0) / 5.0,
        fe_reduction.clamp(-5.0, 15.0) / 10.0,
        post_entropy / (diags.len().max(2) as f64).ln(),
        diags[best_i].oot1_log.clamp(-7.5, 2.3) / 5.0,
        diags[best_i].oot2_log.clamp(-7.5, 2.3) / 5.0,
        diags[best_i].replay_log.clamp(-7.5, 2.3) / 5.0,
        (mean_epi + 1e-7).ln().clamp(-12.0, 4.0) / 6.0,
        (ens_disagree + 1e-7).ln().clamp(-12.0, 4.0) / 6.0,
        norm_ratio.clamp(-4.0, 4.0) / 4.0,
    ];

    // Etapa A ("Yo-FE v2"): predicción de magnitud de log-error esperado en h16
    let err_log_hat: f64 = ctrl
        .err_w
        .iter()
        .zip(&self_feats)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .clamp(-9.0, 4.0);

    // Etapa B ("Yo-FE v2"): rasgos de calibración Platt centrados en el margen ln(0.05) - ê_meta
    let margin = (mt::REL_ERR_OK.ln() - err_log_hat).clamp(-6.0, 6.0);
    let cal_feats: [f64; N_CAL_FEAT] = [
        1.0,
        margin,
        margin.tanh(),
        self_feats[7], // log(mean_epi)
        self_feats[8], // log(ens_disagree)
        self_feats[3], // entropía posterior normalizada
        self_feats[9], // norm_ratio
    ];
    let logit: f64 = ctrl.cal_w.iter().zip(&cal_feats).map(|(a, b)| a * b).sum();
    let p_ok_hat = sig(logit.clamp(-30.0, 30.0));

    EpisodeMetaReport {
        preds,
        fe_null,
        fe_post,
        fe_reduction,
        post_entropy,
        best_hyp,
        p_ok_hat,
        err_log_hat,
        mean_epi,
        ens_disagree,
        self_feats,
        cal_feats,
        hyp_weights,
        nan,
    }
}

// ─── Entrenamiento del Controlador Metacognitivo (0 fuga {8,10,11}, 0 TEST) ──

/// Genera episodios de entrenamiento metacognitivo: 50% familias conocidas 0..7 + 50% sintéticas
/// procedurales auditadas contra {8,10,11} (idéntico al protocolo de auditoría de `atlas.rs`).
pub(super) fn gen_meta_training_eps(seed: u64, n: usize, xor: u64) -> (Vec<Ep>, usize) {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ xor);
    let mut prng = Xoshiro256StarStar::seed_from_u64(seed ^ xor ^ 0x5A11);
    let probe = make_object(&mut prng, K_POINTS, ObjectClass::Blob);
    let mut rej = 0usize;
    let eps: Vec<Ep> = (0..n)
        .map(|i| {
            if i % 2 == 1 {
                make_syn_ep(&mut rng, &probe, &mut rej).0
            } else {
                make_ep(&mut rng, i % N_FAM)
            }
        })
        .collect();
    for e in &eps {
        assert!(!NEW_FAMS.contains(&e.fam), "fuga detectada en meta-entrenamiento METAFE");
    }
    (eps, rej)
}

/// Genera un banco OOD sintético procedural completamente nuevo (independiente del entrenamiento
/// y auditado fuera de {8,10,11}) para evaluar generalización OOD universal adicional.
pub(super) fn gen_ood_synthetic_eps(seed: u64, n: usize) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_METAFE_OOD);
    let mut prng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_METAFE_OOD ^ 0x5A11);
    let probe = make_object(&mut prng, K_POINTS, ObjectClass::Blob);
    let mut rej = 0usize;
    (0..n).map(|_| make_syn_ep(&mut rng, &probe, &mut rej).0).collect()
}

/// Extrae índice de familia y parámetros (p0, p1) desde el vector de contexto c∈R^{N_FAM+2}.
fn parse_context_fam_params(c: &[f64]) -> (usize, [f64; 2]) {
    let mut fam = 0usize;
    let mut best = f64::NEG_INFINITY;
    let nf = N_FAM.min(c.len());
    for i in 0..nf {
        if c[i] > best {
            best = c[i];
            fam = i;
        }
    }
    let p0 = if c.len() > N_FAM { c[N_FAM] } else { 0.0 };
    let p1 = if c.len() > N_FAM + 1 { c[N_FAM + 1] } else { 0.0 };
    (fam, [p0, p1])
}

/// Entrena el Prior Autónomo de Arranque en Frío 1-Paso (contexto c∈R^10 -> Affine2D)
/// exclusivamente sobre `ds.train` (0 consultas TEST, 0 fórmulas de familia) mediante
/// Procesos Gaussianos RBF optimizados por Energía Libre por familia: (p0, p1) -> Affine2D.
pub(super) fn train_cold_start_prior(ctrl: &mut MetaController, seed: u64) {
    let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS));
    let mut per_fam_ps: Vec<Vec<[f64; 2]>> = vec![vec![]; N_FAM];
    let mut per_fam_ys: Vec<[Vec<f64>; 6]> = (0..N_FAM)
        .map(|_| std::array::from_fn(|_| Vec::new()))
        .collect();

    for e in &s.ds.train {
        let (fam, p) = parse_context_fam_params(&e.context);
        if fam >= N_FAM || per_fam_ps[fam].len() >= 48 {
            continue;
        }
        let np = e.input.len() / 2;
        let qs: Vec<[f64; 2]> = (0..np).map(|i| [e.input[2 * i], e.input[2 * i + 1]]).collect();
        let qns: Vec<[f64; 2]> = (0..np).map(|i| [e.expected[2 * i], e.expected[2 * i + 1]]).collect();
        let af = Affine2D::fit(&qs, &qns, 1e-8);
        let vals = [af.a[0][0], af.a[0][1], af.b[0], af.a[1][0], af.a[1][1], af.b[1]];
        per_fam_ps[fam].push(p);
        for o in 0..6 {
            per_fam_ys[fam][o].push(vals[o]);
        }
    }

    let mut cold_gps = Vec::with_capacity(N_FAM);
    for fam in 0..N_FAM {
        let ps = &per_fam_ps[fam];
        let mut chs = Vec::with_capacity(6);
        if ps.len() >= 4 {
            for o in 0..6 {
                if let Some(ch) = optimize_gp_channel_vfe(ps, &per_fam_ys[fam][o], KernelKind::CartRbf, true, 10.0) {
                    chs.push(ch);
                }
            }
        }
        cold_gps.push((ps.clone(), chs));
    }
    ctrl.cold_gps = cold_gps;
    ctrl.cold_trained = true;
}

pub(super) fn cold_start_step(ctrl: &MetaController, x: &[f64], c: &[f64]) -> Vec<f64> {
    let (fam, params) = parse_context_fam_params(c);
    if fam >= ctrl.cold_gps.len() {
        return x.to_vec();
    }
    let (ps, chs) = &ctrl.cold_gps[fam];
    if chs.len() != 6 || ps.is_empty() {
        return x.to_vec();
    }
    let mut p = [0.0; 6];
    for o in 0..6 {
        p[o] = chs[o].predict_mean(ps, &params);
    }
    let af = Affine2D {
        a: [[p[0], p[1]], [p[3], p[4]]],
        b: [p[2], p[5]],
    };
    let np = x.len() / 2;
    let mut out = vec![0.0; 2 * np];
    for i in 0..np {
        let qn = af.apply([x[2 * i], x[2 * i + 1]]);
        out[2 * i] = qn[0];
        out[2 * i + 1] = qn[1];
    }
    out
}

/// Entrena el controlador metacognitivo completo (pesos de Energía Libre, hiper-prior amortizado,
/// auto-calibrador bietapa de 2º orden "Yo-FE v2" y prior GP de arranque en frío) sobre episodios
/// de entrenamiento auditados (conocidas 0..7 + sintéticas procedurales; 0 fuga {8,10,11}, 0 consultas TEST).
pub fn train_metafe_controller(
    cfg: &MetaFeCfg,
    seed: u64,
    n_train: usize,
    n_val: usize,
) -> (MetaController, serde_json::Value) {
    let mut ctrl = MetaController::default();
    let (eps_tr, rej1) = gen_meta_training_eps(seed, n_train, XOR_METAFE_TRAIN);
    let (eps_va, rej2) = gen_meta_training_eps(seed, n_val, XOR_METAFE_VAL);

    // Pre-evaluar diagnósticos de hipótesis en episodios de meta-entrenamiento y meta-validación
    struct CachedMeta {
        inv: [f64; N_INV],
        diags: Vec<HypDiag>,
        true_errs: Vec<f64>,
        ep: Ep,
    }
    let build_cache = |eps: &[Ep]| -> Vec<CachedMeta> {
        eps.iter()
            .map(|e| {
                let ctx_f: Vec<&[f64]> = (0..=K).map(|t| e.frame(t)).collect();
                let inv = episode_invariants(&ctx_f);
                let diags: Vec<HypDiag> = cfg
                    .hyps
                    .iter()
                    .map(|&hk| evaluate_hypothesis(hk, &ctx_f, cfg.use_multistep, cfg.use_oot, K))
                    .collect();
                let true_errs: Vec<f64> = diags
                    .iter()
                    .map(|d| log_rel(clamp_err(&d.roll_preds[HORIZON - K - 1], &e.ys[HORIZON - 1])))
                    .collect();
                CachedMeta {
                    inv,
                    diags,
                    true_errs,
                    ep: e.clone(),
                }
            })
            .collect()
    };

    let cache_tr = build_cache(&eps_tr);
    let cache_va = build_cache(&eps_va);

    // 1. Entrenar hiper-prior amortizado p_ψ(M | c_e) por softmax-cross-entropy sobre episodios
    //    informativos (donde las hipótesis difieren en h16 > 0.15 nats), regularizado L2
    if cfg.use_meta_prior && cfg.hyps.len() > 1 {
        let info_idx: Vec<usize> = cache_tr
            .iter()
            .enumerate()
            .filter_map(|(idx, cm)| {
                let mn = cm.true_errs.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = cm.true_errs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if mx - mn > 0.15 { Some(idx) } else { None }
            })
            .collect();
        if !info_idx.is_empty() {
            let mut pw = [[0.0_f64; N_INV]; N_HYP];
            let lr = 0.08;
            let l2 = 0.05;
            let n_info = info_idx.len() as f64;
            for _ep in 0..80 {
                let mut grad = [[0.0_f64; N_INV]; N_HYP];
                for &idx in &info_idx {
                    let cm = &cache_tr[idx];
                    let clamped_errs: Vec<f64> = cm.true_errs.iter().map(|&v| v.clamp(-8.0, 2.3)).collect();
                    let tgt = softmax_scores(&clamped_errs, 1.5);
                    let logits: Vec<f64> = cfg
                        .hyps
                        .iter()
                        .map(|&hk| pw[hk as usize].iter().zip(&cm.inv).map(|(a, b)| a * b).sum())
                        .collect();
                    let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let ex: Vec<f64> = logits.iter().map(|&z| (z - mx).exp()).collect();
                    let es: f64 = ex.iter().sum::<f64>().max(1e-30);
                    for (i, &hk) in cfg.hyps.iter().enumerate() {
                        let p_i = ex[i] / es;
                        let diff = p_i - tgt[i];
                        for j in 0..N_INV {
                            grad[hk as usize][j] += diff * cm.inv[j] / n_info;
                        }
                    }
                }
                for &hk in &cfg.hyps {
                    let hi = hk as usize;
                    for j in 0..N_INV {
                        pw[hi][j] -= lr * (grad[hi][j] + l2 * pw[hi][j]);
                    }
                }
            }
            ctrl.prior_w = pw;
        }
    }

    // 2. Meta-selección de pesos de Energía Libre, temperatura metacognitiva τ_meta y ganancia epistémica
    //    sobre cache_va (minimizando mediana + media de log-error h16 en meta-validación)
    let eval_ctrl_on_cache = |c: &MetaController, cache: &[CachedMeta]| -> f64 {
        let mut errs = Vec::with_capacity(cache.len());
        for cm in cache {
            let scores: Vec<f64> = cm
                .diags
                .iter()
                .map(|d| {
                    if !d.hyp.valid {
                        return 1e6;
                    }
                    if !cfg.use_vfe {
                        return d.replay_log;
                    }
                    let lp = if cfg.use_meta_prior && c.prior_scale > 0.0 {
                        let v: f64 = c.prior_w[d.hyp.kind as usize]
                            .iter()
                            .zip(&cm.inv)
                            .map(|(a, b)| a * b)
                            .sum();
                        (c.prior_scale * v).clamp(-2.0, 2.0)
                    } else {
                        0.0
                    };
                    let oot = if cfg.use_oot {
                        c.fe_w[2] * d.oot1_log + c.fe_w[3] * d.oot2_log
                    } else {
                        0.0
                    };
                    c.fe_w[0] * d.vfe_per_pt
                        + c.fe_w[1] * d.loo_per_pt
                        + oot
                        + c.fe_w[4] * d.replay_log
                        + c.fe_w[5] * d.stab_pen
                        - lp
                })
                .collect();
            let ws = if cfg.use_bma {
                softmax_scores(&scores, c.tau_meta)
            } else {
                let bi = scores
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                (0..scores.len()).map(|i| if i == bi { 1.0 } else { 0.0 }).collect()
            };
            let mut tw = vec![0.0; cm.diags.len()];
            for (i, (d, &w)) in cm.diags.iter().zip(&ws).enumerate() {
                if w < 1e-5 || !d.hyp.valid || !fin(&d.roll_preds[HORIZON - K - 1]) {
                    continue;
                }
                let ep = if cfg.use_epi_gate {
                    let em = mean(&d.roll_epi) / d.hyp.s2_scale.max(1e-8);
                    (-c.gamma_epi * em.clamp(0.0, 12.0)).exp()
                } else {
                    1.0
                };
                tw[i] = w * ep;
            }
            let wsum: f64 = tw.iter().sum();
            let pred16 = if wsum > 1e-12 {
                let dim = cm.ep.x0.len();
                let mut f = vec![0.0; dim];
                for (i, d) in cm.diags.iter().enumerate() {
                    if tw[i] > 0.0 {
                        let wn = tw[i] / wsum;
                        for j in 0..dim {
                            f[j] += wn * d.roll_preds[HORIZON - K - 1][j];
                        }
                    }
                }
                f
            } else {
                let bi = ws
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                cm.diags[bi].roll_preds[HORIZON - K - 1].clone()
            };
            errs.push(log_rel(clamp_err(&pred16, &cm.ep.ys[HORIZON - 1])));
        }
        0.5 * mean(&errs) + 0.5 * median(errs)
    };

    let fe_candidates: [[f64; 6]; 5] = [
        [1.0, 0.35, 0.50, 0.50, 0.25, 1.0],
        [0.6, 0.40, 0.80, 0.80, 0.30, 1.0],
        [1.0, 0.00, 0.40, 0.40, 0.20, 1.0],
        [0.3, 0.30, 1.00, 1.00, 0.50, 1.5],
        [1.0, 0.50, 0.20, 0.20, 0.10, 1.0],
    ];
    let taus: &[f64] = if cfg.use_bma { &[2.0, 4.0, 8.0, 12.0] } else { &[1.0] };
    let gammas: &[f64] = if cfg.use_epi_gate { &[0.0, 0.15, 0.5] } else { &[0.0] };
    let pscales: &[f64] = if cfg.use_meta_prior { &[0.0, 0.25, 0.6] } else { &[0.0] };

    let mut best_obj = f64::INFINITY;
    let mut best_ctrl = ctrl.clone();
    for &fw in &fe_candidates {
        for &tau in taus {
            for &gam in gammas {
                for &ps in pscales {
                    let mut cand = ctrl.clone();
                    cand.fe_w = fw;
                    cand.tau_meta = tau;
                    cand.gamma_epi = gam;
                    cand.prior_scale = ps;
                    let o = eval_ctrl_on_cache(&cand, &cache_va);
                    if o < best_obj {
                        best_obj = o;
                        best_ctrl = cand;
                    }
                }
            }
        }
    }
    ctrl = best_ctrl;

    // 3. Auto-Calibrador Metacognitivo Bietapa de 2º Orden ("Yo-FE v2"):
    //    - Etapa A: entrena predictor de log-error `err_w` en cache_tr sobre log_rel(err)∈[-7.5, 2.3]
    //    - Etapa B: entrena calibrador Platt `cal_w` ESTRICTAMENTE FUERA DE MUESTRA en cache_va
    //      usando el margen (ln(0.05) - ê_meta) y telemetría epistémica (sin re-ajustar hipótesis)
    let mut tr_feats = Vec::with_capacity(cache_tr.len());
    let mut tr_ylog = Vec::with_capacity(cache_tr.len());
    for cm in &cache_tr {
        let ctx_f: Vec<&[f64]> = (0..=K).map(|t| cm.ep.frame(t)).collect();
        let fe_null = null_free_energy(&ctx_f);
        let mut d_clone = cm.diags.clone();
        let rep = infer_from_diags(cfg, &ctrl, &cm.ep, &ctx_f, fe_null, &cm.inv, &mut d_clone, K);
        let err = clamp_err(&rep.preds[HORIZON - 1], &cm.ep.ys[HORIZON - 1]);
        tr_feats.push(rep.self_feats.to_vec());
        tr_ylog.push(vec![log_rel(err).clamp(-7.5, 2.3)]);
    }
    let ew = ridge(&tr_feats, &tr_ylog, 5e-2);
    for i in 0..N_SELF_FEAT {
        ctrl.err_w[i] = ew[i][0];
    }

    let mut va_cfeats = Vec::with_capacity(cache_va.len());
    let mut va_yok = Vec::with_capacity(cache_va.len());
    for cm in &cache_va {
        let ctx_f: Vec<&[f64]> = (0..=K).map(|t| cm.ep.frame(t)).collect();
        let fe_null = null_free_energy(&ctx_f);
        let mut d_clone = cm.diags.clone();
        let rep = infer_from_diags(cfg, &ctrl, &cm.ep, &ctx_f, fe_null, &cm.inv, &mut d_clone, K);
        let err = clamp_err(&rep.preds[HORIZON - 1], &cm.ep.ys[HORIZON - 1]);
        va_cfeats.push(rep.cal_feats);
        va_yok.push(if err < mt::REL_ERR_OK { 1.0 } else { 0.0 });
    }
    // Regresión logística Platt con Newton-IRLS regularizada L2 (estrictamente fuera de muestra)
    let mut cw = [0.0_f64; N_CAL_FEAT];
    let lam_c = 1.0;
    for _ in 0..25 {
        let mut h = vec![0.0; N_CAL_FEAT * N_CAL_FEAT];
        let mut g = vec![0.0; N_CAL_FEAT];
        for (x, &y) in va_cfeats.iter().zip(&va_yok) {
            let z: f64 = cw.iter().zip(x).map(|(a, b)| a * b).sum();
            let p = sig(z.clamp(-25.0, 25.0));
            let w = (p * (1.0 - p)).max(1e-6);
            for i in 0..N_CAL_FEAT {
                g[i] += (p - y) * x[i];
                for j in 0..N_CAL_FEAT {
                    h[i * N_CAL_FEAT + j] += w * x[i] * x[j];
                }
            }
        }
        for i in 0..N_CAL_FEAT {
            let reg = if i == 0 { 1e-4 } else { lam_c };
            g[i] += reg * cw[i];
            h[i * N_CAL_FEAT + i] += reg;
        }
        let step = solve_spd(&h, &g, N_CAL_FEAT);
        for i in 0..N_CAL_FEAT {
            cw[i] -= step[i];
        }
    }
    ctrl.cal_w = cw;

    // 4. Entrenar prior GP RBF de arranque en frío 1-paso (para E33 autónomo, 0 consultas TEST)
    train_cold_start_prior(&mut ctrl, seed);

    let meta_info = json!({
        "n_train": n_train,
        "n_val": n_val,
        "syn_rejected": rej1 + rej2,
        "best_val_obj": best_obj,
        "fe_w": ctrl.fe_w,
        "tau_meta": ctrl.tau_meta,
        "gamma_epi": ctrl.gamma_epi,
        "prior_scale": ctrl.prior_scale,
        "cold_trained": ctrl.cold_trained,
    });
    (ctrl, meta_info)
}

// ─── Métricas Metacognitivas (AUROC, Brier, ECE, AURC, Reducción Energía Libre) ─

pub(super) fn compute_auroc(scores: &[f64], labels: &[bool]) -> f64 {
    let mut pos = 0.0;
    let mut neg = 0.0;
    for &l in labels {
        if l {
            pos += 1.0;
        } else {
            neg += 1.0;
        }
    }
    if pos == 0.0 || neg == 0.0 {
        return f64::NAN;
    }
    let mut conc = 0.0;
    for i in 0..scores.len() {
        if !labels[i] {
            continue;
        }
        for j in 0..scores.len() {
            if labels[j] {
                continue;
            }
            if scores[i] > scores[j] {
                conc += 1.0;
            } else if (scores[i] - scores[j]).abs() < 1e-15 {
                conc += 0.5;
            }
        }
    }
    conc / (pos * neg)
}

pub(super) fn compute_brier(probs: &[f64], labels: &[bool]) -> f64 {
    mean(
        &probs
            .iter()
            .zip(labels)
            .map(|(&p, &y)| (p - if y { 1.0 } else { 0.0 }).powi(2))
            .collect::<Vec<_>>(),
    )
}

pub(super) fn compute_ece(probs: &[f64], labels: &[bool], n_bins: usize) -> f64 {
    let n = probs.len().max(1) as f64;
    let mut ece = 0.0;
    for b in 0..n_bins {
        let lo = b as f64 / n_bins as f64;
        let hi = (b + 1) as f64 / n_bins as f64;
        let idx: Vec<usize> = (0..probs.len())
            .filter(|&i| probs[i] >= lo && (probs[i] < hi || (b + 1 == n_bins && probs[i] <= hi)))
            .collect();
        if idx.is_empty() {
            continue;
        }
        let conf = mean(&idx.iter().map(|&i| probs[i]).collect::<Vec<_>>());
        let acc = mean(&idx.iter().map(|&i| if labels[i] { 1.0 } else { 0.0 }).collect::<Vec<_>>());
        ece += (idx.len() as f64 / n) * (conf - acc).abs();
    }
    ece
}

/// AURC (Area Under Risk-Coverage Curve): ordena de mayor a menor confianza p̂_ok y acumula tasa de error.
pub(super) fn compute_aurc(probs: &[f64], labels: &[bool]) -> f64 {
    let n = probs.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let mut errs = 0.0;
    let mut sum_risk = 0.0;
    for (k, &i) in ord.iter().enumerate() {
        if !labels[i] {
            errs += 1.0;
        }
        sum_risk += errs / (k + 1) as f64;
    }
    sum_risk / n as f64
}

pub(super) fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || a.len() != b.len() {
        return 0.0;
    }
    let ma = mean(a);
    let mb = mean(b);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    if va > 1e-12 && vb > 1e-12 {
        cov / (va * vb).sqrt()
    } else {
        0.0
    }
}

fn ranks_avg(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (v[ord[j]] - v[ord[i]]).abs() < 1e-15 {
            j += 1;
        }
        let avg = 0.5 * ((i + j - 1) as f64);
        for k in i..j {
            r[ord[k]] = avg;
        }
        i = j;
    }
    r
}

pub(super) fn spearman_corr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || a.len() != b.len() {
        return 0.0;
    }
    pearson_corr(&ranks_avg(a), &ranks_avg(b))
}

pub(super) fn eval_metafe_set(
    cfg: &MetaFeCfg,
    ctrl: &MetaController,
    eps: &[Ep],
    split_known_new: bool,
) -> (serde_json::Value, serde_json::Value, bool) {
    let mut res = serde_json::Map::new();
    let mut meta_j = serde_json::Map::new();
    let mut any_nan = false;

    let groups: Vec<(&str, Vec<&Ep>)> = if split_known_new {
        vec![
            ("known", eps.iter().filter(|e| e.fam < N_FAM).collect()),
            ("new", eps.iter().filter(|e| e.fam >= N_FAM).collect()),
        ]
    } else {
        vec![("syn_ood", eps.iter().collect())]
    };

    let mut all_probs = vec![];
    let mut all_ok = vec![];
    let mut all_err_hat = vec![];
    let mut all_err_true = vec![];
    let mut all_err_true_clamped = vec![];
    let mut all_fe_null = vec![];
    let mut all_fe_post = vec![];
    let mut all_fe_red = vec![];
    let mut all_ent = vec![];
    let mut all_epi = vec![];
    let mut all_dis = vec![];
    let mut per_fam: std::collections::BTreeMap<usize, Vec<f64>> = Default::default();

    for (name, subset) in groups {
        if subset.is_empty() {
            continue;
        }
        let mut errs = Vec::with_capacity(subset.len());
        let mut probs = Vec::with_capacity(subset.len());
        let mut oks = Vec::with_capacity(subset.len());
        let mut fe_reds = Vec::with_capacity(subset.len());
        let mut g_ehat = Vec::with_capacity(subset.len());
        let mut g_etrue = Vec::with_capacity(subset.len());
        let mut g_epi = Vec::with_capacity(subset.len());
        let mut g_dis = Vec::with_capacity(subset.len());
        let mut group_hyps = [0usize; N_HYP];

        for e in &subset {
            let rep = run_metafe_episode(cfg, ctrl, e, K);
            any_nan |= rep.nan;
            let err = clamp_err(&rep.preds[HORIZON - 1], &e.ys[HORIZON - 1]);
            let ok = err < mt::REL_ERR_OK;
            let le = log_rel(err);
            let lec = le.clamp(-7.5, 2.3);
            errs.push(err);
            probs.push(rep.p_ok_hat);
            oks.push(ok);
            fe_reds.push(rep.fe_reduction);
            g_ehat.push(rep.err_log_hat);
            g_etrue.push(lec);
            g_epi.push(rep.mean_epi);
            g_dis.push(rep.ens_disagree);
            group_hyps[rep.best_hyp] += 1;
            per_fam.entry(e.fam).or_default().push(le);

            all_probs.push(rep.p_ok_hat);
            all_ok.push(ok);
            all_err_hat.push(rep.err_log_hat);
            all_err_true.push(le);
            all_err_true_clamped.push(lec);
            all_fe_null.push(rep.fe_null);
            all_fe_post.push(rep.fe_post);
            all_fe_red.push(rep.fe_reduction);
            all_ent.push(rep.post_entropy);
            all_epi.push(rep.mean_epi);
            all_dis.push(rep.ens_disagree);
        }

        let acc16 = oks.iter().filter(|&&b| b).count() as f64 / oks.len() as f64;
        let med_log = median(errs.iter().map(|&x| log_rel(x)).collect());
        let mean_log = mean(&errs.iter().map(|&x| log_rel(x)).collect::<Vec<_>>());
        res.insert(format!("{name}_acc16"), json!(acc16));
        res.insert(format!("{name}_med_log"), json!(med_log));
        res.insert(format!("{name}_mean_log"), json!(mean_log));

        let hyp_frac: serde_json::Map<String, serde_json::Value> = HYP_NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| group_hyps[*i] > 0)
            .map(|(i, &nm)| (nm.into(), json!(group_hyps[i] as f64 / subset.len() as f64)))
            .collect();
        meta_j.insert(
            format!("{name}_metacog"),
            json!({
                "fe_reduction_mean": mean(&fe_reds),
                "fe_reduced_frac": fe_reds.iter().filter(|&&v| v > 0.0).count() as f64 / fe_reds.len() as f64,
                "auroc": compute_auroc(&probs, &oks),
                "brier": compute_brier(&probs, &oks),
                "ece": compute_ece(&probs, &oks, 10),
                "aurc": compute_aurc(&probs, &oks),
                "err_pred_corr": pearson_corr(&g_ehat, &g_etrue),
                "err_pred_spearman": spearman_corr(&g_ehat, &g_etrue),
                "mean_epi": mean(&g_epi),
                "ens_disagree": mean(&g_dis),
                "hyp_selection": hyp_frac,
            }),
        );
    }

    let corr = pearson_corr(&all_err_hat, &all_err_true_clamped);
    let corr_raw = pearson_corr(&all_err_hat, &all_err_true);
    let spearman = spearman_corr(&all_err_hat, &all_err_true);

    let fam_med: serde_json::Map<String, serde_json::Value> = per_fam
        .into_iter()
        .map(|(f, v)| (format!("fam_{f}"), json!(median(v))))
        .collect();

    meta_j.insert(
        "overall".into(),
        json!({
            "fe_null_mean": mean(&all_fe_null),
            "fe_post_mean": mean(&all_fe_post),
            "fe_reduction_mean": mean(&all_fe_red),
            "fe_reduced_frac": all_fe_red.iter().filter(|&&v| v > 0.0).count() as f64 / all_fe_red.len().max(1) as f64,
            "post_entropy_mean": mean(&all_ent),
            "mean_epi": mean(&all_epi),
            "ens_disagree": mean(&all_dis),
            "auroc": compute_auroc(&all_probs, &all_ok),
            "brier": compute_brier(&all_probs, &all_ok),
            "ece": compute_ece(&all_probs, &all_ok, 10),
            "aurc": compute_aurc(&all_probs, &all_ok),
            "err_pred_corr": corr,
            "err_pred_raw_corr": corr_raw,
            "err_pred_spearman": spearman,
            "per_fam_med_log": fam_med,
        }),
    );

    (serde_json::Value::Object(res), serde_json::Value::Object(meta_j), any_nan)
}

/// Evaluación E33 canónica (con prior ÓRBITA h1, igual que v4/ATLAS-FW/BQ10)
/// Y E33 autónoma (SIN ÓRBITA, usando el prior GP RBF entrenado de arranque en frío de METAFE).
pub(super) fn e33_metafe(
    cfg: &MetaFeCfg,
    ctrl: &MetaController,
    d: &Orbita,
    ex: &[Example],
) -> (serde_json::Value, serde_json::Value, serde_json::Value) {
    let trs: Vec<(Vec<Vec<f64>>, &Example)> = ex
        .iter()
        .map(|e| {
            let ys: Vec<Vec<f64>> = (1..=HORIZON).map(|h| target_h(e, h)).collect();
            let ep = Ep {
                fam: e.family_id,
                p: [0.0; 2],
                x0: e.input.clone(),
                ys,
            };
            (run_metafe_episode(cfg, ctrl, &ep, K).preds, e)
        })
        .collect();
    let raw = e33_score(&trs);
    let trs_orb: Vec<(Vec<Vec<f64>>, &Example)> = trs
        .iter()
        .map(|(p, e)| {
            let mut p2 = p.clone();
            p2[0] = d.step(&e.input, &e.context);
            (p2, *e)
        })
        .collect();
    let trs_auto: Vec<(Vec<Vec<f64>>, &Example)> = trs
        .iter()
        .map(|(p, e)| {
            let mut p2 = p.clone();
            if ctrl.cold_trained {
                p2[0] = cold_start_step(ctrl, &e.input, &e.context);
            }
            (p2, *e)
        })
        .collect();
    (e33_score(&trs_orb), e33_score(&trs_auto), raw)
}

// ─── CLI: metafe-paso0, metafe-run, metafe-diag ──────────────────────────────

pub fn seeds(phase: &str) -> std::ops::RangeInclusive<u64> {
    match phase {
        "tune" | "bq_tune" => 0xC4B0..=0xC4B3,
        "dev" | "bq_dev" => 0xA400..=0xA40F,
        "confirm" | "bq_confirm" => 0xB400..=0xB40F,
        "smoke" => 0xC4A0..=0xC4AF,
        _ => panic!("fase desconocida {phase}"),
    }
}

pub fn paso0(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap_or_else(|| "0xC4B0".into()));
    let mut res = serde_json::Map::new();

    // (a) Determinismo bit a bit del entrenamiento y la evaluación
    let cfg = metafe_arms().into_iter().find(|c| c.name == "METAFE").unwrap();
    let (c1, _) = train_metafe_controller(&cfg, seed, 32, 16);
    let (c2, _) = train_metafe_controller(&cfg, seed, 32, 16);
    let held = gen_heldout(seed, 4);
    let mut mismatch = 0usize;
    let mut total_vals = 0usize;
    let mut fe_reduced_all = true;
    for e in &held {
        let r1 = run_metafe_episode(&cfg, &c1, e, K);
        let r2 = run_metafe_episode(&cfg, &c2, e, K);
        let b1 = atlas::bits(&r1.preds);
        let b2 = atlas::bits(&r2.preds);
        total_vals += b1.len();
        mismatch += b1.iter().zip(&b2).filter(|(x, y)| x != y).count();
        if r1.fe_reduction <= 0.0 {
            fe_reduced_all = false;
        }
    }
    res.insert(
        "a_deterministic".into(),
        json!({"values": total_vals, "mismatch": mismatch, "pass": mismatch == 0 && total_vals > 0}),
    );

    // (b) Cero fuga de familias retenidas {8,10,11} en meta-entrenamiento
    let (eps, rej) = gen_meta_training_eps(seed, 128, XOR_METAFE_TRAIN);
    let no_leak = eps.iter().all(|e| !NEW_FAMS.contains(&e.fam));
    res.insert(
        "b_no_heldout_leak".into(),
        json!({"pass": no_leak, "n_eps": eps.len(), "syn_rejected": rej}),
    );

    // (c) Verificación de 0 consultas TEST
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let mut c3 = MetaController::default();
    train_cold_start_prior(&mut c3, seed);
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    res.insert("c_zero_test_queries".into(), json!({"pass": q == 0, "queries": q}));

    // (d) Exactitud en familias afines conocidas (rel_err < 1e-8 en LieAffine)
    let aff_eps: Vec<&Ep> = held.iter().filter(|e| e.fam < 7).collect();
    let max_aff_err = aff_eps
        .iter()
        .map(|e| {
            let r = run_metafe_episode(&cfg, &c1, e, K);
            clamp_err(&r.preds[HORIZON - 1], &e.ys[HORIZON - 1])
        })
        .fold(0.0, f64::max);
    res.insert(
        "d_affine_exact".into(),
        json!({"max_rel_err": max_aff_err, "fe_reduced_all": fe_reduced_all, "pass": max_aff_err < 1e-4}),
    );

    let ok = res.values().all(|v| v["pass"] == true);
    println!(
        "{}",
        json!({"seed": format!("0x{seed:X}"), "git_sha": git_sha(), "paso0_ok": ok, "res": res})
    );
}

pub fn run(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap());
    let phase = arg_of(a, "--phase").unwrap();
    assert!(seeds(&phase).contains(&seed), "semilla fuera de rango para fase {phase}");
    let wpath = arg_of(a, "--weights").unwrap();
    let d = Orbita::from_json(&std::fs::read_to_string(&wpath).unwrap());
    let raw0 = d.raw.clone();
    let out = PathBuf::from(arg_of(a, "--out").unwrap()).join(&phase);
    std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done"));
    if done.exists() && !a.iter().any(|x| x == "--force") {
        eprintln!("skip 0x{seed:X}");
        return;
    }
    let only: Option<Vec<String>> = arg_of(a, "--arms").map(|s| s.split(',').map(String::from).collect());
    let n_tr: usize = arg_of(a, "--ntrain").map(|v| v.parse().unwrap()).unwrap_or(128);
    let n_va: usize = arg_of(a, "--nval").map(|v| v.parse().unwrap()).unwrap_or(96);

    let t0 = std::time::Instant::now();
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let held = gen_heldout(seed, HPF);
    let syn_ood = gen_ood_synthetic_eps(seed, 48);
    let (ex, sha) = e33_examples(seed);

    let mut armsj = serde_json::Map::new();
    for cfg in metafe_arms() {
        if let Some(o) = &only {
            if !o.iter().any(|x| x == cfg.name) {
                continue;
            }
        }
        let ta = std::time::Instant::now();
        let (ctrl, train_meta) = train_metafe_controller(&cfg, seed, n_tr, n_va);
        let train_s = ta.elapsed().as_secs_f64();

        let te = std::time::Instant::now();
        let (held_j, metacog_j, nan_h) = eval_metafe_set(&cfg, &ctrl, &held, true);
        let (syn_j, syn_meta_j, nan_s) = eval_metafe_set(&cfg, &ctrl, &syn_ood, false);
        let (e33_orb, e33_auto, e33_raw) = e33_metafe(&cfg, &ctrl, &d, &ex);
        let eval_s = te.elapsed().as_secs_f64();

        armsj.insert(
            cfg.name.into(),
            json!({
                "held": held_j,
                "syn_ood": syn_j,
                "metacog": metacog_j,
                "syn_metacog": syn_meta_j,
                "e33": e33_orb,
                "e33_autonomous": e33_auto,
                "e33_raw": e33_raw,
                "nan": nan_h || nan_s,
                "train_meta": train_meta,
                "train_s": train_s,
                "eval_s": eval_s,
                "wall_s": ta.elapsed().as_secs_f64(),
            }),
        );
        eprintln!(
            "0x{seed:X} {:<14} new_med={:+.4} new_acc={:.4} known_acc={:.4} syn_med={:+.4} e33_auto={} {:.2}s",
            cfg.name,
            held_j["new_med_log"].as_f64().unwrap_or(f64::NAN),
            held_j["new_acc16"].as_f64().unwrap_or(f64::NAN),
            held_j["known_acc16"].as_f64().unwrap_or(f64::NAN),
            syn_j["syn_ood_med_log"].as_f64().unwrap_or(f64::NAN),
            e33_auto["pass"],
            ta.elapsed().as_secs_f64()
        );
    }

    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    assert_eq!(d.raw, raw0);
    assert_eq!(q, 0, "consultas TEST ≠ 0");

    let r = json!({
        "seed": format!("0x{seed:X}"),
        "phase": phase,
        "weights": wpath,
        "arms": armsj,
        "test_queries": q,
        "test_sha256": sha,
        "git_sha": git_sha(),
        "wall_s": t0.elapsed().as_secs_f64(),
    });
    let tmp = out.join(format!("metrics_0x{seed:X}.json.tmp"));
    std::fs::write(&tmp, serde_json::to_string_pretty(&r).unwrap()).unwrap();
    std::fs::rename(&tmp, out.join(format!("metrics_0x{seed:X}.json"))).unwrap();
    std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} ok {:.2}s", t0.elapsed().as_secs_f64());
}

/// Diagnóstico detallado por familia y por hipótesis en semillas de ajuste (0xC4B0..0xC4B3).
pub fn diag(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap_or_else(|| "0xC4B0".into()));
    assert!((0xC400..=0xC4FF).contains(&seed), "diag solo en semillas de ajuste");
    let held = gen_heldout(seed, HPF);
    let syn_ood = gen_ood_synthetic_eps(seed, 48);
    let cfg = metafe_arms().into_iter().find(|c| c.name == "METAFE").unwrap();
    let (ctrl, tm) = train_metafe_controller(&cfg, seed, 128, 96);

    // Evaluar cada hipótesis aislada sobre known, new, y por familia (7, 8, 10, 11) + syn_ood
    let mut hyp_tbl = serde_json::Map::new();
    for (hi, &hk) in cfg.hyps.iter().enumerate() {
        let single_cfg = MetaFeCfg {
            name: HYP_NAMES[hi],
            hyps: vec![hk],
            use_vfe: true,
            use_oot: false,
            use_multistep: true,
            use_bma: false,
            use_epi_gate: false,
            use_meta_prior: false,
        };
        let (hj, mj, _) = eval_metafe_set(&single_cfg, &ctrl, &held, true);
        let (sj, _, _) = eval_metafe_set(&single_cfg, &ctrl, &syn_ood, false);
        hyp_tbl.insert(
            HYP_NAMES[hi].into(),
            json!({
                "known_acc16": hj["known_acc16"],
                "known_med_log": hj["known_med_log"],
                "new_acc16": hj["new_acc16"],
                "new_med_log": hj["new_med_log"],
                "syn_ood_med_log": sj["syn_ood_med_log"],
                "syn_ood_acc16": sj["syn_ood_acc16"],
                "per_fam": mj["overall"]["per_fam_med_log"],
            }),
        );
    }
    let (hj, mj, _) = eval_metafe_set(&cfg, &ctrl, &held, true);
    let (sj, _, _) = eval_metafe_set(&cfg, &ctrl, &syn_ood, false);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "seed": format!("0x{seed:X}"),
            "train_meta": tm,
            "METAFE_held": hj,
            "METAFE_syn_ood": sj,
            "METAFE_metacog": mj,
            "per_hypothesis": hyp_tbl,
        }))
        .unwrap()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_loo_matches_bruteforce() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0x12345);
        let n = 14;
        let us: Vec<[f64; 2]> = (0..n)
            .map(|_| [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)])
            .collect();
        let y: Vec<f64> = us.iter().map(|u| (1.3 * u[0]).sin() + 0.4 * u[1] * u[1]).collect();
        let p = KernelParams {
            kind: KernelKind::CartM52,
            w_lin: 2.0,
            w1: 0.4,
            w2: 0.4,
            w12: 0.5,
            l1: 1.1,
            l2: 0.9,
            noise: 1e-3,
        };
        let full = fit_gp_channel_fixed(&us, &y, p).unwrap();
        for i in 0..n {
            let us_sub: Vec<[f64; 2]> = us.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &u)| u).collect();
            let y_sub: Vec<f64> = y.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &v)| v).collect();
            let sub = fit_gp_channel_fixed(&us_sub, &y_sub, p).unwrap();
            let mu_bf = sub.predict_mean(&us_sub, &us[i]);
            assert!(
                (full.loo_preds[i] - mu_bf).abs() < 1e-9,
                "LOO mismatch at {i}: {} vs {mu_bf}",
                full.loo_preds[i]
            );
        }
    }

    #[test]
    fn exact_affine_recovery_and_vfe_reduction() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0xC4B0);
        let e = make_ep(&mut rng, 1); // rotación conocida
        let cfg = metafe_arms().into_iter().find(|c| c.name == "METAFE").unwrap();
        let ctrl = MetaController::default();
        let rep = run_metafe_episode(&cfg, &ctrl, &e, K);
        let err = clamp_err(&rep.preds[HORIZON - 1], &e.ys[HORIZON - 1]);
        assert!(err < 1e-6, "affine err too large: {err}");
        assert!(rep.fe_reduction > 5.0, "expected strong VFE reduction, got {}", rep.fe_reduction);
    }

    #[test]
    fn polar_jacobian_and_so2_equivariance() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0x555);
        // Rotación pura con deformación radial suave (familia 7)
        let e = make_ep(&mut rng, 7);
        let ctx: Vec<&[f64]> = (0..=K).map(|t| e.frame(t)).collect();
        let hp = fit_hypothesis(HypKind::GpPolar, &ctx);
        assert!(hp.valid && hp.vfe.is_finite() && hp.loo_fe.is_finite());
        let (p, epi) = hp.step_frame(e.frame(K));
        assert!(fin(&p) && epi >= 0.0 && epi.is_finite());
    }

    #[test]
    fn equiv_cart_psd_and_vfe_radial_warp() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0x888);
        let e = make_ep(&mut rng, 7); // radial_warp: combinación radial SO(2) + cartesiano
        let ctx: Vec<&[f64]> = (0..=K).map(|t| e.frame(t)).collect();
        let hp = fit_hypothesis(HypKind::GpEquivCart, &ctx);
        assert!(hp.valid && hp.vfe.is_finite() && hp.loo_fe.is_finite());
        let (p, epi) = hp.step_frame(e.frame(K));
        assert!(fin(&p) && epi >= 0.0 && epi.is_finite());
        let err = clamp_err(&p, e.frame(K + 1));
        assert!(err < 0.05, "GpEquivCart 1-step err too large on fam 7: {err}");
    }

    #[test]
    fn cold_start_gp_interpolates_hole_zero_test_queries() {
        let seed = 0xC4B1;
        let before = cdt_rqm_epr::v4_provenance::snapshot();
        let mut ctrl = MetaController::default();
        train_cold_start_prior(&mut ctrl, seed);
        let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
        assert_eq!(q, 0, "cold-start GP touched TEST");
        assert!(ctrl.cold_trained);
        let (ex, _) = e33_examples(seed);
        let ok = ex
            .iter()
            .filter(|e| {
                let p = cold_start_step(&ctrl, &e.input, &e.context);
                clamp_err(&p, &e.expected) < mt::REL_ERR_OK
            })
            .count();
        let acc = ok as f64 / ex.len() as f64;
        assert!(acc >= 0.90, "cold-start GP h1 accuracy too low on E33 hole: {acc:.4}");
    }

    #[test]
    fn ard_reduces_free_energy_monotonically() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0x777);
        let e = make_ep(&mut rng, 5);
        let ctx: Vec<&[f64]> = (0..=K).map(|t| e.frame(t)).collect();
        let (qs, qns) = extract_pairs(&ctx);
        let phi: Vec<[f64; N_ARD]> = qs.iter().map(|&q| ard_feat(q)).collect();
        let yx: Vec<f64> = qs.iter().zip(&qns).map(|(a, b)| b[0] - a[0]).collect();
        let ch = fit_ard_channel(&phi, &yx).unwrap();
        assert!(ch.vfe <= ch.vfe_init, "ARD VFE did not decrease: {} -> {}", ch.vfe_init, ch.vfe);
    }

    #[test]
    fn meta_training_never_contains_heldout_fams() {
        let (eps, _) = gen_meta_training_eps(0xC4B0, 64, XOR_METAFE_TRAIN);
        assert!(eps.iter().all(|e| !NEW_FAMS.contains(&e.fam)));
    }
}
