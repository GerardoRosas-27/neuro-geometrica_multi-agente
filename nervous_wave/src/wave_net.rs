//! Complex-valued damped traveling-wave propagation engine on directed graphs
//! with exact analytical Backpropagation Through Time (BPTT, T = 8 physical steps),
//! saturating phase-preserving oscillators, and frozen-synapse myelin calibration.

use crate::domains::DomainData;
use crate::rng::{checksum_f64_slices, Rng64};
use crate::topology::{
    TopologyKind, WaveTopology, HUB_START, N_HUBS, N_NODES, N_PAST_REC,
};
use std::f64::consts::PI;

pub const LATENT_D: usize = 4;
pub const N_STEPS: usize = 8;
pub const READOUT_START: usize = 6; // average over t in {6, 7, 8}
pub const REL_DIM: usize = 2 * N_HUBS; // 32 (16 Re(psi_h) + 16 centered |psi_h|^2)
pub const BETA_SAT: f64 = 0.50;
pub const LAMBDA_SHUNT: f64 = 0.22;
pub const SUSTAIN_DRIVE: f64 = 0.35;

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveTrainMode {
    /// Train sensory wave encoder + all synapses + nodal frequencies/damping; myelin cal locked at 0.
    FullTrain,
    /// Freeze ALL synapses (alpha, theta) & base nodal (omega, gamma) bit-exact; train ONLY 64 myelin_omega knobs + domain encoder/head.
    MyelinCalOnly,
}

#[derive(Clone, Debug)]
pub struct AdamBuf {
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl AdamBuf {
    pub fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    pub fn step(
        &mut self,
        params: &mut [f64],
        grads: &[f64],
        lr: f64,
        wd: f64,
        t: usize,
        grad_clip: f64,
    ) {
        let b1: f64 = 0.9;
        let b2: f64 = 0.999;
        let eps: f64 = 1e-8;
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        let n2: f64 = grads.iter().map(|g| g * g).sum();
        let scale = if n2 > grad_clip * grad_clip {
            grad_clip / n2.sqrt()
        } else {
            1.0
        };
        for i in 0..params.len() {
            let g = grads[i] * scale;
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let mh = self.m[i] / bc1;
            let vh = self.v[i] / bc2;
            params[i] -= lr * (mh / (vh.sqrt() + eps) + wd * params[i]);
        }
    }

    pub fn step_spring(
        &mut self,
        params: &mut [f64],
        grads: &[f64],
        lr: f64,
        k_spring: f64,
        anchor: f64,
        grad_clip: f64,
    ) {
        let n2: f64 = grads.iter().map(|g| g * g).sum();
        let scale = if n2 > grad_clip * grad_clip {
            grad_clip / n2.sqrt()
        } else {
            1.0
        };
        for i in 0..params.len() {
            let g_eff = grads[i] * scale + k_spring * (params[i] - anchor);
            self.m[i] = 0.80 * self.m[i] + g_eff;
            params[i] -= lr * self.m[i];
        }
    }
}

#[derive(Clone, Debug)]
pub struct DomainEncoder {
    pub obs_dim: usize,
    pub w: Vec<f64>,
    pub b: Vec<f64>,
}

impl DomainEncoder {
    pub fn new(obs_dim: usize, rng: &mut Rng64) -> Self {
        let scale = (1.0 / (obs_dim as f64)).sqrt() * 0.85;
        Self {
            obs_dim,
            w: (0..(LATENT_D * obs_dim)).map(|_| rng.normal() * scale).collect(),
            b: vec![0.0; LATENT_D],
        }
    }

    #[inline]
    pub fn forward(&self, x: &[f64]) -> [f64; LATENT_D] {
        let mut u = [0.0; LATENT_D];
        for k in 0..LATENT_D {
            let row = &self.w[k * self.obs_dim..(k + 1) * self.obs_dim];
            let mut acc = self.b[k];
            for j in 0..self.obs_dim {
                acc += row[j] * x[j];
            }
            u[k] = acc;
        }
        u
    }

    #[inline]
    pub fn backward(&self, x: &[f64], du: &[f64; LATENT_D], gw: &mut [f64], gb: &mut [f64]) {
        for k in 0..LATENT_D {
            let dk = du[k];
            gb[k] += dk;
            let row = &mut gw[k * self.obs_dim..(k + 1) * self.obs_dim];
            for j in 0..self.obs_dim {
                row[j] += dk * x[j];
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct DomainHead {
    pub reg_dim: usize,
    pub cls_dim: usize,
    pub w_reg: Vec<f64>,
    pub b_reg: Vec<f64>,
    pub w_cls: Vec<f64>,
    pub b_cls: Vec<f64>,
}

impl DomainHead {
    pub fn new(reg_dim: usize, cls_dim: usize, rng: &mut Rng64) -> Self {
        let s = (1.0 / (REL_DIM as f64)).sqrt() * 0.45;
        Self {
            reg_dim,
            cls_dim,
            w_reg: (0..(reg_dim * REL_DIM)).map(|_| rng.normal() * s).collect(),
            b_reg: vec![0.0; reg_dim],
            w_cls: (0..(cls_dim * REL_DIM)).map(|_| rng.normal() * s).collect(),
            b_cls: vec![0.0; cls_dim],
        }
    }

    pub fn forward(&self, z: &[f64; REL_DIM]) -> (Vec<f64>, Vec<f64>) {
        let mut reg = vec![0.0; self.reg_dim];
        for r in 0..self.reg_dim {
            let mut acc = self.b_reg[r];
            let row = &self.w_reg[r * REL_DIM..(r + 1) * REL_DIM];
            for j in 0..REL_DIM {
                acc += row[j] * z[j];
            }
            reg[r] = acc;
        }
        let mut logits = vec![0.0; self.cls_dim];
        for c in 0..self.cls_dim {
            let mut acc = self.b_cls[c];
            let row = &self.w_cls[c * REL_DIM..(c + 1) * REL_DIM];
            for j in 0..REL_DIM {
                acc += row[j] * z[j];
            }
            logits[c] = acc;
        }
        (reg, logits)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn loss_and_backward(
        &self,
        z: &[f64; REL_DIM],
        reg_pred: &[f64],
        logits: &[f64],
        reg_true: &[f64],
        cls_true: usize,
        w_mse: f64,
        w_ce: f64,
        scale: f64,
        gw_reg: &mut [f64],
        gb_reg: &mut [f64],
        gw_cls: &mut [f64],
        gb_cls: &mut [f64],
    ) -> (f64, [f64; REL_DIM]) {
        let mut loss = 0.0;
        let mut dz = [0.0; REL_DIM];

        if self.reg_dim > 0 && w_mse > 0.0 {
            let inv_r = 1.0 / (self.reg_dim as f64);
            for r in 0..self.reg_dim {
                let err = reg_pred[r] - reg_true[r];
                loss += w_mse * 0.5 * err * err * inv_r;
                let g = scale * w_mse * err * inv_r;
                gb_reg[r] += g;
                let row_g = &mut gw_reg[r * REL_DIM..(r + 1) * REL_DIM];
                let row_w = &self.w_reg[r * REL_DIM..(r + 1) * REL_DIM];
                for j in 0..REL_DIM {
                    row_g[j] += g * z[j];
                    dz[j] += g * row_w[j];
                }
            }
        }

        if self.cls_dim > 0 && w_ce > 0.0 {
            let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut probs = vec![0.0; self.cls_dim];
            let mut sum_e = 0.0;
            for c in 0..self.cls_dim {
                let e = (logits[c] - max_l).exp();
                probs[c] = e;
                sum_e += e;
            }
            for c in 0..self.cls_dim {
                probs[c] /= sum_e;
            }
            loss += -w_ce * probs[cls_true].max(1e-15).ln();
            for c in 0..self.cls_dim {
                let target = if c == cls_true { 1.0 } else { 0.0 };
                let g = scale * w_ce * (probs[c] - target);
                gb_cls[c] += g;
                let row_g = &mut gw_cls[c * REL_DIM..(c + 1) * REL_DIM];
                let row_w = &self.w_cls[c * REL_DIM..(c + 1) * REL_DIM];
                for j in 0..REL_DIM {
                    row_g[j] += g * z[j];
                    dz[j] += g * row_w[j];
                }
            }
        }

        (loss, dz)
    }
}

/// Complex traveling-wave network running on any `WaveTopology`.
#[derive(Clone, Debug)]
pub struct NervousWaveCore {
    pub topo: WaveTopology,
    // Sensory receptor wavevector bank (16 receptors x 4D + 16 phases = 80 params)
    pub w_in: Vec<f64>,   // 16 * 4 = 64
    pub phi_in: Vec<f64>, // 16
    // Per-edge synaptic conduction parameters (2 * E params)
    pub alpha_e: Vec<f64>, // E (amplitude via softplus)
    pub theta_e: Vec<f64>, // E (axonal phase lag / conduction delay)
    // Per-node somatic resonance parameters (2 * 64 = 128 params)
    pub omega_n: Vec<f64>,       // 64 (natural oscillation frequency)
    pub gamma_logit_n: Vec<f64>, // 64 (damping sigmoid(gamma_logit))
    // 64 nodal myelin phase auto-calibration knobs (used ONLY in MyelinCalOnly transfer)
    pub myelin_omega: Vec<f64>, // 64, init 0.0
}

#[derive(Clone, Debug)]
pub struct StepParamCache {
    pub g_e: Vec<f64>,
    pub cos_th: Vec<f64>,
    pub sin_th: Vec<f64>,
    pub k_re: Vec<f64>,
    pub k_im: Vec<f64>,
    pub sig_alpha: Vec<f64>,
    pub gamma: [f64; N_NODES],
    pub cw: [f64; N_NODES],
    pub sw: [f64; N_NODES],
    pub r_re: [f64; N_NODES],
    pub r_im: [f64; N_NODES],
}

impl StepParamCache {
    pub fn from_core(core: &NervousWaveCore, edge_mask: Option<&[bool]>) -> Self {
        let m = core.topo.edges.len();
        let is_cx = core.topo.kind.is_complex_wave();
        let mut g_e = vec![0.0; m];
        let mut cos_th = vec![1.0; m];
        let mut sin_th = vec![0.0; m];
        let mut k_re = vec![0.0; m];
        let mut k_im = vec![0.0; m];
        let mut sig_alpha = vec![0.0; m];

        for e_idx in 0..m {
            let active = edge_mask.map_or(true, |mk| mk[e_idx]);
            if !active {
                continue;
            }
            let edge = core.topo.edges[e_idx];
            let sp = softplus(core.alpha_e[e_idx]);
            let g = edge.sign * sp * core.topo.inv_sqrt_indeg[edge.dst];
            g_e[e_idx] = g;
            sig_alpha[e_idx] = sigmoid(core.alpha_e[e_idx]);
            if is_cx {
                let (st, ct) = core.theta_e[e_idx].sin_cos();
                cos_th[e_idx] = ct;
                sin_th[e_idx] = st;
                k_re[e_idx] = g * ct;
                k_im[e_idx] = g * st;
            } else {
                k_re[e_idx] = g;
            }
        }

        let mut gamma = [0.0; N_NODES];
        let mut cw = [1.0; N_NODES];
        let mut sw = [0.0; N_NODES];
        let mut r_re = [0.0; N_NODES];
        let mut r_im = [0.0; N_NODES];
        for i in 0..N_NODES {
            let gm = sigmoid(core.gamma_logit_n[i]);
            gamma[i] = gm;
            if is_cx {
                let om = core.omega_n[i] + core.myelin_omega[i];
                let (s, c) = om.sin_cos();
                cw[i] = c;
                sw[i] = s;
                r_re[i] = gm * c;
                r_im[i] = gm * s;
            } else {
                r_re[i] = gm;
            }
        }

        Self {
            g_e,
            cos_th,
            sin_th,
            k_re,
            k_im,
            sig_alpha,
            gamma,
            cw,
            sw,
            r_re,
            r_im,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WaveRolloutCache {
    pub u: [f64; LATENT_D],
    pub v: [f64; LATENT_D],
    pub rec_cos: [f64; 16],
    pub rec_sin: [f64; 16],
    pub psi_re: [[f64; N_NODES]; N_STEPS + 1],
    pub psi_im: [[f64; N_NODES]; N_STEPS + 1],
    pub s_re: [[f64; N_NODES]; N_STEPS],
    pub s_im: [[f64; N_NODES]; N_STEPS],
    pub q: [[f64; N_NODES]; N_STEPS],
    pub z_rel: [f64; REL_DIM],
}

#[derive(Clone, Debug)]
pub struct NervousWaveGrads {
    pub w_in: Vec<f64>,
    pub phi_in: Vec<f64>,
    pub alpha_e: Vec<f64>,
    pub theta_e: Vec<f64>,
    pub omega_n: Vec<f64>,
    pub gamma_logit_n: Vec<f64>,
    pub myelin_omega: Vec<f64>,
}

impl NervousWaveGrads {
    pub fn zeros(n_edges: usize) -> Self {
        Self {
            w_in: vec![0.0; 16 * LATENT_D],
            phi_in: vec![0.0; 16],
            alpha_e: vec![0.0; n_edges],
            theta_e: vec![0.0; n_edges],
            omega_n: vec![0.0; N_NODES],
            gamma_logit_n: vec![0.0; N_NODES],
            myelin_omega: vec![0.0; N_NODES],
        }
    }

    pub fn clear(&mut self) {
        self.w_in.fill(0.0);
        self.phi_in.fill(0.0);
        self.alpha_e.fill(0.0);
        self.theta_e.fill(0.0);
        self.omega_n.fill(0.0);
        self.gamma_logit_n.fill(0.0);
        self.myelin_omega.fill(0.0);
    }
}

impl NervousWaveCore {
    pub fn new(kind: TopologyKind, rng: &mut Rng64) -> Self {
        let topo = WaveTopology::build(kind, rng);
        let m = topo.edges.len();
        let mut w_in = vec![0.0; 16 * LATENT_D];
        for i in 0..16 {
            let dir = rng.unit_vec(LATENT_D);
            let rad = rng.uniform_range(0.55, 1.35);
            for d in 0..LATENT_D {
                w_in[i * LATENT_D + d] = dir[d] * rad;
            }
        }
        let phi_in = (0..16).map(|_| rng.uniform_range(-PI, PI)).collect();

        // Initialize synaptic conductances cleanly: slightly stronger myelinated tracts (pathway 3)
        // and moderate inhibitory synapses (pathway 2) so waves propagate without dying out.
        let mut alpha_e = vec![0.0; m];
        let mut theta_e = vec![0.0; m];
        for e_idx in 0..m {
            let pw = topo.edges[e_idx].pathway;
            alpha_e[e_idx] = match pw {
                3 => 0.95 + rng.normal() * 0.12, // myelinated long-range axon
                2 => 0.15 + rng.normal() * 0.12, // lateral inhibitory interneuron
                _ => 0.65 + rng.normal() * 0.15,
            };
            theta_e[e_idx] = if kind.is_complex_wave() {
                rng.uniform_range(-PI, PI)
            } else {
                0.0
            };
        }

        let omega_n = if kind.is_complex_wave() {
            (0..N_NODES).map(|_| rng.uniform_range(-0.9, 0.9)).collect()
        } else {
            vec![0.0; N_NODES]
        };
        // sigmoid(0.4) ~ 0.60 somatic retention (balanced wave echo + decay)
        let gamma_logit_n = (0..N_NODES).map(|_| 0.4 + rng.normal() * 0.15).collect();
        let myelin_omega = vec![0.0; N_NODES];

        Self {
            topo,
            w_in,
            phi_in,
            alpha_e,
            theta_e,
            omega_n,
            gamma_logit_n,
            myelin_omega,
        }
    }

    pub fn edge_count(&self) -> usize {
        self.topo.edges.len()
    }

    pub fn autocal_param_count(&self) -> usize {
        self.myelin_omega.len()
    }

    pub fn reset_myelin(&mut self) {
        self.myelin_omega.fill(0.0);
    }

    /// Exact IEEE-754 checksum over ALL synapses & base somatic parameters (frozen during P4 transfer).
    pub fn frozen_synapses_checksum(&self) -> u64 {
        checksum_f64_slices(&[
            &self.w_in,
            &self.phi_in,
            &self.alpha_e,
            &self.theta_e,
            &self.omega_n,
            &self.gamma_logit_n,
        ])
    }

    /// Permute synaptic weights/delays across edges as a negative control for P4 transfer.
    pub fn with_permuted_synapses(&self, rng: &mut Rng64) -> Self {
        let mut c = self.clone();
        let mut idx: Vec<usize> = (0..c.alpha_e.len()).collect();
        rng.shuffle(&mut idx);
        let a_old = c.alpha_e.clone();
        let t_old = c.theta_e.clone();
        for (i, &p) in idx.iter().enumerate() {
            c.alpha_e[i] = a_old[p];
            c.theta_e[i] = t_old[p];
        }
        c
    }

    pub fn forward(&self, u: [f64; LATENT_D], v: [f64; LATENT_D]) -> WaveRolloutCache {
        let sp = StepParamCache::from_core(self, None);
        self.forward_with_cache(u, v, &sp)
    }

    pub fn forward_masked(
        &self,
        u: [f64; LATENT_D],
        v: [f64; LATENT_D],
        edge_mask: &[bool],
    ) -> WaveRolloutCache {
        let sp = StepParamCache::from_core(self, Some(edge_mask));
        self.forward_with_cache(u, v, &sp)
    }

    /// Inject explicit impulse wave packets at Past and/or Future receptors for P1 physical validation.
    pub fn forward_impulse(
        &self,
        inject_past: bool,
        inject_fut: bool,
        u: [f64; LATENT_D],
        v: [f64; LATENT_D],
    ) -> WaveRolloutCache {
        let sp = StepParamCache::from_core(self, None);
        let is_cx = self.topo.kind.is_complex_wave();
        let mut rec_cos = [0.0; 16];
        let mut rec_sin = [0.0; 16];
        let mut psi_re = [[0.0; N_NODES]; N_STEPS + 1];
        let mut psi_im = [[0.0; N_NODES]; N_STEPS + 1];

        for i in 0..16 {
            let active = (i < N_PAST_REC && inject_past) || (i >= N_PAST_REC && inject_fut);
            if !active {
                continue;
            }
            let src_vec = if i < N_PAST_REC { &u } else { &v };
            let row = &self.w_in[i * LATENT_D..(i + 1) * LATENT_D];
            let mut arg = self.phi_in[i];
            for d in 0..LATENT_D {
                arg += row[d] * src_vec[d];
            }
            let (s, c) = arg.sin_cos();
            rec_cos[i] = c;
            rec_sin[i] = if is_cx { s } else { 0.0 };
            psi_re[0][i] = c;
            psi_im[0][i] = rec_sin[i];
        }
        self.rollout_from_initial(u, v, rec_cos, rec_sin, psi_re, psi_im, &sp)
    }

    pub fn forward_with_cache(
        &self,
        u: [f64; LATENT_D],
        v: [f64; LATENT_D],
        sp: &StepParamCache,
    ) -> WaveRolloutCache {
        let is_cx = self.topo.kind.is_complex_wave();
        let mut rec_cos = [0.0; 16];
        let mut rec_sin = [0.0; 16];
        let mut psi_re = [[0.0; N_NODES]; N_STEPS + 1];
        let mut psi_im = [[0.0; N_NODES]; N_STEPS + 1];

        for i in 0..16 {
            let src_vec = if i < N_PAST_REC { &u } else { &v };
            let row = &self.w_in[i * LATENT_D..(i + 1) * LATENT_D];
            let mut arg = self.phi_in[i];
            for d in 0..LATENT_D {
                arg += row[d] * src_vec[d];
            }
            let (s, c) = arg.sin_cos();
            rec_cos[i] = c;
            rec_sin[i] = if is_cx { s } else { 0.0 };
            psi_re[0][i] = c;
            psi_im[0][i] = rec_sin[i];
        }
        self.rollout_from_initial(u, v, rec_cos, rec_sin, psi_re, psi_im, sp)
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_from_initial(
        &self,
        u: [f64; LATENT_D],
        v: [f64; LATENT_D],
        rec_cos: [f64; 16],
        rec_sin: [f64; 16],
        mut psi_re: [[f64; N_NODES]; N_STEPS + 1],
        mut psi_im: [[f64; N_NODES]; N_STEPS + 1],
        sp: &StepParamCache,
    ) -> WaveRolloutCache {
        let mut s_re = [[0.0; N_NODES]; N_STEPS];
        let mut s_im = [[0.0; N_NODES]; N_STEPS];
        let mut q = [[0.0; N_NODES]; N_STEPS];
        let edges = &self.topo.edges;

        for t in 0..N_STEPS {
            let mut acc_re = [0.0; N_NODES];
            let mut acc_im = [0.0; N_NODES];
            let mut e_inh = [0.0; N_NODES];

            // Self resonant oscillation: gamma * exp(i omega) * psi(t)
            for i in 0..N_NODES {
                let pr = psi_re[t][i];
                let pi = psi_im[t][i];
                acc_re[i] = sp.r_re[i] * pr - sp.r_im[i] * pi;
                acc_im[i] = sp.r_re[i] * pi + sp.r_im[i] * pr;
            }
            // Sustained sensory wave-packet injection at t in {0, 1}
            if t < 2 {
                for i in 0..16 {
                    acc_re[i] += SUSTAIN_DRIVE * psi_re[0][i];
                    acc_im[i] += SUSTAIN_DRIVE * psi_im[0][i];
                }
            }
            // Directed axonal propagation: excitatory complex wave superposition + divisive shunting inhibition
            for (e_idx, edge) in edges.iter().enumerate() {
                let pr = psi_re[t][edge.src];
                let pi = psi_im[t][edge.src];
                if edge.sign > 0.0 {
                    let kr = sp.k_re[e_idx];
                    let ki = sp.k_im[e_idx];
                    acc_re[edge.dst] += kr * pr - ki * pi;
                    acc_im[edge.dst] += kr * pi + ki * pr;
                } else {
                    let g_abs = -sp.g_e[e_idx];
                    e_inh[edge.dst] += g_abs * (pr * pr + pi * pi);
                }
            }
            // Phase-preserving saturating oscillator with GABAergic divisive shunting normalization
            for i in 0..N_NODES {
                let sr = acc_re[i];
                let si = acc_im[i];
                s_re[t][i] = sr;
                s_im[t][i] = si;
                let qi = 1.0 / (1.0 + BETA_SAT * (sr * sr + si * si) + LAMBDA_SHUNT * e_inh[i]).sqrt();
                q[t][i] = qi;
                psi_re[t + 1][i] = qi * sr;
                psi_im[t + 1][i] = qi * si;
            }
        }

        // Hub readout averaged over t in {READOUT_START..=N_STEPS}
        let n_win = (N_STEPS + 1 - READOUT_START) as f64;
        let inv_win = 1.0 / n_win;
        let mut z_rel = [0.0; REL_DIM];
        for k in 0..N_HUBS {
            let h = HUB_START + k;
            let mut sum_re = 0.0;
            let mut sum_int = 0.0;
            for t in READOUT_START..=N_STEPS {
                let pr = psi_re[t][h];
                let pi = psi_im[t][h];
                sum_re += pr;
                sum_int += pr * pr + pi * pi - 0.5;
            }
            z_rel[k] = sum_re * inv_win;
            z_rel[N_HUBS + k] = sum_int * inv_win;
        }

        WaveRolloutCache {
            u,
            v,
            rec_cos,
            rec_sin,
            psi_re,
            psi_im,
            s_re,
            s_im,
            q,
            z_rel,
        }
    }

    #[allow(dead_code)]
    pub fn backward(
        &self,
        cache: &WaveRolloutCache,
        dz_rel: &[f64; REL_DIM],
        mode: WaveTrainMode,
        grads: Option<&mut NervousWaveGrads>,
    ) -> ([f64; LATENT_D], [f64; LATENT_D]) {
        let sp = StepParamCache::from_core(self, None);
        self.backward_with_cache(cache, &sp, dz_rel, mode, grads)
    }

    pub fn backward_with_cache(
        &self,
        cache: &WaveRolloutCache,
        sp: &StepParamCache,
        dz_rel: &[f64; REL_DIM],
        mode: WaveTrainMode,
        mut grads: Option<&mut NervousWaveGrads>,
    ) -> ([f64; LATENT_D], [f64; LATENT_D]) {
        let is_cx = self.topo.kind.is_complex_wave();
        let full = mode == WaveTrainMode::FullTrain;
        let myelin = mode == WaveTrainMode::MyelinCalOnly;

        let n_win = (N_STEPS + 1 - READOUT_START) as f64;
        let inv_win = 1.0 / n_win;

        let mut d_out_re = [[0.0; N_NODES]; N_STEPS + 1];
        let mut d_out_im = [[0.0; N_NODES]; N_STEPS + 1];
        for k in 0..N_HUBS {
            let h = HUB_START + k;
            let gr = dz_rel[k] * inv_win;
            let gi = dz_rel[N_HUBS + k] * inv_win;
            for t in READOUT_START..=N_STEPS {
                d_out_re[t][h] += gr + 2.0 * gi * cache.psi_re[t][h];
                d_out_im[t][h] += 2.0 * gi * cache.psi_im[t][h];
            }
        }

        let mut cur_re = [0.0; N_NODES];
        let mut cur_im = [0.0; N_NODES];
        let mut dpsi0_re = [0.0; 16];
        let mut dpsi0_im = [0.0; 16];
        let edges = &self.topo.edges;

        for t in (0..N_STEPS).rev() {
            for i in 0..N_NODES {
                cur_re[i] += d_out_re[t + 1][i];
                cur_im[i] += d_out_im[t + 1][i];
            }

            let mut ds_re = [0.0; N_NODES];
            let mut ds_im = [0.0; N_NODES];
            let mut d_e_inh = [0.0; N_NODES];
            for i in 0..N_NODES {
                let qi = cache.q[t][i];
                let q3 = qi * qi * qi;
                let sr = cache.s_re[t][i];
                let si = cache.s_im[t][i];
                let cr = cur_re[i];
                let ci = cur_im[i];
                let dot = cr * sr + ci * si;
                let coeff = BETA_SAT * q3 * dot;
                ds_re[i] = qi * cr - coeff * sr;
                ds_im[i] = qi * ci - coeff * si;
                d_e_inh[i] = -0.5 * LAMBDA_SHUNT * q3 * dot;
            }

            if t < 2 {
                for i in 0..16 {
                    dpsi0_re[i] += SUSTAIN_DRIVE * ds_re[i];
                    dpsi0_im[i] += SUSTAIN_DRIVE * ds_im[i];
                }
            }

            let mut next_re = [0.0; N_NODES];
            let mut next_im = [0.0; N_NODES];

            // Self somatic oscillation backward
            for i in 0..N_NODES {
                let dsr = ds_re[i];
                let dsi = ds_im[i];
                let rr = sp.r_re[i];
                let ri = sp.r_im[i];
                next_re[i] += rr * dsr + ri * dsi;
                next_im[i] += -ri * dsr + rr * dsi;

                if let Some(ref mut gr) = grads {
                    let pr = cache.psi_re[t][i];
                    let pi = cache.psi_im[t][i];
                    let d_r_re = dsr * pr + dsi * pi;
                    let d_r_im = -dsr * pi + dsi * pr;
                    if full {
                        let d_gamma = d_r_re * sp.cw[i] + d_r_im * sp.sw[i];
                        gr.gamma_logit_n[i] += d_gamma * sp.gamma[i] * (1.0 - sp.gamma[i]);
                    }
                    if is_cx && (full || myelin) {
                        let d_omega = sp.gamma[i] * (-d_r_re * sp.sw[i] + d_r_im * sp.cw[i]);
                        if full {
                            gr.omega_n[i] += d_omega;
                        } else if myelin {
                            gr.myelin_omega[i] += d_omega;
                        }
                    }
                }
            }

            // Directed axonal edges backward (excitatory wave + shunting divisive inhibition)
            for (e_idx, edge) in edges.iter().enumerate() {
                if edge.sign > 0.0 {
                    let dsr = ds_re[edge.dst];
                    let dsi = ds_im[edge.dst];
                    let kr = sp.k_re[e_idx];
                    let ki = sp.k_im[e_idx];
                    next_re[edge.src] += kr * dsr + ki * dsi;
                    next_im[edge.src] += -ki * dsr + kr * dsi;

                    if full {
                        if let Some(ref mut gr) = grads {
                            let pr = cache.psi_re[t][edge.src];
                            let pi = cache.psi_im[t][edge.src];
                            let d_k_re = dsr * pr + dsi * pi;
                            let d_k_im = -dsr * pi + dsi * pr;
                            let d_g = d_k_re * sp.cos_th[e_idx] + d_k_im * sp.sin_th[e_idx];
                            gr.alpha_e[e_idx] += d_g
                                * edge.sign
                                * self.topo.inv_sqrt_indeg[edge.dst]
                                * sp.sig_alpha[e_idx];
                            if is_cx {
                                let d_th = sp.g_e[e_idx]
                                    * (-d_k_re * sp.sin_th[e_idx] + d_k_im * sp.cos_th[e_idx]);
                                gr.theta_e[e_idx] += d_th;
                            }
                        }
                    }
                } else {
                    let d_e = d_e_inh[edge.dst];
                    let g_abs = -sp.g_e[e_idx];
                    let pr = cache.psi_re[t][edge.src];
                    let pi = cache.psi_im[t][edge.src];
                    next_re[edge.src] += 2.0 * d_e * g_abs * pr;
                    next_im[edge.src] += 2.0 * d_e * g_abs * pi;
                    if full {
                        if let Some(ref mut gr) = grads {
                            let e_src = pr * pr + pi * pi;
                            gr.alpha_e[e_idx] += d_e
                                * e_src
                                * self.topo.inv_sqrt_indeg[edge.dst]
                                * sp.sig_alpha[e_idx];
                        }
                    }
                }
            }

            cur_re = next_re;
            cur_im = next_im;
        }

        // Backprop into initial sensory wave packets psi(0) -> (u, v, w_in, phi_in)
        let mut du = [0.0; LATENT_D];
        let mut dv = [0.0; LATENT_D];
        for i in 0..16 {
            let dr = cur_re[i] + dpsi0_re[i];
            let di = cur_im[i] + dpsi0_im[i];
            let d_arg = if is_cx {
                -dr * cache.rec_sin[i] + di * cache.rec_cos[i]
            } else {
                // psi_re(0) = cos(arg), psi_im(0) = 0; compute sin(arg) cleanly
                let src_vec = if i < N_PAST_REC { &cache.u } else { &cache.v };
                let row = &self.w_in[i * LATENT_D..(i + 1) * LATENT_D];
                let mut arg = self.phi_in[i];
                for d in 0..LATENT_D {
                    arg += row[d] * src_vec[d];
                }
                -dr * arg.sin()
            };

            let row = &self.w_in[i * LATENT_D..(i + 1) * LATENT_D];
            if i < N_PAST_REC {
                for d in 0..LATENT_D {
                    du[d] += d_arg * row[d];
                }
            } else {
                for d in 0..LATENT_D {
                    dv[d] += d_arg * row[d];
                }
            }
            if full {
                if let Some(ref mut gr) = grads {
                    gr.phi_in[i] += d_arg;
                    let src_vec = if i < N_PAST_REC { &cache.u } else { &cache.v };
                    let grow = &mut gr.w_in[i * LATENT_D..(i + 1) * LATENT_D];
                    for d in 0..LATENT_D {
                        grow[d] += d_arg * src_vec[d];
                    }
                }
            }
        }

        (du, dv)
    }
}

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub lr_encoder: f64,
    pub lr_core: f64,
    pub lr_myelin: f64,
    pub lr_head: f64,
    pub wd_encoder: f64,
    pub wd_core: f64,
    pub wd_myelin: f64,
    pub wd_head: f64,
    pub w_mse: f64,
    pub w_ce: f64,
    pub latent_norm_reg: f64,
    pub latent_jitter: f64,
    pub mode: WaveTrainMode,
}

#[derive(Clone, Debug)]
pub struct NervousWaveModel {
    pub encoder: DomainEncoder,
    pub core: NervousWaveCore,
    pub head: DomainHead,
}

impl NervousWaveModel {
    pub fn new(
        obs_dim: usize,
        reg_dim: usize,
        cls_dim: usize,
        core: NervousWaveCore,
        rng: &mut Rng64,
    ) -> Self {
        Self {
            encoder: DomainEncoder::new(obs_dim, rng),
            core,
            head: DomainHead::new(reg_dim, cls_dim, rng),
        }
    }

    pub fn predict_pair(
        &self,
        xa: &[f64],
        xb: &[f64],
        edge_mask: Option<&[bool]>,
    ) -> (Vec<f64>, Vec<f64>, [f64; REL_DIM]) {
        let u = self.encoder.forward(xa);
        let v = self.encoder.forward(xb);
        let cache = match edge_mask {
            Some(mk) => self.core.forward_masked(u, v, mk),
            None => self.core.forward(u, v),
        };
        let (reg, logits) = self.head.forward(&cache.z_rel);
        (reg, logits, cache.z_rel)
    }

    pub fn train(
        &mut self,
        domain: &DomainData,
        train_indices: &[usize],
        cfg: &TrainConfig,
        rng: &mut Rng64,
    ) {
        let pre_hash = self.core.frozen_synapses_checksum();

        let mut opt_ew = AdamBuf::new(self.encoder.w.len());
        let mut opt_eb = AdamBuf::new(self.encoder.b.len());
        let mut opt_hw_r = AdamBuf::new(self.head.w_reg.len());
        let mut opt_hb_r = AdamBuf::new(self.head.b_reg.len());
        let mut opt_hw_c = AdamBuf::new(self.head.w_cls.len());
        let mut opt_hb_c = AdamBuf::new(self.head.b_cls.len());

        let mut opt_win = AdamBuf::new(self.core.w_in.len());
        let mut opt_pin = AdamBuf::new(self.core.phi_in.len());
        let mut opt_alp = AdamBuf::new(self.core.alpha_e.len());
        let mut opt_the = AdamBuf::new(self.core.theta_e.len());
        let mut opt_om = AdamBuf::new(self.core.omega_n.len());
        let mut opt_gm = AdamBuf::new(self.core.gamma_logit_n.len());
        let mut opt_my = AdamBuf::new(self.core.myelin_omega.len());

        let mut gw_e = vec![0.0; self.encoder.w.len()];
        let mut gb_e = vec![0.0; self.encoder.b.len()];
        let mut gw_r = vec![0.0; self.head.w_reg.len()];
        let mut gb_r = vec![0.0; self.head.b_reg.len()];
        let mut gw_c = vec![0.0; self.head.w_cls.len()];
        let mut gb_c = vec![0.0; self.head.b_cls.len()];
        let mut g_core = NervousWaveGrads::zeros(self.core.edge_count());

        let n_tr = train_indices.len();
        let bs = cfg.batch_size.min(n_tr).max(1);
        let mut order: Vec<usize> = train_indices.to_vec();
        let mut cursor = order.len();

        for step in 1..=cfg.steps {
            let sp = StepParamCache::from_core(&self.core, None);
            let frac = (step as f64) / (cfg.steps as f64);
            let mult = if step <= 15 {
                (step as f64) / 15.0
            } else {
                0.10 + 0.90 * 0.5 * (1.0 + (PI * frac).cos())
            };

            gw_e.fill(0.0);
            gb_e.fill(0.0);
            gw_r.fill(0.0);
            gb_r.fill(0.0);
            gw_c.fill(0.0);
            gb_c.fill(0.0);
            g_core.clear();

            let inv_bs = 1.0 / (bs as f64);
            for _ in 0..bs {
                if cursor >= order.len() {
                    rng.shuffle(&mut order);
                    cursor = 0;
                }
                let idx = order[cursor];
                cursor += 1;
                let pair = &domain.pairs[idx];
                let xa = &domain.obs[pair.a];
                let xb = &domain.obs[pair.b];

                let mut u = self.encoder.forward(xa);
                let mut v = self.encoder.forward(xb);
                if cfg.latent_jitter > 0.0 {
                    for d in 0..LATENT_D {
                        u[d] += rng.normal() * cfg.latent_jitter;
                        v[d] += rng.normal() * cfg.latent_jitter;
                    }
                }
                let cache = self.core.forward_with_cache(u, v, &sp);
                let (reg_pred, logits) = self.head.forward(&cache.z_rel);
                let (_loss, dz) = self.head.loss_and_backward(
                    &cache.z_rel,
                    &reg_pred,
                    &logits,
                    &pair.reg,
                    pair.cls,
                    cfg.w_mse,
                    cfg.w_ce,
                    inv_bs,
                    &mut gw_r,
                    &mut gb_r,
                    &mut gw_c,
                    &mut gb_c,
                );
                let (mut du, mut dv) =
                    self.core
                        .backward_with_cache(&cache, &sp, &dz, cfg.mode, Some(&mut g_core));

                if cfg.latent_norm_reg > 0.0 {
                    let target_r2 = 1.44;
                    let ru2: f64 = u.iter().map(|x| x * x).sum();
                    let rv2: f64 = v.iter().map(|x| x * x).sum();
                    let cu = inv_bs * cfg.latent_norm_reg * 2.0 * (ru2 - target_r2);
                    let cv = inv_bs * cfg.latent_norm_reg * 2.0 * (rv2 - target_r2);
                    for d in 0..LATENT_D {
                        du[d] += cu * u[d];
                        dv[d] += cv * v[d];
                    }
                }

                self.encoder.backward(xa, &du, &mut gw_e, &mut gb_e);
                self.encoder.backward(xb, &dv, &mut gw_e, &mut gb_e);
            }

            opt_ew.step(&mut self.encoder.w, &gw_e, cfg.lr_encoder * mult, cfg.wd_encoder, step, 3.0);
            opt_eb.step(&mut self.encoder.b, &gb_e, cfg.lr_encoder * mult, 0.0, step, 3.0);
            if self.head.reg_dim > 0 {
                opt_hw_r.step(&mut self.head.w_reg, &gw_r, cfg.lr_head * mult, cfg.wd_head, step, 3.0);
                opt_hb_r.step(&mut self.head.b_reg, &gb_r, cfg.lr_head * mult, 0.0, step, 3.0);
            }
            if self.head.cls_dim > 0 {
                opt_hw_c.step(&mut self.head.w_cls, &gw_c, cfg.lr_head * mult, cfg.wd_head, step, 3.0);
                opt_hb_c.step(&mut self.head.b_cls, &gb_c, cfg.lr_head * mult, 0.0, step, 3.0);
            }

            match cfg.mode {
                WaveTrainMode::FullTrain => {
                    opt_win.step(&mut self.core.w_in, &g_core.w_in, cfg.lr_core * mult, cfg.wd_core, step, 3.0);
                    opt_pin.step(&mut self.core.phi_in, &g_core.phi_in, cfg.lr_core * mult, 0.0, step, 3.0);
                    opt_alp.step(&mut self.core.alpha_e, &g_core.alpha_e, cfg.lr_core * mult, cfg.wd_core, step, 3.0);
                    if self.core.topo.kind.is_complex_wave() {
                        opt_the.step(&mut self.core.theta_e, &g_core.theta_e, cfg.lr_core * mult, 0.0, step, 3.0);
                        opt_om.step(&mut self.core.omega_n, &g_core.omega_n, cfg.lr_core * mult, 0.0, step, 3.0);
                    }
                    opt_gm.step(&mut self.core.gamma_logit_n, &g_core.gamma_logit_n, cfg.lr_core * mult, cfg.wd_core, step, 3.0);
                }
                WaveTrainMode::MyelinCalOnly => {
                    let cal_start = (cfg.steps * 45) / 100;
                    if step > cal_start && self.core.topo.kind.is_complex_wave() {
                        opt_my.step_spring(
                            &mut self.core.myelin_omega,
                            &g_core.myelin_omega,
                            cfg.lr_myelin * mult,
                            cfg.wd_myelin,
                            0.0,
                            2.0,
                        );
                    }
                }
            }
        }

        if cfg.mode == WaveTrainMode::MyelinCalOnly {
            assert_eq!(
                pre_hash,
                self.core.frozen_synapses_checksum(),
                "FROZEN SYNAPSE VIOLATION during MyelinCalOnly!"
            );
        }
    }
}
