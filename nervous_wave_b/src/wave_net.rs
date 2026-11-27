//! Complex traveling-wave network with Kuramoto Phase-Coherence Selective Lateral Inhibition,
//! compact 4D DomainEncoder -> 16 bipolar phase-coded wave receptors, StepParamCache for 10x speed,
//! exact analytical BPTT (< 1e-5), and two-phase resonant spring auto-calibration for Prueba B.

use crate::domains::DomainData;
use crate::rng::Rng64;
use crate::topology::{EdgeKind, GraphTopology, N_FUT_REC, N_HUBS, N_PAST_REC, N_SPARSE};
use std::f64::consts::PI;

pub const LATENT_D: usize = 4;
pub const MAX_N: usize = N_SPARSE; // 128
pub const T_STEPS: usize = 6;
pub const READOUT_START: usize = 4; // average over t in {4, 5, 6}
pub const READOUT_DIM: usize = N_HUBS * 2; // 32 hubs x [Re(psi_h), |psi_h|^2 - 0.5] = 64
pub const BETA_SAT: f64 = 0.25;
pub const LAMBDA_COH: f64 = 0.45;
pub const EPS_AMP: f64 = 1e-8;
pub const SUSTAIN_DRIVE: f64 = 0.30;

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
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

#[derive(Clone, Debug)]
pub struct Perturbation {
    pub edge_mask: Vec<f64>,   // 1.0 active, 0.0 lesioned (empty = all active)
    pub phase_noise: Vec<f64>, // additive phase perturbation per edge (empty = 0.0)
}

impl Perturbation {
    pub fn none() -> Self {
        Self {
            edge_mask: Vec::new(),
            phase_noise: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdamState {
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub t: u64,
}

impl AdamState {
    pub fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64], lr: f64, wd: f64, clip: f64) {
        self.t += 1;
        let b1: f64 = 0.9;
        let b2: f64 = 0.999;
        let eps: f64 = 1e-8;
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);

        let mut norm2 = 0.0;
        for &g in grads {
            norm2 += g * g;
        }
        let scale = if norm2 > clip * clip {
            clip / norm2.sqrt()
        } else {
            1.0
        };

        for i in 0..params.len() {
            let g = grads[i] * scale;
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * params[i]);
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpringState {
    pub vel: Vec<f64>,
}

impl SpringState {
    pub fn new(n: usize) -> Self {
        Self { vel: vec![0.0; n] }
    }

    /// Damped harmonic resonant spring update around anchor (0.0 for phase, 1.0 for scale).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        params: &mut [f64],
        grads: &[f64],
        anchor: f64,
        lr: f64,
        spring_k: f64,
        damp: f64,
        clip: f64,
        lo: f64,
        hi: f64,
    ) {
        let mut norm2 = 0.0;
        for &g in grads {
            norm2 += g * g;
        }
        let scale = if norm2 > clip * clip {
            clip / norm2.sqrt()
        } else {
            1.0
        };
        for i in 0..params.len() {
            let force = -(grads[i] * scale) - spring_k * (params[i] - anchor);
            self.vel[i] = damp * self.vel[i] + lr * force;
            params[i] = (params[i] + self.vel[i]).clamp(lo, hi);
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepParamCache {
    pub g_e: Vec<f64>,
    pub sig_alpha: Vec<f64>,
    pub c_e: Vec<f64>,
    pub s_e: Vec<f64>,
    pub k_re: Vec<f64>,
    pub k_im: Vec<f64>,
    pub sig_g: [f64; MAX_N],
    pub gamma: [f64; MAX_N],
    pub cn: [f64; MAX_N],
    pub sn: [f64; MAX_N],
    pub sig_tau: [f64; MAX_N],
    pub q_gate: [f64; MAX_N],
}

impl StepParamCache {
    pub fn build(net: &NervousWaveNet, pert: &Perturbation) -> Self {
        let e_len = net.topo.edges.len();
        let n = net.topo.n_nodes;
        let has_mask = !pert.edge_mask.is_empty();
        let has_noise = !pert.phase_noise.is_empty();

        let mut g_e = vec![0.0; e_len];
        let mut sig_alpha = vec![0.0; e_len];
        let mut c_e = vec![1.0; e_len];
        let mut s_e = vec![0.0; e_len];
        let mut k_re = vec![0.0; e_len];
        let mut k_im = vec![0.0; e_len];

        for ei in 0..e_len {
            let mask = if has_mask { pert.edge_mask[ei] } else { 1.0 };
            if mask == 0.0 {
                continue;
            }
            let g = softplus(net.alpha_e[ei]) * mask;
            g_e[ei] = g;
            sig_alpha[ei] = sigmoid(net.alpha_e[ei]) * mask;
            if net.phase_enabled {
                let th = net.theta_e[ei] + if has_noise { pert.phase_noise[ei] } else { 0.0 };
                let (st, ct) = th.sin_cos();
                c_e[ei] = ct;
                s_e[ei] = st;
                k_re[ei] = g * ct;
                k_im[ei] = g * st;
            } else {
                k_re[ei] = g;
            }
        }

        let mut sig_g = [0.0; MAX_N];
        let mut gamma = [0.0; MAX_N];
        let mut cn = [1.0; MAX_N];
        let mut sn = [0.0; MAX_N];
        let mut sig_tau = [0.0; MAX_N];
        let mut q_gate = [0.0; MAX_N];

        for i in 0..n {
            let sg = sigmoid(net.gamma_logit_n[i]);
            sig_g[i] = sg;
            gamma[i] = 0.25 + 0.65 * sg;
            if net.phase_enabled {
                let (s, c) = (net.omega_n[i] + net.myelin_omega[i]).sin_cos();
                cn[i] = c;
                sn[i] = s;
            }
            if net.coherence_gate_enabled {
                let st = sigmoid(net.tau_n[i]);
                sig_tau[i] = st;
                q_gate[i] = LAMBDA_COH * st;
            }
        }

        Self {
            g_e,
            sig_alpha,
            c_e,
            s_e,
            k_re,
            k_im,
            sig_g,
            gamma,
            cn,
            sn,
            sig_tau,
            q_gate,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NervousWaveNet {
    pub topo: GraphTopology,
    pub obs_dim: usize,
    pub reg_dim: usize,
    pub cls_dim: usize,
    pub phase_enabled: bool,
    pub coherence_gate_enabled: bool,

    // Compact sensory adapter (obs_dim -> 4D latent wave coordinates, domain-specific):
    pub w_enc: Vec<f64>, // LATENT_D * obs_dim
    pub b_enc: Vec<f64>, // LATENT_D

    // Frozen nervous tissue parameters (100% bit-exact during transfer):
    pub w_in: Vec<f64>,          // 16 * LATENT_D (receptor wavevectors)
    pub phi_in: Vec<f64>,        // 16 (receptor base phases)
    pub alpha_e: Vec<f64>,       // E synaptic conductance logits (g_e = softplus(alpha_e) > 0)
    pub theta_e: Vec<f64>,       // E axonal conduction phase delays
    pub omega_n: Vec<f64>,       // N intrinsic nodal frequencies
    pub gamma_logit_n: Vec<f64>, // N somatic retention logits
    pub tau_n: Vec<f64>,         // N Kuramoto coherence-inhibition gate logits

    // Transfer resonant spring auto-calibration knobs:
    pub myelin_omega: Vec<f64>,  // N nodal myelin phase calibration knobs (init 0.0)
    pub sensory_scale: Vec<f64>, // 16 receptor gain calibration knobs (init 1.0)
    pub sensory_phase: Vec<f64>, // 16 receptor phase calibration knobs (init 0.0)

    // Linear readout from 64-dim hub features (domain-specific):
    pub w_reg: Vec<f64>, // reg_dim * READOUT_DIM
    pub b_reg: Vec<f64>, // reg_dim
    pub w_cls: Vec<f64>, // cls_dim * READOUT_DIM
    pub b_cls: Vec<f64>, // cls_dim
}

#[derive(Clone, Debug)]
pub struct ForwardTrace {
    pub u_lat: [f64; LATENT_D],
    pub v_lat: [f64; LATENT_D],
    pub past_cos: [f64; N_PAST_REC],
    pub past_sin: [f64; N_PAST_REC],
    pub fut_cos: [f64; N_FUT_REC],
    pub fut_sin: [f64; N_FUT_REC],
    pub psi_re: [[f64; MAX_N]; T_STEPS + 1],
    pub psi_im: [[f64; MAX_N]; T_STEPS + 1],
    pub amp_j: [[f64; MAX_N]; T_STEPS],
    pub i_re: [[f64; MAX_N]; T_STEPS],
    pub i_im: [[f64; MAX_N]; T_STEPS],
    pub s_amp: [[f64; MAX_N]; T_STEPS],
    pub d_incoh: [[f64; MAX_N]; T_STEPS],
    pub pre_re: [[f64; MAX_N]; T_STEPS],
    pub pre_im: [[f64; MAX_N]; T_STEPS],
    pub inv_r: [[f64; MAX_N]; T_STEPS],
    pub feat: [f64; READOUT_DIM],
    pub reg_pred: Vec<f64>,
    pub cls_logits: Vec<f64>,
    pub cls_probs: Vec<f64>,
    pub hub_r_coh: f64,
}

#[derive(Clone, Debug)]
pub struct GradAccum {
    pub w_enc: Vec<f64>,
    pub b_enc: Vec<f64>,
    pub w_in: Vec<f64>,
    pub phi_in: Vec<f64>,
    pub alpha_e: Vec<f64>,
    pub theta_e: Vec<f64>,
    pub omega_n: Vec<f64>,
    pub gamma_logit_n: Vec<f64>,
    pub tau_n: Vec<f64>,
    pub myelin_omega: Vec<f64>,
    pub sensory_scale: Vec<f64>,
    pub sensory_phase: Vec<f64>,
    pub w_reg: Vec<f64>,
    pub b_reg: Vec<f64>,
    pub w_cls: Vec<f64>,
    pub b_cls: Vec<f64>,
}

impl GradAccum {
    pub fn zeros_like(net: &NervousWaveNet) -> Self {
        Self {
            w_enc: vec![0.0; net.w_enc.len()],
            b_enc: vec![0.0; net.b_enc.len()],
            w_in: vec![0.0; net.w_in.len()],
            phi_in: vec![0.0; net.phi_in.len()],
            alpha_e: vec![0.0; net.alpha_e.len()],
            theta_e: vec![0.0; net.theta_e.len()],
            omega_n: vec![0.0; net.omega_n.len()],
            gamma_logit_n: vec![0.0; net.gamma_logit_n.len()],
            tau_n: vec![0.0; net.tau_n.len()],
            myelin_omega: vec![0.0; net.myelin_omega.len()],
            sensory_scale: vec![0.0; net.sensory_scale.len()],
            sensory_phase: vec![0.0; net.sensory_phase.len()],
            w_reg: vec![0.0; net.w_reg.len()],
            b_reg: vec![0.0; net.b_reg.len()],
            w_cls: vec![0.0; net.w_cls.len()],
            b_cls: vec![0.0; net.b_cls.len()],
        }
    }

    pub fn zero_out(&mut self) {
        self.w_enc.fill(0.0);
        self.b_enc.fill(0.0);
        self.w_in.fill(0.0);
        self.phi_in.fill(0.0);
        self.alpha_e.fill(0.0);
        self.theta_e.fill(0.0);
        self.omega_n.fill(0.0);
        self.gamma_logit_n.fill(0.0);
        self.tau_n.fill(0.0);
        self.myelin_omega.fill(0.0);
        self.sensory_scale.fill(0.0);
        self.sensory_phase.fill(0.0);
        self.w_reg.fill(0.0);
        self.b_reg.fill(0.0);
        self.w_cls.fill(0.0);
        self.b_cls.fill(0.0);
    }

    pub fn scale(&mut self, s: f64) {
        for v in &mut self.w_enc { *v *= s; }
        for v in &mut self.b_enc { *v *= s; }
        for v in &mut self.w_in { *v *= s; }
        for v in &mut self.phi_in { *v *= s; }
        for v in &mut self.alpha_e { *v *= s; }
        for v in &mut self.theta_e { *v *= s; }
        for v in &mut self.omega_n { *v *= s; }
        for v in &mut self.gamma_logit_n { *v *= s; }
        for v in &mut self.tau_n { *v *= s; }
        for v in &mut self.myelin_omega { *v *= s; }
        for v in &mut self.sensory_scale { *v *= s; }
        for v in &mut self.sensory_phase { *v *= s; }
        for v in &mut self.w_reg { *v *= s; }
        for v in &mut self.b_reg { *v *= s; }
        for v in &mut self.w_cls { *v *= s; }
        for v in &mut self.b_cls { *v *= s; }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainMode {
    Full,
    ResonantCalOnly,
}

impl NervousWaveNet {
    pub fn new(
        topo: GraphTopology,
        obs_dim: usize,
        reg_dim: usize,
        cls_dim: usize,
        rng: &mut Rng64,
    ) -> Self {
        let n = topo.n_nodes;
        let e = topo.edges.len();
        let phase_enabled = topo.kind.phase_enabled();
        let coherence_gate_enabled = topo.kind.coherence_gate_enabled();

        let enc_scale = (1.0 / obs_dim as f64).sqrt() * 0.85;
        let w_enc = (0..(LATENT_D * obs_dim))
            .map(|_| enc_scale * rng.normal())
            .collect();
        let b_enc = vec![0.0; LATENT_D];

        // 16 sensory receptor wavevectors on 4D sphere (rad in [0.55, 1.35]) + phases
        let mut w_in = vec![0.0; N_PAST_REC * LATENT_D];
        for r in 0..N_PAST_REC {
            let dir = rng.unit_vec(LATENT_D);
            let rad = rng.uniform_range(0.55, 1.35);
            for d in 0..LATENT_D {
                w_in[r * LATENT_D + d] = dir[d] * rad;
            }
        }
        let phi_in = (0..N_PAST_REC)
            .map(|_| rng.uniform_range(-PI, PI))
            .collect();

        // Positive axonal conductances g_e = softplus(alpha_e), scaled fairly by degree:
        // ~0.85 for sparse d_in=8 and ~0.30 for dense d_in=64 (identical wave propagation gain).
        let target_g_base = 2.40 / topo.mean_in_deg.sqrt();
        let mut alpha_e = Vec::with_capacity(e);
        let mut theta_e = Vec::with_capacity(e);
        for edge in &topo.edges {
            let mult = match edge.kind {
                EdgeKind::Commissure => 1.20,
                EdgeKind::Afferent | EdgeKind::HubProjection => 1.05,
                _ => 0.95,
            };
            let g0 = (target_g_base * mult).max(0.08);
            let base_a = (g0.exp() - 1.0).max(1e-4).ln();
            alpha_e.push(base_a + 0.15 * rng.normal());
            theta_e.push(if phase_enabled {
                rng.uniform_range(-PI, PI)
            } else {
                0.0
            });
        }

        let mut omega_n = Vec::with_capacity(n);
        let mut gamma_logit_n = Vec::with_capacity(n);
        let tau_n = vec![0.0; n];
        for _ in 0..n {
            omega_n.push(if phase_enabled {
                rng.uniform_range(-0.45, 0.45)
            } else {
                0.0
            });
            gamma_logit_n.push(0.20 + 0.20 * rng.normal());
        }

        let ro_scale = (1.0 / READOUT_DIM as f64).sqrt() * 0.45;
        let w_reg = (0..(reg_dim * READOUT_DIM))
            .map(|_| ro_scale * rng.normal())
            .collect();
        let w_cls = (0..(cls_dim * READOUT_DIM))
            .map(|_| ro_scale * rng.normal())
            .collect();

        Self {
            topo,
            obs_dim,
            reg_dim,
            cls_dim,
            phase_enabled,
            coherence_gate_enabled,
            w_enc,
            b_enc,
            w_in,
            phi_in,
            alpha_e,
            theta_e,
            omega_n,
            gamma_logit_n,
            tau_n,
            myelin_omega: vec![0.0; n],
            sensory_scale: vec![1.0; N_PAST_REC],
            sensory_phase: vec![0.0; N_PAST_REC],
            w_reg,
            b_reg: vec![0.0; reg_dim],
            w_cls,
            b_cls: vec![0.0; cls_dim],
        }
    }

    /// Bit-exact IEEE-754 FNV-1a checksum over ALL frozen nervous tissue parameters.
    pub fn frozen_tissue_checksum(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mix = |h: &mut u64, x: f64| {
            let bits = x.to_bits();
            for b in bits.to_le_bytes() {
                *h ^= b as u64;
                *h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for &v in &self.w_in { mix(&mut h, v); }
        for &v in &self.phi_in { mix(&mut h, v); }
        for &v in &self.alpha_e { mix(&mut h, v); }
        for &v in &self.theta_e { mix(&mut h, v); }
        for &v in &self.omega_n { mix(&mut h, v); }
        for &v in &self.gamma_logit_n { mix(&mut h, v); }
        for &v in &self.tau_n { mix(&mut h, v); }
        h
    }

    /// Clone trained tissue onto a new target domain, freezing w_in, phi_in, alpha_e, theta_e,
    /// omega_n, gamma_logit_n, tau_n 100% bit-exact.
    pub fn clone_for_transfer(
        &self,
        new_obs_dim: usize,
        new_reg_dim: usize,
        new_cls_dim: usize,
        permute_synapses: bool,
        rng: &mut Rng64,
    ) -> Self {
        let mut alpha_e = self.alpha_e.clone();
        let mut theta_e = self.theta_e.clone();
        if permute_synapses {
            let e = alpha_e.len();
            let mut perm: Vec<usize> = (0..e).collect();
            rng.shuffle(&mut perm);
            let old_a = alpha_e.clone();
            let old_t = theta_e.clone();
            for i in 0..e {
                alpha_e[i] = old_a[perm[i]];
                theta_e[i] = old_t[perm[i]];
            }
        }

        let enc_scale = (1.0 / new_obs_dim as f64).sqrt() * 0.85;
        let w_enc = (0..(LATENT_D * new_obs_dim))
            .map(|_| enc_scale * rng.normal())
            .collect();
        let b_enc = vec![0.0; LATENT_D];

        let ro_scale = (1.0 / READOUT_DIM as f64).sqrt() * 0.45;
        let w_reg = (0..(new_reg_dim * READOUT_DIM))
            .map(|_| ro_scale * rng.normal())
            .collect();
        let w_cls = (0..(new_cls_dim * READOUT_DIM))
            .map(|_| ro_scale * rng.normal())
            .collect();

        Self {
            topo: self.topo.clone(),
            obs_dim: new_obs_dim,
            reg_dim: new_reg_dim,
            cls_dim: new_cls_dim,
            phase_enabled: self.phase_enabled,
            coherence_gate_enabled: self.coherence_gate_enabled,
            w_enc,
            b_enc,
            w_in: self.w_in.clone(),
            phi_in: self.phi_in.clone(),
            alpha_e,
            theta_e,
            omega_n: self.omega_n.clone(),
            gamma_logit_n: self.gamma_logit_n.clone(),
            tau_n: self.tau_n.clone(),
            myelin_omega: vec![0.0; self.topo.n_nodes],
            sensory_scale: vec![1.0; N_PAST_REC],
            sensory_phase: vec![0.0; N_PAST_REC],
            w_reg,
            b_reg: vec![0.0; new_reg_dim],
            w_cls,
            b_cls: vec![0.0; new_cls_dim],
        }
    }

    #[allow(dead_code)]
    pub fn forward(&self, obs_a: &[f64], obs_b: &[f64], pert: &Perturbation) -> ForwardTrace {
        let cache = StepParamCache::build(self, pert);
        self.forward_masked(obs_a, obs_b, &cache, true, true)
    }

    pub fn forward_cached(&self, obs_a: &[f64], obs_b: &[f64], cache: &StepParamCache) -> ForwardTrace {
        self.forward_masked(obs_a, obs_b, cache, true, true)
    }

    pub fn forward_masked(
        &self,
        obs_a: &[f64],
        obs_b: &[f64],
        cache: &StepParamCache,
        enable_past: bool,
        enable_fut: bool,
    ) -> ForwardTrace {
        let n = self.topo.n_nodes;
        let d = self.obs_dim;
        let syn_scale = self.topo.syn_scale;
        let syn_scale2 = syn_scale * syn_scale;

        // 1) Compact 4D sensory encoding
        let mut u_lat = [0.0; LATENT_D];
        let mut v_lat = [0.0; LATENT_D];
        for k in 0..LATENT_D {
            let row = k * d;
            let mut ua = self.b_enc[k];
            let mut vb = self.b_enc[k];
            for j in 0..d {
                let w = self.w_enc[row + j];
                ua += w * obs_a[j];
                vb += w * obs_b[j];
            }
            u_lat[k] = ua;
            v_lat[k] = vb;
        }

        // 2) Phase-coded plane-wave sensory transduction into 16 Past + 16 Future receptors
        let mut u_re = [0.0; MAX_N];
        let mut u_im = [0.0; MAX_N];
        let mut past_cos = [0.0; N_PAST_REC];
        let mut past_sin = [0.0; N_PAST_REC];
        let mut fut_cos = [0.0; N_FUT_REC];
        let mut fut_sin = [0.0; N_FUT_REC];

        for r in 0..N_PAST_REC {
            let row = r * LATENT_D;
            let mut pa = self.phi_in[r] + self.sensory_phase[r];
            let mut pb = self.phi_in[r] + self.sensory_phase[r];
            for k in 0..LATENT_D {
                let wk = self.w_in[row + k];
                pa += wk * u_lat[k];
                pb += wk * v_lat[k];
            }
            let (sa, ca) = pa.sin_cos();
            let (sb, cb) = pb.sin_cos();
            past_cos[r] = ca;
            past_sin[r] = sa;
            fut_cos[r] = cb;
            fut_sin[r] = sb;

            let sc = self.sensory_scale[r];
            let np = self.topo.past_receptors[r];
            let nf = self.topo.future_receptors[r];
            if enable_past {
                u_re[np] = sc * ca;
                if self.phase_enabled {
                    u_im[np] = sc * sa;
                }
            }
            if enable_fut {
                u_re[nf] = sc * cb;
                if self.phase_enabled {
                    u_im[nf] = sc * sb;
                }
            }
        }

        let mut psi_re = [[0.0; MAX_N]; T_STEPS + 1];
        let mut psi_im = [[0.0; MAX_N]; T_STEPS + 1];
        let mut amp_j = [[0.0; MAX_N]; T_STEPS];
        let mut i_re = [[0.0; MAX_N]; T_STEPS];
        let mut i_im = [[0.0; MAX_N]; T_STEPS];
        let mut s_amp = [[0.0; MAX_N]; T_STEPS];
        let mut d_incoh = [[0.0; MAX_N]; T_STEPS];
        let mut pre_re = [[0.0; MAX_N]; T_STEPS];
        let mut pre_im = [[0.0; MAX_N]; T_STEPS];
        let mut inv_r = [[0.0; MAX_N]; T_STEPS];

        // Strictly causal sensory pulse at t = 0 into sensory receptors only
        psi_re[0][..n].copy_from_slice(&u_re[..n]);
        psi_im[0][..n].copy_from_slice(&u_im[..n]);

        let mut hub_r_coh_sum = 0.0;
        let mut hub_r_coh_cnt = 0usize;

        for t in 0..T_STEPS {
            for j in 0..n {
                let pr = psi_re[t][j];
                let pi = psi_im[t][j];
                amp_j[t][j] = (pr * pr + pi * pi + EPS_AMP).sqrt();
            }

            // Synaptic traveling-wave sum I_i and Kuramoto amplitude sum S_i
            for (ei, edge) in self.topo.edges.iter().enumerate() {
                let g = cache.g_e[ei];
                if g == 0.0 {
                    continue;
                }
                let j = edge.src;
                let dst = edge.dst;
                let pr = psi_re[t][j];
                let pi = psi_im[t][j];
                i_re[t][dst] += cache.k_re[ei] * pr - cache.k_im[ei] * pi;
                i_im[t][dst] += cache.k_im[ei] * pr + cache.k_re[ei] * pi;
                s_amp[t][dst] += g * amp_j[t][j];
            }

            let drive_scale = SUSTAIN_DRIVE;
            for i in 0..n {
                let pr = psi_re[t][i];
                let pi = psi_im[t][i];
                let gm = cache.gamma[i];
                let self_r = gm * (cache.cn[i] * pr - cache.sn[i] * pi);
                let self_i = gm * (cache.sn[i] * pr + cache.cn[i] * pi);
                let sr = self_r + syn_scale * i_re[t][i] + drive_scale * u_re[i];
                let si = self_i + syn_scale * i_im[t][i] + drive_scale * u_im[i];
                pre_re[t][i] = sr;
                pre_im[t][i] = si;

                let s2 = s_amp[t][i] * s_amp[t][i];
                let i2 = i_re[t][i] * i_re[t][i] + i_im[t][i] * i_im[t][i];
                let d_inc = syn_scale2 * (s2 - i2).max(0.0);
                d_incoh[t][i] = d_inc;

                let v = 1.0 + BETA_SAT * (sr * sr + si * si) + cache.q_gate[i] * d_inc;
                let r = 1.0 / v.sqrt();
                inv_r[t][i] = r;
                psi_re[t + 1][i] = r * sr;
                psi_im[t + 1][i] = r * si;
            }

            if t + 1 >= READOUT_START {
                for &h in &self.topo.hubs {
                    let i_mag = (i_re[t][h] * i_re[t][h] + i_im[t][h] * i_im[t][h]).sqrt();
                    hub_r_coh_sum += i_mag / (s_amp[t][h] + 1e-6);
                    hub_r_coh_cnt += 1;
                }
            }
        }

        let win = (T_STEPS - READOUT_START + 1) as f64;
        let mut feat = [0.0; READOUT_DIM];
        for t in READOUT_START..=T_STEPS {
            for (hi, &h) in self.topo.hubs.iter().enumerate() {
                let pr = psi_re[t][h];
                let pi = psi_im[t][h];
                feat[2 * hi] += pr / win;
                feat[2 * hi + 1] += (pr * pr + pi * pi - 0.5) / win;
            }
        }

        let mut reg_pred = vec![0.0; self.reg_dim];
        for o in 0..self.reg_dim {
            let mut acc = self.b_reg[o];
            let row = o * READOUT_DIM;
            for f in 0..READOUT_DIM {
                acc += self.w_reg[row + f] * feat[f];
            }
            reg_pred[o] = acc;
        }

        let mut cls_logits = vec![0.0; self.cls_dim];
        let mut max_l = f64::NEG_INFINITY;
        for c in 0..self.cls_dim {
            let mut acc = self.b_cls[c];
            let row = c * READOUT_DIM;
            for f in 0..READOUT_DIM {
                acc += self.w_cls[row + f] * feat[f];
            }
            cls_logits[c] = acc;
            if acc > max_l {
                max_l = acc;
            }
        }
        let mut sum_e = 0.0;
        let mut cls_probs = vec![0.0; self.cls_dim];
        for c in 0..self.cls_dim {
            let e = (cls_logits[c] - max_l).exp();
            cls_probs[c] = e;
            sum_e += e;
        }
        for c in 0..self.cls_dim {
            cls_probs[c] /= sum_e;
        }

        ForwardTrace {
            u_lat,
            v_lat,
            past_cos,
            past_sin,
            fut_cos,
            fut_sin,
            psi_re,
            psi_im,
            amp_j,
            i_re,
            i_im,
            s_amp,
            d_incoh,
            pre_re,
            pre_im,
            inv_r,
            feat,
            reg_pred,
            cls_logits,
            cls_probs,
            hub_r_coh: if hub_r_coh_cnt > 0 {
                hub_r_coh_sum / hub_r_coh_cnt as f64
            } else {
                0.0
            },
        }
    }

    /// Exact analytical BPTT using prebuilt `StepParamCache`. Returns scalar loss.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_cached(
        &self,
        obs_a: &[f64],
        obs_b: &[f64],
        target_reg: &[f64],
        target_cls: usize,
        reg_weight: f64,
        cls_weight: f64,
        cache: &StepParamCache,
        trace: &ForwardTrace,
        grads: &mut GradAccum,
    ) -> f64 {
        let n = self.topo.n_nodes;
        let d = self.obs_dim;
        let syn_scale = self.topo.syn_scale;
        let syn_scale2 = syn_scale * syn_scale;

        let mut loss = 0.0;
        let mut d_feat = [0.0; READOUT_DIM];

        if reg_weight > 0.0 && self.reg_dim > 0 {
            let inv_r = reg_weight / self.reg_dim as f64;
            for o in 0..self.reg_dim {
                let diff = trace.reg_pred[o] - target_reg[o];
                loss += 0.5 * inv_r * diff * diff;
                let g_o = inv_r * diff;
                grads.b_reg[o] += g_o;
                let row = o * READOUT_DIM;
                for f in 0..READOUT_DIM {
                    grads.w_reg[row + f] += g_o * trace.feat[f];
                    d_feat[f] += g_o * self.w_reg[row + f];
                }
            }
        }

        if cls_weight > 0.0 && self.cls_dim > 0 {
            loss += -cls_weight * (trace.cls_probs[target_cls].max(1e-12)).ln();
            for c in 0..self.cls_dim {
                let y = if c == target_cls { 1.0 } else { 0.0 };
                let g_c = cls_weight * (trace.cls_probs[c] - y);
                grads.b_cls[c] += g_c;
                let row = c * READOUT_DIM;
                for f in 0..READOUT_DIM {
                    grads.w_cls[row + f] += g_c * trace.feat[f];
                    d_feat[f] += g_c * self.w_cls[row + f];
                }
            }
        }

        let win = (T_STEPS - READOUT_START + 1) as f64;
        let mut d_psi_re = [0.0; MAX_N];
        let mut d_psi_im = [0.0; MAX_N];
        let mut d_u_re = [0.0; MAX_N];
        let mut d_u_im = [0.0; MAX_N];

        for t in (0..T_STEPS).rev() {
            if t + 1 >= READOUT_START {
                for (hi, &h) in self.topo.hubs.iter().enumerate() {
                    let pr = trace.psi_re[t + 1][h];
                    let pi = trace.psi_im[t + 1][h];
                    let g_re = d_feat[2 * hi] / win;
                    let g_en = d_feat[2 * hi + 1] / win;
                    d_psi_re[h] += g_re + 2.0 * pr * g_en;
                    d_psi_im[h] += 2.0 * pi * g_en;
                }
            }

            let mut d_i_re = [0.0; MAX_N];
            let mut d_i_im = [0.0; MAX_N];
            let mut d_s_amp = [0.0; MAX_N];
            let mut next_d_psi_re = [0.0; MAX_N];
            let mut next_d_psi_im = [0.0; MAX_N];

            let drive_scale = SUSTAIN_DRIVE;

            for i in 0..n {
                let sr = trace.pre_re[t][i];
                let si = trace.pre_im[t][i];
                let r = trace.inv_r[t][i];
                let dpr = d_psi_re[i];
                let dpi = d_psi_im[i];

                let d_r = dpr * sr + dpi * si;
                let d_v = -0.5 * r * r * r * d_r;

                let d_sr = r * dpr + 2.0 * BETA_SAT * d_v * sr;
                let d_si = r * dpi + 2.0 * BETA_SAT * d_v * si;

                if self.coherence_gate_enabled {
                    let d_inc = trace.d_incoh[t][i];
                    let d_q = d_v * d_inc;
                    grads.tau_n[i] += d_q * LAMBDA_COH * cache.sig_tau[i] * (1.0 - cache.sig_tau[i]);

                    let d_d = d_v * cache.q_gate[i];
                    d_s_amp[i] = 2.0 * syn_scale2 * d_d * trace.s_amp[t][i];
                    d_i_re[i] = syn_scale * d_sr - 2.0 * syn_scale2 * d_d * trace.i_re[t][i];
                    d_i_im[i] = syn_scale * d_si - 2.0 * syn_scale2 * d_d * trace.i_im[t][i];
                } else {
                    d_i_re[i] = syn_scale * d_sr;
                    d_i_im[i] = syn_scale * d_si;
                }

                if drive_scale > 0.0 {
                    d_u_re[i] += drive_scale * d_sr;
                    d_u_im[i] += drive_scale * d_si;
                }

                let pr = trace.psi_re[t][i];
                let pi = trace.psi_im[t][i];
                let rot_r = cache.cn[i] * pr - cache.sn[i] * pi;
                let rot_i = cache.sn[i] * pr + cache.cn[i] * pi;
                let d_gamma = d_sr * rot_r + d_si * rot_i;
                grads.gamma_logit_n[i] += d_gamma * 0.65 * cache.sig_g[i] * (1.0 - cache.sig_g[i]);

                if self.phase_enabled {
                    let d_phi = cache.gamma[i] * (-d_sr * rot_i + d_si * rot_r);
                    grads.omega_n[i] += d_phi;
                    grads.myelin_omega[i] += d_phi;
                }

                next_d_psi_re[i] += cache.gamma[i] * (cache.cn[i] * d_sr + cache.sn[i] * d_si);
                next_d_psi_im[i] += cache.gamma[i] * (-cache.sn[i] * d_sr + cache.cn[i] * d_si);
            }

            // Edge gradients (coherent wave branch + positive-conductance Kuramoto amplitude branch)
            for (ei, edge) in self.topo.edges.iter().enumerate() {
                let g = cache.g_e[ei];
                if g == 0.0 {
                    continue;
                }
                let j = edge.src;
                let dst = edge.dst;
                let pr = trace.psi_re[t][j];
                let pi = trace.psi_im[t][j];
                let wr = cache.c_e[ei] * pr - cache.s_e[ei] * pi;
                let wi = cache.s_e[ei] * pr + cache.c_e[ei] * pi;
                let dir = d_i_re[dst];
                let dii = d_i_im[dst];

                let mut d_g = dir * wr + dii * wi;
                if self.phase_enabled {
                    grads.theta_e[ei] += g * (-dir * wi + dii * wr);
                }

                next_d_psi_re[j] += cache.k_re[ei] * dir + cache.k_im[ei] * dii;
                next_d_psi_im[j] += -cache.k_im[ei] * dir + cache.k_re[ei] * dii;

                let ds = d_s_amp[dst];
                if ds != 0.0 {
                    let aj = trace.amp_j[t][j];
                    d_g += ds * aj;
                    let coeff = ds * g / aj;
                    next_d_psi_re[j] += coeff * pr;
                    next_d_psi_im[j] += coeff * pi;
                }

                grads.alpha_e[ei] += d_g * cache.sig_alpha[ei];
            }

            d_psi_re = next_d_psi_re;
            d_psi_im = next_d_psi_im;
        }

        // Initial injection at t = 0: psi[0] = u
        for i in 0..n {
            d_u_re[i] += d_psi_re[i];
            d_u_im[i] += d_psi_im[i];
        }

        // Backprop through bipolar plane-wave receptors -> 4D DomainEncoder
        let mut d_u_lat = [0.0; LATENT_D];
        let mut d_v_lat = [0.0; LATENT_D];

        for r in 0..N_PAST_REC {
            let np = self.topo.past_receptors[r];
            let nf = self.topo.future_receptors[r];
            let sc = self.sensory_scale[r];
            let ca = trace.past_cos[r];
            let sa = trace.past_sin[r];
            let cb = trace.fut_cos[r];
            let sb = trace.fut_sin[r];

            let (d_pa, d_pb, d_sc) = if self.phase_enabled {
                let dpa = sc * (-d_u_re[np] * sa + d_u_im[np] * ca);
                let dpb = sc * (-d_u_re[nf] * sb + d_u_im[nf] * cb);
                let dsc = (d_u_re[np] * ca + d_u_im[np] * sa) + (d_u_re[nf] * cb + d_u_im[nf] * sb);
                (dpa, dpb, dsc)
            } else {
                let dpa = sc * (-d_u_re[np] * sa);
                let dpb = sc * (-d_u_re[nf] * sb);
                let dsc = d_u_re[np] * ca + d_u_re[nf] * cb;
                (dpa, dpb, dsc)
            };

            grads.sensory_scale[r] += d_sc;
            grads.sensory_phase[r] += d_pa + d_pb;
            grads.phi_in[r] += d_pa + d_pb;

            let row = r * LATENT_D;
            for k in 0..LATENT_D {
                grads.w_in[row + k] += d_pa * trace.u_lat[k] + d_pb * trace.v_lat[k];
                let wk = self.w_in[row + k];
                d_u_lat[k] += d_pa * wk;
                d_v_lat[k] += d_pb * wk;
            }
        }

        for k in 0..LATENT_D {
            let du = d_u_lat[k];
            let dv = d_v_lat[k];
            grads.b_enc[k] += du + dv;
            let row = k * d;
            for j in 0..d {
                grads.w_enc[row + j] += du * obs_a[j] + dv * obs_b[j];
            }
        }

        loss
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        domain: &DomainData,
        train_indices: &[usize],
        mode: TrainMode,
        epochs: usize,
        batch_size: usize,
        lr: f64,
        reg_weight: f64,
        cls_weight: f64,
        rng: &mut Rng64,
    ) {
        let mut opt_wenc = AdamState::new(self.w_enc.len());
        let mut opt_benc = AdamState::new(self.b_enc.len());
        let mut opt_win = AdamState::new(self.w_in.len());
        let mut opt_phiin = AdamState::new(self.phi_in.len());
        let mut opt_alpha = AdamState::new(self.alpha_e.len());
        let mut opt_theta = AdamState::new(self.theta_e.len());
        let mut opt_omega = AdamState::new(self.omega_n.len());
        let mut opt_gamma = AdamState::new(self.gamma_logit_n.len());
        let mut opt_tau = AdamState::new(self.tau_n.len());
        let mut opt_wreg = AdamState::new(self.w_reg.len());
        let mut opt_breg = AdamState::new(self.b_reg.len());
        let mut opt_wcls = AdamState::new(self.w_cls.len());
        let mut opt_bcls = AdamState::new(self.b_cls.len());

        let mut spr_myelin = SpringState::new(self.myelin_omega.len());
        let mut spr_sscale = SpringState::new(self.sensory_scale.len());
        let mut spr_sphase = SpringState::new(self.sensory_phase.len());

        let mut grads = GradAccum::zeros_like(self);
        let mut order = train_indices.to_vec();
        let pert = Perturbation::none();
        let warmup_epochs = (epochs / 4).max(1);

        for ep in 0..epochs {
            rng.shuffle(&mut order);
            let frac = ep as f64 / epochs.max(1) as f64;
            let lr_cur = lr * (0.15 + 0.85 * 0.5 * (1.0 + (PI * frac).cos()));

            for chunk in order.chunks(batch_size.max(1)) {
                grads.zero_out();
                let cache = StepParamCache::build(self, &pert);
                for &idx in chunk {
                    let p = &domain.pairs[idx];
                    let oa = &domain.obs[p.a];
                    let ob = &domain.obs[p.b];
                    let tr = self.forward_cached(oa, ob, &cache);
                    self.backward_cached(
                        oa,
                        ob,
                        &p.reg,
                        p.cls,
                        reg_weight,
                        cls_weight,
                        &cache,
                        &tr,
                        &mut grads,
                    );
                }
                grads.scale(1.0 / chunk.len() as f64);

                match mode {
                    TrainMode::Full => {
                        opt_wenc.step(&mut self.w_enc, &grads.w_enc, lr_cur * 1.15, 8.0e-4, 3.0);
                        opt_benc.step(&mut self.b_enc, &grads.b_enc, lr_cur * 1.15, 0.0, 3.0);
                        opt_win.step(&mut self.w_in, &grads.w_in, lr_cur, 5.0e-4, 3.0);
                        opt_phiin.step(&mut self.phi_in, &grads.phi_in, lr_cur, 0.0, 3.0);
                        opt_alpha.step(&mut self.alpha_e, &grads.alpha_e, lr_cur, 3.0e-4, 3.0);
                        if self.phase_enabled {
                            opt_theta.step(&mut self.theta_e, &grads.theta_e, lr_cur * 1.4, 0.0, 3.0);
                            opt_omega.step(&mut self.omega_n, &grads.omega_n, lr_cur, 0.0, 3.0);
                        }
                        opt_gamma.step(&mut self.gamma_logit_n, &grads.gamma_logit_n, lr_cur, 3.0e-4, 3.0);
                        if self.coherence_gate_enabled {
                            opt_tau.step(&mut self.tau_n, &grads.tau_n, lr_cur, 3.0e-4, 3.0);
                        }
                        opt_wreg.step(&mut self.w_reg, &grads.w_reg, lr_cur * 1.25, 2.5e-3, 3.0);
                        opt_breg.step(&mut self.b_reg, &grads.b_reg, lr_cur * 1.25, 0.0, 3.0);
                        opt_wcls.step(&mut self.w_cls, &grads.w_cls, lr_cur * 1.25, 2.5e-3, 3.0);
                        opt_bcls.step(&mut self.b_cls, &grads.b_cls, lr_cur * 1.25, 0.0, 3.0);
                    }
                    TrainMode::ResonantCalOnly => {
                        // NEVER touch w_in, phi_in, alpha_e, theta_e, omega_n, gamma_logit_n, tau_n!
                        opt_wenc.step(&mut self.w_enc, &grads.w_enc, lr_cur * 1.1, 1.5e-2, 3.0);
                        opt_benc.step(&mut self.b_enc, &grads.b_enc, lr_cur * 1.1, 0.0, 3.0);
                        opt_wreg.step(&mut self.w_reg, &grads.w_reg, lr_cur * 1.2, 2.5e-2, 3.0);
                        opt_breg.step(&mut self.b_reg, &grads.b_reg, lr_cur * 1.2, 0.0, 3.0);
                        opt_wcls.step(&mut self.w_cls, &grads.w_cls, lr_cur * 1.2, 2.5e-2, 3.0);
                        opt_bcls.step(&mut self.b_cls, &grads.b_cls, lr_cur * 1.2, 0.0, 3.0);

                        if ep >= warmup_epochs {
                            if self.phase_enabled {
                                spr_myelin.step(
                                    &mut self.myelin_omega,
                                    &grads.myelin_omega,
                                    0.0,
                                    lr_cur * 2.0,
                                    0.025,
                                    0.78,
                                    3.0,
                                    -PI,
                                    PI,
                                );
                            }
                            spr_sphase.step(
                                &mut self.sensory_phase,
                                &grads.sensory_phase,
                                0.0,
                                lr_cur * 1.5,
                                0.025,
                                0.78,
                                3.0,
                                -PI,
                                PI,
                            );
                            spr_sscale.step(
                                &mut self.sensory_scale,
                                &grads.sensory_scale,
                                1.0,
                                lr_cur * 1.2,
                                0.05,
                                0.78,
                                3.0,
                                0.35,
                                2.00,
                            );
                        }
                    }
                }
            }
        }
    }
}
