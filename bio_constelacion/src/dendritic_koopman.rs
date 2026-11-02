//! Neurona Híbrida Piramidal Dendrítica-Koopman (`DendriticKoopmanNeuron`).
//!
//! Combina lo mejor de las dos arquitecturas:
//! 1. **De CONSTELACIÓN v4 (`Neuron`)**:
//!    - Bloques $2 \times 2$ de rotación-escala de Koopman contractivos $K_j = \rho_j R(\omega_j)$
//!      con $\rho_j \in [0.90, 0.99]$ que garantizan radio espectral $\sigma_{\max}(K) < 1$.
//!    - Levantamiento no-lineal monótono tipo Brenier $\psi(v) = v + \eta_b R^\top \sigma(R v + b)$
//!      con sensibilidad dendrítica local $J_\psi^\top$ acotada analíticamente.
//! 2. **De NEOCORTEX (`PyramidalNeuron`, Larkum 2013; Sacramento et al., 2018; Payeur et al., 2021)**:
//!    - Segregación biofísica en **3 compartimentos** (Basal $\mathbf{v}_B \in \mathbb{R}^m$,
//!      Apical $\mathbf{v}_A \in \mathbb{R}^m$, Soma $\mathbf{u}_S \in \mathbb{R}^m$).
//!    - Potencial de meseta de $\text{Ca}^{2+}$ apical ($p_{\text{Ca}} \in (0, 1)$) modulado por
//!      cancelación inhibitoria de células de Martinotti (**SST+**) y desinhibición **VIP+**.
//!    - Acoplamiento asociativo **BAC** (*Backpropagating Action Potential Activated $\text{Ca}^{2+}$*)
//!      y regulación homeostática **BCM** (*Bienenstock, Cooper & Munro, 1982; Turrigiano & Nelson, 2004*).

#![allow(clippy::needless_range_loop)]

use crate::rng::Xoshiro256StarStar;

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Unidad Piramidal Multicompartimental con Dinámica Intrínseca de Koopman.
#[derive(Clone, Debug)]
pub struct DendriticKoopmanNeuron {
    pub m: usize,
    pub h: usize,
    /// Radios de contracción por bloque 2D ($\rho_j \in [0.90, 0.99]$).
    pub rho: Vec<f64>,
    /// Frecuencias angulares intrínsecas ($\omega_j \in [-\pi/4, \pi/4]$).
    pub omega: Vec<f64>,
    /// Bloques ortogonales escalados $2 \times 2$: `[r cos w, -r sin w, r sin w, r cos w]`.
    pub k_blocks: Vec<[f64; 4]>,
    /// Proyecciones dendríticas intrínsecas $R \in \mathbb{R}^{h \times m}$ y sesgos $b \in \mathbb{R}^h$.
    pub r_proj: Vec<f64>,
    pub b_proj: Vec<f64>,
    pub eta_b: f64,
    /// Excitabilidad intrínseca homeostática por canal ($b_{\text{homeo}} \in \mathbb{R}^m$).
    pub intrinsic_excitability: Vec<f64>,
    /// Umbral deslizante BCM ($\theta_M$) y tasa media de actividad.
    pub bcm_threshold: f64,
    pub avg_activity: f64,
    /// Peso local plástico Piramidal -> Interneurona SST+ (Sacramento et al., 2018).
    pub w_sst_cancel: Vec<f64>,
}

/// Estado compartimental instantáneo por punto espacial de una neurona híbrida.
#[derive(Clone, Debug)]
pub struct CompartmentalUnitState {
    /// Potencial basal ($\mathbf{v}_B \in \mathbb{R}^m$): flujo aferente *bottom-up* / estado previo.
    pub v_basal: Vec<f64>,
    /// Potencial apical ($\mathbf{v}_A \in \mathbb{R}^m$): error/contexto *top-down* neto tras SST+.
    pub v_apical: Vec<f64>,
    /// Potencial somático ($\mathbf{u}_S \in \mathbb{R}^m$) integrado con inhibición perisomática PV+.
    pub u_soma: Vec<f64>,
    /// Activaciones ocultas del levantamiento dendrítico $\mathbf{s} = \sigma(R K \mathbf{u}_S + b)$.
    pub s_gate: Vec<f64>,
    /// Probabilidad de ráfaga por meseta de calcio apical ($p_{\text{Ca}} \in (0, 1)$).
    pub ca_plateau: f64,
}

impl CompartmentalUnitState {
    pub fn fresh(m: usize, h: usize) -> Self {
        Self {
            v_basal: vec![0.0; m],
            v_apical: vec![0.0; m],
            u_soma: vec![0.0; m],
            s_gate: vec![0.5; h],
            ca_plateau: 0.5,
        }
    }

    pub fn reset(&mut self) {
        self.v_basal.fill(0.0);
        self.v_apical.fill(0.0);
        self.u_soma.fill(0.0);
        self.s_gate.fill(0.5);
        self.ca_plateau = 0.5;
    }
}

impl DendriticKoopmanNeuron {
    pub fn new(rng: &mut Xoshiro256StarStar, m: usize, h: usize, eta_b: f64) -> Self {
        assert_eq!(m % 2, 0, "La dimensión m de bloques Koopman debe ser par");
        let nb = m / 2;
        let mut rho = Vec::with_capacity(nb);
        let mut omega = Vec::with_capacity(nb);
        let mut k_blocks = Vec::with_capacity(nb);

        for _ in 0..nb {
            let r = rng.gen_range(0.90, 0.99);
            let w = rng.gen_range(-std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_4);
            rho.push(r);
            omega.push(w);
            k_blocks.push([r * w.cos(), -r * w.sin(), r * w.sin(), r * w.cos()]);
        }

        let sd = (1.0 / (h.max(1) as f64)).sqrt();
        let r_proj = (0..h * m).map(|_| rng.randn() * sd).collect();
        let b_proj = (0..h).map(|_| rng.gen_range(-1.0, 1.0)).collect();

        Self {
            m,
            h,
            rho,
            omega,
            k_blocks,
            r_proj,
            b_proj,
            eta_b,
            intrinsic_excitability: vec![0.0; m],
            bcm_threshold: 0.10,
            avg_activity: 0.25,
            w_sst_cancel: vec![0.05; m],
        }
    }

    #[inline]
    pub fn k_mul(&self, x: &[f64], out: &mut [f64]) {
        for j in 0..self.m / 2 {
            let kb = &self.k_blocks[j];
            out[2 * j] = kb[0] * x[2 * j] + kb[1] * x[2 * j + 1];
            out[2 * j + 1] = kb[2] * x[2 * j] + kb[3] * x[2 * j + 1];
        }
    }

    #[inline]
    pub fn kt_mul(&self, x: &[f64], out: &mut [f64]) {
        for j in 0..self.m / 2 {
            let kb = &self.k_blocks[j];
            out[2 * j] = kb[0] * x[2 * j] + kb[2] * x[2 * j + 1];
            out[2 * j + 1] = kb[1] * x[2 * j + 1] + kb[3] * x[2 * j + 1];
        }
    }

    /// Evalúa el operador generativo intrínseco $\psi(K \mu)$ modulado por el acoplamiento
    /// dendrítico BAC y la excitabilidad homeostática.
    pub fn forward_lifted(&self, mu: &[f64], ca_plateau: f64, s_out: &mut [f64], out: &mut [f64]) {
        let mut kv = vec![0.0; self.m];
        self.k_mul(mu, &mut kv);
        for j in 0..self.m {
            kv[j] += self.intrinsic_excitability[j];
        }

        for i in 0..self.h {
            let row = &self.r_proj[i * self.m..(i + 1) * self.m];
            let mut acc = self.b_proj[i];
            for j in 0..self.m {
                acc += row[j] * kv[j];
            }
            s_out[i] = sigmoid(acc);
        }

        // Ganancia asociativa BAC (Larkum, 2013): la meseta de Ca2+ apical modula suavemente
        // la amplitud de las ramas no-lineales sin romper la monotonicidad ni el radio espectral.
        let bac_gain = 1.0 + 0.15 * (2.0 * ca_plateau - 1.0);
        let eff_eta_b = self.eta_b * bac_gain;

        out[..self.m].copy_from_slice(&kv[..self.m]);
        if eff_eta_b != 0.0 {
            for i in 0..self.h {
                let c = eff_eta_b * s_out[i];
                let row = &self.r_proj[i * self.m..(i + 1) * self.m];
                for j in 0..self.m {
                    out[j] += c * row[j];
                }
            }
        }
    }

    /// Propagación dendrítica local $J_\psi^\top g = K^\top (g + \eta_b R^\top \text{diag}(s(1-s)) R g)$
    /// para relajación predictiva dentro de la columna cortical.
    pub fn local_dendritic_jacobian_t(&self, s_gate: &[f64], g: &[f64], out: &mut [f64]) {
        let mut u = g[..self.m].to_vec();
        if self.eta_b != 0.0 {
            for i in 0..self.h {
                let row = &self.r_proj[i * self.m..(i + 1) * self.m];
                let mut dot = 0.0;
                for j in 0..self.m {
                    dot += row[j] * g[j];
                }
                let coeff = self.eta_b * s_gate[i] * (1.0 - s_gate[i]) * dot;
                for j in 0..self.m {
                    u[j] += coeff * row[j];
                }
            }
        }
        self.kt_mul(&u, out);
    }

    /// Integra los 3 compartimentos (Basal, Apical con cancelación SST+ y desinhibición VIP+,
    /// Soma con inhibición divisiva PV+) en un paso de relajación cortical.
    #[allow(clippy::too_many_arguments)]
    pub fn step_compartments(
        &mut self,
        st: &mut CompartmentalUnitState,
        basal_drive: &[f64],
        apical_top_down: &[f64],
        pv_inhibition: f64,
        vip_gate: f64,
        step_size_vec: &[f64],
        learn_sst: bool,
    ) {
        let mut apical_energy = 0.0;
        for j in 0..self.m {
            st.v_basal[j] = basal_drive[j];

            // Interneurona SST+ (Martinotti) predice la señal apical esperada a partir del soma;
            // VIP+ desinhibe el penacho apical ante novedad/ACh (Pfeffer et al., 2013; Sacramento et al., 2018)
            let raw_sst = self.w_sst_cancel[j] * st.u_soma[j];
            let effective_sst = raw_sst / (1.0 + 0.8 * vip_gate.max(0.0));
            let ap_net = apical_top_down[j] - effective_sst;
            st.v_apical[j] = ap_net;
            apical_energy += ap_net;

            if learn_sst {
                let sst_err = apical_top_down[j] - raw_sst;
                self.w_sst_cancel[j] =
                    (self.w_sst_cancel[j] + 0.005 * sst_err * st.u_soma[j]).clamp(-1.5, 1.5);
            }

            // Paso somático precondicionado por curvatura local + control de ganancia divisivo PV+
            let k = step_size_vec[j] / (1.0 + 0.25 * pv_inhibition.max(0.0));
            st.u_soma[j] = (st.u_soma[j] + k * (basal_drive[j] + 0.35 * ap_net)).clamp(-10.0, 10.0);
        }

        let mean_apical = apical_energy / (self.m.max(1) as f64);
        st.ca_plateau = sigmoid(1.8 * mean_apical);
    }

    /// Regulación homeostática lenta (Turrigiano & Nelson, 2004; BCM 1982).
    pub fn update_homeostasis(&mut self, mean_soma_abs: f64) {
        self.avg_activity = 0.96 * self.avg_activity + 0.04 * mean_soma_abs;
        let sq = mean_soma_abs * mean_soma_abs;
        self.bcm_threshold = (0.96 * self.bcm_threshold + 0.04 * sq).clamp(0.01, 2.0);
        let rate_err = 0.30 - self.avg_activity;
        for b in &mut self.intrinsic_excitability {
            *b = (*b + 0.001 * rate_err).clamp(-0.25, 0.25);
        }
    }

    pub fn spectral_radius_k(&self) -> f64 {
        self.rho.iter().copied().fold(0.0, f64::max)
    }

    #[inline]
    pub fn spectral_radius(&self) -> f64 {
        self.spectral_radius_k()
    }
}
