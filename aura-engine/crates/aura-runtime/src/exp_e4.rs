//! Experimento E4 (Parte 1 de la Propuesta): Núcleo Neuromórfico Event-Driven `CorticalFabric`
//! (256 neuronas, 3 bandas Gamma/Beta/Theta, CSR + retardos axónicos reales, atención galga U(1),
//! Energía Libre de Laplace, álgebra tropical por minicolumna, 3 factores + NE/ACh/DA).
//!
//! Flujo de 1,200 ticks de bytes crudos delta-modulados:
//! - Ticks 0..600: Motivo estructurado M1 (específico de la semilla).
//! - Ticks 600..1200: Cambio abrupto a motivo M2 (prueba de volatilidad NE + readaptación local).
//!
//! Brazos pareados: `cortical_full`, `cortical_frozen`, `cortical_no_phase_gate`, `cortical_no_neuromod`.

use aura_cortex::{CorticalFabric, FabricConfig};
use aura_math::Xorshift64;

#[derive(Clone, Debug)]
pub struct E4Row {
    pub seed: u64,
    pub phase: String,
    pub arm: String,
    pub fe_regime1_early: f64,
    pub fe_regime1_late: f64,
    pub fe_regime2_early: f64,
    pub fe_regime2_late: f64,
    pub fe_late_mean: f64,
    pub sensory_fe_late: f64,
    pub mean_spike_rate_pct: f64,
    pub ne_pre_shift: f64,
    pub ne_post_shift: f64,
    pub ne_shift_delta: f64,
    pub ticks_per_sec: f64,
    pub eval_ms: f64,
}

fn generate_motifs(seed: u64) -> ([u8; 8], [u8; 8]) {
    let mut rng = Xorshift64::from_seed(seed).split_stream(0xE400_4D4F);
    let mut m1 = [0u8; 8];
    let mut m2 = [0u8; 8];
    for i in 0..8 {
        // Garantizar transiciones de bits bien definidas en M1 y contraste fuerte en M2.
        let b = (rng.next_u64() & 0xFF) as u8;
        m1[i] = b | if (i & 1) == 0 { 0x05 } else { 0x50 };
        m2[i] = !m1[i] ^ 0xA5;
    }
    (m1, m2)
}

fn eval_e4_arm(
    seed: u64,
    phase: &str,
    name: &str,
    cfg: FabricConfig,
    m1: &[u8; 8],
    m2: &[u8; 8],
) -> E4Row {
    let mut fabric = CorticalFabric::build_hierarchical_sheet(seed, cfg);
    let t0 = std::time::Instant::now();

    let mut prev_byte = 0u8;
    let mut fe_r1_early = 0.0f64;
    let mut fe_r1_late = 0.0f64;
    let mut fe_r2_early = 0.0f64;
    let mut fe_r2_late = 0.0f64;
    let mut sens_fe_late = 0.0f64;
    let mut spikes_sum = 0usize;
    let mut ne_pre = 0.0f64;
    let mut ne_post = 0.0f64;

    const TOTAL_TICKS: usize = 1200;
    for tick in 0..TOTAL_TICKS {
        // Inyectar bytes cada 2 ticks en régimen estacionario (M1 y M2); en el borde del cambio
        // de régimen (600..620) ocurre una ráfaga de transición inesperada que invierte los bits.
        if (600..620).contains(&tick) {
            let curr = !prev_byte ^ m2[tick % m2.len()];
            fabric.inject_raw_byte(prev_byte, curr);
            prev_byte = curr;
        } else if tick % 2 == 0 {
            let step_idx = tick / 2;
            let curr = if tick < 600 {
                m1[step_idx % m1.len()]
            } else {
                m2[step_idx % m2.len()]
            };
            fabric.inject_raw_byte(prev_byte, curr);
            prev_byte = curr;
        }
        let st = fabric.step();
        spikes_sum += st.spike_count;

        let fe = st.free_energy as f64;
        if (40..200).contains(&tick) {
            fe_r1_early += fe;
        } else if (400..600).contains(&tick) {
            fe_r1_late += fe;
            sens_fe_late += st.sensory_free_energy as f64;
        } else if (600..760).contains(&tick) {
            fe_r2_early += fe;
        } else if (1000..1200).contains(&tick) {
            fe_r2_late += fe;
            sens_fe_late += st.sensory_free_energy as f64;
        }

        if (570..600).contains(&tick) {
            ne_pre += st.neuromod.noradrenaline as f64;
        } else if (600..630).contains(&tick) {
            ne_post += st.neuromod.noradrenaline as f64;
        }
    }

    let elapsed_s = t0.elapsed().as_secs_f64().max(1e-9);
    let r1_early = fe_r1_early / 160.0;
    let r1_late = fe_r1_late / 200.0;
    let r2_early = fe_r2_early / 160.0;
    let r2_late = fe_r2_late / 200.0;
    let ne_pre_mean = ne_pre / 30.0;
    let ne_post_mean = ne_post / 30.0;

    E4Row {
        seed,
        phase: phase.to_string(),
        arm: name.to_string(),
        fe_regime1_early: r1_early,
        fe_regime1_late: r1_late,
        fe_regime2_early: r2_early,
        fe_regime2_late: r2_late,
        fe_late_mean: 0.5 * (r1_late + r2_late),
        sensory_fe_late: sens_fe_late / 400.0,
        mean_spike_rate_pct: (spikes_sum as f64) * 100.0 / ((TOTAL_TICKS * fabric.n_neurons) as f64),
        ne_pre_shift: ne_pre_mean,
        ne_post_shift: ne_post_mean,
        ne_shift_delta: ne_post_mean - ne_pre_mean,
        ticks_per_sec: (TOTAL_TICKS as f64) / elapsed_s,
        eval_ms: elapsed_s * 1000.0,
    }
}

pub fn run_e4_for_seed(seed: u64, phase: &str) -> Vec<E4Row> {
    let (m1, m2) = generate_motifs(seed);
    let arms = [
        ("cortical_full", FabricConfig::default()),
        (
            "cortical_frozen",
            FabricConfig {
                enable_plasticity: false,
                ..FabricConfig::default()
            },
        ),
        (
            "cortical_no_phase_gate",
            FabricConfig {
                enable_phase_gate: false,
                ..FabricConfig::default()
            },
        ),
        (
            "cortical_no_neuromod",
            FabricConfig {
                enable_neuromod: false,
                ..FabricConfig::default()
            },
        ),
    ];
    arms.iter()
        .map(|&(name, cfg)| eval_e4_arm(seed, phase, name, cfg, &m1, &m2))
        .collect()
}
