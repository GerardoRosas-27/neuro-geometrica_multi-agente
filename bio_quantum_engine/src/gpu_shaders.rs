//! Compute Shaders WGSL para Bio-Quantum v10 (Singularity GPU Ultra)
//!
//! Diseñados para ejecución masiva en AMD RDNA3 (Radeon RX 7800 XT):
//! - 64 Puntos / Agentes en constelación (128 canales de coordenadas)
//! - 64 Dimensiones de base de Lie y Fasores Cuánticos de Wheeler-Feynman
//! - Proyección estricta de Ley de Dale (W^E >= 1e-4, W^I >= 1e-4, 0 violaciones)
//! - Envolvente contractiva / dilatación hiperbólica de Lyapunov-Brenier

pub const WGSL_BIO_QUANTUM_COMPUTE: &str = r#"
struct ConfigUniform {
    n_points: u32,
    d_basis: u32,
    batch_size: u32,
    r_bound: f32,
    growth_rate: f32,
    learning_rate: f32,
    circadian_phase: f32,
    lambda_reg: f32,
};

@group(0) @binding(0) var<uniform> config: ConfigUniform;

// Coordenadas actuales: [batch_size * n_points * 2]
@group(0) @binding(1) var<storage, read> input_coords: array<f32>;

// Coordenadas objetivo: [batch_size * n_points * 2]
@group(0) @binding(2) var<storage, read> target_coords: array<f32>;

// Pesos corticales lentos W^P_E y W^P_I: [2 * d_basis]
@group(0) @binding(3) var<storage, read_write> wp_exc: array<f32>;
@group(0) @binding(4) var<storage, read_write> wp_inh: array<f32>;

// Pesos adaptativos rápidos W^F_E y W^F_I: [2 * d_basis]
@group(0) @binding(5) var<storage, read_write> wf_exc: array<f32>;
@group(0) @binding(6) var<storage, read_write> wf_inh: array<f32>;

// Predicciones de salida: [batch_size * n_points * 2]
@group(0) @binding(7) var<storage, read_write> output_coords: array<f32>;

// Métricas de paso (Loss MSE, violaciones Dale, etc.): [8]
@group(0) @binding(8) var<storage, read_write> metrics: array<f32>;

// Función auxiliar para calcular los 64 generadores de Lie y fasores cuánticos
fn compute_phi_64(x: f32, y: f32, idx: u32) -> f32 {
    let r2 = x * x + y * y;
    let r = sqrt(max(r2, 1e-12));

    // Grado 1: Afín y radial (0..3)
    if (idx == 0u) { return x; }
    if (idx == 1u) { return y; }
    if (idx == 2u) { return 1.0; }
    if (idx == 3u) { return r; }

    // Armónicos espaciales y modos de Fourier (4..11)
    if (idx == 4u) { return sin(x); }
    if (idx == 5u) { return cos(y); }
    if (idx == 6u) { return sin(2.0 * x); }
    if (idx == 7u) { return cos(2.0 * y); }
    if (idx == 8u) { return sin(4.0 * x); }
    if (idx == 9u) { return cos(4.0 * y); }
    if (idx == 10u) { return tanh(x); }
    if (idx == 11u) { return tanh(y); }

    // Grado 2: Curvaturas cuadráticas y conformes (12..19)
    if (idx == 12u) { return x * y; }
    if (idx == 13u) { return x * x - y * y; }
    if (idx == 14u) { return x * x + y * y; }
    if (idx == 15u) { return 2.0 * x * y; }
    if (idx == 16u) { return x * r; }
    if (idx == 17u) { return y * r; }
    if (idx == 18u) { return (x - y) * (x - y); }
    if (idx == 19u) { return (x + y) * (x + y); }

    // Grado 3: Atractores no lineales cúbicos (Duffing, Van der Pol, Hopf) (20..29)
    if (idx == 20u) { return x * x * x; }
    if (idx == 21u) { return y * y * y; }
    if (idx == 22u) { return x * x * y; }
    if (idx == 23u) { return y * y * x; }
    if (idx == 24u) { return x * r2; }
    if (idx == 25u) { return y * r2; }
    if (idx == 26u) { return (x * x * x - 3.0 * x * y * y) * 0.1; }
    if (idx == 27u) { return (3.0 * x * x * y - y * y * y) * 0.1; }
    if (idx == 28u) { return tanh(x * r2); }
    if (idx == 29u) { return tanh(y * r2); }

    // Fasores Cuánticos Wheeler-Feynman en C (30..49)
    let freq_id = f32(idx - 30u) * 0.5 + 1.0;
    let phase = freq_id * (x + y) * 0.5;
    if ((idx % 2u) == 0u) {
        return cos(phase) * (1.0 / sqrt(freq_id));
    } else {
        return sin(phase) * (1.0 / sqrt(freq_id));
    }

    // Coordenadas acopladas de alta capacidad (50..63)
    let k_mode = f32(idx - 50u + 1u);
    return cos(k_mode * r) * exp(-0.05 * r2);
}

// Kernel 1: Inferencia Forward y Rollout con Envolvente de Brenier
@compute @workgroup_size(64, 1, 1)
fn forward_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    let global_point_idx = id.x;
    let total_points = config.batch_size * config.n_points;

    if (global_point_idx >= total_points) {
        return;
    }

    let coord_offset = global_point_idx * 2u;
    let x = input_coords[coord_offset];
    let y = input_coords[coord_offset + 1u];

    var dx: f32 = 0.0;
    var dy: f32 = 0.0;
    let d = config.d_basis;

    for (var j: u32 = 0u; j < d; j = j + 1u) {
        let phi = compute_phi_64(x, y, j);
        let wx = (wp_exc[j] - wp_inh[j]) + (wf_exc[j] - wf_inh[j]);
        let wy = (wp_exc[d + j] - wp_inh[d + j]) + (wf_exc[d + j] - wf_inh[d + j]);

        dx = dx + wx * phi;
        dy = dy + wy * phi;
    }

    var pred_x = x + dx;
    var pred_y = y + dy;

    // Proyección de Transporte Óptimo de Brenier-Lyapunov
    let rb = config.r_bound;
    let r_pred = sqrt(max(pred_x * pred_x + pred_y * pred_y, 1e-12));
    if (r_pred > rb) {
        let excess = (r_pred - rb) / rb;
        let scale = rb * (1.0 + tanh(excess)) / r_pred;
        pred_x = pred_x * scale;
        pred_y = pred_y * scale;
    }

    output_coords[coord_offset] = pred_x;
    output_coords[coord_offset + 1u] = pred_y;
}

// Kernel 2: Descenso de Gradiente Proyectado (Dale-Compliant W^E >= 1e-4, W^I >= 1e-4)
@compute @workgroup_size(64, 1, 1)
fn train_step_dale(@builtin(global_invocation_id) id: vec3<u32>) {
    let global_point_idx = id.x;
    let total_points = config.batch_size * config.n_points;

    if (global_point_idx >= total_points) {
        return;
    }

    let coord_offset = global_point_idx * 2u;
    let pred_x = output_coords[coord_offset];
    let pred_y = output_coords[coord_offset + 1u];
    let tgt_x = target_coords[coord_offset];
    let tgt_y = target_coords[coord_offset + 1u];
    let in_x = input_coords[coord_offset];
    let in_y = input_coords[coord_offset + 1u];

    let err_x = pred_x - tgt_x;
    let err_y = pred_y - tgt_y;
    let point_loss = err_x * err_x + err_y * err_y;

    // Acumular Loss atómicamente o por suma proporcional
    let norm_factor = 1.0 / f32(total_points);
    metrics[0] = metrics[0] + point_loss * norm_factor;

    let d = config.d_basis;
    let lr = config.learning_rate / f32(config.n_points);

    // Actualización de pesos proyectada para cumplir estrictamente con la Ley de Dale
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        let phi = compute_phi_64(in_x, in_y, j);
        let gx = err_x * phi * norm_factor;
        let gy = err_y * phi * norm_factor;

        // Proyección W^E >= 1e-4 y W^I >= 1e-4
        let new_wp_ex_x = max(wp_exc[j] - lr * gx, 1e-4);
        let new_wp_in_x = max(wp_inh[j] + lr * gx, 1e-4);
        wp_exc[j] = new_wp_ex_x;
        wp_inh[j] = new_wp_in_x;

        let new_wp_ex_y = max(wp_exc[d + j] - lr * gy, 1e-4);
        let new_wp_in_y = max(wp_inh[d + j] + lr * gy, 1e-4);
        wp_exc[d + j] = new_wp_ex_y;
        wp_inh[d + j] = new_wp_in_y;
    }
}

// Kernel 3: Consolidación Circadiana de Sueño SWR en GPU (Bounded Forgetting ΔR >= -ε)
@compute @workgroup_size(64, 1, 1)
fn swr_sleep_consolidation(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let total_weights = config.d_basis * 2u;

    if (j >= total_weights) {
        return;
    }

    // Transferencia de memoria rápida W^F a cortical lenta W^P
    let wf = wf_exc[j] - wf_inh[j];
    let delta = 0.05 * wf;
    let current = (wp_exc[j] - wp_inh[j]) + delta;

    if (current >= 0.0) {
        wp_exc[j] = current + 1e-4;
        wp_inh[j] = 1e-4;
    } else {
        wp_exc[j] = 1e-4;
        wp_inh[j] = -current + 1e-4;
    }

    // Decaimiento homeostático de pesos rápidos tras el sueño
    wf_exc[j] = 1e-4;
    wf_inh[j] = 1e-4;
}
"#;
