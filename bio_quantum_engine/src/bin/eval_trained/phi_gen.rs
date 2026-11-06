// AUTOGENERADO por docs/bq10_gpu_amd/wgsl2rs.py desde los shaders WGSL (x50: rama; x20: zip)
use super::w::*;
#[allow(unused_parens,clippy::all)]
pub fn compute_phi_1024(x: f32, y: f32, idx: u32) -> f32 {
    let r2 = x * x + y * y;
    let r = sqrt(max(r2, 1e-12));
    let theta = atan2(y, x);

    // [Sección 1: 0..63] Generadores Fundamentales de Lie, Afines, Polinomiales y Conformes (64 dimensiones)
    if (idx == 0u32) { return x; }
    if (idx == 1u32) { return y; }
    if (idx == 2u32) { return 1.0; }
    if (idx == 3u32) { return r; }
    if (idx == 4u32) { return x * y; }
    if (idx == 5u32) { return x * x - y * y; }
    if (idx == 6u32) { return x * r; }
    if (idx == 7u32) { return y * r; }
    if (idx == 8u32) { return x * x * x; }
    if (idx == 9u32) { return y * y * y; }
    if (idx == 10u32) { return x * x * y; }
    if (idx == 11u32) { return y * y * x; }
    if (idx == 12u32) { return x * r2; }
    if (idx == 13u32) { return y * r2; }
    if (idx == 14u32) { return tanh(x); }
    if (idx == 15u32) { return tanh(y); }
    if (idx == 16u32) { return x * x * x * x - y * y * y * y; }
    if (idx == 17u32) { return x * y * r2; }
    if (idx == 18u32) { return exp(-0.1 * r2) * x; }
    if (idx == 19u32) { return exp(-0.1 * r2) * y; }
    if (idx == 20u32) { return sin(x * y); }
    if (idx == 21u32) { return cos(x * y); }
    if (idx == 22u32) { return tanh(x * r); }
    if (idx == 23u32) { return tanh(y * r); }
    if (idx == 24u32) { return (x * x * x - 3.0 * x * y * y) * 0.1; }
    if (idx == 25u32) { return (3.0 * x * x * y - y * y * y) * 0.1; }
    if (idx == 26u32) { return 1.0 / (1.0 + r); }
    if (idx == 27u32) { return r / (1.0 + r2); }
    if (idx == 28u32) { return (x - y) * (x - y); }
    if (idx == 29u32) { return (x + y) * (x + y); }
    if (idx == 30u32) { return x * (1.0 - x * x); }
    if (idx == 31u32) { return y * (1.0 - y * y); }
    if (idx == 32u32) { return x * x * x * x * x * 0.05; }
    if (idx == 33u32) { return y * y * y * y * y * 0.05; }
    if (idx == 34u32) { return x * x * x * y * y * 0.05; }
    if (idx == 35u32) { return y * y * y * x * x * 0.05; }
    if (idx == 36u32) { return exp(-0.05 * r2) * (x * x - y * y); }
    if (idx == 37u32) { return exp(-0.05 * r2) * 2.0 * x * y; }
    if (idx == 38u32) { return sin(2.0 * theta) * r; }
    if (idx == 39u32) { return cos(2.0 * theta) * r; }
    if (idx == 40u32) { return sin(3.0 * theta) * r2 / (1.0 + r); }
    if (idx == 41u32) { return cos(3.0 * theta) * r2 / (1.0 + r); }
    if (idx == 42u32) { return log(1.0 + r2) * x; }
    if (idx == 43u32) { return log(1.0 + r2) * y; }
    if (idx == 44u32) { return 1.0 / (1.0 + r2 * r2); }
    if (idx == 45u32) { return (x * x * x * x - 6.0 * x * x * y * y + y * y * y * y) * 0.05; }
    if (idx == 46u32) { return (4.0 * x * x * x * y - 4.0 * x * y * y * y) * 0.05; }
    if (idx == 47u32) { return tanh(x * x + y * y); }
    if (idx == 48u32) { return tanh(x - y); }
    if (idx == 49u32) { return tanh(x + y); }
    if (idx == 50u32) { return cos(x) * sin(y); }
    if (idx == 51u32) { return sin(x) * cos(y); }
    if (idx == 52u32) { return cos(x) * cos(y); }
    if (idx == 53u32) { return sin(x) * sin(y); }
    if (idx == 54u32) { return x * exp(-abs(x)); }
    if (idx == 55u32) { return y * exp(-abs(y)); }
    if (idx == 56u32) { return (x * y) / (1.0 + r2); }
    if (idx == 57u32) { return r * (1.0 - exp(-r)); }
    if (idx == 58u32) { return atan(x); }
    if (idx == 59u32) { return atan(y); }
    if (idx == 60u32) { return sinh(clamp(x * 0.5, -3.0, 3.0)); }
    if (idx == 61u32) { return sinh(clamp(y * 0.5, -3.0, 3.0)); }
    if (idx == 62u32) { return (x * x - y * y) / (1.0 + r2); }
    if (idx == 63u32) { return (2.0 * x * y) / (1.0 + r2); }

    // [Sección 2: 64..383] Armónicos Espaciales 2D y Modos de Fourier (320 dimensiones)
    if (idx < 384u32) {
        let mode_idx = idx - 64u32;
        let k = f32c(mode_idx / 2u32 + 1u32) * 0.125;
        let angle = k * (x * 0.8 + y * 0.6) + f32c(mode_idx % 4u32) * 0.785398;
        if ((mode_idx % 2u32) == 0u32) {
            return sin(angle) / sqrt(k + 0.5);
        } else {
            return cos(angle) / sqrt(k + 0.5);
        }
    }

    // [Sección 3: 384..703] Fasores de Wheeler-Feynman Retardados y Avanzados en C (320 dimensiones)
    if (idx < 704u32) {
        let wf_idx = idx - 384u32;
        let freq = f32c(wf_idx / 4u32 + 1u32) * 0.15;
        let m_ang = f32c(wf_idx % 8u32 + 1u32);
        let phase_ret = freq * r - m_ang * theta;
        let phase_adv = freq * r + m_ang * theta;
        let ret_wave = cos(phase_ret);
        let adv_wave = sin(phase_adv);
        let handshake = (ret_wave + adv_wave) * 0.5;
        return handshake * exp(-0.005 * r2);
    }

    // [Sección 4: 704..1023] Modos Radiales de Bessel, Cuenca CDT y Deformación Geométrica (320 dimensiones)
    let b_idx = idx - 704u32;
    let m_bessel = f32c(b_idx % 32u32 + 1u32);
    let scale_r = f32c(b_idx / 32u32 + 1u32) * 0.18;
    let bessel_approx = cos(scale_r * r - 0.5 * m_bessel * 3.14159) / sqrt(max(scale_r * r, 0.4));
    let angular = sin(m_bessel * theta);
    return bessel_approx * angular * exp(-0.004 * r2);
}

#[allow(unused_parens,clippy::all)]
pub fn compute_phi_512(x: f32, y: f32, idx: u32) -> f32 {
    let r2 = x * x + y * y;
    let r = sqrt(max(r2, 1e-12));
    let theta = atan2(y, x);

    // [Sección 1: 0..31] Generadores Fundamentales de Lie, Afines y Polinomiales
    if (idx == 0u32) { return x; }
    if (idx == 1u32) { return y; }
    if (idx == 2u32) { return 1.0; }
    if (idx == 3u32) { return r; }
    if (idx == 4u32) { return x * y; }
    if (idx == 5u32) { return x * x - y * y; }
    if (idx == 6u32) { return x * r; }
    if (idx == 7u32) { return y * r; }
    if (idx == 8u32) { return x * x * x; }
    if (idx == 9u32) { return y * y * y; }
    if (idx == 10u32) { return x * x * y; }
    if (idx == 11u32) { return y * y * x; }
    if (idx == 12u32) { return x * r2; }
    if (idx == 13u32) { return y * r2; }
    if (idx == 14u32) { return tanh(x); }
    if (idx == 15u32) { return tanh(y); }
    if (idx == 16u32) { return x * x * x * x - y * y * y * y; }
    if (idx == 17u32) { return x * y * r2; }
    if (idx == 18u32) { return exp(-0.1 * r2) * x; }
    if (idx == 19u32) { return exp(-0.1 * r2) * y; }
    if (idx == 20u32) { return sin(x * y); }
    if (idx == 21u32) { return cos(x * y); }
    if (idx == 22u32) { return tanh(x * r); }
    if (idx == 23u32) { return tanh(y * r); }
    if (idx == 24u32) { return (x * x * x - 3.0 * x * y * y) * 0.1; }
    if (idx == 25u32) { return (3.0 * x * x * y - y * y * y) * 0.1; }
    if (idx == 26u32) { return 1.0 / (1.0 + r); }
    if (idx == 27u32) { return r / (1.0 + r2); }
    if (idx == 28u32) { return (x - y) * (x - y); }
    if (idx == 29u32) { return (x + y) * (x + y); }
    if (idx == 30u32) { return x * (1.0 - x * x); }
    if (idx == 31u32) { return y * (1.0 - y * y); }

    // [Sección 2: 32..191] Armónicos Espaciales 2D y Modos de Fourier (160 dimensiones)
    if (idx < 192u32) {
        let mode_idx = idx - 32u32;
        let k = f32c(mode_idx / 2u32 + 1u32) * 0.25;
        let is_sin = (mode_idx % 2u32) == 0u32;
        let angle = k * (x * 0.8 + y * 0.6);
        if (is_sin) {
            return sin(angle) / sqrt(k);
        } else {
            return cos(angle) / sqrt(k);
        }
    }

    // [Sección 3: 192..351] Fasores de Wheeler-Feynman Retardados y Avanzados en C (160 dimensiones)
    if (idx < 352u32) {
        let wf_idx = idx - 192u32;
        let freq = f32c(wf_idx / 2u32 + 1u32) * 0.3;
        let m_ang = f32c(wf_idx % 4u32 + 1u32);
        let ret_wave = cos(freq * r - m_ang * theta);
        let adv_wave = sin(freq * r + m_ang * theta);
        let handshake = (ret_wave + adv_wave) * 0.5;
        return handshake * exp(-0.01 * r2);
    }

    // [Sección 4: 352..511] Modos Radiales de Bessel, Cuenca CDT y Curvatura Extrema (160 dimensiones)
    let b_idx = idx - 352u32;
    let m_bessel = f32c(b_idx % 16u32 + 1u32);
    let scale_r = f32c(b_idx / 16u32 + 1u32) * 0.35;
    let bessel_approx = cos(scale_r * r - 0.5 * m_bessel * 3.14159) / sqrt(max(scale_r * r, 0.5));
    let angular = sin(m_bessel * theta);
    return bessel_approx * angular * exp(-0.008 * r2);
}
