//! Utilidades deterministas de números pseudoaleatorios compartidas por los
//! motores nativos.
//!
//! Estas funciones existían como copias literales en seis módulos; la
//! duplicación arriesgaba divergencias silenciosas entre capas del mismo
//! modelo físico. Cada función produce exactamente los mismos valores que las
//! copias originales, para conservar bit a bit la reproducibilidad de los
//! experimentos publicados.
//!
//! Los mezcladores de fase (`blend_phase`/`blend_angle`) permanecen en sus
//! módulos: las tres variantes (delta angular lineal, mezcla polar compleja y
//! variante `f64`) difieren en precisión y comportamiento ante el clamp, así
//! que unificarlas cambiaría resultados.

/// Finalizador SplitMix64: mezcla de 64 bits de alta calidad y bajo coste.
#[inline(always)]
pub fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Uniforme en [0, 1) a partir de los 24 bits altos del valor mezclado.
#[inline(always)]
pub fn unit_from_u64(value: u64) -> f32 {
    ((value >> 40) as f32) * (1.0 / (1_u32 << 24) as f32)
}

/// Uniforme en [0, 1) en doble precisión, mismos 24 bits altos.
#[inline(always)]
pub fn unit_f64_from_u64(value: u64) -> f64 {
    ((value >> 40) as f64) * (1.0 / (1_u64 << 24) as f64)
}

/// Uniforme en [-1, 1) derivado de una semilla (SplitMix64 + 24 bits altos).
#[inline(always)]
pub fn signed_unit(seed: u64) -> f32 {
    2.0 * unit_from_u64(splitmix64(seed)) - 1.0
}

/// Gaussiana estándar aproximada por suma de 6 uniformes (Irwin–Hall), a
/// partir de un contador. Determinista e independiente del hilo: permite
/// ruido térmico reproducible dentro de regiones paralelas.
#[inline(always)]
pub fn gaussian_from_counter(seed: u64, counter: u64) -> f32 {
    let base = seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    unit_from_u64(splitmix64(base))
        + unit_from_u64(splitmix64(base ^ 0xA24B_AED4_963E_E407))
        + unit_from_u64(splitmix64(base ^ 0x9FB2_1C65_1E98_DF25))
        + unit_from_u64(splitmix64(base ^ 0xC13F_A9A9_02A6_328F))
        + unit_from_u64(splitmix64(base ^ 0x91E1_0DA5_C79E_7B1D))
        + unit_from_u64(splitmix64(base ^ 0xD1B5_4A32_D192_ED03))
        - 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_are_deterministic_and_bounded() {
        for seed in [0, 1, 0xDEAD_BEEF, u64::MAX] {
            assert_eq!(splitmix64(seed), splitmix64(seed));
            let unit = unit_from_u64(splitmix64(seed));
            assert!((0.0..1.0).contains(&unit));
            let unit64 = unit_f64_from_u64(splitmix64(seed));
            assert!((0.0..1.0).contains(&unit64));
            let signed = signed_unit(seed);
            assert!((-1.0..1.0).contains(&signed));
        }
    }

    #[test]
    fn gaussian_has_zero_mean_and_finite_range() {
        let sum: f64 = (0..100_000_u64)
            .map(|counter| f64::from(gaussian_from_counter(7, counter)))
            .sum();
        assert!(gaussian_from_counter(7, 0).abs() <= 3.0);
        assert!((sum / 100_000.0).abs() < 0.02);
    }
}
