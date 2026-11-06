//! gemma_campo: sustrato de campo que predice estados ocultos futuros (z) de Gemma 2 2B congelado.
//!
//! Arquitectura (ver docs/gemma_campo_chat/PLAN.md):
//! * Parte LENTA: dz_{t+1} = z_{t+1} - z_t = W^T phi(z_t..z_{t-C+1}), phi = [1 | lineal C*d | cuadrático z_t (d(d+1)/2)].
//!   Ajuste cerrado M7 "ridge por grado" (lambda distinto por bloque de grado, elegido en val) en f64,
//!   refinado con SGD-momentum en GPU (WGSL sin carreras) sobre todo train.
//! * Parte RÁPIDA: matriz A por conversación, NLMS online sobre el residuo de la lenta
//!   (A <- gamma*A + eta * r z^T / (|z|^2+eps)); solo inferencia, eta/gamma elegidos en val (eta=0 la apaga).
pub mod data;
pub mod features;
pub mod cpu;
pub mod gpu;

pub const WGSL: &str = include_str!("train.wgsl");

/// Valida el WGSL con naga sin necesitar GPU.
pub fn validate_wgsl() -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(WGSL).map_err(|e| e.emit_to_string(WGSL))?;
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).map_err(|e| format!("{e:?}"))?;
    Ok(())
}
