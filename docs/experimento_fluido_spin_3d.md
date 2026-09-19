# Fluido de spin sobre el simulador NS 3D

**Rama:** `exp/fluido-3d-astra`  
**Módulo:** `src/spin_fluid3d.rs` (rejilla compartida con `fluid3d_astra`)  
**Aislado de** RQM / XXZ / VMC / campo-sin-tokens.

## Objetivo

Simular **colapsos de onda**, **propagación** e **interferencia** en un fluido de spin continuum acoplable al Navier–Stokes 3D ya existente.

## Modelo

Envolvente compleja de ondas de spin transversales \(\psi = \psi_r + i\psi_i\), con magnetización longitudinal \(m_z = \sqrt{\max(0,1-|\psi|^2)}\):

\[
i\,\partial_t\psi = -\alpha\nabla^2\psi - \beta|\psi|^2\psi - i\,(u\cdot\nabla)\psi
\]

| Símbolo | Rol |
|---------|-----|
| \(\alpha>0\) | dispersión → propagación de paquetes |
| \(\beta>0\) | no linealidad **enfocante** → colapso si la amplitud es supercrítica (3D) |
| \(u\) | velocidad opcional de `Fluid3D` (advección) |

Integración: RK2 explícito, Laplaciano 7 puntos periódico, \(N=16\).

## Resultados (`cargo test --release --lib spin_fluid3d`)

| Prueba | Resultado |
|--------|-----------|
| Propagación | COM \(8.00\to8.92\) en \(x\), \(\|disp\|=0.918\) (40 pasos) |
| Interferencia constructiva | pico simple \(0.278\) → doble \(0.436\) (ratio **1.57**) |
| Colapso enfocante | débil \(0.220\to0.219\); fuerte \(0.850\to1.160\) (max **1.160**) |
| Advección por NS | amplitud estable \(0.400\to0.398\); fluido con \(E>0\) |

4/4 tests OK.

```bash
cargo test --release --lib spin_fluid3d -- --nocapture
```

## No-claims

- No es el líquido XXZ / VMC del motor unificado.
- No modifica la capa RQM de `main`.
- Rejilla \(16^3\): laboratorio de dinámica de ondas, no resolución continua de blow-up físico.
