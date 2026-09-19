# Experimento aislado: fluido 3D (forma Astra / Navier–Stokes)

**Rama:** `exp/fluido-3d-astra`  
**Aislado de** la línea campo/encoder/lingüística (PRs #10–#12).  
**No es** una formalización Lean ni un claim del problema del milenio.

## Contexto “Astra 5.6”

En septiembre 2026, reportes públicos asociaron **GPT-6 Astra** (y un modelo interno más capaz) con trabajo Lean sobre una **singularidad en tiempo finito** de Navier–Stokes 3D (estiramiento de vórtice). Aquí solo tomamos la **forma matemática** del régimen: NS 3D incompresible + forzamiento de vórtice estirado, y entrenamos un **sustituto residual** sobre un simulador pequeño.

## Ecuaciones

\[
\partial_t u + (u\cdot\nabla)u = -\nabla p + \nu\Delta u + f,\qquad \nabla\cdot u = 0
\]

Discretización: **Stable Fluids** (Stam) en rejilla \(16^3\), periodicidad, proyección Jacobi, difusión + advección semi-Lagrangiana. Forzamiento \(f\): tubo de vórtice alineado con \(z\) con componente axial de estiramiento.

## Modelo aprendido

- MLP residual: predice \(\Delta u\) sobre \(u_t\) para horizonte multi-paso (`skip=6`).
- Baseline: **persistencia** (\(u_{t+\mathrm{skip}}\approx u_t\)).
- Entrenamiento: 48 pares train / 24 test, 40 épocas, seed `0xA57A`.

## Resultados (`cargo test --release`)

| Prueba | Resultado |
|--------|-----------|
| Proyección (div L2) | OK (`div < 0.12`) |
| Vórtice estirado (30 pasos) | \(E: 0 \to 0.00807\), \(\|u\|_{\max}\approx0.366\), \(\mathrm{div}\approx0.0055\) |
| Sustituto vs persistencia | MSE \(0.004063\to0.000147\) vs persist \(0.000152\) (**gana**) |

Comando:

```bash
cargo test --release --lib fluid3d_astra -- --nocapture
```

(o, en el crate lab local: `cd fluid3d-astra && cargo test --release -- --nocapture`)

## Alcance / no-claims

- No demuestra singularidad finita ni resuelve Millennium.
- Rejilla \(16^3\) es un laboratorio mínimo; no es resolución científica de blow-up.
- Independiente del sustrato de campo sin tokens.
