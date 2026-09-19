# Inferencia: colapso de spin → RQM nativo

**Rama:** `exp/fluido-3d-astra`  
**Módulo:** `src/spin_fluid_rqm_infer.rs`  
**Motor RQM:** `NativeThermoRqmEprSubstrate` (el mismo de `main`, cableado solo aquí).

## Idea

La capa de fluido de spin actúa como **dinámica de inferencia**:

1. **Entrada** → paquete \(\psi\) (centro / momento / amplitud por clase).
2. **Colapso** NLS enfocante (`SpinFluid3D`).
3. **Features** discretos: celda gruesa del pico, bin de amplitud, bin de fase.
4. **RQM nativo** aprende `features → etiqueta` y responde `query` sobre el colapso.

No sustituye RQM en `main`; es un experimento de si el colapso de ondas de spin aporta un código espacial útil para relaciones RQM.

## Resultados (`cargo test --release --lib spin_fluid_rqm_infer`)

| Prueba | Resultado |
|--------|-----------|
| Features espaciales distintos | 4 entradas → picos en celdas distintas (p.ej. 57/58/54/58) |
| Identidad tras train | accuracy **0.00 → 1.00** (24 pares, 12 relaciones) |
| Mapeo desplazado \(i\to i+1\) | **4/4** correctos |

```bash
cargo test --release --lib spin_fluid_rqm_infer -- --nocapture
```

## API mínima

```rust
let mut eng = SpinFluidRqmInfer::new();
eng.train_identity_epoch(6);
let report = eng.infer(2); // predicted / confidence / feature_nodes
```
