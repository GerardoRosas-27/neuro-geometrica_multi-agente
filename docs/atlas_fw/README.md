# Mejor versión: ATLAS-FW

**ATLAS-FW** es la mejor versión lograda hasta ahora. Es un aprendiz en contexto (adaptación rápida) montado **sobre la red CONSTELACIÓN v4 ya entrenada**: no toca el entrenamiento de v4, solo sustituye el estimador de adaptación rápida (NLMS de v4) por una pila de mecanismos con interruptor. Con todos los interruptores apagados reproduce v4 **bit a bit** (test `off_is_v4_bit_exact`, `atlas-paso0`: 8448 valores, 0 discrepancias; `artifacts/atlas_fw/off_eq_v4_check.json`).

- Código: [`orbita/src/constelacion4/atlas.rs`](../../orbita/src/constelacion4/atlas.rs) (subcomandos `orbita atlas-h0`, `atlas-paso0`, `atlas-run`).
- Prerregistros: [`PREREG.md`](PREREG.md) · [`PREREG_ABLACIONES.md`](PREREG_ABLACIONES.md).
- Resultados: [`artifacts/atlas_fw/RESULTADOS.md`](../../artifacts/atlas_fw/RESULTADOS.md) · [`artifacts/atlas_fw/ablaciones/RESULTADOS_ABLACIONES.md`](../../artifacts/atlas_fw/ablaciones/RESULTADOS_ABLACIONES.md).
- Ramas de experimento: [`exp/atlas-fw`](https://github.com/GerardoRosas-27/neuro-geometrica_multi-agente/tree/exp/atlas-fw) @ `ea434ac` (desciende de [`exp/constelacion-v4`](https://github.com/GerardoRosas-27/neuro-geometrica_multi-agente/tree/exp/constelacion-v4) @ `fd80320`, que ya está en `main`).

## Arquitectura y mecanismos

Predicción: x̂(t+1) = x(t) + Σ_j (W^P_j + Θ_j) ψ_j, con ψ = [φ_v4 ; s·g(x)].

| Mecanismo | Qué hace | Aporte medido (ablaciones) |
|---|---|---|
| **M1** | Biblioteca Koopman/SINDy: 17 funciones genéricas (polinomios ≤3, sin/cos(2·), tanh(2·), r·x, r·y; escala 0,3) concatenadas a φ_v4 | **Contribuye** (DEV y CONFIRM) |
| **M2** | Estimador disperso iterado (ridge + STLSQ, 3 iteraciones) en lugar de NLMS | **Contribuye** (DEV y CONFIRM) |
| M3 | Vector de contexto c_e (dim 9) + hiperred lineal → prior Θ0 | No significativo (Δ = 0) |
| M4 | Meta-entrenamiento con familias sintéticas auditadas (nunca {8,10,11}) | No significativo (Δ −0,0750 [−0,1614, +0,0074] en DEV) |
| M5 | Objetivo de meta-validación = rollout h5..h16 | No significativo (Δ = 0) |
| **M6** | Compuerta CLS: si el residuo precuencial en contexto de v4 ≤ el de ATLAS, usa v4 | **Contribuye y protege `known_acc16`** (sin M6 falla el no-daño) |

**M3, M4 y M5 no aportan nada medible**; se conservan como interruptores (`Mech`) por trazabilidad y para futuros experimentos, pero la ganancia proviene de **M1 + M2 + M6**. El brazo evaluado y recomendado es `ALL` (los seis activos), que es el que se confirmó.

## Resultados (copiados literalmente de los RESULTADOS)

Primaria: `new_med_log` (med log rel_err h16 en familias nuevas; <0 = mejor), V − v4. Bootstrap pareado por semilla (10 000 réplicas), 16 semillas por fase, 0 consultas TEST.

### Experimento principal (`artifacts/atlas_fw/RESULTADOS.md`)

| fase | brazo | new_med_log | V−v4 new_med_log (IC) | V−v4 known_acc16 IC95 | E33 | NaN |
|---|---|---|---|---|---|---|
| DEV (0xA480…0xA48F) | v4 | -0.9347 | — | — | 16/16 | 0 |
| DEV | ALL | -1.4495 | -0.5149 [-0.6132, -0.4200] | +0.0210 [+0.0176, +0.0249] | 16/16 | 0 |
| CONFIRM (0xB480…0xB48F) | v4 | -0.9042 | — | — | 16/16 | 0 |
| CONFIRM | **ALL** | **-1.3125** | **-0.4082 [-0.4985, -0.3190]** | **+0.0225 [+0.0171, +0.0278]** | **16/16** | 0 |

Veredicto: **ALL: PASA (DEV y CONFIRM)**. Ningún mecanismo aislado pasa el gate DEV (M1 solo daña `known_acc16` −0.6538 y E33 2/16).

### Ablaciones leave-one-out (`artifacts/atlas_fw/ablaciones/RESULTADOS_ABLACIONES.md`)

Δ = (ALL−Mi) − ALL en new_med_log (Δ>0 ⇒ Mi contribuye). ALL reproduce la ganancia en semillas nuevas: V−v4 = −0.3938 [−0.5124, −0.2762] (ABL_DEV) y **−0.3742 [−0.4856, −0.2462] (ABL_CONFIRM)**, E33 16/16.

| brazo | Δ ABL_DEV | Δ ABL_CONFIRM | no-daño |
|---|---|---|---|
| ALL-M1 | +0.3957 [+0.2407, +0.5538] | +0.3850 [+0.2037, +0.5330] | OK |
| ALL-M2 | +0.2049 [+0.0588, +0.3579] | +0.2250 [+0.0771, +0.3592] | OK |
| ALL-M3 | +0.0000 [+0.0000, +0.0000] | — | OK |
| ALL-M4 | -0.0750 [-0.1614, +0.0074] | — | OK |
| ALL-M5 | +0.0000 [+0.0000, +0.0000] | — | OK |
| ALL-M6 | +0.0754 [+0.0250, +0.1397] | +0.0639 [+0.0208, +0.1255] | **FALLA** (known_acc16 −0.3096 vs v4 en CONFIRM) |

## Reproducción

```bash
cd orbita && cargo build --release && cargo test --release && cd ..
# Experimento principal: DEV → gate → CONFIRM → RESULTADOS.md (JOBS=7 por defecto)
bash orbita/scripts/atlas_fw.sh dev v4,M1,M2,M3,M4,M5,M6,ALL
bash orbita/scripts/atlas_fw.sh confirm v4,ALL
python3 scripts/analyze_atlas_fw.py dev confirm --md
# Ablaciones leave-one-out
bash orbita/scripts/atlas_fw_abl.sh abl_dev v4,ALL,ALL-M1,ALL-M2,ALL-M3,ALL-M4,ALL-M5,ALL-M6
bash orbita/scripts/atlas_fw_abl.sh abl_confirm v4,ALL,ALL-M1,ALL-M2,ALL-M6
python3 scripts/analyze_atlas_fw_abl.py abl_dev abl_confirm --md
# Una semilla suelta
orbita/target/release/orbita atlas-run --phase confirm --seed 0xB480 \
  --weights artifacts/orbita/confirm/weights_0xB400.json --out artifacts/atlas_fw --arms v4,ALL
```

Los scripts saltan semillas con `.done` existente: borra `artifacts/atlas_fw/<fase>/` (o `ablaciones/<fase>/`) para recomputar desde cero.

Las cadenas `atlas_fw_chain.sh` / `atlas_fw_abl_chain.sh` automatizan todo pero **hacen commit+push a `exp/atlas-fw`** (la de ablaciones se niega a correr en otra rama); en `main` usa los comandos sueltos de arriba. Los pesos ÓRBITA-v1 (`artifacts/orbita/{dev,confirm}/weights_*.json`) ya están en `main`.
