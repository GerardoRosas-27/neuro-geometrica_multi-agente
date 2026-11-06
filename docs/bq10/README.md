# BQ10-GPU (Bio-Quantum v10 en GPU AMD)

Este documento describe la línea **BQ10-GPU** del crate `bio_quantum_engine/`
(rama `exp/bq10-gpu-amd`, tip integrado `6808294`). No es el mismo experimento
que el **BQ10 canónico** ya presente en `main` (`orbita atlas-run --arms v4,BQ10`,
docs en [`docs/bio_quantum_v10/`](../bio_quantum_v10/)): aquel corre sobre el
arnés CST4 de ÓRBITA, con contraste pareado contra v4 y ATLAS-FW.

**La mejor versión canónica del laboratorio no cambia con esta integración.**
Sigue siendo el BQ10 del arnés CST4 (CONFIRM: BQ10−v4 new_med_log −0.6083
[−0.7415, −0.4652]; BQ10−ATLAS-FW −0.1563 [−0.2692, −0.0434]). ATLAS-FW
permanece como la ganadora previa documentada (CONFIRM V−v4 −0.4082
[−0.4985, −0.3190]). Las cifras de este arnés propio **no se pueden comparar**
con esas tablas.

## Qué es

Motor Rust (`bio_quantum_engine`) con:

- Unidades piramidales con sinapsis excitatorias e inhibitorias separadas
  (Ley de Dale: \(W^E, W^I \ge 0\)).
- Entrenamiento masivo en GPU AMD Radeon RX 7800 XT vía WGPU (backend
  reportado: `D3D12` / `Gl`) y shaders WGSL.
- Escalado del modelo en tres tamaños (x10, x20, x50).
- Un arnés de evaluación *propio* (`eval dev` / `eval confirm`) y, más
  tarde, un evaluador `eval_trained` que sí carga los pesos exportados
  por `gpu-train`.

## Arquitectura: x10 / x20 / x50

Cifras de entrenamiento tomadas de
`bio_quantum_engine/training_results_gpu_rx7800xt.json` en el commit de
cada escala (el JSON del tip solo conserva el run x50). Hardware común:
AMD Radeon RX 7800 XT.

| Escala | Commit | Puntos | `d_basis` | Sinapsis \(W^E,W^I\) | Batch | Episodios | Tiempo (s) | eps/s | MSE inicial → final | Dale |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| x10 | `6a20747` | 256 | 256 | 1 024 | 2 048 | 614 400 | 8.272 | 74 273.3 | 0.000065 → 0.000060 (−8.20 %) | 0 |
| x20 | `d39a5c5` | 512 | 512 | 2 048 | 4 096 | 1 638 400 | 34.185 | 47 927.4 | 0.000146 → 0.000253 (**+73.93 %**) | 0 |
| **x50** (esta versión) | `85cf480` / JSON del tip | 1 024 | 1 024 | 4 096 | 4 096 | **1 638 400** | **86.804** | 18 874.7 | 0.000552 → 0.000478 (−13.47 %) | 0 |

El log impreso de x50 redondea el tiempo a **86.80 s** y la reducción a
**13.5 %**. Telemetría por lote: [`docs/RESULTADOS_GPU_RX7800XT.md`](../RESULTADOS_GPU_RX7800XT.md).
En ese log, el lote 100 ya reporta `Loss = 0.000478`, el mismo valor que el
lote 400; la «reducción» compara el primer lote (0.000552) con el último, no
una curva monótona.

x10 y x20 (código de entrenamiento) ya estaban en `main` vía PR #49
(`928c8d5`, merge de `d39a5c5`). Esta integración añade el paquete zip x20,
toda la escala x50, la evaluación multi-semilla y el trabajo posterior de la
rama (pesos cargados, Gemma, ciclo infinito).

## Evaluación del arnés propio (no carga los pesos GPU)

Comando: `bio_quantum_engine eval <dev|confirm>`.

El motor de `eval` se construye con `BioQuantumConstellationEngine::new`
(configuración CPU por defecto, 72 sinapsis) y **no lee**
`weights_x50.bin` / `weights_x20.bin`. Los números de abajo miden ese motor
adaptativo por episodio, no el modelo entrenado en GPU. Fuente original:
[`docs/bq10_gpu_amd/logs/eval_dev.log`](../bq10_gpu_amd/logs/eval_dev.log) y
[`eval_confirm.log`](../bq10_gpu_amd/logs/eval_confirm.log). Documentado
también en [`docs/bq10_gpu_amd/CONFIRMACION.md`](../bq10_gpu_amd/CONFIRMACION.md).

«Med log err» = media, entre 16 semillas, de la mediana del log-error
relativo por semilla (no es la mediana global del arnés CST4). 16 episodios
de familias conocidas + 16 OOD por semilla (64 total), umbral `rel_err < 0.05`,
E33 = acc ≥ 80 % en horizontes `{1,2,4,8,16}`. Semillas DEV `0xA400–0xA40F`,
CONFIRM `0xB400–0xB40F`.

| Fase | known acc16 | known med log | OOD acc16 | OOD med log | E33 | Dale |
|---|---:|---:|---:|---:|---|---:|
| DEV | 100.00 % | −12.7469 | 99.80 % | −8.1930 | 16/16 | 0 |
| CONFIRM | 100.00 % | −12.8178 | 100.00 % | −7.8845 | 16/16 | 0 |

Un log-error de ≈ −12.75 equivale a un error relativo del orden de \(3\times10^{-6}\):
el arnés propio es una dinámica sintética mucho más simple que CST4.

## Evaluación *con* los pesos entrenados (`eval_trained`)

Código: `bio_quantum_engine/src/bin/eval_trained/`. Port CPU f32 del kernel
WGSL `forward_pass`. Informe:
[`docs/bq10_gpu_amd/EVAL_PESOS_ENTRENADOS.md`](../bq10_gpu_amd/EVAL_PESOS_ENTRENADOS.md).
Logs: `docs/bq10_gpu_amd/logs/ev_{x50,x20,persist}_{dev,confirm}.log`.

| Modelo | Fase | known acc16 | known med log | new acc16 | new med log | E33 | Dale |
|---|---|---:|---:|---:|---:|---|---:|
| x50 entrenado | DEV | 0.00 % | +0.591 | 0.00 % | +0.449 | 0/16 | 0 |
| x50 entrenado | CONFIRM | 0.00 % | +0.599 | 0.00 % | +0.446 | 0/16 | 0 |
| x20 entrenado | DEV | 0.00 % | +2.438 | 0.00 % | +0.715 | 0/16 | 0 |
| x20 entrenado | CONFIRM | 0.00 % | +1.603 | 0.00 % | +0.490 | 0/16 | 0 |
| Persistencia (pred = último frame) | DEV | 28.52 % | −1.155 | 3.03 % | −0.525 | 0/16 | — |
| Persistencia | CONFIRM | 31.45 % | −1.204 | 0.88 % | −0.527 | 0/16 | — |

Con los pesos GPU el error relativo final es > 1 (log > 0), peor que la
persistencia. Chequeo `trainloss` (misma distribución que `gpu-train`, 1 paso):
MSE x50 = 0.1769, x20 = 0.0668, persistencia = 0.0047, frente al 0.000478
reportado por el shader.

## Limitaciones

1. **`eval` ignora el entrenamiento GPU.** Con o sin `gpu-train`, esos números
   salen del motor CPU por defecto.
2. **La GPU se entrenó con una espiral/círculo sintético** (rotación
   \(\theta+0.08\), contracción \(r\cdot 0.995\)), no con las familias CST4.
3. **La pérdida del shader no es fiable.** Muchos hilos escriben
   `metrics[0] = metrics[0] + …` y actualizan `wp_exc[j]` sin atómicos
   (carrera de datos). Eso encaja con la discrepancia MSE 0.000478 vs 0.1769
   del port CPU y con una curva que no desciende de forma monótona.
4. **El arnés no es el canónico de ÓRBITA/CST4.** No hay baseline pareado de
   v4 en este arnés, ni Paso 0 (hash v4 `1985bdbad5eee41f` en `0xC420`), ni
   bootstrap pareado (10 000, seed 12345, δ=0.05). La impresión
   «+14.5× vs 6.9 % en v4» era un `println!` fijo: **no verificada** y
   retirada del binario en esta integración. Los logs históricos
   (`eval_dev.log`, `eval_confirm.log`) conservan el texto original.
5. **No hay prerregistro** de esta línea GPU (a diferencia de
   [`docs/bio_quantum_v10/PREREG.md`](../bio_quantum_v10/PREREG.md)).
6. **No es comparable con ATLAS-FW ni con el BQ10 canónico.** Un −12.75 de
   este arnés no se puede poner al lado de un −0.41 / −0.61 del arnés CST4.
7. **Binarios en el repo** (todos ≪ 50 MB; no hace falta Git LFS):

   | Archivo | Tamaño |
   |---|---:|
   | `bio_quantum_v10_x20_trained.zip` | 32 756 B |
   | `bio_quantum_v10_x50_trained.zip` | 43 319 B |
   | `bio_quantum_engine/weights_x20.bin` | 8 192 B |
   | `bio_quantum_engine/weights_x50.bin` | 16 384 B |
   | `bio_quantum_engine/weights_infinite_vram.bin` | 16 384 B |

## Trabajo posterior en la misma rama (exploratorio)

Incluido porque formaba parte de `exp/bq10-gpu-amd` al integrar. No sustituye
al protocolo canónico ni se usa para reclamar superioridad frente a v4 o
ATLAS-FW:

- Gemma 2 ↔ BQ10-GPU: [`docs/RESULTADOS_GEMMA_BQ10_GPU.md`](../RESULTADOS_GEMMA_BQ10_GPU.md)
- Benchmark Gemma vs híbrido: [`docs/BENCHMARK_GEMMA_VS_HIBRIDO_GPU.md`](../BENCHMARK_GEMMA_VS_HIBRIDO_GPU.md)
- Memoria / razonamiento continuo: [`docs/DINAMICA_MEMORIA_RAZONAMIENTO_GPU.md`](../DINAMICA_MEMORIA_RAZONAMIENTO_GPU.md)
- Ciclo infinito de auto-aprendizaje: [`docs/CICLO_INFINITO_AUTOAPRENDIZAJE_GPU.md`](../CICLO_INFINITO_AUTOAPRENDIZAJE_GPU.md)
  (el commit `6808294` menciona 21 135 ciclos; detalle en ese informe y en
  `bio_quantum_engine/infinite_learning_results.json`).

## Cómo reproducir

```bash
# Compilar el crate (independiente; no entra en el workspace de la raíz)
cargo build --release --manifest-path bio_quantum_engine/Cargo.toml

# Evaluación del arnés propio (CPU; no carga weights_x50.bin)
./bio_quantum_engine/target/release/bio_quantum_engine eval dev
./bio_quantum_engine/target/release/bio_quantum_engine eval confirm

# Evaluación con pesos GPU exportados (port CPU del forward WGSL)
cargo run --release --manifest-path bio_quantum_engine/Cargo.toml \
  --bin eval_trained -- bio_quantum_engine/weights_x50.bin dev

# Entrenamiento GPU x50 (requiere adaptador WGPU; en esta máquina se corrió
# sobre AMD RX 7800 XT / D3D12). No se reejecuta en CI.
# cargo run --release --manifest-path bio_quantum_engine/Cargo.toml -- gpu-audit
# cargo run --release --manifest-path bio_quantum_engine/Cargo.toml -- gpu-train 400 4096 0.05
```

Para el BQ10 *canónico* (arnés CST4, sí comparable con v4 y ATLAS-FW):

```bash
cargo build --release --manifest-path orbita/Cargo.toml
orbita/target/release/orbita atlas-run --phase bq_dev --seed 0xA400 \
  --weights artifacts/orbita/dev/weights_0xA400.json \
  --out artifacts/bio_quantum_v10 --arms v4,BQ10
```
