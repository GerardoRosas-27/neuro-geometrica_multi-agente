# RESULTADOS: Línea Principal — Entrenamiento Continuo gemma_campo con Arquitectura BQ10 (Run 3h BQ10)

> Solo números reales copiados de los archivos de salida verificados (`eval_test.json`, `test_zmse.json`, `model.json`, `muestras.txt`, `master_train_6h_log.txt`).

## Identificación
| Campo | Valor |
|---|---|
| Línea de Investigación | **Línea Principal**: Sustrato de campo continuo token a token con BQ10 |
| Rama / Repositorio | `exp/gemma-campo-chat` / `exp/gemma-campo-infinito` (`wt_gemma_campo_chat` / `investigcion-IA`) |
| Arquitectura del Campo | **BQ10** (`M1: decay` + `M2: momentum` + `M6: dual-rate fast/slow` + `M7: low-rank context projection da=10`) |
| GPU / Kernel / Backend | AMD Radeon RX 7800 XT (`wgpu`/Vulkan en Rust + DirectML `privateuseone:0` en Python) |
| Modelo Base Congelado | `qwen3_vl_4b_hf` ($L=13$ de 36 capas, $H=2560$, subespacio contextual $k_{\text{skip}}=16$) |
| Innovación Causal | Proyección ortogonal de innovación de token $e(x_t)$ acoplada en tiempo real |
| Duración del Entrenamiento | **3.02 horas continuas** (67 ciclos de shards `0..8` con replay y decaimiento térmico) |
| Total Shards / Conversaciones | 9 shards (`0..8`), 1,800 conversaciones train (held-out fijo: 60 conv val, 60 conv test) |

## Fase 0 (Verificación de Kernel y Paridad)
| Puerta | Resultado | Veredicto |
|---|---|---|
| `cargo test --release` (7 tests) | 7 pasados (1 GPU + 6 unitarios Rust) | **PASA** |
| `validate-wgsl` (naga) | WGSL shader validado sin errores | **PASA** |
| `check-gpu` ($F=2337, d=64, n=2048$) | $\max\|W_{\text{cpu}} - W_{\text{gpu}}\| = 1.490\times 10^{-8}$, determinista = true | **PASA** |
| Sanity Identity $\max\|\text{embeds} - \text{ids}\|$ | `0.0` en GPU `float16` ($< 0.05$) | **PASA** |

## Test: Predicción del Estado Oculto $z$-MSE (Bootstrap $B=10000$, Semilla 12345)
| Predictor / Condición | $z$-MSE Medio | IC 95% |
|---|---|---|
| **campo (con innovación causal $e(x_t)$)** | **0.195968** | [0.190373, 0.201465] |
| campo_sin_tok (solo contexto $\le t-1$) | 0.414278 | [0.404943, 0.422627] |
| lenta (media móvil exponencial $\gamma=0.999$) | 0.415831 | [0.406719, 0.423992] |
| repeat (línea base repetir último estado) | 1.238695 | [1.214170, 1.260486] |

### Contrastes Estadísticos (1-paso)
| Contraste | Diferencia Media | IC 95% | $P(\le 0)$ | Veredicto |
|---|---|---|---|---|
| **campo − repeat** | **−1.042726** | [−1.065392, −1.017930] | 1.000 | **ÉXITO** (H1) |
| **campo − lenta** | **−0.219862** | [−0.225814, −0.213134] | 1.000 | **ÉXITO** |
| **campo_sin_tok − repeat** | **−0.824417** | [−0.844211, −0.804111] | 1.000 | **ÉXITO** |

## Test: Rollouts Multi-Paso $k=1..8$ ($z$-MSE campo vs repeat)
| $k$ | $z$-MSE campo | $z$-MSE repeat | Diferencia campo − repeat | IC 95% |
|---|---|---|---|---|
| **1** | 0.195968 | 1.289652 | **−1.093683** | [−1.121295, −1.064237] |
| **2** | 0.566041 | 1.486495 | **−0.920454** | [−0.944623, −0.896860] |
| **3** | 0.655881 | 1.590589 | **−0.934708** | [−0.955071, −0.914705] |
| **4** | 0.690488 | 1.615537 | **−0.925049** | [−0.947013, −0.902519] |
| **5** | 0.726738 | 1.660146 | **−0.933408** | [−0.955072, −0.911288] |
| **6** | 0.750655 | 1.654174 | **−0.903519** | [−0.924540, −0.883399] |
| **7** | 0.785029 | 1.696356 | **−0.911328** | [−0.931740, −0.890992] |
| **8** | 0.795985 | 1.686998 | **−0.891013** | [−0.911499, −0.868853] |

> En todos los horizontes $k \in \{1..8\}$, el campo BQ10 supera a repeat con significancia estadística ($P \le 0 = 1.0$). **H1k: ÉXITO**.

## Test: Generación y Perplejidad en LLM Congelado ($k_{\text{skip}}=16$ + $e(x_t)$)
| Condición | NLL / token | PPL | $\Delta\text{NLL}$ vs base |
|---|---|---|---|
| **gemma (Qwen 4B intacto sin intervenir)** | 1.714556 | 5.55421 | — |
| **oracle (proyección PCA directa del estado real)** | 1.715037 | 5.55688 | +0.000481 |
| **campo (BQ10 continuo)** | **1.733631** | **5.66117** | **+0.019074** |
| **repeat (intervención con estado repetido)** | 3.049482 | 21.10442 | +1.334926 |

### Contrastes de Hipótesis Prerregistradas de NLL
| Hipótesis | Métrica | Valor Obtenido | Criterio de Éxito | Veredicto |
|---|---|---|---|---|
| **H2** | $\Delta\text{NLL}_{\text{campo}-\text{oracle}}$ | **+0.018593** (IC95: [+0.010154, +0.027282]) | $\le +0.05$ | **ÉXITO** |
| **H3** | $\Delta\text{NLL}_{\text{campo}-\text{repeat}}$ | **−1.315852** (IC95: [−1.389505, −1.242132]) | $< 0$ | **ÉXITO** |
| **OLV** | Olvido catastrófico Shard 0 | **+0.064%** | $\le 5.0\%$ | **ÉXITO** |
| **REP** | Reproducibilidad bit a bit | $\max\|W_a - W_b\| = \mathbf{0.000\times 10^{0}}$ | $< 1.0\times 10^{-6}$ | **ÉXITO** |
| **EST** | Estabilidad de pesos | $\|W\|_{\max} = 1.482 < 10.0$, divergencias = 0 | $\|W\| < 10.0$ | **ÉXITO** |

## Veredicto Consolidado Línea Principal
Todas las 7 hipótesis prerregistradas (**H1, H1k, H2, H3, OLV, EST, REP**) son **EXITOSAS**. BQ10 en sustrato continuo logra mantener la perplejidad del LLM en **5.66** (a sólo 0.019 nats del modelo completo) sin requerir almacenar ni recomputar la historia de atención.
