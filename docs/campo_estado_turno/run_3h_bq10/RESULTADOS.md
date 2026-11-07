# RESULTADOS: Segunda Línea de Investigación — Campo de Estado por Turno con BQ10 (Run 3h BQ10)

> Solo números reales copiados de los archivos verificados (`eval_test.json`, `model.json`, `adapter_info.json`, `adapter_log.txt`, `baselines_log.json`).

## Identificación
| Campo | Valor |
|---|---|
| Línea de Investigación | **Segunda Línea**: Campo de estado continuo por turno con arquitectura BQ10 |
| Repositorio / Rama | `investigcion-IA` / `exp/campo-estado-turno` |
| Arquitectura del Campo | **BQ10** en dinámicas de turno ($z_t \in \mathbb{R}^{64}$, $F=2721$ modos ortogonales de Lie acoplados) |
| Decodificador / Interfaz LLM | Adaptador Soft-Prompt $z \to P \cdot H$ ($P=4$ tokens virtuales, ancho 512) entrenado en GPU DirectML |
| GPU / Dispositivo | AMD Radeon RX 7800 XT (`wgpu`/Vulkan en Rust + DirectML `privateuseone:0` en PyTorch) |
| Modelo Base Congelado | `qwen3_vl_4b_hf` ($L=13$, `float16`, pesos congelados intactos) |
| Duración Total del Entrenamiento | **3.0 horas** (1.5h campo continuo Rust/GPU + 1.5h decodificador soft-prompt 5 épocas) |
| Dataset / División | `all.jsonl` (train: 1,600 conv, 12,385 turnos; val: 200 conv; test: 200 conv, 759 turnos assistant) |

## Fase 0 (Verificación de Integridad y Pesos)
| Puerta | Criterio | Resultado | Veredicto |
|---|---|---|---|
| Invariancia de Pesos LLM | $\text{SHA}_{\text{antes}} == \text{SHA}_{\text{después}}$ | `78685cdf6081c3802b007200ddbba61ae24eb29c7ca25416baeacdefef7484ee` (idéntico) | **PASA** |
| Sanity Identity | $\max\|\text{embeds} - \text{ids}\| < 0.05$ | **0.0** | **PASA** |
| Kernel de Dinámicas BQ10 | $\max\|W_{\text{cpu}} - W_{\text{gpu}}\| < 10^{-6}$ | $1.490\times 10^{-8}$ (determinista) | **PASA** |

## T1 y T2: Predicción de Estados Latentes por Turno ($z$-MSE en Turnos Assistant Test)
| Predictor | $z$-MSE Medio |
|---|---|
| **campo (BQ10 $F=2721$)** | **0.368282** |
| mlp (control no lineal 2 capas) | 0.386873 |
| ridge (regresión regularizada) | 0.406254 |
| mean (media global histórica) | 0.987285 |
| repeat (repetir último turno) | 1.091795 |
| repeat_asst (repetir último turno de asistente) | 1.143433 |

### Contrastes Estadísticos (Bootstrap $B=10000$, Semilla 12345, $n=200$ Conversaciones)
| Hipótesis | Contraste | Diferencia Media | IC 95% | Criterio de Éxito | Veredicto |
|---|---|---|---|---|---|
| **T1** | campo − repeat | **−0.723514** | [−0.755422, −0.692156] | Sup $< 0$ ($P \le 0 = 1.0$) | **ÉXITO** |
| **T1** | campo − mean | **−0.619003** | [−0.647813, −0.590987] | Sup $< 0$ ($P \le 0 = 1.0$) | **ÉXITO** |
| **T2** | campo − mlp | **−0.018591** | [−0.024680, −0.012612] | Sup $\le 0.019344$ (5% margen) | **ÉXITO** |

> El campo BQ10 supera a repeat en un **66.3%** y bate al MLP no lineal en **0.0186** unidades de $z$-MSE.

## T3 y T4: Decodificación de Texto y NLL en el LLM Congelado (759 Turnos Assistant Test)
| Condición de Decodificación | NLL / token | PPL |
|---|---|---|
| **none** (sin prefijo de estado latente) | 1.945794 | 6.999187 |
| **shuf** (prefijo con estado de otra conversación aleatoria) | 1.040223 | 2.829848 |
| **mlp** (prefijo con estado predicho por MLP) | 0.627911 | 1.873691 |
| **campo (prefijo con estado predicho por BQ10)** | **0.628333** | **1.874483** |
| **true** (oráculo: prefijo con el estado latente real) | 0.584070 | 1.793322 |

### Contrastes de Eficacia y Especificidad Contextual
| Hipótesis | Contraste | Diferencia Media | IC 95% | Veredicto |
|---|---|---|---|---|
| **T3 (Eficacia Decodificación)** | $\text{NLL}(\text{campo}) − \text{NLL}(\text{none})$ | **−1.317461** | [−1.349952, −1.285876] | **ÉXITO** ($P \le 0 = 1.0$) |
| **T4 (Especificidad de Diálogo)** | $\text{NLL}(\text{campo}) − \text{NLL}(\text{shuf})$ | **−0.411890** | [−0.442310, −0.380553] | **ÉXITO** ($P \le 0 = 1.0$) |
| **Brecha vs Oráculo Real** | $\text{NLL}(\text{campo}) − \text{NLL}(\text{true})$ | **+0.044263** | [+0.035962, +0.054389] | — |

## Veredicto Consolidado Segunda Línea
Todas las 4 hipótesis prerregistradas (**T1, T2, T3, T4**) y los criterios de **No-Daño** son **EXITOSAS**. Inyectar el estado latente predicho por BQ10 reduce la perplejidad de generación de 7.00 a 1.87 (reducción del 73.2% en incertidumbre), superando el control de barajado en 0.41 nats.
