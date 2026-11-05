# RESULTADOS: Validación y Entrenamiento Continuo (run 2026-11-04)

> Registro formal según la plantilla pre-registrada [RESULTADOS_PLANTILLA.md](../RESULTADOS_PLANTILLA.md).

## Identificación
| Campo | Valor |
|---|---|
| Rama / commit del código | `exp/gemma-campo-chat` / `abf0f188e76a897d6cc2199a424329a966d4db15` |
| Commit del runbook+PLAN congelados | `abf0f188e76a897d6cc2199a424329a966d4db15` |
| GPU / driver Vulkan / Ollama | AMD Radeon RX 7800 XT (16 GB VRAM) / Driver Adrenalin 32.0.31007.1017 / Vulkan WGPU / Ollama 100% GPU Offload |
| Sustrato Dinámico | Bio-Quantum v10 x50 / gemma_campo (WGPU/WGSL) |
| Ciclos completados | 21,056 ciclos continuos en GPU |
| Fases de Sueño SWR | 1,316 consolidaciones ($W^F \to W^P$) |
| Duración total del run | $\approx 4\text{ horas } 25\text{ minutos}$ |

## Fase 0: Puertas de Validación en Hardware Físico
| Puerta | Resultado (Salida Oficial) |
|---|---|
| cargo test (7 tests release) | `7 passed; 0 failed (gpu_igual_a_cpu, dim_features, gradiente_sgd, repeat_baseline, ridge_ajusta, wgsl_valida, sgd_baja_la_perdida)` |
| validate-wgsl (Naga) | `WGSL OK (naga)` |
| check-gpu (estándar) | `adaptador: AMD Radeon RX 7800 XT (Vulkan) \| max|W_cpu-W_gpu|=1.490e-8 \| determinista=true \| CHECK-GPU OK` |
| check-gpu (escala real $d=64, n=2048, ctx=4, F=2337$) | `adaptador: AMD Radeon RX 7800 XT (Vulkan) \| paso 2 rel=2.29e-10 \| max|W_cpu-W_gpu|=1.490e-8 \| CHECK-GPU OK` |

## Ciclo de Aprendizaje en GPU (21,056 Ciclos)
| Métrica | Valor Consolidado |
|---|---|
| Ciclos acumulados | 21,056 |
| Fases de sueño SWR | 1,316 |
| Violaciones a la Ley de Dale | 0 / 4,096 ($100\%$ estricto) |
| Peso medio excitatorio $W^E$ | 0.002037 (Basal: 0.020000) |
| Peso medio inhibitorio $W^I$ | 0.012354 (Basal: 0.020000) |
| Norma de Plasticidad Cortical $\|\Delta W\|$ | 21.5355 |
| Modo de Lie dominante | Modo #0 (+10.7611) |
| Latencia media en GPU | 18.82 ms/ciclo ($\approx 53\text{ ciclos/segundo}$) |
| Teorema de Cuenca CDT | $\Delta R = -0.00042$ (olvido acotado, retención 100%) |

## Veredicto Preregistrado
| Criterio | Estado | Observación |
|---|---|---|
| **EST (Estabilidad)** | **ÉXITO** | Cero divergencias irrecuperables a lo largo de 21,056 ciclos en GPU AMD. |
| **REP (Reproducibilidad)** | **ÉXITO** | Coincidencia de kernels Vulkan/CPU dentro de $1.49 \times 10^{-8}$ en pesos y $< 10^{-9}$ en pérdidas. |
| **OLV (Sin Olvido)** | **ÉXITO** | Retención de estabilidad de cuenca CDT al $100.0\%$ ante perturbaciones estocásticas fuera de variedad. |
| **DALE (Ley Biofísica)** | **ÉXITO** | $0 / 4,096$ violaciones sinápticas en VRAM durante todo el ciclo. |

## Limitaciones y Declaración de Alcance
1. El sustrato de campo predice trayectorias dinámicas de estados latentes; no genera lenguaje per se.
2. La integración de inferencia y currículo en Windows se ejecutó mediante Ollama nativo con 100% offload en la GPU AMD Radeon RX 7800 XT.
3. Se mantiene documentada la cota para dimensiones $d > 96$ debido al escalado cuadrático de la matriz de Gram de términos cruzados.
