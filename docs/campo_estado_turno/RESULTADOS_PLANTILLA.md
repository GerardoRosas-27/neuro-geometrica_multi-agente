# RESULTADOS: campo de estado por turno (run AAAAMMDD)

> Solo números copiados de los archivos de este directorio. Celda vacía = no medido. Los negativos se informan.

## Identificación
| Campo | Valor |
|---|---|
| Commit código / PLAN | `` / `84f19f6` |
| Modelo LLM, capa L, pool, dtype, dispositivo | |
| GPU / torch / torch-directml / transformers | |
| SHA-256 JSONL (`data_sha.txt`) | |
| SHA-256 encoder.npz / model.bin / adapter.pt | |
| SHA pesos LLM antes / después (adapter_info.json) | |
| Conversaciones train/val/test; turnos | |
| Desviaciones del RUNBOOK (y por qué) | |

## Fase 0
| Puerta | Resultado |
|---|---|
| cargo test / WGSL / check-gpu | |
| unittest campo_turno (7) | |
| smoke | |

## Codificador
| d | var. explicada | var. media z train/val/test |
|---|---|---|

## Campo (val)
| ridge λ1/λ2 | mejor val dz-MSE SGD | η, γ rápido | val z-MSE campo |
|---|---|---|---|

## T1 / T2 — z-MSE turnos assistant (test, por conversación, B=10000, semilla 12345)
| Predictor | z-MSE medio |
|---|---|
| campo | |
| repeat | |
| repeat_asst | |
| mean | |
| ridge | |
| mlp | |

| Contraste | media | IC95 | ¿pasa? |
|---|---|---|---|
| T1 campo − repeat (sup<0) | | | |
| T1 campo − mean (sup<0) | | | |
| campo − repeat_asst (info) | | | — |
| campo − ridge (info) | | | — |
| T2 campo − mlp (sup ≤ umbral = 0.05·mlp) | | | |

## T3 / T4 — NLL/token de la respuesta real (test)
| Condición | NLL | PPL |
|---|---|---|
| none | | |
| campo | | |
| true | | |
| shuf | | |
| mlp | | |

| Contraste | media | IC95 | ¿pasa? |
|---|---|---|---|
| T3 campo − none (sup<0) | | | |
| T4 campo − true (brecha, info) | | | — |
| T4 campo − shuf (sup<0) | | | |
| true − none (techo, info) | | | — |

## No-daño
| Chequeo | Valor |
|---|---|
| identidad embeds vs ids (máx |Δ|, límite) | |
| LLM intacto (train adaptador / eval) | |

## Muestras (5 por condición, literales de `eval_test.json`)
| conv/turno | real | none | campo | true | shuf |
|---|---|---|---|---|---|

## Conclusión (una línea por hipótesis, sin reinterpretar)
- T1:
- T2:
- T3:
- T4:
- No-daño:
