# RESULTADOS: entrenamiento continuo gemma_campo (run <fecha>)

> Solo números copiados de los archivos indicados. Las celdas vacías significan "no medido". Los negativos se informan.

## Identificación
| Campo | Valor |
|---|---|
| Rama / commit del código | |
| Commit del runbook+PLAN congelados | |
| GPU / driver Vulkan / ROCm / torch | |
| `data_sha.txt` (JSONL) y fuentes | |
| SHA-256 encoder.npz / model.bin final | |
| Shards completados / conv train totales / tokens | |
| Duración total | |

## Fase 0
| Puerta | Resultado (copiar salida) |
|---|---|
| cargo test (7) | |
| check-gpu (pequeño / d=64) | |
| parity.py | |
| sanity_identity_maxdiff (Gemma real) | |

## Codificador (shard 0)
| d_field | var. explicada | recon rel. train / val / test |
|---|---|---|

## Entrenamiento (de `stream_log.jsonl`)
| shard | lr | val_dz_mse_lenta | val_z_mse_campo | olvido_shard1_z_mse | reintentos |
|---|---|---|---|---|---|
Divergencias: __ (recuperadas __ / no recuperadas __)

## Test: z-MSE (eval_test.json, bootstrap 10000, semilla 12345)
| k | campo | repeat | campo−repeat media [IC95] | P(Δ≤0) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |
| campo−lenta (k=1) | | | | |

## Test: perplejidad
| Condición | NLL/token | PPL |
|---|---|---|
| gemma | | |
| oracle (cuello PCA) | | |
| repeat-last | | |
| campo | | |

| Δ (por conversación) | media [IC95] | P(Δ≤0) |
|---|---|---|
| campo−gemma | | |
| campo−oracle | | |
| campo−repeat | | |
| oracle−gemma | | |

## Olvido
| shard | z-MSE justo tras entrenarlo | z-MSE con el ckpt final | ratio |
|---|---|---|---|

## Veredicto (éxito / fallo / inconcluso, con la cifra que lo decide)
| H1 | H1k (2,4,8) | H2 (≤0,05 nat) | H3 | OLV | EST | REP |
|---|---|---|---|---|---|---|

## Muestras (de `muestras.txt`, 5 prompts × {campo, gemma})

## Limitaciones y lo que NO se puede afirmar
(el campo no añade capacidad de lenguaje; predice h_t sin ver el token t; val reutilizada para η/γ por shard; …)
