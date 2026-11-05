# RESULTADOS: entrenamiento continuo gemma_campo (run 20261104)

> Solo números copiados de los archivos indicados (`eval_test.json`, `stream_log.jsonl`, `train_logs/shard_*_train_log.txt`, `fase0.txt`, `olvido_final.txt`, `rep_check.json`). Las celdas vacías significan "no medido". Los negativos se informan.

## Identificación
| Campo | Valor |
|---|---|
| Rama / commit del código | `exp/gemma-campo-infinito` / `4987f5623d73c0ccaa58aa20b2380923479a53cc` ([`commit.txt`](commit.txt)) |
| Commit del runbook+PLAN congelados | `abf0f18` ([`RUNBOOK_ENTRENAMIENTO_INFINITO.md`](../RUNBOOK_ENTRENAMIENTO_INFINITO.md)) / `82271cb` ([`PLAN.md`](../PLAN.md)) |
| GPU / driver Vulkan / ROCm / torch | `AMD Radeon RX 7800 XT (Vulkan)` (campo `wgpu`) + `privateuseone:0 AMD Radeon RX 7800 XT` (`torch==2.4.1`, `torch-directml==0.2.5.dev240914`, `transformers==4.57.6`; ver [`env_freeze.txt`](env_freeze.txt)); modelo base congelado: `qwen/qwen3-vl-4b` ($L=13$ de 36 capas, $H=2560$) |
| `data_sha.txt` (JSONL) y fuentes | `2e9ca4ca11abb7cb36f12645926c7d87ef6cb54020a33b19d460b0f1a85e6dbe  all.jsonl` ([`data_sha.txt`](data_sha.txt)) — fuente: `HuggingFaceTB/everyday-conversations-llama3.1-2k` (`train_sft`, 2,260 conversaciones) |
| SHA-256 encoder.npz / model.bin final | `encoder.npz`: `41e6c15b3ecc807ab55fbef1b74b6de0586740b9eb7adb80f7e44197b5633152` <br> `model.bin`: `d8fae45bc96b6ea237e0edffbd679a6d02b67ad6abe8507a1434ba10e255734a` ([`sha.txt`](sha.txt)) |
| Shards completados / conv train totales / tokens | `9 shards (0..8)` completados / `1,800 conv train` (`301,606 tokens train` acumulados) + held-out fijo en shard 0 (`60 conv val / 9,600 tokens`, `60 conv test / 9,985 tokens`) |
| Duración total | `31 min 26 s` bucle de shards (`2026-11-04T23:51:10-06:00` a `2026-11-05T00:22:36-06:00`) + `10 min 42 s` evaluación final y REP (`00:25:05` a `00:35:47`) |

## Fase 0
| Puerta | Resultado (copiar salida) |
|---|---|
| cargo test (7) | `test gpu_igual_a_cpu ... ok` (`1 passed; 0 failed`) + `6 passed; 0 failed` unitarios (`dim_features`, `gradiente_sgd_coincide_con_diferencias_finitas`, `repeat_baseline_y_rapido_causal`, `ridge_ajusta_y_baja_de_repeat`, `wgsl_valida`, `sgd_baja_la_perdida`) + `WGSL OK (naga)` ([`fase0.txt`](fase0.txt)) |
| check-gpu (pequeño / d=64) | Pequeño ($d=8, n=96$): `adaptador: AMD Radeon RX 7800 XT (Vulkan)` `max\|W_cpu-W_gpu\|=1.490e-8  determinista=true  CHECK-GPU OK` <br> Escala real ($d=64, n=2048, \text{ctx}=4, F=2337$): `max\|W_cpu-W_gpu\|=1.490e-8  determinista=true  CHECK-GPU OK` ([`fase0.txt`](fase0.txt)) |
| parity.py | `paridad numpy vs Rust: max err relativo = 1.72e-08` — `PARIDAD OK` ([`parity.txt`](parity.txt)) |
| sanity_identity_maxdiff (modelo real) | `0.0` en GPU `float16` ([`g5_eval_test.json`](g5_eval_test.json) y [`eval_test.json`](eval_test.json); `0.005409926176071167` en CPU `float32`, ambos $< 0.05$) |

## Codificador (shard 0)
| d_field | var. explicada | recon rel. train / val / test |
|---|---|---|
| `64` ($H=2560$, `fit_tokens=32703`) | `0.998367060873689` (`99.84%`) | `0.0016339428257197142` / `0.0017189040081575513` / `0.0017252065008506179` ([`encoder_info_shard0.json`](encoder_info_shard0.json)) |

## Entrenamiento (de [`stream_log.jsonl`](stream_log.jsonl))
| shard | lr | val_dz_mse_lenta | val_z_mse_campo | olvido_shard1_z_mse | reintentos |
|---|---|---|---|---|---|
| 0 | `0.05` | `0.43141` | `0.428104` | `null` | `0` |
| 1 | `0.048795` | `0.423617` | `0.420708` | `0.407513` | `0` |
| 2 | `0.0476731` | `0.418779` | `0.416145` | `0.404656` | `0` |
| 3 | `0.0466252` | `0.417151` | `0.41465` | `0.402799` | `0` |
| 4 | `0.0456435` | `0.415353` | `0.412983` | `0.403927` | `0` |
| 5 | `0.0447214` | `0.415231` | `0.412838` | `0.403589` | `0` |
| 6 | `0.0438529` | `0.414183` | `0.411846` | `0.40323` | `0` |
| 7 | `0.0430331` | `0.414183` | `0.411846` | `0.40323` | `0` |
| 8 | `0.0422577` | `0.414183` | `0.411846` | `0.40323` | `0` |

Divergencias: `0` (recuperadas `0` / no recuperadas `0`)

## Test: z-MSE ([`eval_test.json`](eval_test.json), bootstrap 10000, semilla 12345)
| k | campo | repeat | campo−repeat media [IC95] | P(Δ≤0) |
|---|---|---|---|---|
| 1 (todas las pos.) | `0.417240` | `1.238695` | `-0.821455` `[ -0.841033, -0.801295 ]` | `1.0` |
| 1 (rollout stride 4) | `0.412197` | `1.289652` | `-0.877455` `[ -0.902244, -0.852541 ]` | `1.0` |
| 2 | `0.567080` | `1.486495` | `-0.919416` `[ -0.943512, -0.896036 ]` | `1.0` |
| 4 | `0.691082` | `1.615537` | `-0.924456` `[ -0.946453, -0.901925 ]` | `1.0` |
| 8 | `0.793802` | `1.686998` | `-0.893195` `[ -0.913777, -0.871005 ]` | `1.0` |
| campo−lenta (k=1) | `0.417240` | `0.418979` (lenta) | `-0.001739` `[ -0.002217, -0.001320 ]` | `1.0` |

## Test: perplejidad ([`eval_test.json`](eval_test.json))
| Condición | NLL/token | PPL |
|---|---|---|
| gemma (`qwen3-vl-4b` intacto) | `1.714566` | `5.554262` |
| oracle (cuello PCA) | `2.946072` | `19.031062` |
| repeat-last | `11.837952` | `138406.683540` |
| campo | `5.890709` | `361.661495` |

| Δ (por conversación) | media [IC95] | P(Δ≤0) |
|---|---|---|
| campo−gemma | `+4.176143` `[ 4.001667, 4.356745 ]` | `0.0` |
| campo−oracle | `+2.944636` `[ 2.853318, 3.037305 ]` | `0.0` |
| campo−repeat | `-5.947243` `[ -6.175854, -5.718967 ]` | `1.0` |
| oracle−gemma | `+1.231507` `[ 1.083606, 1.404715 ]` | `0.0` |
| repeat−gemma | `+10.123386` `[ 9.987274, 10.259928 ]` | `0.0` |

## Olvido ([`olvido_final.txt`](olvido_final.txt) y [`stream_log.jsonl`](stream_log.jsonl))
| shard | z-MSE justo tras entrenarlo | z-MSE con el ckpt final | ratio |
|---|---|---|---|
| 1 | `0.407513` (mín. en el run: `0.402799` en shard 3) | `0.403230` | `0.989490` (vs mín. del run: `1.001070 <= 1.05`) |
| 2 | `0.405996` | `0.412632` | `1.016345` |
| 3 | `0.407580` | `0.411625` | `1.009924` |

## Veredicto (éxito / fallo / inconcluso, con la cifra que lo decide)
| H1 | H1k (2,4,8) | H2 (≤0,05 nat) | H3 | OLV | EST | REP |
|---|---|---|---|---|---|---|
| **ÉXITO** (`ic95[1] = -0.801295 < 0`) | **ÉXITO** ($k=2$: `ic95[1] = -0.896036 < 0`; $k=4$: `ic95[1] = -0.901925 < 0`; $k=8$: `ic95[1] = -0.871005 < 0`) | **FALLO** (`ic95 = [2.853318, 3.037305] > 0.05`) | **ÉXITO** (`ic95[1] = -5.718967 < 0`) | **ÉXITO** (`0.403230 / 0.402799 = 1.00107 <= 1.05`) | **ÉXITO** (`0` abortos / `0` divergencias) | **ÉXITO** (`rep_predict_byte_identical=true`, `max_nll_diff=0.0`, `max_train_rel_diff=0.0`; ver [`rep_check.json`](rep_check.json)) |

## Muestras (de [`muestras.txt`](muestras.txt), 5 prompts × {campo, gemma})
```text
=== PROMPT: Hi! Any tips for a rainy weekend? ===
[campo] Assistant: Hi! ’s — —’s,..’s,...,,.,,,,,,,,,,,,,,,,,,,,,,
[gemma] Assistant: Hi! 😊 Rainy weekends can be cozy, creative, or even productive — here are some fun and practical tips to make the most of it:
=== PROMPT: Can you explain how rainbows form? ===
[campo] Assistant: Certainly! Rain’s — form the a — of — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —
[gemma] Assistant: Certainly! Rainbows form due to a combination of **refraction**, **reflection**, and **dispersion** of light in water droplets — typically in the atmosphere after rain. Here’s a step
=== PROMPT: I need help planning a birthday party. ===
[campo] Assistant: I! happy’s to to you you. to’s you to to to..’s, to to to to. to to to to..’s to to to to.. to to to
[gemma] Assistant: I’d be happy to help you plan a birthday party! Let’s start by answering a few key questions to make it perfect for your guest and your vibe. Here’s how we can proceed:
=== PROMPT: What should I cook tonight? ===
[campo] Assistant: I! happy! to! you!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[gemma] Assistant: I can help you pick a delicious and easy dinner! Here are a few options, depending on your mood and what you have on hand:
=== PROMPT: How do I start learning guitar? ===
[campo] Assistant: Great! — you’re on to. a’s thing..’s..’s.......................
[gemma] Assistant: Great question! Starting to learn guitar is exciting and rewarding. Here’s a step-by-step guide to help you get started:
```

## Limitaciones y lo que NO se puede afirmar
1. **El campo no añade capacidad de lenguaje:** Las capas 1–13, las capas 14–36 y el LM head pertenecen al modelo base congelado (`qwen/qwen3-vl-4b`).
2. **Predicción causal sin ver el token actual ($t$) y fallo esperado de H2:** El campo predice $\hat{z}_t$ a partir de $z_{\le t-1}$ **sin observar el token $t$**, mientras que `oracle` codifica y decodifica $h_t$ habiendo observado ya el token $t$. Por ello $\Delta\text{NLL}_{\text{campo-oracle}} = +2.9446\text{ nat/token}$ (`> 0.05`, fallo de H2) y $\Delta\text{NLL}_{\text{campo-gemma}} = +4.1761\text{ nat/token}$, tal como anticipaba el diseño del experimento en `RUNBOOK_ENTRENAMIENTO_INFINITO.md` §5. Asimismo, aunque `oracle` explica el `99.84%` de la varianza en la capa 13, la compresión lineal a $d_{\text{field}}=64$ introduce un coste de cuello de botella de $\Delta\text{NLL}_{\text{oracle-gemma}} = +1.2315\text{ nat/token}$ (PPL pasa de `5.55` a `19.03`).
3. **La mejora contundente en $z$-MSE (H1, H1k) y frente a `repeat` (H3) no equivale a generación fluida en rollout libre:** En `chat_demo.py` (`--mode campo`), aunque los primeros tokens mantienen la intención semántica inicial (`"Hi!"`, `"Certainly! Rain..."`, `"Great!"`), al sustituir el estado de la última posición sin acceso a la identidad exacta del token actual en la capa 13, la generación degenera en repeticiones de puntuación.
4. **Selección repetida sobre `val`:** Los hiperparámetros rápidos ($\eta, \gamma$) se eligen sobre `val` en cada shard (`eta=0.03, gamma=1.0`), lo que hace las métricas de `val` ligeramente optimistas; el split de `test` (`60` conversaciones) se evaluó una única vez al finalizar.
