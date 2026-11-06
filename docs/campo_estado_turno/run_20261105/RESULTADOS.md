# RESULTADOS: campo de estado por turno (run 20261105)

> Solo números copiados de los archivos de este directorio. Celda vacía = no medido. Los negativos se informan.

## Identificación
| Campo | Valor |
|---|---|
| Commit código / PLAN | `317c38aebb962f0efc910e89bbecfd21985c0c6c` (`317c38a`) / `84f19f6` |
| Modelo LLM, capa L, pool, dtype, dispositivo | `D:/agentesIA/qwen3_vl_4b_hf` (`qwen/qwen3-vl-4b`), `L=13`, `pool=mean`, `dtype=float16`, `device=dml` (`privateuseone:0`) |
| GPU / torch / torch-directml / transformers | `AMD Radeon RX 7800 XT (Vulkan / DirectML)` / `torch==2.4.1` / `torch-directml==0.2.5.dev240914` / `transformers==4.57.6` |
| SHA-256 JSONL (`data_sha.txt`) | `2e9ca4ca11abb7cb36f12645926c7d87ef6cb54020a33b19d460b0f1a85e6dbe */c/ct_work/all.jsonl` |
| SHA-256 encoder.npz / model.bin / adapter.pt | `8f5813d05fb95cdc6f437fe376127494b88b933ac241ae8d4d449be5a26b3fa0` / `13d2dca20170d895c427d13adb2c64d917f4e808ab9c707b4c49ada222dd6cdf` / `9501be7da8cf688ba59848f1c6fe528a77fa88d999bb6ca7ea4ccc75067db479` |
| SHA pesos LLM antes / después (adapter_info.json) | `78685cdf6081c3802b007200ddbba61ae24eb29c7ca25416baeacdefef7484ee` / `78685cdf6081c3802b007200ddbba61ae24eb29c7ca25416baeacdefef7484ee` |
| Conversaciones train/val/test; turnos | `train: 1600 conv, 12385 turnos, H=2560; val: 200 conv, 1565 turnos, H=2560; test: 200 conv, 1548 turnos, H=2560` |
| Desviaciones del RUNBOOK (y por qué) | 1) `MODEL=D:/agentesIA/qwen3_vl_4b_hf` (`qwen/qwen3-vl-4b`, `LAYER=13`): alternativa prevista en `RUNBOOK.md` y `PLAN.md` §1.1 al no estar definido `HF_TOKEN` para `google/gemma-2-2b`. 2) `torch_directml.disable_tiled_resources(True)` vía `sitecustomize.py` en `PYTHONPATH` (sin tocar código del repo): evita degradación de tablas de páginas D3D12 Tiled Resources en `torch-directml` tras cientos de pasadas con longitudes dinámicas. |

## Fase 0
| Puerta | Resultado |
|---|---|
| cargo test / WGSL / check-gpu | `cargo test --release`: `1 passed` (`gpu_igual_a_cpu`) + `6 passed` (`unit`); `validate-wgsl`: `WGSL OK (naga)`; `check-gpu` (defecto y `--n 2048 --d 64 --ctx 4`): `max|W_cpu-W_gpu|=1.490e-8 determinista=true CHECK-GPU OK` |
| unittest campo_turno (7) | `Ran 7 tests in 9.119s — OK` |
| smoke | `OK` (`best val dz-mse SGD=0.135186; fast elegido eta=0.03 gamma=1 val z-mse=0.134833; mejor época=1 val NLL=5.5084; LLM intacto=True; identidad_ok=True`) |

## Codificador
| d | var. explicada | var. media z train/val/test |
|---|---|---|
| `64` | `0.7814175983846907` | `0.9999166131019592` / `0.9685525894165039` / `0.9627315402030945` |

## Campo (val)
| ridge λ1/λ2 | mejor val dz-MSE SGD | η, γ rápido | val z-MSE campo |
|---|---|---|---|
| `1e-3 / 1e-2` (`lam_deg=[0.0, 0.001, 0.01]`, `val_mse(dz)=0.571910`) | `0.571910` | `eta=0, gamma=1` (`0.0, 1.0`) | `0.571910` (global `gemma_campo`, `repeat=1.250798`); `0.37941965281963347` (turnos assistant en `eval_val`) |

## T1 / T2 — z-MSE turnos assistant (test, por conversación, B=10000, semilla 12345)
| Predictor | z-MSE medio |
|---|---|
| campo | `0.3685527244210243` |
| repeat | `1.0917954957485199` |
| repeat_asst | `1.1434330561757087` |
| mean | `0.9872847780585289` |
| ridge | `0.4062537436932325` |
| mlp | `0.38687286220490935` |

| Contraste | media | IC95 | ¿pasa? |
|---|---|---|---|
| T1 campo − repeat (sup<0) | `-0.7232427713274956` | `[-0.7550630325078964, -0.692163755280897]` | `true` (`p_le0=1.0`, `n=200`) |
| T1 campo − mean (sup<0) | `-0.6187320536375046` | `[-0.6475693066623062, -0.5908669548947364]` | `true` (`p_le0=1.0`, `n=200`) |
| campo − repeat_asst (info) | `-0.7748803317546844` | `[-0.8113546610828489, -0.7396679133661092]` | — |
| campo − ridge (info) | `-0.03770101927220822` | `[-0.04452476458996535, -0.0311911224257201]` | — |
| T2 campo − mlp (sup ≤ umbral = 0.05·mlp) | `-0.018320137783885002` | `[-0.024368760069832206, -0.012354790948331359]` | `true` (`umbral=0.01934364311024547`, `p_le0=1.0`, `n=200`) |

## T3 / T4 — NLL/token de la respuesta real (test)
| Condición | NLL | PPL |
|---|---|---|
| none | `1.9457939405404034` | `6.999186587659266` |
| campo | `0.6475953880107791` | `1.9109402301150247` |
| true | `0.6301269838271273` | `1.8778490205803278` |
| shuf | `0.9346708592687366` | `2.5463752039981356` |
| mlp | `0.6469799832911522` | `1.9097645902627474` |

| Contraste | media | IC95 | ¿pasa? |
|---|---|---|---|
| T3 campo − none (sup<0) | `-1.2981985525296238` | `[-1.3309107688946231, -1.2654467306231854]` | `true` (`p_le0=1.0`, `n=200`) |
| T4 campo − true (brecha, info) | `0.01746840418365173` | `[0.012052836236839417, 0.023955998619875577]` | — |
| T4 campo − shuf (sup<0) | `-0.2870754712579576` | `[-0.3118449081435307, -0.2625430659422041]` | `true` (`p_le0=1.0`, `n=200`) |
| true − none (techo, info) | `-1.3156669567132757` | `[-1.3496889347499663, -1.2817227115775984]` | — |

## No-daño
| Chequeo | Valor |
|---|---|
| identidad embeds vs ids (máx \|Δ\|, límite) | `0.0`, límite `0.05` (`identidad_ok: true`) |
| LLM intacto (train adaptador / eval) | `true` / `true` (`78685cdf6081c3802b007200ddbba61ae24eb29c7ca25416baeacdefef7484ee`) |

## Muestras (5 por condición, literales de `eval_test.json`)
| conv/turno | real | none | campo | true | shuf |
|---|---|---|---|---|---|
| `1813 / 1` | `" Hello! How can I help you today?\n"` | `" Hello! How can I assist you today? 😊"` | `" Hello! How can I help you today?"` | `" Hello! How can I help you today?"` | `" Hello! How can I help you today?"` |
| `1813 / 3` | `" Yes, there are many adaptive sports options. Some popular ones include wheelchair basketball, adaptive skiing, and para-swimming. These sports are designed to be inclusive and accessible for people with different abilities.\n"` | `" Absolutely! There are many wonderful sports and physical activities that are accessible and beneficial for people with disabilities. The key is finding activities that match your abilities, interests, and level of support. Here are some great options:\n\n### 🏃‍♂️ Adaptive Sports & Activities:\n\n1. **Wheel"` | `" Yes, there are many sports that are accessible to people with disabilities. For example, wheelchair basketball, adaptive swimming, and wheelchair tennis are all great options."` | `" Yes, there are many sports that are accessible to people with disabilities. For example, wheelchair basketball, adaptive swimming, and wheelchair tennis are all great options."` | `" Absolutely! There are many sports and physical activities that are excellent for people with disabilities. They promote physical health, mental well-being, social connection, and independence. Here are some great options:\n\n### 1. **Adaptive Sports (Officially Designed for People with Disabilities)**\nThese are sports modified"` |
| `1813 / 5` | `" There are many exercises that can be modified or adapted for people who use wheelchairs. Some examples include chair yoga, wheelchair aerobics, and resistance band exercises. These exercises can help improve strength, flexibility, and overall fitness.\n"` | `" Great question! There are many excellent exercise options for people who use wheelchairs — they’re not just for competition, but also for health, strength, balance, and fun. Here are some great options:\n\n### 1. **Wheelchair Sports (for fun and fitness)**\n- **Wheel"` | `" There are many exercise options for people who use wheelchairs, including wheelchair aerobics, cycling, and strength training. These activities can help improve strength, balance, and overall fitness."` | `" There are many exercise options for people who use wheelchairs, including wheelchair aerobics, wheelchair yoga, and adaptive cycling. These activities can help improve strength, flexibility, and cardiovascular health."` | `" Great question! There are many great exercise options for people who use wheelchairs. Some popular ones include:\n\n1. Wheelchair aerobics – A fun and low-impact way to improve cardiovascular health and strength.\n2. Wheelchair tennis – A great way to improve coordination, agility, and"` |
| `2198 / 1` | `" Hello! How can I help you today?\n"` | `" Hi there! How can I assist you today? 😊"` | `" Hello! How can I help you today?"` | `" Hello! How can I help you today?"` | `" Hello! How can I help you today?"` |
| `2198 / 3` | `" There are many varieties, but some popular types include Espresso, Cappuccino, Latte, and Mocha.\n"` | `" Great question! There are many popular types of coffee, depending on your taste, region, and brewing method. Here are some of the most widely enjoyed:\n\n### 1. **Espresso**\n- The base for many coffee drinks.\n- Strong, concentrated, and rich in flavor.\n- Often used"` | `" Some popular types of coffee include espresso, latte, cappuccino, and Americano."` | `" Some popular types of coffee include espresso, latte, cappuccino, and Americano."` | `" Some popular types of coffee include espresso, latte, cappuccino, macchiato, and Americano. Each has its own unique flavor and preparation method. Would you like to know more about any of these?"` |

## Conclusión (una línea por hipótesis, sin reinterpretar)
- T1: Pasa (`pasa=true`): en test (`n=200`), el campo reduce el z-MSE en turnos del asistente frente a `repeat` (`-0.7232`, IC95 `[-0.7551, -0.6922]`) y frente a `mean` (`-0.6187`, IC95 `[-0.6476, -0.5909]`).
- T2: Pasa (`pasa=true`): en test (`n=200`), la diferencia `campo − mlp` es `-0.0183` con IC95 `[-0.0244, -0.0124]`, cuyo extremo superior (`-0.0124`) es menor o igual que el umbral prerregistrado `0.05·mlp = 0.0193`.
- T3: Pasa (`pasa=true`): en test (`n=200` conversaciones, `759` turnos), condicionar con `adapter(ẑ_campo)` reduce la NLL/token frente a `none` en `-1.2982` (IC95 `[-1.3309, -1.2654]`; PPL baja de `6.9992` a `1.9109`).
- T4: Pasa (`pasa=true`): en test (`n=200`), `adapter(ẑ_campo)` reduce la NLL/token frente al control barajado `shuf` de otra conversación en `-0.2871` (IC95 `[-0.3118, -0.2625]`; brecha `campo − true = +0.0175`, IC95 `[0.0121, 0.0240]`).
- No-daño: Pasa (`identidad_ok=true`, `llm_intacto=true`): diferencia máxima entre `inputs_embeds` e `input_ids` de `0.0` (`< 0.05`) y hash SHA-256 de los pesos del LLM idéntico antes y después del entrenamiento del adaptador y tras la evaluación (`78685cdf6081c3802b007200ddbba61ae24eb29c7ca25416baeacdefef7484ee`).
