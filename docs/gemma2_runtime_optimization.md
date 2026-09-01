# Runtime Gemma 2 optimizado

## Ruta de inferencia

`QuantizedGemma2` carga directamente GGUF mediante Candle 0.11. Los pesos
lineales y el embedding permanecen cuantizados. `QMatMul::embedding` desquantiza
únicamente las filas solicitadas; si el LM head está ligado al embedding, ambos
comparten el mismo `Arc<QTensor>`.

Cada capa global usa `candle_nn::kv_cache::KvCache` con capacidad máxima
preasignada. Las capas de ventana Gemma 2 usan `RotatingKvCache`, por lo que
retienen únicamente la ventana local. Las máscaras causal y local se construyen
una vez por forward y se comparten entre capas equivalentes.

La atención agrupada es híbrida: el prefill materializa K/V para conservar un
matmul batched eficiente, mientras el decode token-a-token procesa cada query
head contra su cabeza KV sin copiar todo el contexto repetido.

## Runtime y sesiones

`native_gemma2_runtime` centraliza:

- prefill y decode;
- sampling reproducible;
- decoder incremental de `tokenizers`;
- métricas TTFT, prefill y tokens/s;
- reutilización KV cuando el prompt nuevo extiende exactamente el prefijo
  procesado;
- callback por token para subsistemas concurrentes;
- `generate_observed_with_logits`, que permite mezclar un sesgo externo
  (CTP) **antes** de cada `sample`, sin un segundo prefill Transformer.

Si el historial retokenizado no coincide con el prefijo almacenado, el runtime
limpia la cache y hace un prefill completo. Nunca reutiliza una cache ambigua.

## Acople fasorial concurrente

`gemma_phasor_coupling` crea un worker Rust propietario de
`NativePhasorThermodynamicEngine`. El hilo Candle sólo envía `(token, position)`
por canal. El worker proyecta cada token de forma determinista a nodos,
inyecta amplitud/fase y ejecuta pasos termodinámicos mientras Gemma calcula el
siguiente token.

Esta separación evita compartir tensores o bloquear el hot path del LLM. El
estado fasorial resultante sirve para memoria, routing o diagnóstico de la
siguiente interacción; **ese worker no modifica logits** durante la generación
actual.
El preset interactivo acumula estímulos sobre 256 nodos y evoluciona cada
16 tokens para limitar la competencia por CPU; el config permite aumentar
resolución y frecuencia en entrenamiento offline.

El sesgo de logits en vigilia lo aplica otra ruta: el híbrido CTP del chat
circadiano (§ siguiente). Los dos mecanismos no se sustituyen.

## Ciclo circadiano (vigilia barata, aprendizaje en sueño)

`native_gemma2_circadian_chat` es el chat unificado. Gemma permanece congelado.
El núcleo CTP y las máscaras de capas se actualizan de noche.

### Vigilia (fase 1 + 3)

1. `plan_wake_prefill` / `prepare_forward`: como máximo un forward. Si el
   prompt extiende `cached_tokens` con la misma máscara, sólo se calcula el
   sufijo. Las máscaras sparse recordadas **no** se aplican mientras la KV
   pueda extenderse.
2. `adopt_prefill` entrega logits y KV al runtime.
3. El híbrido observa embeddings (`W_emb·√d`): cola de `thermo_window` (64)
   si no hay cache; si hay cache, sólo el sufijo nuevo.
4. Un paso CTP + `logits_from_hidden` produce un sesgo de vocabulario.
   Mezcla RMS con `wake_blend = 0.25`:
   `ℓ' = ℓ + α · RMS(ℓ)/RMS(b) · b`.
5. El mismo sesgo se aplica en cada paso de decode vía
   `generate_observed_with_logits`.
6. Tras generar, se observan los tokens nuevos para el siguiente turno.
7. `observe()` escribe en un anillo. **No** flushea rutas de día.

`/limpiar` llama a `reset_context()` (ventana de embeddings), no a `reset()`
(eso borraría `sleep_cycles` y contadores). Restaura historial desde
`thermo.cdt` o, si falta, desde el journal de vigilia. Un checkpoint CTP
corrupto falla en voz alta; no se sustituye en silencio por un sustrato vacío.

El log por turno incluye `prefill=N cache=true|false` y, si hubo mezcla,
`[ctp blend=… phi=… mixed=…]`.

### Sueño (fase 2)

`/sueño`, `--sleep-only` o al salir:

1. Replay de hasta 8 prompts (búfer más reciente, luego journal no entrenado).
2. Forward completo + candidatos sparse. `logit_agreement` mide calidad.
3. ≥ 0.50 y sparse → se quedan en memoria de trabajo para el día siguiente.
4. ≥ 0.92, sparse y spin-gate → rutas lentas verificadas.
5. Basura o máscara de todas las capas → descarte.
6. `train_thermo_from_dataset` + persistir `adaptive/` y `thermo.cdt`.

Umbrales: `min_runtime_quality = 0.50`, `min_verified_quality = 0.92`,
`max_sleep_replays = 8`. El umbral 0.92 sigue siendo estricto: una sesión
puede terminar con `verified_routes = 0` y aun así retener máscaras de
trabajo.

```powershell
cargo run --release --bin native_gemma2_circadian_chat -- --chat NOMBRE
cargo run --release --bin native_gemma2_circadian_chat -- --chat NOMBRE --sleep-only
```

Persistencia: `data/native_gemma2_circadian/NOMBRE/{adaptive,thermo.cdt,wake,sleep}`.

El chat adaptativo (`native_gemma2_adaptive_chat`) comparte fases 1 y 2 pero
**no** aplica sesgo CTP: no carga el híbrido.

### Qué está medido

- Prefill incremental: segundo turno sólo sufijo (prueba GGUF ignorada).
- Sueño: un replay descubre máscaras en GGUF (`sleep_discovers_masks_on_gemma2_gguf`).
- Sesgo: vocabulario finito, `phi_norm > 0`, `mix_logits` puede cambiar top-1
  (`wake_bias_from_embeddings_on_gemma2_gguf`, 26 s).
- `wake_blend` ausente en JSON antiguo → 0.25.

No está medido: calidad de chat con `α > 0` frente a Gemma denso, ni ahorro
real de capas durante un decode incremental.

## Benchmark

```powershell
cargo run --release --bin native_gemma2_benchmark -- \
  --prompt-lengths 32,256,1024,2048 \
  --generated 64 \
  --repetitions 3
```

El CSV informa:

- tiempo de carga;
- RSS antes/después;
- almacenamiento del embedding;
- TTFT;
- throughput de prefill y decode;
- hit de reutilización KV.

### Resultado CPU medido

GGUF local Gemma 2 2B, release, `target-cpu=native`, 30 de julio de
2026:

- carga: 5.94 s;
- RSS incremental tras cargar: 1,716 MiB;
- embedding cuantizado: 461 MiB;
- prefill: 18.65–19.53 tokens/s en prompts de 32 y 256 tokens;
- decode: 10.58–11.13 tokens/s;
- TTFT a 256 tokens: 13.73 s;
- TTFT al extender la misma sesión: 0.98 s, 14.0 veces menor.

La referencia publicada anterior era 2.04–2.68 tokens/s; la nueva ruta midió
entre 3.9 y 5.5 veces esa velocidad. Es una comparación con la evidencia
histórica del proyecto, no un benchmark pareado de commits.

Una prueba de calidad produjo una definición coherente de entropía con
30 tokens, 11.47 tokens/s sin worker fasorial y salida determinista para la misma
semilla.

Con el preset fasorial final (256 nodos, un paso cada 16 tokens), una comparación
inmediata del mismo binario midió 5.94 tokens/s con acople y 5.89 sin acople,
diferencia indistinguible del ruido de la máquina. Los valores absolutos bajaron
durante esa corrida por carga del sistema; se usa el par únicamente para evaluar
el overhead concurrente.

El modelo no se versiona. Las corridas deben registrar hash y cuantización del
GGUF, CPU/GPU, RAM y flags de compilación.

## Backends

El build por defecto es CPU portable dentro de la CPU objetivo configurada.

```powershell
cargo build --release --features mkl
cargo build --release --features cuda
```

CUDA es opt-in porque requiere toolkit y driver compatibles. MKL debe aceptarse
sólo si el benchmark demuestra mejora: gran parte del tiempo se consume en
`QMatMul` cuantizado, donde BLAS puede no ser el cuello dominante.

En la máquina de validación no se detectó `nvidia-smi`. El build MKL compiló,
pero el ejecutable no inició por una DLL de runtime ausente
(`STATUS_DLL_NOT_FOUND`); por ello CPU Candle sigue siendo el backend validado y
MKL/CUDA permanecen opt-in.

## Política de shards

No se generan shards para inferencia CPU de un solo dispositivo. Dividir el
GGUF no reduce FLOPs y puede aumentar fallos de página. Un formato por capas se
justifica únicamente para:

1. asignar rangos de capas a varios dispositivos;
2. cargar selectivamente modelos que no caben en RAM/VRAM;
3. distribuir artefactos grandes.

`native_gguf_paged_thermo` continúa siendo un almacén de aristas
termodinámicas; sus shards F32 no deben usarse como pesos del Transformer.

## Límites pendientes

- Los kernels QKV, GeGLU, RMSNorm/residual y softcap/softmax no están fusionados.
  Se implementarán sólo después de perfilar el runtime optimizado.
- El batching continuo y paged attention multiusuario requieren soporte
  adicional en Candle.
- La vigilia ya está limitada a un prefill. El sueño rejuega hasta 8 prompts
  para descubrir máscaras; aplicar esas máscaras en decode incremental
  rompería la KV cache y no está habilitado.
- El sesgo CTP cambia logits; no hay todavía ablación de calidad lingüística
  ni de `wake_blend`.

## V8 — rutas de capas vs 26/26 vs Ollama (31 ago 2026)

Protocolo T0: 3 prompts (`BENCH_PROMPTS`), temperature 0,01, seed
`0x4745_4D4D_4132`, device CPU, `repetitions=2` (mediana de tok/s).
Máquina: Windows CPU. Gemma 2 2B GGUF local + Ollama `gemma2:2b` en
`127.0.0.1:11434`.

La cifra de 8 tokens (2,50 tok/s denso / 2,72 sparse / 14,66 Ollama) queda
**retirada**. Ese N infla el denominador con el arranque del bucle; no se cita
como figura publicada.

Comando:

```powershell
cargo test --release --lib layer_route_benchmark -- --nocapture --test-threads=1
cargo run --release --bin native_gemma2_circadian_chat -- --bench-routes --max-tokens 64
```

`decode_tok_s` incluye sample + decode UTF-8. `model_decode_tok_s` =
generated / `model_decode_seconds` en nativo. En la fila `ollama`,
`model_decode_tok_s` = `decode_tok_s` (eval tok/s de Ollama ya es decode del
modelo; no hay split sample/texto comparable).

### CSV canónico (`--bench-routes --max-tokens 64`, Windows CPU, 31 ago 2026 ~20:52)

```
backend,prompt_id,executed_layers,layer_count,kl_vs_dense,decode_tok_s,model_decode_tok_s,ttft_s,lrc_hit,fallback,generated_tokens
native_dense,0,26,26,0.000000,6.6687,7.3309,1.7513,0,0,24
native_sparse,0,23,26,0.591629,6.9507,7.6597,1.5352,0,1,64
ollama,0,26,26,0.000000,13.2591,13.2591,0.2025,0,0,28
native_dense,1,26,26,0.000000,6.5005,7.1248,2.1932,0,0,27
native_sparse,1,23,26,0.471780,7.2410,7.9959,1.9209,0,1,64
ollama,1,26,26,0.000000,12.9074,12.9074,0.1779,0,0,35
native_dense,2,26,26,0.000000,6.6097,7.3241,1.9067,0,0,14
native_sparse,2,23,26,1.082224,7.1696,7.9004,1.7108,0,1,64
ollama,2,26,26,0.000000,12.2754,12.2754,0.1926,0,0,45
```

| backend | capas | KL vs denso | decode tok/s | model_decode tok/s | LRC hit | fallback |
|---|---:|---:|---:|---:|---:|---:|
| native_dense | 26/26 | — | 6,59 | 7,26 | 0 | 0 |
| native_sparse | 23/26 | 0,72 | 7,12 | 7,85 | 0 | 1 |
| ollama `gemma2:2b` | 26/26 | — | 12,81 | 12,81 | — | — |

N real por fila (EOS puede cortar antes del pedido): denso 24/27/14, sparse
64/64/64, Ollama 28/35/45. Pedir 64 no alarga el denso si el modelo emite EOS.
No se declara victoria frente a Ollama.

Sparse sigue ~8 % más rápido que denso nativo, pero KL media 0,72 > 0,15: el
producto **no promociona** la ruta (fallback 100 %, hit LRC 0). El test lib
con N pedido = 32 en la misma máquina también pasó (tok/s denso 5,88 / sparse
6,38 / Ollama 11,69). Las cifras no se citan como ventaja del preprint.

### T2.1 — Ablation KL por capa (31 ago 2026 ~21:04 CT)

Protocolo: 1 prompt (`BENCH_PROMPTS[0]`, «Explica en una frase qué es un residual.»),
prefill denso con traza, luego 24 prefills sparse que apagan **una sola** capa
`i ∈ 1..24` (no 0 ni la última). Gemma 2 2B GGUF, Windows CPU. Test
`layer_kl_ablation_measures_one_layer_skips_on_gemma2_gguf` en 63 s de
wall-clock (más ~4 min de compile release). CSV canónico:
`docs/v8_layer_kl_ablation.csv`. Filas ordenadas por KL ascendente = cola de
skip calibrada. `conservative_candidate_mask` **no** se tocó.

```
layer,sliding,delta_rms,kl,top1
7,0,1.716770,0.018273,1
21,0,3.012524,0.021460,1
8,1,1.290439,0.023733,1
12,1,1.642857,0.031118,1
20,1,2.632482,0.035451,1
15,0,2.120014,0.036453,1
9,0,1.674762,0.043925,1
11,0,1.622980,0.044089,1
6,1,1.543405,0.045520,1
23,0,4.160556,0.049675,1
14,1,1.671279,0.051977,1
19,0,2.281575,0.063634,1
10,1,1.548858,0.064504,1
18,1,2.098229,0.070332,1
16,1,2.048503,0.079036,1
1,0,1.509647,0.082673,1
13,0,1.758550,0.087647,1
22,1,3.393774,0.091118,1
24,1,7.130063,0.145864,1
17,0,2.247718,0.161124,1
3,0,1.458264,0.228592,0
5,0,1.383746,0.241503,0
4,1,1.731286,0.261560,0
2,1,1.471900,0.479236,0
```

| umbral | capas |
|---|---|
| KL ≤ 0,05 | 7, 21, 8, 12, 20, 15, 9, 11, 6, 23 |
| 0,05 < KL ≤ 0,15 | 14, 19, 10, 18, 16, 1, 13, 22, 24 |
| KL > 0,15 | 17, 3, 5, 4, 2 |

**Veredicto Camino S:** no se aparca. Diez capas caben en KL ≤ 0,05 al
apagarlas **solas**; diecinueve en KL ≤ 0,15. Las caras (2, 4, 5, 3, 17)
rompen top-1 salvo la 17. El skip actual de 3 capas a la vez (KL media 0,72
en V8) no implica que una capa suelta sea cara: `delta_rms` no rankea como
KL (p. ej. capa 23, `delta_rms` 4,16 y KL 0,050; capa 5, `delta_rms` 1,38 y
KL 0,242). T2.2 puede usar esta cola. Este PR no cambia la máscara.

### T1.1 — Perfil del decode token a token (31 ago 2026 ~21:21 CT)

Camino K. Solo medición: no se reescriben kernels, máscaras, skip ni CTP.
Windows CPU, Gemma 2 2B GGUF, release, `target-cpu=native`. Comando:

```powershell
$env:GEMMA2_PROFILE="1"
$env:GEMMA2_BENCH_PROMPT_COUNT="1"
$env:GEMMA2_BENCH_REPS="1"
cargo run --release --bin native_gemma2_circadian_chat -- --bench-routes --max-tokens 64
```

Un prompt (`BENCH_PROMPTS[0]`, «Explica en una frase qué es un residual.»),
1 repetición. El bench **no** mezcla CTP (`on_logits` es no-op). Los
contadores `Tensor::new` / `hidden.clone` / `QMatMul::forward` viven detrás
de `GEMMA2_PROFILE=1` (apagados por defecto; el chat release no los paga).
`input_alloc_s` cronometra `Tensor::new(&[token])` + `unsqueeze` por paso.

CSV de esa corrida:

```
backend,prompt_id,executed_layers,layer_count,kl_vs_dense,decode_tok_s,model_decode_tok_s,ttft_s,lrc_hit,fallback,generated_tokens,model_frac,logits_s,text_s,input_alloc_s,tensor_new,hidden_clone,qmatmul_fwd,seq1_fwds,last_seq1_tensor_new,last_seq1_hidden_clone,last_seq1_qmatmul
native_dense,0,26,26,0.000000,6.3957,7.0113,1.7549,0,0,24,0.9122,0.328837,0.000458,0.000124,24,624,4392,24,0,26,183
native_sparse,0,23,26,0.591629,6.9995,7.7170,1.8187,0,1,64,0.9070,0.848551,0.001000,0.000313,64,1472,10368,64,0,23,162
ollama,0,26,26,0.000000,13.0065,13.0065,0.5337,0,0,28,1.0000,0.000000,0.000000,0.000000,0,0,0,0,0,0,0
```

Fracciones del decode nativo (denso, 24 tokens por EOS; sparse, 64):

| backend | modelo / decode | sampler (logits, sin CTP) | UTF-8 | Tensor::new+unsqueeze |
|---|---:|---:|---:|---:|
| native_dense | 91,22 % | 0,328837 s (~8,8 %) | 0,000458 s (~0,01 %) | 0,000124 s (~0,003 %) |
| native_sparse | 90,70 % | 0,848551 s (~9,3 %) | 0,001000 s (~0,01 %) | 0,000313 s (~0,003 %) |

Dentro de **un** `forward` seq=1 (columna `last_seq1_*`):

| backend | Tensor::new | hidden.clone | QMatMul::forward |
|---|---:|---:|---:|
| dense 26/26 | 0 | 26 | 183 |
| sparse 23/26 | 0 | 23 | 162 |

`Tensor::new` en el decode está **fuera** de `forward` (el `Tensor::new(&[token])`
del bucle). Por eso `last_seq1_tensor_new=0` y el total es 1 por token generado
(24 / 64). `QMatMul::forward` por paso = capas×(4 attn + 3 MLP) + 1 lm_head
(26×7+1=183; 23×7+1=162). `hidden.clone` = una vez por capa ejecutada.

**H1 verdadera:** `model_decode` ≥ 85 % del decode (91,22 % denso, 90,70 %
sparse). El hueco vs Ollama (13,01 tok/s eval) es de kernels, no de sampler
ni UTF-8.

**H2 falsa:** `Tensor::new(&[token])` + `unsqueeze` por paso no es visible
(>5 %). Son ~0,003 % del decode (0,000124 s denso / 0,000313 s sparse).

**H3 no se midió aquí.** El 8,8–9,3 % de logits es `LogitsProcessor::sample`
del bench, sin CTP. No se «arregla» CTP para ganar a Ollama.

Este PR no cambia máscaras, skip, early-exit, cuenca, MKL ni llama.cpp.
