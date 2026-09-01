# Plan: nativo Rust más rápido que Ollama (o honestamente no)

**Rama:** `feature/optimizacion-velocidad-rutas`
**Fecha:** 31 de agosto de 2026
**Para:** bot autónomo. Trabajar **una tarea a la vez**. No mezclar cuenca/CL-MPM.
**No es código.** Este archivo es el mapa. Cada tarea dice archivos, comando y
criterio de hecho (puede fallar).

El preprint de cuenca no reclama este resultado (opción C). Esto es ingeniería
de producto del chat nativo.

---

## 0. Meta y no-metas

**Meta A (calidad de skip):** una máscara o un early-exit con
`KL(denso ‖ barato) ≤ 0,15` en el set V8, y hit LRC > 0.

**Meta B (velocidad de kernels):** el decode nativo denso 26/26, **64 tokens**,
queda a ≤ 1,3× de Ollama `gemma2:2b` en la misma máquina, mismos prompts.

**Meta C (ganar de verdad):** nativo (el camino que el producto usaría:
sparse si KL ok, si no denso) **tok/s > Ollama** en ese protocolo.

El bot **no** declara victoria con 8 tokens generados. El V8 actual usó 8
tokens: el arranque del bucle infla el denominador. Protocolo canónico abajo.

**No-metas:**

- No entrenar pesos GGUF / LoRA / adapters.
- No quinto binario. Extender `layer_route_benchmark.rs` y
  `--bench-routes`.
- No `exit_after` hasta tener la tabla de KL por capa (T2.1).
- No subir `max_skip_fraction` a 0,25 si KL mediana > 0,15.
- No tocar `consolidation_basin_experiment`.
- No citar tok/s de 8 tokens como cifra publicada.

---

## 1. Hechos ya medidos (no repetir el diagnóstico)

Corrida V8, Gemma 2 2B GGUF local, CPU, 2 prompts, **8 tokens**:

| backend | capas | KL | tok/s | LRC hit | fallback |
|---|---:|---:|---:|---:|---:|
| nativo 26/26 | 26 | — | 2,50 | 0 | 0 |
| nativo sparse | 23/26 | **0,53** | 2,72 | 0 | **1** |
| Ollama `gemma2:2b` | 26 | — | **14,66** | — | — |

Implicaciones, en este orden:

1. **El cuello no es saltar 3 capas.** 23 vs 26 da +9 % (2,50 → 2,72).
   Ollama está ~5–6× por encima. Eso es kernels / runtime, no máscaras.
2. **El skip actual es inutilizable.** KL 0,53 ≫ 0,15 → el LRC no promociona
   (`promote` rechaza). Hit 0, fallback 100 %. El producto sigue en 26/26.
3. **Hay dos problemas independientes.** Resolver (2) no cierra el gap con
   Ollama. Resolver (1) no hace útil el skip. El bot debe tratarlos como
   **caminos paralelos**, no como un solo PR.

Cifra histórica (64 tokens, denso, sin skip): ~10,6–11,1 tok/s nativo. Si se
reproduce, el hueco vs Ollama ~14,7 baja a ~1,3–1,4×, no a 6×. **Primera
tarea: repetir V8 con 64 tokens** antes de optimizar nada.

---

## 2. Protocolo canónico (todas las tareas lo usan)

Misma máquina, mismo GGUF (`resolve_gemma2_model_path`), mismo Ollama
`gemma2:2b` si el daemon responde.

```text
prompts     = BENCH_PROMPTS (3) en src/layer_route_benchmark.rs
generated   = 64
temperature = 0.01
seed        = 0x4745_4D4D_4132
device      = cpu
repetitions = 2 (mediana de tok/s)
```

Métricas obligatorias en el CSV (ya existen las columnas):

- `executed_layers`, `layer_count`
- `kl_vs_dense` (última posición del **prompt**, no de la generación)
- `decode_tok_s` y `model_decode_seconds` si se añade la columna
- `lrc_hit`, `fallback`
- fila `ollama` si `127.0.0.1:11434` responde

Comando:

```powershell
cargo test --release --lib native_sparse_vs_dense_and_ollama -- --nocapture --test-threads=1
cargo run --release --bin native_gemma2_circadian_chat -- --bench-routes --max-tokens 64
```

**Regla:** cada PR que pretenda “más rápido” pega el CSV en el mensaje de
commit o en `docs/gemma2_runtime_optimization.md` § V8. Sin CSV no hay
claim.

---

## 3. Tres caminos (elegir por evidencia, no por gusto)

```text
T0 medición 64 tokens
        │
        ├─► Camino K  kernels / runtime decode     ── cierra el 5× vs Ollama
        ├─► Camino S  skip/exit con KL ≤ 0,15      ── hace útil LRC
        └─► Camino E  especulativo (borrador barato + 1 verify)
```

Después de T0:

| Si 64 tokens muestran… | El bot hace |
|---|---|
| nativo denso ≥ 0,75× Ollama | K es ajuste fino; priorizar S |
| nativo denso < 0,5× Ollama | **K primero.** S no va a ganar |
| KL de 1 capa skip ≤ 0,05 | S es viable; ablation por capa |
| KL de 1 capa skip > 0,15 | skip por `delta_rms` está mal; S cambia de métrica o se aparca |
| early-exit k=20 KL ≤ 0,15 y tok/s +20 % | reabrir V7 |

Camino E solo si K o S ya dieron un decode barato (early-exit o sparse
calibrado). No especular sobre 26 capas densas: no hay borrador.

---

## 4. Tareas

Notación: `T{camino}.{n}`. Independientes si no listan `Blocked-by`.
Un PR = una tarea. Tests verdes del módulo tocado + CSV si hay GGUF.

---

### T0 — Medición honesta a 64 tokens

**Estado (31 ago 2026):** hecho. CSV canónico de `--bench-routes --max-tokens 64` (Windows CPU) en `docs/gemma2_runtime_optimization.md` § V8. La cifra de 8 tokens queda retirada.

**Por qué:** el 2,5 tok/s de 8 tokens no es comparable a Ollama ni a la
cifra histórica de 11 tok/s.

**Archivos:** `src/layer_route_benchmark.rs` (default `generated_tokens=64`,
`prompt_count=3`). Test GGUF puede usar 32 si 64 se pasa de 10 min; el
`--bench-routes` usa 64.

**Pasos:**

1. Default de `RouteSpeedConfig.generated_tokens` → 64.
2. Añadir a `RouteSpeedRow` (opcional pero útil):
   `model_decode_tok_s = generated / model_decode_seconds`.
   Hoy `decode_tok_s` incluye sample + decode de texto.
3. Correr el test con `--test-threads=1` y `--bench-routes`.
4. Pegar CSV en `docs/gemma2_runtime_optimization.md` sustituyendo la
   tabla de 8 tokens, con nota de N y máquina.

**Gate:**

```powershell
cargo test --release --lib layer_route_benchmark -- --nocapture --test-threads=1
```

**Pasa:** hay una fila nativo_dense, nativo_sparse, y ollama si el daemon
está; `generated_tokens ≥ 32`; `model_decode_tok_s` se imprime.

**No hacer:** no cambiar máscaras ni kernels en este PR.

---

### T1.1 — Perfil del decode token a token (Camino K)

**Blocked-by:** T0 (para no optimizar ruido).

**Por qué:** Ollama (llama.cpp) gana en el matmul cuantizado y en no
asignar tensores por token. Hay que ver **dónde** se va el tiempo nativo.

**Archivos:** `src/native_gemma2.rs` (`forward_with_mask`, `attention` seq=1),
`src/native_gemma2_runtime.rs` (el bucle `for _ in 0..max_tokens`).

**Pasos:**

1. En una corrida de 64 tokens, loguear fracciones:
   `model_decode_seconds / decode_seconds`,
   `logits_processing_seconds`, `text_decode_seconds`.
2. Dentro de un `forward` seq=1, contar (debug, no en caliente):
   llamadas a `Tensor::new`, clones de `hidden`, `QMatMul::forward`.
3. Anotar en el doc: % del decode que es modelo vs sampler vs UTF-8.

**Hipótesis a confirmar o matar:**

- H1: `model_decode` ≥ 85 % del decode → kernels.
- H2: `Tensor::new(&[token])` + `unsqueeze` por paso es visible (>5 %).
- H3: `on_logits` / CTP en el chat (no en el bench) come tiempo. El bench
  no mezcla CTP; no “arreglar” CTP para ganar a Ollama.

**Gate:** un párrafo con porcentajes en el doc V8, obtenido de una corrida,
no inventado.

**Pasa:** H1 o H2 queda verdadero o falso con números.

---

### T1.2 — Quitar alocaciones del hot path seq=1

**Blocked-by:** T1.1 si H2 es verdadero. Si H2 es falso, **saltar**.

**Archivos:** `src/native_gemma2.rs`, `src/native_gemma2_runtime.rs`.

**Pasos:**

1. Reutilizar un buffer `Tensor` de shape `[1,1]` para el token siguiente
   (escribir el u32 in-place o `from_vec` en storage ya reservado).
2. No clonar `hidden` para `layer_input` cuando `capture_trace=false`
   (el bench y el chat de vigilia van con `false`).
3. No construir máscaras de atención en seq=1 (ya retorna `None`; no
   tocar).
4. Evitar `to_vec1` en el camino de generate (el sampler de Candle toma
   `&Tensor`; no copiar logits a CPU Vec salvo para KL).

**Gate:** T0 otra vez, 64 tokens. CSV denso vs denso anterior.

**Pasa:** `model_decode_tok_s` denso sube ≥ 10 % **o** se demuestra con
contador que las alocaciones ya no aparecen. Si no sube, revertir el PR.

---

### T1.3 — Features de Candle: threads, MKL, backend

**Blocked-by:** T1.1 (H1 verdadero).

**Archivos:** `Cargo.toml`, `.cargo/config.toml`, `docs/gemma2_runtime_optimization.md`.
El doc ya dice: MKL compiló y el exe no arrancó por DLL; CUDA opt-in; CPU
Candle es el backend validado.

**Pasos, en este orden, cada uno un mini-PR si cambia números:**

1. `RAYON_NUM_THREADS` / `candle` intra-op: medir 1, 2, 4, 8 hilos en el
   mismo bench. Pegar tabla. Dejar el default que gane en decode seq=1
   (a menudo **1 hilo** gana en batch=1; no asumir más hilos = más rápido).
2. Documentar `cargo run --release --features mkl --bin native_gemma2_circadian_chat -- --bench-routes`
   solo si el DLL carga. Si no carga, `wontfix` con el error, no insistir.
3. No introducir `llama-cpp-sys` en este paso.

**Gate:** CSV 64 tokens, denso, 1 vs N hilos.

**Pasa:** se elige un default de threads versionado en config o env
documentado. tok/s denso no baja.

---

### T1.4 — Fusión RMSNorm + residual (solo si el perfil lo pide)

**Blocked-by:** T1.1.

El doc de runtime ya lista: QKV, GeGLU, RMSNorm/residual, softcap/softmax
**no fusionados**. Fusionar a ciegas es un pozo.

**Pasos:**

1. Si T1.1 dice que el matmul Q (`QMatMul`) es ≥ 70 % del forward, **no
   fusionar RMSNorm**. El gap vs llama.cpp está en el kernel cuantizado,
   no en el add.
2. Si RMSNorm+add es ≥ 15 %, escribir un `rms_norm_add_inplace` sobre
   `&mut [f32]` del residual para seq=1 (d_model=2304 en 2B) y usarlo
   **solo** en decode seq=1.
3. Tests de bit-exactitud vs `layer.attention_norm.forward` + add, rtol
   1e-4, un vector sintético. Sin GGUF.

**Pasa:** test sintético verde y, si se cablea, CSV con ≥ 5 % tok/s o
revert.

**No hacer:** reescribir `QMatMul` de Candle. Eso es parche a la
dependencia. Si el matmul es el cuello, ir a T1.5.

---

### T1.5 — Backend llama.cpp opcional (escape hatch)

**Blocked-by:** T1.2 y T1.3 hechos, y nativo denso sigue < 0,5× Ollama.

**Por qué:** Ollama *es* llama.cpp. Competir con Candle QMatMul en CPU
puede ser un techo. Un backend `llama` detrás del mismo
`Gemma2Session` permitiría ganar tok/s **sin** fingir que Candle es
llama.cpp.

**Archivos nuevos:** `src/native_llama_backend.rs` (feature `llama`).
No es quinto binario. El chat elige backend por env `GEMMA2_BACKEND=candle|llama`.

**Pasos:**

1. Feature Cargo `llama` opcional (`llama-cpp-2` o equivalente que lea el
   **mismo** GGUF).
2. Trait mínimo: `prefill(&[u32]) -> logits`, `decode(token) -> logits`,
   KV propia del backend.
3. `Gemma2Session` no cambia su API pública; un enum interno.
4. Bench V8 gana una fila `native_llama`.
5. Máscaras de capas (LRC) en v1 del backend llama: **no**. Gana tok/s
   denso primero. Skip es Camino S sobre Candle.

**Gate:** `--features llama` + bench. Si no compila en Windows, documentar
y no forzar.

**Pasa:** `native_llama` tok/s ≥ 0,85× Ollama (mismo GGUF, 64 tokens).
Si no, el feature queda detrás de cfg y no se vende.

**No hacer:** llamar al HTTP de Ollama desde el chat como “runtime nativo”.
Eso no es nativo.

---

### T2.1 — Ablation KL por capa (Camino S)

**Estado (31 ago 2026):** hecho. CSV en `docs/v8_layer_kl_ablation.csv` y § T2.1
de `docs/gemma2_runtime_optimization.md`. Diez capas con KL ≤ 0,05 al apagarlas
solas; cinco con KL > 0,15. Camino S middle-skip **no** se aparca. La máscara
`conservative_candidate_mask` no se cambió.

**Blocked-by:** nada. Puede ir en paralelo a T0/T1.

**Por qué:** el ranking por `delta_rms` + “preferir locales” saltó 3 capas
con KL 0,53. No sabemos **cuáles** ni si **una sola** capa ya rompe KL.

**Archivos:** `src/layer_route_benchmark.rs` función
`run_layer_kl_ablation(model, prompt_tokens) -> Vec<(layer, kl, top1)>`.

**Pasos:**

1. Prefill denso con traza.
2. Para cada capa `i ∈ 1..24` (no 0 ni última): máscara que solo apaga `i`.
   Prefill sparse. `logits_kl` + `top1_agree`.
3. CSV: `layer,sliding,delta_rms,kl,top1`.
4. Ordenar por KL ascendente. Esa es la cola de skip **calibrada**.

**Gate:** test GGUF (omitir si no hay modelo) con 1 prompt, 26 prefills
sparse (caro pero es one-shot). Guardar CSV en
`docs/v8_layer_kl_ablation.csv` o pegarlo en el doc.

**Pasa:** existe una lista de capas con KL ≤ 0,05 y otra con KL > 0,15.
Si **ninguna** capa tiene KL ≤ 0,15, el Camino S de middle-skip se
**aparca** y se va a T2.3 (early-exit) o se declara no viable.

**No hacer:** no cambiar `conservative_candidate_mask` en este PR. Solo
medir.

---

### T2.2 — Máscara greedy por presupuesto de KL, no por `delta_rms`

**Blocked-by:** T2.1.

**Archivos:** `src/adaptive_gemma2.rs` (`conservative_candidate_mask` o
función nueva `kl_budget_mask`).

**Pasos:**

1. Nueva función: apaga capas en orden de KL incremental (de T2.1, o
   estimado) mientras `KL_acumulada ≤ 0,15` y `executed ≥ 8` y no hay dos
   globales consecutivos.
2. Si T2.1 dice que 0 capas caben en el presupuesto, la función devuelve
   `LayerExecutionMask::all` y el test lo aserta.
3. El LRC promociona **esta** máscara, no la de `delta_rms`.
4. No subir `max_skip_fraction` a ciegas.

**Gate:** V8. `mean_kl ≤ 0,15` **o** `fallback_rate = 1` y máscara = 26/26
(honesto). Nunca KL 0,53 con `fallback=0`.

**Pasa:** o hay skip con KL ok, o el sistema se niega a saltar. Ambos son
éxito de ingeniería. Lo que no es éxito: skip con KL alto.

---

### T2.3 — Early-exit (reabrir V7) con tabla k

**Blocked-by:** T2.1. Si middle-skip no deja capas baratas, early-exit
puede: las últimas capas son las que se omiten, el residual ya “decidió”.

**Archivos:** `src/native_gemma2.rs` (`forward_with_mask` +
`exit_after: Option<usize>`), `src/layer_route_cache.rs` (`SkipKind::EarlyExit`).

**Pasos:**

1. Si `exit_after = Some(k)`, tras la capa `k` aplicar
   `norm + lm_head` al hidden actual (el head está ligado al embedding,
   misma `d_model`). No ejecutar `k+1…25`.
2. KV de capas `> k` no se toca.
3. Máscara fija por generación: `exit_after` es parte de la ruta LRC.
4. Tabla k ∈ {12, 16, 20, 23} × 1 prompt: KL, tok/s, capas.
5. Elegir el menor k con KL ≤ 0,15. Si ninguno, `wontfix` V7.

**Gate:** test de unidad sin GGUF (k fuera de rango, k=last ≡ denso).
Test GGUF de la tabla.

**Pasa:** un k con KL ≤ 0,15 y tok/s ≥ +15 % vs denso **o** tabla que
demuestra que no existe.

**No hacer:** mezclar middle-skip y early-exit en el mismo turno.

---

### T2.4 — Calibración offline de rutas (sueño de verdad)

**Blocked-by:** T2.2 o T2.3 con al menos una ruta KL-ok.

**Archivos:** `replay_prompt_for_mask` en `adaptive_gemma2.rs`.

**Pasos:**

1. El sueño usa la máscara calibrada (T2.2/T2.3), no `progressive_candidate_masks`
   por `delta_rms`.
2. Promueve a LRC solo si KL ≤ 0,15 (ya es la regla).
3. Un fixture: 2 prompts parecidos; el segundo debe `lrc_hit=1` en V8.

**Gate:** V8 con `lrc_hit_rate ≥ 0,3` en los 3 prompts del set (el primero
puede ser miss; 2 y 3 hit si la huella solapa).

**Pasa:** hit > 0 y fallback < 1. Eso es el concepto LRC **por fin** vivo.

---

### T3.1 — Decode especulativo (Camino E)

**Blocked-by:** T2.3 con un k usable **o** T2.2 con ≥ 4 capas apagadas.

**Idea:** el borrador corre early-exit/sparse; cada 4 tokens un paso denso
verifica el último. Si coincide, se ahorran 3 forwards densos. Si no, se
rollbackea 1 token y se sigue denso.

**Archivos:** `src/native_gemma2_runtime.rs` (nuevo
`generate_speculative`), no un binario.

**Pasos:**

1. API: `draft_mask` + `verify_every: usize`.
2. Métrica: tok/s de pared, tasa de aceptación del borrador, KL no aplica
   igual (el denso verifica).
3. Si aceptación < 50 %, el especulativo **pierde** (paga draft+verify).
   Gate de abandono: aceptación ≥ 70 % y tok/s ≥ denso + 15 %.

**No hacer:** si T2 no produjo un borrador barato. Especular 26 vs 26 es
peor.

---

### T4 — Cablear el camino ganador en el chat

**Blocked-by:** al menos un camino con CSV que mejore tok/s **usable**
(KL ok o backend llama).

**Archivos:** `native_gemma2_circadian_chat.rs`, `agent_graph.rs` (FastTalker
usa la máscara calibrada; si LRC miss → denso, no sparse sucio).

**Pasos:**

1. FastTalker solo si `lookup_confident` y la ruta tiene `mean_kl ≤ 0,15`.
2. Banner: `layers=a/b backend=candle|llama tok/s=`.
3. Un turno de calibración 1-de-32 denso (ya diseñado, no duplicar).

**Pasa:** chat en frío = 26/26 o llama denso; tras un `/sueño` con KL ok,
el turno siguiente puede ser FastTalker. Nunca FastTalker con KL 0,53.

---

## 5. Orden recomendado para el bot (cola)

Ejecutar en este orden. Parar si el gate falla y no forzar la siguiente.

```text
T0          medición 64 tokens
T2.1        ablation KL por capa          } paralelo a T0 si hay GGUF
T1.1        perfil decode
     │
     ├─ si H2: T1.2 alocaciones
     ├─ si H1: T1.3 threads / MKL
     └─ si matmul ≥ 70 % y sigue < 0,5× Ollama: T1.5 llama.cpp
T2.2        máscara por presupuesto KL    (si T2.1 dejó capas)
T2.3        early-exit                    (si T2.1/T2.2 no bastan)
T2.4        LRC hit real
T3.1        especulativo                  (solo con borrador barato)
T4          chat
```

**DoD de la rama (definition of done):**

1. CSV 64 tokens en el doc V8.
2. O bien nativo (camino producto) tok/s > Ollama, o bien un párrafo
   explícito: *Candle CPU no alcanza llama.cpp; el chat usa backend X /
   se documenta el techo*.
3. Ninguna ruta LRC promocionada con KL > 0,15.
4. `cargo test --release --lib layer_route_benchmark -- --test-threads=1`
   verde.

---

## 6. Desglose por tipo de tarea (para asignar)

| ID | Tipo | GGUF | Riesgo | Impacto esperado vs Ollama |
|---|---|---|---|---|
| T0 | medición | sí | bajo | 0; limpia el 6× vs 1,3× |
| T1.1 | perfil | sí | bajo | 0; decide K |
| T1.2 | runtime | sí para gate | bajo | +10–30 % nativo si H2 |
| T1.3 | config | sí | bajo | +0–2× (threads mal usados restan) |
| T1.4 | kernel | sintético + GGUF | medio | poco si el cuello es QMatMul |
| T1.5 | backend nuevo | sí | alto | puede igualar Ollama |
| T2.1 | medición | sí | bajo | decide si S vive |
| T2.2 | algoritmo | sí | medio | 0 vs Ollama; habilita LRC |
| T2.3 | algoritmo | sí | medio | +15–40 % nativo si k bajo |
| T2.4 | integración | sí | bajo | hit LRC |
| T3.1 | algoritmo | sí | alto | +si aceptación alta |
| T4 | cableado | no obligatorio | bajo | producto |

---

## 7. Criterios de aborto (el bot se detiene y escribe)

Escribir `docs/v8_aborto.md` de 20 líneas y no abrir más PRs de velocidad si:

1. T0: nativo 64 tokens ≥ 0,9× Ollama **y** el usuario solo quería “ganar”.
   Entonces el trabajo ya está; no optimizar más.
2. T2.1: **cero** capas con KL ≤ 0,15. Middle-skip muerto. Solo T2.3 o
   abandonar S.
3. T2.3: ningún k con KL ≤ 0,15. Early-exit muerto.
4. T1.5: `llama` no compila en Windows. Documentar. No pelear con bindgen
   tres PRs.
5. Cualquier PR que suba tok/s y **suba** KL mediana por encima de 0,15
   sin marcar `fallback=1`. Revert.

---

## 8. Archivos que el bot puede tocar

| Puede | No puede |
|---|---|
| `src/native_gemma2.rs` | `src/consolidation_basin_experiment.rs` |
| `src/native_gemma2_runtime.rs` | `src/bin/archive/**` salvo leer |
| `src/layer_route_benchmark.rs` | nuevo `src/bin/native_*` |
| `src/layer_route_cache.rs` | trainers infinitos |
| `src/adaptive_gemma2.rs` | papers de cuenca como resultado |
| `src/bin/native_gemma2_circadian_chat.rs` | |
| `src/agent_graph.rs` (solo T4) | |
| `Cargo.toml` features opcionales | |
| `docs/gemma2_runtime_optimization.md` | |
| `docs/plan_nativo_gana_ollama.md` (este) | |

---

## 9. Prompt corto para arrancar al bot

> Trabaja **solo T0** de `docs/plan_nativo_gana_ollama.md`. No kernels, no
> early-exit, no quinto binario. Cambia el default de generated_tokens a 64,
> corre `layer_route_benchmark` con `--test-threads=1`, pega el CSV en
> `docs/gemma2_runtime_optimization.md`. Si Ollama no responde, déjalo
> `n/a`. Commit en `feature/optimizacion-velocidad-rutas`.

Cuando T0 esté mergeado en la rama, el siguiente mensaje es el mismo
cambiando `T0` por `T2.1` o `T1.1` según la tabla del §3.
