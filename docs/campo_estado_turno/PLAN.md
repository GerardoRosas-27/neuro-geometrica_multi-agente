# PLAN (preregistro) — campo de ESTADO por TURNO (`exp/campo-estado-turno`)

> Este documento se congela en un commit propio ANTES de cualquier ejecución que no sea smoke.
> Cualquier cambio posterior se registra como enmienda fechada al final (sección "Enmiendas"), nunca reescribiendo lo anterior.
> El test se evalúa UNA sola vez. Los resultados negativos se informan igual que los positivos.

## 0. Motivación (qué falló antes)
En `exp/gemma-campo-chat` (run 20261104, `docs/gemma_campo_chat/run_20261104/`) el campo sustituía el estado oculto de la capa 13
**token a token**. Para chat fracasó: PPL 362 frente a 5.6 del modelo intacto y el texto degenera. Lectura: obligamos al campo a
aprender *lenguaje* (la dinámica fina token a token), que es justo lo que el LLM ya hace bien.

**Idea del usuario:** el campo NO debe aprender lenguaje. Debe predecir un **ESTADO** (un vector por turno) a partir de parámetros de
entrada (los estados de los turnos previos). El LLM es solo la **periferia lingüística**: codifica turnos a estado y decodifica estado
a texto. Los pesos del LLM no se tocan.

## 1. Diseño
1. **Codificar** (LLM congelado, por defecto `google/gemma-2-2b`; configurable, p. ej. `Qwen/Qwen3-VL-4B-Instruct`/el usado en el run anterior).
   Cada conversación se lee entera una vez; para cada turno se toma el estado `hidden_states[L]` (por defecto `L=13`) y se agrega sobre los
   tokens del turno: `mean` (por defecto) o `last`. → PCA blanqueado → `z_turn ∈ R^d`, `d=64` por defecto.
   El codificador (media, componentes, escala) se ajusta SOLO con turnos de train y se congela; se registra su SHA-256.
2. **Campo** (crate Rust `gemma_campo`, sin cambios: ridge por grado + SGD en GPU wgpu + NLMS rápido, elegidos en val).
   Secuencia por conversación = `z` de los turnos en orden (user, assistant, user, ...). La predicción en la posición del turno
   assistant `k` usa `z` de los `ctx` turnos anteriores, **incluido el turno user actual**. `ctx=4` por defecto.
   Nota: el crate se entrena sobre todas las transiciones de turno (también predice turnos user); solo se EVALÚA en turnos assistant.
   **Baselines/controles** (misma entrada): `repeat` (z del último turno, = turno user actual), `repeat_asst` (z del último turno
   assistant anterior; si no existe, el user), `mean` (media de z de turnos assistant de train), `ridge` lineal (numpy, sin cuadráticos,
   λ elegido en val, entrenado solo en objetivos assistant), `mlp` (torch, 2 capas ocultas de 256, GELU, early stopping en val,
   entrenado solo en objetivos assistant).
3. **Decodificar**: adaptador aprendido `z → P` vectores de soft-prompt (`P=4`), MLP `d→512→P·H`, prefijados a los embeddings de
   entrada. Entrada del LLM: `[soft-prompt(z)] + texto de la conversación previa + "Assistant:"` y se puntúa la respuesta real.
   Se entrena SOLO el adaptador (LLM congelado, `requires_grad=False`, SHA de los pesos comprobado antes y después) con el `z`
   VERDADERO de la respuesta (teacher forcing del estado), con ruido gaussiano `σ=0.1·` (en unidades blanqueadas) por defecto.
   Pérdida: NLL media de los tokens de la respuesta. Selección de época por NLL en val con `z_true`.

## 2. Datos y splits
- Dataset: `HuggingFaceTB/everyday-conversations-llama3.1-2k` (`train_sft`) o JSONL local `{"messages":[...]}`; mismo origen que el run anterior.
- Deduplicación por hash del texto; split POR CONVERSACIÓN `train/val/test = 0.8/0.1/0.1`, semilla `12345`; `n_conv` por defecto 2000.
- Turnos truncados a `max_turn_tokens=128`; contexto de decodificación truncado por la izquierda a `max_ctx_tokens=384`.
- Se comprueba disyunción de splits (ids y hashes). λ, η, γ, época del adaptador y del MLP se eligen en val. Test se evalúa una vez.

## 3. Hipótesis y criterios (todos sobre TEST, unidad estadística = conversación)
Bootstrap pareado por conversación, `B=10000`, semilla `12345`, IC95 por percentiles. Métrica por conversación = media sobre sus
turnos assistant evaluables (posición ≥1).

- **T1 (el campo predice el estado):** z-MSE(campo) − z-MSE(repeat) con IC95 superior < 0 **y** z-MSE(campo) − z-MSE(mean) con
  IC95 superior < 0. (Se informan también `repeat_asst` y `ridge`.)
- **T2 (vs control MLP):** se informa z-MSE(campo) − z-MSE(mlp). Éxito si IC95 superior ≤ +0.05·z-MSE(mlp) (no peor que el MLP por más de un 5 %).
- **T3 (el estado predicho ayuda a decodificar):** NLL/token de la respuesta real con `adapter(ẑ_campo)` − NLL sin condicionamiento
  (mismo texto de contexto, sin soft-prompt), IC95 superior < 0.
- **T4 (el estado lleva información de ESTA conversación):** se informa la brecha NLL(ẑ_campo) − NLL(z_true). Control barajado:
  `z_shuf` = `z_true` de un turno assistant de OTRA conversación de test (permutación sin puntos fijos, semilla 12345);
  NLL(ẑ_campo) − NLL(z_shuf) con IC95 superior < 0.
- **No-daño:** (a) identidad: NLL sin condicionamiento por la ruta `inputs_embeds` == ruta `input_ids` (|Δ| máx < 1e-4 en fp32, < 0.05 en fp16);
  (b) SHA de los pesos del LLM idéntico antes/después de entrenar el adaptador; (c) 5 muestras de generación (greedy, 60 tokens)
  por condición (`none`, `campo`, `true`, `shuf`) en test, copiadas literalmente.

Si T1 falla, T3/T4 se informan igualmente. Ninguna hipótesis se reinterpreta a posteriori.

## 4. Lo que NO se afirma
- Que el campo "entiende" o "aprende lenguaje": por diseño no lo hace.
- Resultados de smoke (modelo diminuto aleatorio): solo validan la tubería, no son evidencia.

## 5. Reproducibilidad
Se guardan: commit, `pip freeze`, SHA-256 del JSONL, `encoder.npz`, `model.bin`, `adapter.pt`, pesos del LLM (hash del state_dict),
logs de entrenamiento y `eval_test.json`. Comandos exactos en `RUNBOOK.md`; plantilla en `RESULTADOS_PLANTILLA.md`.

## Enmiendas
(ninguna)
