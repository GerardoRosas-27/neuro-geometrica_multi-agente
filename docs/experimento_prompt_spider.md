# Experimento: Prompt Spider — «cada palabra, un fork»

> **Estado:** experimental, en `main`. Pestaña **Prompt Spider** de la app web
> (`agentic_web`). Definición de cada métrica/panel de la UI, pipeline y
> esquema del export: [`prompt_spider_metricas.md`](prompt_spider_metricas.md).

## 1. Origen

Recreación de un vídeo de TikTok (`PROMPT SPIDER // EVERY WORD, ONE FORK`). La
idea: en lugar de apuntar un crawler a la web, se rastrea **el propio prompt**.
Cada palabra se trata como un **punto de decisión**, no como texto plano, y se
evalúa con un conjunto fijo de preguntas:

| Pregunta (UI) | Clave | Significado |
|---|---|---|
| ¿relevante para la tarea? | `relevant` | ¿la palabra importa para lo que pide el prompt? |
| ¿afirmación con fuente? | `grounded` | ¿lo que afirma está respaldado dentro del prompt? |
| ¿es específico? | `ambiguous` | ¿es ambigua / deíctica / vaga? |
| ¿requiere aprobación? | `approval` | ¿compromete algo hacia fuera (publicar, enviar, pagar…)? |

Cada pregunta produce una probabilidad y el enrutado es **deliberadamente
determinista** (no tiene opinión propia):

| Original (TikTok) | Esta app |
|---|---|
| `p ≥ 0.95` → código / ejecución, sin LLM (ramas `if`) | `p ≥ umbral` (0.95 por defecto, slider 0.80–0.99) → **code** |
| `p < 0.95` → Claude Sonnet 5.5, único modelo que razona | `p < umbral` → **LLM activo de la app** (Gemma local o API OpenAI-compatible elegida en *Modelos / API*), etiquetado con su nombre real |
| lo que parece requerir aprobación → al humano | `P(aprobación) ≥ 0.5` → **tú** (cola de aprobación; Aprobar / Rechazar) |
| `ask` con confianza 0.544 queda pendiente en vez de alucinar | `ask` tiene `P(aprobación)=0.544` → queda pendiente; además, cualquier fallo/duda del LLM deja la palabra pendiente |

Cifras del vídeo (una corrida suya): 168 de 175 decisiones resueltas
automáticamente, 133 en ramas `if`, 24 escaladas a Sonnet, 11 a aprobación
humana, confianza media 0.925. Las nuestras están en §5.

## 2. Arquitectura

- `src/prompt_spider.rs` (sin feature `web`, corre en el smoke de CI):
  tokenizador, contexto del prompt, scorer, router, prompts del LLM y parseo
  robusto del JSON del LLM. Todo puro y testeado.
- `src/web/spider_job.rs` (feature `web`): job asíncrono, cola de aprobación,
  persistencia en `data/spider_runs/<id>.json` (env `SPIDER_RUNS_DIR`) y
  endpoints.
- `web/static/spider.js` + sección `#main-spider` en `index.html` + estilos
  `.sp-*` en `styles.css`: la vista. El navegador **no decide** nada; solo
  dibuja eventos del servidor.

### Flujo de una corrida

1. `tokenize(prompt)` → tokens con spans en caracteres, línea y sección
   (`<role>`, `<objective>`, …). Etiquetas y marcadores de lista (`1.`, `-`) se
   **leen** (cuentan en «palabras») pero no son forks.
2. Para cada palabra/número (fork): `score_token` → `P(sí)` para las 4
   preguntas; `p` = certeza de la pregunta **menos segura** (`max(q, 1-q)`);
   si `P(aprobación) ≥ 0.5`, la pregunta decisiva es aprobación y
   `p = P(aprobación)`.
3. `route`: aprobación → `you`; `p ≥ umbral` → `code`; resto → `llm`.
4. Las palabras `llm` van a un worker que las agrupa según el **perfil del LLM
   que atiende** (corrida corta/completa, §4b). API externa: lotes de 8 con un
   *system prompt* fijo que exige JSON
   `{"verdicts":[{"i","relevant","grounded","ambiguous","needs_approval","p","note"}]}`
   (temperatura 0). Gemma local: prompt compacto por palabra y lectura de
   `P(sí)` (sin generar texto, §4b). El crawler sigue caminando mientras tanto;
   la UI muestra el progreso de cada llamada.
5. Veredicto: si el LLM **no** pide aprobación, **no** la marca ambigua y su
   `p ≥ piso` (0.70) → resuelta por el LLM. Si no → pendiente para ti. Si el
   LLM falla, no responde a tiempo, devuelve basura o se salta una palabra →
   pendiente con la causa. **Nunca se inventa un veredicto.**
6. Fin: resumen, evento `done`, JSON en `data/spider_runs/`. Aprobar/Rechazar
   actualiza la corrida y la vuelve a guardar.

### Endpoints (todos bajo `/api/*` → exigen sesión `MASTER_SECRET`)

| Método | Ruta | Uso |
|---|---|---|
| GET | `/api/spider/sample` | prompt de ejemplo + umbral/piso por defecto |
| POST | `/api/spider/start` | `{prompt?, threshold?, llm_floor?, pace_ms?, batch_size?, mode?: auto\|corta\|completa, max_escalations?}` → `run_id`, LLM usado, `mode`, `profile`, `fallback_available` (409 si ya corre) |
| POST | `/api/spider/stop` | detener (lo escalado sin veredicto queda pendiente) |
| GET | `/api/spider/status` | corrida completa (tokens, decisiones, llamadas al LLM) + resumen + LLM activo + respaldo + perfiles `corta`/`completa` por defecto |
| GET | `/api/spider/events?after=N` | eventos por sondeo |
| GET | `/api/spider/stream?after=N` | SSE (`event: spider`, mismo patrón que la consola de entrenamiento) |
| POST | `/api/spider/decide` | `{run_id?, index, approve}` — solo palabras pendientes (409 si no) |
| GET | `/api/spider/runs` | corridas guardadas (resumen) |
| GET | `/api/spider/export?id=` | JSON descargable (`run` + `summary`); sin claves de API |

## 3. Heurísticas (deterministas, en `score_token`)

Léxicos en inglés y español (`STOPWORDS`, `AMBIGUITY_MARKERS`,
`CLAIM_MARKERS`, `APPROVAL_TERMS`, `KNOWN_ENTITIES`).

| Regla | relevant | grounded | ambiguous | Efecto típico |
|---|---|---|---|---|
| Stopword / token estructural (`the`, `de`, `with`…) | 0.03 | 0.99 | 0.03 | p≈0.97 → code |
| Número / formato (`2026`, `1080x1920`, `45`) | 0.97 | 0.99 | 0.02 | p=0.97 → code |
| Palabra de contenido | 0.93 base; +0.04 en `<objective>`; +0.03 en `<role>`; +0.02 por repetición (máx +0.04); +0.05 entidad conocida; +0.02 si ≥5 letras, +0.01 más si ≥7 | 0.97 si se repite / entidad / en role u objective; si no 0.955 | 0.04 | palabras cortas y sueltas (`Use`, `cut`, `flag`) → 0.93 → llm |
| Marcador de ambigüedad (`it`, `this`, `something`, `anything`, `vague`, `short`, `clear`…) | ≥0.5 | ≤0.9 | 0.38 | p=0.62 → llm |
| Afirmación sin fuente (`best`, `every`, `never`, `complete`, `launch-ready`…) | — | 0.40 si aparece una vez fuera de role/objective; si no 0.80 | — | → llm |
| Nombre propio desconocido (mayúscula a mitad de frase, aparece una vez, no es entidad) | — | 0.80 | — | → llm |
| Verbo/sustantivo de aprobación | — | — | — | `P(aprobación)` léxica: `approve` 0.97, `publish` 0.95, `send` 0.92, `ship` 0.90, `pay` 0.94, `emails` 0.62, **`ask` 0.544**, `request` 0.53 → you |

Notas:

- Todas las cifras se redondean a 3 decimales; la misma entrada da siempre la
  misma salida (hay test de determinismo).
- Inicio de frase (tras `.`, `:`, salto de línea, etiqueta, marcador) no cuenta
  como nombre propio.
- El umbral solo mueve palabras entre `code` y `llm`; nunca saca algo de
  `you`.

## 4b. Respaldo Gemma local y corridas cortas

**Problema en producción** (Railway, CPU): una corrida con Gemma local mandó
1 lote de ~11 palabras con el prompt completo (~1000+ tokens de prefill);
Gemma agotó el plazo de 60 s → 0 resueltas, todo pendiente.

**Respaldo**: si el LLM activo es una API externa (p. ej. docker-llm) y falla
por conexión, timeout, HTTP 5xx, 401/403/404 o sin modelo, la corrida usa
**Gemma local (respaldo)** —si el GGUF está cargado— para esa palabra y el
resto de la corrida. Aviso: «docker-llm no disponible → Gemma local» (evento
`fallback`, `run.fallback`, `llm_calls[].fallback`). El chat crudo (OFF)
hace lo mismo (`RAW_CHAT_FALLBACK_RESERVE_SECS`, 30 s reservados para Gemma),
coherente con el respaldo que ya tenía el decoder del campo (ON). Gemma local
sigue siendo el proveedor activo por defecto.

**Perfiles** (env por proveedor `SPIDER_LOCAL_*` / `SPIDER_API_*`, genéricas
`SPIDER_LLM_TIMEOUT_SECS`, `SPIDER_BATCH_SIZE`, `SPIDER_MAX_ESCALATIONS`,
`SPIDER_LLM_MAX_TOKENS`):

| | lote | plazo/llamada | máx. escaladas | prompt |
|---|---|---|---|---|
| Gemma local · **corta** (defecto con Gemma) | 1 | 45 s | 6 | compacto, P(sí) |
| Gemma local · completa | 2 | 90 s | — | compacto, P(sí) |
| API · corta | 8 | 60 s | 24 | completo, JSON (max_tokens 1200) |
| API · completa (defecto con API) | 8 | 60 s | — | completo, JSON |

Lo que excede el límite queda pendiente con «límite de corrida corta»; tras 2
(corta) o 3 (completa) fallos seguidos, el resto queda pendiente sin consultar.

**Por qué P(sí) y no JSON generado con Gemma 2 2B** (medido en el box, Q3_K_L,
8 vCPU, prompt de ejemplo, 20 palabras escaladas):

| Variante | Resultado |
|---|---|
| Prompt completo + JSON (antes, producción) | lote de ~11 palabras → timeout de 60 s en Railway, 0 resueltas (reporte del usuario; no re-medido) |
| Compacto, lotes de 2, `p` numérica generada | 16–26 s por lote; `p=0.0` o veredicto ausente → 0 resueltas de 20 |
| Compacto, pidiendo `p` numérica | ~300 tokens, ~10 s; copia el esquema (`"p":0.0-1.0`) → JSON inválido |
| Compacto categórico `sí/no/dudo` generado | ~190 tokens, ~5.5 s, 4 tokens generados; responde **«sí» a las 20**, también a la pregunta inversa («¿es vaga?») |
| `P(sí)` del siguiente token, opciones `<sí\|no>` | P(sí) normalizada 0.82–1.00 en las 20 (sesgo de aquiescencia) |
| Igual con opciones `<no\|sí>` | 0.09–0.99: cambia mucho con el orden (sesgo de primera opción) |

Decisión: para Gemma local se lee `P(sí)` (un forward, sin generar) con las
**dos órdenes** de opciones y `p = min(P₁, P₂)`: solo resuelve si el «sí» es
consistente. Es la probabilidad del propio modelo (no se inventa), con un
sesgo medido corregido de forma conservadora.

## 4. UI (pestaña *Prompt Spider*)

Recrea el layout del vídeo con estética de terminal oscura:

- Barra superior: contadores `palabras · forks · code · <llm> · tú`.
- Izquierda `studio.prompt`: editor (prompt de ejemplo original: un *chief of
  staff* de un estudio de contenido con `<prompt><role><objective>…`, 159
  palabras); durante la corrida pasa a vista resaltada por ruta (verde code,
  naranja LLM, amarillo tú; borde discontinuo = esperando al LLM).
- Centro: araña en `<canvas>`: líneas desde la línea del prompt de cada palabra
  hasta nodos en un arco (etiquetas rosas con borde del color de la ruta),
  abanico desde la palabra actual y línea animada de la última decisión.
- Derecha: tres cajas `if p ≥ 0.95 → code`, `p < 0.95 → <LLM real>`,
  `solo aprobación → tú`.
- Abajo: *decision stream* (palabra, pregunta tipada, p, ruta), «enrutadas en
  esta sesión» + barras, sparkline de confianza con el umbral y la media,
  rejilla de palabras leídas (x/N), esfera de cobertura %, panel «dónde pensó»
  con el lema «el modelo solo piensa donde el código no puede», y la cola de
  aprobación con **Aprobar / Rechazar**.
- Controles: umbral (slider), ritmo de la animación, **corrida**
  (auto / corta / completa), Ejecutar / Detener, Editar, Ejemplo, Exportar
  JSON y un **?** con la ayuda de métricas (enlaza a
  `prompt_spider_metricas.md`).
- Línea de progreso por llamada al LLM («llamada k → LLM: «palabra» ·
  escaladas s/cap · plazo» + «esperando… Xs») y aviso amarillo cuando entra
  el respaldo.
- Responsive: probado a 360, 412 y 1280 px (sin scroll horizontal; las
  pestañas siguen envolviendo en filas).

## 5. Resultados de una corrida local

Prompt de ejemplo, umbral 0.95, piso del LLM 0.70, lotes de 8. LLM: **servidor
mock OpenAI-compatible** (determinista, local; no es un modelo real), así que
las cifras del LLM muestran el *plumbing*, no la calidad de un modelo:

| Métrica | Valor |
|---|---|
| tokens leídos (palabras) | 175 |
| forks (palabras/números) | 159 |
| code (ramas `if`) | 129 (81 %) |
| escaladas al LLM | 20 (3 lotes) |
| resueltas por el LLM | 13 |
| a ti (aprobación + dudas del LLM) | 17 (10 por léxico de aprobación + 7 que el LLM no resolvió) |
| resueltas automáticamente | 142 de 159 |
| confianza media | 0.933 |
| `ask` | `P(aprobación)=0.544` → pendiente |

Con Gemma local sin GGUF (`GEMMA2_AUTO_DOWNLOAD=0`), las escaladas quedan
**pendientes** con «LLM no disponible» (test de integración), en vez de
inventar un veredicto.

### 5b. Corrida corta real con Gemma local (GGUF Q3_K_L, CPU 8 vCPU del box)

`GEMMA2_GGUF=models/gemma-2-2b-it-Q3_K_L.gguf cargo test --release --features web --lib real_gemma_spider -- --ignored --nocapture`
(`SPIDER_REAL_MODES=corta`), prompt de ejemplo, umbral 0.95:

| Métrica | Valor |
|---|---|
| forks / code | 159 / 129 |
| escaladas / enviadas / por límite | 20 / 6 / 14 |
| llamadas / fallidas | 6 / 0 |
| latencia por llamada (1 palabra = 2 lecturas de P(sí), ~180 tokens de prefill c/u) | 10.4–16.6 s (mediana ≈ 11 s; ~5 s por lectura) |
| resueltas por Gemma | 2 (`launch-ready` p=0.991, `complete` p=0.717) |
| «Gemma duda» | 4 (`clear` 0.427, `short` 0.314, `it` 0.239, `Use` 0.091) |
| tú (final) | 28 |
| tiempo total | 72.5 s |

Comparación: la versión anterior en el mismo box (prompt completo, lotes de 10)
tardó 16–26 s por lote y resolvió **0** de 20. En Railway (CPU más lenta; chat
crudo ON medía 18–23 s) se espera ~2× por llamada, dentro del plazo de 45 s.

## 6. Limitaciones

- Las heurísticas son **léxicas** y calibradas a mano sobre el prompt de
  ejemplo (inglés con apoyo de español). No entienden sintaxis ni negación
  (`never publish` sigue disparando `publish` → aprobación, lo que aquí es
  conservador).
- Las «probabilidades» no están calibradas estadísticamente: son puntuaciones
  transparentes, no frecuencias observadas.
- La calidad del veredicto escalado depende del LLM activo. Gemma 2 2B en CPU
  es un juez débil (sesgos de aquiescencia y de orden medidos en §4b): el
  `min` de las dos órdenes es conservador y deja muchas palabras pendientes;
  para resolver más usa una API externa.
- Una sola corrida activa a la vez (por servidor). Los eventos se guardan en
  memoria (últimos 4000); la corrida completa sí se persiste en JSON.
- El ritmo (`pace_ms`) es solo para la animación; con `pace_ms=0` una corrida
  del ejemplo sin LLM tarda milisegundos.
- No reproduce el vídeo exactamente: su prompt, sus preguntas internas y sus
  umbrales intermedios no son públicos; esta es una reconstrucción a partir de
  las capturas y la descripción.
