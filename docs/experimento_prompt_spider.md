# Experimento: Prompt Spider — «cada palabra, un fork»

> **Estado:** experimental (rama `exp/prompt-spider`, sin merge a `main`).
> Pestaña **Prompt Spider** de la app web (`agentic_web`).

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
4. Las palabras `llm` se agrupan en lotes (8 por defecto) y un worker las manda
   al LLM activo con un *system prompt* fijo que exige JSON
   `{"verdicts":[{"i","relevant","grounded","ambiguous","needs_approval","p","note"}]}`
   (temperatura 0, plazo `SPIDER_LLM_TIMEOUT_SECS`, 60 s por defecto). El
   crawler sigue caminando mientras tanto.
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
| POST | `/api/spider/start` | `{prompt?, threshold?, llm_floor?, pace_ms?, batch_size?}` → `run_id`, LLM usado (409 si ya corre) |
| POST | `/api/spider/stop` | detener (lo escalado sin veredicto queda pendiente) |
| GET | `/api/spider/status` | corrida completa (tokens, decisiones, llamadas al LLM) + resumen + LLM activo |
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
- Controles: umbral (slider), ritmo de la animación, Ejecutar / Detener,
  Editar, Ejemplo, Exportar JSON.
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

Con Gemma local sin GGUF (`GEMMA2_AUTO_DOWNLOAD=0`), las 20 escaladas quedan
**pendientes** con «LLM no disponible» (test de integración), en vez de
inventar un veredicto.

## 6. Limitaciones

- Las heurísticas son **léxicas** y calibradas a mano sobre el prompt de
  ejemplo (inglés con apoyo de español). No entienden sintaxis ni negación
  (`never publish` sigue disparando `publish` → aprobación, lo que aquí es
  conservador).
- Las «probabilidades» no están calibradas estadísticamente: son puntuaciones
  transparentes, no frecuencias observadas.
- La calidad del veredicto escalado depende del LLM activo; con Gemma 2 2B en
  CPU puede ser lento (plazo por lote configurable) y su JSON menos fiable; el
  parser lo tolera, pero lo que no parsea queda pendiente.
- Una sola corrida activa a la vez (por servidor). Los eventos se guardan en
  memoria (últimos 4000); la corrida completa sí se persiste en JSON.
- El ritmo (`pace_ms`) es solo para la animación; con `pace_ms=0` una corrida
  del ejemplo sin LLM tarda milisegundos.
- No reproduce el vídeo exactamente: su prompt, sus preguntas internas y sus
  umbrales intermedios no son públicos; esta es una reconstrucción a partir de
  las capturas y la descripción.
