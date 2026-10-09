# Prompt Spider — documentación técnica y métricas

> Pestaña **Prompt Spider** de la app web (`agentic_web`). Código:
> `src/prompt_spider.rs` (lógica pura), `src/web/spider_job.rs` (job,
> endpoints, LLM), `web/static/spider.js` (vista). Diseño y resultados
> medidos: [`experimento_prompt_spider.md`](experimento_prompt_spider.md).

## 1. ¿Para qué sirve?

El Prompt Spider («spider tracker») **audita un prompt palabra por palabra**
antes de dárselo a un agente. En vez de mandar el prompt entero a un modelo
y esperar, un crawler determinista lo recorre y trata cada palabra como un
**punto de decisión (fork)**. Para cada una decide quién debe «pensar»:

- el **código** (reglas deterministas, gratis, instantáneas),
- el **LLM activo** (solo donde el código duda),
- **tú** (lo que compromete algo hacia fuera o el LLM no pudo resolver).

Casos de uso:

- **Revisar un prompt de agente antes de ejecutarlo**: localizar palabras
  vagas (`it`, `something`, `anything`), afirmaciones sin fuente (`best`,
  `every`, `never`, `launch-ready`) y verbos con efectos externos (`publish`,
  `send`, `ship`, `pay`, `ask`) que deberían pasar por aprobación humana.
- **Medir dónde hace falta un modelo**: el resumen dice qué fracción del prompt
  se resuelve sin LLM (ramas `if`) y cuántas llamadas al modelo hicieron falta.
- **Comparar modelos/umbrales**: misma entrada, distinto LLM activo o
  umbral → distintas cifras de `code / LLM / tú` y de confianza.
- **Cola de aprobación**: lo riesgoso nunca se resuelve solo; queda en una
  lista con Aprobar / Rechazar y se exporta en JSON.

Lo que **no** es: no reescribe el prompt, no ejecuta la tarea y sus
probabilidades heurísticas no están calibradas estadísticamente (son
puntuaciones transparentes y deterministas).

## 2. Pipeline

```
prompt ──tokenize──▶ tokens ──score_token (4 preguntas)──▶ p ──route──┬─▶ code  (p ≥ umbral)
                                                                     ├─▶ LLM   (p < umbral) ──▶ resuelta | tú (pendiente)
                                                                     └─▶ tú    (P(aprobación) ≥ 0.5)
```

1. **Tokenizar** (`tokenize`): palabras, números, etiquetas (`<role>`,
   `</objective>`) y marcadores de lista (`1.`, `-`), con línea y sección.
   Todos los tokens se **leen** (cuentan en «palabras»), pero solo
   palabras y números son **forks**.
2. **Cuatro preguntas** por fork (`score_token`), cada una con `P(sí)`:
   `relevant` (¿relevante para la tarea?), `grounded` (¿afirmación con fuente
   en el prompt?), `ambiguous` (¿es ambigua? — la UI la muestra como
   «¿es específico?»), `approval` (¿requiere aprobación humana?).
3. **p del fork**: certeza de la pregunta **menos segura**,
   `p = min over q (max(P(sí), 1 − P(sí)))`; esa pregunta es la «pregunta
   tipada» que se muestra. Si `P(aprobación) ≥ 0.5`, la pregunta decisiva es
   aprobación y `p = P(aprobación)` (p. ej. `ask` → 0.544).
4. **Router** (`route`, sin opinión propia):
   aprobación → **tú**; `p ≥ umbral` → **code**; resto → **LLM**.
5. **Escalado al LLM** (worker asíncrono; el crawler no se detiene):
   - **API externa** (docker-llm u otra OpenAI-compatible): lotes de 8
     palabras con el prompt completo; pide JSON
     `{"verdicts":[{"i","relevant","grounded","ambiguous","needs_approval","p","note"}]}`.
   - **Gemma local** (GGUF en CPU): prompt **compacto** por palabra (contexto
     ±6 palabras + pregunta tipada, ~180 tokens) y **sin generar texto**: se
     lee `P(sí)` del siguiente token con un forward. Se pregunta dos veces
     con el orden de opciones invertido (`<sí|no|dudo>` y `<no|sí|dudo>`)
     porque Gemma 2 2B tiene un sesgo fuerte por la primera opción;
     `p = min(P(sí)₁, P(sí)₂)` (normalizadas `sí/(sí+no)`).
   - Un veredicto **resuelve** solo si no pide aprobación, no marca la
     palabra ambigua y `p ≥ piso` (0.70). Si no → pendiente para ti.
6. **Respaldo Gemma local**: si el LLM activo es una API externa y falla por
   conexión, timeout, HTTP 5xx, 401/403/404 o falta de modelo, la corrida
   cambia a **Gemma local (respaldo)** (si el GGUF está cargado) para esa
   palabra y el resto de la corrida. Se ve como aviso amarillo
   «`docker-llm` no disponible → Gemma local» y se guarda en
   `run.fallback`. Un JSON malo **no** activa el respaldo (la API respondió).
   El chat crudo (modo OFF) hace lo mismo: reserva 30 s
   (`RAW_CHAT_FALLBACK_RESERVE_SECS`) del plazo para Gemma y etiqueta la
   respuesta `Gemma local (respaldo)`.
7. **Nunca se inventa**: cualquier fallo (timeout, sin JSON, sin veredicto,
   límite de corrida, detenido) deja la palabra **pendiente con el motivo**.
8. Fin: evento `done`, resumen y JSON en `data/spider_runs/<id>.json`.

### Corrida corta vs completa

Selector **corrida** en los controles:

| | Gemma local · corta | Gemma local · completa | API · corta | API · completa |
|---|---|---|---|---|
| palabras por llamada (`batch_size`) | 1 | 2 | 8 | 8 |
| plazo por llamada | 45 s | 90 s | 60 s | 60 s |
| máx. escaladas por corrida | **6** | sin límite | 24 | sin límite |
| prompt | compacto (P(sí)) | compacto (P(sí)) | completo (JSON) | completo (JSON) |
| fallos seguidos antes de abandonar | 2 | 3 | 2 | 3 |

- **auto** (por defecto): corta con Gemma local, completa con API. Si la API
  cae a mitad y el modo es auto, el respaldo usa el perfil **corta** de Gemma.
- Al llegar al límite de la corrida corta, las escaladas restantes **no se
  envían**: quedan pendientes con «límite de corrida corta (N escaladas al
  LLM); pendiente sin consultar» y cuentan en `llm_capped`.
- Tras N fallos seguidos del LLM, el resto queda pendiente «… falló N veces
  seguidas; pendiente sin consultar» (evita esperar 45 s × palabra).

Variables de entorno (prioridad: específica del proveedor → genérica →
defecto). Específicas: `SPIDER_LOCAL_<X>` (Gemma) y `SPIDER_API_<X>` (API);
genérica: `SPIDER_<X>`, con `X` ∈ `LLM_TIMEOUT_SECS`, `BATCH_SIZE`,
`MAX_ESCALATIONS` (solo corta; `0` = sin límite), `LLM_MAX_TOKENS` (API y
modo no compacto). La petición `POST /api/spider/start` acepta además
`mode` (`auto|corta|completa`), `batch_size` y `max_escalations`.

## 3. Umbral y piso

- **Umbral** (slider, 0.80–0.99, por defecto 0.95): `p ≥ umbral` → code.
  Subirlo manda más palabras al LLM; bajarlo, menos. Nunca saca nada de la
  cola de aprobación (eso depende solo de `P(aprobación)`).
- **Piso del LLM** (0.70, `llm_floor` en la API): confianza mínima del
  veredicto del LLM para darlo por resuelto.

## 4. Métricas y paneles de la UI

### Contadores superiores

| Contador | Definición exacta |
|---|---|
| **palabras** | Tokens **leídos** por el crawler (cursor): palabras, números, etiquetas y marcadores. |
| **forks** | Palabras/números ya evaluados con las 4 preguntas (decisiones creadas). |
| **code** | Forks con ruta final `code` (`p ≥ umbral`): resueltos por reglas, sin modelo. |
| **LLM** (muestra el nombre del LLM) | Forks **escalados** al LLM (`first_route = llm`), incluidos los que después quedaron para ti, los no enviados por límite y los fallidos. |
| **tú** | Forks con ruta **final** `you`: piden aprobación (léxico) **o** el LLM no los resolvió. |

Nota: `code + tú + resueltas por el LLM = forks` cuando la corrida terminó.

### Cajas de ruta (derecha)

- **if p ≥ umbral → code**: número de forks `code`; pie «sin llamada al modelo».
- **p < umbral → LLM**: número de escaladas; pie
  «N llamadas · M resueltas · K por límite» — N = llamadas al LLM (incluye
  fallidas y la de la API que activó el respaldo), M = escaladas resueltas por
  el LLM, K = no enviadas por el límite de corrida corta.
- **solo aprobación → tú**: forks con ruta final `you`; pie
  «P pendientes · A✓ R✗» (pendientes / aprobadas / rechazadas).

### Línea de progreso del LLM y aviso de respaldo

- Naranja: «llamada k → LLM: «palabra» · escaladas s/cap · plazo Ts» con un
  contador «esperando… Xs» mientras la llamada está en curso; al terminar,
  «llamada k → LLM: n palabra(s), v veredicto(s), Ys» o «… falló (Ys): motivo».
- Amarillo: «⚠ docker-llm no disponible → Gemma local» cuando entra el respaldo.
- Pie inferior: corrida, tamaño de lote, plazo y límite efectivos.

### Decision stream

Últimos 40 forks (más reciente arriba). Columnas:

1. **%**: posición del token en el prompt (`índice / tokens_total`).
2. **palabra**.
3. **pregunta tipada**: la pregunta menos segura (o aprobación).
4. **p**: `final_p` (ver abajo).
5. **ruta**: `→ code`, `→ <LLM>` (resuelta por el LLM), `→ tú (pend.)`,
   `→ tú (✓/✗)`; `…` = esperando al LLM.

### Enrutadas en esta sesión

Número grande = forks enrutados; subtítulo «de F palabras del prompt (T
tokens)». Barras: proporción de forks por **ruta final** — code, LLM
(resueltas por el LLM) y tú.

### Confianza (sparkline) y media

Serie de `final_p` de cada fork en orden de llegada, con línea discontinua en
el umbral. `final_p` = p heurística del código; **se sustituye por la p del
LLM** cuando el LLM devolvió veredicto para esa palabra (resuelta o no). Las
palabras con fallo / límite / sin veredicto conservan la p heurística.
**media** = promedio aritmético de `final_p` sobre todos los forks
(`summary.avg_confidence`).

### Palabras leídas (rejilla)

Un punto por **token** (incluye etiquetas): gris = no leído, tenue = leído
sin fork, color = ruta del fork. «x / N» = tokens leídos / total.

### Cobertura del prompt (esfera)

`cobertura = tokens leídos / tokens totales`. Significa **leído**, no
resuelto: una corrida terminada tiene 100 % aunque queden pendientes.

### Dónde pensó

- **code**: «N if-statements» = forks resueltos por reglas.
- **LLM**: «N llamadas · M resueltas · F fallidas» (llamadas = peticiones al
  modelo; con Gemma local cada llamada lee P(sí) dos veces por palabra).
- **tú**: «N aprobaciones (P pendientes)» = forks con ruta final tú.

### Pendientes de aprobación

Lista de forks con ruta final `tú` (pendientes primero). Cada tarjeta muestra
palabra, p heurística, pregunta, contexto ±6 y el **motivo**:

| Motivo (texto) | Significado |
|---|---|
| `requiere aprobación (p=…)` | El léxico detectó un verbo con efectos externos (`publish`, `send`, `ask`…). Nunca pasa por el LLM. |
| `<LLM> pide aprobación humana` | El LLM marcó `needs_approval`. |
| `<LLM> la marca ambigua (p=…)` | El LLM la considera ambigua. |
| `<LLM> duda (p=… < 0.70)` | LLM inseguro: su p no llega al piso (con Gemma: p = min de las dos lecturas). |
| `<LLM> no devolvió veredicto: pendiente` | El JSON no traía esa palabra. |
| `<LLM> falló; pendiente (no se inventa): …` | Error del LLM: timeout, conexión, JSON inválido, etc. |
| `<LLM> falló N veces seguidas; pendiente sin consultar` | Se dejó de llamar tras fallos consecutivos. |
| `límite de corrida corta (N escaladas al LLM); pendiente sin consultar` | Corrida corta: no se envió. |
| `detenido antes del veredicto del LLM` / `sin veredicto del LLM (corrida detenida)` | Pulsaste Detener. |

**Aprobar / Rechazar**: `POST /api/spider/decide {run_id, index, approve}`.
Solo sobre pendientes (409 si no). Cambia `status` a `approved`/`rejected`,
`resolved_by = "you"`; la ruta sigue siendo `tú`. Actualiza contadores
(✓/✗, `summary.approved/rejected`) y vuelve a guardar el JSON. No cambia
las heurísticas; sí queda como **episodio de vigilia** del decoder del campo
(etiqueta `user`), que se consolida en el siguiente sueño (pestaña Sueño o
«Entrenamiento Prompt Spider»). Ver §7.

## 5. Export JSON (`GET /api/spider/export?id=`)

```jsonc
{
  "run": {
    "id": "uuid", "started_ms": 0, "finished_ms": 0,
    "state": "running|done|stopped",
    "prompt": "…", "threshold": 0.95, "llm_floor": 0.7,
    "batch_size": 1, "pace_ms": 60,
    "llm": {"id": "gemma_local|<id API>", "label": "Gemma local", "kind": "local|openai_compatible", "model": "…", "unavailable": "…?"},
    "mode": "corta|completa",
    "profile": {"mode": "corta", "batch_size": 1, "timeout_secs": 45, "max_tokens": 16,
                "max_escalations": 6, "compact": true, "max_consecutive_failures": 2},
    "fallback": {"from": "API · docker-llm (…)", "to": "Gemma local (respaldo)", "reason": "…", "message": "docker-llm no disponible → Gemma local"},
    "tokens": [{"index": 0, "text": "…", "lower": "…", "kind": "word|number|tag|list_marker",
                "start": 0, "end": 0, "line": 0, "section": "role", "sentence_start": false}],
    "cursor": 175,
    "decisions": [{
      "index": 12, "word": "vague", "question": "relevant|grounded|ambiguous|approval",
      "p": 0.62, "scores": {"relevant": 0, "grounded": 0, "ambiguous": 0, "needs_approval": 0},
      "reasons": ["…"], "first_route": "code|llm|you", "route": "code|llm|you",
      "status": "resolved|escalating|pending|approved|rejected", "final_p": 0.62,
      "verdict": {"index": 12, "relevant": true, "grounded": true, "ambiguous": false, "needs_approval": false, "p": 0.4, "note": "…"},
      "resolved_by": "code|<LLM>|you", "note": "motivo", "capped": true
    }],
    "llm_calls": [{"batch": 1, "words": [12], "ok": true, "seconds": 9.8, "verdicts": 1,
                   "model": "gemma2-gguf·P(sí)", "error": "…?", "llm": "Gemma local", "fallback": true}]
  },
  "summary": {
    "tokens_total": 175, "words_read": 175, "forks_total": 159, "forks": 159, "code": 129,
    "llm_escalated": 20, "llm_resolved": 0, "escalating": 0, "you": 30, "pending": 30,
    "approved": 0, "rejected": 0, "auto_resolved": 129, "avg_confidence": 0.915,
    "coverage": 1.0, "llm_calls": 6, "llm_failures": 0, "llm_sent": 6, "llm_capped": 14
  }
}
```

Con el **decoder del campo** activo la corrida añade `run.decider = "campo"`,
`run.field_model` (nombre, entrenado, ejemplos, sueños, hold-out), en cada
escalada `decision.field = {probs[5], p_ok, novelty, micros, resolved,
reason}` (`resolved_by = "campo"` si la resolvió) y en el resumen `decider`,
`field_escalated`, `field_resolved` y `field_latency_us` (mediana). Las
corridas con LLM tienen `decider = "llm"` y `field_* = 0`.

Campos opcionales (`verdict`, `note`, `capped`, `fallback`, `error`…) se
omiten cuando están vacíos. `auto_resolved = code + llm_resolved`;
`llm_sent` = palabras enviadas en llamadas (incluye las fallidas y la llamada
de la API que activó el respaldo, por lo que puede superar `llm_escalated`).
Nunca incluye claves de API.

## 6. Cómo interpretar una corrida (ejemplo real)

Prompt de ejemplo, umbral 0.95, **Gemma local** (Q3_K_L en CPU de 8 vCPU),
corrida **corta** (medido, ver `experimento_prompt_spider.md` §5):

- 175 tokens leídos (**cobertura 100 %**), **159 forks**.
- **code 129** (81 %): stopwords, números, palabras repetidas o en
  `<role>/<objective>` — resueltas sin modelo.
- **LLM 20** escaladas → solo **6 enviadas** (límite corto), 6 llamadas,
  0 fallidas, 10.4–16.6 s por llamada, 72.5 s en total.
  - **2 resueltas** por Gemma (`launch-ready` p=0.991, `complete` p=0.717).
  - 4 «Gemma local duda» (`clear` 0.427, `short` 0.314, `it` 0.239,
    `Use` 0.091).
  - 14 pendientes con «límite de corrida corta».
- **tú 28** = 10 por léxico de aprobación (`ship`, `publish`, `ask`…) + 18
  escaladas no resueltas. `auto_resolved = 129 + 2 = 131`; media 0.915.

Lectura: el 81 % del prompt no necesitó modelo; las palabras dudosas
(`vague`, `anything`, `it`, `best`…) y los verbos con efectos externos son
exactamente lo que conviene revisar a mano o con un LLM mejor. La corrida
corta termina en ~1 min con Gemma en CPU sin timeouts; para resolver más,
usa **completa** (≈ 20 × 11 s) o una API externa (docker-llm).

## 7. Decoder del campo (interruptor en la barra del Spider)

Con **Decoder del campo** activo (se recuerda en el navegador), las palabras
escaladas (`p < umbral`) **no** van al LLM: las decide el router del campo
del Spider (líquido + CDT + RQM/EPR) entrenado en **Entrenamiento Prompt
Spider** (tarjeta al final de la pestaña). Arquitectura, entrenamiento y el
experimento de sustrato: [`prompt_spider_decoder_campo.md`](prompt_spider_decoder_campo.md).

Cambios en los paneles con el decoder activo:

| Panel | Con LLM | Con decoder del campo |
|---|---|---|
| Nombre del escalado (contador, caja, chip, barras, stream) | etiqueta del LLM | `campo` |
| Pie de la caja `p < umbral` | «N llamadas · M resueltas · K por límite» | «campo · M resueltas · 0 llamadas · X µs» (X = mediana de latencia por decisión) |
| Dónde pensó | «N llamadas · M resueltas» | «decoder del campo · M resueltas · P a ti · X µs» |
| `final_p` / sparkline | p del LLM si devolvió veredicto | `P(ok)` calibrada del campo |
| Línea de progreso del LLM | llamadas en curso | no aparece (no hay llamadas) |

Una escalada se resuelve por el campo (`resolved_by = campo`, ruta final
`llm`) solo si `P(ok) ≥ piso (0.70)`, `P(aprobación) < 0.5`,
`P(ambigua) < 0.5` y la **novedad** CDT está dentro de lo visto en
entrenamiento. Si no, queda **pendiente** con uno de estos motivos:

| Motivo | Significado |
|---|---|
| `campo: decoder del campo sin entrenar; pendiente (no se inventa)` | Aún no hay modelo: entrena primero. |
| `campo: fuera de distribución (novedad a > b); pendiente` | El estado líquido no se parece a ningún engrama CDT consolidado. |
| `campo: el campo pide aprobación (P=…)` | La cabeza `needs_approval` ≥ 0.5. |
| `campo: el campo la marca ambigua (P=…)` | La cabeza `ambiguous` ≥ 0.5. |
| `campo: el campo duda (P(ok)=… < 0.70)` | Confianza calibrada bajo el piso. |

Las palabras de aprobación (ruta inicial `tú`) y las de código no cambian:
el campo solo sustituye al LLM en las escaladas.

### Tarjeta «Entrenamiento Prompt Spider»

- **datasets** (2 / 6 / 12 / ∞), **prompts/dataset**, **maestro LLM** (0 / 4 /
  12 escaladas por dataset consultadas en vivo al LLM activo), **corridas
  guardadas** (usar etiquetas de `data/spider_runs`), **sustrato nuevo**
  (empezar en blanco).
- Tabla: en las **escaladas** del hold-out fijo — H1 (vocabulario visto, 24
  prompts) y H2 (vocabulario nuevo, 24 prompts) — exactitud de `ok`, ECE (10
  bins), Brier, % resueltas por el campo y precisión de lo resuelto, frente a
  la heurística (scorer + piso) en las mismas palabras; exactitud / ECE /
  Brier por pregunta en todas las palabras de H1; latencia mediana; en datos
  reales, exactitud frente a tus Aprobar/Rechazar y acuerdo con el maestro LLM
  (1 de cada 5 palabras etiquetadas se reserva para esto).
- Registro: dataset → vigilia → sueño (clases CDT, celdas RQM, repetición) →
  evaluación → checkpoint (`data/checkpoints/spider_field/`).

