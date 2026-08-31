# Runtime de rutas de capas y red multiagente

**Velocidad primero: un Gemma congelado, muchas rutas, ningún segundo modelo en el camino feliz**

**Fecha:** 31 de agosto de 2026
**Rama:** `feature/optimizacion-velocidad-rutas`
**Estado:** V1–V4 implementados (sin GGUF en CI). V5–V8 pendientes.

### Qué hay en código (para el bot que continúe)

| PR | Código | Tests |
|---|---|---|
| V1 KV-A | `plan_wake_prefill` aplica máscara recordada; si cambia, no reutiliza KV | `kv_a_*`, `later_wake_turn_*` |
| V2 LRC | `src/layer_route_cache.rs`, persistencia `layer-routes.json` | 7 tests del módulo |
| V3 chat | log `layers=a/b route=hit\|miss tok/s=…`, `observe_layer_route_turn` | binario `native_gemma2_circadian_chat` compila |
| V4 KL sueño | `replay_prompt_for_mask` promociona por KL ≤ 0,15, sin umbral 0,92 | `kl_promotion_*`, `lrc_promoted_route_*` |

Pendiente: V5 (local vs global), V6 (grafo de 6 nodos), V7 (early exit), V8 (benchmark CSV). Los tests GGUF siguen `#[ignore]`.
**Alcance:** Gemma 2 nativo (Candle/GGUF) + grafo de agentes sobre *un* runtime.
**Fuera de alcance:** tesis de cuenca / CL-MPM (otra línea de trabajo), entrenar pesos del LLM, chip físico, consciencia.

El preprint de cuenca no reclama periferia lingüística (opción C). Esto es
ingeniería de producto: hacer que el chat nativo **conteste más rápido**.

---

## 0. Tesis de este ciclo

Gemma 2 2B **ya está entrenado**. Lo que falta no es más gradiente sobre el
Transformer. Lo que falta es **memoria de ejecución**:

> Recordar qué subconjunto de capas (y qué agente) resolvió una clase de
> pregunta, para no volver a encender las 26 en la siguiente.

Dos niveles de «ruta de conexión», el mismo contrato:

| Nivel | Qué se prende | Qué se aprende | Dónde vive |
|---|---|---|---|
| L1 · capas | bits de `LayerExecutionMask` (26 capas) | `huella(pregunta) → máscara` | tabla de rutas, no pesos GGUF |
| L2 · agentes | qué nodo del grafo habla | `huella(pregunta) → agente` | grafo dirigido con pesos de utilidad |

Aprendizaje = estadísticas de rutas (éxito, KL, latencia, refuerzo de aristas).
**Cero** actualización de `QMatMul`. Los pesos permanecen congelados.

Invariante de velocidad:

> El camino feliz ejecuta **un** `forward_with_mask` sobre **una** instancia
> de `QuantizedGemma2`. Ni dos modelos, ni dos prefills densos, ni sueño
> síncrono en el turno.

---

## 1. Qué hay hoy (el nudo)

Inventario que este diseño reutiliza, no reescribe.

### 1.1 Runtime que ya es rápido en lo básico

- `QuantizedGemma2` carga GGUF con Candle. KV por capa (`KvCache` global,
  `RotatingKvCache` en ventana).
- Prefill incremental: segundo turno ~14× menos TTFT si el prefijo coincide
  (0,98 s vs 13,7 s a 256 tokens, evidencia de
  `docs/gemma2_runtime_optimization.md`).
- Decode medido ~10,6–11,1 tok/s en CPU nativo; histórico 2,0–2,7 tok/s.
- `forward_with_mask` **ya salta capas**: `continue` sobre el residual
  (identidad). La máscara no puede cambiar a `position > 0` sin limpiar KV.

### 1.2 Rutas que ya existen y no se usan de día

`adaptive_gemma2` recuerda máscaras. `plan_wake_prefill` las **ignora** si la
KV se puede extender:

```text
si prompt empieza por cached_tokens:
    máscara = máscara_de_la_cache     # casi siempre 26/26
    reutilizar KV
si no:
    máscara = recalled_mask o 26/26
    prefill completo
```

Motivo correcto: cambiar de máscara a mitad de generación invalida la KV.
Efecto: **el producto nunca ahorra capas en un chat real**, porque el segundo
turno siempre extiende. El sueño descubre máscaras; la vigilia no las aplica.
Evidencia histórica: 0 rutas sparse verificadas, fallback 100 %, 26/26 capas.

### 1.3 Presupuesto de skip actual

`AdaptiveGemma2Config`:

- `max_skip_fraction = 0,15` → ~3–4 capas de 26
- `minimum_executed_layers = 8`
- no se saltan capa 0 ni la última
- no se saltan dos adyacentes
- el candidato se elige por `delta_rms` bajo (capa casi identidad)

Gemma 2 alterna atención **local / global**. Saltar una capa global es más
peligroso que una local. Hoy el ranking no distingue.

### 1.4 Por qué no basta «entrenar más»

El trainer infinito y el núcleo CTP aprenden *otra cosa* (relaciones, sesgo
de logits). No apagan capas en decode. Mezclar ese aprendizaje con velocidad
de chat es el error que P3/opción C ya recortó. Esta rama no lo reabre.

---

## 2. Principios

1. **Pesos congelados.** El GGUF es de solo lectura.
2. **Una instancia en RAM.** Varios agentes son políticas, no copias del
   modelo.
3. **Máscara fija por generación.** Se elige *antes* del prefill del turno.
   Nunca en el token 3 del decode.
4. **Fallar hacia denso.** Si la ruta es incierta, 26/26. La corrección es
   un fallback, no un segundo intento silencioso que duplica el coste.
5. **Medir capas, no relatos.** La métrica primaria es
   `tok/s`, `TTFT`, `capas_ejecutadas/26`, `tasa_fallback`, `KL` frente a
   denso en un set fijo. No «calidad subjetiva del 2B».
6. **No quinto binario.** Se extiende `native_gemma2_circadian_chat` y el
   benchmark archivado se reactiva *o* se añade una subflag. P5 sigue
   vigente.
7. **No tocar cuenca.** `consolidation_basin_experiment` y el plan CL-MPM
   viven en `main` / la otra línea. Esta rama no los modifica.

---

## 3. Arquitectura L1 — Layer Route Cache (LRC)

El módulo nuevo de diseño se llama **LRC**. No es un motor termodinámico
nuevo. Es una tabla `(huella → máscara)` con KV *por ruta* o, más barato,
generaciones *por turno* con máscara elegida al inicio.

### 3.1 Modelo de datos

```text
LayerRoute {
    id:                u64
    fingerprint:       ActivationFingerprint   # ya existe (tokens + rms)
    mask:              LayerExecutionMask      # bits, 26 capas
    skip_kind:         MiddleSkip | EarlyExit
    hits:              u64
    misses_as_fallback:u64
    mean_kl:           f32                     # vs logits densos, última posición
    top1_agree:        f32
    mean_decode_toks:  f32
    mean_executed:     u8
    last_generation:   u64
    confidence:        f32                     # hits / (hits+fallback), con prior
}

RouteTable {
    routes:            Vec<LayerRoute>         # tope ~2048 (ya en config)
    default_mask:      LayerExecutionMask::all(26)
    min_confidence:    f32                     # p.ej. 0.55
    max_kl_promote:    f32                     # p.ej. 0.10
}
```

Persistencia: JSON junto a `adaptive/routes.json` o un `layer-routes.json`
aparte. **No** mezclar payload de máscara con cápsulas opacas del router
termodinámico hasta que LRC demuestre ahorro. El `ThermoAssociativeRouter`
actual es demasiado pesado (sustrato CDT) para el hot path de un chat.

Primera implementación: `HashMap` de huella discretizada (top-k tokens del
turno de usuario, sin historial largo) + overlap ≥ 0,35, que es el umbral
que ya usa `recall_working_memory`.

### 3.2 Huella barata (antes de cualquier forward)

Hoy la huella de activación exige un forward con traza RMS. Eso mata el
ahorro: pagas 26 capas para decidir saltar 3.

Contrato nuevo:

```text
fingerprint_wake(user_turn_tokens) =
    top_k tokens del turno actual (no del historial)
    + longitud + hash de los últimos 32 tokens de usuario
```

Costo: tokenizar lo que de todos modos se tokeniza. Cero matmul.

La huella de *activación* (RMS por capa) queda para el **sueño** o para una
sonda 1-de-N, nunca para el lookup del turno.

### 3.3 El cambio de política que desbloquea la velocidad

Hoy: «si puedo reutilizar KV, no cambio la máscara».
Mañana: **la máscara se elige al empezar el turno de usuario**, no al
extender el decode.

Tres políticas de KV, de menor a mayor RAM. Se implementa **KV-A**. KV-B/C
son escaleras si KV-A no da el +20 % tok/s.

#### KV-A — generación por turno (recomendada en CPU 2B)

```text
al llegar el mensaje de usuario:
    route = LRC.lookup(fingerprint_wake(mensaje))
    mask  = route.mask si confidence ≥ umbral else 26/26
    clear_kv_cache()                  # un prefill, máscara ya fija
    prefill(historial + mensaje, mask)
    decode(..., mask)                 # misma máscara, KV coherente
    LRC.observe(route, latency, executed_layers)
```

Se pierde la reutilización KV *entre turnos* cuando la máscara cambia. Se
conserva dentro del decode (el tramo caro: ~10 tok/s × N tokens de
respuesta).

Compensación: si el turno 2+ **repite la misma ruta** (misma máscara), se
puede extender KV exactamente como hoy. El caso frecuente de un chat
temático («sigue en español, misma tarea») reutiliza.

Regla:

```text
si máscara_nueva == máscara_activa y prompt extiende cached_tokens:
    reutilizar KV          # igual que ahora, pero sparse-sparse
si no:
    prefill con máscara_nueva
```

Eso corta el nudo de `plan_wake_prefill` sin inventar un cache dual.

#### KV-B — banco de KV por ruta

Una KV cache por `LayerRoute.id`. RAM ≈ (rutas calientes) × (capas
ejecutadas) × (ctx) × (kv dim). En 2B CPU con ctx 2048, más de 2–3 bancos
calientes se come el ahorro. Sólo si el usuario mantiene *varios* temas en
paralelo.

#### KV-C — denso + sparse en paralelo

Mantener KV 26/26 para fallback instantáneo y KV sparse para el camino
feliz. Duplica memoria. Útil en GPU. No en el portátil CPU del laboratorio.

### 3.4 Dos geometrías de apagado

El código actual **salta capas intermedias** (identidad). Hay otra geometría
más rentable en decode: **salida temprana**.

| Geometría | Qué apaga | KV de las capas apagadas | Encaje con el runtime |
|---|---|---|---|
| **Middle skip** (existe) | 3–4 capas internas de bajo `delta_rms` | no se actualizan; las posteriores sí, con un residual «sin esa capa» | `forward_with_mask` ya lo hace |
| **Early exit** (nuevo) | la cola: capas `k+1 … 25` | no existen | aplicar `norm + lm_head` al hidden de la capa `k` (el head está ligado al embedding, misma `d_model`) |

Early exit ahorra las capas **más caras que quedan** (todo el resto del
stack). Middle skip ahorra un 15 % por diseño conservador.

Orden de prueba:

1. Middle skip **en vigilia** con KV-A (lo que el hardware ya sabe hacer).
2. Medir tok/s y KL.
3. Si el techo de 15 % no llega a +20 % tok/s, subir `max_skip_fraction`
   **solo** sobre capas *locales* (ventana), nunca sobre dos globales
   consecutivas.
4. Early exit como segundo experimento: `k ∈ {12, 16, 20, 23}` en un set
   fijo, KL vs denso. Elegir el menor `k` con KL ≤ 0,10.

No mezclar middle skip y early exit en el mismo turno hasta tener tablas
separadas. Son rutas distintas en LRC (`skip_kind`).

### 3.5 Sonda de calidad sin pagar 2× siempre

Comparar sparse vs denso en **cada** turno duplica el coste y anula el
proyecto.

Política:

```text
con probabilidad p = 1/N  (N=16 al inicio):
    además del forward sparse, un forward denso SÓLO del último token
    (un decode denso de 1 token, no un prefill)
    KL y top-1 se registran en la ruta
si KL > umbral:
    confidence ↓, próxima vez fallback denso
si KL ≤ umbral:
    confidence ↑, se promociona
en sueño (/sleep, no en el turno):
    replay de 2 prompts, misma sonda
```

El decode denso de un token sobre KV **densa** exigiría tener esa KV. Bajo
KV-A no la hay. Alternativas honestas:

- **Sonda de sueño:** replay offline, no duele al usuario. Es el sitio
  correcto para KL.
- **Sonda de turno 1-de-N:** hacer el turno *entero* denso de vez en cuando
  (el usuario paga un turno lento). Más simple, peor UX.
- **Sonda de primer token:** prefill denso vs sparse *solo logits del último
  prompt token* (un prefill extra). Caro.

Recomendación: **sonda en sueño + 1 turno denso de calibración cada N=32**.
No KL en el camino feliz.

Proxy de calidad en caliente, gratis:

- entropía de logits (`output_confidence` ya existe)
- el usuario repite la pregunta / usa `/limpiar` → señal negativa
- el verificador L2 rechaza → señal negativa

### 3.6 Camino de vigilia (secuencia)

```text
tokenizar(mensaje)
huella = fingerprint_wake(tokens_del_turno)
ruta   = LRC.lookup(huella)

si ruta.confidence ≥ τ:
    mask = ruta.mask
sino:
    mask = 26/26
    ruta = None          # fallback, no se finge un hit

si mask == mask_activa AND prefix match:
    prefill sólo sufijo
sino:
    clear KV; prefill historial+mensaje con mask

decode con mask fija
observe: latency, executed_layers, fallback?, proxy_calidad
si el verificador L2 falla:
    UN fallback denso (clear KV, 26/26, regenerar)
    marcar ruta.misses_as_fallback++
    no hay tercer intento
```

Un fallback denso por turno como máximo. Si el denso también falla el
verificador (idioma, «soy un LLM»), se abstiene. No se enciende un segundo
LLM.

### 3.7 Qué se persiste y qué es efímero

| Artefacto | Vida | Ruta de disco |
|---|---|---|
| `LayerRoute` promocionada (KL sueño ok) | lenta | `adaptive/layer-routes.json` |
| hits del turno, buffer de 24 | rápida | `adaptive-state.json` (ya existe el anillo) |
| KV | efímera | RAM, se pierde al cambiar de máscara o `/limpiar` |
| pesos GGUF | inmutables | archivo de modelo |

El journal de Dyamon sigue ignorado por Git.

---

## 4. Arquitectura L2 — grafo de agentes-nodo

Un agente **no es un Gemma**. Es un nodo con rol, presupuesto y una ruta L1
preferida. El grafo aprende *qué nodo debe hablar*, igual que LRC aprende
*qué capas encender*.

### 4.1 Por qué un grafo y no N modelos

En CPU, N×2B no cabe y no es rápido. La red multiagente de este diseño es
**una** red de políticas sobre un único runtime, análoga a Mixture-of-Depths
más Mixture-of-Agents *sin* mezclar pesos.

Más adelante, si hay GPU o varios procesos, un nodo *puede* ser un proceso
con su propio GGUF. El contrato del grafo no cambia: aristas, huella,
utilidad. El transport (in-process vs IPC) es un detalle de la capa de
mailbox.

### 4.2 Nodo

```text
AgentNode {
    id:          NodeId              # u32, estable
    role:        Router | FastTalker | DenseTalker | Verifier | Compiler | Memory
    route_pref:  Option<LayerRouteId>  # L1 por defecto de este agente
    claim:       FingerprintSketch     # de qué preguntas se adueña
    budget:      { max_new_tokens, max_executed_layers, timeout_ms }
    mailbox:     VecDeque<Message>
    stats:       { calls, accepts, rejects, mean_ms }
}
```

`Memory` no genera tokens: es el LRC + historial. `Verifier` no genera
tokens si puede evitarlo: reglas.

### 4.3 Arista

```text
AgentEdge {
    src, dst:    NodeId
    weight:      f32                 # utilidad, no analogía física
    successes:   u32
    failures:    u32
    mean_latency_ms: f32
    last_used:   u64
}
```

Aprendizaje de arista (bandit contextual, no backprop):

```text
reward = -normalized_latency + λ * proxy_calidad - μ * fallback_denso
weight ← (1-α) weight + α * reward
```

`λ, μ, α` versionados en config. Un turno actualiza **una** arista
`Router → hablante` (y `hablante → Verifier` si el verificador corre).

### 4.4 Roles mínimos (no crecer la lista en el primer PR)

| Rol | ¿Forward Gemma? | Capas | Función |
|---|---|---|---|
| **Router** | no | 0 | lookup LRC + lookup grafo; elige un hablante |
| **FastTalker** | sí, 1 vez | máscara sparse L1 | respuesta corta, camino feliz |
| **DenseTalker** | sí, 1 vez | 26/26 | fallback y calibración 1-de-N |
| **Verifier** | no | 0 | español, no «soy un LLM», abstener si no hay receta cuando el modo es compilador |
| **Compiler** | sí, opcional | ruta propia o densa | `gemma_operator_bridge`: texto → receta; ya existe |
| **Memory** | no | 0 | LRC + anillo de experiencias |

Seis nodos. El grafo inicial:

```text
        [Router]
        /      \
[FastTalker]  [DenseTalker]
        \      /
       [Verifier]
           |
        usuario

[Compiler] cuelga del Router sólo si la huella parece receta/QUBO/plan.
[Memory] es consultado por Router, no está en el camino de tokens.
```

### 4.5 Router sin LLM

El router **no** puede ser otro Gemma: duplicaría TTFT.

Implementación prevista:

1. `fingerprint_wake` (ya pagada).
2. Match contra `AgentNode.claim` (overlap de huella, el mismo overlap de
   0,35).
3. Si hay ruta L1 con `confidence ≥ τ` → FastTalker con esa máscara.
4. Si la huella coincide con el DSL de recetas (`{...}` / palabras QUBO) →
   Compiler.
5. Si no → DenseTalker (o FastTalker con máscara all, que es lo mismo).
6. Empate: ε-greedy sobre `AgentEdge.weight`.

Costo objetivo del router: < 1 ms. Si alguien propone «un clasificador
Transformer», se rechaza en esta rama.

### 4.6 Verifier sin LLM

Reglas, en orden, sobre el texto ya generado (barato):

1. ¿Hay caracteres latinos y proporción de español (lista de tokens
   frecuentes + `GEMMA2_FORCED_LANGUAGE`)? Si el output es inglés y el
   system prompt era español → fail.
2. ¿Contiene «I'm a large language model» / «soy un modelo de lenguaje»? →
   fail (el system prompt ya lo prohíbe; esto es el diente).
3. Modo compilador: ¿hay receta parseable? Si no → fail y abstención.
4. Longitud 0 → fail.

Fail → como máximo un DenseTalker. Segundo fail → abstenerse.

No se usa un juez 2B. Un juez 2B es otro forward.

### 4.7 Mailbox y ejecución

Mensaje:

```text
Message {
    from, to: NodeId
    kind:     Query | Reply | Reject | Metric
    tokens:   Option<Vec<u32>>
    text:     Option<String>
    route:    Option<LayerRouteId>
    deadline: Instant
}
```

Ejecución **síncrona** en un hilo (el del chat). No hay runtime actor
asíncrono en v1: el overhead no paga en un único usuario CPU. El mailbox es
una cola en memoria para dejar el contrato listo a IPC.

Paralelismo permitido en v1:

- lookup LRC ∥ lookup grafo (ambos < 1 ms, irrelevante)
- **no** FastTalker ∥ DenseTalker
- el worker fasorial (`gemma_phasor_coupling`) puede seguir en su hilo:
  evidencia previa de overhead nulo sobre tok/s; **no escribe logits**

### 4.8 Aprendizaje del grafo (qué se guarda)

Tras cada turno:

```text
Router → hablante_usado    += reward
hablante → Verifier        += 1 si pass, 0 si fail
Memory.LRC.observe(...)    # L1
```

No se guardan pesos de atención. Se guarda `agent-graph.json`:

```text
{ "nodes": [...], "edges": [...], "generation": u64 }
```

Igual que las rutas de capas: sesionable, `.gitignore` del estado de chat,
un fixture de ejemplo en `data/examples/`.

### 4.9 Extensión futura (no v1)

- Un nodo = proceso con GGUF propio, mailbox por stdin JSON.
- Un nodo = experto LoRA. **Fuera:** contradice pesos congelados y opción C.
- Un nodo = otro modelo (TinyLlama ya está en `ollama-models/`). Tentador
  para el Router. Se evalúa **después** de que el router sin LLM esté
  medido. Si se hace, TinyLlama sólo clasifica (max 8 tokens), nunca habla
  al usuario.
- Consenso entre varios hablantes. Prohibido en v1: N forwards.

---

## 5. Cómo encajan L1 y L2

```text
usuario
  │
  ▼
tokenizar ──────────────────────────────────────────────┐
  │                                                     │
  ▼                                                     │
Router (0 capas)                                        │
  ├─ Memory.LRC.lookup(huella)  → mask, confidence      │
  ├─ grafo.lookup(huella)       → AgentNode             │
  └─ decide hablante                                    │
         │                                              │
         ▼                                              │
   FastTalker | Compiler | DenseTalker                  │
         │                                              │
         ▼                                              │
   QuantizedGemma2.forward_with_mask(mask)   ◄──────────┘
         │
         ▼
   Verifier (0 capas)
         │
         ├─ pass → texto al usuario; reward +
         └─ fail → DenseTalker una vez; si otra vez fail, abstener
         │
         ▼
   LRC.observe + grafo.observe
```

La «red de conexiones que aprende» es literalmente:

- **vertical:** capas 0–25, bits que se prenden juntos porque sirvieron;
- **horizontal:** aristas Router→FastTalker que se refuerzan cuando esa
  pareja (huella, máscara) acertó rápido.

No es un líquido de espines. Es una tabla y un grafo. Los nombres se quedan
en tabla y grafo.

---

## 6. Métricas de hecho (el gate de esta rama)

Set fijo de prompts (español, identidad, seguimiento de turno, receta
mínima, abstención). N=30, semilla versionada. Comparar contra **denso
26/26** en la misma máquina, mismo GGUF.

| Métrica | Cómo se mide | Pasa v1 | Pasa v2 |
|---|---|---|---|
| decode tok/s | benchmark existente, 64 tokens gen. | ≥ +15 % vs denso | ≥ +30 % |
| TTFT turno 1 | prefill historial vacío | no peor que −10 % | — |
| TTFT turno 2+ misma ruta | extensión KV sparse | ≤ TTFT denso actual de extensión | — |
| capas ejecutadas | `trace.executed_layers` | media ≤ 22/26 | ≤ 18/26 |
| tasa fallback denso | turnos DenseTalker / total | ≤ 40 % | ≤ 20 % |
| KL vs denso (sueño) | última posición, replay | mediana ≤ 0,15 | ≤ 0,08 |
| top-1 agree (sueño) | idem | ≥ 0,70 | ≥ 0,85 |
| hits LRC | lookup con confidence ≥ τ | ≥ 20 % tras 50 turnos de calentamiento | ≥ 50 % |
| router | tiempo | < 1 ms p95 | < 1 ms |

Si tok/s sube y KL explota, no se publica la cifra de velocidad. El par
(tok/s, KL) se imprime junto, como la tabla de cuenca imprime pre/post.

Baseline obligatorio en el mismo CSV: denso 26/26. Sin él no hay ventaja.

---

## 7. Cómo implementarlo (sin escribirlo ahora)

Orden de PRs. Cada uno es independiente y medible. Ninguno toca cuenca.

### PR-V1 — Política KV-A en vigilia

**Qué:** cambiar `plan_wake_prefill` para que una máscara *recordada* gane
incluso si hay KV, **cuando la máscara coincide**; y para que un cambio de
máscara limpie KV y re-prefill en lugar de ignorar la ruta.

**Dónde:** `src/adaptive_gemma2.rs` (`plan_wake_prefill`, `prepare_forward`),
tests que ya cubren «el segundo turno no aplica sparse». Esos tests se
*actualizan* al contrato nuevo, no se borran: el segundo turno *sí* aplica
sparse si la ruta es la misma; si la ruta cambia, se documenta el clear KV.

**No hacer:** descubrir máscaras nuevas. Usar `recalled_mask` que ya existe.

**Gate:** un test de unidad sin GGUF (máscaras sintéticas) + el test GGUF
hoy `ignored` se corre a mano una vez.

### PR-V2 — LRC de huella barata

**Qué:** tabla `huella_de_turno → LayerRoute` independiente del sustrato
CDT. Lookup < 1 ms. Persistencia `layer-routes.json`.

**Dónde:** módulo nuevo `src/layer_route_cache.rs`. Lo consume
`AdaptiveThermoMemory::recall` *antes* del router termodinámico. Si LRC
hit, no se toca CDT.

**Gate:** tests de lookup/overlap/confianza; el camino sin hit es idéntico
al denso.

### PR-V3 — Middle skip en el chat circadiano

**Qué:** el chat usa LRC+KV-A. Banner: capas ejecutadas por turno.
Métrica en el log: `layers=20/26 route=hit|miss tok/s=…`.

**Dónde:** `src/bin/native_gemma2_circadian_chat.rs`. No hay binario nuevo.

**Gate:** 10 turnos sintéticos (prompts fijos) con y sin LRC, CSV.

### PR-V4 — Sonda de sueño y promoción

**Qué:** `/sueño` compara logits último-token sparse vs denso en ≤ 2
prompts. Promueve a LRC si KL ≤ umbral. No spin-gate, no PEPS, no VMC.
El umbral 0,92 actual es el motivo de «0 rutas». Se sustituye por KL.

**Dónde:** `consolidate_sleep_with_model` / `discover_sleep_masks`. Se deja
de exigir `min_verified_quality=0.92` *y* spin-gate para *promover a LRC*.
El spin-gate puede quedarse como feature `research`.

**Gate:** en un fixture, una máscara de 3 skips con KL bajo se promociona;
una máscara absurda (apagar 20 capas) no.

### PR-V5 — Capas locales vs globales

**Qué:** el ranking de skip prefiere capas de *ventana deslizante*.
Prohibido saltar dos capas globales consecutivas. Subir
`max_skip_fraction` a 0,25 *solo* si el gate de KL de PR-V4 sigue verde.

**Dónde:** `conservative_candidate_mask`. El modelo ya sabe qué capa es
sliding (`layer.sliding_window`).

### PR-V6 — Grafo de 6 nodos, router sin LLM

**Qué:** `src/agent_graph.rs`. Router = lookup. FastTalker/DenseTalker
delegan en el runtime. Verifier = reglas. JSON de aristas.

**Dónde:** el chat construye el grafo al arrancar. Un turno = un recorrido
Router→hablante→Verifier.

**Gate:** test sin GGUF del router (huellas sintéticas) y del verifier
(strings). Un test de integración opcional ignored-GGUF.

### PR-V7 — Early exit (sólo si V3–V5 no llegan a +15 % tok/s)

**Qué:** `forward_with_mask` acepta `exit_after: Option<usize>`. Aplica
`norm+lm_head` al hidden de esa capa. Rutas LRC con `skip_kind=EarlyExit`.

**Gate:** tabla k=12,16,20,23 vs denso. Se elige un k o se abandona.

### PR-V8 — Benchmark de la rama

**Qué:** reactivar el binario de benchmark *desde archive* o una flag
`--bench-routes` en el chat. CSV: tok/s, TTFT, capas, fallback, KL sueño.

**No:** un «motor unificado» nuevo.

---

## 8. Mapa de archivos previstos (cuando toque código)

| Archivo | Rol |
|---|---|
| `src/layer_route_cache.rs` | L1, nuevo |
| `src/agent_graph.rs` | L2, nuevo |
| `src/adaptive_gemma2.rs` | KV-A, dejar de ignorar máscaras en vigilia |
| `src/native_gemma2.rs` | early exit opcional (PR-V7); skip ya está |
| `src/bin/native_gemma2_circadian_chat.rs` | cablear Router→hablante→Verifier |
| `src/native_gemma2_runtime.rs` | system prompt intacto; métricas de capas en el log |
| `docs/gemma2_runtime_optimization.md` | apéndice de cifras de esta rama, no reescribir historia |

`lib.rs` exporta los dos módulos nuevos sin feature flag: son runtime de
chat, no investigación de cuenca. No van a `research`.

---

## 9. Decisiones clave

1. **No se entrena Gemma.** La «inteligencia» nueva es memoria de ejecución.
2. **Un solo GGUF en RAM.** Agente ≠ modelo.
3. **KV-A (máscara por turno, clear si cambia).** Es el menor cambio que
   permite skip en decode, que es el tramo lento.
4. **Huella de lookup sin forward.** Huella de activación sólo en sueño.
5. **KL en sueño sustituye al umbral 0,92 + spin-gate** para promover rutas.
   Ese umbral es la causa de 0 rutas.
6. **Router y verifier sin LLM.** El camino feliz es un forward, no tres.
7. **Fallback denso como máximo una vez por turno.**
8. **Middle skip primero, early exit después.** El primero ya está en el
   kernel; el segundo es más ahorro potencial y más riesgo de calidad.
9. **Esta rama no toca cuenca ni el trainer gated.**
10. **No quinto binario.** Chat + métricas.

---

## 10. Qué no hacer

- No lanzar el trainer infinito «para que aprenda capas».
- No aplicar máscara distinta en el token 2 del decode.
- No verificar cada turno con un prefill denso.
- No crear un agente por capa (26 agentes × mailbox). El agente es un rol,
  la granularidad fina es L1.
- No llamar «EPR» o «CDT» a una `HashMap` de máscaras.
- No mezclar `wake_blend` CTP en este diseño: es otro eje (calidad de
  logits, no de capas). Si se toca, es un PR aparte con A/B, no esta tesis.
- No introducir LoRA, adapters ni fine-tune del 2B.
- No consenso multi-hablante en v1.
- No fusionar a `main` sin la tabla (tok/s, KL, capas, fallback).

---

## 11. Relación con el resto del laboratorio

| Línea | Rama / doc | Relación |
|---|---|---|
| Cuenca y memoria de *K* patrones | `main`, `docs/arquitectura_siguiente_ciclo.md` | ortogonal; no compartir PRs |
| P0–P6 higiene | `docs/revision_proyecto.md` | se respeta (4 binarios, opción C) |
| Runtime Gemma ya medido | `docs/gemma2_runtime_optimization.md` | este doc es la continuación de sus «límites pendientes»: *aplicar máscaras en decode incremental* |
| Chat circadiano | `native_gemma2_circadian_chat` | único punto de entrada a cablear |

Frase para un extraño:

> El laboratorio de cuenca pregunta si escribir un patrón deforma un
> paisaje. Esta rama pregunta si el chat nativo puede contestar con 20
> capas en vez de 26 y recordar ese atajo la próxima vez, con un router
> de agentes que no es otro modelo.

---

## 12. Preguntas abiertas (no bloquean el diseño v1)

1. ¿N=16 o N=32 para el turno denso de calibración? Default N=32.
2. ¿Early exit se evalúa si middle skip ya da +15 %? Default: no, se aparca.
3. ¿TinyLlama como router, después de medir el router-hash? Default: no en
   v1.
4. ¿Rehidratar historial con la máscara nueva (prefill largo) o recortar a
   los últimos 2 turnos al cambiar de ruta? Default: recortar a 2 turnos
   para proteger TTFT; el system prompt siempre se reinyecta
   (`gemma2_system_prefix`).

Cualquier respuesta distinta se anota aquí antes de escribir código.

---

## 13. Criterio de cierre de la rama

La rama está lista para fusionar cuando:

1. existe la tabla del §6 en un CSV versionado o pegado en
   `docs/gemma2_runtime_optimization.md`;
2. el camino feliz es 1 forward;
3. LRC tiene hits > 0 en una sesión de 50 turnos del set fijo;
4. el grafo tiene 6 nodos y el router no llama a Gemma;
5. `cargo test --release --lib -- --skip scientific` sigue verde;
6. no se ha modificado el experimento de cuenca.

Hasta entonces, este archivo es el mapa. No el código.
