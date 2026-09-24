# Propuesta — Mejoras y experimentos · ciclo siguiente

**Rama:** `exp/mejoras-siguiente-ciclo`  
**Base:** `origin/main` @ `e05cef1` (PR #26 · E8–E30 + Clean-Room v2 + UI agentica)  
**Fecha:** 2026-09-23 (America/Mexico_City)  
**Alcance:** plan serio + roadmap; **no** es un merge a main ni un claim científico nuevo.  
**Calidad de razonamiento:** brief arquitecto-IA (honestidad > narrativa; métricas antes que intuición).

---

## 0. Resumen ejecutivo (1 página)

Main ya productizó el stack **líquido (hot-path) → CDT (post-sueño) → RQM (índice/fallback) → Gemma (solo periferia)** con UI Chat / Entrenamiento / Sueño / Pruebas e Infinito. Clean-Room DEV (`0xA300–0xA307`) cerró con **leakage = 0** y un histograma mixto: muchos NULL/PARTIAL, E21/E22 claramente débiles, FASE_A POSITIVE.

Este ciclo **no** inventa un motor nuevo ni revive gas-core. Prioriza:

1. **Cerrar el cuello de generalización de regla** (E21 extrapolación, E22 composición RQM-OFF).
2. **Endurecer E13 holdout** (seen ya ok; unseen/transitividad abierta) y **consolidar E15** a 8 semillas.
3. **Producto UI/API** que separe smoke vs bench, haga honestos los paneles, y no mezcle chat con train.
4. **Diferidos protocolarios** solo cuando el DEV esté listo: confirmation `0xB300–0xB30F`, E30 OS restart real, Gemma generative datasets.

**Éxito del ciclo (falsable):** ≥1 de E21/E22 pasa de NEGATIVE-dominante a PARTIAL estable (≥6/8 seeds) **sin** leakage>0 ni RQM-ON en el brazo de autonomía; E13 unseen ≥0.50 en ≥4/8 seeds **sin** insertar pares holdout; UI smoke <30 s y suite DEV documentada aparte.

---

## 1. Diagnóstico breve

### 1.1 Qué ya está en main (`e05cef1`)

| Bloque | Estado en tip |
|---|---|
| E8–E10 líquido | Integrados; protocolos en `docs/experimentos_8_9_10_liquido.md` |
| E11–E17 campo entrenable | Históricos + hallazgos E13/E15 WIP |
| E18–E30 Clean-Room v2 | Harness `field_autonomy_stage2_v2.rs`; DEV 8 seeds; hyperparams LOCKED |
| Anti-contaminación | Auditor hard; manifests sellados; leakage=0 en corrida documentada |
| UI agentica | `agentic_web` · Chat / Train / Sleep / Tests · Infinito · decoder-only |
| Arquitectura productizada | Líquido infer · CDT sueño · RQM índice/fallback · Gemma periferia |

### 1.2 Clean-Room DEV — números conocidos (no inventar)

Fuente: `docs/resultados_etapa_2_cleanroom_v2.md` (wall ~12.1 s, 120 filas).

| Agregado | Valor |
|---|---|
| PARTIAL | 58 |
| NULL | 46 |
| POSITIVE | 8 (FASE_A_cleanroom_field_only ×8) |
| NEGATIVE | 8 (sobre todo E21) |
| leakage>0 | **0** |
| Confirmation `0xB300–0xB30F` | **no corridas** |

Dolores claros por experimento:

- **E21** — NEGATIVE:6, NULL:2 → interpolación ≠ extrapolación; el campo no generaliza parámetro OOD.
- **E22** — NEGATIVE:2, NULL:2, PARTIAL:4 → composición débil con RQM OFF (el brazo que importa).
- **E13 (histórico)** — PARTIAL: dyn seen ok tras endurecimiento; **unseen ~0.33** (holdout `lobo→animal`, `águila→ser-vivo`).
- **E15** — smoke endurecido PASS en 1 seed; **suite 8 semillas pendiente** de re-lock.
- **E30** — PARTIAL ×8 in-process; **OS restart real diferido**.
- **E18/E19/E20/E23/E28/E29** — PARTIAL dominante: hay señal, no cierre.

### 1.3 Producto UI — dolores de ingeniería (no claims)

- Chat / Train / Sleep / Tests existen; falta **contrato explícito smoke vs bench largo** en UI y docs de operador.
- Train Infinito escribe checkpoints por dataset; falta **telemetría de anti-leakage** visible (leakage score, FIELD_ONLY flag, seed family).
- Tests evalúan fuse ya entrenado; no sustituyen Clean-Room DEV ni confirmation.
- Gemma generative datasets: hoy curriculum ampliado + encode; **no** hay benchmark lingüístico Stage-2 post-math.

### 1.4 Contexto de arquitectura (no reabrir tesis)

`docs/arquitectura_siguiente_ciclo.md` fija memoria *K* patrones, ΔR gate, curva vs Hopfield — eso es tesis v2 del preprint. Este ciclo experimental **no** la reemplaza; opera en el laboratorio de campo/UI ya landed. Cualquier puente a tesis v2 va por P7–P12, no por cherry-pick de E21.

---

## 2. Principios (inviolables)

1. **Anti-contaminación Clean-Room §29+ prevalece** para cualquier corrida E18+.
2. **No cherry-pick de seeds** por rendimiento; DEV = `0xA300–0xA307`; confirmation = `0xB300–0xB30F` (intocables hasta lock).
3. **Field-only en evals de autonomía:** RQM / NN / table / attractor OFF en el brazo que se reporta como evidencia de D_φ.
4. **UI smoke ≠ bench largo:** smoke <30–60 s, 1 seed o subset; suite DEV/confirm en CLI/`--release` documentada, no en botón de “Pruebas” por defecto.
5. **No retune post-TEST;** hyperparams LOCKED hasta nueva fase TRAIN→DEV explícita.
6. **Gemma solo periferia:** encode/decode/dataset gen; nunca “Gemma inventa el target” en math benches.
7. **Gas-core eliminado a propósito — NO revivir.**
8. **No inventar resultados:** plantilla de resultados vacía hasta corrida real; CSV con provenance.
9. **Docs-first si el diseño no está cerrado;** stub de código solo si no rompe `cargo check`.
10. **Push solo a esta rama experimental;** nunca force-push; nunca merge a main desde este brief.

---

## 3. Mejoras de producto (UI / API / train / sleep / tests / Gemma)

Prioridad = impacto × riesgo × esfuerzo. Criterio de éxito = medible en una sesión de operador.

### P0 — Contrato smoke vs bench + paneles honestos

| Ítem | Qué | Criterio de éxito | Esfuerzo |
|---|---|---|---|
| **P0.1** Etiquetas UI | Cada acción Train/Tests marca `mode=smoke\|dev\|confirm` y seed family | Operador ve badge; smoke no lanza `0xB3xx` | S |
| **P0.2** API `/api/tests/run` | Default = smoke fuse (identidad, latencia líquido, recall); flag `suite=stage2_v2_dev` solo CLI/docs | Smoke <30 s; DEV no bloquea event-loop Axum | M |
| **P0.3** Telemetría anti-leak | Status/events incluyen `leakage_score`, `field_only`, `rqm_eval=off` | Visible en panel Entrenamiento y CSV de job | S |
| **P0.4** Separación Chat↔Train | Chat no escribe ejemplos a dataset; UI deja explícito “decoder only” | Test de regresión web:: ya existente + assert en UI copy | S |

### P1 — Train Infinito operable y Sueño auditable

| Ítem | Qué | Criterio de éxito | Esfuerzo |
|---|---|---|---|
| **P1.1** Checkpoint schema v2 | Campos: seed_family, hyper_lock_id, encoder_init_hash, leakage | `latest.json` valida schema; jobs viejos marcados legacy | M |
| **P1.2** Stop + drain | Detener cancela batch en curso y flushea evento `cancelled` | No orphan jobs; status `idle` <2 s tras stop | S |
| **P1.3** Sueño report | UI muestra F, simetría, handshake, engramas Δ, RQM pods | Coincide con `sleep_consolidate` report bit a bit | S |
| **P1.4** Dataset periphery tag | `source_tag` obligatorio (`lexicon_synth` / `gemma_gen` / `numeric_cleanroom`) | Filtro en Tests por tag | S |

### P2 — Gemma periferia y DX

| Ítem | Qué | Criterio de éxito | Esfuerzo |
|---|---|---|---|
| **P2.1** Lexicon-first Railway | Deploy sin GGUF sigue siendo default; GGUF opcional volumen | `/health` documenta `llm_mode` | S (ya casi) |
| **P2.2** Generative datasets (prep) | Diseño de curriculum Stage-2 lingüístico **después** de math lock | Spec en docs; **no** claim hasta bench | M (docs) |
| **P2.3** DX operador | README + deploy: tabla “qué botón ≠ qué suite científica” | Extraño no confunde Pruebas UI con Clean-Room | S |

---

## 4. Experimentos nuevos y endurecimientos (E31+ / E13–E30)

Convención de veredicto (igual Clean-Room):

- **PASS** — criterio primario + leakage=0 + controls no explican el lift.
- **PARTIAL** — señal direccional, no cierra; documentar gap.
- **FAIL / NEGATIVE** — hipótesis rechazada o peor que control.
- **NULL** — no discrimina (ruido / techo / infra).

Seeds reservadas (propuesta; **no usar confirmation**):

| Familia | Rango | Uso |
|---|---|---|
| DEV Clean-Room (lock) | `0xA300–0xA307` | Solo re-run si hyperparams sin retune; preferir no tocar |
| DEV ciclo siguiente | `0xA310–0xA31F` | Endurecimientos E21/E22/E30-OS / E31+ en DEV |
| Confirmación (intocable) | `0xB300–0xB30F` | Solo tras lock explícito TRAIN→DEV→LOCK |
| E13/E15 histórico | `0xE1100–0xE1107` | Re-run 8 semillas endurecimiento |
| Smoke UI | `0x51D0_0001` | 1-shot UX; nunca en papers |

### 4.1 Endurecimiento E13 — holdout is-a (prioridad científica alta)

| Campo | Contenido |
|---|---|
| **Hipótesis** | Scores jerárquicos multi-label + R residual por rama + soft parent `mamífero→animal` suben unseen sin leakage de pares holdout |
| **Método** | Partir de hallazgos `docs/hallazgos_e13_e15_endurecimiento.md`; ablation: −residual / −animal_ball / −tax_ancestors; eval `(cue,label)` con ancestros; **prohibido** soft-target en cues holdout |
| **Métricas** | `D_dynamic` seen/unseen; holdout bit (`lobo→animal`, `lobo→mamífero`, `águila→ser-vivo`); leakage; RQM/table holdout (=0 esperado) |
| **Seeds** | `0xE1100–0xE1107` (8); smoke previo `0xE1100` ya documentado |
| **PASS** | unseen ≥0.50 **y** ≥2/3 holdout bits en ≥6/8 seeds; leakage=0 |
| **PARTIAL** | seen≥0.9 y unseen≥0.40 en ≥4/8 **o** 1 holdout bit nuevo estable |
| **FAIL** | unseen≤0.33 media **o** leakage>0 **o** lift solo con RQM-ON |
| **Riesgo leakage** | Medio (jerarquía puede filtrar label); mitigar con auditor de pares y no soft en holdout cues |
| **Esfuerzo** | M–L |

### 4.2 Consolidar E15 — composición factorial (prioridad media)

| Campo | Contenido |
|---|---|
| **Hipótesis** | Compose+Dφ+manifold mantiene structure 3/3 y rank≥0.75 en 8 semillas |
| **Método** | Re-run suite; opcional holdouts sujeto/verbo OOD; periferia léxico (GGUF solo si protocolo lo pide) |
| **Métricas** | structure, rank_acc, margin_E, d_man, stab |
| **Seeds** | `0xE1100–0xE1107` |
| **PASS** | structure 3/3 y rank≥0.70 en ≥6/8 |
| **PARTIAL** | PASS en smoke pero <6/8 en suite |
| **FAIL** | margin_E≈0 o structure <2/3 media |
| **Riesgo leakage** | Bajo |
| **Esfuerzo** | S–M |

### 4.3 Endurecimiento E21 — extrapolación de parámetro (P0 científico Stage-2)

| Campo | Contenido |
|---|---|
| **Hipótesis** | Separar loss/currículo interp vs extrap + curriculum de parámetros densos en frontera + regularización de D_φ en dirección del parámetro reduce NEGATIVE sin lookup |
| **Método** | Mantener TRAIN dx∈{−2..2} / TEST dx=3 (y análogos θ, scale); **añadir** métricas separadas interp/extrap; curriculum “borde” (dx=±2) con más peso; ablations: STATIC, linear, RQM-ON (control, no claim), table |
| **Métricas** | accuracy/cos interp; accuracy/cos extrap; Δ vs STATIC; leakage; sample efficiency |
| **Seeds** | DEV ciclo `0xA310–0xA317` (8); **no** retocar lock A300 salvo A/B documentado |
| **PASS** | extrap ≥ interp−0.15 **y** extrap > STATIC+0.20 en ≥6/8; leakage=0; RQM-OFF |
| **PARTIAL** | extrap mejora vs baseline actual pero <6/8 o solo en 1 familia de regla |
| **FAIL** | extrap≈azar **o** lift solo con RQM/table |
| **Riesgo leakage** | Medio-alto (generador debe auditar que targets TEST no estén en CDT/train) |
| **Esfuerzo** | L |

### 4.4 Endurecimiento E22 — composición RQM-OFF (P0 científico Stage-2)

| Campo | Contenido |
|---|---|
| **Hipótesis** | Entrenar T1, T2 con rollout corto de composición **sintética no-test** (T2∘T1 sobre estados train-only) + loss de conmutador aproximado mejora PARTIAL→PASS sin ver T2(T1(x_test)) |
| **Método** | Prohibido entrenar compuesto de estados TEST; permitido compose sobre órbita train auditada; comparar D_φ RQM-OFF vs STATIC / linear / RQM compose (control) |
| **Métricas** | cos/err compose; ranking vs controles; leakage; h∈{1,2} stability |
| **Seeds** | `0xA318–0xA31F` |
| **PASS** | D_φ RQM-OFF > todos controles no-oráculo en ≥6/8; leakage=0 |
| **PARTIAL** | Gana a STATIC pero no a linear **o** 4–5/8 |
| **FAIL** | RQM compose es el único brazo bueno **atribuido** a autonomía |
| **Riesgo leakage** | Alto (compose train puede acercarse a test orbit); auditor same_orbit obligatorio |
| **Esfuerzo** | L |

### 4.5 E30-OS — persistencia con reinicio de proceso real

| Campo | Contenido |
|---|---|
| **Hipótesis** | Serialize encoder+D_φ+CDT → exit proceso → reload → novel-state match in-process ±ε |
| **Método** | Example/binario hijo `stage2_e30_os_restart`; ablaciones P0–P4 del protocolo §43; **no** contar in-process como OS |
| **Métricas** | Δ métrica pre/post restart; wall; integridad hash checkpoint |
| **Seeds** | `0xA320–0xA323` (4 smoke) luego 8 |
| **PASS** | P0 ≥ in-process−ε; P1–P4 discriminan dónde vive la info |
| **PARTIAL** | Reload ok pero ε grande o solo P0 |
| **FAIL** | P3 (solo CDT retrieval) explica todo el lift |
| **Riesgo leakage** | Bajo (infra) |
| **Esfuerzo** | M |

### 4.6 E31 — Sample-efficiency de experiencia consolidada (nuevo)

| Campo | Contenido |
|---|---|
| **Hipótesis** | Con presupuesto N∈{4,8,16,32} episodios, Dynamic+CDT alcanza umbral de regla con menos N que Dynamic sin CDT (pareado) |
| **Método** | Extensión E25/E26; curvas N vs rule_gen; bootstrap CI; **FIELD_ONLY** |
| **Métricas** | N* para alcanzar thresh; AUC; paired Δ |
| **Seeds** | `0xA330–0xA337` |
| **PASS** | N*_CDT ≤ 0.5·N*_noCDT en ≥6/8 **y** controls corrupt/irrelevant no lo replican |
| **PARTIAL** | Tendencia en 4–5/8 |
| **FAIL** | Curvas solapan CI |
| **Riesgo leakage** | Medio |
| **Esfuerzo** | M |

### 4.7 E32 — Estabilidad de rollout largo con recuperación (nuevo)

| Campo | Contenido |
|---|---|
| **Hipótesis** | Tras E23 PARTIAL, perturbación a h=16 + un paso de “re-anclaje” por experiencia consolidada (no teacher forcing de estado true) reduce error a h=64 |
| **Método** | Rollout 1…64; branch con/sin re-anclaje CDT-experience; **sin** insertar x_true |
| **Métricas** | err(h), cos(h), recovery_gain |
| **Seeds** | `0xA338–0xA33F` |
| **PASS** | recovery_gain>0 estable ≥6/8; leakage=0 |
| **PARTIAL** | Gain solo a h≤32 |
| **FAIL** | Re-anclaje ≡ teacher forcing disfrazado (auditor) |
| **Riesgo leakage** | Alto (definir re-anclaje sin target) |
| **Esfuerzo** | L |

### 4.8 E33 — UI↔ciencia bridge (smoke de producto, no paper)

| Campo | Contenido |
|---|---|
| **Hipótesis** | Un operador puede: train smoke 2 batches → sleep → tests smoke y obtener JSON con `field_only`, latencia líquido p50/p95, 0 leakage probes |
| **Método** | Script/`cargo test --features web` + checklist manual; **no** sustituye Clean-Room |
| **Métricas** | tiempo total; asserts schema; no panic Infinito stop |
| **Seeds** | `0x51D0_0001` |
| **PASS** | Checklist verde en CI web smoke |
| **PARTIAL** | Manual ok, CI flaky |
| **FAIL** | Chat contamina train **o** tests lanzan suite DEV |
| **Riesgo leakage** | N/A producto |
| **Esfuerzo** | S–M |

### 4.9 Confirmation suite (diferido gate)

Correr `0xB300–0xB30F` **solo** cuando:

1. Hyperparams LOCKED sin retune post-DEV de este ciclo.
2. E21 o E22 alcanzaron PARTIAL estable en `0xA31x`.
3. Manifests sellados y docs de resultados rellenados (no plantilla).

Hasta entonces: **reservadas / no corridas**.

---

## 5. Roadmap 2–3 sprints (solo esta rama)

### Sprint 1 — Instrumentación + cuellos E21/E22 (docs→código mínimo)

1. Land esta propuesta + plantilla resultados + README pointer (**este commit**).
2. P0.1–P0.4 (badges, API default smoke, telemetría leakage, copy Chat).
3. Diseño detallado E21/E22 (generador + auditor same_orbit) en follow-up doc o sección changelog de resultados.
4. Smoke E33 checklist (tests web existentes + gaps).
5. **No** tocar confirmation; **no** merge main.

**Exit Sprint 1:** operador no puede confundir Pruebas UI con Clean-Room; specs E21/E22 listos para implementar.

### Sprint 2 — Implementar endurecimientos científicos

1. E21 curriculum borde + métricas interp/extrap separadas (`0xA310–0xA317`).
2. E22 compose-train auditado RQM-OFF (`0xA318–0xA31F`).
3. E13 re-run 8 semillas + ablations; E15 suite 8 semillas.
4. Rellenar `docs/resultados_ciclo_siguiente.md` + CSV **solo con corridas reales**.

**Exit Sprint 2:** ≥1 de {E21, E22} en PARTIAL estable **o** informe negativo honesto; E15 consolidado o gap claro.

### Sprint 3 — Persistencia, eficiencia, gate confirmation

1. E30-OS restart real.
2. E31 sample-efficiency (si Sprint 2 no fue negativo total).
3. E32 solo si E23/E22 dan base.
4. Decisión go/no-go confirmation `0xB300–0xB30F`.
5. P2.2 spec Gemma generative datasets (docs); implementación bench solo post-math.

**Exit Sprint 3:** informe go/no-go a main (humano decide merge); confirmation corrida o explícitamente aplazada.

---

## 6. Qué NO hacer

- **NO** merge a `main` desde esta rama sin decisión humana + CI verde + resultados no inventados.
- **NO** force-push; **NO** reescribir historia de `e05cef1`.
- **NO** revivir **gas-core**.
- **NO** cherry-pick seeds ni mirar `0xB3xx` “por curiosidad”.
- **NO** atribuir a D_φ un lift de RQM/table/NN/attractor.
- **NO** meter targets de TEST en CDT, prototypes, soft-labels o animal-ball.
- **NO** retune hyperparams mirando TEST o confirmation.
- **NO** lanzar suite DEV/confirm desde el botón Pruebas por defecto.
- **NO** poner Gemma a “salvar” math benches (§44 etapa 2).
- **NO** citar 100 % post-sueño / Hopfield empate como ventaja de autonomía de campo.
- **NO** abrir otro motor unificado / quinto binario de tesis; reutilizar `agentic_web` + examples.
- **NO** mezclar CSV pre-cleanroom (`0xE1800..`) con v2 (`0xA3xx`).
- **NO** declarar PASS con 1 smoke seed (vale para E13/E15).

---

## 7. Matriz de prioridad (top)

| # | Ítem | Tipo | Pri | Por qué ahora |
|---|---|---|---|---|
| 1 | E21 extrapolación | Exp | P0 | Único bloque NEGATIVE-dominante del Clean-Room |
| 2 | E22 composición RQM-OFF | Exp | P0 | Núcleo de “regla ∘ regla” sin lookup |
| 3 | UI smoke≠bench + leakage telemetry | Prod | P0 | Evita falsos claims desde producto |
| 4 | E13 holdout 8 semillas | Exp | P1 | Seen resuelto; unseen es el claim abierto |
| 5 | E15 suite 8 semillas | Exp | P1 | Smoke PASS; falta estadística |
| 6 | E30-OS restart | Exp | P1 | Cierra diferido de persistencia real |
| 7 | E31 sample-efficiency | Exp | P2 | Fortalece E25/E26 si E21/E22 no mueren |
| 8 | Confirmation `0xB300–0xB30F` | Gate | P2 | Solo post-lock; no antes |

---

## 8. Referencias (tip main)

- `docs/resultados_etapa_2_cleanroom_v2.md`
- `docs/etapa_2_autonomia_campo_experimentos.md` (§26 Qué NO, §29+ Clean-Room)
- `docs/hallazgos_e13_e15_endurecimiento.md`
- `docs/arquitectura_siguiente_ciclo.md`
- `docs/deploy_railway_agentic.md`
- README · Etapa 2 / UI

---

## 9. Estado de este documento

| Artefacto | Estado |
|---|---|
| Esta propuesta | **Landed** en `exp/mejoras-siguiente-ciclo` |
| Resultados numéricos del ciclo | Plantilla + sección implementado/pendiente → `docs/resultados_ciclo_siguiente.md` |
| Código P0 producto + E21/E22 harden | **Implementado** (corridas PASS pendientes) |
| Merge a main | **No** |

Cuando existan corridas: llenar la plantilla, adjuntar CSV, y enlazar SHAs de harness. Hasta entonces, cualquier cifra fuera de Clean-Room DEV / hallazgos E13-E15 citados arriba es **inválida**.
