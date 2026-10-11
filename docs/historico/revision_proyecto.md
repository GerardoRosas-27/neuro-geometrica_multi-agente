# Revisión del proyecto CDT-RQM-EPR

**Fecha:** 30 de agosto de 2026
**Alcance:** working tree de `main` (`770e4d9`, `optimizacion de respuesta`), tracking `origin/main`
**Cambio local no versionado al leer:** `data/native_gemma2_circadian/dyamon/wake/journal.jsonl`
**Autor de la revisión:** Grok 4.6 (xAI), lectura directa del checkout

Este documento resume lo leído, el diagnóstico y un plan de mejoras. No modifica código. No afirma que las hipótesis físicas o la cognición general estén demostradas.

---

## 1. Qué se leyó

### 1.1 Documentos

- `README.md`
- `Cargo.toml`
- `src/lib.rs`
- `.github/workflows/ci.yml`
- `docs/paper.md`
- `docs/paper_inferencia_fasorial_consolidacion_cdt.md` (preprint 0.5, 19 ago 2026)
- `docs/unified_spin_cognitive_engine.md`
- `docs/gemma2_runtime_optimization.md`

### 1.2 Código de referencia

- `src/consolidation_basin_experiment.rs`
- `src/cognitive_generalization_benchmark.rs`
- `src/emergent_cognition_training.rs`
- `src/unified_spin_cognitive_engine.rs`
- inventario de `src/bin/` (41 binarios)

### 1.3 Artefactos de datos

- `data/gemma2_developmental_infinite_training/summary.json`
- `data/native_gemma2_future_infinite_training/latest.json`
- `data/native_gemma2_circadian/dyamon/wake/journal.jsonl`

No se ejecutaron tests ni binarios. No se leyeron las ramas `feature/*`; el usuario indicó que el trabajo vigente está en `main`.

---

## 2. Qué es el proyecto

Laboratorio nativo en Rust (`cdt_rqm_epr`) de un sistema de **dos tiempos**:

1. **Vigilia:** inferencia por relajación de un campo de fasores (mínimo de energía libre `F = U − T·S`).
2. **Sueño:** consolidación selectiva y transaccional sobre un sustrato tipo CDT.

Gemma 2 2B (GGUF/Candle) se declara periferia lingüística: pesos congelados, un prefill, sesgo CTP `wake_blend = 0.25`. El aprendizaje de máscaras y del núcleo CTP ocurre de noche.

Arquitectura unificada documentada:

```text
CDT simplicial pyrochlore
  -> líquido de espines XXZ
  -> RQM de amplitud/fase
  -> EPR predictivo
  -> capa cognitiva relacional
  -> gate de conocimiento consolidado
```

Los manuscritos acotan bien el alcance: no conciencia, no julios, no generalización conceptual, tres niveles de evidencia (interna / algorítmica / física).

---

## 3. Diagnóstico

### 3.1 Fortalezas

- Ingeniería real: crate nativo, runtime Gemma 2, minimizador Armijo, sueño transaccional, CI (`fmt`, `clippy -D warnings`, `cargo test --release --lib`).
- Honestidad científica rara: resultados negativos publicados (0 rutas sparse, fallback 100 %), niveles de evidencia, disclaimers en README y papers.
- El experimento de cuenca es el núcleo defendible: pre/post, misma cue y solver, Z₂ como fallo, acoplamiento nulo sólo en adquisición, ocho semillas, gate versionado en config.
- Protocolos con ablación y lesión (simetría, CDT) en el ciclo causal de `emergent_cognition_training`.
- El paper de inferencia fasorial (`docs/paper_inferencia_fasorial_consolidacion_cdt.md`) es el candidato a preprint; está más limpio que `docs/paper.md`.

### 3.2 Problema central

Casi toda la “cognición” vive en **fixtures sintéticos con techo del 100 %**.

Memoria exacta, composición A→B→C, transferencia isomórfica, abstención OOD, descubrimiento de familia: 100 %. Cuando todo pasa, el protocolo y el motor se diseñaron juntos. La órbita se inyecta, el catálogo de transformaciones está predefinido, el patrón de la cuenca se escribe ya verificado. Eso es composición de grafo más escritura de atractor, no cognición emergente.

Los papers lo dicen. El volumen de binarios, nombres físicos y métricas al 100 % ahoga esa honestidad.

### 3.3 Metáfora frente a implementación

| Nombre en el relato | Qué hay en el código |
|---|---|
| CDT / Pachner | Grafo magnético con fases; Pachner no se simula (el README lo admite) |
| Líquido de espines / EPR físico | XXZ exacto hasta ~16 espines; EPR es utilidad predictiva clásica |
| 3D-PEPS | Scaffold; no cubre todos los enlaces físicos; Hamiltonian no disponible |
| LLM periférico | Gemma genera el texto; CTP es un sesgo α = 0,25 sobre logits |
| SO cognitivo | Orquestación de módulos + chat; no un sistema operativo |

Un revisor preguntará si esto es física o un grafo con fase. Hoy es lo segundo, con analogías útiles y nombres que sobran.

### 3.4 Fragmentación

Hay demasiados motores en paralelo: CDT legado, fasorial, híbrido, unificado spin, circadiano, adaptativo, future-guided, VMC, DMRG, PEPS. `src/lib.rs` ya marca huérfanos:

- `emergent_cognition_training`: el entrenador infinito reimplementa su propio gate
- `engine_comparison`: sólo tests propios
- `native_plastic_symmetry_network`: nadie la consume
- `variational_spin_liquid_vmc`: validado, sin consumidores fuera de tests

41 binarios. Un paper o un colaborador no puede decir cuál es *el* sistema.

### 3.5 Evidencia no reproducible o estancada

- Currículo de 5 fases y 221,6 millones de ejemplos: evidencia de sesión; artefactos no versionados. `paper.md` lo declara. No citar como resultado vigente.
- Entrenamiento de desarrollo (`data/gemma2_developmental_infinite_training/summary.json`): ciclo 491, etapa máxima 4, `symbolic_accuracy = 0`, `language_plan_accuracy = 0`, `functional_cognition_gate = false`, 5 ciclos aceptados de 491. Más ciclos no sustituyen un protocolo que no pasa.
- Future-guided: 98 ciclos, 83 gates, 70 consolidados, un atractor. Evidencia de mecanismo bajo entrada repetida, no de generalización semántica (el preprint ya lo dice).
- Chat Dyamon: Gemma 2 2B mezcla idiomas, no retiene el nombre, `quality` 0,0–0,49, 26/26 capas. El producto no demuestra el claim de periferia lingüística.

### 3.6 Ausencias que impiden publicar la tesis fuerte

- Baseline externo (Hopfield moderno, memoria asociativa, GNN, RAG, LoRA).
- Tarea que el motor no vea inyectada en el código.
- Medida de olvido catastrófico longitudinal (`ΔR ≥ −ε` del preprint, todavía no hecha).
- Calidad de chat con `α > 0` frente a Gemma denso; ahorro real de capas en decode incremental.

---

## 4. Cómo leer el proyecto, en una frase

Laboratorio de **atractores y consolidación de dos tiempos**, con higiene experimental en un núcleo pequeño, envuelto en un relato demasiado grande (cognición, CDT físico, spin, LLM periférico, SO cognitivo).

Si se recorta al núcleo, hay preprint técnico. Si se deja como cognición emergente, se cae en revisión.

---

## 5. Mejoras (orden de prioridad)

### P0 — Una tesis, un motor, un paper

**Tesis recomendada:**

> Una consolidación CDT de un patrón verificado deforma el paisaje fasorial y amplía de forma causal la cuenca de recuperación.

Eso ya tiene protocolo (`consolidation_basin_experiment`), gate en CI y disclaimers correctos.

Acciones:

1. Declarar en README y en el preprint que ese es el resultado principal. El resto es infraestructura o trabajo en curso.
2. Congelar o archivar motores que no alimentan esa tesis (VMC, PEPS, red plástica, comparador huérfano) detrás de un feature flag o `docs/archive.md`.
3. Unificar manuscritos: `paper_inferencia_fasorial_consolidacion_cdt.md` es el paper; `paper.md` pasa a bitácora histórica o se recorta a apéndice de compatibilidad.
4. Integrar o retirar `emergent_cognition_training`: o el entrenador infinito usa ese gate, o el módulo se elimina del crate público.

Criterio de hecho: un extraño puede ejecutar un comando y obtener el resultado principal sin leer 41 binarios.

### P1 — Romper el techo del 100 %

Los tests cognitivos actuales no pueden fallar de forma informativa.

Acciones:

1. Añadir un holdout que **no** esté hardcodeado: más nodos, patrón no inyectado, corrupción > 40 %, ruido de grafo, semilla no usada en desarrollo.
2. Reportar intervalos (media ± desviación, no sólo 100 %). El gate de cuenca ya pide ganancia mínima; extender eso a generalización.
3. Si un nivel sigue al 100 % tras endurecerlo, el nivel está mal diseñado: bajarlo o marcarlo como smoke, no como evidencia.
4. Separar en CI: smoke rápido (formato, tipos, un ensayo) vs. gate científico pesado.

Criterio de hecho: al menos una métrica cognitiva principal deja de ser 1,0 en el reporte publicado, o se explica por qué el techo es el resultado esperado y se añade una tarea más dura que sí discrimine.

### P2 — Baselines externos

Sin comparación, no hay ventaja.

Acciones, en el mismo fixture de cuenca (32 nodos, mismas cues, mismo presupuesto de iteraciones):

| Baseline | Qué mide |
|---|---|
| Hopfield / Hopfield moderno | memoria asociativa clásica |
| Relajación fasorial sin consolidación CDT | ya existe el brazo pre; formalizarlo como baseline |
| GNN pequeño o Hebb sobre las mismas aristas | aprendizaje relacional estándar |
| (opcional) RAG o LoRA sobre Gemma | sólo si el claim incluye lenguaje |

Publicar tiempo, energía del modelo, recuperación y saturación. El unificado ya es ~400× más lento que el legado en 24 ensayos; esa cifra debe ir con la ventaja funcional, no esconderse.

Criterio de hecho: una tabla con al menos dos métodos ajenos al crate.

### P3 — Gemma o es periferia o se saca del claim

Hoy el chat no demuestra el paper.

Opciones (elegir una):

**A. Periferia de verdad.** Gemma sólo compile a receta y verbalice un resultado del motor. Si el motor no infiere, el chat se abstiene. Prohibido responder “I’m a large language model”.

**B. Híbrido medido.** Conservar el sesgo CTP, pero publicar A/B: `α = 0` vs `α = 0,25` en un set fijo de prompts (español, identidad, retención de turno). Si no hay ganancia, bajar α o retirar el claim.

**C. Sacar el LLM del preprint.** El paper de cuenca no lo necesita. El chat queda como demo de ingeniería.

Además: system prompt estable, idioma forzado, identidad Dyamon fuera del historial de usuario. El journal actual muestra que 2B sin ancla no retiene ni el nombre.

Criterio de hecho: o hay métrica de chat que gane a Gemma denso, o el preprint no habla de periferia lingüística como resultado.

### P4 — Dejar de entrenar infinito sin gate

El currículo de desarrollo está atascado en etapa 4. El trainer de 221 M ejemplos no está en el checkout. Seguir girando ciclos no produce evidencia nueva.

Acciones:

1. Parar `native_gemma2_spin_infinite_trainer` hasta que `symbolic_accuracy` y el gate funcional pasen en una corrida corta y versionada.
2. Versionar un checkpoint mínimo reproducible (o un script que lo regenere en < N minutos) para cada claim numérico del paper.
3. Tratar cifras históricas (ciclo 116350, 221,6 M, 115876 wake) como apéndice “sesión”, nunca como tabla principal.
4. Si un checkpoint no cabe en Git, publicar hash, comando y semilla; no el número suelto.

Criterio de hecho: cada cifra del preprint se regenera con un comando documentado.

### P5 — Higiene del crate

- Un binario de entrada por rol: chat, experimento de cuenca, trainer, visualizador. El resto a `examples/` o `src/bin/archive/`.
- Resolver huérfanos de `lib.rs` (integrar o `#[cfg]` de investigación).
- No añadir nombres de física (Pachner, EPR, CDT) a APIs nuevas salvo que el código implemente la operación.
- Conservar `target-cpu=native` en local; en CI usar un target portable o documentar que los números de wall-clock no son comparables entre máquinas.
- El journal de Dyamon no debería ensuciar `git status` si es dato de sesión: `.gitignore` o ruta bajo `data/` ya ignorada, con un fixture pequeño de ejemplo.

### P6 — Secuencia experimental (la del propio paper)

El preprint ya escribe el orden correcto. Cumplirlo:

```text
algoritmo -> ventaja computacional -> escalabilidad -> mapa físico
```

No saltar a hardware, consciencia o SO cognitivo. El siguiente experimento, después de P1–P2, es **escala**: 128 / 512 / 2048 nodos en cuenca, no otro motor.

Olvido acotado: implementar el ΔR del §4 del preprint sobre un set retenido *después* de consolidar un patrón nuevo. Hoy no existe.

---

## 6. Qué no hacer

- No añadir otro binario “unificado” hasta archivar dos existentes.
- No citar 100 % como evidencia de cognición.
- No mezclar energía del funcional con julios.
- No presentar Gemma 2 2B como motor de razonamiento.
- No reanudar el trainer infinito para “ver si pasa”.
- No publicar `paper.md` y el preprint de fasores como si fueran el mismo resultado.

---

## 7. Plan mínimo de 30 días

Semana 1: recortar README a la tesis de cuenca; marcar huérfanos; decidir A/B/C para Gemma.

Semana 2: holdout duro + dos baselines en el mismo fixture; CI verde.

Semana 3: ΔR de retención o escala 128/512 nodos; una sola tabla para el preprint.

Semana 4: un paper de 8–12 páginas a partir de `paper_inferencia_fasorial_consolidacion_cdt.md`; bitácora histórica fuera del cuerpo.

Entregable: un comando, una tabla, un manuscrito. Si eso no cabe, el proyecto sigue siendo laboratorio, no resultado.

---

## 8. Comandos del núcleo vigente

```powershell
cargo test --lib consolidation_basin_experiment -- --nocapture
cargo run --release --bin native_consolidation_basin_experiment
cargo test --release --lib
```

El resto de binarios es contexto. No es el resultado.

---

## 9. Siguiente documento

Evaluación de P0–P6 (tests + binario canónico, 31 ago 2026) y arquitectura
del ciclo siguiente: [`arquitectura_siguiente_ciclo.md`](arquitectura_siguiente_ciclo.md).
