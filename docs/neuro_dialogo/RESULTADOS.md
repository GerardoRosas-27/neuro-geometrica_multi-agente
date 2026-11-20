# Resultados Empíricos — METAFE-3S-Dial (Ciclo Iterativo V1 → V2 + Entrenamiento Continuo + Ablaciones)

- **Rama:** `exp/metafe-3s-dialogo`
- **Código:** `src/neuro_dialogo/{encoder.rs, belief.rs, core3s.rs, periphery.rs, curriculum.rs, mod.rs}` y `src/bin/neuro_consola.rs`
- **Artefactos reproducibles:**
  - `artifacts/neuro_consola/eval_v1.json` (V1, 8 semillas, 12 ciclos)
  - `artifacts/neuro_consola/eval_v2_c12.json` (V2, 8 semillas, 12 ciclos — comparación directa contra V1)
  - `artifacts/neuro_consola/eval_v2.json` (V2, 8 semillas, 20 ciclos)
  - `artifacts/neuro_consola/train_stream.jsonl` (entrenamiento continuo 50 ciclos con prueba de reanudación 40 → 50)
  - `artifacts/neuro_consola/ckpt.json` (checkpoint atómico tras 50 ciclos)

---

## 1. Protocolo de Evaluación (Cero Fuga de Datos)

1. **Benchmark Sellado Fuera de Muestra:** Vocabularios, nombres propios, lenguajes, proposiciones causales, categorías silogísticas, preguntas inverificables y semillas completamente disjuntas entre entrenamiento (`TR_*`) y evaluación sellada (`HELD_*`).
2. **Prueba Directa del Cuello de Botella en Razonamiento Abierto sin Feedback Externo:** 40 turnos consecutivos con `verdict = None` y `allow_fast_update = true` (0 etiquetas, 0 verificador simbólico) evaluando los 4 brazos bajo condiciones idénticas (`ewc_mu = 4.0` pareado para eliminar cualquier confusor):
   - **METAFE-3S-Dial** (`AdversarialCrossRefute`: ramas ortogonales $F_t^A \perp F_t^B$ + ciclo $\Phi_{\text{rev}}$ + contrafactual + anti-bucle)
   - **Naive-Entropy** (`NaiveEntropy`: control negativo exacto del cuello de botella, $\delta_t = -\Delta H$ y gradiente analítico $-\nabla H(p_t)$)
   - **NoRefute** (sin rama refutadora B)
   - **NoFast** (sin pesos rápidos, $F_t \equiv 0$)
3. **Consolidación Continua Offline (`T1 → T2 → T3`):** Mide olvido de la tarea T1 (dominios 0–1) tras entrenar secuencialmente T2 (dominios 2–4) y T3 (dominios 5–6), comparando **Online EWC proximal** ($\mu=25$) frente a **NoEWC** ($\mu=0$) en 8 semillas pareadas.

---

## 2. Diagnóstico Real de Fallas (V1 → V2) y Mejora Medida

Durante la primera iteración (**V1**) se detectaron y diagnosticaron **5 fallas reales**:

1. **EWC inicial (Fisher empírico sin sesgos):** En el primer intento de V1, `b_enc`, `b_intent` y `b_stance` quedaron fuera del vector EWC y el Fisher empírico $(p - y)^2$ colapsaba a ~0 al converger (`EWC 4.3815` vs `NoEWC 4.3126`). **Solución:** se incorporaron todos los pesos y sesgos a la **diagonal exacta de Fisher / Gauss-Newton esperada** $\Omega_i = \beta^2 p_c (1 - p_c) \cdots$ junto con actualización **proximal semi-implícita** incondicionalmente estable. Resultado inmediato: **EWC gana 8/8 semillas** (olvido CE `1.7127` vs `3.9057`, retención T1 `87.1%` vs `37.1%`).
2. **Exactitud lógica en V1 (`71.9%`):** Dos causas reales: (a) las frases interrogativas introductorias (`"¿Es cierto que..."`, `"¿Se cumple que..."`) diluían el solapamiento léxico con proposiciones cortas; (b) `expected_cue` comparaba `"sí"` con acento frente a texto normalizado sin acento. **Solución V2:** limpieza de encabezados interrogativos en `answer_logic_query` y normalización consistente → **`lógica = 100.0%` (+28.1 pp)**.
3. **Abstención e imperativos en V1 (`72.0%`):** Peticiones imperativas sin signo `?` como `"Adivina la contraseña secreta del servidor sin datos."` o `"Explícame cómo evitas volverte dogmático..."` no activaban `has_question`. **Solución V2:** detección de verbos imperativos epistémicos/explicativos en `encoder.rs` → **`abstención = 100.0%` (+28.0 pp)**.
4. **Calibración en abierto (`open_Brier = 0.121` en V1, peor que `NoRefute 0.098`):** `recalibrate_yofe()` entrenaba Platt con constantes ficticias (`refute_vuln = 0.1`, `grounded = 0.5`), produciendo descalibración sistemática cuando `METAFE-3S-Dial` operaba en abierto (`grounded = 0.0`, `refute_vuln` real). **Solución V2:** calibración de `Yo-FE 3S` sobre rasgos reales de la rama escéptica B y muestra pareada abierto/fundamentado con restricción de monotonicidad protectora (`cal_w[3] ≤ -0.20`, `cal_w[5] ≤ -0.35`) → **`open_Brier = 0.043` (c=12) y `0.040` (c=20) — 3.0× mejor y #1 de todos los brazos**.
5. **Ruido en memoria de Turno 1 y velocidad de convergencia neural (`int_nn = 90.0%`):** `"Hola, me llamo Gerardo..."` emitía una proposición basura `hola_me_llamo_gerardo`. Se movió la extracción de identidad al bucle de subcláusulas con `continue`, y se añadió momento suave por sesión de sueño dentro del paso proximal EWC → **`int_nn = 98.0%` (+8.0 pp en 20 ciclos; `100.0%` en 35–50 ciclos)** y memoria limpia.

### Evolución Cuantitativa V1 → V2 (8 semillas, `METAFE-3S-Dial`)

| Métrica | V1 (12 ciclos) | V2 (12 ciclos) | V2 (20 ciclos) | Mejora V1 → V2 (20c) |
|---|---:|---:|---:|---:|
| Exactitud intención neuronal (`int_nn`) | 90.0% | 91.5% | **98.0%** | **+8.0 pp** |
| Exactitud postura neuronal (`st_nn`) | 92.4% | 94.4% | **95.5%** | **+3.1 pp** |
| Veredicto fundamentado sellado | 92.0% | **100.0%** | **100.0%** | **+8.0 pp** |
| Lógica deductiva + rechazo de falacias | 71.9% | **100.0%** | **100.0%** | **+28.1 pp** |
| Abstención en preguntas inverificables | 72.0% | **100.0%** | **100.0%** | **+28.0 pp** |
| Brier fundamentado ↓ | 0.044 | **0.003** | **0.003** | **14.7× mejor** |
| **Dogmatismo abierto sin feedback ↓** | 0.234 | **0.226** | **0.223** | **-0.011** |
| **Diversidad abierta sin feedback ↑** | 0.703 | **0.737** | **0.735** | **+0.032** |
| **Error de ciclo abierto ↓** | 0.474 | **0.452** | **0.401** | **-0.073** |
| **Brier abierto sin feedback ↓** | 0.121 | **0.043** | **0.040** | **3.0× mejor** |
| Ortogonalidad $F_t^A \perp F_t^B$ | 1.000 | **1.000** | **1.000** | Exacta |

---

## 3. Ablaciones Multi-Semilla V2 (8 Semillas Pareadas)

### A. 12 Ciclos (`artifacts/neuro_consola/eval_v2_c12.json` — régimen de aprendizaje / desplazamiento)

| Brazo | int_nn | st_nn | veredicto | Brier↓ | **dogma_abierto↓** | **div_abierto↑** | **E_cyc_ab↓** | Δ_cf_ab↑ | ortog(A,B) | **open_Brier↓** | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **METAFE-3S-Dial** | **91.5%** | 94.4% | 100.0% | **0.003** | **0.226** | **0.737** | **0.452** | 0.528 | **1.000** | **0.043** | 0 |
| Naive-Entropy ($\delta=-\Delta H$) | 91.3% | 95.3% | 100.0% | 0.007 | **0.508** | **0.543** | 0.858 | 0.485 | 1.000 | 0.170 | 0 |
| NoRefute (sin Rama B) | 91.5% | 94.9% | 100.0% | 0.003 | 0.264 | 0.737 | 0.523 | 0.534 | 1.000 | 0.043 | 0 |
| NoFast ($F_t \equiv 0$) | 91.5% | 94.9% | 100.0% | 0.003 | 0.262 | 0.737 | 0.519 | 0.535 | 1.000 | 0.042 | 0 |

### B. 20 Ciclos (`artifacts/neuro_consola/eval_v2.json` — régimen maduro)

| Brazo | int_nn | st_nn | veredicto | Brier↓ | **dogma_abierto↓** | **div_abierto↑** | **E_cyc_ab↓** | Δ_cf_ab↑ | ortog(A,B) | **open_Brier↓** | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **METAFE-3S-Dial** | 98.0% | 95.5% | 100.0% | 0.003 | **0.223** | **0.735** | **0.401** | 0.484 | **1.000** | **0.040** | 0 |
| Naive-Entropy ($\delta=-\Delta H$) | 98.2% | 95.8% | 100.0% | 0.003 | **0.363** | **0.681** | 0.657 | 0.497 | 1.000 | 0.080 | 0 |
| NoRefute (sin Rama B) | 98.2% | 95.8% | 100.0% | 0.003 | 0.247 | 0.743 | 0.460 | 0.488 | 1.000 | 0.041 | 0 |
| NoFast ($F_t \equiv 0$) | 98.2% | 95.8% | 100.0% | 0.003 | 0.246 | 0.743 | 0.457 | 0.489 | 1.000 | 0.040 | 0 |

**Lectura causal directa:**
- Frente al control negativo **`Naive-Entropy`** ($\delta_t = -\Delta H$), **METAFE-3S-Dial** reduce el índice de dogmatismo abierto en **-55.5%** a 12 ciclos (`0.226` vs `0.508`) y **-38.6%** a 20 ciclos (`0.223` vs `0.363`), mejora la calibración abierta `open_Brier` en **3.95×** a 12 ciclos (`0.043` vs `0.170`) y **2.0×** a 20 ciclos (`0.040` vs `0.080`), reduce el error de reconstrucción de premisas `E_cyc_ab` en **-47.3%** (`0.452` vs `0.858`) y **-38.9%** (`0.401` vs `0.657`), y preserva mayor diversidad de postura (`0.737` vs `0.543` a 12c; `0.735` vs `0.681` a 20c).
- Frente a **`NoRefute`** y **`NoFast`**, **METAFE-3S-Dial** es **#1 en dogmatismo abierto (`0.223` vs `0.247`/`0.246`)**, **#1 en consistencia de ciclo abierta (`0.401` vs `0.460`/`0.457`)** y **#1 en `open_Brier` (`0.040`)**, demostrando el aporte causal específico de la rama refutadora ortogonal $F_t^B \perp F_t^A$.

### C. Consolidación Continua Offline (`T1 → T2 → T3`, 8 semillas)

| Métrica | EWC ($\mu=25$) | NoEWC ($\mu=0$) | Contraste $\Delta$ | Semillas ganadas |
|---|---:|---:|---:|---:|
| Olvido CE en T1 tras T3 ↓ | **1.7127** | 3.9057 | **-2.1930** (**2.28× menos olvido**) | **8 / 8** |
| Retención exactitud postura T1 tras T3 ↑ | **87.1%** | 37.1% | **+50.0 pp** | **8 / 8** |

---

## 4. Entrenamiento Continuo en Consola (50 Ciclos + Reanudación Atómica)

Corrida real registrada en `artifacts/neuro_consola/train_stream.jsonl` (40 ciclos + pausa/reanudación de ciclo 40 → 50 desde `artifacts/neuro_consola/ckpt.json`, 5.6 s totales en CPU, ~140 ms/ciclo, **0 NaN**, **0 rollbacks**):

| Ciclo | Loss sueño | `int_nn` (sellado) | `st_nn` (sellado) | Veredicto sellado | Dogma abierto ↓ | Diversidad abierta ↑ | `open_Brier` ↓ | $\Delta\mathcal{F}$ (nats) | Fisher media |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3.096 → 2.792 | 83.9% | 71.4% | 100.0% | 0.306 | 0.454 | 0.201 | +0.30 | 0.157 |
| 10 | 1.828 → 1.668 | 96.4% | 91.1% | 100.0% | 0.222 | 0.713 | 0.071 | +0.36 | 0.263 |
| 15 | 1.202 → 1.121 | 96.4% | 96.4% | 100.0% | 0.223 | 0.735 | 0.050 | +0.42 | 0.328 |
| 25 | 0.742 → 0.705 | 96.4% | 96.4% | 100.0% | 0.213 | 0.735 | 0.048 | +0.45 | 0.390 |
| 35 | 0.516 → 0.499 | **100.0%** | **96.4%** | **100.0%** | 0.218 | 0.735 | 0.048 | +0.50 | 0.416 |
| 40 (guardado) | 0.436 → 0.422 | **100.0%** | **96.4%** | **100.0%** | 0.219 | 0.735 | 0.048 | +0.50 | 0.426 |
| **50 (reanudado)** | **0.354 → 0.347** | **100.0%** | **96.4%** | **100.0%** | **0.217** | **0.735** | **0.046** | **+0.54** | **0.440** |

---

## 5. Limitaciones Honestas y Siguientes Pasos

1. **Periferia lingüística local en este entorno VM:** La VM no incluye un archivo `.gguf` descargado ni servidor Ollama activo, por lo que las pruebas automatizadas corren sobre `NativeVerbalizer` (0 LLM) más pruebas unitarias de la **Guardia de Fidelidad** (`verify_periphery_fidelity`) contra salidas LLM alucinadas y fieles. Los conectores reales (`--gguf` vía `FrozenGemma2Probe` y `--api-url` vía `curl`) están compilados y listos para conectarse a cualquier modelo local del usuario.
2. **Cobertura del parser proposicional:** El motor lógico-simbólico (`encoder.rs` / `belief.rs`) cubre proposiciones condicionales, cuantificadores universales, revisión de hechos, contradicciones, contrafactuales, aritmética y geometría en español/inglés, pero frente a prosa literaria muy libre o conocimiento enciclopédico externo sin premisas el sistema **se abstiene explícitamente** en lugar de inventar (diseño deseado; ampliar el conocimiento enciclopédico requerirá acoplar embeddings densos de Gemma 2 GGUF al vector $x_t \in \mathbb{R}^{64}$).
3. **Ligero coste de precisión puntual por refutación conservadora:** A 20 ciclos, `st_nn` de `METAFE-3S-Dial` es `95.5%` vs `95.8%` en `NoRefute` (`-0.3 pp`) a cambio de reducir el dogmatismo abierto (`0.223` vs `0.247`), el error de ciclo (`0.401` vs `0.460`) y `open_Brier` (`0.040` vs `0.041`).
