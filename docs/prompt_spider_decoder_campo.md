# Prompt Spider · Decoder del campo (EXPERIMENTAL)

Rama `exp/spider-field-decoder`. Conecta el Prompt Spider al campo
(líquido + CDT + RQM/EPR) igual que el chat: con el interruptor **Decoder del
campo** activo, las palabras **escaladas** (`p < umbral`) las decide un
router del campo entrenado para el Spider **en vez del LLM**. Código, el
scorer determinista y la cola de aprobación no cambian.

Código: [`src/spider_field.rs`](../src/spider_field.rs) (router puro),
[`src/web/spider_field_job.rs`](../src/web/spider_field_job.rs) (estado,
entrenamiento, endpoints), [`src/spider_field_experiment.rs`](../src/spider_field_experiment.rs)
(experimento de sustrato), integración en [`src/web/spider_job.rs`](../src/web/spider_job.rs).

## 1. Cómo funciona el campo del chat (lo que se reutiliza)

El chat (`AppState.fuse = FusedLiquidCdt`, 8 conceptos) codifica el mensaje a
un concepto, infiere en el líquido (`WavePredictCore`), consulta engramas CDT
(escritos solo en el sueño) y el índice relacional RQM/EPR; Gemma solo
verbaliza. El entrenamiento (`/api/train/*`) genera datasets, hace
`infer/observe/teach_relation`, duerme (`sleep_optimize`) entre datasets y
guarda checkpoints por dataset. El Spider copia **ese patrón** (vigilia →
sueño → checkpoint), pero con un sustrato propio (ver §4: decisión por
experimento).

## 2. Arquitectura del router del campo

Una decisión = un fork escalado.

1. **Codificación** (`encode_decision`): 53 rasgos locales por token (tipo,
   léxicos del scorer —stopword, ambigüedad, afirmación, aprobación, entidad,
   pista de fuente—, mayúscula, inicio de frase, longitud, frecuencia, en
   objetivo/rol, puntuación antes/después, las 4 puntuaciones heurísticas,
   sección one-hot, hashing con signo de identidad (8) y sufijo (6)) en una
   **ventana ±3** (7 pasos) + 12 rasgos globales (4 puntuaciones, `p`,
   pregunta decisiva, ruta inicial).
2. **Inferencia líquida** (`LiquidEncoder`): reservorio LTC de 48 neuronas con
   constante de tiempo dependiente de la entrada (τ = 1 + 3σ(a·u)), pesos
   fijos por semilla; lee la ventana token a token. Estado = h(centro) ⊕
   h(final).
3. **CDT** (`CdtBasinMemory`): 10 engramas (5 cabezas × sí/no) en bloques
   disjuntos de un `NativeThermoCdtSubstrate`; se consolidan **solo en el
   sueño** (8 pasos termodinámicos, plantilla amp/cos/sin). En inferencia se
   leen similitudes y la **novedad** → guardia fuera de distribución (umbral
   = percentil 99.5 de la novedad de entrenamiento + 0.02).
4. **RQM/EPR** (`RqmIndex`): celda LSH de 8 bits del estado líquido → cue de
   un `NativeThermoRqmEprSubstrate`; `train_observed_transition` en el sueño y
   destilado a una tabla de priors (la inferencia lee la tabla, no muta).
5. **Lectura**: 5 cabezas logísticas (relevant, grounded, ambiguous,
   needs_approval, **ok**) con Adam + L2 y **escalado de temperatura** por
   cabeza en un split de calibración (~20 % de los prompts, por grupo).
6. **Decisión**: resuelve (`resolved_by = campo`) solo si está entrenado,
   dentro de distribución, `P(aprob) < 0.5`, `P(ambigua) < 0.5` y
   `P(ok) ≥ piso` (0.70, el mismo del LLM). Si no → **pendiente** con motivo.
   Nunca inventa: sin entrenar, todo escalado queda pendiente.

Persistencia: `data/spider_field/model.json` (lectura, estado CDT, conteos/caché
RQM; los pesos líquidos se regeneran de la semilla). La repetición no se
persiste (ver limitaciones).

## 3. Entrenamiento Prompt Spider

Tarjeta al final de la pestaña Prompt Spider (`POST /api/spider/field/train`,
`/stop`, `GET /status`, `/events`). Cada dataset:

- **Sintéticos con verdad por construcción** (`synth_prompt`): marcadores de
  ambigüedad resueltos o no por un número entre paréntesis, deícticos con/sin
  sustantivo, afirmaciones con/sin fuente, nombres propios definidos por
  aposición o no, relleno irrelevante, instrucciones never/always, verbos de
  contenido; 5 % de ruido de etiquetas solo en entrenamiento.
- **Scorer**: palabras de alta confianza (`code` → respuestas del scorer y ok;
  `tú` → aprobación, no-ok) del prompt de ejemplo y de cada prompt guardado.
- **Usuario**: Aprobar/Rechazar de `data/spider_runs` (y de la corrida actual).
- **Maestro LLM**: veredictos guardados de corridas con LLM (nunca los del
  propio campo) y, opcional, consultas en vivo al LLM activo (Gemma local o
  API, con respaldo) sobre escaladas reales aún sin etiqueta.
- Vigilia (`observe`) → **sueño** (CDT + RQM + reajuste sobre una repetición
  de hasta 24 000 ejemplos) → evaluación en hold-out fijo (24 prompts H1 de
  vocabulario visto + 24 H2 de vocabulario nuevo, etiquetas limpias, y 1 de
  cada 5 palabras reales etiquetadas) → **checkpoint por dataset**
  (`data/checkpoints/spider_field/spider_field_<ts>_ds_<n>.json` + `latest.json`:
  semillas, ejemplos reales con etiquetas, conteos por fuente, informe de
  sueño y métricas) y modelo guardado.
- **Compatibilidad con el sueño**: cada Aprobar/Rechazar se observa en vigilia;
  la pestaña **Sueño** consolida esos episodios en el mismo ciclo
  (`run_one_sleep_cycle`), o se difiere si la repetición es < 400 (tras
  reiniciar la app) o hay un entrenamiento en curso.

## 4. Experimento de sustrato (A vs B vs C)

Pregunta: ¿sustrato **en blanco** o **capa externa acoplada** al campo del
chat? Mismo hold-out, 5 semillas, mismos datos por semilla.

- **A · blanco**: líquido + CDT + RQM propios del Spider.
- **B1 · núcleo congelado + lectura**: la tabla de respuesta del campo del
  chat (copia, solo lectura) por cue léxico + lectura.
- **B2 · núcleo congelado + capa líquido/CDT/RQM**: B1 + la capa de A.
- **C · ajuste fino del núcleo compartido**: la experiencia del Spider se
  escribe en el campo del chat (`teach_relation` + sueño) y se lee de vuelta.
- Ablaciones: A sin CDT, A sin RQM, sin campo (logística sobre rasgos crudos).
- **Interferencia**: exactitud del mapa relacional del chat
  (`cue → cue+1`) y máx. |Δscore| antes/después, sobre una copia.
- **Maestro**: Gemma 2 2B-it **Q3_K_L** (el GGUF por defecto de la app),
  prompt compacto P(sí) en las dos órdenes, en 20 escaladas del prompt de
  ejemplo + 50 escaladas de H1 ([`data/spider_field_teacher_gemma.json`](data/spider_field_teacher_gemma.json)).

Reproducir (≈10 min en 8 núcleos):

```bash
SPIDER_TEACHER=docs/data/spider_field_teacher_gemma.json \
SPIDER_FIELD_OUT=docs/data/spider_field_substrate_results \
cargo test --release --lib spider_field_substrate_experiment -- --ignored --nocapture
# etiquetas del maestro (≈15 s/palabra en CPU):
GEMMA2_GGUF=models/gemma-2-2b-it-Q3_K_L.gguf N=70 OUT=docs/data/spider_field_teacher_gemma.json \
cargo test --release --features web --lib real_gemma_teacher_labels -- --ignored --nocapture
```

### Resultados (media ± desviación sobre 5 semillas)

Semillas: [11, 22, 33, 44, 55] · prompts de entrenamiento/semilla: 240 en 6 datasets · ruido de etiquetas 5 % · hold-out H1 6854 palabras (1123 escaladas), H2 vocabulario desplazado 6689 (949 escaladas), prompt de ejemplo 20 escaladas · piso 0.70 · umbral 0.95

### Router en escaladas (H1, en distribución)

| brazo | exactitud ok | ECE ok | Brier ok | resuelto % | precisión resuelto | resuelto erróneo % | pendiente % | latencia µs (mediana) |
|---|---|---|---|---|---|---|---|---|
| heurística (scorer, piso) | 0.788 | 0.109 | 0.116 | 73.8 | 0.945 | 4.1 | 26.2 | — |
| A · blanco (líquido+CDT+RQM) | 0.931 ± 0.006 | 0.079 ± 0.010 | 0.057 ± 0.002 | 79.1 ± 2.2 | 0.978 ± 0.007 | 1.8 ± 0.6 | 20.9 ± 2.2 | 31.7 ± 0.3 |
| B1 · núcleo congelado + lectura | 0.897 ± 0.002 | 0.094 ± 0.009 | 0.073 ± 0.001 | 79.3 ± 2.6 | 0.968 ± 0.010 | 2.5 ± 0.8 | 20.7 ± 2.6 | 0.7 ± 0.1 |
| B2 · núcleo congelado + capa líquido/CDT/RQM | 0.931 ± 0.008 | 0.080 ± 0.011 | 0.056 ± 0.002 | 79.4 ± 2.0 | 0.976 ± 0.006 | 1.9 ± 0.6 | 20.6 ± 2.0 | 31.5 ± 0.4 |
| C · ajuste fino del núcleo compartido | 0.929 ± 0.006 | 0.079 ± 0.010 | 0.058 ± 0.002 | 79.4 ± 2.1 | 0.977 ± 0.007 | 1.8 ± 0.6 | 20.6 ± 2.1 | 29.2 ± 0.4 |
| A sin CDT | 0.930 ± 0.006 | 0.078 ± 0.009 | 0.057 ± 0.002 | 79.4 ± 2.0 | 0.977 ± 0.007 | 1.8 ± 0.6 | 20.6 ± 2.0 | 30.1 ± 0.2 |
| A sin RQM | 0.930 ± 0.004 | 0.079 ± 0.009 | 0.058 ± 0.002 | 78.8 ± 2.2 | 0.978 ± 0.008 | 1.8 ± 0.7 | 21.2 ± 2.2 | 32.0 ± 3.0 |
| sin campo (logística cruda) | 0.898 ± 0.004 | 0.090 ± 0.004 | 0.073 ± 0.001 | 79.1 ± 2.3 | 0.971 ± 0.008 | 2.3 ± 0.8 | 20.9 ± 2.3 | 0.5 ± 0.0 |

### Router en escaladas (H2, vocabulario desplazado)

| brazo | exactitud ok | ECE ok | resuelto % | precisión resuelto | resuelto erróneo % |
|---|---|---|---|---|---|
| heurística | 0.710 | 0.092 | 70.2 | 0.872 | 9.0 |
| A · blanco (líquido+CDT+RQM) | 0.883 ± 0.014 | 0.064 ± 0.006 | 74.3 ± 2.7 | 0.934 ± 0.009 | 4.9 ± 0.7 |
| B1 · núcleo congelado + lectura | 0.835 ± 0.005 | 0.059 ± 0.010 | 75.9 ± 2.6 | 0.908 ± 0.010 | 7.0 ± 1.0 |
| B2 · núcleo congelado + capa líquido/CDT/RQM | 0.884 ± 0.015 | 0.061 ± 0.007 | 74.7 ± 2.9 | 0.934 ± 0.009 | 4.9 ± 0.7 |
| C · ajuste fino del núcleo compartido | 0.879 ± 0.012 | 0.063 ± 0.005 | 74.9 ± 2.6 | 0.935 ± 0.012 | 4.9 ± 1.0 |
| A sin CDT | 0.882 ± 0.014 | 0.065 ± 0.008 | 74.5 ± 2.5 | 0.935 ± 0.009 | 4.9 ± 0.7 |
| A sin RQM | 0.881 ± 0.014 | 0.064 ± 0.008 | 74.0 ± 3.0 | 0.935 ± 0.010 | 4.8 ± 0.8 |
| sin campo (logística cruda) | 0.838 ± 0.007 | 0.063 ± 0.007 | 75.9 ± 2.1 | 0.910 ± 0.009 | 6.9 ± 0.9 |

### Por pregunta (todas las palabras de H1): exactitud / ECE / Brier

| brazo | relevant | grounded | ambiguous | needs_approval | ok |
|---|---|---|---|---|---|
| heurística | 0.987 / 0.022 / 0.013 | 0.980 / 0.034 / 0.012 | 0.987 / 0.038 / 0.008 | 1.000 / 0.005 / 0.002 | 0.954 / 0.031 / 0.033 |
| A · blanco (líquido+CDT+RQM) | 0.991 / 0.059 / 0.009 | 0.997 / 0.052 / 0.006 | 0.997 / 0.055 / 0.006 | 1.000 / 0.048 / 0.003 | 0.980 / 0.060 / 0.018 |
| B1 · núcleo congelado + lectura | 0.987 / 0.051 / 0.013 | 0.992 / 0.052 / 0.008 | 0.994 / 0.056 / 0.007 | 1.000 / 0.048 / 0.002 | 0.972 / 0.060 / 0.023 |
| B2 · núcleo congelado + capa líquido/CDT/RQM | 0.991 / 0.060 / 0.009 | 0.997 / 0.052 / 0.006 | 0.997 / 0.055 / 0.006 | 1.000 / 0.050 / 0.003 | 0.980 / 0.059 / 0.018 |
| C · ajuste fino del núcleo compartido | 0.991 / 0.059 / 0.009 | 0.997 / 0.052 / 0.006 | 0.997 / 0.055 / 0.006 | 1.000 / 0.048 / 0.003 | 0.980 / 0.059 / 0.018 |
| A sin CDT | 0.991 / 0.059 / 0.009 | 0.997 / 0.052 / 0.006 | 0.997 / 0.055 / 0.006 | 1.000 / 0.050 / 0.003 | 0.979 / 0.059 / 0.018 |
| A sin RQM | 0.991 / 0.059 / 0.009 | 0.997 / 0.052 / 0.006 | 0.997 / 0.055 / 0.006 | 1.000 / 0.048 / 0.003 | 0.979 / 0.060 / 0.018 |
| sin campo (logística cruda) | 0.987 / 0.051 / 0.013 | 0.992 / 0.052 / 0.008 | 0.994 / 0.056 / 0.007 | 1.000 / 0.048 / 0.002 | 0.972 / 0.058 / 0.024 |

### Exactitud de `ok` por ruta inicial (H1)

| brazo | code | llm (escaladas) | tú |
|---|---|---|---|
| A · blanco (líquido+CDT+RQM) | 0.989 ± 0.001 | 0.931 ± 0.006 | 1.000 ± 0.000 |
| B1 · núcleo congelado + lectura | 0.986 ± 0.000 | 0.897 ± 0.002 | 1.000 ± 0.000 |
| B2 · núcleo congelado + capa líquido/CDT/RQM | 0.990 ± 0.001 | 0.931 ± 0.008 | 1.000 ± 0.000 |
| C · ajuste fino del núcleo compartido | 0.989 ± 0.001 | 0.929 ± 0.006 | 1.000 ± 0.000 |
| A sin CDT | 0.989 ± 0.001 | 0.930 ± 0.006 | 1.000 ± 0.000 |
| A sin RQM | 0.989 ± 0.001 | 0.930 ± 0.004 | 1.000 ± 0.000 |
| sin campo (logística cruda) | 0.986 ± 0.000 | 0.898 ± 0.004 | 1.000 ± 0.000 |

### Interferencia sobre el campo del chat, prompt real y maestro

| brazo | chat exactitud antes → después | máx. |Δ score| chat | escaladas del ejemplo resueltas % | acuerdo con maestro | exactitud campo en subset del maestro | entrenamiento s |
|---|---|---|---|---|---|---|
| A · blanco (líquido+CDT+RQM) | 1.000 ± 0.000 → 1.000 ± 0.000 | 0.000 ± 0.000 | 55.0 ± 8.9 | 0.349 ± 0.047 (n=70) | 0.936 ± 0.023 | 24.4 ± 0.5 |
| B1 · núcleo congelado + lectura | 1.000 ± 0.000 → 1.000 ± 0.000 | 0.000 ± 0.000 | 50.0 ± 0.0 | 0.349 ± 0.023 (n=70) | 0.908 ± 0.010 | 4.6 ± 0.1 |
| B2 · núcleo congelado + capa líquido/CDT/RQM | 1.000 ± 0.000 → 1.000 ± 0.000 | 0.000 ± 0.000 | 54.0 ± 7.3 | 0.351 ± 0.043 (n=70) | 0.928 ± 0.030 | 24.5 ± 0.6 |
| C · ajuste fino del núcleo compartido | 1.000 ± 0.000 → 0.100 ± 0.094 | 8.499 ± 0.085 | 57.0 ± 10.8 | 0.343 ± 0.053 (n=70) | 0.936 ± 0.027 | 25.4 ± 0.6 |
| A sin CDT | 1.000 ± 0.000 → 1.000 ± 0.000 | 0.000 ± 0.000 | 55.0 ± 9.5 | 0.349 ± 0.045 (n=70) | 0.936 ± 0.023 | 23.3 ± 0.4 |
| A sin RQM | 1.000 ± 0.000 → 1.000 ± 0.000 | 0.000 ± 0.000 | 55.0 ± 8.9 | 0.349 ± 0.047 (n=70) | 0.936 ± 0.027 | 11.9 ± 0.4 |
| sin campo (logística cruda) | 1.000 ± 0.000 → 1.000 ± 0.000 | 0.000 ± 0.000 | 49.0 ± 2.0 | 0.357 ± 0.018 (n=70) | 0.904 ± 0.008 | 3.9 ± 0.1 |

Maestro (gemma-2-2b-it-Q3_K_L.gguf) solo: {"holdout_words":50,"median_seconds_per_word":10.863405494,"router":{"escalated":50,"ok":{"accuracy":0.34,"brier":0.4586220400000001,"ece":0.53944,"n":50},"pending_pct":96.0,"resolved_pct":4.0,"resolved_precision":1.0,"wrong_resolved_pct":0.0},"sample_resolved_pct":20.0,"sample_words":20}

Datos crudos: [`data/spider_field_substrate_results.json`](data/spider_field_substrate_results.json).

### Lectura de los números

- **El campo supera a la heurística** en las escaladas: exactitud de `ok`
  0.931 vs 0.788 (H1) y 0.883 vs 0.710 (H2, vocabulario nuevo); resuelve más
  (79 % vs 74 %) con menos errores (1.8 % vs 4.1 % de escaladas resueltas mal
  en H1; 4.9 % vs 9.0 % en H2).
- **La ganancia la aporta el líquido**: sin campo 0.898 → A 0.931 (+0.033,
  ≫ desviación). **CDT y RQM no mueven la exactitud** de forma medible (A sin
  CDT 0.930, A sin RQM 0.930, dentro de ±0.006). Se mantienen por diseño:
  CDT da la guardia de novedad y ambos son la vía de consolidación en el
  sueño; RQM duplica el tiempo de entrenamiento (11.9 s → 24.4 s) sin coste
  de latencia.
- **B (núcleo congelado)**: B1 (solo el núcleo del chat) ≈ sin campo
  (0.897): el campo del chat (8 conceptos relacionales) **no aporta
  información** sobre las preguntas del Spider. B2 empata con A (0.931) — la
  ganancia es la capa propia, no el acoplamiento.
- **C (ajuste fino compartido)**: misma exactitud (0.929) pero **olvido
  catastrófico** del chat: exactitud relacional 1.000 → 0.100, |Δscore| 8.5.
- **Maestro Gemma**: con el piso 0.70 resuelve 4 % de las escaladas de H1
  (exactitud de `ok` vs verdad 0.34, ~10.9 s/palabra); el acuerdo del campo
  con el maestro es bajo (~0.35) porque Gemma 2B casi nunca supera el piso,
  no porque el campo falle: en ese mismo subconjunto el campo acierta 0.936
  frente a la verdad.
- **Latencia**: ~32 µs por decisión (A) frente a ~10.9 s por palabra con
  Gemma local en CPU.

### Decisión

**Por defecto: A · sustrato en blanco (líquido + CDT + RQM/EPR).** Empata con
la mejor variante acoplada (B2) en exactitud, calibración y % resuelto, no
depende del estado del chat (que no se restaura de checkpoints al arrancar)
y tiene **interferencia cero**. B1 no aporta y C destruye el chat, así que
se descartan. La capa externa (B2) queda soportada en código
(`Coupling::FrozenCore`) pero sin beneficio medido.

## 5. Uso (UI)

1. Pestaña **Prompt Spider** → tarjeta **Entrenamiento Prompt Spider** (al
   final): elegir datasets (6 por defecto), prompts/dataset (40), maestro LLM
   (opcional) y **Entrenar**. La tabla muestra el hold-out tras cada sueño.
2. Activar **Decoder del campo** en la barra de controles (se recuerda) y
   **Ejecutar**. El nombre del escalado pasa a `campo`; las resueltas muestran
   `resolved_by = campo` y la `P(ok)` calibrada; el resto queda en
   **pendientes** con el motivo del campo.
3. Aprobar/Rechazar alimenta la vigilia del campo; la pestaña **Sueño** (o el
   siguiente entrenamiento) la consolida.

API: `POST /api/spider/start {"field_decoder": true}`; métricas en
`summary.decider/field_escalated/field_resolved/field_latency_us` (ver
[`prompt_spider_metricas.md`](prompt_spider_metricas.md) §7).

## 6. Limitaciones

- La verdad de los sintéticos la define el generador: H1/H2 miden
  generalización dentro de esa gramática (H2 con vocabulario disjunto), no
  sobre prompts reales arbitrarios. En el prompt de ejemplo el campo resuelve
  ~55 % de las escaladas sin verdad con la que puntuarlas.
- La ECE de `ok` en escaladas (~0.08) es mayor que en todas las palabras
  (~0.06): las escaladas son la parte difícil y la temperatura se ajusta con
  todas las palabras.
- CDT/RQM no mejoran la exactitud medida; su valor es la consolidación y la
  guardia de novedad, no el acierto.
- La repetición no se guarda: tras reiniciar, el primer dataset reajusta con
  sus propios ejemplos y el sueño de la pestaña Sueño se difiere hasta tener
  ≥ 400.
- El maestro Gemma 2B (Q3_K_L, CPU) es muy conservador y lento; las etiquetas
  de maestro son pocas (70).
- Los tests web (`--features web`) no corren en CI; se ejecutaron en local.
