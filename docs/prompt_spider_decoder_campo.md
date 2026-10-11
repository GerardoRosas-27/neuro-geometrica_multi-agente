# Prompt Spider · Decoder del campo (arquitectura v4)

Con el interruptor **Decoder del campo** activo, las palabras **escaladas**
(`p < umbral`) las decide el router del campo entrenado para el Spider **en vez
del LLM**. Código, el scorer determinista y la cola de aprobación no cambian.

Desde esta versión el router corre sobre la **arquitectura v4** (receta §16) y
se eliminaron todos los módulos legacy (`liquid_cdt_*`, `native_thermo*`,
`relational_field`, `wave_predict_core`, `entanglement`, `native_rng`) y la
dependencia `num-complex`. La versión anterior (sustrato «A»: reservorio LTC +
CDT termodinámica + RQM/EPR) se documenta solo como referencia numérica.

Código: [`src/spider_field.rs`](../src/spider_field.rs) (router puro),
[`src/web/spider_field_job.rs`](../src/web/spider_field_job.rs) (estado,
entrenamiento, endpoints), [`src/spider_field_experiment.rs`](../src/spider_field_experiment.rs)
(experimento), integración en [`src/web/spider_job.rs`](../src/web/spider_job.rs).

## 1. Arquitectura del router (v4, §16)

1. **Rasgos de decisión** (`encode_decision` → `decision_context`): rasgos
   locales del token (tipo, léxicos del scorer, mayúscula, longitud,
   frecuencia, sección, puntuación vecina, hashing de identidad/sufijo),
   media de la ventana ±3 y rasgos globales (4 puntuaciones heurísticas, `p`,
   pregunta decisiva, ruta inicial).
2. **Vigilia**: cada ejemplo etiquetado entra en la CDT del Spider
   (experiencia) y en la repetición.
3. **Sueño = consolidación → teacher**: lectura logística por cabeza
   (relevant, grounded, ambiguous, needs_approval, ok); la **máscara de
   confianza** marca pares ejemplo×cabeza donde el teacher acierta con margen.
4. **Dφ (§16)**: `Dphi::gated_h` (cabeza local-lineal con compuerta)
   transporta el estado 2-D de cada cabeza `q₀=(0.5,1)` a `(P,1)` condicionado
   por el contexto; entrenamiento one-step (Adam, coseno) con 50 % etiquetas
   reales + 50 % **replay de confianza**. Teacher y CDT se descartan tras el
   sueño: la inferencia no consulta memoria.
5. **Calibración**: temperatura por cabeza en un split por prompt → `P(ok)`
   calibrada.
6. **Guardia OOD/novedad**: radio estandarizado del contexto contra el
   cuantil 99.5 % visto en el sueño (solo estadísticas; sin módulos legacy).
7. **Router**: resuelve (`resolved_by = campo`) solo si está entrenado,
   dentro de distribución, `P(aprob) < 0.5`, `P(ambigua) < 0.5` y
   `P(ok) ≥ piso` (0.70). Si no → **pendiente** con motivo. **Nunca inventa**:
   sin entrenar, todo escalado queda pendiente.

Persistencia: `data/spider_field/model.json` (`MODEL_VERSION = 2`; un modelo
v1 legacy no se carga y hay que reentrenar).

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

## 4. Re-medición en hold-out (v4 vs versión anterior A)

Mismo protocolo que la versión anterior: semillas 11, 22, 33, 44, 55; 240
prompts sintéticos/semilla en 6 datasets; ruido 5 %; hold-out H1 (vocabulario
visto) y H2 (vocabulario desplazado); maestro Gemma 2 2B-it Q3_K_L en 70
escaladas. Reproducir (≈8 min en 8 núcleos):

```bash
SPIDER_TEACHER=docs/data/spider_field_teacher_gemma.json \
SPIDER_FIELD_OUT=docs/data/spider_field_v4_results \
cargo test --release --lib spider_field_substrate_experiment -- --ignored --nocapture
```

Re-ejecutado dos veces (sesión anterior y esta): resultados idénticos salvo
latencia/tiempo.

### Comparación con la versión anterior (A, líquido+CDT+RQM), escaladas

| métrica | A (anterior) | **v4 (por defecto)** | v4 4× pasos Dφ | teacher solo |
|---|---|---|---|---|
| H1 exactitud `ok` | **0.931** | 0.898 ± 0.028 | 0.978 ± 0.022 | 0.930 ± 0.008 |
| H1 ECE `ok` | **0.079** | 0.123 ± 0.010 | 0.105 ± 0.019 | 0.091 ± 0.013 |
| H1 resuelto % | 79.1 | **80.4 ± 3.3** | 81.5 ± 1.3 | 78.3 ± 1.3 |
| H1 resuelto erróneo % | 1.8 | **1.4 ± 0.7** | 0.5 ± 0.6 | 1.1 ± 0.4 |
| H2 exactitud `ok` | **0.883** | 0.846 ± 0.036 | 0.955 ± 0.023 | 0.880 ± 0.011 |
| latencia / decisión | ~32 µs | ~70 µs | ~70 µs | ~1 µs |
| entrenamiento (5 sueños) | 11.9–24.4 s | ~15 s | ~38 s | ~7 s |

**Lectura honesta.** Con el presupuesto pre-registrado, el router v4 por
defecto es **peor** que A en exactitud de `ok` (−0.033 H1, −0.037 H2) y en
calibración (ECE 0.123 vs 0.079), aunque resuelve algo más y **se equivoca
menos al resolver** (1.4 % vs 1.8 %). Las diferencias en exactitud están en
torno a 1 desviación típica entre semillas. El teacher logístico solo ya
iguala a A; Dφ con el presupuesto por defecto queda por debajo de su propio
teacher (infraentrenado). Con 4× pasos de Dφ, v4 supera a A en todo salvo ECE
(0.105 vs 0.079), a ~2.5× tiempo de entrenamiento. Ese brazo se midió en este
mismo hold-out, por lo que **no** se adopta como defecto aquí (sería ajustar
sobre el test); es la candidata obvia para un próximo pre-registro con
semillas nuevas. La guardia OOD casi no cambia las cifras en H1/H2 (marca
≈0.4 % de escaladas como pendientes).

### Tablas completas (v4)

### Router en escaladas (H1, en distribución)

| brazo | exactitud ok | ECE ok | Brier ok | resuelto % | precisión resuelto | resuelto erróneo % | pendiente % | latencia µs (mediana) |
|---|---|---|---|---|---|---|---|---|
| heurística (scorer, piso) | 0.788 | 0.109 | 0.116 | 73.8 | 0.945 | 4.1 | 26.2 | — |
| V4 · Dφ gated + replay de confianza + OOD | 0.898 ± 0.028 | 0.123 ± 0.010 | 0.070 ± 0.010 | 80.4 ± 3.3 | 0.982 ± 0.009 | 1.4 ± 0.7 | 19.6 ± 3.3 | 70.3 ± 4.2 |
| V4 replay completo (sin máscara) | 0.912 ± 0.018 | 0.108 ± 0.012 | 0.062 ± 0.004 | 78.6 ± 1.6 | 0.988 ± 0.007 | 0.9 ± 0.6 | 21.4 ± 1.6 | 70.4 ± 3.3 |
| V4 sin guardia OOD | 0.898 ± 0.028 | 0.123 ± 0.010 | 0.070 ± 0.010 | 80.8 ± 3.2 | 0.982 ± 0.009 | 1.5 ± 0.7 | 19.2 ± 3.2 | 71.5 ± 5.2 |
| teacher logístico (sin Dφ) | 0.930 ± 0.008 | 0.091 ± 0.013 | 0.059 ± 0.001 | 78.3 ± 1.3 | 0.986 ± 0.005 | 1.1 ± 0.4 | 21.7 ± 1.3 | 1.0 ± 0.0 |
| V4 con 4× pasos de Dφ | 0.978 ± 0.022 | 0.105 ± 0.019 | 0.032 ± 0.009 | 81.5 ± 1.3 | 0.994 ± 0.007 | 0.5 ± 0.6 | 18.5 ± 1.3 | 66.2 ± 0.9 |

### Router en escaladas (H2, vocabulario desplazado)

| brazo | exactitud ok | ECE ok | resuelto % | precisión resuelto | resuelto erróneo % |
|---|---|---|---|---|---|
| heurística | 0.710 | 0.092 | 70.2 | 0.872 | 9.0 |
| V4 · Dφ gated + replay de confianza + OOD | 0.846 ± 0.036 | 0.126 ± 0.018 | 76.1 ± 4.0 | 0.950 ± 0.017 | 3.9 ± 1.5 |
| V4 replay completo (sin máscara) | 0.873 ± 0.021 | 0.108 ± 0.020 | 74.1 ± 1.1 | 0.957 ± 0.008 | 3.2 ± 0.6 |
| V4 sin guardia OOD | 0.846 ± 0.036 | 0.126 ± 0.018 | 76.5 ± 3.9 | 0.950 ± 0.017 | 3.9 ± 1.5 |
| teacher logístico (sin Dφ) | 0.880 ± 0.011 | 0.083 ± 0.010 | 72.6 ± 3.0 | 0.952 ± 0.013 | 3.5 ± 1.1 |
| V4 con 4× pasos de Dφ | 0.955 ± 0.023 | 0.112 ± 0.025 | 74.6 ± 1.7 | 0.987 ± 0.011 | 0.9 ± 0.8 |

### Por pregunta (todas las palabras de H1): exactitud / ECE / Brier

| brazo | relevant | grounded | ambiguous | needs_approval | ok |
|---|---|---|---|---|---|
| heurística | 0.987 / 0.022 / 0.013 | 0.980 / 0.034 / 0.012 | 0.987 / 0.038 / 0.008 | 1.000 / 0.005 / 0.002 | 0.954 / 0.031 / 0.033 |
| V4 · Dφ gated + replay de confianza + OOD | 0.990 / 0.066 / 0.011 | 0.993 / 0.062 / 0.009 | 0.994 / 0.068 / 0.010 | 1.000 / 0.060 / 0.005 | 0.973 / 0.075 / 0.022 |
| V4 replay completo (sin máscara) | 0.990 / 0.070 / 0.011 | 0.995 / 0.067 / 0.009 | 0.992 / 0.068 / 0.010 | 1.000 / 0.062 / 0.006 | 0.975 / 0.073 / 0.021 |
| V4 sin guardia OOD | 0.990 / 0.066 / 0.011 | 0.993 / 0.062 / 0.009 | 0.994 / 0.068 / 0.010 | 1.000 / 0.060 / 0.005 | 0.973 / 0.075 / 0.022 |
| teacher logístico (sin Dφ) | 0.990 / 0.059 / 0.010 | 0.997 / 0.055 / 0.006 | 0.997 / 0.059 / 0.006 | 1.000 / 0.048 / 0.003 | 0.978 / 0.063 / 0.019 |
| V4 con 4× pasos de Dφ | 0.999 / 0.058 / 0.006 | 0.998 / 0.057 / 0.005 | 0.996 / 0.061 / 0.007 | 1.000 / 0.055 / 0.004 | 0.995 / 0.065 / 0.010 |

### Exactitud de `ok` por ruta inicial (H1)

| brazo | code | llm (escaladas) | tú |
|---|---|---|---|
| V4 · Dφ gated + replay de confianza + OOD | 0.988 ± 0.002 | 0.898 ± 0.028 | 1.000 ± 0.000 |
| V4 replay completo (sin máscara) | 0.987 ± 0.002 | 0.912 ± 0.018 | 1.000 ± 0.000 |
| V4 sin guardia OOD | 0.988 ± 0.002 | 0.898 ± 0.028 | 1.000 ± 0.000 |
| teacher logístico (sin Dφ) | 0.987 ± 0.000 | 0.930 ± 0.008 | 1.000 ± 0.000 |
| V4 con 4× pasos de Dφ | 0.998 ± 0.002 | 0.978 ± 0.022 | 1.000 ± 0.000 |

### Prompt real y maestro

| brazo | escaladas del ejemplo resueltas % | acuerdo con maestro | exactitud campo en subset del maestro | entrenamiento s |
|---|---|---|---|---|
| V4 · Dφ gated + replay de confianza + OOD | 50.0 ± 10.5 | 0.317 ± 0.048 (n=70) | 0.908 ± 0.037 | 15.4 ± 0.9 |
| V4 replay completo (sin máscara) | 41.0 ± 6.6 | 0.354 ± 0.040 (n=70) | 0.920 ± 0.022 | 14.7 ± 0.2 |
| V4 sin guardia OOD | 54.0 ± 13.2 | 0.311 ± 0.051 (n=70) | 0.908 ± 0.037 | 14.9 ± 0.4 |
| teacher logístico (sin Dφ) | 46.0 ± 4.9 | 0.360 ± 0.025 (n=70) | 0.936 ± 0.029 | 6.2 ± 0.2 |
| V4 con 4× pasos de Dφ | 44.0 ± 11.1 | 0.309 ± 0.033 (n=70) | 0.988 ± 0.016 | 38.3 ± 3.0 |

Maestro (gemma-2-2b-it-Q3_K_L.gguf) solo: {"holdout_words":50,"median_seconds_per_word":10.863405494,"router":{"escalated":50,"ok":{"accuracy":0.34,"brier":0.4586220400000001,"ece":0.53944,"n":50},"pending_pct":96.0,"resolved_pct":4.0,"resolved_precision":1.0,"wrong_resolved_pct":0.0},"sample_resolved_pct":20.0,"sample_words":20}


Datos crudos: [`data/spider_field_v4_results.json`](data/spider_field_v4_results.json).
Los números de A provienen de la versión anterior de este documento
(experimento de sustrato A/B/C; código eliminado).

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
  generalización dentro de esa gramática, no sobre prompts reales.
- v4 por defecto es menos exacto y peor calibrado que A (ver §4); el beneficio
  de esta versión es arquitectónico (una sola arquitectura v4, ~6 000 líneas
  legacy menos, inferencia sin memoria), no de métricas.
- La repetición no se guarda: tras reiniciar, el sueño de la pestaña Sueño se
  difiere hasta tener ≥ 400 ejemplos.
- El maestro Gemma 2B (Q3_K_L, CPU) es muy conservador y lento; pocas
  etiquetas (70).
- Los tests web (`--features web`) no corren en CI; se ejecutaron en local.
