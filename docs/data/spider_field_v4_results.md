Semillas: [11, 22, 33, 44, 55] · prompts de entrenamiento/semilla: 240 en 6 datasets · ruido de etiquetas 5 % · hold-out H1 6854 palabras (1123 escaladas), H2 vocabulario desplazado 6689 (949 escaladas), prompt de ejemplo 20 escaladas · piso 0.70 · umbral 0.95

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
