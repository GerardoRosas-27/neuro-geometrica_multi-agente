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
