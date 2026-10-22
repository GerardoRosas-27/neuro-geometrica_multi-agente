# Resultados — Cognitive Transformer + ÓRBITA (atención estado↔memoria CDT-RQM)

Prerregistro `docs/prereg_cognitive_transformer_orbita.md` (`5634286`), empujado antes de ejecutar DEV/CONFIRM. Diseño `docs/cognitive_transformer_orbita.md` (`3dfb95c`).
Código `orbita/cto/` (`cto.py`, `analyze_cto.py`, `chain.sh`) sin cambios entre DEV y CONFIRM. Una sola cadena `setsid nohup chain.sh` (4 trabajos, `nice 10`), hora CST (UTC−6):
calib 2026-10-20 13:23 → train DEV 13:23 → 2026-10-21 11:59 → flujo DEV → análisis DEV 12:23 → train CONFIRM 14:15 → flujo CONFIRM → análisis CONFIRM 15:06 → `chain end` 15:06.
64/64 redes por fase (4 brazos entrenados × 16 seeds), 16/16 flujos por fase, `analyze_*.err` vacíos. sha256 de ÓRBITA idéntico antes/después 16/16 en ambas fases; 0 queries a TEST.
τ_k (P95 sorpresa ÓRBITA, H DEV fam 1–2): τ₂ = 0.3792, τ₄ = 0.3710, τ₈ = 0.3691.
Tablas completas: `artifacts/orbita/cognitive_transformer/summary_{dev,confirm}.md` (re-ejecutado `analyze_cto.py` el 2026-10-21: salida y `res_*.json` idénticos byte a byte).
Todas las cifras de este documento salen de esas tablas / `res_*.json`, salvo las marcadas **[diagnóstico post hoc]** (ver §5), que no son prerregistradas y no entran en ningún veredicto.

## 1. Veredicto global
- **El Cognitive Transformer NO mejora a ÓRBITA en nada de lo prerregistrado. La rama fracasa** (éxito = H1 ∧ H7 ∧ H6: ninguna se acepta).
- **La consolidación CDT-RQM, tal como está implementada, destruye la predicción**: con banco vacío (n = 0) CT queda algo por debajo de ÓRBITA en conocidas
  (CONFIRM acc16 0.1729 vs 0.2427), y en cuanto el banco tiene episodios consolidados (n = 150, ~20 episodios) CT cae a **0.0000** en conocidas y se queda ahí hasta n = 2400.
  H1 refutada (−0.2427 [−0.2744, −0.2134]) y H9 refutada (olvido −0.1729 [−0.1987, −0.1484], superior < −δ).
- **En familias nuevas 8–11 hay suelo: todos los brazos aprendidos y ÓRBITA dan 0.0000 en acc16 (k = 2, CONFIRM)**, así que H2, H3, H7 y las partes «nuevas» de H4–H6
  dan diferencias exactamente 0 con IC95 degenerado [0, 0] → inconclusas por construcción (no hay señal que medir). Sólo ORACLE_EP sale del suelo (0.0781).
- El único criterio que se cumple es **H10** (AUROC de sorpresa CT n=2400 − n=0 < 0 en DEV y CONFIRM), pero **por el mecanismo contrario al hipotetizado**: lo nuevo no se vuelve
  menos sorprendente (su sorpresa CT mediana sube), lo conocido se vuelve mucho más sorprendente porque CT colapsa. Formalmente aceptada, sustantivamente vacía.
- No es un bug de cálculo/indexado en la métrica primaria (verificado, §5); es un fallo de diseño real: la memoria consolidada por sorpresa sólo contiene dinámicas ajenas,
  la recuperación top-R no tiene umbral, y la atención (entrenada con un banco homogéneo de familias conocidas) confía en ella. Además, el entrenamiento de 4000 pasos no consiguió
  que ningún módulo nuevo superase a ÓRBITA congelada ni siquiera en su propia distribución de entrenamiento.

## 2. Hipótesis (primario k = 2, n = 2400, acc16; media de 16 seeds [IC95 bootstrap pareado, 10 000 réplicas, semilla 12345]; δ = 0.05)
Regla del prerregistro: **aceptada** si el criterio se cumple en DEV **y** en CONFIRM; **refutada** si en CONFIRM el IC95 queda entero del lado contrario (< 0; en H9 superior < −δ; en H10 > 0);
**inconclusa** en otro caso. Para hipótesis compuestas (H4, H5, H6, H7, H8: todos los IC95 > 0) basta un componente refutado para refutar la conjunción.

| id | cantidad | criterio | DEV | CONFIRM | veredicto |
|---|---|---|---|---|---|
| **H1** | CT − ORB, conocidas 0–7 | IC95 > 0 | −0.3247 [−0.3608, −0.2856] | −0.2427 [−0.2744, −0.2134] | **refutada** |
| **H2** | CT − ORB, nuevas 8–11 | IC95 > 0 | −0.0010 [−0.0029, +0.0000] | +0.0000 [+0.0000, +0.0000] | **inconclusa** (suelo: ambos 0) |
| **H3** | CT n=2400 − CT n=0, nuevas | IC95 > 0 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | **inconclusa** (suelo) |
| **H4** | CT − CT_SHUF, nuevas | IC95 > 0 | +0.0000 [0, 0] | +0.0000 [0, 0] | — |
|  | CT − CT_RAND, nuevas | IC95 > 0 | +0.0000 [0, 0] | +0.0000 [0, 0] | — |
|  | CT − CT_SHUF, conocidas | IC95 > 0 | +0.0000 [0, 0] | +0.0000 [0, 0] | — |
|  | CT − CT_RAND, conocidas | IC95 > 0 | −0.0044 [−0.0093, −0.0005] | −0.0029 [−0.0054, −0.0010] | **H4 refutada** (este componente entero < 0) |
| **H5** | CT − SELF, nuevas | IC95 > 0 | +0.0000 [0, 0] | +0.0000 [0, 0] | — |
|  | CT − SELF, conocidas | IC95 > 0 | −0.0020 [−0.0039, −0.0005] | −0.0024 [−0.0044, −0.0010] | **H5 refutada** (ver salvedad SELF, §5.3) |
| **H6** | CT − MLP, nuevas | IC95 > 0 | +0.0000 [0, 0] | +0.0000 [0, 0] | — |
|  | CT − MLP, conocidas | IC95 > 0 | −0.0044 [−0.0083, −0.0015] | −0.0020 [−0.0044, +0.0000] | **H6 inconclusa** (CONFIRM sup = 0 exacto, no entero < 0) |
| **H7** | CT − NOORB, nuevas | IC95 > 0 | +0.0000 [0, 0] | +0.0000 [0, 0] | — |
|  | CT − NOORB, conocidas | IC95 > 0 | +0.0000 [0, 0] | +0.0000 [0, 0] | **H7 inconclusa** (ambos brazos en 0) |
| **H8** | CT − ORACLE_EP, nuevas | IC95 > 0 | +0.0000 [0, 0] | −0.0781 [−0.0781, −0.0781] | **H8 refutada** |
|  | CT − ORACLE_KNN, nuevas | IC95 > 0 | +0.0000 [0, 0] | +0.0000 [0, 0] | — |
| **H9** | olvido: CT n=2400 − n=0, conocidas | IC95 inf > −δ | −0.2329 [−0.2642, −0.2031] | −0.1729 [−0.1987, −0.1484] | **refutada** (sup < −δ: olvido masivo) |
| **H10** | AUROC sorpresa CT n=2400 − n=0 | IC95 < 0 | −0.0775 [−0.1216, −0.0456] | −0.0704 [−0.1064, −0.0408] | **aceptada formalmente** (mecanismo contrario, §4) |

Criterio del documento §24 («Attention+ÓRBITA > Attention ≥≈ ÓRBITA > baseline»): CT − NOORB = 0 (ambos en 0), NOORB − ORB conocidas CONFIRM −0.2427 [−0.2744, −0.2134]
(DEV −0.3247 [−0.3608, −0.2856]), CT − MLP conocidas −0.0020 [−0.0044, +0.0000]. No se cumple ninguna desigualdad del §24. **Éxito de la rama: NO.**

Nota sobre H8 (ORACLE_EP): el IC95 degenerado [−0.0781, −0.0781] es correcto: ORACLE_EP no depende de la seed (ajuste afín sobre los mismos k pasos del conjunto H fijo por fase,
sin contexto ni pesos), así que da 0.0781 en las 16 seeds de CONFIRM y 0.0000 en las de DEV; CT da 0 en todas.

Secundarios k = 4 y k = 8 (CONFIRM, sin criterio): mismo patrón. H1 −0.2729 [−0.3066, −0.2412] (k=4), −0.3579 [−0.3906, −0.3262] (k=8); H9 −0.2007 [−0.2334, −0.1689] (k=4),
−0.3071 [−0.3345, −0.2783] (k=8); H10 −0.0623 [−0.0962, −0.0368] (k=4), −0.0536 [−0.0844, −0.0310] (k=8); en nuevas todas las diferencias CT − X = 0.0000 salvo vs ORACLE_EP (−0.0938).

## 3. Precisión por brazo (k = 2, acc16, media de 16 seeds; «n = 2400 / n = 0»)

| brazo | DEV conocidas | DEV nuevas | CONFIRM conocidas | CONFIRM nuevas | CONFIRM conocidas 1–2 | CONFIRM rel_err16 mediana conocidas |
|---|---|---|---|---|---|---|
| ORB | 0.3247 / 0.3247 | 0.0010 / 0.0010 | 0.2427 / 0.2427 | 0.0000 / 0.0000 | 0.3555 / 0.3555 | 0.1154 |
| CT | 0.0000 / 0.2329 | 0.0000 / 0.0000 | **0.0000 / 0.1729** | 0.0000 / 0.0000 | 0.0000 / 0.2324 | 0.9469 |
| CT_NOCDT | 0.2329 / 0.2329 | 0.0000 / 0.0000 | 0.1729 / 0.1729 | 0.0000 / 0.0000 | 0.2324 / 0.2324 | 0.1299 |
| CT_RAND | 0.0044 / 0.0088 | 0.0000 / 0.0000 | 0.0029 / 0.0049 | 0.0000 / 0.0000 | 0.0020 / 0.0000 | 0.5425 |
| CT_SHUF | 0.0000 / 0.1050 | 0.0000 / 0.0000 | 0.0000 / 0.0791 | 0.0000 / 0.0000 | 0.0000 / 0.1152 | 0.9704 |
| SELF | 0.0020 / 0.0020 | 0.0000 / 0.0000 | 0.0024 / 0.0024 | 0.0000 / 0.0000 | 0.0059 / 0.0059 | 0.4151 |
| MLP | 0.0044 / 0.0557 | 0.0000 / 0.0000 | 0.0020 / 0.0571 | 0.0000 / 0.0000 | 0.0000 / 0.0840 | 0.4450 |
| NOORB | 0.0000 / 0.0005 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.9374 |
| ORACLE_EP | 0.8828 / 0.8828 | 0.0000 / 0.0000 | 0.8750 / 0.8750 | 0.0781 / 0.0781 | 1.0000 / 1.0000 | 0.0000 |
| ORACLE_KNN | 0.0010 / 0.5068 | 0.0000 / 0.0000 | 0.0010 / 0.4438 | 0.0000 / 0.0068 | 0.0000 / 0.8477 | 0.9874 |

CT por punto de control (k = 2, conocidas): DEV 0.2329 → 0.0005 → 0.0000 → 0.0000; CONFIRM 0.1729 → 0.0000 → 0.0000 → 0.0000 (n = 0, 150, 600, 2400).
Banco CDT final: DEV media 290.6 episodios, fracción de familias nuevas 0.8045; CONFIRM 284.4 episodios, 0.7165.

**Sorpresa / AUROC (nuevas vs conocidas, k = 2):**

| | DEV | CONFIRM |
|---|---|---|
| ÓRBITA | 0.9396 [0.9173, 0.9601] | 0.9144 [0.8968, 0.9321] |
| CT n = 0 | 0.9421 [0.9237, 0.9599] | 0.9124 [0.8960, 0.9293] |
| CT n = 2400 | 0.8647 [0.8204, 0.9035] | 0.8419 [0.8018, 0.8766] |

Coste (CONFIRM): parámetros entrenables CT 4744, SELF 4392, MLP 4755, NOORB 5442 (ÓRBITA congelada 2602). Entrenamiento CT 717 s/seed, NOORB 673, SELF 140, MLP 92;
evaluación + flujo 763 s/seed. (En DEV los tiempos son mayores —CT 9372 s/seed— por compartir CPU con `exp/orbita-tf`.)

## 4. Interpretación
1. **La memoria consolidada envenena la predicción.** El mismo CT pasa de 0.1729 (banco vacío) a 0.0000 (banco con episodios consolidados) en conocidas. CT_NOCDT (mismos pesos, sin CDT)
   conserva 0.1729 a n = 2400, lo que aísla la causa en el contenido recuperado del banco. El oráculo no aprendido ORACLE_KNN, que usa **las mismas entradas** de memoria con un núcleo
   gaussiano fijo, sufre exactamente el mismo colapso (0.4438 → 0.0010): no es un defecto de la red de atención sino del **contenido de la memoria y de la recuperación**.
2. **Por qué el banco es dañino.** La regla prerregistrada consolida sólo episodios con sorpresa ÓRBITA > τ: el banco se llena de familias nuevas (72–80 %) y de episodios conocidos que
   ÓRBITA predice mal. La recuperación es top-R = 4 por distancia de firma **sin umbral**, así que todo episodio de H recibe 4 episodios ajenos, con residuos e_j grandes y de otra dinámica.
   La atención se entrenó con un banco de 512 episodios de familias conocidas sin filtrar por sorpresa (mismas dinámicas que la consulta), y aprendió a confiar en él;
   con el banco del flujo pone la mayor parte de su masa en esas entradas ajenas [diagnóstico post hoc §5.2] y el Δz se dispara.
3. **Familias nuevas: suelo absoluto.** Con k = 2 ni ÓRBITA ni ningún brazo aprendido acierta una sola trayectoria nueva a 16 pasos en CONFIRM; todas las diferencias son 0 exacto.
   La consolidación no lo cambia (H3 = 0). No podemos decir que la memoria «no ayuda» en nuevas de forma refutada; sólo que no hay ninguna señal y que el resultado es inconcluso.
4. **H10 es un artefacto del colapso.** [diagnóstico post hoc, sobre `eval_*.csv`] Sorpresa CT mediana (k = 2, CONFIRM): conocidas 0.011 (n=0) → 0.240 (n=2400); nuevas 0.188 → 0.371
   (DEV: conocidas 0.009 → 0.242; nuevas 0.210 → 0.405). Lo nuevo es **más** sorprendente con memoria, no menos; el AUROC baja porque lo conocido sube aún más. La hipótesis del
   documento («la memoria hace menos sorprendente lo nuevo») no recibe apoyo, aunque el criterio numérico prerregistrado se cumpla.
5. **Sin atención y sin memoria tampoco se gana.** SELF, MLP y NOORB quedan muy por debajo de ÓRBITA (≤ 0.06 en conocidas). NOORB (dinámica libre) queda en 0: ÓRBITA es imprescindible
   como núcleo, pero la atención añadida encima no aporta nada medible.
6. Coherente con los experimentos anteriores (ÓRBITA crece, latente largo plazo, ÓRBITA-TF): añadir módulos entrenados sobre/alrededor de ÓRBITA con presupuestos cortos no supera a
   ÓRBITA congelada, y las familias nuevas no afines siguen sin resolverse a acc16 con k ≤ 8.

## 5. Anomalías y comprobación de bug
### 5.1 Lo que se comprobó y no es bug
- **Reproducibilidad:** `analyze_cto.py dev|confirm` re-ejecutado: salidas y `res_*.json` idénticos byte a byte.
- **Ceros exactos en nuevas:** ORB ya da 0.0000 en nuevas (CONFIRM) y 0.0010 en DEV; todos los brazos aprendidos 0.0000 → diferencias 0 con IC95 [0, 0]. Es un efecto suelo,
  no un fallo del bootstrap.
- **CT n=0 ≡ CT_NOCDT:** idénticos por construcción (a n = 0 el banco del flujo está vacío). Consistente.
- **ORACLE_EP con IC95 degenerado:** no depende de la seed (ver §2). Consistente.
- **Banco reconstruido:** [diagnóstico post hoc] reconstruir el flujo S de la seed 0xB400 da 20 episodios a n = 150, igual que `meta_0xB400.json`.
- **Colapso CT 0.1729 → 0.0000 es genuino** [diagnóstico post hoc, `artifacts/orbita/cognitive_transformer/diag_colapso.{py,txt}`; k = 2, 1 seed por fase, red CT guardada, sin reentrenar]:

  | seed | ORB | CT sin banco | CT + banco tipo entrenamiento (512 ep. conocidas) | CT + banco flujo n=150 | CT + banco flujo n=2400 |
  |---|---|---|---|---|---|
  | CONFIRM 0xB400, contexto Gemma, conocidas | 0.273 | 0.234 | 0.195 | 0.000 | 0.000 |
  | DEV 0xA400, contexto Gemma, conocidas | 0.242 | 0.180 | 0.180 | 0.000 | 0.000 |

  Con un banco de la misma distribución que el de entrenamiento CT funciona casi como sin banco; sólo el banco consolidado por sorpresa lo hunde. Masa de atención sobre entradas CDT
  con el banco n = 150: 0.88 (0xB400) y 0.83 (0xA400); ‖Δz‖ medio en conocidas pasa de 0.12 → 0.77 (0xB400) y 0.25 → 0.58 (0xA400), frente a ‖z‖ ≈ 1.6.
  Es colapso real por desajuste de distribución de la memoria + recuperación sin umbral, no un error de indexado.

### 5.2 Problemas reales encontrados (no afectan al primario, k = 2, h16)
- **Bug en la métrica secundaria h8 con k = 8:** `eval_H` indexa `pr[j, h − k − 1]`; para h = k = 8 el índice es −1 y compara la predicción del paso 256 con x_8. Por eso ORB «conocidas h8»
  sale 0.0000 a k = 8. La columna h8 de las tablas k = 8 no es válida (debería ser N/A: x_8 es observado). h8 a k = 2, 4 y todos los h ≥ 16 son correctos.
- **El entrenamiento de los módulos nuevos no funcionó.** [diagnóstico post hoc] En la propia distribución de entrenamiento (256 episodios nuevos del generador, familias 0–7, k = 2):
  ORB acc16 0.172 / CT (banco tipo entrenamiento) 0.148 / SELF 0.004 en 0xB400; 0.145 / 0.090 / 0.000 en 0xA400. Con W_O inicializada a 0 el punto de partida es ÓRBITA, así que
  4000 pasos Adam (lr 3e-3) empeoraron el modelo incluso dentro de distribución; la pérdida registrada no baja de forma clara (p. ej. CT 0xB400: 0.134 en el paso 250, 0.237 en el 4000;
  SELF 0.370 → 0.452). Probable inestabilidad de optimización a través del rollout de 14–16 pasos y/o pérdida log1p(rel_err²) desalineada con acc16. No es un bug de código
  identificado, pero sí invalida las comparaciones contra SELF/MLP/NOORB como pruebas de «arquitectura»: son comparaciones entre modelos mal entrenados.
- **Desajuste train/test del banco** (diseño, no código): en entrenamiento el banco son 512 episodios sin filtrar de familias conocidas; en el flujo, sólo episodios sorprendentes.
  El prerregistro lo fijó así; el resultado muestra que esa elección es la que provoca el colapso.

### 5.3 Salvedades
- H5 y H6 se refutan/quedan inconclusas comparando CT contra brazos que también están prácticamente en 0; no dicen nada fuerte sobre atención vs auto-atención vs MLP.
- El contexto Gemma (D2/numnorm) rebaja a ÓRBITA respecto al contexto verdadero (diag.: 0.273 vs 0.359 en 0xB400), igual para todos los brazos.

## 6. Limitaciones
- 4000 pasos de entrenamiento (frente a 256 000 de ÓRBITA); sin rejilla de hiperparámetros (prerregistrado así).
- Las entidades no interactúan en estas familias: la atención sólo puede explotar estado↔memoria (riesgo 2 del documento).
- Sin decoder estado→lenguaje, sin paráfrasis nuevas; el LLM sólo entra como contexto ya extraído.
- Los diagnósticos de §5 son post hoc, con 1 seed por fase, y no cambian ningún veredicto.
- El bug h8@k=8 afecta sólo a una columna secundaria.

## 7. Próximos pasos (no prerregistrados; requieren prerregistro nuevo si se ejecutan)
1. **Recuperación con compuerta:** umbral de distancia de firma (o «no recuperar» como opción aprendida) y peso de mezcla CT/ÓRBITA que pueda volver a 0; probar primero que CT con banco
   del flujo no cae por debajo de CT_NOCDT (criterio de no-daño) antes de pedir mejora.
2. **Entrenar con el banco de la misma distribución que el test** (consolidado por sorpresa, con familias ajenas) para que la atención aprenda a ignorar memoria irrelevante.
3. **Arreglar la optimización** antes de comparar arquitecturas: comprobar que cualquier módulo nuevo supera a ÓRBITA en su propia distribución de entrenamiento (lr menor,
   rollout corto con curriculum, pérdida de un paso + rollout); si no lo hace, no tiene sentido evaluar en H.
4. Corregir h8 para k ≥ 8 (marcar N/A).
5. Para familias nuevas el suelo persiste con k = 2: medir con k mayor o métricas continuas (rel_err) para tener señal; la pista de TF-PE en nuevas (exp/orbita-tf) sigue pendiente de prerregistro.
