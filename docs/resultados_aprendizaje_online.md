# Resultados: ÓRBITA aprende dinámicas nuevas en línea (biblioteca de familias aprendidas)

Prerregistro `docs/prereg_orbita_aprendizaje_online.md` (d391036), publicado y empujado antes de ejecutar. Código: `orbita/src/learn.rs`
(`learn-eval`), `orbita/src/fam.rs` (`fam-set --rng`; sin `--rng` reproduce byte a byte el conjunto anterior 0xFA4E), `orbita/llm/learn_eval.py`,
`learn_chain.sh`. Ejecución 2026-10-19 (hora CST, UTC−6): extracción Gemma DEV 01:12:45–01:17:10, CONFIRM 01:17:24–≈01:22, calibración τ
01:22:06, malla 01:22:49, DEV 01:22:52, CONFIRM 01:23:00 (8 núcleos, ≈5,9 GB RAM disponibles; un solo proceso Gemma). 0 queries v4 a TEST
en todas las ejecuciones, 0 parse_fail, sha256 de los pesos ÓRBITA idéntico antes/después en todas las ejecuciones (núcleo congelado intacto).
Tablas completas: `artifacts/orbita/aprendizaje_online/summary_{dev,confirm}.md`; por episodio: `artifacts/orbita/aprendizaje_online/{dev,confirm}/eval/`
(la línea `library=` de cada `.err` lista los módulos con su familia base g0 y las familias verdaderas de sus episodios).

Antes del flujo sólo se corrió una prueba de humo del binario sobre 20 filas del conjunto antiguo (`demo_fam`, seed 0xA400) para medir tiempo
(0,17 s); no se usó para ninguna decisión.

## Calibración y selección (sólo DEV, regla prerregistrada)
- τ (P95 de S_known en familias 1–2 de DEV, 512 episodios): **τ₁ = 0,012439**, **τ₂ = 0,013989**.
- Malla (acc16 media DEV, k = 2): LEARN elige **D = 2, λ = 1e-4** (0,3512; empata con D = 3, λ = 1e-4; regla de empate → D = 2). ORACLE_FAM
  elige D = 2, λ = 1e-4 (0,3621) y ORACLE_EP D = 2, λ = 1e-4 (0,7125). Las tres elecciones caen en el borde inferior de la malla de λ.
  Tabla completa: `artifacts/orbita/aprendizaje_online/grid.log`.

## acc16 (media de 16 seeds [IC95]; k = 2, primario)
| grupo | fase | BASE | OBS2 | LEARN2 | LEARN_EP2 | FINAL2 | ORACLE_FAM2 | ORACLE_EP2 | ORACLE_LS2 |
|---|---|---|---|---|---|---|---|---|---|
| nuevas 8–9 | DEV | 0,0000 | 0,0059 | 0,0293 [0,0254, 0,0312] | 0,0312 | 0,0312 | 0,0059 | 0,0312 | 0,0625 |
| nuevas 8–9 | CONFIRM | 0,0000 | 0,0000 | 0,0371 [0,0293, 0,0449] | 0,0625 | 0,0293 | 0,0098 | 0,0625 | 0,0312 |
| conocidas 1–2 | DEV | 0,3496 | 0,5586 | 0,6055 [0,5508, 0,6582] | 0,6055 | 0,6055 | 0,2207 | 1,0000 | 1,0000 |
| conocidas 1–2 | CONFIRM | 0,2051 | 0,4766 | 0,5684 [0,5195, 0,6113] | 0,5703 | 0,5625 | 0,1973 | 1,0000 | 1,0000 |
| conocidas 0–7 | DEV | 0,3101 | 0,4043 | 0,4316 [0,4014, 0,4604] | 0,4341 | 0,4312 | 0,4512 | 0,8828 | 0,8828 |
| conocidas 0–7 | CONFIRM | 0,2354 | 0,3643 | 0,4053 [0,3818, 0,4282] | 0,4033 | 0,3984 | 0,4365 | 0,8750 | 0,8750 |

acc8 nuevas 8–9 (k = 2): OBS2 0,0039 / 0,0020; LEARN2 0,0293 / 0,1094; LEARN_EP2 0,0391 / 0,1191; ORACLE_EP2 0,0312 / 0,1250; ORACLE_LS2
0,0938 / 0,1562 (DEV / CONFIRM).

## Hipótesis (IC95 bootstrap pareado por seed; DEV / CONFIRM)
| | DEV | CONFIRM | veredicto |
|---|---|---|---|
| **HL1** acc16 LEARN2 − OBS2, nuevas 8–9 (> 0) | +0,0234 [+0,0156, +0,0293] | +0,0371 [+0,0293, +0,0449] | **aceptada** (efecto mínimo, ver lectura) |
| HL1' (k = 1) LEARN1 − OBS1, nuevas | +0,0039 [−0,0039, +0,0117] | +0,0078 [+0,0020, +0,0156] | **no aceptada** (falla en DEV) |
| **HL2** LEARN2 − OBS2, conocidas 0–7 (> −δ) | +0,0273 [+0,0195, +0,0356] | +0,0410 [+0,0322, +0,0503] | **aceptada** |
| **HL2** LEARN2 − OBS2, conocidas 1–2 (> −δ) | +0,0469 [+0,0215, +0,0801] | +0,0918 [+0,0703, +0,1133] | **aceptada** |
| **HL2** FINAL2 − OBS2, conocidas 0–7 (> −δ) | +0,0269 [+0,0195, +0,0352] | +0,0342 [+0,0259, +0,0425] | **aceptada** |
| **HL2** FINAL2 − OBS2, conocidas 1–2 (> −δ) | +0,0469 [+0,0215, +0,0801] | +0,0859 [+0,0645, +0,1074] | **aceptada** (+ sha256 idéntico) |
| **HL3** LEARN2 − LEARN_EP2, nuevas (> 0) | −0,0020 [−0,0059, +0,0000] | −0,0254 [−0,0332, −0,0176] | **rechazada** (en CONFIRM la biblioteca empeora) |
| **HL4** LEARN2 − ORACLE_FAM2, nuevas (> −δ) | +0,0234 [+0,0156, +0,0293] | +0,0273 [+0,0137, +0,0391] | **aceptada** (incluso IC95 > 0) |
| HL4b LEARN2 − ORACLE_EP2, nuevas (reportada) | −0,0020 [−0,0059, +0,0000] | −0,0254 [−0,0332, −0,0176] | LEARN2 no supera al oráculo por episodio |

k = 1 (secundario), CONFIRM: HL2 LEARN1 − OBS1 conocidas 0–7 +0,0093 [+0,0054, +0,0137], 1–2 +0,0293 [+0,0156, +0,0449]; FINAL1 − OBS1 0–7
+0,0044 [+0,0020, +0,0073], 1–2 +0,0156 [+0,0039, +0,0273] (en DEV también > −δ: ver summary). HL3 k = 1: +0,0000 [−0,0059, +0,0059] /
−0,0508 [−0,0586, −0,0430]. HL4 k = 1: +0,0059 [0,0000, +0,0117] / +0,0000 [−0,0098, +0,0098].

## Reportado sin criterio (k = 2; DEV / CONFIRM)
- **Detección por sorpresa**: AUROC de S_known nuevas vs conocidas 0–7 0,9868 [0,9846, 0,9889] / 0,9706 [0,9683, 0,9729]; nuevas vs 1–2
  0,9996 / 0,9938. Tasa de disparo (sorpresa > τ₂): nuevas 1,0000 / 0,9980; conocidas 1–2 0,0508 / 0,1152; v4 no iterables 0,1979 / 0,2253.
- **Biblioteca**: módulos por seed 20,88 [20,31, 21,50] / 19,25 [18,81, 19,69] (de ellos con mayoría de familia nueva 17,06 / 14,19);
  reutilización (episodios nuevos asignados a un módulo existente) 0,4004 / 0,5000; pureza 0,6097 / 0,5664. Es decir, con sólo 2 familias nuevas
  la biblioteca se fragmenta en muchos módulos y mezcla familias.
- **Curva de aprendizaje** acc16 LEARN2 en nuevas por orden de aparición (1–4 / 5–8 / 9–16): DEV 0,0234 / 0,0312 / 0,0312; CONFIRM 0,0469 /
  0,0234 / 0,0391: no hay mejora con la experiencia.
- **Olvido**: FINAL2 − LEARN2 en nuevas +0,0020 [0,0000, +0,0059] / −0,0078 [−0,0156, 0,0000].
- **Texto multilingüe** (CONFIRM, nuevas, LEARN2 vs OBS2 = 0 en todos): es 0,0398, en 0,0341, fr 0,0341; formato con palabras 0,0227 [0,0089,
  0,0405] vs resto 0,0454 [0,0298, 0,0607]. 0 parse_fail en todo el flujo (p̂ viene del parser numnorm también para las familias nuevas).
- LEARN2 también mejora las v4 no iterables: +0,0208 [+0,0143, +0,0280] / +0,0241 [+0,0156, +0,0339].

## Exploratorio (post hoc, NO prerregistrado; `artifacts/orbita/aprendizaje_online/explo_horizonte_*.md`)
1. **¿Qué episodios nuevos se resuelven?** En las 16 seeds, los aciertos acc16 de LEARN2 en nuevas vienen de **un solo ejemplo por conjunto y
   familia**, casi lineal (p₂ pequeño): DEV idx 129 (fam 8, p = (−0,902, 0,100)), 15/16 seeds; CONFIRM idx 141 (fam 8, p = (−0,033, −0,072)),
   6/16, e idx 155 (fam 9, p = (−0,138, −0,016)), 13/16. ORACLE_EP2 resuelve exactamente esos mismos (16/16). Ningún método resuelve a h16
   un caso genuinamente no lineal.
2. **Horizonte de acierto** (pasos consecutivos correctos tras y_k, binario con columnas añadidas; columnas comunes idénticas a la ejecución
   principal: verificado): nuevas, k = 2: OBS2 0,371 / 0,643 → LEARN2 1,447 [1,385, 1,516] / 2,104 [2,041, 2,160] (LEARN_EP2 1,453 / 2,250;
   ORACLE_EP2 1,406 / 2,250; ORACLE_LS2 1,750 / 1,531). El módulo aprende la dinámica **local** (≈ +1 a +1,5 pasos), pero el error se acumula
   y no extrapola a 14 pasos. En conocidas 1–2 el horizonte es 12,2 / 12,0 (LEARN2) frente a 14,0 del ajuste directo.

## Lectura
- **El bucle funciona como mecanismo**: la sorpresa detecta casi perfectamente lo desconocido (AUROC ≈ 0,97–0,99), se crea/reutiliza un
  módulo sin tocar el núcleo (sha256 idéntico), y **no hay degradación ni olvido**: en conocidas LEARN y FINAL incluso mejoran sobre OBS
  (el adaptador residual corrige parte del error iterado de ÓRBITA en los episodios conocidos que disparan: 1–2 0,05 / 0,12, v4 no iterables 0,20 / 0,23).
- **Pero no aprende dinámicas nuevas no afines de forma útil**: HL1 es estadísticamente positiva y replica, pero el nivel absoluto es
  0,03–0,04 de acc16 (frente a 0,57–0,61 en conocidas) y se explica por 1–2 ejemplos casi lineales. Con k ≤ 2 pasos (6–12 pares de puntos)
  ningún aprendiz local —ni el residual sobre ÓRBITA, ni los oráculos directos con los mismos datos— extrapola 14 pasos de una dinámica no
  lineal; el cuello de botella son los datos observados y el horizonte, no la detección.
- **La biblioteca no ayuda (HL3 rechazada)**: reutilizar módulos entre episodios iguala (DEV) o empeora (CONFIRM) al módulo por episodio.
  Causa probable (consistente con ORACLE_FAM ≪ ORACLE_EP incluso en familias afines conocidas: 0,20–0,22 vs 1,0): la parametrización
  Φ ⊗ [1, p̂₁, p̂₂] no representa la dependencia no lineal en p (p. ej. ángulos 0,6·p), así que agrupar episodios con distintos p sesga el
  ajuste; además la asignación por sorpresa fragmenta (≈ 20 módulos/seed, pureza ≈ 0,57–0,61).
- **Frente al oráculo justo**: LEARN2 es no inferior (y superior) a ORACLE_FAM2, pero no supera a ORACLE_EP2 (−0,025 en CONFIRM). En familias
  afines, una regresión directa con 2 pasos (ORACLE_EP2/LS2 = 1,0) supera claramente a ÓRBITA + re-identificación + adaptador (≈ 0,57–0,61):
  el núcleo congelado actúa más como sesgo que como prior útil cuando hay observaciones.

## Limitaciones
- Conjuntos fijos por fase (160 ejemplos; el bootstrap por seed captura variación de modelos, orden e idioma, no de datos); familias nuevas
  sólo 2 (8, 9) con 16 ejemplos cada una.
- λ elegido en el borde de la malla; espacio de módulos restringido a polinomios de grado ≤ 3 ⊗ lineal en p̂; no se probaron adaptadores en
  el espacio latente de Koopman ni más pasos observados (k > 2), que serían el siguiente experimento natural (prerregistrado aparte).
- El exploratorio es post hoc y no confirma nada; sólo describe.
