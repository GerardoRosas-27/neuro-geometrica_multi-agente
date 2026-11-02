# Resultados: latente de Koopman con k > 2 (Exp. 1) y flujo largo (Exp. 2)

Prerregistro `docs/prereg_orbita_latente_largo_plazo.md` (20edf0c) y driver (2111a04), ambos empujados antes de ejecutar. Código `orbita/src/latent.rs`,
`orbita/llm/latlp.py`, `latlp_chain.sh`. Ejecución 2026-10-19 (hora CST, UTC−6), una sola cadena `setsid nohup` (sobrevivió a las interrupciones
de la sesión, sin relanzar nada): Gemma DEV H 04:09:10–04:13:23, DEV S desde 04:13:23, CONFIRM H 04:31:34–04:36:18, CONFIRM S desde 04:36:18
(un solo proceso Gemma); τ 05:21:29; malla DEV 06:03:13; Exp. 1 CONFIRM 06:04:39; Exp. 2 DEV 06:14:24, CONFIRM 06:30:57. 0 queries v4 a TEST,
0 parse_fail (H y S, DEV), sha256 de los pesos de ÓRBITA idéntico antes/después en todas las ejecuciones. Tablas completas en
`artifacts/orbita/latente_largo_plazo/summary_exp{1,2}_{dev,confirm}.md`; por episodio en `per_episodio.tar.gz`.

## Calibración y selección (sólo DEV)
τ₂ = 0,014656, τ₄ = 0,022020, τ₈ = 0,037142 (P95 de S_known, familias 1–2, H DEV, 512 episodios por k).
Selección (`grid.log`): LAT λ₀ = 1e-4, λ_m = 1e-3 (puntuación 0,4478); LAT_EP λ₀ = 1e-4 (0,4367); ORACLE_EP D = 2, λ = 1e-4 (0,7063);
ORACLE_MS D = 2, λ = 1e-4 (0,6562). LAT y los dos oráculos quedaron en el borde inferior de sus mallas de λ.

## Exp. 1: acc16 en familias nuevas 8–9 (media de 16 seeds [IC95]; DEV / CONFIRM)
| k | OBS | RES | RES_EP | LAT_EP | LAT | FINAL | ORACLE_EP | ORACLE_MS | ORACLE_LS |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0 / 0,0234 | 0 / 0,0234 | 0 / 0,0234 | 0 / 0,0234 | 0 / 0,0293 | 0 / 0,0312 | 0 / 0,0312 | 0 / 0,0312 | 0 / 0,0312 |
| 4 | 0 / 0,0254 | 0 / 0,0254 | 0 / 0,0254 | 0 / 0,0469 | 0 / 0,0430 | 0 / 0,0469 | 0 / 0,0312 | 0 / 0,0312 | 0 / 0,0938 |
| 8 | 0 / 0,0293 | 0 / 0,0664 | 0 / 0,0664 | 0,0020 / 0,1348 | 0,0020 / 0,1250 [0,1035, 0,1465] | 0,0020 / 0,1250 | 0 / 0,0625 | 0 / 0,0625 | 0 / 0,0938 |

En el conjunto H de DEV **ningún** método, incluidos los oráculos, acierta a h16 en las familias nuevas (todo 0 salvo LAT/LAT_EP k = 8: 0,0020).
En CONFIRM, con k = 8, LAT llega a 0,1250: familia 8 0,2461 [0,2031, 0,2891] (ORACLE_EP 0,1250), familia 9 0,0039.

### Hipótesis Exp. 1 (nuevas 8–9; DEV / CONFIRM)
| | k = 2 | k = 4 | k = 8 | veredicto |
|---|---|---|---|---|
| **H1a** LAT_EP − RES_EP > 0 | 0 [0, 0] / 0 [0, 0] | 0 [0, 0] / +0,0215 [+0,0117, +0,0332] | +0,0020 [0, +0,0059] / +0,0684 [+0,0449, +0,0918] | **rechazada** en los 3 k (falla en DEV) |
| **H1b** LAT − RES > 0 | 0 / +0,0059 [0, +0,0117] | 0 / +0,0176 [+0,0078, +0,0312] | +0,0020 [0, +0,0059] / +0,0586 [+0,0391, +0,0781] | **rechazada** (DEV) |
| **H2** LAT − LAT_EP > 0 | 0 / +0,0059 [0, +0,0117] | 0 / −0,0039 [−0,0117, +0,0039] | 0 [0, 0] / −0,0098 [−0,0195, −0,0020] | **rechazada** (en k = 8 CONFIRM la memoria empeora) |
| **H3** LAT − OBS > 0 | 0 / +0,0059 [0, +0,0117] | 0 / +0,0176 [+0,0078, +0,0312] | +0,0020 [0, +0,0059] / +0,0957 [+0,0742, +0,1172] | **rechazada** (DEV) |
| **H4** LAT − ORACLE_EP > −δ | 0 [0, 0] / −0,0020 [−0,0078, +0,0039] | 0 [0, 0] / +0,0117 [0, +0,0254] | +0,0020 [0, +0,0059] / +0,0625 [+0,0410, +0,0840] | **aceptada** (no inferioridad; en DEV trivial, ambos ≈ 0) |
| **H4** LAT − ORACLE_MS > −δ | igual que ORACLE_EP en los 3 k | | | **aceptada** (ídem) |

**H5** (sin olvido; LAT − OBS y FINAL − OBS > −δ en conocidas 0–7 y 1–2): **aceptada** en los 3 k y ambas fases, con mejora. LAT − OBS en 0–7: k = 2
+0,0483 [+0,0356, +0,0610] / +0,0703 [+0,0557, +0,0864]; k = 4 +0,0601 / +0,0850; k = 8 +0,1006 [+0,0850, +0,1187] / +0,1279 [+0,1133, +0,1436].
En 1–2: k = 2 +0,0742 / +0,1074; k = 4 +0,0801 / +0,1172; k = 8 +0,0586 / +0,0859. FINAL − OBS en 0–7: +0,0562 / +0,0840 (k = 2) … +0,1172 /
+0,1460 (k = 8); en 1–2 entre +0,0605 y +0,1250. Aun así, en las conocidas el ajuste directo por episodio sigue arriba (ORACLE_EP 0,875–0,891 en 0–7
frente a LAT 0,45–0,67).
Reportado: AUROC de S_known nuevas vs conocidas 0,989–0,990 (DEV), 0,958–0,966 (CONFIRM); uso de la memoria en nuevas 0,12–0,29; pureza del prior
recuperado en nuevas 0,75–0,91.

## Exp. 2: flujo largo (2000 episodios por seed; held-out H con la memoria congelada)
acc16 en H, nuevas 8–9, por episodios vistos n (CONFIRM; en DEV todo es 0,0000 salvo k = 8: 0,0020 para n ≤ 500, 0,0039 en n = 1000, 0 en n = 2000):
| k_eval | n=0 | 125 | 250 | 500 | 1000 | 2000 | OBS | ORACLE_EP | ORACLE_MS |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0,0234 | 0,0293 | 0,0293 | 0,0312 | 0,0312 | 0,0352 [0,0312, 0,0410] | 0,0234 | 0,0312 | 0,0312 |
| 4 | 0,0469 | 0,0352 | 0,0391 | 0,0410 | 0,0410 | 0,0391 | 0,0254 | 0,0312 | 0,0312 |
| 8 | 0,1348 | 0,1230 | 0,1172 | 0,1094 | 0,1152 | 0,1055 [0,0820, 0,1309] | 0,0293 | 0,0625 | 0,0625 |

### Criterios «aprende a largo plazo» (k_eval = 2 primario; DEV / CONFIRM)
| | DEV | CONFIRM | |
|---|---|---|---|
| **L1** pendiente acc16 nuevas vs log₂ n > 0 | 0 [0, 0] | +0,0014 [−0,0000, +0,0029] | **no** |
| **L2** n = 2000 − n = 0 > 0 | 0 [0, 0] | +0,0117 [+0,0039, +0,0195] | **no** (falla en DEV) |
| **L3** n = 2000 − ORACLE_EP > 0 y − ORACLE_MS > 0 | 0 [0, 0] (ambos) | +0,0039 [0, +0,0098] (ambos) | **no** |
| **L4** conocidas 0–7: n = 2000 − OBS > −δ; n = 2000 − n = 125 > −δ | +0,1797 [+0,1582, +0,2012]; +0,1284 [+0,1064, +0,1494] | +0,1958 [+0,1758, +0,2163]; +0,1001 [+0,0815, +0,1211] | **sí** |

**Veredicto Exp. 2: NO aprende a largo plazo** según el criterio prerregistrado (L1–L3 fallan). Secundarios: k = 4 y k = 8 tampoco.
En CONFIRM, k = 8, el aprendizaje acumulado **empeora** lo nuevo: L2 −0,0293 [−0,0508, −0,0078]; L1 −0,0037 [−0,0082, +0,0006]. Sí supera a los
oráculos por episodio (L3 +0,0430 [+0,0195, +0,0684]), pero eso ya ocurre con n = 0 (es el módulo, no la memoria).

Reportado sin criterio:
- **En conocidas sí aprende a largo plazo**: acc16 0–7 en H (k = 2) 0,4624 → 0,6084 en DEV y 0,4395 → 0,5767 en CONFIRM, creciendo en cada
  punto de control. Conocidas 1–2 a n = 2000 − OBS: k = 2 +0,3516 / +0,4062; k = 4 +0,2539 / +0,3301; k = 8 +0,0820 / +0,1445, es decir más
  beneficio cuanto menos pasos se observan (transferencia a episodios cortos), pero sólo en dinámicas que ÓRBITA ya conoce. En el flujo,
  LAT − OBS en conocidas sube de +0,0794 / +0,0853 (episodios 0–249) a +0,1763 / +0,2025 (1750–1999).
- Familias nuevas en el flujo (precuencial, k mixto): LAT 0,04–0,08 por bloque sin tendencia, a la par de LAT_EP (p. ej. CONFIRM 1750–1999: LAT
  0,0646, LAT_EP 0,0586, OBS 0,0123).
- Memoria: 43,5 / 46,9 entradas a n = 125 y 1103,6 / 1152,3 a n = 2000 (crece sin compactar). Pureza del prior en nuevas a n = 2000 (k = 2):
  0,9287 / 0,8841: la recuperación por sorpresa sí agrupa bien (arregla la pureza de e2ac4ab, 0,57–0,61), pero el prior agrupado no mejora el ajuste.

## Lectura
1. **El módulo latente multipaso es mejor que el residual cuando hay más pasos**: con k = 8 en CONFIRM, LAT_EP 0,1348 vs RES_EP 0,0664 y vs los
   oráculos por episodio 0,0625, y es el único método que extrapola algo de la familia 8 (0,25). Pero no replica en DEV, donde el conjunto H no
   tiene ningún caso nuevo resoluble por ningún método, así que H1/H3 se rechazan según la regla.
2. **La memoria nueva agrupa bien pero no transfiere a lo nuevo**: pureza ≈ 0,88–0,93, pero LAT ≈ LAT_EP (H2 rechazada) y, a largo plazo, el
   prior acumulado no ayuda (k = 2) o perjudica (k = 8) en familias nuevas. Hipótesis (no probada): para dinámicas no lineales que dependen de p,
   promediar adaptadores de episodios distintos da un prior sesgado; con k = 8 el prior aleja el ajuste de lo que se lograría sin él.
3. **Sin olvido, y con aprendizaje acumulativo real en lo conocido**: en todas las comparaciones de conocidas, LAT/FINAL/memoria larga mejoran a
   OBS y la mejora crece con la experiencia (hasta +0,41 en 1–2 con k = 2). Núcleo intacto (sha256).
4. Cuello de botella: extrapolar 8–14 pasos de una dinámica no afín nueva desde ≤ 8 pasos observados con rel_err < 0,05; ni ÓRBITA + adaptador
   ni los oráculos lo logran salvo en la familia 8 con k = 8.

## Limitaciones
- Sólo 2 familias nuevas; H con 16 ejemplos por familia. En DEV el techo medido es 0 para todos los métodos, lo que hace las hipótesis de
  superioridad imposibles de aceptar con la regla «DEV y CONFIRM»; el resultado positivo de CONFIRM (k = 8) es de una sola fase.
- λ seleccionados en el borde de la malla. Memoria sin compactación (≈ 1100 entradas). Mezcla del flujo fija (20 % nuevas) y k uniforme {2, 4, 8}.
- La métrica binaria acc16 (umbral 0,05) es muy estricta para dinámicas no afines; no se reportan métricas continuas porque no estaban
  prerregistradas.
