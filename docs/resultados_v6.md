# Resultados v6 — hipótesis H1–H7 sobre v4 (§16)

**Rama:** `exp/v6-hipotesis` · **Fecha:** 2026-10-16  
**Código:** `b1c7628` (mismo SHA en DEV y CONFIRM; `dirty=false`).  
**Pre-registro:** `docs/preregistro_v6.md` (commiteado antes de ejecutar).  
**Seeds:** DEV `0xC600–0xC60F`, CONFIRM `0xD600–0xD60F` (16+16). Presupuesto 256 000 pasos Adam × batch 32.  
**Artefactos:** `artifacts/v6/{dev,confirm}/` → `config.json`, `dataset_manifest.jsonl`, `metrics.csv`, `summary.md`, `seeds/`, `paired_stats.txt` (salida de `scripts/v6_analyze.py`).  
**Nota:** se lanzó con `V6_IGNORE_V5=1` porque el usuario descartó v5; H2 queda **pendiente** del veredicto v4 vs v5 y no entra en el veredicto de v6.

Reproducir: `python3 scripts/v6_analyze.py artifacts/v6/dev` · `python3 scripts/v6_analyze.py artifacts/v6/confirm`.

## 0. Veredicto global

**V6_all no le gana a E3_v4 ni a los controles.** En ambas fases `V6_all − E3_v4` en acc h16 es fuertemente negativo (IC95 enteramente < 0), `acc1` empeora ≈0.20–0.22, y `V6_all − V6_all_mlp` también es negativo (el MLP del combo rinde mejor que el campo combinado). El control barajado `V6_shuffled` falla el gate E33 en **16/16** semillas (cumple ese requisito de sanidad). Ninguna de H1, H3, H4, H5a, H6, H7 se acepta bajo el criterio pre-registrado (pasar en DEV **y** CONFIRM). H2 (PRM) mejora vs su referencia multi-horizonte, pero está fuera del veredicto v6.

| Hipótesis | DEV | CONFIRM | Veredicto |
|---|---|---|---|
| H1 sesgo | FALLA | FALLA | **rechazada** |
| H2 PRM* | IC>0 vs mh | IC>0 vs mh | **pendiente** (no entra en v6) |
| H3 escala replay | FALLA | FALLA | **rechazada** |
| H4 Sobolev | FALLA | FALLA | **rechazada** |
| H5a soft replay | FALLA | FALLA | **rechazada** |
| H5b ruido η | diagnóstico | diagnóstico | ver §H5b |
| H6 semigrupo | FALLA | FALLA | **rechazada** |
| H7 normal | FALLA | FALLA | **rechazada** |
| **V6_all** | FALLA | FALLA | **rechazada** |

\*H2 no forma parte del veredicto hasta `docs/comparacion_v4_v5.md`.

## 1. Gates (PASS / 16; clean+fresh = true en todos los brazos)

| Brazo | DEV PASS | CONFIRM PASS | acc_h16 DEV | acc_h16 CONFIRM |
|---|---|---|---|---|
| E3_v4 | 11 | 14 | 0.2197 | 0.2559 |
| E3_v4_V1reproj | 13 | 14 | 0.2696 | 0.3066 |
| E3_mlp | 12 | 12 | 0.2275 | 0.1875 |
| E3_shuffled | 0 | 0 | 0.0000 | 0.0000 |
| H1_bias | 8 | 11 | 0.1885 | 0.1748 |
| H2_mh_eqc* | 0 | 0 | 0.0088 | 0.0176 |
| H2_prm_eqc* | 3 | 2 | 0.0820 | 0.0928 |
| H3_x0.5 | 11 | 12 | 0.1963 | 0.1953 |
| H3_x1 | 10 | 12 | 0.2402 | 0.1953 |
| H3_x2 | 14 | 12 | 0.2178 | 0.2256 |
| H3_x4 | 14 | 16 | 0.2090 | 0.2578 |
| H4_sobolev | 12 | 14 | 0.2207 | 0.2314 |
| H4_sobolev_mlp | 12 | 13 | 0.2100 | 0.2061 |
| H5_soft | 9 | 12 | 0.2168 | 0.2402 |
| H5_eta1 | 9 | 14 | 0.1914 | 0.2344 |
| H5_eta2 | 9 | 14 | 0.1914 | 0.2178 |
| H5_eta4 | 9 | 13 | 0.1709 | 0.1943 |
| H5_eta8 | 6 | 11 | 0.1631 | 0.1758 |
| H6_semigroup | 0 | 0 | 0.0127 | 0.0185 |
| H7_normal | 10 | 14 | 0.2119 | 0.2471 |
| **V6_all** | **0** | **0** | 0.0058 | 0.0039 |
| V6_all_mlp | 2 | 4 | 0.1133 | 0.1182 |
| V6_all_H2* | 1 | 0 | 0.1025 | 0.0801 |
| V6_shuffled | 0 | 0 | 0.0000 | 0.0000 |

Auditoría: `clean=true`, `leak=0`, `q_test=0`, `fresh_ok=true` en todas las semillas de ambos fases (salida de `v6_analyze.py`).

## 2. Contrastes pre-registrados (acc_h16 salvo nota; media [IC95] bootstrap 10 000, semilla 0x6B00)

### H1 — sesgo (éxito: IC95 > 0 en acc h16 en ambas fases; acc1 no peor > 0.02)

| Contraste | DEV | CONFIRM |
|---|---|---|
| H1_bias − E3_v4 (acc_h16) | −0.0312 [−0.0664, +0.0137] | −0.0810 [−0.1338, −0.0225] |
| H1_bias − E3_v4 (acc1) | −0.0117 [−0.0186, −0.0056] | −0.0095 [−0.0152, −0.0027] |

**Rechazada.** En CONFIRM el IC de acc h16 es enteramente negativo; en DEV incluye 0. Además acc1 empeora (IC < 0) aunque dentro del margen 0.02.

### H2 — PRM vs multi-horizonte (pendiente; no entra en veredicto v6)

| Contraste | DEV | CONFIRM |
|---|---|---|
| H2_prm_eqc − H2_mh_eqc (acc_h16) | +0.0733 [+0.0537, +0.0928] | +0.0752 [+0.0508, +0.1006] |

La máscara PRM gana a la pérdida multi-horizonte sin máscara en ambas fases (IC > 0), pero ambos brazos fallan el gate E33 casi siempre (0–3/16). Lectura diagnóstica pendiente de `comparacion_v4_v5.md`.

### H3 — escala del buffer de replay (éxito: pendiente rel_err₁ vs log N < 0 con IC que excluye 0)

Diferencias pareadas de `rel_err1` vs E3_v4 (flujo infinito):

| Brazo | DEV | CONFIRM |
|---|---|---|
| H3_x0.5 − E3_v4 | +0.0004 [−0.0003, +0.0010] | +0.0001 [−0.0005, +0.0006] |
| H3_x1 − E3_v4 | +0.0001 [−0.0003, +0.0005] | +0.0003 [−0.0002, +0.0007] |
| H3_x2 − E3_v4 | +0.0000 [−0.0005, +0.0006] | −0.0001 [−0.0008, +0.0005] |
| H3_x4 − E3_v4 | +0.0005 [+0.0002, +0.0009] | −0.0001 [−0.0008, +0.0005] |

**Rechazada.** No hay caída monótona de rel_err₁ con N; las diferencias están en ~10⁻⁴ y casi todos los IC incluyen 0. El cuello no parece ser el tamaño del buffer de replay (saturación / irrelevancia de N frente al flujo infinito de E3).

### H4 — destilación Sobolev (éxito: interacción campo−MLP en acc h16, IC95 > 0)

| Contraste | DEV | CONFIRM |
|---|---|---|
| (H4_field − E3_v4) − (H4_mlp − E3_mlp) | +0.0186 [−0.0449, +0.0830] | −0.0430 [−0.0957, +0.0117] |

**Rechazada.** Ambos IC incluyen 0; en CONFIRM la media es negativa.

### H5a — replay suave (éxito: IC lim.inf. > −0.02; acc1 no peor > 0.02)

| Contraste | DEV | CONFIRM |
|---|---|---|
| H5_soft − E3_v4 (acc_h16) | −0.0029 [−0.0391, +0.0342] | −0.0156 [−0.0596, +0.0283] |

**Rechazada.** El límite inferior del IC queda por debajo de −0.02 en ambas fases.

### H5b — sensibilidad a ruido η (diagnóstico; éxito: pendiente log-log dstep vs η con IC < 1)

Media de `dstep_vs_E3` por η:

| η | DEV | CONFIRM |
|---|---|---|
| 1 % | 0.00365 | 0.00253 |
| 2 % | 0.00519 | 0.00393 |
| 4 % | 0.00666 | 0.00498 |
| 8 % | 0.00842 | 0.00777 |

`dstep` crece con η de forma sub-lineal a ojo (factor ~2.3× al multiplicar η ×8 en DEV), compatible con acoplamiento suave; no se reporta IC de pendiente en `v6_analyze.py`. Acc h16 y PASS caen al subir η (peor con más contaminación).

### H6 — semigrupo (éxito: H6 − E3_mlp IC > 0 y H6 − E3_v4 ≥ 0)

| Contraste | DEV | CONFIRM |
|---|---|---|
| H6_semigroup − E3_mlp | −0.2149 [−0.2549, −0.1748] | −0.1689 [−0.2041, −0.1358] |
| H6_semigroup − E3_v4 | −0.2071 [−0.2539, −0.1651] | −0.2373 [−0.2871, −0.1885] |

**Rechazada.** Gate 0/16 en ambas fases; mucho peor que MLP y que v4.

### H7 — contracción normal (éxito: H7(V0) − E3_v4(V1reproj) lim.inf. > −0.02 y H7 − E3_v4(V0) > 0)

| Contraste | DEV | CONFIRM |
|---|---|---|
| H7_normal(V0) − E3_v4(V1reproj) | −0.0576 [−0.0703, −0.0430] | −0.0596 [−0.0840, −0.0352] |
| H7_normal − E3_v4 | −0.0078 [−0.0186, +0.0039] | −0.0088 [−0.0303, +0.0127] |

**Rechazada.** Pierde claramente frente a E3 con V1reproj (IC enteramente < −0.02); vs E3_v4 sin exigir ganancia (IC incluye 0).

### V6_all — combinación H1+H4+H5a+H6+H7 (éxito: V6_all − E3_v4 IC > 0, acc1 no peor > 0.02, V6_all − V6_all_mlp IC > 0; V6_shuffled falla E33 en ≥15/16)

| Contraste | DEV | CONFIRM |
|---|---|---|
| V6_all − E3_v4 (acc_h16) | −0.2139 [−0.2617, −0.1738] | −0.2520 [−0.2998, −0.2041] |
| V6_all − E3_v4 (acc1) | −0.2022 [−0.2136, −0.1907] | −0.2178 [−0.2300, −0.2063] |
| V6_all − V6_all_mlp | −0.1074 [−0.1445, −0.0772] | −0.1143 [−0.1367, −0.0928] |
| V6_all_H2 − V6_all* | +0.0967 [+0.0655, +0.1299] | +0.0762 [+0.0518, +0.1064] |

`V6_shuffled`: FAIL gate **16/16** en DEV y CONFIRM (cumple el control de sanidad).

**V6_all rechazada.** El combo destroza el baseline: 0/16 PASS, acc h16 ≈ 0.006/0.004, peor que su propio control MLP. Apilar hipótesis que individualmente no ayudan (y H6 que sí daña) produce un modelo inutilizable bajo el protocolo E33.

## 3. Lectura

1. **Baseline E3_v4 sigue siendo el mejor brazo de campo** (11–14/16 PASS; acc_h16 0.22–0.26). El MLP gated empató o superó al campo en media en DEV y quedó por debajo en CONFIRM; el shuffled colapsa (0/16), como se esperaba.
2. **Ninguna hipótesis H1/H3–H7 mejora acc h16 de forma pre-registrada.** H6 es activamente dañina (0/16). H1 y H5a son neutras o levemente peores. H3 satura. H4 no muestra interacción campo>MLP. H7 no sustituye a V1reproj.
3. **H2 PRM** ayuda frente a multi-horizonte sin máscara, pero ambos fallan E33; fuera del veredicto hasta cerrar v4 vs v5.
4. **V6_all falla el criterio de éxito de v6.** No se recomienda mergear esta combinación a `main`. Las λ/ρ fijadas a priori (§3 del preregistro) quedan como limitación declarada: no se re-tunearon tras DEV.

## 4. Archivos de análisis

Salida cruda de `scripts/v6_analyze.py` guardada en:

- `artifacts/v6/dev/paired_stats.txt`
- `artifacts/v6/confirm/paired_stats.txt`
