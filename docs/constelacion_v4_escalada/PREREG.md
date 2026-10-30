# CONSTELACIÓN v4-ESCALADA — Preregistro (DEV 0xA450–0xA45F, CONFIRM 0xB450–0xB45F)

Se fija y se sube con push **antes** de cualquier ejecución DEV.
- Código congelado en este commit: `orbita/src/constelacion4/escalada.rs`, comando `cst4esc-run`.
- `cst4esc-run` rechaza cualquier flag extra en dev/confirm y asserta los rangos de semilla.
- 0 consultas TEST (assert). NaN = fallo. Sin reajuste entre DEV y CONFIRM.

## Sistema
- Control: `Cfg::v4()` (PREREG de v4), re-ejecutado en las semillas nuevas como brazo `v4`.
- Variantes separables (tamaños en PLAN.md):
  - **P:** capas [16,8,8,4,4];
  - **A:** capas [32,16,8];
  - **T:** flujos ×4 (A = 12288; B = 768+768);
  - **PAT:** [32,16,16,8,8] con flujos ×4.
- Con P=A=T=off se obtiene v4 exacta. Verificado bit a bit (PLAN.md §Smoke).
- No se usan M1/M2/M3.

## Datos y semillas
- Mismos generadores que v4: held-out H (HPF=16 por familia, conocidas 0–7, nuevas {8,10,11}), flujo A, flujo B y TEST E33 sellado.
- Semillas **nuevas**: DEV 0xA450–0xA45F y CONFIRM 0xB450–0xB45F. No usadas antes en ningún worktree.
- **ÓRBITA-v1** congelada: la semilla nueva 0x{A,B}45i usa `artifacts/orbita/{dev,confirm}/weights_0x{A,B}40i.json`, los mismos pesos que el v4 original.
  - **Desviación declarada:** ÓRBITA-v1 no se re-entrena para las semillas nuevas.
  - Es un modelo fijo evaluado en el H y el E33 de la semilla nueva, que no vio en entrenamiento.

## Métricas (como v4, más una informativa)
- **Primaria:** por semilla, mediana de log rel_err h16 en familias nuevas, k=4.
- **h1:** E33 acc en h1.
- **E33:** pass si la región es ≥ 8 con acc ≥ 0.5 desde h1. Objetivo ≥ 15/16.
- **Conocidas, no-daño:** acc16 known final − pre.
- **Sorpresa AUROC:** AUROC de h1_log por episodio en el flujo B, positivo = familia nueva.
  - Es una métrica nueva. v4 no la registraba; aquí se mide también en el brazo v4.

## Estadística (idéntica a v4)
- Bootstrap pareado por semilla, 10000 réplicas, semilla 12345, IC percentil, δ = 0.05.
- IC95, excepto en "V − v4" sobre la primaria para P, A y T: ahí IC 98.75% (Bonferroni ×4). PAT es la hipótesis principal y usa IC95.

## Hipótesis (V ∈ {P, A, T, PAT})
- **H1, mejora vs v4:** IC sup de (V − v4) en new_med_log < 0 en DEV **y** CONFIRM (regla v4).
  - "Mejora material" si IC sup < −δ (equivale a que el IC inf de la mejora sea > δ). Se reporta, pero no es requisito.
  - REFUTADA si el IC inf en CONFIRM es > 0.
- **H2, no-daño en conocidas:** IC inf de (final − pre) en acc16 conocidas > −δ, **y** IC inf de (V − v4) en acc16 conocidas > −δ, en ambas fases.
- **H3, vs ÓRBITA-v1:** IC95 sup de (V − ÓRBITA) en new_med_log < 0 en ambas fases.
- **H4, E33:** ≥ 15/16 en ambas fases, y IC inf de (V − v4) en acc h1 > −δ.
- **Secundaria:** IC inf de (V − v4) en sorpresa AUROC > −δ.

## Gate DEV → CONFIRM
- v4 corre siempre en CONFIRM.
- Una variante V pasa a CONFIRM solo si en DEV cumple todo lo siguiente:
  - 16/16 semillas, 0 NaN, 0 colapso y 0 TEST;
  - media(V − v4) en new_med_log < 0;
  - H2 OK;
  - E33 ≥ 15/16.
- Si ninguna pasa: no hay CONFIRM y el veredicto es "escalar no mejora v4".

## Criterio "candidata a main"
V es candidata si cumple **todo** en DEV y CONFIRM:
- H1 (regla v4);
- H2;
- H3;
- H4 (E33 ≥ 15/16 y h1 no inferior);
- sorpresa no inferior;
- 0 NaN, 0 colapso, 0 TEST y 16/16.

Si hay varias candidatas, se propone la de menos parámetros (y luego menos tiempo) entre las que estén a ≤ δ de la mejor media CONFIRM (V − v4). La decisión de pasar a main es del usuario.

## Expectativa declarada (sin números inventados)
- El diagnóstico de v4 indicó que el entrenamiento lento ×4 no movió la primaria, así que **T** probablemente no mejora.
- P y A son inciertas.
- Es plausible un resultado nulo.
