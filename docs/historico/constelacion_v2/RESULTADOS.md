# CONSTELACIÓN v2 — Resultados F2 (DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F)

Preregistro: `PREREG.md`. Análisis: `python3 scripts/analyze_constelacion_v2.py dev confirm` → `artifacts/constelacion_v2/analysis.json`. Bootstrap pareado por semilla (10 000 réplicas, semilla 12345, IC95 percentil), δ=0,05. n=16 semillas por fase, 0 NaN en todos los brazos, 0 consultas TEST, ÓRBITA-v1 = ESTELA k4 en 16/16 semillas por fase. Sin reajuste entre DEV y CONFIRM. CONFIRM: las 16 semillas completaron en la ejecución original (`.done` + `metrics_*.json` válidos + `ok` en log); no hubo que relanzar ninguna.

**Contexto.** F0 y F1 ya habían FALLADO (ver `F0.md`, `F1.md`): F0 — la neurona única no supera al trivial ni en h1 ni en h16 (4/4); F1 — brecha +2,15 a +2,69 log frente a ÓRBITA-v1 en conocidas (4/4), red ≈ predicción nula. F2 se ejecutó por orden expresa del usuario con expectativa declarada de primaria no aceptada.

## Tabla (media [IC95])

| Criterio | DEV | CONFIRM |
|---|---|---|
| Primaria: med log rel_err h16 nuevas, CST − ÓRBITA-v1 | +0,807 [+0,702, +0,908] | +0,798 [+0,688, +0,916] |
| No-daño: acc16 conocidas, final − pre (δ=0,05) | 0,000 [0,000, 0,000] | 0,000 [0,000, 0,000] |
| E33 pase (CST / ÓRBITA) | 0/16 / 16/16 | 0/16 / 16/16 |
| E33 acc h16, CST − ÓRBITA | −0,313 [−0,369, −0,259] | −0,323 [−0,376, −0,268] |
| Atribución: CST − fixed_w | +0,0080 [+0,0023, +0,0155] | +0,0047 [+0,0010, +0,0106] |
| Atribución: CST − no_err | +0,0000 [−0,0048, +0,0049] | +0,0012 [−0,0045, +0,0096] |
| Atribución: CST − t1 | +0,0033 [−0,0014, +0,0096] | +0,0034 [−0,0004, +0,0088] |
| (informativo) CST − no_prec | −0,039 [−0,071, −0,012] | −0,108 [−0,169, −0,052] |
| (informativo) CST − no_prune | −0,034 [−0,071, −0,007] | −0,013 [−0,023, −0,005] |
| (informativo) CST − no_sleep | −0,031 [−0,056, −0,011] | −0,094 [−0,165, −0,036] |
| (informativo) CST − one_layer | −0,062 [−0,116, −0,022] | −0,021 [−0,040, −0,005] |
| (sólo reporte) CST − backprop | −0,276 [−0,602, −0,055] | −0,153 [−0,341, −0,035] |
| Colapso de capa CST (semillas) | 4/16 | 5/16 |

Medias brutas nuevas med log: CST +0,008 (DEV) / +0,005 (CONFIRM) ≈ 0 = predicción nula; ÓRBITA −0,799 / −0,793. acc16 de CST = 0 en todo.

## Veredicto por criterio

- **Primaria: REFUTADA.** IC95 inferior > 0 en CONFIRM (+0,688): CONSTELACIÓN es peor que ÓRBITA-v1 sola en nuevas (≈ +0,8 log). También en DEV.
- **No-daño: OK formal, pero vacío** — acc16 de CST es 0 antes y después; no hay nada que dañar.
- **E33: FALLA** — 0/16 en ambas fases (ÓRBITA 16/16).
- **Atribución a PC: NO.** Ningún control requerido tiene IC sup < 0; frente a fixed_w CST es incluso significativamente peor (IC > 0 en ambas fases). Las diferencias son ~1e-3 log, todas en el régimen de predicción nula.
- Los controles no_prec/no_prune/no_sleep/one_layer/backprop son peores que CST, pero sólo porque se alejan más de 0 (predicción nula); no indican aprendizaje útil.

**Conclusión:** negativo consistente con F0/F1. CONSTELACIÓN v2 (configuración congelada) no aprende más allá de la predicción nula y queda claramente por debajo de ÓRBITA-v1. Resultado preregistrado: primaria refutada en CONFIRM.
