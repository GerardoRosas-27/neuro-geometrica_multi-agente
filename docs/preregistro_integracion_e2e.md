# Prerregistro — integración end-to-end D3-num (texto → Gemma/D2 + parser → ÓRBITA-v1 → CDT-RQM-epr) y robustez numérica

Rama `exp/orbita-llm`. Base: ganador A (D3-num) de `docs/resultados_d3_vs_robusta.md` (951fbf7). Este documento se
comprometió y publicó **antes** de extraer embeddings o evaluar cualquier texto de robustez. δ = 0,05. IC95 bootstrap
pareado sobre las 16 seeds (10 000 remuestreos, semilla 12345), como en las fases anteriores. DEV (0xA400–0xA40F) es
exploratoria; las hipótesis se deciden en CONFIRM (0xB400–0xB40F). Nada cambia entre DEV y CONFIRM.

## (a) Pipeline reproducible (módulo/CLI en `orbita/`, sin tocar v4)
- `orbita/llm/numnorm.py`: normalización numérica multilingüe es/en/fr: dígitos con `.`/`,`, palabras (`cero coma treinta
  y uno`, `zero point three one`, `zéro virgule trente et un`, incluidos `quatre-vingt-dix-sept`, `soixante et onze`),
  fracciones (`un cuarto`, `three tenths`, `trois dixièmes`, `quatre-vingt-quinze centièmes`), signos (`menos/minus/moins`)
  y unidades: radianes por defecto, grados ×π/180, vueltas ×2π (`un cuarto de vuelta` = π/2). Convención fijada: **p se
  interpreta en radianes**. Enteros en palabras sueltos (≠ cero) sin unidad se ignoran (evita artículos `un/una/one`).
  Resultado recortado a [−1, 1] como en D3.
- `orbita/llm/e2e.py run --phase P --seed S --idx N --text "…"`: texto → `orbita gemma-embed --per-token --exit-after 12`
  → encoder D2 de esa seed (sólo argmax de familia) + `numnorm.parse2` → ĉ = one-hot ++ p → `orbita llm-eval`
  (ÓRBITA-v1 congelada, rollout h=1…32; CDT-RQM-epr consolidado en modo Full sobre TRAIN: familia inferida por el teacher
  `cdt_fam_hat` y residuo `cdt_residual16` = rel_err(ÓRBITA h16, teacher h16)). **Fallback**: si el parser no encuentra
  dos cantidades, p = p̂ de D2.
- Test de humo `orbita/llm/test_e2e.sh`: tests unitarios del parser (ida y vuelta de −1,00…1,00 en 5 formatos × 3 idiomas,
  frases con dos valores, compatibilidad exacta con el parser de dígitos de D3 en plantillas 0–5) + un texto por la cadena
  completa con 0 queries. Debe pasar antes de DEV.

## (b) Prueba de robustez
- Mismos 16 ejemplos TEST iterables por seed que D2/D3 (`llm_subset`), 6 textos por ejemplo: 3 idiomas × **plantillas
  nuevas 6–7** (redacción nunca vista por D2, entrenado en 0,1,4,5; D3NUM_DIG usa 2–3). Split por plantilla.
- Formato de cada valor determinista: f₁ = (idx + plantilla + idioma) mod 5, f₂ = (f₁ + 2) mod 5, con formatos
  {palabras, fracción, grados (2 decimales), vueltas (4 decimales), radianes con unidad}. Los formatos sólo representan p
  redondeado a 2 decimales (error de representación ≤ 3,1·10⁻⁴ en vueltas, ≤ 8,7·10⁻⁵ en grados).
- Condiciones (todas con ÓRBITA-v1 congelada): TRUE (c real), D2_DIG y D3NUM_DIG (idénticos a la fase D3, textos con
  dígitos), D2_R (ĉ de D2 sobre textos nuevos), **E2E_R** (pipeline: familia D2 + numnorm, fallback D2).
- 0 queries a TEST auditado (contador de `llm-eval`); se reporta `parse_fail` total, por seed y por formato.

## Hipótesis (CONFIRM decide)
| id | contraste | aceptación | refutación |
|---|---|---|---|
| H-R0 | tasa parse_fail de E2E_R | IC95 superior < 0,05 | IC95 inferior > 0,05 |
| H-R1 | acc16 E2E_R − D3NUM_DIG | IC95 inferior > −δ | IC95 superior < −δ |
| H-R2 | acc16 E2E_R − D2_R | IC95 > 0 | IC95 superior < 0 |
| H-R3 | acc16 E2E_R − TRUE | IC95 inferior > −δ | IC95 superior < −δ |
| H-C2a | acierto de familia del teacher CDT sobre la predicción de ÓRBITA (E2E_R) − 1/8 | IC95 > 0 | IC95 superior < 0 |
| H-C2b | AUROC (residuo16 → fallo de ÓRBITA a h16) − 0,5 | IC95 > 0 | IC95 superior < 0 |
| H-P1 | test de humo (`test_e2e.sh`) | pasa | falla |
En otro caso: inconclusa. Descriptivo (sin hipótesis): acc por idioma y por formato, familia correcta de D2 en textos
nuevos vs dígitos, max|p̂−p|, residuo mediano en aciertos/fallos.

## Ejecución
`/workspace/orbita-data/llm_e2e/chain.sh` (`setsid nohup nice -n 0`), reanudable; `.done`: `smoke.done`,
`<fase>/robust.tok.done`, `dev.done`, `confirm.done`; log `chain.log`. Resultados en `docs/resultados_integracion_e2e.md`.
