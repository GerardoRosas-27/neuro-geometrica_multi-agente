# Prerregistro — ¿lo consolidado por CDT-RQM-epr aporta a la inferencia de ÓRBITA? DIRECTO vs SUEÑO (+poda)

Rama `exp/orbita-llm`, base e2e (925015f). Comprometido y publicado **antes** de calcular cualquier rasgo, predicción o
métrica de DEV/CONFIRM (la cadena sólo había empezado a extraer embeddings de Gemma del pool de entrenamiento). δ = 0,05;
IC95 bootstrap pareado sobre 16 seeds (10 000, semilla 12345). DEV 0xA400–0xA40F exploratoria, CONFIRM 0xB400–0xB40F
decide; nada cambia entre fases. 0 queries a TEST auditado (contadores de `llm-eval` y `cdt-feat`).

## Mecanismos existentes usados (no se inventa ninguno; v4 no se modifica)
Del crate v4 (`src/field_autonomy_v4.rs`, `src/v4_distill.rs`, `src/field_v4.rs`, mismos en `origin/main`):
- Vigilia/CDT: `collect_experience(train, ExpMode::Real, seed)` → `CdtStore` (experiencia episódica sobre TRAIN).
- Consolidación: `consolidate(&cdt, ConsolidateMode::Full)` → `LearningSignal` (una `FamilyRule` ridge por familia,
  48 números por regla, `n_numbers()`); control existente `ConsolidateMode::Corrupt` (E36-D, coeficientes permutados).
- Máscara de confianza: `family_fit_residual` + `TRUST_TAU = 0.01` → familias *trusted* (receta §16, `train_trusted_on`).
- Sueño: `sleep_consolidate` (field_v4) = consolidar toda la CDT → replay restringido a familias de confianza →
  **borrar la CDT** episódica; replay generativo `replay_sample` / `pseudo_experiences` (E45: pseudo-experiencias desde la
  señal cuando los episodios ya se borraron).
- Poda: en v4 la única poda de la memoria consolidada es la máscara *trusted* (reglas con residuo ≥ τ no se reproducen).
  Hay otras podas en el repo pero de otros sustratos, no aplicables a CDT/ÓRBITA: `prune_weak_routes` (rutas RQM fasoriales,
  `src/web/sleep_optimize.rs`, sólo `archive/pre-v4-main`) y las máscaras dispersas de Gemma en `adaptive_gemma2.rs`
  (`sleep_flush…`, ramas field-autonomy-v4 / live-train-llm-cdt / predictive-field-v5). No se usan.

## Señales (orbita `cdt-feat`, nuevo subcomando de ÓRBITA, sólo lee el crate)
- **full** (DIRECTO): señal consolidada de la CDT de vigilia, tal cual.
- **sleep** (SUEÑO): misma experiencia → `consolidate(Full)` → máscara trusted → **poda** de reglas no confiables →
  borrado de la CDT → `pseudo_experiences` (tantas como ejemplos TRAIN) desde la señal podada → `collect_experience` +
  `consolidate(Full)` offline sobre el replay. Tamaño tras poda = `n_numbers()` / reglas de la señal resultante.
- **corrupt**: `ConsolidateMode::Corrupt`.
- Rasgos por texto y familia candidata g, con p leído por `numnorm` (fallback D2) y el estado de entrada (TRAIN para el
  pool; entrada del ejemplo TEST iterable en evaluación, como `llm-eval`): r_g = rel_err(ÓRBITA h16 con c_g, teacher h16
  con g) (10 si la familia no está en la señal), sup_g = p en el rango de la regla, tr_g = familia de confianza.
  φ_g = [log(r_g + 10⁻³), sup_g, tr_g].

## Condiciones (ÓRBITA-v1 congelada; cuerpo de D2 congelado; p = numnorm como en e2e)
Entrenamiento = cabeza de familia de D2 (`f`, iniciada en D2) + 3 pesos de fusión w: logit_g = f(u)_g + w·φ_g, CE con
la familia verdadera, Adam lr 0,01, 600 pasos full-batch, sobre un **pool3 nuevo** (128 contextos, semilla 0xC0DF, sin el
hueco de TRAIN, plantillas de entrenamiento 0,1,4,5 → 1536 textos). Evaluación en plantillas nunca vistas (split por
plantilla): **R** = textos de robustez e2e (plantillas 6–7, primario) y DIG = dígitos (plantillas 2–3, secundario).
- **BASE** = E2E actual (familia argmax de D2 sin reentrenar). TRUE = c real.
- **NOCDT**: mismo fine-tune de `f`, φ = 0 (sin CDT; controla el efecto del propio entrenamiento).
- **DIRECTO**: φ de la señal full alimentada directo a la decisión de familia (contexto de ÓRBITA).
- **SUENO**: φ de la señal tras sueño + poda.
- **CORRUPT**: φ de la señal Corrupt (CDT aleatorizado entre familias).
- **SHUF**: φ de la señal full barajado entre textos (permutación fija por seed).

## Hipótesis (CONFIRM, conjunto R; A − B en acc16)
| id | contraste | aceptación | refutación |
|---|---|---|---|
| H-D1 / H-S1 | DIRECTO / SUENO − BASE | IC95 > 0 | IC95 sup < 0 |
| H-D2 / H-S2 | DIRECTO / SUENO − NOCDT (aporte del CDT más allá del fine-tune) | IC95 > 0 | IC95 sup < 0 |
| H-D3 / H-S3 | DIRECTO / SUENO − CORRUPT | IC95 > 0 | IC95 sup < 0 |
| H-D4 / H-S4 | DIRECTO / SUENO − SHUF | IC95 > 0 | IC95 sup < 0 |
| H-SD | SUENO − DIRECTO | IC95 > 0 (el sueño aporta) | IC95 sup < 0; si IC95 ⊂ (−δ, δ): «equivalentes» |
| H-F1 / H-F2 / H-SDf | error de familia DIRECTO / SUENO − BASE; SUENO − DIRECTO | IC95 < 0 | IC95 inf > 0 |
Otro caso: inconclusa. «Lo consolidado aporta» sólo si H-D2 (o H-S2) **y** H-D3/H-D4 (o H-S3/H-S4) se aceptan.
Descriptivo: acc1/acc8 de todos los contrastes, acc vs TRUE, AUROC de r(familia elegida) → fallo a h16 (con la señal de
cada condición; NOCDT/BASE con full), tamaño de la señal (números y reglas) full vs sleep, pesos w aprendidos, DIG.

## Ejecución
`/workspace/orbita-data/llm_cdt/chain.sh` (copia en `orbita/llm/cdt_chain.sh`), `setsid nohup nice -n 0`, reanudable;
`.done`: `pool3.tok.done`, `dev.done`, `confirm.done`; log `chain.log`. Resultados: `docs/resultados_cdt_sueno.md`.
