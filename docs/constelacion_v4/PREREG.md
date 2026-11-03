# CONSTELACIÓN v4 — Preregistro F2 (DEV 0xA400–0xA40F, luego CONFIRM 0xB400–0xB40F)

Fijado y subido con push **antes** de cualquier ejecución DEV v4.
- Código congelado en este commit: `orbita/src/constelacion4/mod.rs`, comando `cst4-run`.
- Runner: `orbita/scripts/constelacion_v4.sh` (JOBS=7, nice 10, setsid nohup, reanudable con `.done`; solo se relanzan semillas caídas).
- Análisis: `scripts/analyze_constelacion_v4.py`.

**Contexto.** v3 cumplió la primaria y el no-daño, pero E33 dio 0/16, con h1 ≈ 0.03. Diagnóstico en DIAGNOSTICO.md: x̂(1) se predice antes de observar ninguna transición y sin contexto, así que equivale a persistencia.

## Sistema congelado (`Cfg::v4` = `Cfg::v3` más los cambios siguientes)
1. **Cabeza 1-paso informada por ÓRBITA (`orb_prior`).**
   - Solo cuando el ejemplo trae contexto (E33) y aún no se ha observado ninguna transición (W^F = 0): x̂(1) = ÓRBITA(x0, c).
   - El resto de la dinámica, la plasticidad y el rollout no cambian.
   - La primaria y el no-daño no usan esta cabeza: el held-out se evalúa en h16.
   - Se reporta también `e33_v3head`, que es la misma red con la cabeza v3 (control informativo, `no_orbprior`).
2. **t_f = 128** iteraciones NLMS del campo rápido por frame (v3: 32).
3. **Más entrenamiento:**
   - flujo A: N_A = 3072 episodios (×4);
   - flujo B: 192 nuevos + 192 conocidos (×4).
   - Nota: en tuning el presupuesto lento no movió la primaria (DIAGNOSTICO.md), pero se usa el máximo viable.

El tuning se hizo solo en 0xC4xx (C400–C403 y C410–C413). `cst4-run` rechaza flags de tuning, `--no-*`, `--v2` y `--v3` en dev/confirm.

**Gates previos:**
- Paso 0: todo PASS, incluida la nueva 0h (`--v3` ≡ binario v3, bit a bit).
- F0: PASS.
- F1: gap PASS, sin colapso, E33 4/4 con región 16.

## Criterios (idénticos a v3, salvo el objetivo de E33)
- **Primaria:** por semilla, mediana de log rel_err h16 en familias nuevas {8, 10, 11}, k = 4, CONSTELACIÓN − ÓRBITA-v1.
  - ACEPTADA si IC95 sup < 0 en DEV **y** en CONFIRM.
  - REFUTADA si IC95 inf > 0 en CONFIRM.
  - En otro caso, no concluyente.
- **No-daño:** acc16 known final − pre, con IC95 inf > −δ.
- **E33:** pass por semilla si la región es ≥ 8 con acc ≥ 0.5, empezando en h1. **Objetivo v4: ≥ 15/16 semillas en DEV y en CONFIRM.**
- **Controles:** no_err, no_prec, t1, no_prune, no_sleep, fixed_w, one_layer, backprop y no_fast (informativo).
  - Atribución a PC: CST − {fixed_w, no_err, t1} con IC95 sup < 0 en ambas fases.
- **Estadística:**
  - bootstrap pareado por semilla, 10000 réplicas, semilla 12345, IC95 percentil, δ = 0.05;
  - 0 consultas TEST (assert);
  - NaN = fallo;
  - sin reajuste entre DEV y CONFIRM.
- **Informativo:** v4 − v3 pareado por semilla (cst new_med_log); E33 de la cabeza v3 frente a la de v4.

**Expectativa declarada:**
- E33 ≈ 16/16;
- primaria ≈ −0.1 (como v3, quizá un poco mejor por t_f);
- no-daño OK;
- atribución a PC NO;
- ganancia en h1 atribuible a la cabeza ÓRBITA, no a PC.

**Coste.** Smoke 0xC420 a pleno presupuesto: cst ≈ 377 s, no_prec ≈ 379 s, no_err ≈ 66 s → ≈ 1 h por semilla y ≈ 2–3 h por fase con 7 jobs.
