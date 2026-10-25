# CONSTELACIÓN v3 — Preregistro F2 (DEV 0xA400–0xA40F, luego CONFIRM 0xB400–0xB40F)

Este documento se fijó y se subió con push **antes** de cualquier ejecución DEV v3.
- Código congelado en este commit: `orbita/src/constelacion3/mod.rs`, comando `cst3-run`.
- Análisis: `scripts/analyze_constelacion_v3.py`.
- Runner: `orbita/scripts/constelacion_v3.sh` (JOBS=7, nice 10, lanzado con setsid nohup, reanudable con `.done`).

**Contexto.** v2 quedó refutada: DEV +0.807, CONFIRM +0.798. El diagnóstico y las correcciones están en DIAGNOSTICO.md.

Resultados de las fases previas:
- Paso 0: todo PASS.
- F0: PASS (≈8 nats mejor que la trivial).
- F1: gap known PASS, sin colapso, **E33 FALLA** (acc h1 ≈ 0.03).

Expectativa declarada antes de DEV:
- primaria con Δ ≈ −0.07 a −0.10, quizá no concluyente con 16 semillas;
- E33 falla;
- atribución a PC "NO", porque la ganancia viene del campo rápido.

**Sistema congelado (`Cfg::v3`).** Igual que v2 (PREREG v2) salvo:
- w_init = 1.0 y B_in = 40;
- Π ∈ [0.1, 10], normalizada por la media geométrica de cada capa;
- paso de inferencia κ/max(curv+Π+Π^τ, 1), con κ=0.1 y T=8;
- cabeza temporal x̂(t+1) = x(t) + (W^P + W^F)·φ, con φ = [x; 0.3/√d·F_1(μ_1); 1];
- W^P lento: regla de tres factores, η_P = 1e-3, Π^P ∈ [0.1, 10];
- W^F rápido (campo líquido por episodio): NLMS, η_F = 1, t_f = 32 iteraciones por frame, reinicio en cada episodio, sin actualizar con frames imaginados;
- rollout reinyectado; h1 por predict1;
- CDT solo en sueño (sin cambios).

El tuning se hizo solo en 0xC4xx (C400–C403, C410–C414). `cst3-run` rechaza flags de tuning, `--no-*` y `--v2` en dev/confirm.

**Protocolo, primaria, no-daño, E33 y estadística:** idénticos a PREREG v2.
- **Primaria:** por semilla, mediana de log rel_err h16 en H nuevas {8,10,11}, k=4, CONSTELACIÓN − ÓRBITA-v1 sola.
  - Aceptada si IC95 sup < 0 en DEV **y** en CONFIRM.
  - Refutada si IC95 inf > 0 en CONFIRM.
  - En otro caso, no concluyente.
- **No-daño:** acc16 known final − pre, con IC95 inf > −δ.
- **E33:** pass si la región es ≥ 8 con acc ≥ 0.5.
- **Estadística:** bootstrap pareado por semilla, 10000 réplicas, semilla 12345, IC95 percentil, δ = 0.05.
- 0 consultas TEST (assert).
- Una semilla con NaN cuenta como fallo.
- Sin reajuste entre DEV y CONFIRM.

**Controles:** los mismos de v2 (no_err, no_prec, t1, no_prune, no_sleep, fixed_w, one_layer, backprop) más **no_fast** (sin W^F), que es informativo.
- **Atribución a PC:** CST − {fixed_w, no_err, t1} con IC95 sup < 0 en ambas fases (regla de v2 sin cambios).
- **Informativo:** comparación pareada v3 − v2 por semilla (cst new_med_log).

**Coste medido.** Smoke 0xC414: 663 s por semilla (10 brazos, 1 hilo) → ≈ 35 min por fase con 7 jobs.
- Smoke 0xC414 (1 semilla): primaria −0.069, no_fast −3.35, fixed_w ≈ 0, t1 ≈ 0, E33 cst 0/1.
