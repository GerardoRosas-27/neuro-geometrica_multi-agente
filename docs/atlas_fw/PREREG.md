# ATLAS-FW — Prerregistro

Escrito y publicado ANTES de cualquier ejecución DEV. Rama `exp/atlas-fw` desde `exp/constelacion-v4` @ `fd80320`. No toca `main`. 0 TEST queries (asertado en cada métrica).

## Base y benchmark (idénticos a CONSTELACIÓN v4)
- Base: `Cfg::v4` (entrenamiento flujo A/B, N_A=3072, N_NEW=N_KNOWN_B=192), held-out `gen_heldout(seed,16)`, k=K=4, horizonte h16 (clamp_err), `known_acc16` (REL_ERR_OK=0.05), E33 (pass si región ≥ 8).
- v4 se entrena UNA vez por semilla (idéntico a `cst4-run --arms cst`); todos los brazos reutilizan esa red (clonada, `fast=false`) y solo cambian el estimador de adaptación rápida.
- Verificado: `atlas-run` brazo v4 ≡ `cst4-run --arms cst` bit a bit (hash, held, pre, colapso, E33, órbita) en 0xC4A0 (na=48, nb=8): `artifacts/atlas_fw/off_eq_v4_check.json`. Todo off ≡ v4 (test `off_is_v4_bit_exact`; paso0: 8448 valores, 0 discrepancias).
- Pesos ÓRBITA-v1: semilla 0x{A,B}48i ↔ `artifacts/orbita/{dev,confirm}/weights_0x{A,B}40i.json`.

## Mecanismos (interruptor por mecanismo, `orbita/src/constelacion4/atlas.rs`)
- M1 biblioteca Koopman/SINDy (17 funciones: polinomios ≤3, sin/cos(2·), tanh(2·), r·x, r·y; escala 0.3).
- M2 estimador disperso iterado (ridge + STLSQ, 3 iteraciones) en lugar de NLMS.
- M3 contexto (dim 9) + hiperred lineal ridge (λ_H=1e-2) → prior Θ0.
- M4 meta-entrenamiento con 50% de familias sintéticas (Rot/Scale/Shear/SinShear/TanhShear/Radial/Twist/Quad), auditadas contra las reglas retenidas {8,10,11} (rejilla 9×9, tolerancia 0.05) y excluidas; rechazo de inestables.
- M5 objetivo de meta-validación = rollout h5..h16 (en vez de 1 paso).
- M6 compuerta CLS: residuo precuencial t=1..K; si v4 ≤ ATLAS usa v4. M6 solo ≡ v4 por construcción (declarado).
- Rejilla de meta-selección congelada: α∈{0,0.5,1}, λ∈{1e-3,1e-2,1e-1}, thr∈{0,0.05}; N_FIT=192, N_VAL=96 (sólo familias de entrenamiento). M3/M4/M5 solos pueden elegir α=0 (≡ v4); se declara y se reporta.
- Brazos: v4, M1, M2, M3, M4, M5, M6, ALL.

## H0 (cota oráculo; ejecutada antes del prereg, semillas DEV/CONFIRM de v4 0xA400–0xA40F / 0xB400–0xB40F)
Archivos `artifacts/atlas_fw/h0_{dev,confirm}_v4seeds.json`. Oráculo familia verdadera − v4 (new_med_log): DEV −26.195 [−26.332, −26.056]; CONFIRM −26.12 [−26.203, −26.05]. Criterio de falsación (oráculo − v4 > −0.2) NO se cumple → H0 no falsada; hay margen. Oráculo biblioteca genérica − v4: DEV −0.419 [−0.527, −0.301]; CONFIRM −0.258 [−0.336, −0.161].

## Semillas (nuevas, no solapadas con ninguna usada: A40,A41,A45,A46,A47 / B ídem)
- DEV: 0xA480–0xA48F (16). CONFIRM: 0xB480–0xB48F (16). Smoke (no inferencial): 0xC4A0–0xC4AF.

## Análisis (`scripts/analyze_atlas_fw.py`)
- Bootstrap pareado por semilla, 10000 réplicas, semilla 12345, IC percentil. δ=0.05.
- Primaria: V − v4 en new_med_log. Singles M1..M6: IC al nivel 1−0.05/6 (Bonferroni). ALL: IC95.
- Pasa fase si: 16/16 semillas, 0 NaN, IC sup(V−v4 new_med_log) < 0; no-daño: IC95 inf(V−v4 known_acc16) > −δ e IC95 inf(known_acc16 final − pre) > −δ; E33 pass ≥ 15/16.
- Gate DEV → CONFIRM solo con brazos que pasan (+v4). Veredicto PASA si pasa en DEV y CONFIRM; REFUTADO si en CONFIRM IC inf > 0.
- Validez: veredicto SOLO con 16/16 semillas sin error (.done, rc=0, sin panic, JSON válido con todos los brazos, finitos); si no, `CHAIN_STOP_ERROR` sin veredicto.

## Expectativa
Sin números. Se espera que ALL mejore new_med_log respecto a v4 sin daño en conocidas gracias a M6; M1 solo puede dañar conocidas.
