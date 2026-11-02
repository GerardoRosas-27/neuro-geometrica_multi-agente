# PREREGISTRO — CONSTELACIÓN v4: mejoras M1/M2/M3

Rama `exp/constelacion-v4-mejoras` (base `exp/constelacion-v4` @ fd80320). Especificación: contexto de Investigación IA (M1 tasa de W^F modulada por sorpresa con momento/olvido; M2 regla local meta-aprendida por ES; M3 MLP tanh añadido a W^F). Congelado antes de DEV; sin reajuste en CONFIRM.

## Valores congelados
`frozen.json` sha256 `f5d608fa1258a9088ce9334949ce7fa90db6e22b2b40a6682d5f86f878aef724`
- **M1**: γ=0.25, β=0.5, λ=0, μ_S=-7.0891, σ_S=2.8845 (calib. 0xD420–D423, 16384 pasos); m_max=min(2,2(1−β)/η_F), m_min=min(0.25,m_max). **M1-ctrl**: m constante = m̄=0.9320 (media de m de M1 en calibración).
- **M2**: ES (pop 16, σ .05, lr .02, L2 0.1, 50 gen) meta-train familias 0–5 / meta-val 6–7, semillas 0xC430–C437. Ganancia meta-val = 0.00 ≤ 0.02 log ⇒ **M2 NO ACEPTADA, no va a DEV** (ningún θ superó a θ0; val de θ0 5.36 vs ≤1.6 para todos los θ perturbados). Resultado negativo documentado: `tuning/meta.json`.
- **M3**: H=32, η_V=0.1, η_U=0.05, s3=32, ρ=1e-3. Nota: todas las configs M3 empeoran fuertemente el criterio en familias conocidas en tuning (−5.76 vs v4 −10.12); se elige el mejor de la rejilla igualmente (η_V=0.1 es el borde inferior).

## Tuning (desviaciones)
- Semillas sólo 0xC4xx/0xD4xx (sustituyen C6xx/C7xx/A6xx/B6xx de la especificación). μ_S/σ_S calibrados en 0xD420–D423 (no en DEV).
- Red base v4 entrenada sólo con stream A por semilla (0xC420–C423); cada config se evalúa eval-only en familias conocidas retenidas; criterio = media de med log k=2 y k=8 (no usa familias nuevas ni TEST). Rejilla: v4, 27 M1, 15 M3 → `tuning/`.
- Pesos ÓRBITA en tuning: `artifacts/orbita/dev/weights_0xA400.json` (no existen pesos tune).
- phat=[0,0] en reid (el v4 original pasaba p verdadero a orbita_held: sólo afecta métricas ÓRBITA secundarias). Contexto acotado a K frames incluso con k=8.

## Paso 0 (`tuning/paso0.json`) — PASS
Bit a bit vs v4 con mejora apagada (M1 γ=β=λ=0 camino general, M2 θ0, M3 H=0, M3 H=32 η=0) en 0xC420/C421: hashes idénticos. M1 llamadas ÓRBITA=0 con γ=0. Causalidad: 0 fugas; reid_prefixes ≡ reid (176 checks). Sin NaN (M1 m_max β∈{0,.5,.9}, M3 máx, M3 default). Reproducción completa: `cst4-run --arms cst` vs `cst4m-run --arm v4` smoke 0xC420: hash 1985bdbad5eee41f en ambos. 0 consultas TEST.

## Diseño
DEV 0xA400–A40F, CONFIRM 0xB400–B40F; pesos ÓRBITA `artifacts/orbita/{dev,confirm}/weights_<S>.json`. Brazos DEV: v4, m1, m1_ctrl, m3, m3_rf (exploratorio), m3_orb (residual sobre ÓRBITA, exploratorio). Cada mejora vs v4 solo.
- **Primaria**: Δ med log rel_err h16, familias nuevas, k=4; IC98.33% bootstrap pareado (Bonferroni×3); Holm como sensibilidad. Bootstrap 10000, semilla 12345.
- **No-daño**: IC95 Δacc16 conocidas y Δacc h16 E33, límite inferior > −δ (δ=0.05).
- **Secundarias**: vs ÓRBITA-v1 congelada (métricas de artifacts/constelacion_v4); h1 vs persistencia; M1 vs M1-ctrl.
- **Compuertas DEV por idea**: pass = sup IC98.33 < 0 ∧ no-daño ∧ NaN≤1 semilla. Stop (no CONFIRM) = inf IC > 0 ∨ IC95 no-daño entero < −δ ∨ NaN>1 semilla. Las ideas no detenidas van a CONFIRM (+ m1_ctrl si va M1). ALL (M1+M3, exploratorio) se corre en DEV sólo si alguna idea pasa, y va a CONFIRM si no se detiene.
- **Confirmada** = pass en DEV y CONFIRM. Comprobación: hash brazo v4 = hash cst de artifacts/constelacion_v4.
- Cadena: `orbita/scripts/cst4m_chain.sh`; análisis `scripts/analyze_constelacion_v4_mejoras.py`.
