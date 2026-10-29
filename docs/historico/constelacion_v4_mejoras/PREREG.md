# PREREGISTRO ÔÇö CONSTELACI├ôN v4: mejoras M1/M2/M3

Rama `exp/constelacion-v4-mejoras` (base `exp/constelacion-v4` @ fd80320). Especificaci├│n: contexto de Investigaci├│n IA (M1 tasa de W^F modulada por sorpresa con momento/olvido; M2 regla local meta-aprendida por ES; M3 MLP tanh a├▒adido a W^F). Congelado antes de DEV; sin reajuste en CONFIRM.

## Valores congelados
`frozen.json` sha256 `f5d608fa1258a9088ce9334949ce7fa90db6e22b2b40a6682d5f86f878aef724`
- **M1**: ╬│=0.25, ╬▓=0.5, ╬╗=0, ╬╝_S=-7.0891, ¤â_S=2.8845 (calib. 0xD420ÔÇôD423, 16384 pasos); m_max=min(2,2(1ÔêÆ╬▓)/╬À_F), m_min=min(0.25,m_max). **M1-ctrl**: m constante = m╠ä=0.9320 (media de m de M1 en calibraci├│n).
- **M2**: ES (pop 16, ¤â .05, lr .02, L2 0.1, 50 gen) meta-train familias 0ÔÇô5 / meta-val 6ÔÇô7, semillas 0xC430ÔÇôC437. Ganancia meta-val = 0.00 Ôëñ 0.02 log ÔçÆ **M2 NO ACEPTADA, no va a DEV** (ning├║n ╬© super├│ a ╬©0; val de ╬©0 5.36 vs Ôëñ1.6 para todos los ╬© perturbados). Resultado negativo documentado: `tuning/meta.json`.
- **M3**: H=32, ╬À_V=0.1, ╬À_U=0.05, s3=32, ¤ü=1e-3. Nota: todas las configs M3 empeoran fuertemente el criterio en familias conocidas en tuning (ÔêÆ5.76 vs v4 ÔêÆ10.12); se elige el mejor de la rejilla igualmente (╬À_V=0.1 es el borde inferior).

## Tuning (desviaciones)
- Semillas s├│lo 0xC4xx/0xD4xx (sustituyen C6xx/C7xx/A6xx/B6xx de la especificaci├│n). ╬╝_S/¤â_S calibrados en 0xD420ÔÇôD423 (no en DEV).
- Red base v4 entrenada s├│lo con stream A por semilla (0xC420ÔÇôC423); cada config se eval├║a eval-only en familias conocidas retenidas; criterio = media de med log k=2 y k=8 (no usa familias nuevas ni TEST). Rejilla: v4, 27 M1, 15 M3 ÔåÆ `tuning/`.
- Pesos ├ôRBITA en tuning: `artifacts/orbita/dev/weights_0xA400.json` (no existen pesos tune).
- phat=[0,0] en reid (el v4 original pasaba p verdadero a orbita_held: s├│lo afecta m├®tricas ├ôRBITA secundarias). Contexto acotado a K frames incluso con k=8.

## Paso 0 (`tuning/paso0.json`) ÔÇö PASS
Bit a bit vs v4 con mejora apagada (M1 ╬│=╬▓=╬╗=0 camino general, M2 ╬©0, M3 H=0, M3 H=32 ╬À=0) en 0xC420/C421: hashes id├®nticos. M1 llamadas ├ôRBITA=0 con ╬│=0. Causalidad: 0 fugas; reid_prefixes Ôëí reid (176 checks). Sin NaN (M1 m_max ╬▓Ôêê{0,.5,.9}, M3 m├íx, M3 default). Reproducci├│n completa: `cst4-run --arms cst` vs `cst4m-run --arm v4` smoke 0xC420: hash 1985bdbad5eee41f en ambos. 0 consultas TEST.

## Dise├▒o
DEV 0xA400ÔÇôA40F, CONFIRM 0xB400ÔÇôB40F; pesos ├ôRBITA `artifacts/orbita/{dev,confirm}/weights_<S>.json`. Brazos DEV: v4, m1, m1_ctrl, m3, m3_rf (exploratorio), m3_orb (residual sobre ├ôRBITA, exploratorio). Cada mejora vs v4 solo.
- **Primaria**: ╬ö med log rel_err h16, familias nuevas, k=4; IC98.33% bootstrap pareado (Bonferroni├ù3); Holm como sensibilidad. Bootstrap 10000, semilla 12345.
- **No-da├▒o**: IC95 ╬öacc16 conocidas y ╬öacc h16 E33, l├¡mite inferior > ÔêÆ╬┤ (╬┤=0.05).
- **Secundarias**: vs ├ôRBITA-v1 congelada (m├®tricas de artifacts/constelacion_v4); h1 vs persistencia; M1 vs M1-ctrl.
- **Compuertas DEV por idea**: pass = sup IC98.33 < 0 Ôêº no-da├▒o Ôêº NaNÔëñ1 semilla. Stop (no CONFIRM) = inf IC > 0 Ôê¿ IC95 no-da├▒o entero < ÔêÆ╬┤ Ôê¿ NaN>1 semilla. Las ideas no detenidas van a CONFIRM (+ m1_ctrl si va M1). ALL (M1+M3, exploratorio) se corre en DEV s├│lo si alguna idea pasa, y va a CONFIRM si no se detiene.
- **Confirmada** = pass en DEV y CONFIRM. Comprobaci├│n: hash brazo v4 = hash cst de artifacts/constelacion_v4.
- Cadena: `orbita/scripts/cst4m_chain.sh`; an├ílisis `scripts/analyze_constelacion_v4_mejoras.py`.
