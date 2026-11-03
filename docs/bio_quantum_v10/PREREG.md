# PREREG — Bio-Quantum v10 (exp/bio-quantum-v10-rust) portado al arnés CONSTELACIÓN v4

Escrito y empujado **antes** de TUNE/DEV. Sin re-ajuste entre DEV y CONFIRM. 0 consultas TEST.

## Auditoría del benchmark original (`bio_quantum_engine --benchmark-suite`) — NO comparable
- Generador propio (`protocol_cst4.rs`): 6 puntos, dinámicas reescritas a mano; las familias "nuevas" 8/10/11 son Duffing/Lorenz-like/Van der Pol inventadas, **no** las `rule` 8/10/11 del arnés v4; familias conocidas tampoco son `v4_dataset::apply_rule`.
- "DEV/CONFIRM" = **una sola semilla** cada una (0xA400 / 0xB400 vía `evaluate_phase_rust(seed,64)`), sin entrenamiento v4 por semilla, sin bootstrap.
- Números de v4 **hardcodeados** en la tabla final (6.9 %, 87.5 %, 0.0 %…), no se ejecuta v4.
- "E33" propio (h∈{1,2,4,8,16} sobre su generador), no el E33 de v4 (`e33_score`). `new_acc16 = 100 %` y med log −8.7 reflejan un problema trivial para un ridge polinómico exacto, no capacidad comparable.
- Conclusión: se descarta su número; se porta la idea al arnés v4.

## Port (orbita/src/constelacion4/atlas.rs, sobre ATLAS-FW de main 1c845d6)
Ideas transferibles de `unified_engine.rs`: adaptación rápida por ridge exacto sobre base polinómica ≤3 (≈ M1+M2 de ATLAS), **DA-STR** (Tikhonov por grado: cúbicos penalizados más que afines) y **cota Brenier-Lyapunov** con dilatación hiperbólica en el rollout. No portados: Ley de Dale (proyección W=W^E−W^I es cosmética, no cambia la predicción), memoria CDT por episodio y prior tálamo-cortical (usan contexto/episodios no disponibles sin fuga), BTSP/piramidales (no afectan la salida en v10).
- **M7 DA-STR**: λ_j = λ·da^(grado_j−1) sobre la biblioteca (grados `LIB_DEG`), φ_v4 sin cambio; implementado por reescalado de columnas.
- **M8 Brenier**: r_b = 2.5·max r(cuadros 0..K)·g^(h−K), g=(r_K/r_0)^(1/K) si >1.03; compresión suave tanh (como v10).
- Brazos: `CORE` = M1+M2+M6 (ATLAS sin M3–M5, ablación), `BQ{da}` = CORE+M7, `BQR{da}` = CORE+M7+M8, da∈{3,10,30}. `ALL` = ATLAS-FW (mejor versión, main). `v4` = CONSTELACIÓN v4 @ fd80320.
- Con M7/M8 apagados el código previo es idéntico (tests `atlas` 7/7 OK, incl. `off_is_v4_bit_exact`).

## Paso 0
`cst4-run --phase smoke --seed 0xC420 --arms cst` con este binario: hash **1985bdbad5eee41f** (= referencia) → PASS (`paso0_cst4run_0xC420.json`).

## Ajuste (sólo 0xC4B0–0xC4B3)
Grilla fija: v4, CORE, BQ3/10/30, BQR3/10/30. Regla congelada en `scripts/bq10_tune_pick.py`: menor media new_med_log con known_acc16 ≥ v4−0.05, E33 4/4, 0 NaN; empate → menor da sin M8; ninguno → CORE. Resultado → `frozen.json` (commit+push automático antes de DEV). Brazos DEV/CONFIRM: v4, ALL, CORE, primario y su hermano (BQ↔BQR mismo da).

## Diseño y métricas
- DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F (pesos ÓRBITA congelados `artifacts/orbita/{dev,confirm}/weights_<S>.json`), presupuesto v4 completo.
- **Primaria**: mediana log rel_err h16 familias nuevas k=4 (`new_med_log`), primario − v4. Éxito: IC95 sup < 0 y |Δ| ≥ δ=0.05, en DEV **y** CONFIRM.
- **Co-primaria declarada**: primario − ALL (ATLAS-FW), misma regla.
- No-daño: known_acc16 Δ IC95 inf > −0.05; E33 16/16; 0 NaN.
- Bootstrap pareado por semilla 10 000, seed 12345, IC95 percentil. Veredicto final = CONFIRM; negativos se reportan tal cual.
