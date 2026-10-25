# MAREA — Preregistro (congelado antes de DEV)

**Sistema** (`orbita/src/marea/mod.rs`, modelo `artifacts/marea/pretrain/model.json`, hash aec4e1359cce050d):
1. Predictor JEPA latente congelado: z=E(x_t/s_t) (DZ=16, MLP+skip), ẑ = z + P([z, c_t]); objetivo JEPA contra E_EMA(x_{t+1}/s_t) + SIGReg de momentos (estilo LeJEPA); sonda dec post-hoc (ridge + MLP); x̂ = s_t·dec(ẑ). Preentrenado en fam. 0–5 (semilla 0xD400), 200k pasos.
2. Memoria test-time (Titans/ATLAS): ẑ += W[z;1]; tras observar x_{t+1}: S ← β S + η m r φᵀ, W ← (1−λ) W + S, r = E(x_{t+1}/s_t) − ẑ.
3. Tercer factor m = tanh(a(b − ln err_t)); θ=(η,β,λ,a,b,ρ) meta-aprendidos por ES en fam. 6–7 (no vistas por JEPA), semilla 0xD401: η=0.1146, β=0.0257, λ=0.00357, a=−1.802, b=−5.162, ρ=0.00244.
4. Niveles: W rápido por episodio; W lento ← W lento + ρ(W − W lento) si el episodio ganó.
5. Hopfield ±: clave = c_t normalizado, valor = W final, signo = ganancia vs η=0 en el episodio; recuperación en t=K=4 (softmax β=8 sobre +, se omite si el mejor − es más similar); capacidad 64, desalojo por menor |ganancia|.
6. Rollback por bloque de 4 pasos: si ln err del bloque > η=0 + 0.05 ⇒ W ← snapshot, S ← 0.
7. Actor recursivo (TRM): y ← y + A([s, y]) R=3 veces sobre el resumen de transiciones observadas; c_t → predictor.

**Protocolo.** Flujo por semilla = gen_stream de ECO (48×{8,10,11} nuevas + 48 conocidas, 192 episodios), memoria persiste entre episodios. En cada t=0..15 se predice x_{t+1} desde x_t verdadero, luego se observa y se aprende (precuencial).

**Primaria.** Pendiente OLS de ln rel_err_t vs t dentro del episodio, media en nuevas; MAREA − MAREA(η=0). H1 ACEPTADA si IC95 superior < 0 en DEV y CONFIRM; REFUTADA si IC95 inferior > 0 en CONFIRM; si no, no concluyente.
**Secundaria.** Media ln err en nuevas, MAREA − η=0 < 0 (IC95 sup < 0 en ambas fases).
**No-daño (veto).** Media ln err en conocidas, MAREA − η=0: IC95 superior < +δ en ambas fases.
**Línea base externa.** ÓRBITA+LAT (LAT ajustado en los 4 primeros pasos), media ln err 1 paso t≥4 en nuevas: MAREA − ÓRBITA+LAT (informativo, con veredicto mejor/peor/no concluyente).
**Controles** (MAREA − control, pendiente y media): η=0, m aleatorio U(−1,1), sin rollback, sin Hopfield, claves barajadas, JEPA aleatorio (enc/pred/actor aleatorios, sólo sonda; mismo θ), K=1 (actor con 1 recursión).
**Estadística.** Bootstrap pareado por semilla 10000, semilla 12345, δ=0.05 (`scripts/analyze_marea.py`).
**Reglas.** 0 consultas TEST (assert). DEV 0xA400–0xA40F, luego CONFIRM 0xB400–0xB40F sin reajuste. Pesos ÓRBITA por semilla sólo para la línea base.

**Desviaciones / decisiones declaradas.** (1) Un único modelo preentrenado para todas las semillas: la varianza entre semillas es sólo de datos/flujo. (2) «K=1» se interpreta como recursiones del actor (inferencia; el modelo se entrenó con R=3). (3) JEPA a escala de juguete (DX=12), no V-JEPA 2. (4) La sonda dec no es parte del JEPA puro (lectura necesaria para rel_err). (5) SIGReg simplificado a momentos (media/covarianza), no la prueba de Epps–Pulley de LeJEPA.
