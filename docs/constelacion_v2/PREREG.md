# CONSTELACIÓN v2 — Preregistro F2 (DEV 0xA400–0xA40F, luego CONFIRM 0xB400–0xB40F)

Fijado antes de cualquier ejecución DEV. Código congelado en este commit (`orbita/src/constelacion/mod.rs`, `cst-run`); análisis `scripts/analyze_constelacion_v2.py`; runner `orbita/scripts/constelacion_v2.sh` (JOBS=7, nice 10).

**Contexto honesto.** F0 y F1 fallaron (ver F0.md, F1.md: CONSTELACIÓN ≈ predicción nula, brecha +2,2 a +2,7 log frente a ÓRBITA-v1 en conocidas). Por la spec esto era STOP; el usuario ordenó continuar. Expectativa declarada: primaria no aceptada.

**Sistema (valores congelados, `Cfg::base`).** m=8, h=8, η_b=0,1, init aleatoria estable (ρ~U(0,90,0,99), ω~U(−π/4,π/4), R~N(0,1/h), b~U(−1,1)); L=3, n=(16,8,4); T=8, κ=0,1, tol ΔE/E 1e-3; η_W=0,01, λ=1e-6, W~N(0,1/fan_in)·0,1; η_Π=1e-3, Π∈[0,01,100]; τ∈[1,10], ζ=0,01, ρ_τ=0,01; P=500 pasos, w_min=1e-4, u_min=0,01·Ū (2 ventanas), crecimiento 1 par/capa/P; B_in=10 (L1 por fila), ρ_max=0,99; clip μ ±10; sueño cada N_s=16 episodios, 16 réplicas priorizadas por E (excluye >p95), α_s=0,1, episodios − sólo precisión; vigilancia: buffer de 16 conocidas, si E sube >5 % 3 sueños seguidos → η_W/2. Semilla de init = semilla del experimento.

**Protocolo por semilla.** Flujo A: 768 episodios conocidas 0–7 (96/fam) → eval «pre» en H conocidas. Flujo B (= ECO): 48×{8,10,11} + 48 conocidas = 192, barajado. Eval final en H (16/familia, 0–7 y 8,10,11; generador idéntico a ESTELA/ECO), k=4 observados, sin plasticidad, predicción de y16 por rollout descendente desde t=4. ÓRBITA-v1 sola = pesos `artifacts/orbita/{dev,confirm}/weights_S.json` congelados, ctx verdadero (conocidas) / reid k=4 (nuevas), rollout 16 desde x0 (= brazo `orbita` k4 de ESTELA; se verifica igualdad con ESTELA por semilla).

**Primaria (única).** Por semilla, mediana de log rel_err h16 (acotado a 10) en H nuevas {8,10,11}, k=4: CONSTELACIÓN − ÓRBITA-v1 sola. **Aceptada** si IC95 sup < 0 en DEV **y** CONFIRM; **refutada** si IC95 inf > 0 en CONFIRM; si no, no concluyente.
**No-daño (guarda).** acc16 H conocidas: CONSTELACIÓN final − CONSTELACIÓN pre (tras flujo A); IC95 inf > −δ.
**E33.** Reportado frente a ÓRBITA-v1 (pass = región ≥ 8 con acc ≥ 0,5); lectura del TEST iterable v4 como en ECO (contador de consultas = 0, assert). Para CONSTELACIÓN h≤4 = predicción a un paso filtrada, h>4 = rollout desde t=4.
**Controles (mismo binario, flags):** no_err (reservorio: x entra a la capa superior por U_in fija, sin ε ascendente, delta sólo en W_1), no_prec (Π≡1), t1 (T=1), no_prune (C≡1, τ fijo), no_sleep, fixed_w (η_W=0, Π sí aprende), one_layer (L=1, n=28), backprop (W por gradiente truncado de 1 paso de la pérdida h1 a través de la generación descendente, mismo η y número de pasos; Π con su regla local — desviación declarada: Π no está en el camino de predicción).
**Atribución a PC:** CONSTELACIÓN − {fixed_w, no_err, t1} en primaria con IC95 sup < 0 en ambas fases. backprop sólo se reporta.
**Estadística.** Bootstrap pareado por semilla, 10 000 réplicas, semilla 12345, IC95 percentil, δ=0,05. 0 consultas TEST (assert). NaN → semilla cuenta como fallo (rel_err=10).
**Reglas.** Sin reajuste entre DEV y CONFIRM; `cst-run` rechaza hiperparámetros por CLI en dev/confirm. Ajuste sólo en 0xC4xx (hecho: F0/F1/tune/smoke 0xC405).
**Costo medido.** Smoke 0xC405: 321 s/semilla (9 brazos, 1 hilo) → ≈ 18–20 min por fase con 7 jobs.
