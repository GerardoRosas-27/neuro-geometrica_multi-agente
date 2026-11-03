# PREREG — CONSTELACIÓN v4: campo rápido solo (FIELD_ONLY), sin ÓRBITA

Rama `exp/cst-v4-campo-solo` desde `main` @ 4e0b04a. **Exploratorio**, corrida rápida. Pedido: "entrenar el campo rápido de constelación v4, solo, sin órbitas".

## Brazos (mismo binario `cst4-run`, mismo flujo A+B, misma semilla)
- **v4** (`cst`): CONSTELACIÓN v4 congelada (W lento, W^P, Π, poda, sueño, W^F por episodio y cabeza x̂(1) delegada a ÓRBITA en E33).
- **FIELD_ONLY** (`field_only`): `orb_prior=false` (ninguna llamada a ÓRBITA; x̂(1) = x(0)+(W^P+W^F)φ con W^F=0 al inicio ⇒ persistencia en el 1er paso) y **sólo W^F aprende**: η_W=η_P=η_Π=λ=ζ=ρ_τ=0, sin poda, sin sueño. La red jerárquica queda en su inicialización aleatoria (φ = rasgos fijos), W^P=0. La predicción sale del campo rápido NLMS sobre φ.
- **NO_ORB** (`no_orb`, descomposición): v4 con `orb_prior=false` únicamente (todo lo lento sigue aprendiendo).
- **Persistencia**: x̂(t+h)=x(t) (campo `trivial` / `e33_persist`).
Valores v4 congelados; ningún reajuste.

## Semillas y métricas
DEV 0xA400–0xA40F (si es lento: 0xA400–0xA407, se declara). Pesos ÓRBITA `artifacts/orbita/dev/weights_<S>.json` (sólo para E33 de v4). 0 consultas TEST (aserción en binario).
- **Primaria**: mediana log rel_err h16, familias nuevas, k=4 (`held.new_med_log`). Δ = FIELD_ONLY − v4 y FIELD_ONLY − persistencia; IC95 bootstrap pareado por semilla (10000, seed 12345) de la media de Δ.
- **Secundarias**: acc16 familias conocidas (`held.known_acc16`), E33 (región, 16/16 = pass) y acc E33; mismos contrastes, también NO_ORB − v4.
- Lectura: FIELD_ONLY "mejor" si sup IC95(Δ vs v4) < 0; "peor" si inf > 0; si no, indistinguible. Sin CONFIRM salvo resultado muy prometedor y rápido (entonces 0xB400–0xB40F sin reajuste).

## Paso 0
`cst4-run --phase smoke --seed 0xC420 --arms cst` debe dar hash 1985bdbad5eee41f (añadir brazos no altera `cst`).
