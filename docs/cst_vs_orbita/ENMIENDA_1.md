# ENMIENDA 1 al PREREG CST vs ÓRBITA (antes de DEV)

Fecha: 2026-10-26 (antes de lanzar DEV). Se registran todas las desviaciones respecto de `PREREG.md` (7abd4e7) y los valores congelados. Todo el ajuste se hizo **sólo** en semillas 0xC400, 0xC401, 0xD400, 0xD401; ninguna semilla DEV (0xA400–0xA40F) ni CONFIRM (0xB400–0xB40F) se tocó.

## 1. Presupuesto de entrenamiento (medido, no estimado)
- CST (Cfg::v4, `orb_prior=false`, desde cero en flujos A+B): 3840 episodios + 3840 episodios de replay de sueño (240 sueños) = **122 880 transiciones presentadas** (medido en 0xC400 por `cvo-eval`, campo `train_samples`).
- ÓRBITA-eq: **3840 pasos Adam × batch 32 = 122 880 transiciones**, lr 3e-3 → 3e-5 coseno, `lam_h 0.01`, `mh_x 1`, desde cero.
- Desviación: el PREREG decía "una pasada ordenada"; ÓRBITA-eq muestrea i.i.d. del mismo conjunto de 61 440 transiciones (52 224 conocidas + 9 216 nuevas) de A+B, que es su régimen nativo de entrenamiento. Mismo dato, mismo número de muestras presentadas.
- Sin replay del teacher (`replay_fraction=0`) ⇒ la pérdida multi-horizonte queda inactiva. `train_orbita.py` sólo cambia cuando no hay `replay.bin`/replay_fraction=0; la ruta original queda intacta.
- Familias nuevas: contexto = vector cero para ÓRBITA-eq (no conoce su etiqueta; igual que CST, que nunca recibe contexto).

## 2. LAT
- Implementado como LAT_EP (sin biblioteca de memoria λ_m del PREREG): `reid` (familia y p desde k=4 cuadros) + `fit_gn` con lam0=1e-4 sobre los k cuadros, y rollout desde el cuadro 4.

## 3. Evaluación homologada
- Held-out: todos los brazos observan cuadros 0..4 y predicen h16; ÓRBITA usa `reid` también en familias conocidas (sin etiquetas verdaderas).
- **E33 primario = modo k4** (observa target_h 0..4, sin contexto; Δ∈{1,2,4,8,16,32,60}; región ≥ 8 ⇒ pasa). Información idéntica para todos.
- E33-ctx (x0 + contexto verdadero, h 1..64) = protocolo original de ÓRBITA, **sólo informativo** (CST no usa contexto; CST+prior inyecta x̂1 de ÓRBITA-eq).
- Exactitud a 1 paso con contexto sobre todo TEST: informativa, todos los brazos.
- Control "trivial" del PREREG no se implementa en cvo; persistencia sí.

## 4. lr congelado
Ajuste en 4 semillas tune (lr 3e-3 vs 1e-3): l1 final menor con 3e-3 en las 4; ÓRBITA-eq congelado medias known/new med log = −1.337/−0.730 (3e-3) vs −1.084/−0.668 (1e-3). Con LAT −2.413/−0.328 vs −2.649/−0.345 (mixto). **Se congela lr=3e-3** (default de ÓRBITA-v1).

## 5. Paso 0 (reproducibilidad)
- CST de `cvo-eval` vs brazo `cst` de `cst4-run` (smoke, 0xC400): **hash idéntico 8655a689dffdb540**, 240 sueños, held idéntico (known_acc16 0.875, known med log −9.438, new med log −0.837). La ruta de entrenamiento es bit a bit la de v4.
- Paridad PyTorch→Rust de ÓRBITA-eq (0xC400): max_l2_diff 8.9e-16. La cadena exige < 1e-9 por semilla.
- Tiempos: CST ≈ 380–400 s por brazo; ÓRBITA-eq entrenamiento ≈ 11 s; eval ÓRBITA ≈ 5 s (congelado), 11 s (LAT).

## 6. Hipótesis y análisis (sin cambio de fondo)
H1/H2/H3 contra ÓRBITA-eq+LAT, como en el PREREG; H2 con E33-k4 acc a Δ=16. Bootstrap pareado 10000, semilla 12345, δ=0.05. Script: `scripts/analyze_cst_vs_orbita.py`. Regla de paro: NaN/colapso en >2/16 semillas DEV (o <16 semillas completas) ⇒ no se lanza CONFIRM.
