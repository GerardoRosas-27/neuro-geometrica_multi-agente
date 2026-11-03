# CONSTELACIÓN v2 — Paso 0 (bloqueante: 0a–0c)

Rama `exp/constelacion-v2` desde `main` 590dd93. Spec: `constelacion-spec.md` v2 (2026-10-25). Código: `orbita/src/constelacion/mod.rs` (`orbita cst-paso0 --seed 0xC400`). Artefactos: `artifacts/constelacion_v2/paso0_run{1,2}.json` (dos procesos nuevos).

Nota sobre el encargo: la plantilla «γ=0 ≡ ÓRBITA bit a bit» no aplica a la v2 (la v2 quita ÓRBITA como ancla; la red parte de cero). Se aplica el Paso 0 de la spec v2 (0a–0e). La equivalencia con ÓRBITA se sustituye por: la referencia ÓRBITA-v1 se evalúa en el mismo H/k=4 que ESTELA (`orbita_held`, = brazo `orbita` k4 de ESTELA), y en F2 se compara por semilla con los valores publicados de ESTELA.

| Puerta | Criterio | Resultado |
|---|---|---|
| 0a mismo camino | `constelacion_step(train)` es la única función; train con plasticidad off (η_W=η_Π=λ=ζ=ρ_τ=0, sin poda/sueño) vs eval, los 9 brazos, 16 episodios × 17 cuadros | **PASA**: 27 936 valores, 0 discrepancias |
| 0b reproducibilidad | misma semilla → mismos pesos/métricas en 2 procesos nuevos | **PASA**: pesos `8e14c5880d854d46`, métricas `a139b204454bcfbc`, datos (H+flujo B) `a35386552788de67` idénticos en ambos |
| 0c estabilidad init | ρ(K)<1, cota J_ψ, 1000 pasos sin aprendizaje sin divergencia | **PASA**: 28 neuronas, máx ρ(K)=0,9896, cota ‖J_ψ‖ ≤ 1,107; F libre 1000 pasos máx ‖μ‖=23,3 (finito; capa sin contracción global por J_ψ>1, acotada); inferencia 1000 pasos máx E=5,29 |
| 0d gradiente local [D] | corr(ΔW_PC, −∂L/∂W por DF) > 0,9, red L=2 n=(2,1) | **PASA**: corr media 0,99999 (5 ensayos). Primer intento con Π^τ_top=1000 y κ=0,05 dio 0,05: la inferencia era inestable (κ·Π≫2). Se corrigió sólo la numérica de la prueba (Π^τ_top=50, κ=0,02, T=20 000), no la red |
| 0e costo | μs por punto-paso, 1 hilo | **203 μs** train, **180 μs** eval (spec estimaba ≈400). W = 10 496 pesos; 76 números fijos por neurona |

Estimación real (para PREREG): ≈40–45 s por brazo en flujo A (768 episodios) en 1 hilo; F2 ≈ 9 brazos × ~60 s ≈ 9 min/semilla → ≈ 25–30 min por fase con 7 jobs. CPU es suficiente; no GPU.

**Veredicto Paso 0: PASA (0a, 0b, 0c); 0d y 0e informativos, 0d pasa.**
