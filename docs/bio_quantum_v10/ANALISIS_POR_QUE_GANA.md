# ¿Por qué BQ10 (CORE + M7 DA-STR) le gana a ATLAS-FW (ALL)?

Rama `exp/bio-quantum-v10-rust` @ 7d9cf75. **Análisis sólo de lectura**: código + artefactos existentes (`artifacts/bio_quantum_v10/{bq_tune,bq_dev,bq_confirm}`, `analysis.json`, `DIAGNOSTICO_M8.md`, `docs/atlas_fw/README.md`). 0 corridas nuevas incluidas, 0 TEST, nada congelado se tocó.

> Nota de transparencia: se lanzó un análisis instrumentado (8 semillas 0xC4C0–C7 + 4 con red v4 sin entrenar) y luego se canceló por petición de rapidez; sus salidas se descartaron sin leerlas y el código de instrumentación se revirtió. Por eso **no hay** desglose por familia, por horizonte h1..h16, por k ni normas de coeficientes: los artefactos guardados sólo tienen agregados por semilla (h16, familias nuevas juntas). Esos puntos quedan marcados como *no medido*.

## 1. Qué hace M7 en el código (exacto)

`atlas.rs`:
- `LIB_DEG = [1,1,2,2,2,3,3,3,3,1,1,2,2,1,1,2,2]` → grado de las 17 funciones de biblioteca de M1 (`a,b | a²,ab,b² | a³,a²b,ab²,b³ | sin2a,sin2b | cos2a−1,cos2b−1 | tanh2a,tanh2b | r·a,r·b`).
- `colscale`: s_j = 1 para las columnas φ_v4 (j < dp) y s_j = da^{−(grado−1)/2} para la biblioteca.
- `Online::refresh` (sólo si `m.m7`): multiplica las columnas de ψ por s, ajusta `sparse_fit` (ridge λ + STLSQ) en el espacio reescalado y des-escala w_j ← s_j·u_j.

Equivalencia: ridge sobre u con penalización λ‖u‖² y w = s·u ⇒ penalización **λ·da^{grado−1}·w_j²** sobre el coeficiente original. Con da=10 y el λ elegido (1e-3 en 100 % de semillas DEV/CONFIRM/tune): lineal 1e-3, cuadrático **1e-2**, cúbico **1e-1** (×1, ×10, ×100). φ_v4 no cambia. Con `thr=0` (elegido siempre por BQ10) STLSQ no actúa, así que M7 es puro Tikhonov ponderado por grado.

Dónde entra: sólo en la adaptación rápida en contexto (cuadros 1..K=4) y en la selección de (λ, thr) de `meta_train` (que usa el mismo `Online`). No toca la red v4, ni la compuerta M6, ni el rollout, ni ÓRBITA.

**Régimen**: d = 148 columnas (dp=131 φ_v4 + 17 biblioteca) contra K·6 puntos = 24 filas por salida. El sistema está muy **sub-determinado**; con λ=1e-3 el ridge es casi el interpolante de norma mínima. M7 cambia *la métrica de esa norma mínima*: entre todas las soluciones que ajustan los 4 cuadros, prefiere la que pone masa en términos lineales/φ_v4 y no en cúbicos (que explotan al iterar el rollout de 12 pasos).

## 2. Diferencias BQ10 vs ATLAS-FW por componente

| componente | qué hace | ALL (ATLAS-FW) | CORE | BQ10 |
|---|---|---|---|---|
| red v4 entrenada (W^P, φ_v4 = estados/lectura, Π/precisiones, W^F) | da φ_v4 y la predicción lenta x(t)+W^P φ | sí (misma red por semilla, hash idéntico) | sí | sí |
| M1 biblioteca 17 fns ×0,3 | amplía ψ = [φ_v4; s·g(x)] | sí | sí | sí |
| M2 ridge+STLSQ en contexto | reemplaza NLMS de v4 | sí | sí | sí |
| M3 hiperred contexto → Θ0 | prior | sí, pero α=0 elegido ⇒ inactivo | no | no |
| M4 familias sintéticas | meta-datos | sí (sólo afecta Θ̄/H, que α=0 anula) | no | no |
| M5 objetivo rollout h5..h16 | criterio de elección de (λ,thr) | sí | no (one-step h5) | no (one-step h5) |
| M6 compuerta CLS vs v4 | elige v4 si su residuo en contexto ≤ ATLAS | sí | sí | sí |
| **M7 DA-STR** | Tikhonov ×da^{grado−1} sobre biblioteca | no | no | **sí, da=10** |
| M8 Brenier | cota de radio en rollout | no | no | no (BQR10 = BQ10 en mediana, ver DIAGNOSTICO_M8) |

Consecuencia: en CONFIRM **ALL ≡ CORE** (Δ = 0.0000 exacto, 16/16 semillas); en DEV difieren sólo en 0xA40A y 0xA40F (vía M5/thr). Por tanto BQ10 − ALL ≈ **efecto aislado de M7**.

## 3. Dependencias (verificado en código)

- **Red v4**: ambos la usan íntegra y congelada en evaluación (`run_ep`: `net.c.fast=false`, sin plasticidad). Usan W^P (parte lenta en `pred`), φ_v4 de `net.feat` (estados inferidos μ / lectura), la dinámica de inferencia (`constelacion_step`, que usa Π, W) para estados y rollout (`step_ex`). W^F (NLMS rápido) de v4 **no** se usa en la ruta ATLAS (M2 lo sustituye), pero sí en la rama v4 de la compuerta M6. Que el aprendizaje lento de v4 casi no aporte a la primaria es un resultado previo (no re-medido aquí).
- **ÓRBITA**: sólo entra en E33 (`e33_atlas`: sustituye x̂(1) por `d.step` si `orb_prior`). No entra en la primaria (`eval_held_atlas` no la usa). Artefactos existentes: E33 con ÓRBITA 16/16 para v4, ALL, CORE y BQ10 en DEV y CONFIRM; **sin ÓRBITA (`e33_v3head`) 0/16 para los cuatro**, acc h1 0,056 (DEV) / 0,052 (CONFIRM). ⇒ E33 depende de ÓRBITA de forma idéntica en ambos; M7 no aporta nada a E33.
- Ablaciones "sin v4" (red sin entrenar / compuerta forzada): *no medido* (cancelado).

## 4. Evidencia cuantitativa (artefactos existentes)

### 4.1 Por semilla, Δ new_med_log BQ10 − ALL
- DEV: gana en **13/16** semillas; rango −0,443 … +0,182; media −0,139 [−0,220, −0,055].
- CONFIRM: gana en **11/16**; rango −0,548 … +0,241; media −0,156 [−0,269, −0,043].
- Pierde en 0xA408 (+0,13), 0xA40B (+0,18), 0xA40F (+0,09), 0xB401 (+0,08), 0xB403 (+0,17), 0xB404 (+0,12), 0xB40A (+0,24), 0xB40F (+0,07).

### 4.2 Dosis–respuesta λ (tune 0xC4B0–B3, 4 semillas, CORE = da "1")
| brazo | new_med_log | new_acc16 | known_acc16 | frac v4 (nuevas) | frac v4 (conocidas) |
|---|---|---|---|---|---|
| CORE | −1,208 | 0,146 | 0,900 | 0,099 | 0,875* |
| BQ3 | −1,339 | 0,167 | 0,893 | 0,099 | 0,875 |
| BQ10 | **−1,416** | 0,146 | 0,885 | 0,115 | 0,867 |
| BQ30 | −1,415 | 0,125 | 0,879 | 0,130 | 0,852 |
(*CORE conocidas de DEV/CONFIRM ≈0,873.) Mediana: mejora monotónica hasta 10 y **meseta** en 30; las 4 semillas mejoran con cada paso CORE→BQ3→BQ10. known_acc16 y new_acc16 **empeoran monotónicamente** con da.

### 4.3 Compuerta M6
Fracción de episodios nuevos en que sale v4: DEV 0,090 (ALL) vs 0,090 (BQ10); CONFIRM 0,087 vs 0,095. Conocidas: 0,874→0,858 (DEV), 0,872→0,864 (CONFIRM). ⇒ La compuerta **casi no cambia**: la ganancia no viene de enrutar distinto, sino de que la rama ATLAS predice mejor en el ~91 % de nuevos donde ya se usaba.

### 4.4 Generalización fuera de contexto (proxy de varianza)
`meta.best_obj` = media log err **1 paso en h5** (fuera de los 4 cuadros vistos) sobre 96 episodios de meta-validación, mismo criterio para CORE y BQ10:
- CORE −7,32 (DEV/CONFIRM, todas las semillas en [−7,54, −7,06]); BQ10 **−8,07** ([−8,25, −7,91]). Δ ≈ **−0,75 nats en 32/32 semillas**.
Esto es error de generalización 1-paso a un cuadro no visto con los mismos 24 puntos de ajuste: el regularizador reduce el sobreajuste del interpolante, no cambia la capacidad.

### 4.5 Coeficientes cúbicos vs lineales; desglose por familia / horizonte / k
*No medido* (no hay coeficientes ni errores por h/familia guardados). Por construcción, la penalización cúbica efectiva es ×100 la lineal; la predicción cuantitativa de norma queda pendiente.

## 5. Hipótesis principal

**M7 gana por reducción de varianza (sobreajuste) en un ajuste sub-determinado de 24 puntos × 148 columnas**: con λ chico, ALL/CORE elige un interpolante que reparte masa en términos cúbicos/cuadráticos que ajustan los 4 cuadros pero amplifican el error al iterar 12 pasos; DA-STR desplaza la solución hacia términos lineales/φ_v4, más estables en el rollout.

A favor:
- Mecanismo matemático exacto (Tikhonov por grado) y que actúa sólo en la adaptación en contexto.
- Error 1-paso fuera de contexto mejora −0,75 nats en 32/32 semillas (4.4).
- Dosis–respuesta monotónica hasta da=10 con meseta (4.2).
- No es efecto de compuerta (4.3) ni de M8 (DIAGNOSTICO_M8) ni de M3–M5 (inactivos).

En contra / límites (coste de sesgo):
- **new_acc16 baja** (DEV 0,155→0,145; CONFIRM 0,156→0,148) y known_acc16 baja ~0,01–0,016 (DEV 0,897→0,888; CONFIRM 0,903→0,887): cuando la dinámica verdadera *sí* necesita términos cuadráticos/cúbicos para quedar bajo 5 % de error, el encogimiento introduce sesgo. M7 mejora la **mediana** (episodios de error moderado) a costa de los casos casi exactos. Sigue pasando no-daño (> −0,05).
- da=30 ya no mejora la mediana y empeora más los acc ⇒ el óptimo está en el balance sesgo–varianza, no en "más regularización".
- 8/32 semillas pierden (hasta +0,24): el efecto es de población, no uniforme.
- E33: sin efecto (depende sólo de ÓRBITA).
- La mejora 1-paso (−0,75) es mucho mayor que la de h16 (−0,15): la parte de la ganancia que sobrevive al rollout está limitada por la dinámica de la red v4 en el rollout.

Pendiente (requiere correr, no hecho): desglose por familia 8/10/11 y por horizonte, barrido k (2–8) para comprobar que el efecto decrece al aumentar puntos de contexto (predicción clave de la hipótesis de varianza), normas de coeficientes por grado, y ablación con red v4 sin entrenar.
