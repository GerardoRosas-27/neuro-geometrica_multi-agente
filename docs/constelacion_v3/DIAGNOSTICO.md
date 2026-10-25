# Constelación v3 — Diagnóstico de v2

Todo el diagnóstico usa solo semillas de tuning 0xC4xx/0xD4xx. No se ejecutó nada en DEV (0xA4xx) ni en CONFIRM (0xB4xx). Se hicieron 0 queries TEST.
Artefactos: `artifacts/constelacion_v3/diag/*.json` (comandos `cst3-diag` y `cst3-oracle`).

Punto de partida: v2 quedó refutada. Primario (med log rel_err h16, familias nuevas, k=4, CST−ÓRBITA): DEV +0.807 [0.702, 0.908], CONFIRM +0.798 [0.688, 0.916].

## 1. Por qué la salida va a ≈0
- Al inicio, rms(pred)/rms(x) = 0.0067. Tras 192 episodios (0xC410) llega a 0.12. rms(W_1) ≈ 0.08 sea cual sea el init (con w_init=1: 0.078).
- La causa son dos topes que actúan juntos:
  - el tope L1 por fila B_in=10 sobre 128 entradas: 3257 reescalados de fila en 192 episodios;
  - el reescalado espectral (28 en 192 episodios; 226–1040 en las semillas base de 768 episodios).
  Con ellos la ganancia máxima por entrada es ≈0.08.
- Sin los topes (`--no-spec`): rms W = 0.357, pred/x = 0.32, h1 = −0.13. La amplitud sube, pero h16 no mejora.
- μ rms ≈ 0.14 frente a x rms ≈ 0.99. El estado latente es pequeño y la lectura h16 se hace solo desde la capa superior, desenrollada con un F_L fijo y contractivo (ρ ≤ 0.99). Así la predicción decae a 0.
- Ninguna de estas variantes da ganancia en h16 (h16 known queda ≈ 0.000–0.04):
  - T=64
  - η_W=0.1
  - sin poda
  - sin sueño
  - sin precisión
- Oráculos lineales (ridge sobre los estados de la red, 0xC410, n=768):

| oráculo | med log h1 / h16 |
|---|---|
| estados capa 1 → x(t+1) | −2.01 (h1) |
| x4, x3 → x5 | −1.54 |
| trivial (persistencia) | −1.60 (h1) / −0.19 (h16) |
| todos los μ → y16 | +0.47 |
| rollout superior → y16 | +0.01 |
| x0..4 → y16 | +0.29 |

  La información de h1 está en la red. Ninguna lectura **global** sirve para h16 porque la dinámica cambia por episodio y la red no tiene contexto.
- Oráculo de ajuste afín **in-context** por episodio (`cst3-oracle`, λ=1e-6), 0xC410/0xC411:
  - known −12.3/−12.3 y nuevas −1.34/−1.17;
  - trivial: known −0.19/−0.24 y nuevas +0.23/+0.11.

**Causa raíz:** v2 no tenía adaptación rápida por episodio ni residuo de persistencia. Una regla lenta y global no puede representar familias cuyo parámetro cambia en cada episodio.

## 2. Por qué colapsan las capas
- En las semillas colapsadas Π_0 queda en pi_min=0.01 en el 100 % de las unidades. Ocurre en DEV 0xA400, 0xA407, 0xA408 y 0xA40A (son del informe v2; no se re-ejecutaron) y en 0xC412, que además tiene η reducido a la mitad.
- Es un bucle de realimentación positiva: errores grandes → Π_0 baja (estimación ML de precisión) → ε ponderado débil → no hay aprendizaje → los errores siguen grandes. Mientras tanto Π en las capas superiores sube a 2.7–5 y domina la energía.
- La inferencia larga diverge. En 0xC413 la energía E es 0.242 con T=8 y 17455 con T=512: κ·Π > 2/curvatura es inestable, y el truncado a T=8 lo ocultaba.

## 3. Por qué la neurona F0 no batía la trivial
v2, 1 neurona, familia 0, 0xC410:
- h1: −0.92, último cuarto −1.23, frente a trivial −1.64.
- h16 known: −0.15 frente a trivial −0.29.
- Con los mismos estados, los oráculos de h16 llegan a −5 / −8.9.
- Con `--no-spec`: h1 −1.18 y h16 −0.149.

Causas:
- no hay residuo de persistencia (predice x̂ desde cero en vez de Δx);
- los topes limitan la ganancia;
- h16 sale de un K fijo en rollout libre, sin realimentar la propia predicción ni ajustar al episodio.

## 4. Correcciones v3 (manteniendo el espíritu)
Se mantiene:
- neuronas ÓRBITA no entrenadas (Koopman en bloques + Brenier fijos);
- campo líquido;
- regla local de tres factores;
- CDT solo en sueño.

Cambios:
1. **Cabeza temporal con residuo:** x̂(t+1) = x(t) + (W^P + W^F)·φ, con φ = [x(t); 0.3/√d·F_1(μ_1); 1].
2. **W^P lento**, regla de tres factores η_P/τ · Π^P · e · φ (factor de utilidad τ), η_P = 1e-3, con precisión ML Π^P ∈ [0.1, 10].
3. **W^F rápido**, campo líquido por episodio: NLMS local en lote sobre los 6 puntos, t_f=32 iteraciones por frame, η_F=1. Se reinicia en cada episodio y no se actualiza con frames imaginados.
4. **Π normalizada** por la media geométrica de cada capa, con límites [0.1, 10]. Esto rompe el bucle de colapso.
5. **Paso de inferencia precondicionado** κ/max(curv+Π+Π^τ, 1). Es estable con T grande.
6. **Rollout reinyectado:** cada predicción se usa como observación imaginada.
7. w_init=1.0 y B_in=40 (menos estrangulamiento de amplitud).
8. Nuevo brazo de control informativo `no_fast` (sin W^F).

Tuning solo en 0xC410–0xC413 (NA=768, NB=48), familias nuevas, CST vs ÓRBITA:
- config congelada D: −1.006/−1.00, −1.205/−1.07, −1.127/−0.90, −0.923/−0.90 → Δ medio ≈ −0.10;
- known ≈ −9.5, sin colapso.

Riesgos conocidos y anotados antes de DEV:
- La exactitud de h1 en E33 es ≈0.02–0.09, porque la primera predicción no tiene información de transición. La región será 0 y E33 probablemente falla por definición.
- La ganancia viene sobre todo del campo rápido sobre [x;1]; feat_s aporta poco. Es probable que la atribución a PC falle (fixed_w ≈ t1).
