# CONSTELACIÓN v4 — Diagnóstico de E33 (solo semillas de tuning 0xC4xx; ninguna DEV/CONFIRM)

Datos: `artifacts/constelacion_v4/tune/*/smoke/metrics_0xC41{0..3}.json` (brazo cst, ÓRBITA de referencia `artifacts/orbita/dev/weights_0xA40i.json`, igual que F1 v3) y F1 v3 (0xC400–0xC403).

## Qué exige E33 exactamente (código `e33_score`, sin cambios desde v2)
- Ejemplos: los 64 ejemplos TEST v4 de familias iterables (rotate + reflect), leídos como `orbita eval` (contador de consultas TEST = 0, sha verificado).
- Episodio: se observa x0 y luego frames reales x1..x4 (k = 4). Después hay rollout reinyectado hasta h = 16.
- acc_h = fracción de ejemplos con rel_err < 0.05 en h ∈ {1, 2, 4, 8, 16}.
- **región** = el mayor h tal que acc ≥ 0.5 en **todos** los h' ≤ h, **empezando en h1**. **pass ⇔ región ≥ 8.**
- Por tanto, si acc_h1 < 0.5 la región vale 0, aunque acc_h16 = 1.0. Así ocurría en v3.

## Por qué acc h1 ≈ 0.03 en v3
- **Qué es la predicción h1.** `preds[0]` = predict1 en t=1. Se hace tras observar solo x0, con W^F recién reiniciado (= 0): x̂(1) = x0 + W^P·φ(x0).
- **W^P no puede inferir la transformación del episodio.** Es lento y global, y la familia o el parámetro cambian en cada ejemplo. La red **no recibe el contexto** e.context. x̂(1) queda prácticamente en persistencia.
- **Números en 0xC410–0xC413 (acc h1 / h2 / h4 / h8 / h16):**

| semilla | persistencia (x̂=x0) | CST v3 (cabeza v3) | ÓRBITA (usa contexto) |
|---|---|---|---|
| 0xC410 | 0.08 / 0.55 / 0.53 / 0.52 / 0.52 | 0.08 / 0.81 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 / 0.64 / 0.30 |
| 0xC411 | 0.02 / 0.52 / 0.50 / 0.50 / 0.50 | 0.02 / 0.70 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 0.94 / 0.56 / 0.22 |
| 0xC412 | 0.06 / 0.56 / 0.52 / 0.50 / 0.50 | 0.06 / 0.92 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 / 0.83 / 0.30 |
| 0xC413 | 0.09 / 0.56 / 0.55 / 0.52 / 0.53 | 0.09 / 0.78 / 1.0 / 1.0 / 1.0 | 0.98 / 0.98 / 0.95 / 0.53 / 0.19 |

  - El h1 de CST v3 coincide **exactamente** con la persistencia (0.02–0.09). F1 v3 en 0xC400–0xC403 daba 0.02–0.05.
  - Persistencia ≈ 0.5 a partir de h2: las reflexiones son involuciones.
  - **Conclusión:** sin contexto, la información disponible en h1 es insuficiente. **No es un problema de entrenamiento:** ningún presupuesto de entrenamiento de W^P lo arregla, porque la transformación es por ejemplo.
- **Cuello secundario.** Con t_f = 32, h2 (1 transición observada) da 0.70–0.92, con poco margen sobre 0.5.

## ¿Ayuda más entrenamiento? (0xC410–0xC413, brazo cst)

| variante | primaria (CST − ÓRBITA, new med log h16) | acc E33 h2 | coste |
|---|---|---|---|
| v3, N_A = 768 | −0.008 / −0.189 / −0.161 / −0.095 | 0.81 / 0.70 / 0.92 / 0.77 | 98 s |
| N_A = 3072 (×4) | −0.008 / −0.189 / −0.161 / −0.095 (idéntica) | igual | 485 s |
| N_B = 192 + 192 (×4) | −0.008 / −0.189 / −0.161 / −0.094 | igual | 146 s |
| t_f = 64 | −0.009 / −0.189 / −0.211 / −0.075 | 0.98 / 1.0 / 1.0 / 0.98 | 97 s |
| **t_f = 128** | −0.009 / −0.189 / −0.238 / −0.075 | **1.0 / 1.0 / 1.0 / 1.0** | 104 s |
| t_f = 256 | = t_f 128 | 1.0 | 111 s |
| η_F = 0.5 | ≈ igual | 0.33–0.48 (peor) | 94 s |

- **Entrenamiento lento más largo.** Más episodios de flujo A o B **no cambian** la primaria: la ganancia la aporta W^F dentro de cada episodio. Solo mueve el known med log en ±1 nat, con acc16 known sin cambios (0.875).
- **Paso del campo rápido.** Más iteraciones NLMS por frame (t_f) sí mejoran h2 y, ligeramente, la primaria en 0xC412.
