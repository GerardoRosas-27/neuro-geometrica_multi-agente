# Preregistro — CONSTELACIÓN v4 vs ÓRBITA con entrenamiento igualado

Estado: **congelado antes de cualquier ejecución DEV/CONFIRM.** Rama `exp/mejor-arquitectura`. Ninguna corrida DEV/CONFIRM se lanza con este documento; cualquier cambio posterior exige enmienda fechada antes de DEV.

## 1. Pregunta
Con los mismos datos, el mismo presupuesto de entrenamiento y el mismo acceso a adaptación en test, ¿CONSTELACIÓN v4 predice mejor que ÓRBITA? La revisión rápida (`REVISION_RAPIDA.md`) mezcló tres factores: arquitectura, adaptación por episodio (W^F) y entrenamiento distinto.

## 2. Brazos
| Brazo | Descripción |
|---|---|
| **CST** | CONSTELACIÓN v4 congelada (`Cfg::v4`, `cst4-run`), **sin** `orb_prior` (cabeza v3 en h1) para que no dependa de ÓRBITA. |
| **CST+prior** (secundario) | CST v4 tal cual (h1 de ÓRBITA). Sólo informativo. |
| **ÓRBITA-eq** | ÓRBITA-v1 (`orbita/`, sin cambios de arquitectura) reentrenada con el presupuesto igualado de §3. |
| **ÓRBITA-eq+LAT** | ÓRBITA-eq + adaptación por episodio LAT (`orbita/src/latent.rs`, hiperparámetros ya fijados en `resultados_latente_largo_plazo.md`: λ₀ = 1e-4, λ_m = 1e-3). Rival adaptativo equivalente a W^F. |
| Controles | CST `no_fast` (sin W^F); ÓRBITA-v1 original (256 000 pasos); persistencia; trivial. |

## 3. Igualación de entrenamiento
- **Datos:** ambos brazos ven exactamente el mismo flujo de entrenamiento por semilla: el TRAIN generado por la semilla + los flujos A (3072 episodios) y B (192 nuevos + 192 conocidos) de CST v4, en el mismo orden. ÓRBITA-eq **sin** replay del teacher (el replay es información extra que CST no ve). Familias nuevas {8, 10, 11} nunca en entrenamiento.
- **Presupuesto primario = número de transiciones de entrenamiento vistas** (muestras). ÓRBITA-eq: Adam batch 32, lr coseno 3e-3→3e-5 (como ÓRBITA), con pasos = ⌈muestras_CST / 32⌉ y una sola pasada en el mismo orden. Se registra el número exacto de muestras de cada brazo y debe coincidir (±1 batch).
- **Cómputo:** se reporta tiempo de pared y FLOPs estimados por brazo; no es criterio de decisión.
- **Pesos iniciales:** ninguno de los brazos parte de pesos de ÓRBITA-v1 entrenada (CST sin `orb_prior`).
- **Adaptación en test:** mismo contexto k (k = 4 held-out; en E33 las mismas transiciones observadas) para CST y ÓRBITA-eq+LAT.

## 4. Seeds y tuning
- DEV 0xA400–0xA40F y CONFIRM 0xB400–0xB40F (16 + 16), mismo binario y commit en ambas fases, sin reajuste entre fases.
- Tuning (sólo lr de ÓRBITA-eq, en {3e-3, 1e-3}, y verificación de que el número de muestras coincide) **exclusivamente** en 0xC4xx/0xD4xx; congelado y anotado en enmienda antes de DEV.

## 5. Métricas (ambos protocolos)
- **Protocolo CST** (held-out, k = 4, h16): med log rel_err nuevas {8,10,11}; acc16 nuevas; med log y acc16 conocidas.
- **Protocolo ÓRBITA** (E33, n = 64 por semilla, mismo `test_sha256`): acc h1/h2/h4/h8/h16/h32/h64, acc 1 paso, región, pase E33.

## 6. Hipótesis y criterio principal
- **Primaria (H1):** med log rel_err h16 nuevas, k = 4, **CST − ÓRBITA-eq+LAT** < 0. Aceptada si el IC95 superior < 0 en DEV **y** CONFIRM.
- **H2:** acc h16 E33, CST − ÓRBITA-eq+LAT > 0 (IC95 inferior > 0 en ambas fases).
- **H3 (no-daño):** acc16 conocidas, CST − ÓRBITA-eq+LAT > −δ (IC95 inferior > −0,05 en ambas fases).
- **H4 (informativa):** las mismas comparaciones contra ÓRBITA-eq congelada y contra ÓRBITA-v1 original.
- **Atribución:** CST − CST `no_fast` (se espera > 0 de mejora, es decir IC sup < 0); si CST pierde contra ÓRBITA-eq+LAT, la ventaja de la revisión rápida se atribuye a entrenamiento/adaptación, no a la arquitectura.

**Decisión:** CST «gana» sólo si H1 ∧ H3. ÓRBITA «gana» si el IC95 inferior de H1 > 0 en ambas fases y su no-daño simétrico se cumple. En otro caso: **empate / no concluyente**. δ = 0,05; bootstrap pareado por semilla, 10 000 réplicas, semilla 12345, IC95 percentil; estimación = media entre semillas de la diferencia pareada.

## 7. Calidad y parada
- Asserts: 0 consultas TEST, 0 NaN, igualdad de muestras de entrenamiento por brazo, mismo `test_sha256` en todos los brazos.
- Paso 0 (antes de DEV): CST con este binario reproduce bit a bit `cst4-run` v4 en una seed de tuning; ÓRBITA-eq con 256 000 pasos y replay reproduce ÓRBITA-v1 (`weights_0xC5xx`) — si no, se detiene y se corrige.
- Parada: si en DEV hay NaN o colapso de capa en > 2/16 semillas en cualquier brazo, o un fallo de assert, se detiene sin CONFIRM y se reporta. Seeds caídas se relanzan con el mismo binario (reanudable vía `.done`); no se descartan.
- No se modifica nada entre DEV y CONFIRM.

## 8. Tiempo estimado
- CST v4 con todos sus brazos tardó ≈ 3100–4770 s de pared por semilla (`wall_s` en `metrics_0xA400/0xB400`; CST solo ≈ 380–390 s). Con CST, CST+prior, `no_fast` (≈ 3 × 390 s ≈ 20 min/seed) y 7 jobs: ≈ 1 h por fase.
- ÓRBITA-eq: el entrenamiento ÓRBITA-v1 original no reporta tiempo de pared por semilla en los docs (no encontrado). Con presupuesto igualado (≈ 10⁵ muestras frente a 8,2 M originales) se espera muy por debajo de la ejecución original; se medirá en tuning. LAT añade minutos por semilla.
- **Estimación total: ≈ 3–5 h** (tuning + Paso 0 ≈ 1 h; DEV ≈ 1–2 h; CONFIRM ≈ 1–2 h), sujeta a la medición en tuning.

## 9. Pendiente de implementación (antes de DEV, sin tocar seeds DEV/CONFIRM)
- Flag de presupuesto de muestras y desactivar replay en el entrenamiento ÓRBITA.
- Evaluación de CST hasta h64 y acc 1 paso en E33.
- Script `scripts/analyze_cst_vs_orbita.py` con los contrastes de §6.
