# ÓRBITA-Nova — Preregistro congelado (fase 2)

**Estado**: CONGELADO. Escrito y empujado **antes** de generar datos DEV.  
**Rama**: `exp/orbita-nova` (base `main` `590dd93`; fase 0 `413e06b`; fase 1 `eb17263`).  
**Fecha**: 2026-10-22 (America/Mexico_City, UTC−6).  
**Autorización**: reacción ❤️ del usuario al resumen de fase 1 que recomendaba familias primarias **{8, 10, 11}**, excluir **{9}**, y preguntaba si escribir el preregistro así.

Diseño: [`DISENO.md`](DISENO.md). Plan: [`PLAN_EJECUCION.md`](PLAN_EJECUCION.md). Techo: [`FASE1_RESULTADOS.md`](FASE1_RESULTADOS.md).

**Regla dura**: nada de reajuste de código, hiperparámetros, umbrales ya fijados, familias, métricas ni criterios entre DEV y CONFIRM. Tras el commit de este archivo, el único valor que aún se *calibra una vez* en DEV (según §5) es τ / umbral de recuperación sobre H DEV familias 1–2; ese valor se congela en el informe DEV y se reutiliza idéntico en CONFIRM. Sin bytes. Sin `run_v6`. Sin entrenar el núcleo ÓRBITA ni Gemma. 0 queries a TEST.

---

## 1. Familias y horizonte

| conjunto | ids | rol |
|---|---|---|
| Conocidas | 0–7 | no-daño / olvido (N3, N6) |
| **Nuevas primarias** | **{8, 10, 11}** | métrica primaria e hipótesis N1–N5 |
| Nueva fuera del primario | **{9}** (espiral) | **excluida** del primario (suelo blando fase 1); sólo secundaria / sensibilidad si se reporta |

Nombres: 8 giro ondulado, 9 espiral, 10 pulso radial, 11 cizalla saturada (`fam::rule`).

- **k primario**: 4. Secundarios: 2 y 8.
- **k_fit / k_val** (C5, validación retenida intra-episodio):

| k | k_fit | k_val |
|---|---|---|
| 2 | 1 | 1 |
| 4 | **3** | **1** |
| 8 | 6 | 2 |

---

## 2. Seeds, datos y mezcla del flujo

| fase | seeds (n=16) | pesos ÓRBITA |
|---|---|---|
| DEV | `0xA400`–`0xA40F` | `artifacts/orbita/dev/weights_*.json` (congelados; sha256 antes/después) |
| CONFIRM | `0xB400`–`0xB40F` | `artifacts/orbita/confirm/weights_*.json` (idéntico protocolo) |

- Distribución DEV/CONFIRM: episodios **generados** con contexto verdadero (`fam::rule`); **sin** H/Gemma en el criterio principal; **sin bytes**.
- **Flujo por seed**: **2400** episodios. Orden y k por episodio con `default_rng(seed)` (Xoshiro/equivalente documentado en el harness).
- **Mezcla del flujo**: **80 %** conocidas (0–7) / **20 %** nuevas **primarias** {8, 10, 11} (uniforme entre las tres). Fam 9 **no** entra en el flujo primario.
- k de cada episodio del flujo: uniforme en {2, 4, 8} (misma RNG).
- **Held-out H** (evaluación con memoria/biblioteca congelada, sin guardar): 16 episodios/familia en conocidas 0–7 ∪ nuevas primarias {8,10,11} (= 176); RNG de construcción H distinta del flujo S (documentada en el harness al cablear fase 3). k_eval primario = 4; secundarios 2 y 8.
- Puntos de control n ∈ {**0, 150, 300, 600, 1200, 2400**} episodios del flujo vistos (N2 usa 2400 vs 0; N3 usa 2400 vs 150 en conocidas).

Bootstrap: pareado por seed, **10 000** réplicas, semilla **12345**, IC95. δ = **0,05**.

---

## 3. Métricas

- **Primaria**: Δ mediana de `log(rel_err h16)` por episodio (continua; **menor es mejor**), sobre **nuevas {8,10,11}**, **k=4**. Contrastes = diferencia de medianas (o de la métrica agregada por seed documentada en el análisis) entre brazos; signo: negativo = mejora del primer brazo.
- **Secundarias** (reportadas, sin veto salvo donde una hipótesis lo cite): acc16 (umbral rel_err < **0,05**), `log(rel_err)` / acc en h32 y h64, k∈{2,8}, AUROC de sorpresa (nuevas vs conocidas), tasa de recuperación C6, pureza de cluster (fracción de peso/soporte de la misma familia verdadera), tamaño de biblioteca, coste (s/episodio), curvas precuenciales por bloques de 300 episodios.
- **Olvido**: acc16 y mediana `log(rel_err h16)` en conocidas 0–7 a n=0 vs n=2400 (y el contraste N3 a n=150).

---

## 4. Brazos (controles incluidos)

| brazo | definición |
|---|---|
| ÓRBITA (OBS) | núcleo congelado; re-id / free-run como baseline |
| ÓRBITA+C3 | sólo filtro de innovación (estado); sin C4/C6 |
| LAT_EP | adaptador latente denso por episodio (`latent.rs`), sin memoria |
| LAT | LAT con memoria episódica (protocolo latente ya publicado; hiperparámetros de su propio prereg si se reutiliza artefacto; si se re-ejecuta aquí, mismos λ_0/λ_m congelados del ciclo LAT) |
| **NOVA** | C3+C4+C5+C6+C7 completo |
| NOVA−C3 | sin filtro de innovación |
| NOVA−C5 | sin árbitro de validación retenida (siempre acepta C4 si converge) |
| NOVA−C6 | sin biblioteca (sin recuperación ni consolidación) |
| NOVA_SHUF | igual que NOVA pero firmas de biblioteca **permutadas** entre clusters al recuperar |
| **MLP** | mismo nº de parámetros adaptables que C4 (= `2·r·M` = **32** con r=2, M=8), misma inferencia GN/LM por episodio, sin estructura UVᵀ de rango fijo |
| ORACLE_EP | techo por episodio (Module residual Φ deg=2, λ=1e-4; o C4_EP reportado) |
| ORACLE_FAMNET | Module Φ un módulo por familia verdadera (expectativa baja, fase 1) |

Núcleo C2: **nunca** se actualiza. sha256 de pesos idéntico antes/después de cada seed.

---

## 5. Hiperparámetros congelados (y única calibración permitida)

| símbolo | valor | notas |
|---|---|---|
| r (rango C4) | **2** | ΔK = U Vᵀ; init LoRA U~ε, V=0 |
| λ_C4 (prior GN) | **1e-6** | fase 0/1 |
| ε espectral (C5) | **0,5** | aceptar sólo si ‖I+ΔK‖ ≲ 1+ε |
| N_consolidación | **50** | sueño C6 cada 50 episodios del flujo (lote); admisión sólo si mejora retenida y no-daño en conocidas del lote |
| σ ruido (N6) | {**0,01**; **0,05**} | gaussiano en observaciones |
| τ / umbral de recuperación C6 | **calibración única en DEV** | distancia L2 de firma: percentil **95** de la distancia al vecino incorrecto (o de S_known análoga) en episodios H DEV de familias **1–2** (16 seeds pooled), **antes** de correr el flujo DEV completo; el valor numérico se escribe en `FASE3_DEV` / RESULTADOS y **no** se retoca en CONFIRM. Opción «ninguno» obligatoria si ningún vecino ≤ τ. |

No hay otra malla de hiperparámetros entre DEV y CONFIRM. Si el harness aún no implementa C6 real, fase 3 no arranca hasta cablearlo conforme a este documento (stubs actuales = no-op).

---

## 6. Hipótesis N1–N6

Aceptada si se cumple en **DEV y CONFIRM**. Refutada si en CONFIRM el IC95 queda entero del lado contrario. Inconclusa en otro caso. Contrastes sobre la **métrica primaria** salvo donde se indique acc16; «nuevas» = {8,10,11}; k=4 salvo secundarias.

- **N1**: NOVA − LAT_EP, nuevas: IC95 < 0 (Nova mejora a la adaptación por episodio sin memoria).
- **N2**: NOVA(n=2400) − NOVA(n=0), nuevas: IC95 < 0 (aprende con experiencia = largo plazo).
- **N3** (no-daño): NOVA − ÓRBITA en conocidas, acc16, inf. IC95 > −δ; **y** NOVA(n=2400) − NOVA(n=150) en conocidas, inf. IC95 > −δ.
- **N4**: NOVA − NOVA_SHUF **y** NOVA − MLP, nuevas: IC95 < 0 (la estructura importa).
- **N5** (ablaciones): NOVA − NOVA−C5 **y** NOVA − NOVA−C6, nuevas: IC95 < 0.
- **N6** (ruido): con σ ∈ {0,01; 0,05}, NOVA no peor que ÓRBITA+C3 en conocidas (inf. IC95 > −δ).

### Éxito / fracaso de la rama

- **Éxito** ⇔ **N1 ∧ N2 ∧ N3 ∧ N4** (aceptadas en DEV y CONFIRM).
- **Fracaso** ⇔ N1 o N3 **refutada**, **o** N2 **no aceptada** (no hay aprendizaje a largo plazo).
- N5 y N6 se reportan siempre; no vetan el éxito de rama por sí solas.
- Se publica el informe igual en éxito o fracaso (`RESULTADOS.md` o `docs/lecciones/` si falla).

---

## 7. Fuera de alcance (este ciclo)

- Bytes / periferia multilingüe (fase 5, preregistro aparte, sólo si CONFIRM tiene éxito).
- Reentrenar ÓRBITA, Gemma, D2/D3, o invocar `run_v6`.
- Merge a `main`.
- Incluir familia 9 en el primario.
- Reajustar τ, r, N_consolidación, mezcla, familias o criterios tras ver números de DEV.

---

## 8. Cadena de ejecución (referencia; no forma parte del freeze de números)

`setsid nohup`, reanudable con `.done`, `nice 10`, sin compartir CPU con otras ramas. Análisis reproducible desde artefactos bajo `artifacts/orbita/nova_*/`. Tras CONFIRM: fase 6 informe.
