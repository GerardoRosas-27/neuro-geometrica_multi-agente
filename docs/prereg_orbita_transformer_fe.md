# Prerregistro — ÓRBITA-TF: Transformer sobre estados continuos + energía libre vs ÓRBITA-v1

**Rama:** `exp/orbita-tf` (desde `exp/orbita-yo` `0b6f12a`). Escrito y empujado **antes** de entrenar/evaluar DEV y CONFIRM. Fecha 2026-10-20 (CST, UTC−6).
Código nuevo sólo en `orbita/tf/` (`tfe.py`, `eval_tf.py`, `gen_eps.py`, `analyze_tf.py`, `chain.sh`). No se toca `src/` (v4/v5), ni `orbita/src`,
ni pesos de ÓRBITA (se usan los publicados `artifacts/orbita/{dev,confirm}/weights_*.json`, congelados).

## 1. Idea
Sistema tipo ÓRBITA que opera **directamente sobre estados continuos** (6 puntos 2-D = vector de 12) con **arquitectura Transformer (atención)** y un
**objetivo de energía libre variacional** (inferencia activa / codificación predictiva): F = error de predicción ponderado por precisión + complejidad
(KL del latente respecto a su prior). **Sin tokens ni preentrenamiento de lenguaje**: los elementos de atención son vectores de estado; no hay
vocabulario, ni texto, ni Gemma en ningún brazo de este experimento (tampoco en ÓRBITA: el contexto de las familias nuevas es nulo, ver §4).

## 2. Brazos
| brazo | arquitectura | params |
|---|---|---|
| **TFE** (principal) | Elementos de atención: 6 embeddings de punto (Linear 2→12) + 1 elemento latente μ ∈ ℝ⁸ (Linear 8→12). 1 bloque Transformer pre-LN (d = 12, 2 cabezas, FFN 44, GELU), salida residual por punto q' = q + Linear(12→2). Prior amortizado q(μ\|c) = N(m(c), σ(c)²) con MLP tanh 10→24→16. Entrenamiento: μ muestreado (reparametrización) y pérdida = ÓRBITA (relativa 1 paso + λ_H multi-horizonte + sesgo) + β·KL(q‖N(0,I)), **β = 1e-3**. | 2618 |
| **TF** (control sin FE) | Mismo Transformer, μ = m(c) determinista, sin KL. | 2418 |
| **MLP** (control) | MLP por punto tanh [q, c] (12) → 44 → 44 → 2, residual. | 2642 |
| **ÓRBITA-v1** (comparador) | Núcleo de Koopman ortogonal + corrector de Brenier (ICNN), pesos publicados por seed, congelado. | 2602 |

Presupuesto igual a ÓRBITA (`train_orbita.py`): mismos datos por seed (`/workspace/orbita-data/{dev,confirm}/<seed>`, TRAIN real 50 % + replay
del teacher 50 % ponderado por confianza), **256 000 pasos Adam, batch 32, lr coseno 3e-3→3e-5**, λ_H = 0.01 sobre k ∈ {1,2,4,8} con γ = 0.9
(objetivos del teacher, sólo replay; error relativo por paso recortado a 10), λ_B = 1. float64, 1 hilo, determinista por seed.
Parámetros dentro de ±7 % de ÓRBITA (no idénticos; declarado).

## 3. Inferencia en test (energía libre)
Con k observaciones y₁…y_k desde x₀ (contexto **desconocido**):
1. **OBS** (todos los brazos, igual que ÓRBITA `reid`): candidato c ∈ 8 familias × p ∈ {(0,0)} ∪ malla 9×9; se elige el de menor Σ_t min(rel_err, 10).
2. **FE** (sólo TFE): desde μ₀ = m(c*) minimiza F(μ) = Σ_{t≤k} rel_err_t² /(2·0.05²) + ½‖(μ − m(c*))/σ(c*)‖² con 100 pasos Adam (lr 0.03),
   guardando la iteración de menor F. Luego free-run desde y_k hasta t = 16.
3. **PE** (TF y, como ablación, TFE): igual sin el término de complejidad (sólo error de predicción).
Sorpresa/energía libre final F se usa como detector de error.

## 4. Benchmark (sin ds.test; 0 queries a TEST)
- Conjuntos de episodios propios, independientes de la seed: `fam-set --nf 12 --n 64` con RNG **0xD7F0 (DEV)** y **0xC7F0 (CONFIRM)**
  (768 episodios: 64 por familia 0–11; 0–7 = familias v4, 8–11 = nuevas no afines de `fam.rs`). Trayectorias de 16 pasos con la regla verdadera.
- **acc16** = rel_err(ẑ₁₆, y₁₆) < 0.05 (`v4_metrics::correct`), también acc8.
- **Conocidas (c verdadero)**: free-run 16 pasos desde x₀ con c = context_of(fam, p) (para ÓRBITA = `fam-eval`, columna `true_ok16`).
- **k observados (c desconocido)**, k ∈ {2,4,8} (k = 2 primario como en los ciclos anteriores): TFE-FE, TFE-PE, TFE-OBS, TF-PE, TF-OBS, MLP-OBS;
  ÓRBITA vía `lat-eval exp1` con contexto nulo (p̂ = 0): OBS, LAT_EP (adaptación latente de Koopman por episodio), LAT (con memoria), ORACLE_LS;
  τ₂ = 0.014656, τ₄ = 0.022020, τ₈ = 0.037142 y λ₀ = 1e-4, λ_m = 1e-3 fijados en `resultados_latente_largo_plazo.md` (no se recalibran).
- Seeds DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F` (n = 16 por fase); ÓRBITA usa sus pesos publicados de esas mismas seeds.

## 5. Hipótesis (δ = 0.05; IC95 bootstrap pareado por seed, 10 000 remuestreos, semilla 12345; cada una debe cumplirse en DEV **y** CONFIRM)
- **H1 (conocidas, c verdadero, acc16):** TFE − ÓRBITA. *Superior* si IC95 > 0; *no inferior* si límite inferior > −δ. Refutada si IC95 sup < −δ.
- **H2 (nuevas 8–11, k = 2, acc16, adaptación online):** H2a TFE-FE − ÓRBITA-LAT_EP > 0 **y** H2b TFE-FE − ÓRBITA-OBS > 0 (IC95). H2c (vs LAT
  con memoria) y k = 4, 8 secundarios.
- **H3 (calibración, k = 2, todos los 768 episodios):** AUROC de F_TFE para predecir fallo de TFE-FE a h16. H3a: IC95 inferior de AUROC − 0.5 > δ.
  H3b: AUROC_TFE − AUROC_ÓRBITA (s_known → fallo de ÓRBITA-OBS) IC95 > 0. Seeds sin ambas clases se excluyen (se reporta n).
- **H4 (aporte de la energía libre, nuevas k = 2):** H4a TFE-FE − TF-PE > 0; H4b TFE-FE − TFE-OBS > 0 (inferencia iterativa); H4c (sec.)
  TFE-FE − TFE-PE (término de complejidad); H4d conocidas TFE − TF. Controles: TFE-FE − MLP-OBS, TFE − MLP en conocidas.
- **H5 (coste, descriptivo):** segundos de pared de entrenamiento y de evaluación por 768 episodios; predicción: TFE más caro que ÓRBITA.
Etiquetas: **aceptada** (criterio en ambas fases), **refutada** (IC95 de CONFIRM entero del lado contrario), **inconclusa** (resto).

## 6. Protocolo y desviaciones conocidas
- Sin cambios de código ni hiperparámetros entre DEV y CONFIRM; una sola cadena `setsid nohup orbita/tf/chain.sh` (reanudable por `.done` y
  checkpoints cada 10 000 pasos), DEV y luego CONFIRM; análisis `analyze_tf.py`.
- **Sin tuning**: β, d, FFN, lr de inferencia y nº de pasos fijados a priori. Antes de este prerregistro sólo se hizo una prueba de humo (20 000
  pasos, seed 0xA400, evaluada en un conjunto aparte RNG 0xE7F0) para comprobar que el código corre; además, durante la prueba de `lat-eval` se vio
  la primera fila del CSV de ÓRBITA con el conjunto DEV (RNG 0xD7F0), que se borró. Ningún número de esa prueba se usa ni cambió el diseño.
- Limitación conocida: el entrenamiento sólo ve transiciones de las 8 familias v4 con su c; las familias 8–11 son fuera de distribución para
  todos los brazos.
