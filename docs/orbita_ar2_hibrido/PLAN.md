# PLAN — exp/orbita-ar2-hibrido (ÓRBITA × AR2)

Rama `exp/orbita-ar2-hibrido` desde `exp/orbita-vs-timesfm` @ 2d9ff63. Estado: **solo plan + código esqueleto + humo**.
DEV **no** ejecutado. 0 consultas a TEST.

## Motivación (resultados ya publicados, no nuevos)
- dinlat DEV: G(ORB_LIBRE vs AR2, ho) = −0.0647 [−0.1040, −0.0262] → AR2 (VAR(2) ridge α=1) mejor en promedio.
- vstimesfm CONFIRM: G(ORB_SESGO vs TIMESFM, ho) = +0.3701; ORB_LIBRE ≈ TimesFM en conjunto, pierde fuerte en familias
  geométricas (rotar, trasladar, escalar, cizalla) y gana en no geométricas; AR2 supera a TimesFM.
- Lectura: ORB_LIBRE y AR2 fallan en regiones distintas → una combinación dependiente del contexto puede superar a ambos.

## Diseño del híbrido (`orbita/hibrido/run_hib.py`)
Expertos base = exactamente los de `orbita/dinlat/run.py` (mismo código, mismos seeds ⇒ mismos modelos; con
`DINLAT_RES` el script ABORTA si PERS/AR2/ORB_LIBRE/ORB_SESGO no coinciden con `res_<seed>.json` de dinlat).

| Brazo | Definición | Ve etiquetas |
|---|---|---|
| AR2, ORB_LIBRE, ORB_SESGO, PERS | baselines de dinlat | solo ORB_SESGO |
| MIX50 | 0.5·AR2 + 0.5·ORB_LIBRE (control fijo) | no |
| **HIB_GATE** (primario) | ŝ_{t+1} = w_AR2·AR2_t + w_LIB·LIB_t, w = softmax(W f_t + b) | no |
| HIB_HEDGE | w_k ∝ exp(−η·err_pasado_k), η en grilla {0,…,32} sobre cross-fit | no |
| HIB_RES | AR2_t + R(S,t); R = OrbLibre − s_t entrenado sobre residuos de AR2 (≈0 al inicio) | no |
| HIB_GATE_S | compuerta sobre {AR2, ORB_LIBRE, ORB_SESGO} | sí (vía ORB_SESGO) |
| ORACLE_FAM | techo: por familia, mejor de {AR2, ORB_LIBRE} con errores del propio split | n/a (no es modelo) |

**Características de la compuerta (solo pasado):** en el paso t (objetivo s_{t+1}), para cada experto k:
log de la media del error de un paso en transiciones τ=1..t−1 (pred_τ vs s_{τ+1}, τ+1 ≤ t) y log del último error,
más t/T. Nada de s_{t+1} en adelante (test `past_feats` lo verifica perturbando el futuro).

**Entrenamiento de la compuerta sin fuga train→train:** predicciones fuera de pliegue K=4 (cross-fit por episodio
sobre train; AR2 y ORB re-entrenados por pliegue); la compuerta (lineal+softmax, init uniforme, Adam 1500 pasos,
wd 1e-3) se ajusta con MSE de la mezcla. En evaluación usa los expertos entrenados en todo train.
ho/tf nunca se usan para ajustar nada (salvo ORACLE_FAM, que es solo techo descriptivo).

## Datos y semillas
Mismos episodios, embeddings Gemma-2-2B capa 13 cacheados y latentes (PCA 16 + Q_rand) de dinlat:
`/workspace/orbita-data/dinlat/{dev,confirm}`. 16 DEV (0xA400–0xA40F) + 16 CONFIRM (0xB400–0xB40F). No re-embebe.
Controles C1 (permutado) y C2 (barajado) reconstruidos con el mismo rng que dinlat.

## Archivos
- `orbita/hibrido/run_hib.py` — brazos por seed. `analyze.py` — tablas/IC/criterios. `chain.sh PHASE` — orquestación (8 en paralelo).
- `orbita/hibrido/test_hibrido.py` — humo (embebido sintético, STEPS=100): identidad con dinlat, reproducibilidad
  bit a bit, anti-fuga de características, claves. Resultado: OK (~13 s/corrida). Sus números son de datos sintéticos
  y no tienen valor de evidencia.

## Ejecución (pendiente, no hecha)
1. `bash orbita/hibrido/chain.sh dev` → `/workspace/orbita-data/hibrido/dev/analysis.txt`; RESULTADOS.md (DEV).
2. Si `gate_dev_a_confirm` (media G(HIB_GATE vs AR2, ho) > 0): congelar hiperparámetros y `chain.sh confirm`.
   Las hipótesis se deciden SOLO en CONFIRM. Si no pasa: se reporta DEV negativo y no se corre CONFIRM.

## Estimación de cómputo DEV
Medido en el box (8 núcleos), STEPS=200, embebido sintético: dinlat/run.py 4.7 s vs run_hib.py 26.6 s por seed
→ factor ≈5.6× (23 entrenamientos ÓRBITA por seed vs 4 en dinlat: cross-fit 4 pliegues ×{LIBRE,SESGO}, HIB_RES,
y lo mismo en C1/C2). La fase run de dinlat DEV tomó ≈20 min (texts.emb 15:01 → último res 15:22, 27-oct, CST)
⇒ **DEV estimado ≈ 1 h 50 min de reloj (rango 1.5–2.5 h)**, 8 procesos en paralelo, sin Gemma (embeddings cacheados).
CONFIRM: similar. Es una extrapolación, no una medición sobre los datos reales.
