# ÓRBITA-Nova-2 · Experimento 3 — BOCD en lugar del umbral fijo de novedad τ (preregistro)

Rama `exp/orbita-nova2` (desde `6cc6d8a`). Aprobado por Bruster (Nova-2 exp3; propuesta 3 de la página «papers de vanguardia y propuestas»: BONE arXiv:2411.10153 / BOCD, Adams & MacKay 2007, arXiv:0710.3742). Fecha: 2026-10-24 (America/Mexico_City, UTC−6).
Este documento se congela con su commit **antes** de generar cualquier dato (calibración incluida). No modifica PREREG_EXP1/EXP2 ni PREREG de Nova.

Regla dura: entre DEV y CONFIRM no se cambia código, hiperparámetros, umbrales, familias, métricas ni criterios. La única calibración (§2.3) usa sólo familias 1–2 de las seeds DEV y se congela antes de DEV.

## 0. Base

Se conservan los arreglos del exp1 (`DIAGNOSTICO_HARNESS.md`): **C3 activo** en NOVA* (filtro en k pasos + free-run 16−k) y **LAT_EP con τ_LAT(k)** (0,0146557 / 0,0220203 / 0,0371425; λ0=1e-4). τ_C6 = 0,9386641928884383 (sigue usándose para recuperación/clustering de la biblioteca C6, que **no** es un detector de novedad y no se toca en este experimento). Núcleo ÓRBITA congelado. Flujo, held-out, biblioteca y consolidación de NOVA idénticos al exp2.

## 1. Qué se sustituye

Hoy la decisión «¿dinámica conocida o nueva?» es un umbral fijo calibrado una vez: LAT_EP adapta sólo si `s_known(k) > τ_LAT(k)`. Se sustituye por la probabilidad posterior de cambio de régimen de un BOCD sobre la sorpresa por paso, con umbral de decisión 0,5 fijado a priori (decisión bayesiana con coste simétrico), sin calibración por k.

## 2. Detector BOCD (definición congelada)

### 2.1 Observación
Para cada paso observado t = 1..k del episodio: x_t = ln(max(rel_err(ÓRBITA.step(y_{t−1}, c), y_t), 1e-12)) con y_0 = x0, contexto c de `ctx_for_eval` (idéntico al de s_known), rel_err acotado a 10 y 10 si no finito. Es la sorpresa one-step de ÓRBITA (k observaciones por episodio; nunca se usan pasos > k).

### 2.2 Modelo
BOCD (Adams & MacKay) con verosimilitud Normal de media y precisión desconocidas, conjugada Normal-Gamma (predictiva t de Student). Hazard constante **h = 1/16** (a priori: un cambio esperado por horizonte de 16 pasos).
Al inicio de cada episodio el run-length está, con probabilidad 1, en el **run de referencia** (régimen «conocido») con estadístico Normal-Gamma NG_ref. Se procesan las k observaciones con la recursión estándar de BOCD (estados = continuar ref, o un run nuevo empezado en el paso j ≤ t con prior NG_new). **p_cambio = 1 − P(run actual = ref | x_1..x_k)**. Decide «nuevo» ⇔ p_cambio > 0,5.

### 2.3 Priors (fijados así, y sólo así)
- NG_ref inicial: μ_ref = media y s²_ref = varianza de x_t (pasos 1..8) sobre los episodios de calibración de **familias 1–2** (`gen_calib_h(seed, 16)`, 32 episodios/seed) de las **16 seeds DEV** con pesos DEV. Fuerza equivalente n0 = 16: κ = 16, α = 8, β = 8·s²_ref. Se calcula una vez con `orbita nova2-exp3-calib`, se escribe en `artifacts/orbita/nova2_exp3/calib.json` y se copia como constante en el código **antes** de DEV (commit de implementación). Se usa igual en CONFIRM.
- NG_new (run nuevo, poco informativo): μ = μ_ref, κ = 0,1, α = 1, β = s²_ref.
- h = 1/16, umbral 0,5: a priori, no calibrados.

### 2.4 Aprendizaje en línea (sin etiquetas)
Durante el flujo (n=2400), tras cada episodio, si el detector lo decide «conocido» (p_cambio ≤ 0,5) sus k observaciones actualizan NG_ref (actualización conjugada). Nunca se usan etiquetas de familia. En cada checkpoint, la evaluación del held-out usa NG_ref congelado en ese punto (no se actualiza con H).

## 3. Brazos

Controles (como exp2): **ÓRBITA**, **ÓRBITA+C3**, **LAT_EP** (τ_LAT(k)), **NOVA** (C3 activo), NOVA−C3.
Nuevos:
- **LAT_BOCD**: LAT_EP con la decisión `s_known ≤ τ_LAT(k)` sustituida por `p_cambio ≤ 0,5` (BOCD §2). Todo lo demás idéntico (mismo ajuste LAT con λ0).
- **NOVA_BOCD**: NOVA (misma biblioteca C6 compartida, C4/C5/C3 idénticos) pero C4/C5 sólo se disparan si p_cambio > 0,5; si no, ÓRBITA+C3 (C7).
- **LAT_BOCD_SHUF** (control nulo): mismas decisiones de LAT_BOCD permutadas al azar entre los episodios del held-out (por checkpoint y k; RNG Xoshiro con semilla seed ⊕ 0xB0CD ⊕ n ⊕ k). Conserva la tasa de «nuevo» y destruye la asociación episodio-decisión.

## 4. Datos, semillas y métricas

Idénticos a exp1/exp2: `gen_stream` n=2400 (80 % conocidas 0–7 / 20 % nuevas {8,10,11}, k∼U{2,4,8}), H = 16 episodios/familia, checkpoints n ∈ {0,150,300,600,1200,2400}. Semillas: DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F`.
**Métrica primaria**: por seed, mediana de `log(rel_err h16)` sobre episodios H de nuevas {8,10,11}, n=2400; menor = mejor.
Secundarias: acc16 (rel_err<0,05) conocidas/nuevas; known_med_log; detección en H (positivo = familia ∈ {8,10,11}, negativo = 0–7): balanced accuracy (bacc) de BOCD (p_cambio>0,5) y de la regla τ_LAT(k) (s_known>τ_LAT), AUROC de p_cambio y de s_known; tasa de «nuevo» en conocidas (falsos positivos); n efectivo de NG_ref; curva n.

## 5. Hipótesis y criterios

Contraste = media sobre 16 seeds de la diferencia pareada por seed (n=2400); IC95 bootstrap pareado percentil, **10 000 réplicas, semilla 12345** (`scripts/analyze_nova2_exp3.py`). δ = 0,05.

- **F1 (primaria)**: LAT_BOCD − LAT_EP, nuevas, log, **k=2** < 0. **Aceptada** si IC95 sup < 0 en DEV **y** CONFIRM; **refutada** si en CONFIRM IC95 inf > 0; inconclusa en otro caso.
- **F2 (secundarias, no vetan)**: igual con k=4 y k=8; NOVA_BOCD − NOVA nuevas log k=2/4/8.
- **F3 (no-daño, veta)**: LAT_BOCD − LAT_EP y NOVA_BOCD − NOVA en conocidas 0–7, acc16, k=2/4/8: IC95 inf > −δ en ambas fases (las 12 comparaciones).
- **F4 (detección, no veta)**: bacc(BOCD) − bacc(τ_LAT), k=2/4/8; se reporta con IC95.
- **F5 (control nulo, no veta)**: LAT_BOCD − LAT_BOCD_SHUF nuevas log k=2 < 0 (si no, la ganancia no viene de la detección).
- Reportados: NOVA − LAT_EP, NOVA − ÓRBITA, NOVA − NOVA−C3, AUROC.

**Éxito del exp3** ⇔ F1 aceptada ∧ F3 aceptada. `RESULTADOS_EXP3.md` se publica en cualquier caso (negativos incluidos).

Nota honesta a priori: LAT_EP ya decide bien en exp1/exp2 (NOVA−LAT_EP muy negativo en k=2 indica que LAT_EP es débil en nuevas por su ajuste, no necesariamente por su detector); F1 puede salir inconclusa aunque BOCD detecte igual o mejor. F4 mide la detección por separado.

## 6. Ejecución

Binarios `orbita nova2-exp3-calib` (una vez, DEV fam 1–2) y `orbita nova2-exp3`; scripts `orbita/scripts/nova2_exp3.sh` / `nova2_exp3_all.sh` (setsid nohup, `nice 10`, JOBS=7, reanudable por `.done` por seed). Salida `artifacts/orbita/nova2_exp3/{smoke,dev,confirm}/`. Orden: calib → smoke (no analizado) → DEV → CONFIRM, mismo binario y commit. Sin `run_v6`, sin tocar `main`, sin force push.
