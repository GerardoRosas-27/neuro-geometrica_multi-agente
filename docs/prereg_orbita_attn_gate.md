# Prerregistro: ÓRBITA + corrector de atención gateado por sorpresa (`exp/orbita-attn-gate`)

Rama creada desde `exp/orbita-tf` (`dcaadad`). Experimento aislado; no toca v4/v5/run_v6 ni otras ramas. Este documento y el código
(`orbita/attn_gate/`) se empujan **antes** de cualquier ejecución. Cambios posteriores sólo como enmiendas fechadas (§5).

## Motivación
En `exp/orbita-tf` (CONFIRM, k = 2, familias nuevas 8–11) el Transformer de estados con inferencia por error de predicción (TF-PE) obtuvo
acc16 0.0544 frente a 0.0339 de ÓRBITA-LAT_EP (hallazgo NO prerregistrado), pero en familias conocidas TF-PE (0.3721) queda por debajo de
LAT_EP (0.4481). Idea: dejar a ÓRBITA-v1 congelado como predictor y añadir un corrector residual de atención entrenado por error de
predicción, que sólo actúe cuando la sorpresa de ÓRBITA es alta (gate «do-no-harm»).

## 1. Brazos (c desconocido; k ∈ {2,4,8} observaciones; primario k = 2; métrica acc16 = rel. err. < 0.05 en t = 16)
- **ORBITA**: ÓRBITA-v1 congelado (pesos publicados `artifacts/orbita/{dev,confirm}/weights_<seed>.json`), re-identificación OBS sobre 8 familias ×
  ({p̂} ∪ malla 9×9) con las k observaciones y free-run hasta t = 16. Reimplementado en PyTorch (`orb.py`); se exige igualdad episodio a episodio
  con `obs16` de `orbita lat-eval` (Rust). Sorpresa s_known = Σ_{t≤k} min(rel. err., 10)/k del mejor candidato (= `s_known` de Rust).
- **ALWAYS**: x_{t+1} = f_ÓRBITA(x_t) + Δ_corr(x_t, memoria), sin gate.
- **GATE** (principal): ALWAYS si s_known > τ_k, si no ORBITA.
- **LAT_EP**: adaptación latente por episodio de ÓRBITA (`lat-eval exp1`, τ = 0.014656/0.022020/0.037142, λ0 = 1e-4), recalculada aquí.
- **TF-PE**: valores por seed del artefacto `artifacts/orbita/transformer_fe/per_seed_{dev,confirm}.json` (`dcaadad`); mismos episodios y seeds
  (se regeneran con el mismo RNG y se verifica que LAT_EP coincide con el artefacto por seed). No se reentrena.

## 2. Corrector (`orb.py: Corrector`, ~3.0 k parámetros, float64)
Consultas = 6 puntos del estado actual [x_j, f(x)_j]; memoria = un token por transición observada del episodio y punto
[y_{t−1,i}, f(y_{t−1})_i, r_{t,i}], r = y_t − f(y_{t−1}) (error de predicción de ÓRBITA). Atención 2 cabezas (d = 16) con sesgo de distancia
−softplus(w_h)·‖x_j − y_{t−1,i}‖², FFN 48, salida residual Δ_j (init ×0.1).
**Entrenamiento** (`train_corr.py`, una red por seed, ÓRBITA congelado): episodios propios de familias conocidas 0–7 (`gen_eps.py train`,
RNG = seed ⊕ 0x5A17, 64 + 8 de validación por familia; sin ds.test, sin familias 8–11). Por batch: k ∈ {2,4,8} uniforme; base de ÓRBITA = reid
con k obs. o, con prob. 0.5, reid **excluyendo la familia verdadera** (simula familia nueva). Pérdida = error de predicción relativo²
(1 paso teacher-forced + free-run k+1..16, recortado a 10). Adam, 20 000 pasos, batch 32, warmup 500, coseno 1e-3 → 1e-5, clip 1.0.

## 3. Calibración del gate (sólo DEV)
τ_k = percentil 95 de s_known en episodios **TUNE** (RNG 0xE7F0, disjuntos de DEV/CONFIRM) de familias **conocidas** 0–7, agrupando las 16 seeds
DEV. No usa familias nuevas ni episodios DEV/CONFIRM. τ queda fijo para DEV y CONFIRM (`tau.json`). Secundario: q = 0.90 y 0.99.
Nota: s_known (error del mejor ajuste dentro de las 8 familias conocidas) es en sí la distancia a las familias conocidas; no se añade otro gate.

## 4. Protocolo
16 seeds por fase (DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F), 768 episodios por seed (64 × familias 0–11; 8–11 nuevas), mismos conjuntos
que `exp/orbita-tf`. 0 queries a TEST. IC95 bootstrap pareado por seed (10 000 remuestreos, `default_rng(12345)`), δ = 0.05.
Sin cambios de código ni hiperparámetros entre DEV y CONFIRM (salvo §5). Una sola cadena `chain.sh` (flock, setsid nohup, reanudable con
`.done`), sin Gemma (todo es de estados).

## 5. Convergencia (obligatoria antes de CONFIRM)
Por seed DEV: C1 val final ≤ 1.05 × val mínima; C2 |Δ| relativa de la pérdida de entrenamiento media entre los dos últimos 10 % < 5 %;
C3 val final < 0.9 × val inicial; C4 sin pasos no finitos. Si alguna seed falla, la cadena se detiene antes de CONFIRM; se corrige la
optimización (p. ej. más pasos o lr), se documenta como enmienda fechada, se repite DEV completo y sólo entonces CONFIRM. Se publican las
curvas (`artifacts/orbita/attn_gate/curve_*.csv`).

## 6. Hipótesis (k = 2; A − B media por seed [IC95])
| id | cantidad | aceptada si (DEV y CONFIRM) | refutada si (CONFIRM) |
|---|---|---|---|
| H1 | GATE − LAT_EP, nuevas | inf. IC95 > 0 | sup. IC95 < 0 |
| H2 | GATE − ORBITA, conocidas (no inferioridad) | inf. IC95 > −δ | sup. IC95 < −δ |
| H3 | GATE − ALWAYS, conocidas | inf. IC95 > 0 | sup. IC95 < 0 |
| H4 | GATE − TF-PE, conocidas | inf. IC95 > 0 | sup. IC95 < 0 |
Resto: inconclusa. Secundarios (descriptivos, k = 2/4/8): GATE − ORBITA nuevas; ALWAYS − LAT_EP nuevas; GATE − TF-PE nuevas;
GATE − LAT_EP conocidas; ALWAYS − ORBITA nuevas/conocidas; tasas de disparo del gate; q = 0.90/0.99.
Predicción honesta: H2 y H3 probables por construcción; H4 dudosa (TF-PE conocidas 0.37 vs ORBITA-OBS 0.18 en `exp/orbita-tf`); H1 incierta.
