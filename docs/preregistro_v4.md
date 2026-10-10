# Pre-registro v4 — umbrales, seeds y lock (antes de DEV/CONFIRM)

**Rama:** `exp/field-autonomy-v4` · **Fecha:** 2026-10-09 · Escrito y commiteado **antes** de ejecutar seeds DEV (`0xA400–0xA40F`) o CONFIRM (`0xB400–0xB40F`).

## 1. Seeds

| Rol | Seeds | Uso |
|---|---|---|
| SMOKE | `0x5A00` | depuración del pipeline y calibración del lock de **A** (no evidencia) |
| DEV | `0xA400–0xA40F` (16) | primera corrida científica; no se retunea después |
| CONFIRM | `0xB400–0xB40F` (16) | held-out; mismo binario/lock que DEV |

Seeds nuevas: no se reutilizan `0xA300/0xB300` de v3 (salvo el diagnóstico E23 de v3, que por definición usa el modelo v3 y sus seeds DEV `0xA300–0xA30F`).

## 2. Divulgación de la calibración en SMOKE

En la seed SMOKE `0x5A00` se observó que un Dφ que procesa el objeto entero (12-D) memorizaba TRAIN (rel_err TEST 0.59 > STATIC 0.41). Se cambió a Dφ **de campo puntual** (el mismo campo vectorial `F_θ(q,c)` actúa sobre cada punto 2-D; pesos compartidos), y se fijaron `steps=8000`, `train_per_family=24`, mirando **sólo** la accuracy TEST del control A en SMOKE. La corrida SMOKE completa (que también imprime B) se ejecutó una vez para verificar el pipeline; tras verla no se cambió ningún umbral ni hiperparámetro. Esto se declara para transparencia.

## 3. Lock (hash en `config.json` → `hp_lock_sha256`)

`k_points=6, hidden=64, Adam lr=3e-3, batch=16, steps=8000, train/dev/test por familia = 24/12/32, REPLAY_FRACTION=0.5, MARGIN=0.05, REL_ERR_OK=0.05`.

- Correcto = ‖ŷ−y‖/‖y‖ < 0.05. Métrica primaria: accuracy TEST. Secundarias: rel_err, cosine de desplazamiento, energía, manifold distance.
- TEST: 50 % objetos nuevos in-dist, 50 % parámetros en el hueco de interpolación `p1∈(0.10,0.45)` jamás muestreado en TRAIN/DEV.
- Encoder = identidad (periferia congelada); decoder = identidad (independiente, sin parámetros ⇒ `decoder_lookup` imposible).

## 4. Gates por seed (código: `src/field_autonomy_v4.rs`)

| Exp | PASS por seed |
|---|---|
| E31 | Dφ_A supera por ≥0.05 accuracy al **máximo** de {STATIC, LINEAR, RANDOM_DYN, SHUFFLED_LABEL, NN_LOOKUP} en ≥3 familias, leakage=0 |
| E32 | mediana ‖σ_max(J_Dφ) − σ_max(J_true)‖/σ_true ≤ 0.10 en ≥3 familias (Jacobiano por diferencias finitas centrales, power-iteration). Diagnóstico por paso h1..h64 en `e32_trace.csv` |
| E33 | región de estabilidad (máximo H contiguo desde h1 con accuracy ≥0.5 en familias iterables rotate+reflect, free-run sin teacher forcing) ≥ 8 |
| E34 | Dφ(h1) ≥ MLP_DIRECT(h1)+0.05 **y** Dφ(h4 iter) ≥ MLP_DIRECT(h4 iter) **y** Dφ(h1) ≥ STATIC+0.05; mismos params (962), datos, seed, optimizer y pasos |
| E35 | acc(B) ≥ acc(A)+0.05 **y** cdt=rqm=table=nn=attractor=direct=0 en TEST **y** leaked=0 **y** target_seen_cdt=false **y** target_equivalent_seen=false |
| E36 | C(consolidado) ≥ B_raw+0.05 **y** C ≥ max(D corrupto, E irrelevante, F stats-sin-topología, G orden barajado)+0.05 |
| E37 | proceso nuevo cargando sólo el checkpoint Dφ_B: |post−pre| ≤ 0.01, post ≥ A+0.05, queries=0 |
| E38 | P0 (restart, retrieval OFF) ≥ A+0.05 en TEST y TEST' nuevo; P1 (CDT cargado, no consultado) idéntico a P0 con queries=0; P3 (lookup CDT) se reporta como control positivo de lookup |
| E39 | TRAIN círculos → TEST triángulos: B ≥ A+0.05, B > STATIC, sin queries |
| E40 | secuencia rotate→shear→affine→compose: final(B) ≥ final(A)+0.05, olvido(B) ≤ olvido(A)−0.05, final(B) ≥ final(D corrupto)+0.05 |
| E41 | intervención estructurada (swap columnas one-hot translate↔rotate) redirige ≥ 0.5×acc_translate_base; control aleatorio de igual ‖Δ‖_F redirige ≤ 0.10; rollback exacto (hash y accuracy) |
| E42 | N = 8/16/32/64/128 (k=4..64 puntos): B ≥ A+0.05 en **todas** las N, sin queries |
| E43/E44 | NOT_RUN en este ciclo (requieren periferia LLM congelada; no hay LLM en este harness) |

## 5. Gate de experimento

`PASS` si ≥ 9/16 seeds PASS (mayoría) en DEV **y** en CONFIRM; para E35 además IC bootstrap 95 % de mean(B−A) > 0 en ambas. `PARTIAL` si mayoría en una sola fase o 5–8/16. `FAIL` si ≤ 4/16. Cualquier fila `LEAKED` en condición científica ⇒ `LEAKED`.

## 6. Diseño de E35 (resumen)

`collect_experience(TRAIN) → CdtStore` (cada lectura cuenta `cdt_queries`) → `consolidate` → `LearningSignal` (por familia: regla afín puntual con coeficientes cuadráticos en p, 36 números + estadísticos de entrada/parámetros; **ningún par x→y**) → `adapt_dynamics` (mismo presupuesto que A: 8000×16 muestras; 50 % TRAIN, 50 % replay generativo de estados sintéticos nuevos) → `drop(signal, cdt)` → verificación hash TEST → evaluación con contadores thread-local por predicción. Todos los targets de replay se auditan contra TEST (hash canónico y bola de corrección 5 %). A y B comparten init (`seed^0xD0`) y stream de muestreo (`seed^0x7A`).

Hipótesis rival H0e (regularización genérica) se prueba en E36 (D/E/F/G de igual tamaño). La varianza por stream de muestreo se estima con B_raw (replay crudo = misma distribución que A).

## 7. E45 (añadido antes de ejecutarlo)

Gate por seed: max(acc L1, acc L2) ≥ acc T1 − 0.05, queries=0, leaked=0. Experimento PASS si ≥9/16 en DEV y en CONFIRM. Dataset: `train_per_family=12` (96 experiencias), resto igual. Presupuesto por fase = 4000×16 muestras para L1/L2/H. Umbral de gating = REL_ERR_OK (0.05). Rango UVᵀ = 4.

## 8. Seguimiento 1 — consolidación genérica (pre-registrado antes de correr DEV/CONFIRM)

- `consolidate_generic`: kernel ridge con random Fourier features sobre `u=[q, contexto]` → Δq, compartido entre puntos; ningún conocimiento de estructura afín/rotación; replay con entradas gaussianas independientes por punto (media/std por familia) y parámetros uniformes en el rango visto.
- Hiperparámetros elegidos en seeds de tuning **nuevas** `0xC400–0xC407`, sólo sobre la partición DEV, sólo con la accuracy del teacher (`artifacts/v4/generic/tuning_round*.txt`): **D=1600, ℓ=1.4, λ=1e-8·n**. Congelados.
- Ablaciones genéricas: D corrupto = contexto de familia desplazado (+1); F sin topología = RFF sin q (sólo contexto); E/G como antes.
- Re-corrida E35/E36 y E45 con `--consol generic`, mismas seeds DEV/CONFIRM, mismos gates (§4, §7).
- Corrección E45 (aplica a rule y generic): en la fase 2, H consolidaba con episodios de fase 1 ya "borrados"; ahora fase 2 = episodios fase 2 + pseudo-experiencias generadas desde la señal de fase 1 (igual para T1). E45 rule se re-corre y sustituye a la versión anterior.
