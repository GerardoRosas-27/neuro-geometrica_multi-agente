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

## 9. Seguimiento 2 — estabilidad E33/E23 (pre-registrado antes de correr)

- `src/v4_stability.rs` (E33S): Dφ entrenado igual que E35-A (consolidación rule; lock sin cambios); sólo cambia el operador de rollout: V0 baseline (ya residual), V1 re-proyección cada paso a la órbita de similitud(+reflexión) de x0, V2 límite espectral δ/max(1,σ_max(J)), V3 V1+V2. Gate por variante = gate E33 (región de estabilidad ≥8 en rotate+reflect), mayoría en DEV y CONFIRM.
- "Residual form": Dφ v4 ya es residual (V0). "Non-saturating encoder": no aplica en v4 (encoder identidad); en v3 requiere re-entrenar encoder/Dφ → NOT_RUN en este ciclo.
- E23 v3.7 (diagnóstico, seeds 0xA300–0xA30F, LONG lock): V0, V1 (re-proyección al manifold del encoder: punto de grilla más cercano + pattern search), V2 límite espectral, V3 ambos; se reporta cos por horizonte, sin gate.
- Advertencia pre-declarada: V1 inyecta el prior "el objeto se mueve en una órbita de similitud", que es exacto para rotate/reflect; un PASS de V1 no es evidencia de estabilidad intrínseca de Dφ.

## 10. Seguimiento 3 — completar E45 (pre-registrado antes de correr)

Mismo diseño E45 (96 experiencias, 2 fases, borrado, auditoría), corrido con consolidación rule y genérica. Brains nuevos:
- **L2g**: base = L1 tras fase 1; un adaptador UVᵀ (rango 4) por familia seleccionado por el one-hot del contexto; presupuesto fase 2 repartido por familia (total = L2).
- **L3**: paisaje energético cuadrático E(q,c)=½qᵀS(c)q+b(c)·q, S simétrica, S y b lineales en [c,1] (55 números); consolidación = mínimos cuadrados; fase 2 con pseudo-experiencias generadas por su propio modelo de fase 1; inferencia = un paso de gradiente. Limitación declarada: no puede representar flujos antisimétricos (rotación).
Gates (por seed; experimento PASS si ≥9/16 en DEV y CONFIRM):
- **E45b**: max(L1, L2, L2g, L3) ≥ T1 − 0.05, queries/leakage = 0.
- **E45c composición**: aplicar rotate(p1,0) y luego scale(p2,p2) (dos dinámicas aprendidas) para predecir compose(p) en TEST; PASS si el mejor cerebro líquido (L1, L2g, L3, H) encadenado ≥ STATIC + 0.05. Se reporta también la predicción directa con contexto compose.
- **E45a iteración adaptativa**: para Dφ (L1, H) en translate/rotate/scale, iterar n=2..8 sub-pasos con parámetros p/n y parar cuando el cambio relativo < 1 %; PASS si max(ganancia de accuracy adaptativa vs n=1) ≥ 0.05. Se reporta n medio por tercil de dificultad |p|.

## 11. Seguimiento 4 — E43/E44 con periferia LLM congelada (pre-registrado antes de correr)

- Periferia A: `gemma-2-2b-it-Q3_K_L.gguf` (loader existente `native_gemma2`, denso, media del hidden final de la secuencia). Periferia B (E44): `qwen2.5-1.5b-instruct-q4_k_m.gguf` (loader de candle vendorizado `v4_qwen2_hidden`, tokenizer.json oficial de Qwen2.5). Control R: encoder aleatorio congelado (trigramas hash → proyección gaussiana). Embeddings cacheados una vez (`artifacts/v4/lang/emb_*.json`, sha256 en manifest).
- Tarea: relaciones plural y femenino (16 + 14 conceptos; es/en/fr/ja; ja-plural sólo con reduplicación `々`, 7 conceptos). Por seed: 70 % de los conceptos españoles = TRAIN, 30 % held-out. PCA (d=16) ajustada sólo con palabras españolas de TRAIN; Dφ = MLP residual sobre el vector PCA, contexto = relación; 3000 pasos Adam. Decoder = vecino más cercano (coseno) entre todas las formas del idioma evaluado (vocabulario cerrado → se reporta `decoder_lookup`, no es memoria de la tarea).
- Métrica principal: accuracy top-1 en pares de **conceptos novedosos** (no presentes en TRAIN en ningún idioma). Controles: STATIC (identidad), OFFSET (desplazamiento medio por relación — control lineal), Dφ con encoder R.
- **Gate E43** por seed: acc Dφ_A ≥ max(STATIC, Dφ_R) + 0.05 en ≥2 de {en, fr, ja}. Experimento PASS si ≥9/16 en DEV y CONFIRM.
- **Gate E44** por seed: Dφ entrenado con A; en test la entrada pasa por B (o R) alineado al espacio PCA de A con ridge sobre 96 palabras ancla (disjuntas de la tarea, λ=1e-3·traza/n). PASS si acc_B ≥ acc_R + 0.05 y acc_B ≥ 0.5·acc_A (pares novedosos, todos los idiomas). Experimento PASS si ≥9/16 en DEV y CONFIRM.
- Seeds DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F (varían split y init). Sin tuning.

## 12. Plan Dφ/teacher distillation — Fase A: techo de destilación (pre-registrado antes de correr)

Plan: `docs/plan_v4_dphi_teacher_distillation.md` (commit `7a996b4`). Código: `src/v4_distill.rs`, `--exps DIST_A`.

- Dataset, seeds (DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F`), presupuesto (8000×16, Adam 3e-3), init (`seed^0xD0`), Dφ-962 y teacher (`consolidate(Full)`, regla) idénticos al protocolo v4. Sin cambio de arquitectura de inferencia.
- Condiciones: `A_train_only` (Dφ sólo TRAIN), `B_consolidation_route` (ruta existente E35-B: 50 % TRAIN real + 50 % replay del teacher), `D1_teacher_distill` (Dφ aprende **sólo** la salida del teacher: 50 % estados de replay generativo + 50 % entradas TRAIN reetiquetadas por el teacher; nunca ve el ground truth), `M1_mlp_direct_distill` (control MLP no residual, misma capacidad 962, misma destilación), `D4_teacher_distill_4xsteps` (diagnóstico de optimización: 4× pasos, no gate), `T_teacher_direct` (teacher como predictor, post-hoc, lado auditor).
- Todas las etiquetas del teacher pasan por el auditor (hash canónico + bola 5 % contra TEST). Teacher y CDT se destruyen antes de TEST. Diagnóstico teacher→modelo (rel. err.) y σ_max(J) mediano se calculan después de TEST con un teacher re-consolidado del lado del auditor.
- D1 se guarda como checkpoint y se re-evalúa en **proceso nuevo** (`eval-checkpoint P0`).
- **Gate por seed:** acc(D1) ≥ acc(B) + 0.05, leakage=0 y queries=0 en todas las condiciones, y proceso nuevo bit-exacto (|Δ|<1e-12, queries=0).
- **Gate de fase (plan §4):** media pareada D1−B > 0 con IC95 bootstrap (4000 remuestreos, semilla 0xB007) con límite inferior > 0 **en DEV y en CONFIRM**, más leakage=0, queries=0 y proceso nuevo OK en 32/32. Objetivo fuerte (no gate): D1 ≥ 0.70.
- Lectura (plan §5) pre-fijada: (i) D1 alto (≥0.70) y B bajo → el cuello es la consolidación/señal; (ii) D1 ≈ B y lejos del teacher con teacher→Dφ alto → cuello de representación/capacidad → la Fase B (barrido de capacidad) es exactamente la prueba que el plan prescribe, así que se ejecuta aunque el gate A falle **sólo** en este caso, documentándolo; (iii) cualquier leakage/queries>0 o fallo de proceso nuevo → STOP.
- DEV primero, luego CONFIRM sin retune. Smoke (`0x5A00`) sólo para comprobar que corre.

## 13. Fase B — barrido de capacidad (pre-registrado antes de correr; después del FAIL caso ii de la Fase A)

- `--exps DIST_B`. Misma parametrización residual `Ψ' = Ψ + ΔΨ`; ancho oculto H ∈ {64, 136, 272, 544} → 962 / 2042 / 4082 / 8162 parámetros. Ruta de entrenamiento: D1 (destilación directa del teacher, Fase A). Mismos dataset, seeds, init seed, presupuesto (8000×16, Adam 3e-3) y número de pasos para todas las H. Control: MLP directo (no residual) con la misma H en cada punto. Referencias: B ruta de consolidación H64 y teacher directo.
- Reporte: parámetros, wall-clock de entrenamiento, memoria (params×8 B×4: pesos+grad+Adam m,v), rel_err₁, acc, σ_max(J) mediano, teacher→modelo.
- **Gate de fase:** H* = la H más pequeña > 64 cuya diferencia pareada acc(H) − acc(H64) tenga media ≥ +0.05 e IC95 inferior > 0 en DEV; CONFIRM debe reproducir el mismo criterio **para la misma H*** (sin re-elegir). Leakage=0 y queries=0 en todas las condiciones. Si ninguna H lo cumple → FAIL.
- Separación capacidad vs consolidación vs estabilidad: este barrido sólo varía capacidad (la ruta de consolidación queda fija y la estabilidad sólo se mide).
- Paso siguiente pre-fijado: la Fase C se aplica "a la mejor capacidad" (plan §2). Si B pasa, H*; si B falla, H=64 (no hay mejor capacidad demostrada), documentándolo. Una mejora de Dφ que también obtenga el MLP de igual H no se atribuye a la dinámica.

## 14. Fase C — optimización explícita de un paso (pre-registrado antes de correr; tras FAIL de la Fase B)

- `--exps DIST_C`, H = 64 (Fase B no demostró una capacidad mejor; `V4_DPHI_H` sin fijar → 64, registrado en las notas).
- Objetivo de entrenamiento: error de un paso frente a teacher/operador verdadero = etiquetas de la ruta de consolidación (50 % TRAIN real, 50 % replay del teacher, auditado). Receta única fijada a priori, sin búsqueda: 32000 pasos, batch 32, Adam lr 3e-3 con decaimiento coseno a 3e-5, pérdida relativa por ejemplo ‖ŷ−y‖²/‖y‖² (coincide con la métrica). Cómputo 8× el lock (se reporta).
- Condiciones: `C0_lock_budget` (referencia, ruta de consolidación con el lock), `C1_onestep_recipe` (Dφ), `M1_mlp_onestep_recipe` (control MLP igual capacidad y receta).
- Métricas en TEST por familia y por región (`in_dist` / `interp_hole`): rel_err₁, error absoluto, coseno del desplazamiento con el target, distancia al manifold, σ_max(J) mediano, error teacher→Dφ (post-hoc). C1 re-evaluado en proceso nuevo.
- **Gate (meta del plan):** media sobre seeds de rel_err₁ (TEST, recortado a 10) de C1 < 0.05 en DEV **y** CONFIRM, sin lookup ni memoria (leakage=0, queries=0, proceso nuevo bit-exacto). Gate por seed: rel_err₁(C1) < 0.05 + limpio.
- Si falla: STOP antes de la Fase D (el plan exige cumplir la meta de un paso antes de estabilidad/reproyección/E33) y por tanto antes de Ψ64/Gemma E43/E44.

## 15. Fase D — estabilidad y horizonte largo / E33 (pre-registrado antes de correr; tras PASS de la Fase C)

- `--exps DIST_D`. Modelo: C1 de la Fase C (H=64, misma receta y seeds; re-entrenado de forma determinista en el proceso). Protocolo E33 sin cambios: familias iterables rotate+reflect de TEST, free-run sin teacher forcing, horizontes 1/2/4/8/16/32/64, correcto = rel_err<5 % contra la regla aplicada h veces.
- Variantes (plan §2 Fase D): `V0_pure_C1` (Dφ puro), `V1_reproj_C1` (re-proyección por paso a la órbita de similitud(+espejo) de x0; usa sólo x0 y el estado, nunca el target), `V3_reproj_spectral_C1` (V1 + límite espectral δ/max(1,σ_max(J))). V3 se corre siempre pero sólo se interpreta si V1 sigue inestable. Controles: `V0_pure_M1_mlp` (MLP misma receta) y `V0_pure_C0_lock` (Dφ con el lock).
- Diagnóstico por horizonte: accuracy, rel_err, deriva de norma (mediana ‖z‖/‖y‖), energía, distancia al manifold, crecimiento de perturbaciones (δ₀ aleatorio fijo por seed, ‖δ₀‖=1 %·‖x0‖; mediana ‖Φʰ(x+δ₀)−Φʰ(x)‖/‖δ₀‖), σ_max(J) mediano a lo largo del rollout.
- **Gate por variante = gate E33:** región de estabilidad (máximo h contiguo desde h1 con acc ≥ 0.5) ≥ 8, limpio. Experimento: ≥ 9/16 seeds en DEV **y** CONFIRM (§5).
- Atribución: una mejora de V1/V3 sólo se atribuye a la re-proyección comparada con V0 en las mismas seeds; V1 inyecta el prior de órbita de similitud (exacto para rotate/reflect) → no es estabilidad intrínseca. La re-proyección es mecanismo de estabilización, no memoria.
- Núcleo Dφ "resuelto" (condición para pasar a Ψ64/Gemma E43/E44, plan §6): Fase C PASS **y** E33 PASS para V0 o V1 (reportando siempre V0). Si sólo falla E33 → plan §5: estudio manifold/proyección/Jacobiano, sin más epochs automáticos; STOP antes de E43/E44.

## 16. Estudio manifold/proyección/Jacobiano — sesgo por paso (pre-registrado antes de correr DEV/CONFIRM)

**Diagnóstico previo (no gate, divulgado):** `examples/v4_bias_diag.rs`, seeds de tuning nuevas `0xC500–0xC503`, evaluado **sólo en la partición DEV** (TEST no tocado). Logs: `artifacts/v4/bias_diag/round{1..5}_*.txt`. Hallazgos:
1. El sesgo de reflect/radial_warp viene de la **consolidación**: el teacher (regla afín con coeficientes cuadráticos en p) tiene error propio de 9–13 % en reflect (cos/sin(2a), |2a|≤2 no es cuadrático) y 6–10 % en radial_warp (no afín); el resto ≤2 %. C1 copia ese sesgo vía replay.
2. Paridad/orientación: la reflexión **sí** es representable por el campo residual (Δ = R q − q); el problema es la precisión de un MLP tanh para productos q·cos(2a). Una unidad multiplicativa genérica ("campo local-lineal con compuerta": Δ = M(q,c)·q + b(q,c), M y b salidas del mismo MLP; no conoce familias) baja el error en todas las familias.
3. Excluir del replay las familias cuyo residuo de ajuste de consolidación (in-sample sobre la CDT, calculado en consolidación) es ≥ τ=1 % ("replay de confianza") elimina el sesgo heredado de reflect (máscara resultante en las 4 seeds: reflect y radial_warp excluidas).
4. El error restante escala con la optimización: con el gated H136 + confianza, error medio DEV 1.7 % (1×) → 1.0 % (4×) → 0.85 % (8×); rotate 0.5 %, reflect ~1 %. Rollout DEV h8 iterable 0.12–0.38 (1×) → 0.50–0.71 (8×). La re-proyección suma poco (+0–0.1).
5. Entrenamiento multi-paso (BPTT/scheduled sampling): no hay objetivos verdaderos multi-paso sin consultar el generador; los del teacher heredan su sesgo (rotate 0.6 %/paso) → no se usa en este ciclo.

**Diseño `--exps DIST_E`** (mismas seeds DEV/CONFIRM, receta one-step de la Fase C: batch 32, Adam 3e-3→3e-5 coseno, pérdida relativa; constantes fijadas: H=136, multiplicador de pasos 8 → 256 000 pasos, τ=0.01):
- `E0_faseC_H64` (referencia Fase C, 1×), `E1_plain_trusted` (Dφ tanh normal H136 + replay de confianza; aísla la parametrización), `E2_gated_consol` (gated H136 + replay completo; aísla la consolidación de confianza), **`E3_gated_trusted`** (candidato principal), `M3_mlp_gated_trusted` (control MLP: misma cabeza gated, no residual, misma H, receta y datos). Capacidad fija en H136 en E1–E3/M3 (la separación capacidad/consolidación/estabilidad se hace con E1 vs E3 = parametrización, E2 vs E3 = consolidación, V0 vs V1 = estabilización).
- Por familia y región: rel_err₁, error absoluto, coseno, manifold, σ_max(J), teacher→modelo. E3 re-evaluado en proceso nuevo. Auditor sobre todas las etiquetas de replay.
- E33 (protocolo sin cambios: rotate+reflect TEST, free-run, h1…h64, correcto <5 %) para E0, E1, E2, E3 puro (V0), E3 + re-proyección (V1) y M3; reporta crecimiento de perturbaciones, deriva de norma y acc por familia.
- **Gate principal:** E33 `R_E3_V0_pure` región ≥ 8 en ≥ 9/16 seeds en DEV **y** CONFIRM, leakage=0, queries=0, proceso nuevo bit-exacto. V1 se reporta aparte (sólo atribuible a re-proyección contra V0). Meta secundaria (no gate): error medio por paso ≈0.5 %.
- Si E3 V0 (o V1, declarado como estabilizado con prior) pasa en ambas fases → el núcleo Dφ se considera resuelto (§15) y se procede a Ψ64/Gemma E43/E44 con pre-registro propio. Si no → STOP y se documenta. Si M3 iguala a E3, la mejora no se atribuye a la dinámica residual.

## 18. E43/E44 con un Ψ64 NUEVO entrenado sobre Gemma y congelado (opción (a), pre-registrado antes de DEV/CONFIRM)

Código: `training/v4_psi64/e43_psi64.py` (JAX, venv token-free), splits/pares/anclas/encoder aleatorio exportados del harness Rust (`examples/v4_lang_export.rs` → `artifacts/v4/psi64/lang_task.json`, `emb_random512.json`; mismo split por seed que §11 vía `v4_language::train_concepts`). Embeddings: caches de §11 (Gemma-2-2B-it Q3_K_L, pooling medio del hidden final, 2304-d; Qwen2.5-1.5B 1536-d), sha256 en `config.json`.

- **Encoder Ψ64 (receta E45):** MLP gelu 2304→256→64 (W2 init /16), entradas estandarizadas con media/desv. de las palabras TRAIN. Pérdida E45 adaptada: CE de decodificación (concepto TRAIN + forma base/derivada) sobre Ψ(src), Ψ(dst) y sobre la predicción lineal Ψ(src)·A_rel, + 0.1·MSE(Ψ(src)·A_rel, sg Ψ(dst)). Adam 2e-3, 1500 pasos full-batch. **Entrenado sólo con las palabras de los pares españoles de conceptos TRAIN** (≈42 palabras); después se congela y se registran sha256 de los pesos y estadísticas de entrenamiento (`encoder_manifest.jsonl`). Leakage = nº de palabras de conceptos novedosos presentes en el set de entrenamiento (debe ser 0). El control R usa el **mismo** procedimiento sobre el encoder aleatorio de trigramas (512-d).
- **Dφ (receta §16 E3 adaptada a un vector 64-d):** z = Ψ/escala (RMS de Ψ en TRAIN), contexto = relación one-hot; cabeza gated local-lineal **diagonal** (Δ = m(z,c)⊙z + b(z,c); una M completa 64×64 tendría ~575k parámetros con 21 pares), MLP tanh H=136; replay de confianza: teacher = mapa lineal ridge por relación (λ=1e-3·n) en el espacio z, de confianza si su residuo relativo leave-one-out < τ=0.01, replay 50 % con estados gaussianos genéricos (media/desv. de z TRAIN). Pasos 24000 (8× los 3000 de §11), batch 8, Adam coseno 3e-3→3e-5, pérdida relativa, weight decay 1e-4. Controles: MLP (misma cabeza gated, no residual), Dφ plain (residual tanh normal, misma receta), OFFSET lineal, STATIC, y R = encoder aleatorio + su propio Ψ64 + Dφ E3.
- Decoder de vocabulario cerrado (coseno, como §11; `decoder_lookup` = vocabulario cerrado, no memoria de tarea). Sin stores: queries = 0.
- **Gates (iguales que §11):** E43 por seed: acc E3(Gemma-Ψ64) ≥ max(STATIC, E3(R-Ψ64)) + 0.05 en ≥2 de {en, fr, ja}. E44 por seed: Dφ entrenado sobre Gemma-Ψ64; en test la entrada viene de Qwen (o R) alineada con ridge dual sobre las 96 anclas (λ=1e-3·traza/n) al espacio z de Gemma-Ψ64; PASS si pooled_qwen ≥ pooled_R + 0.05 y ≥ 0.5·pooled_gemma. Ambos requieren además leakage=0 y re-evaluación en **proceso nuevo** (checkpoint npz recargado por otro proceso python) idéntica. Experimento PASS si ≥9/16 en DEV **y** CONFIRM.
- Smoke `0x5A00` corrido antes sólo para comprobar el pipeline (E43 wins 1/3, E44 FAIL, teacher LOO ≈0.7 → sin replay); sin cambios de constantes después.
