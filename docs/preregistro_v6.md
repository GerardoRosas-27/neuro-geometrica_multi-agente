# Preregistro v6 — hipótesis H1–H7 sobre la arquitectura v4 (§16)

**Rama:** `exp/v6-hipotesis` (desde `main` @ `49c00a9`). **Fecha:** 2026-10-13. **Estado:** preregistrado, **sin ejecutar**. Este documento no contiene resultados; el smoke (200 pasos) no es evidencia.

**Fuentes:** hipótesis de `docs/hipotesis_math.md` (rama `exp/hipotesis-math`, commit `bb24118`); receta ganadora v4 §16 (`docs/resultados_v4.md`, `docs/preregistro_v4.md` en `exp/field-autonomy-v4`; código en `main`: `v4_distill::train_onestep_trusted_b`, brazo `E3_gated_trusted`). Arnés (caché por semilla, checkpoint intra-semilla determinista, evaluación en proceso nuevo) reutilizado del diseño de v5 (`exp/predictive-field-v5`).

**Código:** `src/v6_hypo.rs`, `src/train_ckpt.rs`, `src/bin/run_v6.rs`; lanzador `scripts/v6_launch.sh`, reanudación `scripts/v6_resume.sh`, análisis `scripts/v6_analyze.py`.

## 1. Base común (idéntica para todos los brazos)

| Elemento | Valor fijo |
|---|---|
| Arquitectura | Dφ gated local-lineal por punto, Δ(q,c) = M(q,c)·q + b(q,c), H = 136, 2590 parámetros en inferencia |
| Teacher | CDT → consolidación `Full` → residuo de ajuste por familia → máscara de confianza τ = 0.01 (`TRUST_TAU`) |
| Datos de entrenamiento | TRAIN (verdad) con prob. 0.5, replay del teacher con prob. 0.5 (`REPLAY_FRACTION`) |
| Pérdida principal | relativa ‖ŷ−y‖²/‖y‖² a 1 paso |
| Presupuesto (= v4 §16 E3) | 256 000 pasos Adam × batch 32, lr coseno 3e-3 → 3e-5 |
| Init / RNG | init `seed ^ 0xD0`; muestreador `seed ^ 0x7A` en **todos** los brazos de 1 paso (números aleatorios comunes → comparaciones pareadas). Ruido H5b y direcciones H7 usan un RNG auxiliar sin estado que no altera el flujo principal |
| Semillas | **DEV 0xC600–0xC60F**, **CONFIRM 0xD600–0xD60F** (16 + 16), smoke 0x5A60 |
| Evaluación | TEST sellado (`verify_test`), acc/rel_err a 1 paso, E33 libre (rotate+reflect) hasta h64, región contigua acc ≥ 0.5 |
| Auditoría memoria cero | `leakage = 0` (hits del auditor sobre targets del teacher + frames de episodios que coinciden con TEST) y `queries = 0` durante TEST; CDT y teacher se destruyen antes de TEST |
| Proceso nuevo | cada modelo se escribe a checkpoint y se evalúa con `run_v6 eval-checkpoint` en otro proceso; se exige acc idéntica y 0 consultas (`fresh_ok`) |
| Controles | **MLP gated** (mismo tamaño, `residual=false`): `E3_mlp`, `H4_sobolev_mlp`, `V6_all_mlp`; **shuffled** (targets de TRAIN permutados dentro de familia, teacher consolidado desde ese TRAIN): `E3_shuffled`, `V6_shuffled` |

Con todos los flags apagados, el bucle de v6 es **bit-idéntico** a `train_onestep_trusted_b` de `main` (test `baseline_bit_identical_to_v4`).

## 2. Hipótesis → cambio en código

Las constantes λ, ρ, ε, σ se fijan **a priori** aquí (no hay fase de tuning; `hipotesis_math.md` proponía tuning en 0xC500–0xC503, se descarta para no tocar datos antes de DEV). Si una hipótesis falla, una explicación posible es la elección de λ; eso se registrará como limitación y no se re-tuneará en CONFIRM.

### H1 — sesgo, no varianza (Gronwall lineal)
- **Cambio:** término `L_bias = λ₁‖E_batch[(Φθ(x)−y)/‖y‖]‖²`, λ₁ = 1.0 (`LAMBDA_H1`). Brazo `H1_bias`.
- **Predicción falsable:** baja el sesgo con signo por paso sobre la trayectoria verdadera (`bias_norm`, `bias_ang`, t = 1..16) y sube acc h16.
- **Métrica:** acc h16 (E33), rel h8, `bias_norm`/`bias_ang`.
- **Éxito:** diferencia pareada `H1_bias − E3_v4` en acc h16 con IC95 bootstrap > 0 en DEV **y** CONFIRM, y acc 1 paso no peor en > 0.02. **Fallo:** IC incluye 0 en cualquier fase, o sube el RMS (rel_err1) sin bajar |bias|.

### H2 — pérdida tipo PRM con prefijo confiable — **PENDIENTE del veredicto v4 vs v5**
- **Cambio:** episodios TRAIN de 8 pasos (misma construcción que v5, auditados), pérdida relativa en k ∈ {1,2,4,8}; en `H2_prm_eqc` el horizonte k cuenta solo si todos los pasos j < k tienen rel_err < 0.05 (`PRM_TAU = REL_ERR_OK`, etiqueta "paso correcto"). Referencia `H2_mh_eqc` = misma pérdida sin máscara (equivale a v5 `P2_gated_eqc`). Ambos con 256 000/8 = 32 000 pasos (mismas evaluaciones de Dφ que E3). `V6_all_H2`: mitad de evaluaciones con la receta V6_all a 1 paso (128 000 pasos) + mitad con PRM (16 000 pasos de horizonte).
- **Predicción:** `H2_prm_eqc` > `H2_mh_eqc` en acc h16 con cómputo igualado.
- **Éxito:** IC95 > 0 en región o acc h16 en ambas fases. **Fallo:** empate o negativo.
- **Estado:** se ejecuta y reporta, pero **no entra en el veredicto de v6** hasta cerrar `docs/comparacion_v4_v5.md`. DEV preliminar de v5: gana v4; si se confirma, H2 se lee solo como diagnóstico.

### H3 — escala muestras–precisión del replay (analogía con 140)
- **Cambio:** el replay deja de ser un flujo generativo infinito y se toma de un buffer finito de N = m × 4096 tripletas de familias confiables, m ∈ {0.5, 1, 2, 4} (`H3_x0.5…H3_x4`), pasos fijos. `E3_v4` (flujo infinito) es el límite N → ∞.
- **Predicción:** rel_err₁ cae monótonamente con N (lineal en log N según la analogía).
- **Métrica:** rel_err₁, acc h16; ajuste de log(1/rel_err₁) contra log N y contra N.
- **Éxito:** pendiente pareada de rel_err₁ vs log N < 0 con IC95 que excluye 0 en ambas fases. **Fallo:** pendiente ≈ 0 (saturación) → "más replay" no es el cuello.
- No se combina en V6_all (V6_all usa el flujo infinito, que es el límite favorable).

### H4 — destilación Sobolev (Jacobiano)
- **Cambio:** en muestras de replay del teacher, `λ₄ Σ_p ‖Jθ,p − J_T,p‖²_F / n_p` con columnas del Jacobiano 2×2 por punto por diferencias finitas (ε = 1e-3) en el modelo **y** en el teacher; λ₄ = 0.1. Brazos `H4_sobolev` (campo) y `H4_sobolev_mlp` (control) → 2×2 con `E3_v4`/`E3_mlp`.
- **Predicción:** baja |σmax(Jθ) − σmax(J*)| (`sigma_gap`) y sube acc h16, más en el campo que en el MLP.
- **Éxito:** interacción (H4_field − E3_v4) − (H4_mlp − E3_mlp) en acc h16 con IC95 > 0 en ambas fases. **Fallo:** ganan igual o nadie gana.

### H5 — peso continuo y sensibilidad (Brenier 1/3)
- **(a) Cambio:** `H5_soft`: el replay se toma de todas las familias con peso w_f = exp(−r_f/τ), τ = 0.01, en vez de máscara dura.
- **(a) Éxito:** acc h16 pareada IC95 ≥ 0 (límite inferior > −0.02) sin empeorar acc 1 paso > 0.02, y menor varianza entre semillas que E3. **Fallo:** diferencia negativa.
- **(b) Cambio:** `H5_eta{1,2,4,8}`: fracción η ∈ {1,2,4,8} % de targets de replay contaminados con ruido relativo σ = 0.1.
- **(b) Métrica:** `dstep_vs_E3` = ‖Δθ_η − Δθ_0‖ relativo RMS en TEST, contra la cota de acoplamiento W₂ ≤ σ√(η·f_replay); pendiente log-log.
- **(b) Éxito:** pendiente con IC95 < 1 (sub-lineal). **Fallo:** pendiente ≥ 1.

### H6 — consistencia de semigrupo
- **Cambio:** s ∈ {1, ½} como entrada extra de la red (H6 entrena con 2726 parámetros); Φ_s = Ψ + s·Δθ(Ψ,c,s); término `λ₆‖Φ₁(q) − Φ½(Φ½(q))‖²/‖y‖²`, λ₆ = 1.0, en todas las muestras. En inferencia s = 1 se pliega exactamente en el sesgo de la primera capa → modelo exportado con el layout estándar de 2590 parámetros y el mismo evaluador.
- **Predicción:** campo + L_SG supera al MLP gated en acc h16.
- **Éxito:** `H6_semigroup − E3_mlp` en acc h16 con IC95 > 0 en DEV y CONFIRM, **y** `H6_semigroup − E3_v4` ≥ 0. **Fallo:** IC incluye 0.

### H7 — contracción normal aprendida (Lyapunov)
- **Cambio:** con P⊥ = proyector ortogonal al espacio tangente de semejanzas span{u, i·u, 1, i} (el mismo que reproyecta V1), dirección aleatoria v ∈ rango(P⊥), `λ₇ [log‖P⊥ Jθ v‖ − log ρ]₊`, ρ = 0.9, λ₇ = 1e-3 (Jθ v por diferencias finitas, ε = 1e-3).
- **Predicción:** sin reproyección en inferencia, `pert_h8` ≤ 0.7 y acc h16 ≥ `E3_v4` con V1.
- **Éxito:** `H7_normal(V0) − E3_v4(V1reproj)` en acc h16 con límite inferior IC95 > −0.02 en ambas fases y `H7_normal − E3_v4(V0)` > 0. **Fallo:** peor que E3 V0 o necesita reproyección.

### v6 = combinación
- `V6_all` = H1 + H4 + H5a + H6 + H7 sobre E3 (H3 no aplica: flujo infinito; H5b es solo diagnóstico; H2 pendiente → `V6_all_H2` aparte).
- **Éxito de v6:** `V6_all − E3_v4` en acc h16 IC95 > 0 en DEV y CONFIRM, sin empeorar acc 1 paso > 0.02, **y** `V6_all − V6_all_mlp` IC95 > 0; `V6_shuffled` debe fallar el gate E33 (región < 8) en ≥ 15/16 semillas. **Fallo:** cualquiera de los anteriores.

## 3. Constantes fijadas (`src/v6_hypo.rs`)
`V6_HIDDEN=136`, `V6_STEPS=256000`, batch 32, lr 3e-3→3e-5, `TRUST_TAU=0.01`, `REPLAY_FRACTION=0.5`, `LAMBDA_H1=1.0`, `LOSS_AT={1,2,4,8}`, `EPISODE_H=8`, `PRM_TAU=0.05`, `H3_N_BASE=4096`, `H3_MULTS={0.5,1,2,4}`, `LAMBDA_H4=0.1`, `FD_EPS=1e-3`, `H5_TAU=0.01`, `H5_ETAS={0.01,0.02,0.04,0.08}`, `H5_SIGMA=0.1`, `LAMBDA_H6=1.0`, `LAMBDA_H7=1e-3`, `RHO_H7=0.9`. No se modifican entre DEV y CONFIRM (mismo binario/SHA; `run_v6` rechaza overrides fuera de smoke).

## 4. Ablación (23 brazos por semilla, orden fijo)
`E3_v4` (baseline, + evaluación V1reproj), `E3_mlp`, `E3_shuffled`, `H1_bias`, `H2_mh_eqc`*, `H2_prm_eqc`*, `H3_x0.5`, `H3_x1`, `H3_x2`, `H3_x4`, `H4_sobolev`, `H4_sobolev_mlp`, `H5_soft`, `H5_eta1`, `H5_eta2`, `H5_eta4`, `H5_eta8`, `H6_semigroup`, `H7_normal`, `V6_all`, `V6_all_mlp`, `V6_shuffled`, `V6_all_H2`* (*pendiente H2).

**Igualdad de cómputo:** todos los brazos usan los mismos pasos de optimizador, batch y muestras (E3); los brazos de horizonte igualan evaluaciones de Dφ. Los términos H4/H6/H7 añaden evaluaciones auxiliares de Dφ por muestra (diferencias finitas, medio paso); esas FLOPs extra se reportan (`train_s`) y no se compensan quitando pasos — es el costo de la hipótesis.

## 5. Estadística
Diferencias pareadas por semilla, IC95 bootstrap percentil (10 000 remuestreos, semilla fija 0x6B00) — `scripts/v6_analyze.py`. Una hipótesis se acepta solo si pasa en DEV **y** CONFIRM. Los negativos se registran. Gate por brazo: región E33 ≥ 8, `clean` (leakage = 0, queries = 0) y `fresh_ok`.

## 6. Desviaciones respecto a `hipotesis_math.md` (declaradas antes de ejecutar)
1. Sin fase de tuning: λ/ρ fijados a priori (§3).
2. H2 usa k ∈ {1,2,4,8} (no 1..8) para que la única diferencia con la referencia v5 sea la máscara.
3. H3: el replay de v4 es un flujo generativo infinito; "N actual" no existe, se define N_base = 4096.
4. H4: J por diferencias finitas en vez de derivadas exactas (evita segundas derivadas de la red).
5. H6: s entra como input sólo durante el entrenamiento (+136 parámetros), plegado exacto en inferencia.
6. H7: ‖P⊥JP⊥‖ se estima con una dirección normal aleatoria por muestra (cota inferior estocástica de la norma espectral), no con la norma espectral exacta.

## 7. Smoke y costo estimado (no es evidencia)
Smoke 0x5A60, 200 pasos, 23 brazos: 16 s de pared en total, todos `clean=true leak=0 q_test=0 fresh_ok=true`; dos corridas idénticas byte a byte (sin contar tiempos); interrumpir y reanudar desde checkpoint (`V6_CKPT_STEPS=50`, `V6_ABORT_AT_STEP=100`) da métricas bit-idénticas. La suma de train_s es ≈ 11.6 s por 200 pasos → ≈ 4.1 h por semilla con el presupuesto completo (×1280) → con 8 núcleos (16 semillas = 2 tandas) ≈ 8–9 h por fase, **≈ 17–19 h DEV+CONFIRM**.
