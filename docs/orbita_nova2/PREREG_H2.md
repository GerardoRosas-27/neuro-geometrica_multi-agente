# ÓRBITA-Nova-2 · H2 NOVA_MIX — mezcla de 4 operadores de la biblioteca, sólo 4 pesos por episodio (preregistro)

Rama `exp/orbita-nova2` (desde `34dd1f7`). Aprobado por Bruster (brief «Tras el veredicto de Nova», propuesta 2; MixER arXiv:2502.05335). Fecha: 2026-10-24 (America/Mexico_City, UTC−6).
Se congela con su commit **antes** de cualquier dato (smoke incluido). No modifica PREREG_EXP1/2/3, PASO0_C6 ni PREREG_H1. Módulo nuevo opt-in `orbita/src/nova2h2.rs`, subcomando `orbita nova2-h2`. Sin calibración nueva; entre DEV y CONFIRM no se cambia nada. 0 consultas TEST.

## 0. Base (igual que H1)
Biblioteca **C6 legada** construida por el flujo NOVA idéntico a exp3/H1 (τ_C6 = 0,9386641928884383, consolidación cada 50). C3 activo en todos los brazos NOVA*. Núcleo ÓRBITA congelado. Datos, H, checkpoints, semillas DEV `0xA400–0xA40F` / CONFIRM `0xB400–0xB40F` como H1.

## 1. NOVA_MIX (definición congelada)
1. **Candidatos**: igual que NOVA_RET (H1): se puntúan **todas** las entradas por s_j = media_{t=1..k} ln rel_err del rollout libre de `lr_step(θ_j)` desde x0; top-4 = 4 menores (empates: índice menor), ordenadas j1..j4.
2. **Operador mezcla**: ΔK(w) = Σ_{i=1..4} w_i ΔK_{j_i}, con ΔK_j = U_j V_jᵀ. Se implementa exactamente como un θ de rango 8: U = [w_1U_{j1} | … | w_4U_{j4}], V = [V_{j1} | … | V_{j4}] (apply_delta es lineal en cada término).
3. **Ajuste por episodio**: sólo w ∈ ℝ⁴ (las entradas no se tocan). `fit_gn` (Gauss-Newton/LM, mismos residuos relativos que C4) sobre los **k** pasos observados, inicio y prior **one-hot w⁰ = (1,0,0,0)** (= NOVA_RET), ridge **λ_w = 1e-2** hacia w⁰ (fijado a priori, sin calibrar).
4. Después, igual que NOVA_RET: C5 `arbitrate(θ_mix, r=8, kf, kv, EPS_SPEC)` + C7 `gate_nodamage`; si acepta paso adaptado, si no ÓRBITA; C3 en la evaluación h16. (Fuga de C5 idéntica a NOVA/NOVA_RET; anotada.)
5. Biblioteca con < 4 entradas: se usan las que haya (w de esa dimensión); vacía → ÓRBITA+C3.

## 2. Brazos
ÓRBITA+C3, NOVA (exp3), NOVA_RET (H1, idéntico), **NOVA_MIX** (§1), **NOVA_MIX_SHUF**: 4 entradas al azar (misma RNG que NOVA_RET_SHUF de H1: Xoshiro256**, seed ⊕ 0x5EF1 ⊕ (n≪20) ⊕ (k≪16) ⊕ índice en H), ordenadas por s_j, prior one-hot en la mejor de las 4, mismos pesos/λ_w/C5/C7/C3.

## 3. Hipótesis y criterios
Media de la diferencia pareada por seed (n=16, n=2400), IC95 bootstrap percentil 10 000 réplicas, semilla 12345 (`scripts/analyze_nova2_h2.py`). δ = 0,05. Métrica: mediana log rel_err h16 en nuevas {8,10,11}.
- **H2 (primaria)**: NOVA_MIX − NOVA, k=2 < 0. Aceptada si IC95 sup < 0 en DEV **y** CONFIRM; refutada si IC95 inf > 0 en CONFIRM; inconclusa si no.
- **Controles clave (a la luz de H1)**: NOVA_MIX − NOVA_MIX_SHUF y NOVA_MIX − ÓRBITA+C3, k=2. **El crédito a la biblioteca exige vencer a SHUF** (IC95 sup < 0 en ambas fases); si H2 se acepta sin eso, se reporta como «no atribuible a la biblioteca». Vencer a ÓRBITA+C3 se exige para hablar de mejora real sobre el núcleo.
- **Secundarias (no vetan)**: lo mismo con k=4/8; NOVA_MIX − NOVA_RET (k=2/4/8; ¿aportan los pesos?).
- **No-daño (veta)**: NOVA_MIX − NOVA, acc16 conocidas 0–7, k=2/4/8: IC95 inf > −δ en ambas fases.
**Éxito H2** ⇔ H2 aceptada ∧ no-daño aceptado. Diagnósticos: distribución de w (media de w_i, |w|₁, cuota max|w_i|/|w|₁, ‖w − w⁰‖), coincidencia de familia del top-1, tasa de paso C5+C7. `RESULTADOS_H2.md` se publica en cualquier caso.

Nota a priori: en H1 el no-daño falló por ~16–21 pp frente a NOVA; 4 pesos dan poca flexibilidad y es probable que vuelva a fallar.
