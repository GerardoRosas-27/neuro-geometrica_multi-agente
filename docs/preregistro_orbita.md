# Pre-registro — ÓRBITA vs v4 §16 (escrito y commiteado ANTES de correr DEV y CONFIRM)

**Rama:** `exp/orbita` (desde `main` `49c00a9`) · **Fecha:** 2026-10-13 · Diseño: `docs/arquitectura_optimizada.md` (rama `exp/arquitectura-optimizada`).
Código **separado** en `orbita/` (crate Rust propio + entrenador PyTorch). No se modifica ningún archivo de v4/v5; la librería v4 sólo se *usa* (dataset sellado, auditor, CDT→consolidación→máscara de confianza, `evaluate`, `onestep_breakdown`, métricas). No se compara con v5.

## 1. Implementación concreta (ÓRBITA-v1) y desviaciones declaradas respecto al diseño
Campo por punto (igual que Dφ: mismos pesos para cada punto 2-D, contexto c ∈ ℝ¹⁰):
- `head(c)` (MLP tanh 10→24→76) → ω (4), s (4), A (28, antisimétrica 8×8), β (8), b_k (16), a_k (16).
- Codificador z = [q; g(q,c)] ∈ ℝ⁸ (g: MLP tanh 12→16→6), decodificador = primeras 2 coordenadas (exacto).
- Núcleo: z' = Q·blkdiag(ρ_j R(ω_j))·Qᵀ z + β, Q = Cayley(A) (ortogonal), ρ_j = e^{s_j} en los 2 bloques "tangentes", e^{−softplus(s_j)} < 1 en los 2 "normales".
- Corrector de Brenier (gradiente de un potencial convexo de una capa), r = 2 iteraciones, η = 0.5: u ← u + η W_cᵀ (softplus(a)⊙σ(W_c u + b)).
- **Parámetros: 2602** (v4 E3/M3: 2590; +0.5 %).
- Pérdida: relativa de un paso (igual que v4) + λ_H Σ_{k∈{1,2,4,8}} 0.9^k ‖[núcleo^k(z)]_{xy} − y_k^{teacher}‖²/‖y_k‖² (sólo ejemplos de replay; objetivos multi-paso = teacher consolidado iterado, **nunca** el generador) + λ_B (sesgo medio)² + λ_C ‖u−núcleo(z)‖². **λ_H = 0.01, multi-horizonte sólo sobre las coordenadas xy, λ_B = 1, λ_C = 0.01.**
- Datos: TRAIN real (50 %) ∪ replay del teacher (50 %), con el replay muestreado ∝ w_f = exp(−r_f/τ), τ = 0.01 (r_f = residuo de ajuste de consolidación por familia, idéntico a §16; con pesos 0/1 coincide con la máscara de confianza de v4). El replay se materializa en un pool de 100 000 muestras generativas por seed (v4 lo genera en línea); todas sus etiquetas pasan por el auditor (hits → flags de fuga en TEST).
- Presupuesto **idéntico** a E3: 256 000 pasos Adam, batch 32, lr coseno 3e-3→3e-5. FLOPs por paso mayores (Cayley 8×8, corrector, multi-horizonte): se reporta tiempo de pared.
- **Desviaciones del diseño (declaradas):** sin verificador/inferencia adaptativa, sin Sobolev, sin Muon (Adam como v4), sin EMA (stop-grad del codificador en línea), multi-horizonte restringido a xy (la versión latente completa destruyó el aprendizaje en tuning).

## 2. Tuning (divulgado, sin tocar DEV/CONFIRM)
Seeds `0xC500, 0xC501`, 32 000 pasos (1×), evaluadas en su propio TEST (seeds independientes de DEV/CONFIRM). Logs `artifacts/orbita/tune/`.
| variante | acc1 C500/C501 | región C500/C501 | h16 C500/C501 |
|---|---|---|---|
| λ_H = 0 | 0.981 / 0.918 | 8 / 4 | 0.219 / 0.047 |
| latente completo λ_H = 0.01 | 0.711 / 0.430 | 0 / 0 | 0 / 0 |
| latente completo λ_H = 0.1 | 0 / 0 | 0 / 0 | 0 / 0 |
| **xy λ_H = 0.01 (elegida)** | 0.973 / 0.961 | 8 / 8 | 0.312 / 0.109 |
| xy λ_H = 0.1 | 0.938 / 0.949 | 4 / 4 | 0.125 / 0.031 |
Con 2 seeds la elección es ruidosa; queda fijada aquí y no se cambia.

## 3. Protocolo (idéntico a v4 §16 DIST_E)
- Seeds DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F`; mismo `generate_and_seal(seed)`, mismo auditor, mismo TEST.
- Un paso: `evaluate(TEST, h=1)` (correcto = rel_err < 5 %), desglose por familia `onestep_breakdown`.
- E33: rotate+reflect de TEST, free-run, h = 1,2,4,8,16,32,64, perturbación δ₀ con la misma semilla (`seed^0xD33`), región = máximo h contiguo con acc ≥ 0.5 (copia fiel de `e33_rollout`, variante V0 sin re-proyección).
- Auditoría: queries TEST = 0, leaked = 0 (con flags de CDT y de replay), proceso nuevo (`orbita fresh`) con |Δacc| < 1e-12 y 0 queries. Paridad Rust↔PyTorch (max ‖Δ‖ < 1e-9) reportada.
- **Comparadores:** v4 `E3_gated_trusted` / `R_E3_V0_pure` y control MLP `M3_mlp_gated_trusted` / `R_M3_V0`, **tomados de los artefactos publicados** `artifacts/v4/distill_e/{dev,confirm}/metrics.csv` (mismas seeds y TEST; código de `main` idéntico al que los generó). No se re-entrenan.

## 4. Criterios (cerrados antes de DEV)
1. **Gate ÓRBITA (como §16):** región ≥ 8, limpio y proceso nuevo OK en ≥ 9/16 seeds, en DEV y en CONFIRM.
2. **ÓRBITA gana a v4** sólo si: (a) gate 1 PASS en ambas fases; **y** (b) diferencia pareada ÓRBITA − E3 en **acc h16** con IC95 bootstrap (10 000 remuestreos, semilla 12345) > 0 en DEV **y** en CONFIRM; **y** (c) acc de un paso media no peor que E3 en más de 0.02 en ninguna fase.
3. **v4 gana** si la diferencia en acc h16 tiene IC95 < 0 en ambas fases. Otro caso: **empate / no concluyente**.
4. Superar al MLP (secundario, se reporta igual): mismo análisis contra M3.
5. Si gana (criterio 2), el documento de resultados propondrá llamarla **v6** (sin renombrar ramas). Si pierde o empata, se reporta tal cual.
6. Sin cambios de código ni hiperparámetros entre DEV y CONFIRM; CONFIRM se corre con el mismo SHA de código que DEV (los commits intermedios sólo añaden artefactos/resultados DEV).
7. La ablación λ_H = 0 no se corre en DEV/CONFIRM en este ciclo (coste); queda pendiente.

Script de análisis: `orbita/analyze_orbita.py`. Ejecución: `orbita/run_phase.sh dev|confirm`.
