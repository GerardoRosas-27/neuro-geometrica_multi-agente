# PREREG — BIO-CONSTELACIÓN (híbrido neocórtex) bajo el protocolo CONSTELACIÓN v4

Rama `exp/hibrido-constelacion-neocortex`. Escrito y empujado **antes** de ejecutar DEV. Sin re-ajuste entre DEV y CONFIRM.

## Paso 0 (port al arnés v4)
- Se descarta `run_hybrid.py` (generador de juguete). El híbrido se re-implementa en `orbita/src/constelacion4/bio.rs` dentro del arnés v4
  (código v4/mejoras copiado tal cual de `exp/constelacion-v4-mejoras @ ce63ac3`): mismo `gen_heldout`/`gen_stream`/`gen_known_stream`
  (familias 0–7 = `v4_dataset::apply_rule`, nuevas 8/10/11 = `rule`), k=4, h16, `clamp_err`/`log_rel`/`REL_ERR_OK`, E33 vía `e33_score`.
- Presupuesto: idéntico a v4 (flujo A N_A=3072 conocidas + flujo B 192×3 nuevas + 192 conocidas), T_F=128 iteraciones rápidas, **sembrado por semilla de evaluación**.
- Fuga E33 h=1 corregida: la predicción del cuadro t se emite antes de adaptar W^F con la transición (t−1→t); h=1 usa sólo W^P (W^F=0).
- Componentes portados: base L2/3 φ∈R¹⁰, latente Koopman L5 contractivo (ρ, ω), W^P lento, W^F rápido (BTSP≈NLMS normalizado),
  Ley de Dale (W=W^E−W^I, ≥1e-4), STP (depresión de recursos que escala ΔW^F), consolidación SWR (W^P += c·W^F por episodio de entrenamiento).
  **No portados** (documentado): prior tálamo-cortical (necesita el contexto verdadero, no disponible en held-out sin fuga), hipocampo CA3/DG, neuromoduladores.
- Verificación: `docs/bio_constelacion/paso0.json` — v4 vía este binario vs binario mejoras, semilla 0xC420 (na=256): hash de pesos y JSON held idénticos.
  Además la cadena re-ejecuta v4 en DEV/CONFIRM y **aborta** si el hash por semilla difiere del de `exp/constelacion-v4-mejoras`.

## Ajuste
Sólo 0xC420–0xC423, grilla fija de 36 configs (`scripts/bio_tune_grid.py`, `tuning/grid.jsonl`), parada = grilla agotada, 0 consultas TEST/E33.
Elegida: `frozen.json["hyb"]` = {eta_p 0.001, eta_f 1.0, t_f 128, cons 0.05, stp_u 0.2, stp_tau 4, rho 0.9, omega 0.3}.

## Brazos
`v4` (cst4m-run, frozen v4), `hyb` (híbrido completo), `hyb_nobio` (ablación: sin Dale/STP/consolidación, misma config), `ref` (ÓRBITA-v1 congelada y persistencia x̂16=x_K).

## Métricas e hipótesis
- **Primaria (única)**: Δ = mediana por semilla de `new_med_log` (log rel_err h16, familias nuevas, k=4): hyb − v4. Éxito si IC95 superior < 0 y |Δ| ≥ δ=0.05.
- Secundarias (sin multiplicidad confirmatoria, Holm reportado): hyb − ÓRBITA-v1 (new_med_log), hyb − persistencia, hyb − hyb_nobio.
- No-daño: known_acc16 hyb − v4 ≥ −0.05 (IC95 inferior); E33 acc h16 hyb − v4 ≥ −0.05.
- Bootstrap pareado por semilla, 10 000 remuestreos, semilla 12345, IC95 percentil. DEV 0xA400–A40F (16), CONFIRM 0xB400–B40F (16).
  Veredicto final = CONFIRM. Resultados negativos se reportan tal cual.
