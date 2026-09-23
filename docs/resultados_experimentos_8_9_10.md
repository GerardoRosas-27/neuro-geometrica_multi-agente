# Resultados experimentos 8/9/10 — inferencia líquida

Rama: `exp/liquid-inference-experiments-8-9-10`

Hardware: ver ejecución (`uname -a`, `rustc -V`).
Modo: `--release`.
Commit al correr: 58b2603.

## Tabla de registro

# Registry — experimentos 8/9/10 (liquid)

| Exp | seed | N | acc_seen | acc_unseen | acc_ood | mean_us | p50 | p95 | p99 | top1 | margin | abs | liq | rqm | sleep_ms | engrams | verdict |
|-----|-----:|--:|---------:|-----------:|--------:|--------:|----:|----:|----:|-----:|-------:|----:|----:|----:|---------:|--------:|---------|
| E8_representation_invariance | 952592 | 15 | 1.000 | 1.000 | 0.500 | 27.847 | 25.609 | 34.181 | 39.939 | 0.698 | 0.229 | 4 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952592 | 8 | 1.000 | 1.000 | 1.000 | 5.551 | 3.904 | 17.309 | 17.309 | 0.872 | 0.841 | 0 | 1 | 6 | 0.500 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952592 | 8 | 1.000 | 1.000 | 1.000 | 3.910 | 3.528 | 6.833 | 6.833 | 0.874 | 0.845 | 0 | 1 | 7 | 0.483 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |
| E8_representation_invariance | 952593 | 15 | 1.000 | 1.000 | 0.625 | 25.039 | 24.922 | 26.639 | 27.140 | 0.634 | 0.210 | 3 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952593 | 8 | 1.000 | 1.000 | 1.000 | 3.309 | 3.664 | 5.015 | 5.015 | 0.872 | 0.841 | 0 | 1 | 6 | 0.498 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952593 | 8 | 1.000 | 1.000 | 1.000 | 3.708 | 3.469 | 6.264 | 6.264 | 0.874 | 0.845 | 0 | 1 | 7 | 0.497 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |
| E8_representation_invariance | 952594 | 15 | 1.000 | 1.000 | 0.500 | 42.235 | 39.634 | 44.003 | 73.267 | 0.700 | 0.229 | 4 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952594 | 8 | 1.000 | 1.000 | 1.000 | 4.648 | 5.333 | 7.440 | 7.440 | 0.872 | 0.841 | 0 | 1 | 6 | 0.695 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952594 | 8 | 1.000 | 1.000 | 1.000 | 4.340 | 5.379 | 5.976 | 5.976 | 0.874 | 0.845 | 0 | 1 | 7 | 0.756 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |


## Fallos / notas por semilla

## seed=0xe8910
E8 failures (4): ["ood_reject:xqz9:Some((3, 0.9598204098139542))", "ood_reject:mesa:Some((1, 0.9314125995691108))", "ood_reject:gato:Some((3, 0.9314085802837553))", "ood_reject:lobo:Some((3, 0.9314082751201338))"]
E9 failures (0): []
E10 failures (0): []
E8 notes: train='perro' concept=3 failures=4 threshold=0.60
E9 notes: control liquid-only compose_ok=0/3 (expect ~0); failures=0
E10 notes: failures=0 variants=A,B,C
periphery: gemma-shaped-lexicon (GGUF absent; honest)

## seed=0xe8911
E8 failures (3): ["ood_reject:mesa:Some((1, 0.9314106161277985))", "ood_reject:gato:Some((3, 0.931408811021837))", "ood_reject:lobo:Some((3, 0.9314098728576061))"]
E9 failures (0): []
E10 failures (0): []
E8 notes: train='perro' concept=3 failures=3 threshold=0.60
E9 notes: control liquid-only compose_ok=0/3 (expect ~0); failures=0
E10 notes: failures=0 variants=A,B,C
periphery: gemma-shaped-lexicon (GGUF absent; honest)

## seed=0xe8912
E8 failures (4): ["ood_reject:chien:Some((1, 0.9598252173752919))", "ood_reject:mesa:Some((1, 0.9598227679157019))", "ood_reject:gato:Some((3, 0.9314113123281987))", "ood_reject:lobo:Some((3, 0.9314103805172929))"]
E9 failures (0): []
E10 failures (0): []
E8 notes: train='perro' concept=3 failures=4 threshold=0.60
E9 notes: control liquid-only compose_ok=0/3 (expect ~0); failures=0
E10 notes: failures=0 variants=A,B,C
periphery: gemma-shaped-lexicon (GGUF absent; honest)


## Controles y honestidad
- E8: periferia léxico si GGUF ausente; no se afirma invariancia cross-lingual.
- E9: liquid-only compose esperado ~0; composición vía `infer_compose` sin teach transitivo.
- E10: abstención / hops en cue OOD; perturbación de onda con margen.
- No se presenta como evidencia de cognición general.

---

## Hardware / toolchain (America/Mexico_City)

```
Linux grok-bot-vm-416735841 6.12.94+ x86_64
rustc 1.98.1 (48a229cea 2026-09-01)
Fecha: 2026-09-23 ~01:58 CST
```

Periferia: gemma-shaped-lexicon (GGUF ausente; honesto). Ψ sin tokens.

## Mejoras de arquitectura (motivadas por resultados)

1. **WavePredictCore** — `RankedPrediction` + `predict_ranked` / `predict_chain` / `predict_ranked_from_observation` (margen top-1/top-2 → abstención E9/E10).
2. **FusedLiquidCdt** — `infer_compose` multi-hop RQM **sin** teach transitivo; `abstain_margin`, `compose_max_hops`; `FuseReport` con top1/top2/margin/energy/abstained/hops.
3. **ConceptAttractorBank (E8)** — matching por coseno a fingerprints de sonda congelada + margen entre atractores (evita colapso del projector compartido).
4. **field_linguistic_layer** — diminutivos ES (`ito/ita/illo/…`) para alinear `perrito` con raíz de `perro` sin tokens en Ψ.

## Controles

- E9 liquid-only compose ≈ 0/3 (no finge composición geométrica).
- E8 no afirma invariancia cross-lingual con léxico.
- No se presenta como evidencia de cognición general.

## Baselines post-cambio

`liquid_cdt_memory`, `liquid_cdt_rqm_fuse`, `liquid_cdt_vs_main`, `wave_predict_core`: OK en `--release`.

## Confirmación de rama

- Push solo a `origin/exp/liquid-inference-experiments-8-9-10`.
- **main no modificado.**
