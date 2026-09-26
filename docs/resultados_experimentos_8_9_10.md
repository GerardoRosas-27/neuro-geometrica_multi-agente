# Resultados experimentos 8/9/10 — inferencia líquida

Rama: `exp/liquid-inference-experiments-8-9-10`

Hardware: ver ejecución (`uname -a`, `rustc -V`).
Modo: `--release`.
Commit al correr: c59b9fe.

## Tabla de registro

# Registry — experimentos 8/9/10 (liquid)

| Exp | seed | N | acc_seen | acc_unseen | acc_ood | mean_us | p50 | p95 | p99 | top1 | margin | abs | liq | rqm | sleep_ms | engrams | verdict |
|-----|-----:|--:|---------:|-----------:|--------:|--------:|----:|----:|----:|-----:|-------:|----:|----:|----:|---------:|--------:|---------|
| E8_representation_invariance | 952592 | 15 | 1.000 | 1.000 | 0.500 | 83.942 | 82.211 | 96.751 | 99.205 | 0.698 | 0.229 | 4 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952592 | 8 | 1.000 | 1.000 | 1.000 | 22.483 | 26.660 | 39.801 | 39.801 | 0.872 | 0.841 | 0 | 1 | 6 | 2.674 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952592 | 8 | 1.000 | 1.000 | 1.000 | 25.979 | 30.775 | 43.882 | 43.882 | 0.874 | 0.845 | 0 | 1 | 7 | 2.751 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |
| E8_representation_invariance | 952593 | 15 | 1.000 | 1.000 | 0.625 | 93.682 | 91.215 | 111.665 | 119.668 | 0.634 | 0.210 | 3 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952593 | 8 | 1.000 | 1.000 | 1.000 | 24.637 | 29.708 | 44.708 | 44.708 | 0.872 | 0.841 | 0 | 1 | 6 | 2.612 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952593 | 8 | 1.000 | 1.000 | 1.000 | 26.927 | 29.837 | 45.167 | 45.167 | 0.874 | 0.845 | 0 | 1 | 7 | 2.874 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |
| E8_representation_invariance | 952594 | 15 | 1.000 | 1.000 | 0.500 | 84.919 | 83.555 | 90.389 | 97.553 | 0.700 | 0.229 | 4 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952594 | 8 | 1.000 | 1.000 | 1.000 | 24.614 | 26.995 | 48.781 | 48.781 | 0.872 | 0.841 | 0 | 1 | 6 | 2.649 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952594 | 8 | 1.000 | 1.000 | 1.000 | 26.527 | 31.717 | 41.066 | 41.066 | 0.874 | 0.845 | 0 | 1 | 7 | 2.716 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |


## Fallos / notas por semilla

## seed=0xe8910
E8 failures (4): ["ood_reject:xqz9:Some((3, 0.9598204098139542))", "ood_reject:mesa:Some((1, 0.9314125995691108))", "ood_reject:gato:Some((3, 0.9314085802837553))", "ood_reject:lobo:Some((3, 0.9314082751201338))"]
E9 failures (0): []
E10 failures (0): []
E8 notes: train='perro' concept=3 failures=4 threshold=0.60
E9 notes: control liquid-only compose_ok=0/3 (expect ~0); failures=0
E10 notes: failures=0 variants=A,B,C
periphery: gemma-gguf

## seed=0xe8911
E8 failures (3): ["ood_reject:mesa:Some((1, 0.9314106161277985))", "ood_reject:gato:Some((3, 0.931408811021837))", "ood_reject:lobo:Some((3, 0.9314098728576061))"]
E9 failures (0): []
E10 failures (0): []
E8 notes: train='perro' concept=3 failures=3 threshold=0.60
E9 notes: control liquid-only compose_ok=0/3 (expect ~0); failures=0
E10 notes: failures=0 variants=A,B,C
periphery: gemma-gguf

## seed=0xe8912
E8 failures (4): ["ood_reject:chien:Some((1, 0.9598252173752919))", "ood_reject:mesa:Some((1, 0.9598227679157019))", "ood_reject:gato:Some((3, 0.9314113123281987))", "ood_reject:lobo:Some((3, 0.9314103805172929))"]
E9 failures (0): []
E10 failures (0): []
E8 notes: train='perro' concept=3 failures=4 threshold=0.60
E9 notes: control liquid-only compose_ok=0/3 (expect ~0); failures=0
E10 notes: failures=0 variants=A,B,C
periphery: gemma-gguf


## Controles y honestidad
- E8: periferia léxico si GGUF ausente; no se afirma invariancia cross-lingual.
- E9: liquid-only compose esperado ~0; composición vía `infer_compose` sin teach transitivo.
- E10: abstención / hops en cue OOD; perturbación de onda con margen.
- No se presenta como evidencia de cognición general.
