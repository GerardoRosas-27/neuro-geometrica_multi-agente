# Resultados experimentos 8/9/10 — inferencia líquida

Rama: `exp/liquid-inference-experiments-8-9-10`

Hardware: ver ejecución (`uname -a`, `rustc -V`).
Modo: `--release`.
Commit al correr: 0cc1a65.

## Tabla de registro

# Registry — experimentos 8/9/10 (liquid)

| Exp | seed | N | acc_seen | acc_unseen | acc_ood | mean_us | p50 | p95 | p99 | top1 | margin | abs | liq | rqm | sleep_ms | engrams | verdict |
|-----|-----:|--:|---------:|-----------:|--------:|--------:|----:|----:|----:|-----:|-------:|----:|----:|----:|---------:|--------:|---------|
| E8_representation_invariance | 952592 | 15 | 1.000 | 1.000 | 0.500 | 21.556 | 20.407 | 22.839 | 34.181 | 0.698 | 0.229 | 4 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952592 | 8 | 1.000 | 1.000 | 1.000 | 2.608 | 2.858 | 3.970 | 3.970 | 0.872 | 0.841 | 0 | 1 | 6 | 0.469 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952592 | 8 | 1.000 | 1.000 | 1.000 | 2.555 | 2.942 | 3.659 | 3.659 | 0.874 | 0.845 | 0 | 1 | 7 | 0.393 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |
| E8_representation_invariance | 952593 | 15 | 1.000 | 1.000 | 0.625 | 20.801 | 20.452 | 21.953 | 22.998 | 0.634 | 0.210 | 3 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952593 | 8 | 1.000 | 1.000 | 1.000 | 3.186 | 3.586 | 4.876 | 4.876 | 0.872 | 0.841 | 0 | 1 | 6 | 0.522 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952593 | 8 | 1.000 | 1.000 | 1.000 | 2.557 | 3.047 | 3.623 | 3.623 | 0.874 | 0.845 | 0 | 1 | 7 | 0.387 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |
| E8_representation_invariance | 952594 | 15 | 1.000 | 1.000 | 0.500 | 20.858 | 20.615 | 22.225 | 22.933 | 0.700 | 0.229 | 4 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952594 | 8 | 1.000 | 1.000 | 1.000 | 3.207 | 3.509 | 4.861 | 4.861 | 0.872 | 0.841 | 0 | 1 | 6 | 0.504 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952594 | 8 | 1.000 | 1.000 | 1.000 | 3.225 | 3.067 | 5.240 | 5.240 | 0.874 | 0.845 | 0 | 1 | 7 | 0.412 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |


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
