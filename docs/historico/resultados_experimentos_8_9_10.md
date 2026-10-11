# Resultados experimentos 8/9/10 — inferencia líquida

> **Aviso de procedencia (nota añadida 2026-10-06).** Este archivo lo escribe
> automáticamente un test de `src/liquid_experiments_8_9_10.rs`. Se
> **regeneró sin querer** en el commit `7619728` (PR #30, 2026-09-26): los
> veredictos coinciden con la corrida original, pero las latencias son de otra
> máquina (≈3–7× mayores), la etiqueta `periphery: gemma-gguf` contradice las
> notas de E8 (léxico) y se perdieron las secciones de hardware, mejoras de
> arquitectura y controles. La versión original (commit al correr `58b2603`,
> periferia léxico) está en `git show 7619728^:docs/resultados_experimentos_8_9_10.md`.

Rama: `exp/liquid-inference-experiments-8-9-10`

Hardware: ver ejecución (`uname -a`, `rustc -V`).
Modo: `--release`.
Commit al correr: db27d89.

## Tabla de registro

# Registry — experimentos 8/9/10 (liquid)

| Exp | seed | N | acc_seen | acc_unseen | acc_ood | mean_us | p50 | p95 | p99 | top1 | margin | abs | liq | rqm | sleep_ms | engrams | verdict |
|-----|-----:|--:|---------:|-----------:|--------:|--------:|----:|----:|----:|-----:|-------:|----:|----:|----:|---------:|--------:|---------|
| E8_representation_invariance | 952592 | 15 | 1.000 | 1.000 | 0.500 | 27.289 | 27.627 | 28.825 | 29.003 | 0.698 | 0.229 | 4 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952592 | 8 | 1.000 | 1.000 | 1.000 | 4.004 | 3.147 | 12.338 | 12.338 | 0.872 | 0.841 | 0 | 1 | 6 | 0.419 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952592 | 8 | 1.000 | 1.000 | 1.000 | 2.465 | 2.945 | 3.683 | 3.683 | 0.874 | 0.845 | 0 | 1 | 7 | 0.383 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |
| E8_representation_invariance | 952593 | 15 | 1.000 | 1.000 | 0.625 | 20.517 | 20.241 | 21.552 | 22.496 | 0.634 | 0.210 | 3 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952593 | 8 | 1.000 | 1.000 | 1.000 | 2.612 | 2.832 | 3.973 | 3.973 | 0.872 | 0.841 | 0 | 1 | 6 | 0.453 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952593 | 8 | 1.000 | 1.000 | 1.000 | 2.473 | 2.892 | 3.775 | 3.775 | 0.874 | 0.845 | 0 | 1 | 7 | 0.380 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |
| E8_representation_invariance | 952594 | 15 | 1.000 | 1.000 | 0.500 | 20.440 | 20.195 | 21.524 | 22.422 | 0.700 | 0.229 | 4 | 0 | 0 | 0.000 | 0→2 | PARTIAL_PASS: stem/paraphrase attractor match; cross-lingual not claimed (lexicon) |
| E9_relational_composition | 952594 | 8 | 1.000 | 1.000 | 1.000 | 2.583 | 2.819 | 3.968 | 3.968 | 0.872 | 0.841 | 0 | 1 | 6 | 0.396 | 0→3 | PASS: direct + composed hops via RQM walk (no transitive teach) |
| E10_future_prediction | 952594 | 8 | 1.000 | 1.000 | 1.000 | 2.524 | 2.948 | 3.835 | 3.835 | 0.874 | 0.845 | 0 | 1 | 7 | 0.386 | 0→3 | PASS_PARTIAL: distance/bifurcation/perturbation via compose+wave |


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
