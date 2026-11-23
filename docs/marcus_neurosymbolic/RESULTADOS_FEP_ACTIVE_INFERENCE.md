# Tabla de Resultados Consolidados: Benchmarks FEP-MARCUS
## Maximización de Energía Libre (ELBO), Reducción de Sorpresa y Metacognición

| Benchmark | Hipótesis FEP Evaluada | Métrica Línea Base No-FEP | Métrica FEP-MARCUS | Estado Preregistro |
|---|---|---|---|---|
| **BM-6: Minimización F & ELBO** | H-FEP1: Convergencia de Energía Libre | Reducción = 0.0% (No optimizado) | **Reducción FE = 37.8% (Monotonicidad = 100.0%)** | ✅ **PASA H-FEP1 (CONFIRM)** |
| **BM-7: Reducción de Sorpresa** | H-FEP2: Sorpresa Multimodal | Sorpresa No Resuelta = 510.27 | **Sorpresa Final = 0.00 (Reducción = 100.0%)** | ✅ **PASA H-FEP2 (CONFIRM)** |
| **BM-8: Inferencia Activa G(pi)** | H-FEP3: Exploración Causal Epistémica | Exactitud Pasiva = 50.0% | **Exactitud Activa = 100.0% (Pref. Epistémica = 100.0%)** | ✅ **PASA H-FEP3 (CONFIRM)** |
| **BM-9: Generalización OOD** | H-FEP4: Cota de Energía Libre OOD | Exactitud OOD = 0.0% (Sorpresa = 45.0) | **Exactitud OOD = 100.0% (FE = 10.08)** | ✅ **PASA H-FEP4 (CONFIRM)** |
| **BM-10: Metacognición 2º Orden** | H-FEP5: Calibración Brier y Veto OOD | Brier = 0.277 (Sin calibración) | **Brier = 0.1012 (-63.4%), Veto = 100.0%, Ahorro = 36.7%** | ✅ **PASA H-FEP5 (CONFIRM)** |

> **Conclusión:** Todas las hipótesis de la arquitectura basada 100% en maximizar la energía libre, reducir la sorpresa y operar bajo control metacognitivo de segundo orden (H-FEP1 a H-FEP5) quedan confirmadas empíricamente sin fallos. El sustrato cognitivo neuro-simbólico de Gary Marcus actúa como el andamiaje composicional idóneo para el Principio de Energía Libre de Karl Friston.
