# Evaluación DEV/CONFIRM con los pesos entrenados en GPU (x50 y x20), solo inferencia

- Pesos: `weights_x50.bin` (zip x50, 4096 f32 = wp_exc[2048]|wp_inh[2048], d_basis=1024) y `weights_x20.bin` (zip x20, 2048 f32, d_basis=512). Formato tal como lo escribe `gpu-train` (`audit_dales_law_gpu`).
- Modelo: port CPU f32 del kernel WGSL `forward_pass` (paso por punto: `q + Σ_j (wpE−wpI + wfE−wfI)·φ_j(q)`, envolvente Brenier r_bound=15). φ transpilada mecánicamente desde los shaders (`wgsl2rs.py`; x50 de la rama, x20 del zip). Código: `bio_quantum_engine/src/bin/eval_trained/`.
- **Congelado**: el modelo GPU no tiene plasticidad en inferencia (solo `train_step_dale`/SWR modifican pesos; no se llaman). wf=1e-4 (init) ⇒ aporte 0. No usa las 4 observaciones (no hay ruta de adaptación). Rollout de 16 pasos desde el frame K_OBS−1.
- Protocolo = `eval` de la rama: mismos generadores/RNG por episodio, 64 eps/semilla, rel_err<0.05, mediana log por semilla promediada en 16 semillas, E33 h∈{1,2,4,8,16} ≥80%.
- Uso: `eval_trained <weights.bin|persistence> <dev|confirm|trainloss>`.

## Resultados (16 semillas por fase)
| Modelo | Fase | known acc16 | known med log | new acc16 | new med log | E33 | Dale viol |
|---|---|---|---|---|---|---|---|
| x50 entrenado | DEV | 0.00% | +0.591 | 0.00% | +0.449 | 0/16 | 0 |
| x50 entrenado | CONFIRM | 0.00% | +0.599 | 0.00% | +0.446 | 0/16 | 0 |
| x20 entrenado | DEV | 0.00% | +2.438 | 0.00% | +0.715 | 0/16 | 0 |
| x20 entrenado | CONFIRM | 0.00% | +1.603 | 0.00% | +0.490 | 0/16 | 0 |
| Persistencia (pred = último frame) | DEV | 28.52% | −1.155 | 3.03% | −0.525 | 0/16 | – |
| Persistencia | CONFIRM | 31.45% | −1.204 | 0.88% | −0.527 | 0/16 | – |
| `eval` por defecto (motor CPU sin estos pesos, se adapta por episodio) | DEV | 100.00% | −12.747 | 99.80% | −8.193 | 16/16 | 0 |
| idem | CONFIRM | 100.00% | −12.818 | 100.00% | −7.885 | 16/16 | 0 |

## Lectura honesta
- Resultados **degenerados**: con los pesos entrenados el error relativo final es >1 (log>0), peor que no predecir nada (persistencia). 0% en todo y E33 0/16, en DEV y en CONFIRM.
- Chequeo de fidelidad (`trainloss`, distribución exacta de `gpu-train`, 1 paso): MSE x50 = 0.1769, x20 = 0.0668, persistencia = 0.0047. El doc reporta MSE final 0.000478. La discrepancia es consistente con un bug del shader: `metrics[0] = metrics[0] + …` y las actualizaciones `wp_exc[j] = …` se hacen desde ~4M hilos sin atómicos (carrera de datos), así que la "pérdida" reportada no es el MSE real y el descenso de gradiente pierde/corrompe actualizaciones. No hay GPU aquí para ejecutar el WGSL original; el port es mecánico y la discrepancia podría deberse en parte a diferencias del port, pero el valor de persistencia y el sentido de las cifras apuntan al entrenamiento.
- Además el modelo GPU se entrenó solo en una rotación-contracción de un círculo (θ+0.08, r·0.995), no en las familias CST4.
