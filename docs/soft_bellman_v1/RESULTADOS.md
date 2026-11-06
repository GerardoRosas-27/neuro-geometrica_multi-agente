# Soft-Bellman v1 — resultados reales

Cierre: 2026-11-06 17:00:28 America/Mexico_City (UTC-6).

## Veredicto
**PASA el criterio primario preregistrado en este benchmark sintético pequeño.** Una implementación Rust con nueve parámetros aprendió costos locales desde acciones de un oráculo y generalizó a mapas no vistos. No es un modelo de lenguaje ni demuestra aprendizaje general, autonomía o sustitución de BQ10.

## Procedimiento y trazabilidad
- Rama `exp/soft-bellman-v1`; crate aislado `soft_bellman/`. No se modificó la aplicación existente.
- Prerregistro ANTES de DEV: `3a1901f8c080540e9c5d4f3f5b0a53a6cb526d96` (`PLAN.md`).
- Código y DEV congelados ANTES de CONFIRM: `43fcd21`; hash SHA256 de `main.rs`: `8f56f2cffac5e70e146b4df639bdb1cc5e87f3ca34dffe6d3650b1b465e5fad4` (`FROZEN.json`). No cambios de hiperparámetros ni selección de checkpoint después de DEV.
- DEV 16/16 semillas 0xA400–0xA40F; CONFIRM 16/16 semillas 0xB400–0xB40F. Cada semilla entrena un modelo independiente con mapas propios: 24 train, 8 validación, 64 test 6×6 y 16 test 3×3×3. Todos los estados libres no-meta se evalúan. Promedio primero por mapa, después por semilla; no tratar miles de estados correlacionados como réplicas independientes.
- 160 actualizaciones full-batch Adam (lr .03, beta .9/.999, epsilon 1e-8, clip global 10), sin early stopping. Temperatura .2, H=número de celdas. Rust f64, sin dependencias externas, CPU, cuatro procesos `nice -n 10`; build `-j 2` en fase separada. Python solo orquesta procesos y calcula estadísticas: no entrena.
- Gradiente analítico inverso completo a través de Soft-Bellman desenrollado, métrica diagonal SPD `G_dd=.01+softplus(w_d·[1,a,b])²` y potencial no negativo. Dos cabezas métricas (eje0 y resto de ejes), una de potencial; 9 parámetros. Costo `1+sqrt(G_dd)+softplus(w_phi·f_destino)`. No matriz Cholesky completa.
- V0=∞ excepto meta0, meta absorbente/fijada0 en todas las capas, logsumexp estable; estados inalcanzables conservan∞. El plan usa horizonte decreciente. Costo≥1.1 > tau·log(6); no se afirma estabilidad general para otros parámetros/grados ni horizonte infinito.

## CONFIRM: mapas 2D no vistos
Regret = costo real de ruta / costo óptimo − 1. En la tabla se expresa como porcentaje del costo óptimo. Fracasos, si ocurren, reciben regret5 (500%); el regret del MLP por ello NO equivale a sobrecosto de rutas exitosas.

| Modelo | NLL acción (nat) | Acción óptima | Llega a meta | Regret |
|---|---:|---:|---:|---:|
| Soft-Bellman entrenado | 0.033595 | 99.1430% | 100.0000% | 0.006907% |
| Mismo campo sin entrenar | 0.292821 | 82.9209% | 100.0000% | 3.625497% |
| Planificador costo uniforme | 0.291898 | 82.3365% | 100.0000% | 3.841467% |
| MLP reactivo 81 parámetros | 0.413241 | 83.3663% | 82.0795% | 90.668873% |

Contraste primario campo−uniforme: **-3.834560 puntos porcentuales**, IC95% **[-4.004103, -3.659101]**; éxito no disminuye (ambos100%). Campo−sin entrenar: -3.618590 pp [-4.177180, -3.028243].

CONFIRM: 1024 mapas test, 29932 inicios/etiquetas; la unidad estadística sigue siendo16 semillas. Bootstrap pareado10000, seed12345, delta.05, percentiles interpolados. Contraste principal preregistrado; resto exploratorio, sin corrección múltiple.

NLL de entrenamiento promedio: 0.296192 antes → 0.029758 después. NLL test: 0.292821 → 0.033595.

## DEV y transferencia 3D

| Fase / split | Regret campo | Regret uniforme | Δ campo−uniforme pp [IC95%] | Éxito campo |
|---|---:|---:|---:|---:|
| DEV / test | 0.007288% | 3.860487% | -3.853199 [-4.060947, -3.642340] | 100.00% |
| DEV / ood3d | 0.010204% | 4.517958% | -4.507754 [-4.857943, -4.156960] | 100.00% |
| CONFIRM / ood3d | 0.007859% | 5.019431% | -5.011572 [-5.438148, -4.585167] | 100.00% |

3D CONFIRM: 256 mapas y 5507 inicios, accuracy 98.8759%. Prueba OOD limitada: misma ley de terreno, eje nuevo comparte parámetros con eje1; no demuestra escalabilidad arbitraria n-D.

## Tiempo real y pruebas

DEV completo (16 modelos, incluye generación/evaluación): 1.791259s de pared; CONFIRM: 1.733040s. Suma de fases 3.524299s, sin compilación/estadística/revisión. Son modelos diminutos de nueve parámetros, no entrenamiento de LLM. Linux x86_64, CPU Intel(R) Xeon(R) Processor, rustc1.99.1; tiempos orientativos con CPU compartida, no un benchmark de velocidad.

Evaluar los64 mapas test (todas las rutas) por modelo CONFIRM: campo 3.9170ms; uniforme 3.7379ms; MLP 16.7509ms. Esto mezcla planificación y rollout, NO latencia de una acción ni comparación de cómputo equiparado.

- `cargo test`: **9/9** (gradientes de los9 parámetros del campo y81 del MLP frente a diferencias finitas, SPD, goal clamp, inalcanzables, softmin extremos, dimensión3, reproducción determinista y mejora de aprendizaje). Véanse `tests.log` y `tests_release.log` (9/9 también en release).
- Reentrenamiento completo independiente de B400 con el mismo binario: pesos, rutas, pérdidas de cada época y métricas idénticos al quitar solo columnas temporales (`replay.json`). No afirmar identidad binaria de archivos que incluyen tiempos.
- Oráculo Dijkstra usa costos verdaderos; sus distancias son el denominador de regret. No hay ventaja sobre el oráculo: el regret aprendido no es exactamente cero.

## Qué significa y qué NO significa
1. Es **planificación inversa supervisada sintética**: el generador conoce una ley oculta, no el modelo. El terreno observable [a,b] y topología de obstáculos sí se entregan; costos verdaderos/valores del oráculo no son entradas al predictor. Se rechazan mapas desconectados antes de entrenar/evaluar. No se ha probado una mezcla de metas alcanzables/inalcanzables como tarea de rendimiento, solo como test unitario.
2. La familia del docente está dentro de la familia del estudiante: escenario favorable realizable. No se han probado sensores reales, otra ley física, mapas grandes ni obstáculos dinámicos. Aprender9 coeficientes no demuestra una nueva IA general.
3. El éxito100% depende fuertemente del planificador explícito, positividad de costos y mapas conectados: los controles uniforme y sin entrenar también llegan100%. La mejora aprendida real es el menor costo/NLL, no haber inventado el algoritmo de búsqueda.
4. El MLP tiene81 parámetros nominales y las mismas actualizaciones/etiquetas, pero es reactivo/local y no propaga obstáculos globales. **NO** está equiparado en cómputo ni capacidad receptiva: no usar esta comparación para decir que Soft-Bellman supera las redes neuronales en general.
5. Se imitan acciones, no se supervisan costos absolutos. Métrica y potencial no identificables unívocamente; aumentar escalas también afila probabilidades con temperatura fija. No se demuestra recuperación de geometría verdadera ni estabilidad de entrenamiento ilimitado.
6. Sin entrenamiento de lenguaje ni integración con el campo de estado/LLM periférico/BQ10. No evalúa pérdidas de lenguaje ni conversación.
7. Paso0 hash `1985bdbad5eee41f`, known_acc16 y E33: **NO EVALUADOS**, no aprobados. No se ejecutó ni modificó `run_v6`, ni se leyó/cambió MASTER_SECRET. No merge, no cambios de `main`.
8. Fuente de ecuaciones: enlace Gemini de la propuesta y ecuaciones proporcionadas en el encargo. El fetch adicional de este ejecutor devolvió HTTP500; no se archivó texto completo ni se atribuyen afirmaciones nuevas a esa página.

## Reproducir
Desde raíz del repo (salida nueva para no sobreescribir evidencia):
```bash
nice -n 10 cargo test --manifest-path soft_bellman/Cargo.toml -j 2
nice -n 10 cargo build --release --manifest-path soft_bellman/Cargo.toml -j 2
nice -n 10 python3 soft_bellman/run.py DEV --workers 4 --output /workspace/soft-bellman-repro
nice -n 10 python3 soft_bellman/run.py CONFIRM --workers 4 --output /workspace/soft-bellman-repro
# Semilla individual; argumento numérico decimal 46080 = 0xB400:
nice -n 10 soft_bellman/target/release/soft-bellman-v1 train 46080 CONFIRM /workspace/soft-bellman-replay-new
# Estadística de la evidencia guardada en docs:
nice -n 10 python3 soft_bellman/summarize.py
```
`run.py` comprueba SHA256 congelado para CONFIRM y rechaza sobrescribir entrenamientos completos. `summarize.py` analiza por defecto `docs/soft_bellman_v1/raw`; para una reproducción externa, copiar el script y ajustar DOC a esa salida (sin reemplazar datos originales).

## Evidencia guardada
`raw/{DEV,CONFIRM}/{semilla}/`: `epochs.csv`161 pérdidas antes/después de160 updates, `metrics.csv`16 filas, `weights.txt`, `stdout.log`, `routes.txt`(3 mapas con terreno, paredes y rutas de los4 modelos). `metrics.csv` agregado512 filas; `summary.json` estadísticas completas; `analysis.log` consola de análisis; `FROZEN.json`, `replay.json`, `tests.log`. No se dejaron trabajos en segundo plano.
