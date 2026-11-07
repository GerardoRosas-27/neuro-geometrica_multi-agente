# Soft-Bellman simplicial 3D v2 — resultados reales

Cierre: 2026-11-06 17:21:21 America/Mexico_City (UTC-6).

## Veredicto
**PASA el criterio primario en este benchmark sintético, pero NO es todavía un solver continuo ni robusto a otra ley de terreno.** Hay un complejo tetraédrico3D auténtico y segmentos ejecutables que atraviesan interiores de caras y volúmenes. La métrica completa mejora a la diagonal en la MISMA discretización y las MISMAS mallas. No se comparan porcentajes de v1-grid con v2-mesh para atribuir mejora.

**Negativos importantes:** la discretización gruesa aún deja una brecha considerable al refinar; al cambiar la ley de impedancia, el modelo entrenado resulta peor que el Euclídeo y que su inicialización. Hodge y Regge son diagnósticos, no componentes entrenados que expliquen el beneficio.

## Trazabilidad y alcance
- Leída la propuesta completa capturada del navegador antes de implementar: `PROPUESTA.md`. Tabla detallada implementado/corregido/diferido en `TRAZABILIDAD.md`.
- Rama `exp/soft-bellman-simplicial-v2` desde v1 `a05c7e9`. Crate aislado `soft_bellman_simplicial/`, sin dependencias; no cambia v1 ni aplicación existente.
- Prerregistro antes de DEV: `4b2c05b2e4da614a584f3df7df6742f4c6b6fe6d`.
- Congelación código/config/DEV antes de CONFIRM: `f694a09`; SHA256 del Rust entrenado `b02e6ab2b3fbc1a530af5ebc1b096c050a64731a95b611e41f7bb1d7dd20a59c`. Ninguna modificación de código entrenado o hiperparámetros después de DEV.
- DEV16/16 semillas0xA400–A40F; CONFIRM16/16 semillas0xB400–B40F. En cada semilla se entrenan4 variantes: completo9 parámetros, diagonal6, edge-only diagonal6, sinPhi6. Son128 trayectorias de optimización en total (32 semillas×4), no un solo entrenamiento compartido.
- Por semilla:12 train,4 validación,24 test,8 OOD geométricos,8 ley cambiada; streams independientes. Cada modelo aprende de todos los estados de su grafo salvo meta, NLL de acciones Dijkstra. Se evalúan rutas desde los mismos vértices originales en todos los modelos. La variante edge-only usa etiquetas de su oráculo restringido (no puede ejecutar acciones interiores), por lo que su NLL/accuracy no comparte dominio con la red Steiner.
- 120 updates Adam full-batch lr.03, beta.9/.999 eps1e-8, clip10, init normal.1, final sin selección de checkpoint; validación solo descriptiva. Generación, geometría y TODO entrenamiento/gradiente en Rust f64. Python solo lanza procesos y calcula estadística.

## Qué implementa exactamente
Freudenthal6 tetraedros/celda; perturbaciones coordenadas ±.06 y18% tetraedros bloqueados; cierre de caras/aristas activo y matrices frontera orientadas. Train/test2×1×1 cubos; OOD2×2×1 con perturbaciones±.09. Se rechazan inversión de tetraedros, vértices aislados y desconexión, así que éxito de alcanzar meta no mide detección de objetivos imposibles.

La métrica es homogénea global G=LLᵀ+.01I con6 pesos: diagL=.5+softplus(w), subdiagL=.8tanh(w). Esto impone longitudes compartidas y factibles para todas las caras/tetraedros, pero restringe a geometrías planas globalmente embebibles. No longitudes independientes ni curvatura aprendida. Phi por cara=.3+softplus(cabeza3 sobre [1,a,b]); costo de segmento=longitudSPD×impedancia de su portador; volumen usa media de4 caras y arista media de caras incidentes activas.

Estados originales+baricentros de caras+baricentros de tetraedros. Una acción por pareja de estados co-contenidos en un tetraedro activo, sin duplicados. Unión de soportes determina si el segmento cruza una cara o un volumen. La ruta es una polilínea3D realmente ejecutable, no un valor baricéntrico convertido arbitrariamente en salto a vértice. La distancia de cada segmento equivale a la fórmula de longitudes baricéntrica de la propuesta, pero el minimizador lambda se aproxima con puntos Steiner: NO se implementa min continuo exacto.

Soft-Bellman estable tau.03 y horizonteH=N estados; V0=∞ salvo meta0, meta fijada0 en todas las capas. Reverse-mode completo de las capas hacia los9 parámetros (derivadas de costo analíticas). Inferencia greedy de Q suave con horizonte decreciente, no muestreo de política; persiste sesgo entrópico por grado, sin afirmación de estabilidad estacionaria.

## CONFIRM: resultado principal en mallas nuevas
Regret=costo verdadero ruta / Dijkstra Steiner grueso −1. Se muestra como porcentaje del costo del oráculo; fracaso se penaliza5 y se reporta éxito separadamente.

| Modelo | Parámetros entrenados | Sobrecosto | Llega a meta | NLL acción nat |
|---|---:|---:|---:|---:|
| Completo SPD + Phi | 9 | 0.010733% | 100.00% | 0.137028 |
| Diagonal + Phi, mismo Steiner | 6 | 0.814048% | 100.00% | 0.484983 |
| Diagonal solo1-esqueleto | 6 | 8.375407% | 100.00% | 0.215390* |
| Completo sinPhi | 6 | 2.259997% | 100.00% | 0.730010 |
| Completo sin entrenar | 0 | 2.420714% | 100.00% | 1.256641 |
| Euclídeo, mismo Steiner | 0 | 2.484227% | 100.00% | 1.095485 |
| Dijkstra de costos verdaderos | 0 | 0.000000% | 100.00% | 0.000000* |

Contraste primario completo−diagonal: **-0.803314 puntos porcentuales**, IC95% **[-0.860493, -0.746124]**, sin disminuir éxito. Bootstrap pareado10000, seed12345, delta.05, percentiles interpolados. Unidad independiente:16 semillas, no miles de estados correlacionados. Solo este contraste es primario; resto exploratorio sin multiplicidad.

CONFIRM: 384 mallas, 4224 inicios originales de ruta y 19848 estados Steiner no-meta etiquetados. Accuracy de acción completo 91.4590% vs diagonal 80.2819%. NLL test antes 1.256641 → después 0.137028; NLL train antes 1.207586 → después 0.129996.

Fracción promedio por ruta de segmentos con portador cara/tetraedro: **47.8983%** en completo;0% en1-esqueleto. No es porcentaje de área/volumen explorado. No se atribuye la diferencia respecto de edge-only solo a SPD: cambian también acciones disponibles y etiquetas restringidas.

\* NLL edge-only usa otros estados/acciones; no es una comparación directa de calidad con NLL Steiner. NLL0/accuracy100% del oráculo son por definición del docente determinista, no resultado de aprendizaje. Todos los controles llegan100%; aprender mejora costo, no inventa búsqueda ni resuelve metas desconectadas.

## DEV y OOD geométrico

| Fase/split | Regret completo | Regret diagonal | Δ pp [IC95%] |
|---|---:|---:|---:|
| DEV/test | 0.009543% | 0.831425% | -0.821882 [-0.892039, -0.755627] |
| DEV/ood_geometry | 0.009159% | 1.026295% | -1.017136 [-1.112157, -0.934636] |
| CONFIRM/ood_geometry | 0.011303% | 1.001995% | -0.990692 [-1.131117, -0.864580] |

OOD geométrico CONFIRM: 128 mallas, 2176 inicios; éxito100%, misma ley de costos y tamaño todavía pequeño. No es una evaluación en grandes complejos o topologías arbitrarias.

## Negativo: cambio de ley de impedancia, sin adaptar

La cabeza Phi aprendida es mal especificada: el docente multiplica por un factor positivo no lineal `1+.45 sin(7a)+.8(b−.5)²`. Se generan mallas nuevas sin ajuste de parámetros sobre este split.

| Modelo CONFIRM | Sobrecosto | NLL |
|---|---:|---:|
| full | 9.280995% | 4.868354 |
| diagonal | 9.986338% | 2.059732 |
| no_phi | 5.015508% | 1.078162 |
| untrained | 5.405981% | 1.883326 |
| euclidean | 5.278389% | 1.612948 |

**El completo entrenado empeora frente al Euclídeo**: Δ **+4.002606 pp**, IC95% [3.285793, 4.807155]. También es peor que sinPhi y su inicialización. No se demuestra robustez a cambios de física/terreno; el positivo principal depende de la ley realizable.

## Refinamiento: todavía NO geodésica continua

Diagnóstico preregistrado sobre3 mallas test por semilla (48 CONFIRM), sin tuning ni entrenamiento fino: Dijkstra Steiner grueso cuesta en promedio **7.4202% menos** que Dijkstra limitado al1-esqueleto, ambos con la misma ley verdadera. Al añadir puntos medios de aristas y puntos faciales(2/3,1/6,1/6), el oráculo refinado cuesta **11.0605% menos** que el grueso. Son medias por malla de reducciones de costo medio, no del regret principal.

Esta brecha de refinamiento es grande: el regret casi cero frente al oráculo grueso NO significa casi óptimo continuo. Dos resoluciones solo demuestran dependencia de discretización y monotonicidad al conservar acciones, no convergencia. El siguiente trabajo útil es refinar/optimizar posiciones de cruce y evaluar el costo extra antes de añadir física de flujo o curvatura.

En esas48 mallas: [7.0, 12.0] tetraedros activos; estados gruesos [43.0, 58.0], acciones [452.0, 702.0]; refinado estados [143.0, 193.0], acciones [4542.0, 7182.0]. Las acciones crecen aproximadamente un orden de magnitud.

## Hodge, Regge y factibilidad: sin atribución inflada

CM Vol² mínimo registrado (muestra48 CONFIRM): 0.025977178003; transición aprendida mínima registrada 0.163192845886. Déficits de bisagras interiores y error de auto-adjunción Hodge por debajo de resolución1e-12 del CSV (no ceros simbólicos probados por esos decimales). Regge≈0 es esperado por la métrica global plana, NO curvatura que la red aprendió a regularizar.

El Hodge1 dimensionado correctamente es `B1ᵀ M0⁻¹ B1 M1 + M1⁻¹ B2 M2 B2ᵀ`, con masas lumped positivas, no dual circumcéntrico exacto. Se prueba auto-adjunción ponderada, PSD y covariancia de orientación. No se usa en el objetivo de rutas y no hay canales ocultos de cocadenas. La barrera CM no se optimiza: la parametrización SPD garantiza realizabilidad y evita fingir que una barrera impide cualquier paso inválido. FlujoB/holonomía y Regge entrenable DIFERIDOS con razones en TRAZABILIDAD.md.

## Pruebas, reproducción y tiempo
- 11/11 tests base en debug y release. Auditoría adicional3 tests sobre copia del código congelado:14/14 en total (`tests_orientation.log`), sin modificar el Rust entrenado.
- Gradientes finitos de todos los parámetros activos de las4 variantes; fronteras B1B2=B2B3=0, coherencia orientada3D, Hodge ponderado/PSD/orientación/gauge, CM/SPD, distancia baricéntrica/permutaciones, Regge interior con frontera excluida, bloqueo/inalcanzables, refinamiento ejecutable, no-leakage de etiquetas y mejora de aprendizaje.
- Replay completo B400 de los4 modelos: pérdidas, geometría, pesos, rutas y métricas idénticos tras retirar únicamente campos de tiempo (`replay.json`). No se afirma identidad de archivos temporizados completos.

DEV16 paquetes de4 modelos: 9.598697s de pared; CONFIRM: 9.349268s; suma 18.947965s. Incluye generación/evaluación/diagnósticos, excluye compilación/estadística/revisión. Cuatro procesos nice10, CPU Intel(R) Xeon(R) Processor, Linux x86_64, rustc1.99.1. La tarea termina; no quedan entrenamientos en segundo plano.

Evaluación de24 mallas test por modelo CONFIRM (planificación+rollout de todos los inicios): completo 10.5311ms, diagonal 9.2880ms, edge-only 0.4856ms. No son latencias de una acción ni cómputo equiparado entre discretizaciones. O(H(N+E)) tiempo/tape conH=N.

```bash
nice -n 10 cargo test --manifest-path soft_bellman_simplicial/Cargo.toml -j 2 -- --test-threads=2
nice -n 10 python3 soft_bellman_simplicial/extra_audit.py
nice -n 10 cargo build --release --manifest-path soft_bellman_simplicial/Cargo.toml -j 2
nice -n 10 python3 soft_bellman_simplicial/run.py DEV --workers 4 --output /workspace/simplicial-repro-new
nice -n 10 python3 soft_bellman_simplicial/run.py CONFIRM --workers 4 --output /workspace/simplicial-repro-new
nice -n 10 python3 soft_bellman_simplicial/summarize.py
```
El agregador lee docs/raw por defecto; para datos reproducidos externos usar una copia del script con DOC apuntando a esa salida, sin sobrescribir evidencia original. Los32 directorios raw conservan121 pérdidas por modelo, pesos,35 filas de métricas, rutas/coords3D, geometría y logs. `metrics.csv`1120 filas, `summary.json`ICs completos, `geometry_summary.json`refinamiento, pruebas y source hash.

## Límites y siguiente paso
Docente dentro de familia aprendible en split principal, supervisión por oráculo y terreno/topología observables. No recuperar necesariamente la métrica física única: se imitan acciones y escalas pueden agudizar NLL con temperatura fija. Falta IO de malla externa y API pública estable; es un ejecutable científico aislado, no integración lista para producción. Falta solver continuo/refinamiento aprendido, geometría local curva y dinámica/cochain learning. No lenguaje ni integración BQ10/LLM periférico. Paso0 hash1985bdbad5eee41f, known_acc16/E33 NO EVALUADOS; main/run_v6/secretos no modificados; no PR ni merge.
