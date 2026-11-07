# Propuesta → implementación / corrección / diferido

Se leyó PROPUESTA.md COMPLETO antes del prerregistro. Las afirmaciones de Gemini no se aceptan como teoremas ni como autoridad para actuar.

| Componente de propuesta | Estado real v2 | Motivo / alcance |
|---|---|---|
| Complejo orientado de vértices/aristas/caras/tetraedros | Implementado | Freudenthal6 tetraedros/celda, cierre activo tras bloquear volúmenes, fronteras B1/B2/B3; B1B2=B2B3=0 y cadena3D coherente. |
| Longitudes de arista aprendibles e intrínsecas | Corregido/restringido | Longitudes inducidas por G=LLᵀ+.01I global con6 parámetros. Todas las caras compartidas tienen la misma longitud y cada tetraedro es realizable. No son pesos independientes por arista, ni modelo puramente libre de coordenadas. |
| Phi positiva por cara | Implementado/restringido | Cabeza compartida3 parámetros sobre terreno de cada cara; generaliza a caras nuevas. Tetraedros usan media de sus4 caras; aristas media de caras activas incidentes. Esta extensión a un medio estratificado está definida por el prototipo, no se atribuye como fórmula exacta del texto fuente. |
| Cruce de interior triangular/tetraédrico | Implementado, aproximado | Estados Steiner de baricentros y segmentos contenidos en tetraedros. Soportes de estados determinan portador geométrico; rutas son polilíneas3D ejecutables y costeadas segmento por segmento. No mero renombrado de grid3D. |
| Fórmula baricéntrica de distancia | Implementado/verificado | Norma SPD de segmentos equivale exactamente a fórmula de longitudes para los puntos muestreados; test de identidad y permutación. |
| min continuo analítico en lambda / V interpolado | Diferido | Optimización aproximada por nodos Steiner, no minimizador continuo ni derivada del teorema de envolvente. Se evita confundir lambda Vj+(1−lambda)Vk (continuación estocástica/interpolada) con ruta determinista que después elige arbitrariamente un vértice. |
| Soft-Bellman diferenciable | Implementado/modificado | Recurrencia sobre estados aumentados con H=N, temperatura.03, objetivo NLL de acciones del oráculo; gradientes analíticos completos. Meta fijada0 y estados inalcanzables∞. No punto fijo garantizado ni diferenciación implícita. |
| Entropía / duplicación de caras | Corregido parcialmente | Una acción por pareja de estados sin duplicados. Horizonte finito y costos positivos, pero no normalización por grado: persiste sesgo entrópico a tau>0. Evaluación es rollout greedy de Q suave con horizonte decreciente, no muestreo; NLL mide distribución suave. |
| Hodge1 | Operador diagnóstico implementado; NO entrenado | L1=B1ᵀ M0⁻¹ B1 M1 + M1⁻¹ B2 M2 B2ᵀ. Usa tamaños coherentes y masas positivas lumped, no star circumcéntrico exacto. Auto-adjunto en producto M1 y PSD: <h,Lh>_M1=||B1 M1h||²_M0inv+||B2ᵀh||²_M2. No hidden features de cocadenas, separación armónica aprendida ni beneficio causal atribuido. |
| Stars dependientes de métrica | Aproximación diagnóstica | M0 suma vol/4; M1 suma vol/(6*l²); M2 suma vol/(4*area²). Un lumping dimensionalmente consistente y positivo, NO cálculo de volúmenes duales circumcéntricos. |
| Regge | Diagnóstico implementado; regularización diferida | Se calculan diedros métricos y déficit2pi SOLO para aristas que no pertenecen a caras frontera. Bajo esta métrica global embebible el interior es plano: déficit≈0 por construcción. No se afirma que haya aprendido curvatura. |
| CM / barrera log de volumen | Positividad y determinante implementados; barrera diferida | SPD global garantiza factibilidad sin confiar en que barrera corrija pasos inválidos. No se optimiza −logVol² (podría inflar escala); no hay métricas intrínsecas curvas independientes. |
| B=d1 A y .5B en Q | Diferido explícitamente | Falta una acción orientada con gauge/holonomía y control de ciclos coherentes. Añadir un escalar de cara sin derivación no demuestra transporte magnético ni direccional. |
| Backprop≡DDP, Snell exacto, estabilidad garantizada, n-D arbitrario | NO afirmado | No se deducen de estos experimentos. Comparación se limita a benchmark sintético3D realizable y perturbaciones declaradas. |

## Coste y alcance del sustrato
Aunque se utiliza una recurrencia de grafo para computar, el grafo de estados NO es el1-esqueleto del complejo original: incluye puntos de caras/tetraedros y segmentos que cortan sus interiores. Sigue siendo una discretización de dimensión finita, no un solver continuo HJB/fast marching exacto. Las operaciones de cocadenas constituyen soporte topológico/diagnóstico separado de la red de rutas; no afirmar razonamiento de orden superior aprendido solo por disponer de matrices de incidencia.

La métrica homogénea representa seis parámetros geométricos globales, no una métrica local arbitraria. Impedancias faciales crean canales espaciales heterogéneos; caminos volumétricos usan el promedio de caras especificado. Las caras bloqueadas no existen salvo cuando son frontera de un tetraedro activo. Segmentos nunca atraviesan el interior de un tetraedro retirado.

La referencia principal es Dijkstra en el grafo Steiner grueso, NO geodésica continua. Refinar añade midpoints de aristas y puntos de cara (2/3,1/6,1/6), conservando acciones y costes previos; se mide brecha restante sin entrenar ni elegir configuración sobre ella. Dos resoluciones no prueban convergencia.
