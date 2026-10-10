# Token-Free Liquid Field — nueva línea experimental

## Propósito

Esta rama investiga una hipótesis separada de `field-autonomy-v4`: después de una codificación semántica inicial, la inferencia puede ejecutarse sobre un estado de campo compacto sin consultar tokens, logits ni el LLM durante el hot path.

**Restricción de arquitectura:** el core de inferencia del proyecto permanece en Rust. No se reemplazará por Python ni se añadirá una dependencia de TensorFlow/JAX/PyTorch al binario de producción solo para entrenar. Se reutilizarán frameworks existentes como herramientas de entrenamiento y experimentación, y se evaluará la exportación de parámetros/operaciones a Rust.

La rama no modifica `main` ni v4. Su propósito es producir evidencia independiente que posteriormente pueda compararse con `main` + v4.

## Rama y aislamiento

- Rama: `exp/token-free-liquid-field`.
- Base: `main` en el momento de crear la rama.
- No reutilizar checkpoints, pesos aprendidos, datasets sellados ni resultados favorables de v4.
- Se puede reutilizar infraestructura genérica y código numérico existente si se documentan origen, versión y equivalencia.
- La futura comparación contra v4 será una evaluación común sellada; no una mezcla anticipada de implementaciones.

## Hipótesis central

`texto → encoder lingüístico → FieldEncoder → Ψ₀ → Liquid Core Rust → Ψ₁…Ψₙ → decoder`

Después de obtener `Ψ₀`:

- el Liquid Core solo recibe estado de campo, contexto estructurado permitido y parámetros;
- no acepta texto, token IDs, tokenizer, logits ni acceso al LLM;
- no calcula probabilidades de todos los tokens para explorar futuros;
- el decoder se ejecuta al final y no puede actuar como tabla de respuestas.

Si Gemma es el encoder, sigue procesando tokens dentro de sí mismo. “Token-free” significa que los tokens no entran al sustrato de campo después del encoder, no que el LLM sea token-free.

## Arquitectura de software: entrenamiento separado de ejecución

### Camino de entrenamiento

`dataset + manifest → Python/JAX/Keras (o PyTorch) → entrenamiento/validación → exportación de parámetros → artefacto versionado`

### Camino de inferencia

`entrada codificada → FieldEncoder compatible → estado Ψ → core Rust → estado final → decoder`

El core Rust es la implementación de referencia de inferencia. El framework Python es un laboratorio para entrenar y comparar funciones candidatas; no debe permanecer en el hot path de producción.

Los artefactos exportados deben incluir:
- arquitectura y versión del esquema;
- dimensiones, orden de tensores, dtype y convención de layout;
- parámetros y SHA-256;
- versión del framework y del exportador;
- seed, configuración, manifest de datos y checksum de cada split;
- tolerancia numérica de equivalencia;
- reporte de pruebas de equivalencia Python/Rust.

No se aceptará “portar pesos” sin especificar semántica de operaciones, activaciones, normalización, orden de actualización y tratamiento de estados recurrentes.

## Frameworks recomendados

### Primera opción de investigación: Keras 3 + backend JAX

- Keras 3 sirve para describir capas y bucles de entrenamiento de forma compacta.
- JAX permite autodiferenciación, compilación XLA y experimentación numérica con funciones de estado.
- Utilizarlo para prototipos de FieldEncoder, dinámica recurrente, pérdidas, ablations y controles.
- No asumir superioridad de rendimiento; medir entrenamiento y exportabilidad.

### Alternativas

- PyTorch: alternativa si las capas personalizadas, el ecosistema o la depuración experimental resultan más sencillos.
- TensorFlow: alternativa válida si su exportación o herramientas de despliegue se ajustan mejor.
- Rust nativo: referencia de inferencia, benchmarks de latencia y eventual aprendizaje local si se implementan y validan actualizaciones numéricas en Rust.

No mantener simultáneamente implementaciones redundantes sin necesidad. Elegir un backend de investigación inicial y fijar versiones en el entorno reproducible. Los controles pequeños pueden implementarse directamente en Rust cuando eso reduzca la complejidad.

## Base teórica

### 1. Estado latente como interfaz

El FieldEncoder comprime una activación de alta dimensión a `Ψ ∈ R^N`:

`Ψ₀ = Eφ(h_encoder(x))`

La meta no es reconstruir cada detalle lexical, sino retener información suficiente para relaciones, transformaciones y predicciones. Dimensiones iniciales: 32, 64, 128; controles más amplios en E45.

### 2. Dinámica recurrente / líquida

`Ψ(t+1) = Ψ(t) + Δt · Fθ(Ψ(t), c(t))`

Una variante continua es `dΨ/dt = Fθ(Ψ,c,t)`. “Líquida” debe referirse a una dinámica dependiente del estado/contexto o a un mecanismo de adaptación definido, no simplemente a repetir un MLP.

Liquid Time-Constant Networks son referencia conceptual para dinámicas continuas con escalas temporales dependientes del estado. Los SSM/selective-state models como Mamba son referencias de ingeniería para estudiar recurrencia eficiente. No se presupone que ninguna de estas familias sea automáticamente superior.

### 3. Predicción en el espacio de campo

El núcleo predice estados `Ψ_t → Ψ̂_{t+1}`. La salida lingüística se decodifica al final. Debe medirse si esto reduce coste sin sacrificar composición, OOD y calidad semántica.

### 4. Sparsity

`A_t = TopK(|Ψ_t|, k)`

Solo se actualizarán componentes seleccionados si el coste de selección no cancela el ahorro. Comparar con la versión densa en el mismo Rust runtime y con calidad emparejada.

### 5. Memoria asociativa opcional

La memoria asociativa puede alterar el estado o la trayectoria, pero no devolver respuestas directas. Nearest-neighbor y lookup se conservan como controles negativos/positivos, no como prueba de dinámica aprendida.

### 6. Plasticidad y consolidación

Separar:
- hot path: `Ψ → Dθ → Ψ'`;
- learning/sleep path: `experiencias → consolidación → Δθ / señal de aprendizaje`.

El gate más importante exige que, después de consolidar, borrar episodios y almacenamiento de respuestas no elimine la mejora en estados nuevos.

## Qué no se afirma todavía

No se afirma conciencia, vida artificial, inteligencia general, causalidad física del campo, superioridad frente a Transformers ni suficiencia universal de una representación compacta. Se investiga si un sustrato dinámico externo puede aprender y ejecutar transformaciones con mejor relación capacidad/coste.

## Principio experimental

Cada experimento debe documentar:
1. qué datos entran al encoder y al core;
2. qué recursos están prohibidos durante el hot path;
3. arquitectura, parámetros y operaciones;
4. latencia separada de encoder, core y decoder;
5. calidad seen/OOD/composición/rollout;
6. pasos y nodos activos;
7. memoria y coste de entrenamiento/consolidación;
8. equivalencia numérica entre modelo entrenado y core Rust;
9. semillas y procedencia;
10. si la velocidad procede de computación reducida o de memorizar respuestas.

## Familias E45–E57

- E45: bottleneck token-free.
- E46: Liquid vs MLP recurrente vs SSM.
- E47: inferencia dispersa.
- E48: stopping adaptativo.
- E49: coarse-to-fine.
- E50: beam geométrico.
- E51: memoria asociativa.
- E52: plasticidad online.
- E53: consolidación vs inferencia.
- E54: kill test del acceso al LLM.
- E55: encoder swap.
- E56: escalabilidad.
- E57: comparación futura contra v4.

## Criterios generales de aceptación

Un resultado solo es `PASS` cuando tiene splits limpios, manifest reproducible, provenance, controles, métricas de calidad y coste, múltiples seeds, y auditoría de acceso a memoria. Una mejora de velocidad con degradación no controlada de calidad será `PARTIAL`.

Para cualquier modelo entrenado fuera de Rust se requiere una prueba de equivalencia de ejecución: mismas entradas y parámetros, error máximo y medio dentro de tolerancias fijadas antes del test, y mismos resultados discretos cuando corresponda. Si la equivalencia falla, no se atribuyen resultados del prototipo al core Rust.

## Descubrimientos buscados

- **Bottleneck suficiente:** 32–128 dimensiones retienen relaciones relevantes con menor coste.
- **Dinámica operacional:** el núcleo compone reglas en estados nunca observados sin lookup.
- **Sparsity útil:** menos cómputo real a calidad equivalente.
- **Inferencia adaptativa:** consultas fáciles convergen pronto y las difíciles reciben más pasos.
- **Consolidación plástica:** experiencia modifica persistentemente el comportamiento; al borrar episodios, se conserva mejora causal.
- **Portabilidad:** los parámetros entrenados con framework producen comportamiento equivalente en Rust.
- **Independencia del LLM:** tras Ψ₀, el core funciona sin llamadas lingüísticas.

## Comparación futura con v4

Solo después de cerrar v4: dataset común sellado, mismos seeds/tareas/decoder comparable, configuraciones congeladas, sin ajuste posterior al test. Competidores mínimos: v4 Dynamic Field, Liquid Rust, Liquid sparse, Liquid plástico/consolidado, baseline LLM, MLP recurrente y SSM.

Métricas: accuracy, OOD, composición, estabilidad a largo horizonte, p50/p95/p99, active_nodes, pasos, memoria, coste de entrenamiento/consolidación, transferencia y persistencia.

La decisión de merge depende de evidencia reproducible, no del objetivo de la rama.
