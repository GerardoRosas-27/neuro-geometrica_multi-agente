# Plan de implementación — V4 Dφ / Teacher Distillation

**Rama:** `exp/field-autonomy-v4`  
**Estado:** plan de implementación, sin merge a `main`  
**Objetivo:** aumentar la fracción de conocimiento consolidado que Dφ puede ejecutar de forma persistente, antes de introducir Ψ64/Gemma en E43/E44.

## 1. Diagnóstico que motiva el plan

Los resultados E31–E45 muestran que la v4 ya tiene una propiedad importante: la experiencia consolidada puede modificar persistentemente los pesos de Dφ y producir comportamiento en TEST sin consultas de CDT/RQM/tablas/NN/attractor ni memoria directa. E35/E37/E38 pasan en DEV y CONFIRM bajo el protocolo auditado.

El cuello de botella actual es la transferencia de conocimiento: el `LearningSignal` teacher alcanza aproximadamente 0.83, mientras Dφ queda alrededor de 0.34–0.38. La documentación también advierte que el teacher incorpora una clase de hipótesis estructurada y que E34 no demuestra una ventaja de Dφ frente a un MLP de capacidad comparable. E33 falla en horizontes largos.

Por tanto, el siguiente ciclo debe distinguir tres causas posibles:

1. **Capacidad de Dφ insuficiente.**
2. **Pérdida/señal de destilación insuficiente o mal condicionada.**
3. **Inestabilidad de rollout que sólo debe atacarse después de mejorar el error de un paso.**

No se debe mezclar estas tres preguntas en un único experimento.

## 2. Orden de implementación

### Fase A — techo de destilación

Implementar una variante de entrenamiento donde Dφ aprenda directamente la salida del `LearningSignal` teacher sobre el mismo espacio de estados, sin cambiar la arquitectura de inferencia.

Comparar:

- teacher directo;
- Dφ actual entrenado contra teacher;
- Dφ actual entrenado mediante la ruta de consolidación existente.

Propósito: medir el techo que permite la arquitectura de Dφ y separar el problema de representación del problema de consolidación.

**No usar TEST para tuning.** Mantener DEV/CONFIRM y seeds independientes del protocolo existente.

### Fase B — barrido de capacidad

Mantener la misma parametrización residual `Ψ' = Ψ + ΔΨ`, pero evaluar al menos:

- Dφ-962 (baseline actual)
- Dφ-~2k
- Dφ-~4k
- Dφ-~8k

El barrido debe mantener comparables dataset, seeds, optimizer budget y número de pasos. Reportar parámetros, tiempo, memoria, error one-step, accuracy y estabilidad Jacobiana.

**Gate:** no declarar mejora sólo por accuracy; exigir mejora reproducible en ambas fases y reportar coste.

### Fase C — entrenamiento de un paso

Para la mejor capacidad, optimizar explícitamente el error de un paso frente al teacher/operador verdadero. Medir por familia y región del manifold:

- `rel_err_1step`;
- error absoluto;
- coseno con el target;
- distancia al manifold;
- `σ_max(J_Dφ)`;
- error teacher→Dφ.

**Meta:** `rel_err_1step < 5 %` sin introducir lookup ni memoria durante TEST.

### Fase D — estabilidad y horizonte largo

Sólo después de cumplir la meta de un paso, evaluar tres variantes:

1. Dφ puro;
2. Dφ + reproyección al manifold por paso;
3. Dφ + reproyección + restricción/regularización espectral, si la segunda sigue siendo inestable.

Repetir E33 con el mismo protocolo pre-registrado. Medir h1/h2/h4/h8/h16/h32/h64, deriva de norma, energía, distancia al manifold y crecimiento de perturbaciones.

La reproyección debe tratarse como mecanismo de estabilización, no como memoria ni como recuperación de respuestas.

## 3. Criterios de no autoengaño

- No consultar CDT/RQM/tabla/NN/attractor/direct memory durante TEST.
- No almacenar target, equivalente de target ni answer table.
- No retunear con CONFIRM.
- Mantener seeds independientes DEV/CONFIRM.
- Mantener manifests y hashes de provenance.
- Comparar contra MLP de capacidad equivalente cuando corresponda.
- Reportar resultados negativos y variantes que no pasen.
- No llamar a una mejora "aprendizaje emergente" si proviene de una clase de hipótesis diseñada a partir del generador.
- No atribuir una mejora de horizonte largo a la reproyección sin un control Dφ puro.

## 4. Gate principal del ciclo

El objetivo inmediato es aumentar la captura del teacher de ~40 % hacia >70 %, pero el primer gate operativo debe ser:

`Dφ teacher-distillation > baseline en DEV y CONFIRM + leakage=0 + queries=0 + proceso nuevo`

Un resultado fuerte sería aproximadamente:

`teacher ~0.83 → Dφ >=0.70`

No es un umbral universal; es un objetivo experimental para decidir si continuar escalando capacidad.

## 5. Decisión después del ciclo

### Si la capacidad mejora mucho

Continuar con la mejor Dφ, alcanzar `<5 %` one-step y después atacar horizonte largo con reproyección.

### Si el teacher directo es alto pero la ruta de consolidación es baja

Investigar la pérdida/regresión de consolidación y no aumentar indefinidamente Dφ.

### Si incluso el teacher directo queda bajo

El cuello es la representación/capacidad de Dφ; reconsiderar parametrización residual antes de E43/E44.

### Si one-step <5 % pero E33 sigue fallando

Pasar al estudio de manifold/proyección/Jacobiano; no entrenar más epochs como respuesta automática.

## 6. Integración posterior con token-free

Después de resolver el núcleo Dφ, usar el `Ψ64` obtenido en E45 de `exp/token-free-liquid-field` como entrada semántica congelada para E43/E44.

Conservar de token-free:

- core Rust sin tokens;
- exportación de parámetros;
- paridad JAX↔Rust;
- pipeline JAX (`lax.scan`/`vmap`);
- Ψ64 como encoder/bottleneck;
- `liquid_ss` y liquid como controles negativos.

No reabrir la hipótesis liquid como candidato principal salvo nueva evidencia que cambie los resultados E46/E46b/H2H/E47/E48.

## 7. Resultado esperado del camino

La secuencia científica buscada es:

`experiencia → CDT/consolidación → teacher → Dφ persistente → eliminación de memoria → respuesta nueva → estabilidad de rollout → Ψ64/Gemma como periferia`.

El objetivo no es demostrar consciencia, vida artificial ni AGI. El objetivo es aislar causalmente si un sustrato compacto puede absorber conocimiento consolidado y ejecutarlo posteriormente sin recuperar la experiencia original.
