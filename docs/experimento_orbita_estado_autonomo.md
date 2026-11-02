# Experimento SA-1 — ÓRBITA aprende dinámica de estados agnóstica al lenguaje

**Rama:** `exp/orbita-state-autonomy`  
**Base:** `main` / CONSTELACIÓN v4  
**Estado:** protocolo y harness inicial; no se integra en `main` hasta superar los gates.

## 1. Pregunta

¿Puede ÓRBITA aprender una ley de transición sobre estados latentes sin que le demos familias geométricas, reglas analíticas ni targets memorizables, y después generalizarla a estados y mecanismos dinámicos no vistos?

La prueba está diseñada para separar cuatro explicaciones que actualmente siguen mezcladas en v4:

1. aprendizaje real de una dinámica de estado;
2. memorización de ejemplos/targets;
3. ventaja específica de la parametrización geométrica de ÓRBITA/CONSTELACIÓN;
4. ventaja debida a la memoria/CDT en vez de a la dinámica.

## 2. Cambio fundamental respecto a v4

El benchmark no usa `apply_rule`, familias geométricas, puntos con significado geométrico ni una clase de hipótesis entregada al modelo.

Cada **episodio** contiene un sistema dinámico oculto de 8 dimensiones. El sistema se genera mediante una transformación aleatoria estable y una pequeña no linealidad. El learner recibe únicamente:

- estado actual `x ∈ R^8`;
- un **fingerprint de mecanismo** `c ∈ R^10`, obtenido de una trayectoria de soporte;
- target `x'`, sólo durante entrenamiento.

El fingerprint no contiene matrices del generador ni el target. En TEST se usan mecanismos e inicializaciones nunca presentes en TRAIN.

Los 8 valores se empaquetan como cuatro pares sólo para reutilizar la representación de entrada de ÓRBITA; no tienen interpretación espacial.

## 3. Hipótesis preregistradas

### H1 — dinámica
ÓRBITA entrenada en SA-1 supera significativamente al control sin núcleo dinámico (`variant=A`) en mecanismos TEST no vistos, en `h=1,4,8,16`.

### H2 — no memoriza targets
El rendimiento de TEST permanece dentro del margen preregistrado al:

- reemplazar el estado inicial por uno nunca visto;
- evaluar mecanismos completamente nuevos;
- alejar TEST de todos los estados TRAIN mediante un umbral de distancia;
- destruir cualquier almacén de episodios antes de TEST.

### H3 — no depende de la familia geométrica
Un benchmark de sistemas aleatorios, sin familias ni reglas conocidas por el learner, mantiene una ventaja sobre los controles estructurales. Si sólo funciona el benchmark geométrico v4, la hipótesis de dinámica general queda refutada.

### H4 — el contexto codifica mecanismo, no respuesta
Permutar `c` entre episodios debe degradar fuertemente el resultado. Si no ocurre, el modelo probablemente ignora el mecanismo y aprende un predictor marginal.

### H5 — CDT no es lookup
Una segunda fase permite consolidar experiencias en CDT, pero TEST se ejecuta con el lookup deshabilitado. La mejora sólo cuenta como consolidación si sobrevive a la eliminación de pares `(x → x')` y mejora mecanismos/estados nuevos.

### H6 — separación lenguaje/core
En una extensión posterior, un encoder LLM congelado produce `c`/estado. ÓRBITA recibe sólo el estado continuo; no recibe tokens ni logits. La métrica primaria sigue siendo predicción del estado futuro, no next-token accuracy.

## 4. Diseño de datos

### TRAIN
- 256 mecanismos.
- 32 episodios por mecanismo.
- estados iniciales aleatorios.
- 4 pasos de soporte + 1 query.

### DEV
- 64 mecanismos distintos.
- mismos rangos, seeds independientes.

### CONFIRM
- 64 mecanismos completamente nuevos.
- seeds congeladas antes de ejecutar DEV.

### TEST-OOD
Tres cortes obligatorios:

1. **new-state:** estado inicial fuera del conjunto de TRAIN;
2. **new-mechanism:** mecanismo nunca visto;
3. **new-composition:** composición de dos operadores latentes que nunca apareció como mecanismo de entrenamiento.

No se reporta únicamente el promedio agregado: cada corte se informa por separado.

## 5. Generador oculto

El mecanismo `m` se genera con seed independiente y produce una transición estable:

`x_{t+1} = tanh(A_m x_t + U_m tanh(V_m x_t) + b_m)`

con `A_m` normalizada para mantener radio espectral estable y rangos fijos para `U,V,b`.

Importante: estas matrices **no se entregan al learner** y no se serializan en los datasets públicos de entrenamiento. Sólo se conserva el seed del episodio para reproducibilidad.

El contexto `c` se obtiene de estadísticas de la trayectoria de soporte: media, media de diferencias, norma de diferencias, dispersión y proyecciones fijas. El learner debe inferir la dinámica útil desde el comportamiento observado, no desde el identificador del generador.

## 6. Controles obligatorios

| Control | Qué falsifica |
|---|---|
| ÓRBITA full | hipótesis principal |
| ÓRBITA A | efecto del núcleo dinámico |
| ÓRBITA B | si basta una matriz libre condicionada |
| ÓRBITA C | efecto del corrector |
| MLP igualado | si la ventaja es sólo capacidad |
| contexto permutado | dependencia real del mecanismo |
| estado barajado | detección de lookup/memoria |
| replay/CDT OFF | atribución a memoria |
| random-context | predictor marginal |

El MLP debe tener parámetros igualados dentro de ±2 %.

## 7. Métricas

Primaria: `median(log relative error)` en `h=16` sobre `new-mechanism`.

Secundarias:

- error por horizonte `1,2,4,8,16`;
- error de un paso;
- estabilidad / divergencia;
- cosine de desplazamiento;
- energía relativa;
- transferencia `new-state` vs `new-mechanism`;
- caída por `context permutation`;
- caída por `CDT OFF`;
- consultas de memoria durante TEST = **0**;
- targets vistos/equivalentes en TRAIN = **0**.

Intervalos: bootstrap pareado por seed, 10 000 réplicas, IC95, misma política que v4. No se reajusta el gate después de mirar CONFIRM.

## 8. Gate de éxito

La hipótesis fuerte sólo se considera apoyada si, en CONFIRM:

1. ÓRBITA full vs MLP igualado tiene IC95 de la diferencia completamente favorable en `new-mechanism h16`;
2. ÓRBITA full supera `variant=A` y el control random-context;
3. `new-mechanism` conserva al menos 50 % del rendimiento relativo de `new-state`;
4. context permutation produce una caída ≥ 30 % de la métrica primaria;
5. TEST tiene 0 consultas CDT/lookup y 0 target-equivalent leakage;
6. el resultado se reproduce con una segunda familia de generadores ocultos no utilizada para ajustar hiperparámetros.

Si falla cualquiera de 1–5, el resultado se etiqueta **NO DEMOSTRADO**, no como éxito parcial.

## 9. Fase CDT-RQM / CONSTELACIÓN

Después de SA-1 se repite exactamente el protocolo con una fase de sueño:

`trayectorias → CDT → consolidación → estado aprendido → ÓRBITA → CONSTELACIÓN`

La CDT puede almacenar experiencias durante WAKE, pero el TEST debe ejecutarse con:

- CDT físicamente vacía o desconectada;
- contador de consultas = 0;
- sin pares de respuesta;
- sin targets TEST.

La variable causal es `consolidation ON/OFF`, no `lookup ON/OFF`.

La pregunta es si la consolidación cambia los parámetros/estado dinámico y produce transferencia a mecanismos nuevos. Si sólo mejora cuando la CDT está consultable, no es consolidación: es recuperación.

## 10. Fase LLM periférica

Sólo después de superar SA-1:

`texto → encoder Gemma/Qwen congelado → estado continuo → ÓRBITA → estado futuro → decoder/verbalizador`

No se entrena tokenización ni next-token prediction. Se evalúa:

- predicción del estado futuro;
- invariancia entre dos formulaciones lingüísticas del mismo estado;
- transferencia entre idiomas si el encoder lo permite;
- ablación del LLM después de producir el estado.

El LLM no puede consultar CDT ni ÓRBITA durante la codificación.

## 11. Interpretación

### Resultado fuerte
ÓRBITA supera controles de capacidad y geometría en mecanismos no vistos, conserva transferencia OOD y la consolidación sigue funcionando sin lookup.

Esto apoyaría la afirmación limitada de que el sistema aprende una dinámica de estados transferible.

### Resultado débil
ÓRBITA funciona sólo en el generador geométrico v4 o sólo con mecanismos conocidos.

Conclusión: la ventaja anterior es específica del benchmark.

### Resultado negativo
ÓRBITA no supera MLP igualado o ignora el contexto.

Conclusión: no hay evidencia suficiente para atribuirle un mecanismo cognitivo de predicción de estados.

## 12. Reproducibilidad

Los seeds DEV/CONFIRM, configuración del generador, hash de datasets, arquitectura, presupuesto y análisis estadístico se congelan antes de CONFIRM. Los scripts de ejecución deben ser reanudables y producir un manifiesto con hashes.

**Principio:** este experimento no intenta demostrar AGI. Intenta demostrar —o refutar— una afirmación mucho más pequeña y falsable: **ÓRBITA puede aprender y extrapolar dinámicas sobre estados latentes sin depender de familias geométricas codificadas ni de recuperación de memoria.**
