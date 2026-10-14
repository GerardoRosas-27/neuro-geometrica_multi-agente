# Protocolo de ablaciones y extrapolación — ÓRBITA

**Rama:** `exp/orbita`  
**Estado:** protocolo posterior a DEV/CONFIRM, sin modificar los resultados ya publicados.  
**Objetivo:** determinar qué componente de ÓRBITA produce la ventaja observada en horizontes largos y comprobar si la ventaja corresponde a una dinámica aprendida o a una combinación accidental de componentes.

> **Regla principal:** este documento fija las pruebas antes de ejecutar las nuevas corridas. No se deben ajustar hiperparámetros usando CONFIRM. Cualquier tuning adicional debe utilizar semillas de tuning nuevas y quedar registrado antes de volver a ejecutar CONFIRM.

---

## 1. Hipótesis que se quieren separar

Los resultados actuales muestran que ÓRBITA obtiene una ventaja clara frente a v4 E3 y MLP M3 en `h8` y `h16`, especialmente en CONFIRM, aunque su accuracy de un paso es ligeramente inferior. El objetivo de este protocolo es separar cuatro posibles explicaciones:

1. **La ventaja proviene del núcleo dinámico `K`.**
2. **La ventaja proviene solamente del corrector de Brenier.**
3. **La ventaja requiere específicamente la estructura espectral/semigrupo de `K`; una matriz aprendida libre no debería reproducirla automáticamente.**
4. **La ventaja es extrapolable más allá del horizonte usado durante entrenamiento.**

La comparación debe conservar, en lo posible, el mismo dataset sellado, semillas, presupuesto de entrenamiento y métricas del experimento principal.

---

# 2. Experimento A — Sin núcleo `K` (corrector-only)

## Pregunta

> ¿El corrector por sí solo explica la mejora de ÓRBITA?

## Construcción

Eliminar completamente la transformación dinámica del núcleo:

```text
z → corrector de Brenier → salida
```

No debe existir `Q`, bloques rotacionales, radios `ρ`, `A`, ni aplicación de `K^h`.

El corrector conserva la misma arquitectura utilizada por ÓRBITA-v1:

- mismo número de iteraciones `r = 2`;
- mismo `η = 0.5`;
- mismas dimensiones;
- misma inicialización/protocolo de optimización.

Si eliminar `K` cambia sustancialmente el número de parámetros, se debe reportar explícitamente la diferencia. **No agregar capacidad artificial para igualar parámetros** sin registrarlo como una segunda variante.

## Entrenamiento

Mantener:

- mismo TRAIN;
- mismo TEST;
- misma separación TRAIN/TEST;
- mismas semillas DEV/CONFIRM;
- mismo número de pasos: `256 000`;
- batch `32`;
- Adam;
- `lr 3e-3 → 3e-5` coseno;
- mismo protocolo de auditoría.

La pérdida multi-horizonte debe adaptarse de forma coherente: al no existir `K^k`, la variante no debe recibir información equivalente mediante un mecanismo alternativo.

## Métrica principal

Comparar contra ÓRBITA completa en:

```text
acc1
h1
h2
h4
h8
h16
h32
h64
región E33
```

## Interpretación

### Si corrector-only ≈ ÓRBITA

La ventaja podría provenir principalmente del corrector y no de la dinámica espectral.

### Si corrector-only << ÓRBITA

Evidencia a favor de que `K` aporta una parte esencial de la estabilidad temporal.

---

# 3. Experimento B — Núcleo `K` libre

## Pregunta

> ¿La ventaja procede de tener cualquier transformación aprendida, o específicamente de imponer la estructura espectral/semigrupo?

## Construcción

Sustituir:

\[
K_h = e^{hA}
\]

por una transformación aprendida libre `K` con presupuesto paramétrico comparable.

La variante debe conservar:

```text
encoder → K libre → corrector → decoder
```

pero eliminar explícitamente las restricciones:

- `Q` ortogonal;
- bloques rotacionales;
- radios derivados de `s`;
- construcción exponencial;
- relación analítica `K_a K_b = K_{a+b}`.

### Importante

No se permite implementar `K` libre y después imponer accidentalmente una parametrización equivalente a la original. La matriz debe ser realmente libre.

## Dos variantes recomendadas

### B1 — `K` libre por paso

Una matriz aprendida aplicada iterativamente:

\[
z_{t+h}=Kz_t
\]

### B2 — `K_h` aprendido por horizonte

Sólo si el coste experimental lo permite, aprender transformaciones independientes para los horizontes evaluados. Esta variante es más potente pero rompe deliberadamente la hipótesis de semigrupo y sirve como control negativo de la estructura.

B1 debe ser la prueba principal.

## Métrica principal

Comparar:

```text
ÓRBITA
K libre
MLP M3
v4 E3
```

en `h1…h64`.

## Interpretación

### Si K libre ≈ ÓRBITA

La ventaja podría deberse simplemente a disponer de un núcleo dinámico explícito.

### Si ÓRBITA > K libre, especialmente en h8/h16+

Esto constituye evidencia a favor de que la **estructura impuesta sobre K**, y no sólo su existencia, aporta estabilidad.

---

# 4. Experimento C — Sin corrector

## Pregunta

> ¿Puede el núcleo estructurado por sí solo producir la ventaja?

## Construcción

Eliminar completamente el corrector:

```text
encoder → K_h → decoder
```

Conservar exactamente el núcleo ÓRBITA:

\[
K_h = e^{hA}
\]

y la misma parametrización espectral.

Esta variante es la ablación más directa de la hipótesis:

> `K` aprende la dinámica global y el corrector aprende solamente el residuo.

## Métricas

Reportar:

- accuracy de un paso;
- error relativo;
- E33;
- región;
- `h1, h2, h4, h8, h16, h32, h64`.

## Interpretación

### Si sin corrector conserva gran parte de la ventaja

La estructura dinámica es probablemente el principal mecanismo responsable.

### Si sin corrector colapsa pero ÓRBITA completa funciona

El resultado apoya una interpretación de **dinámica global + corrección residual**.

### Si sin corrector supera a ÓRBITA completa

Debe investigarse si el corrector está introduciendo sobreajuste o inestabilidad.

---

# 5. Experimento D — Extrapolación fuera del horizonte entrenado

## Pregunta principal

> ¿ÓRBITA aprendió una ley de evolución o simplemente aprendió los horizontes utilizados durante entrenamiento?

Esta es la prueba más importante del protocolo.

## Entrenamiento

No modificar el protocolo principal salvo que se cree una variante explícita de entrenamiento.

La configuración base continúa utilizando los horizontes registrados en el pre-registro:

```text
{1, 2, 4, 8}
```

La evaluación debe extenderse a:

```text
1, 2, 4, 8, 16, 32, 64, 128, 256
```

Los horizontes `128` y `256` **no deben aparecer como objetivos de entrenamiento**.

## Prueba adicional: horizontes no potencia de dos

Para distinguir extrapolación real de memorización de los puntos multi-horizonte, añadir:

```text
3, 6, 12, 24, 48, 96, 192
```

si la implementación lo permite sin alterar la definición de la tarea.

## Métrica

Construir una curva:

\[
E(h)
\]

para cada modelo.

Además de accuracy, registrar:

- error relativo;
- norma del estado;
- región E33;
- tasa de colapso;
- divergencia entre trayectoria predicha y teacher;
- estabilidad numérica.

## Resultado crítico

La hipótesis fuerte sería:

```text
ÓRBITA mantiene un régimen útil de predicción
más allá de h16 y degrada más lentamente que
los baselines y las ablaciones.
```

No se debe exigir que ÓRBITA mantenga accuracy alta indefinidamente. Lo relevante es la **tasa de degradación**.

---

# 6. Diseño estadístico

Para mantener continuidad con el experimento original:

### DEV

```text
0xA400 – 0xA40F
```

### CONFIRM

```text
0xB400 – 0xB40F
```

No cambiar semillas entre variantes salvo que el coste computacional obligue a crear un protocolo explícito de replicación.

Para cada comparación principal reportar:

- media;
- mediana;
- desviación estándar;
- IC95 bootstrap pareado;
- número de semillas donde gana cada variante.

Bootstrap:

```text
10 000 remuestreos
seed = 12345
unidad = semilla
```

El bootstrap debe calcular diferencias por semilla, no diferencias entre medias independientes.

---

# 7. Control de parámetros y cómputo

Para cada variante registrar:

```text
parameters
training_steps
batch_size
optimizer
learning_rate
wall_time
FLOPs estimados
```

No interpretar una mejora como arquitectónica si proviene exclusivamente de mayor presupuesto de cómputo.

ÓRBITA-v1 tiene `2602` parámetros frente a `2590` de E3/M3. Esta diferencia (~0.46 %) debe mantenerse como referencia.

Si una ablación requiere más parámetros por su formulación, reportarlo y, si es viable, añadir una variante con presupuesto igualado.

---

# 8. Auditoría obligatoria

Cada nueva variante debe conservar la auditoría del experimento original:

- `leakage = 0`;
- `queries = 0`;
- TEST sellado;
- flags del auditor registrados;
- proceso nuevo reproducible;
- mismo dataset sellado;
- ningún acceso directo a targets de TEST;
- ningún acceso a CDT/RQM/replay prohibido.

Para Rust ↔ PyTorch, mantener la comprobación numérica cuando la variante tenga implementación equivalente en ambos lados.

---

# 9. Matriz final de ablaciones

| Variante | K estructurado | K libre | Corrector | Objetivo |
|---|---:|---:|---:|---|
| ÓRBITA completa | ✓ | — | ✓ | referencia |
| A — corrector-only | — | — | ✓ | medir aporte de K |
| B — K libre | — | ✓ | ✓ | medir aporte de estructura |
| C — sin corrector | ✓ | — | — | medir aporte del corrector |
| D — extrapolación | ✓ | — | ✓ | medir ley de evolución fuera de entrenamiento |

---

# 10. Criterios de interpretación

No se establecerá un único criterio binario de "ganador". El objetivo es identificar el mecanismo.

## Evidencia fuerte a favor de la estructura ÓRBITA

Se considerará especialmente fuerte si se observa simultáneamente:

1. ÓRBITA > K libre en `h8/h16`.
2. ÓRBITA > corrector-only en `h8/h16`.
3. Sin corrector conserva parte de la ventaja o muestra una dinámica estable aunque tenga menor accuracy absoluta.
4. ÓRBITA mantiene ventaja en `h128/h256` o degrada más lentamente.
5. La ventaja se reproduce en DEV y CONFIRM.

## Evidencia de que el corrector es el mecanismo dominante

Si:

```text
corrector-only ≈ ÓRBITA
```

y el núcleo estructurado sin corrector no aporta mejora relevante.

## Evidencia de que la estructura espectral es innecesaria

Si:

```text
K libre ≈ ÓRBITA
```

especialmente si la equivalencia persiste en horizontes no entrenados.

## Evidencia de extrapolación débil

Si todas las variantes colapsan de forma similar a partir de `h64`, entonces no debe afirmarse generalización indefinida. El resultado puede limitarse a la región temporal demostrada por el benchmark.

---

# 11. Regla contra el sobreajuste experimental

Después de ejecutar DEV:

- no modificar arquitectura para mejorar CONFIRM;
- no cambiar semillas de CONFIRM;
- no seleccionar hiperparámetros por el resultado de CONFIRM;
- no eliminar semillas desfavorables;
- no ocultar ablaciones fallidas.

Si una decisión nueva resulta necesaria, debe registrarse antes de la nueva corrida y, cuando corresponda, utilizar semillas de tuning independientes.

---

# 12. Resultado esperado del protocolo

El objetivo final no es demostrar simplemente que "ÓRBITA gana".

El objetivo es responder cuatro preguntas causales:

```text
¿1. Hace falta K?
        ↓
¿2. Hace falta que K tenga estructura espectral?
        ↓
¿3. Hace falta el corrector?
        ↓
¿4. La dinámica aprendida extrapola fuera del horizonte entrenado?
```

Si las cuatro respuestas favorecen la hipótesis, el siguiente paso será formular una versión más fuerte de la hipótesis arquitectónica:

> **ÓRBITA no obtiene su ventaja principalmente por capacidad paramétrica adicional, sino por imponer una dinámica latente estructurada que separa transporte global y corrección residual y que puede extrapolar temporalmente más allá de los ejemplos usados para supervisarla.**

Esa afirmación deberá considerarse provisional hasta completar las cuatro pruebas.

---

## 13. Artefactos que deben publicarse

Para cada variante:

```text
artifacts/orbita/ablations/<variant>/dev/
artifacts/orbita/ablations/<variant>/confirm/
```

Mínimo:

- `metrics.csv`;
- tabla por semilla;
- estadísticas pareadas;
- configuración/hyperparams;
- auditoría;
- proceso nuevo;
- paridad Rust↔PyTorch cuando aplique;
- curva de error por horizonte;
- resumen reproducible de comandos.

Los resultados deben incorporarse posteriormente a un documento separado de resultados. Este archivo define únicamente el protocolo y no debe modificarse para acomodar resultados obtenidos posteriormente.
