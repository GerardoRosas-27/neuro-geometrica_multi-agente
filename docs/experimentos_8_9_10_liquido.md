# Experimentos 8, 9 y 10 — inferencia líquida

Rama: exp/liquid-inference-experiments-8-9-10
Base: main
Arquitectura: LiquidInfer / WavePredictCore + Thermo CDT + RQM únicamente durante sueño.

Este documento fija el protocolo antes de ejecutar los experimentos. La finalidad es distinguir recuperación, generalización y predicción de relaciones no observadas.

## Preparación común

Desde la raíz:

git checkout exp/liquid-inference-experiments-8-9-10

cargo fmt --all -- --check
cargo test --release --lib liquid_cdt_memory -- --nocapture
cargo test --release --lib liquid_cdt_rqm_fuse -- --nocapture
cargo test --release --lib liquid_cdt_vs_main -- --nocapture

Para resultados reproducibles:
- ejecutar en --release;
- no mezclar ejecuciones con diferentes semillas;
- registrar CPU, Rust y commit;
- publicar media, p50, p95, p99 y exactitud;
- separar seen, unseen y OOD;
- registrar siempre la ruta Liquid, CDT o RQM;
- no considerar un 100 % de identidad como evidencia de generalización.

La inferencia actual de LiquidCdtSystem::infer es deliberadamente pura: no avanza el tick de CDT, no llama RQM y no consulta la memoria consolidada. El sueño escribe en CDT y destila relaciones a RQM.

## Experimento 8 — invariancia de representación / lenguaje

### Pregunta

Si dos expresiones diferentes llegan al mismo estado semántico en la frontera lingüística, ¿el sustrato líquido las trata como el mismo concepto?

Ejemplo:

perro  -> estado latente X -> líquido -> concepto X
dog    -> estado latente X -> líquido -> concepto X
chien  -> estado latente X -> líquido -> concepto X
犬      -> estado latente X -> líquido -> concepto X

### Protocolo fuerte

1. Usar Gemma 2 solamente como periferia lingüística.
2. Entrenar/consolidar únicamente una forma, por ejemplo perro.
3. Obtener su representación en la frontera field_linguistic_layer.
4. No entrenar dog, chien ni 犬.
5. Consultar las variantes.
6. Medir similitud entre fingerprints y recuperación del mismo atractor.
7. Repetir con paráfrasis y ruido controlado.

El resultado fuerte es que perro -> X y las variantes no entrenadas también -> X, sin almacenarlas como claves explícitas.

### Controles

- palabra entrenada vs palabra desconocida;
- palabras semánticamente cercanas pero distintas;
- palabras aleatorias;
- permutación de etiquetas;
- representación congelada vs adaptativa.

### Estado actual

El repositorio ya separa tokens, features y campo, pero este experimento todavía necesita conectar el encoder lingüístico real con el protocolo. No debe reportarse como demostrado mientras esa conexión no exista.

## Experimento 9 — composición de relaciones

### Pregunta

¿El sistema puede componer relaciones que aprendió por separado?

Ejemplo de entrenamiento:

A -> B
B -> C
C -> D

Consultas no vistas:

A -> C
B -> D
A -> D

### Matriz mínima

| Query | Visto durante train | Tipo |
|---|---:|---|
| A→B | Sí | seen |
| B→C | Sí | seen |
| C→D | Sí | seen |
| A→C | No | composicional |
| B→D | No | composicional |
| A→D | No | composicional |

### Métricas

- exactitud por distancia composicional;
- score líquido;
- energía -ln(score);
- margen top-1/top-2;
- número de pasos;
- ruta utilizada;
- abstención cuando la confianza sea insuficiente.

### Controles

Comparar contra:
1. memoria directa de pares;
2. RQM entrenado directamente con los saltos;
3. líquido sin entrenamiento relacional;
4. una versión con relaciones mezcladas.

El resultado interesante es que A→C y A→D sean correctos sin haber sido almacenados como pares explícitos.

Si el rendimiento cae al azar al eliminar los saltos explícitos, el experimento no demuestra composición.

## Experimento 10 — predicción de estados futuros no observados

### Pregunta

¿La dinámica líquida puede predecir un estado que nunca apareció directamente como respuesta a ese cue?

Este experimento ataca directamente la hipótesis de que la inferencia selecciona una posibilidad futura compatible con la dinámica aprendida.

### Variante A — predicción a distancia

Entrenar:

A -> B
B -> C
C -> D

Consultar A -> ?

Esperado: C si la arquitectura compone la dinámica.

### Variante B — bifurcación

Entrenar:

A -> B
A -> C
B -> D
C -> E

Consultar desde A con una condición contextual que determine la rama.

Medir hipótesis, score, confianza, margen top-2, energía y selección/abstención.

### Variante C — perturbación

Después del aprendizaje:

A -> B -> C -> D

introducir ruido en el estado intermedio y medir si la dinámica recupera la trayectoria correcta.

Esto separa recuperación de memoria de relajación hacia un atractor dinámico.

## Matriz conjunta

| Experimento | Qué prueba | No debe confundirse con |
|---|---|---|
| 8 | invariancia de representación / lenguaje | alias manual |
| 9 | generalización composicional | recuperación de pares |
| 10 | predicción/extrapolación dinámica | memorizar respuestas |

La progresión buscada es:

8: representación
↓
9: estructura
↓
10: dinámica/predicción

## Baselines actuales

Estos comandos ya corresponden a la arquitectura líquida integrada:

cargo test --release --lib liquid_cdt_memory -- --nocapture
cargo test --release --lib liquid_cdt_rqm_fuse -- --nocapture
cargo test --release --lib liquid_cdt_vs_main -- --nocapture

liquid_cdt_vs_main debe tratarse como baseline, no como experimento 8/9/10: demuestra la separación del hot path, la latencia de identidad y la capacidad de MAIN/RQM para mapas arbitrarios.

## Registro obligatorio

Cada ejecución debe registrar:

commit
seed
N
train_examples
unseen_examples
ood_examples
accuracy_seen
accuracy_unseen
accuracy_ood
mean_us
p50_us
p95_us
p99_us
top1_score
top2_score
margin
energy
abstentions
liquid_calls
cdt_calls
rqm_calls
sleep_ms
engrams_before
engrams_after

Además: hardware, versión de Rust, modo release/debug, parámetros, número de semillas, distribución train/test y ejemplos exactos de fallos.

## Regla de publicación

No presentar estos experimentos como evidencia de cognición general.

El lenguaje correcto es:
- E8: invariancia/generalización de representación;
- E9: generalización composicional;
- E10: predicción dinámica/extrapolación.

Solo si los tres sobreviven a controles adversariales tendría sentido avanzar hacia una prueba de generalización mucho más amplia.
