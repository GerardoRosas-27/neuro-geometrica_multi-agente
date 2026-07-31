# CDT-RQM-EPR · Sistema Operativo Cognitivo

Motor termodinámico nativo en Rust. El Transformer se conserva como periferia
lingüística; CDT-RQM-EPR mantiene memoria, control, exploración, planificación y
sueño.

## Motor termodinámico fasorial independiente

La arquitectura fasorial se ejecuta por separado y no reemplaza el motor CDT,
RQM/EPR ni el motor unificado:

```powershell
cargo run --release --bin native_phasor_thermodynamic
```

`NativePhasorThermodynamicEngine` reutiliza únicamente la configuración,
topología, amplitudes, fases y temperaturas iniciales de
`NativeThermoCdtSubstrate`. Después mantiene su propio estado `Complex32` y
evoluciona mediante:

- Laplaciano magnético disperso con las fases de arista CDT;
- energía de acoplamiento por interferencia;
- potencial radial que impide el mínimo trivial de amplitud cero;
- término entrópico `F = U - T·S`;
- estímulos complejos, ruido térmico y enfriamiento gradual.

Para buscar directamente el mínimo, `minimize_free_energy` aplica sincronización
topológica de gauge `O(E)`, gradiente precondicionado y búsqueda de línea de
Armijo. Cada paso aceptado reduce la energía libre. El benchmark verifica el
resultado contra un mínimo global conocido:

```powershell
cargo run --release --bin native_phasor_minimum_benchmark
```

Comparación aislada entre el minimizador global actual y un bucle de inferencia
activa local:

```powershell
cargo run --release --bin native_phasor_active_inference_benchmark
```

La variante activa aplica Metropolis-within-Gibbs a un fasor complejo por vez,
calcula deltas de energía usando sólo sus aristas incidentes, mantiene la
entropía mediante estadísticos suficientes y contrasta ese valor con una
estimación Monte Carlo. El benchmark incluye una ablación Gibbs sin gradiente
local y usa el mismo estado inicial y presupuesto de barridos para las tres
variantes.

El resultado selecciona el gradiente global precondicionado con Armijo como
solver canónico de producción: ganó en tiempo y energía libre final en las tres
escalas evaluadas (128, 512 y 2.048 nodos). Gibbs y Active Inference permanecen
como variantes experimentales; el benchmark termina con error si Armijo deja de
ganar ambas métricas.

Los movimientos de Pachner no se simulan en este benchmark: el motor fasorial
actual conserva un grafo magnético, no un complejo simplicial con caras,
orientaciones y restricciones de manifold. Un simple alta/baja de aristas no
sería una prueba válida de movimientos de Pachner.

Comparación pareada contra el CDT anterior:

```powershell
cargo run --release --bin native_thermodynamic_attractor_comparison
```

El benchmark entrena ambos sustratos con el mismo dataset Walsh/Hebbiano, usa
los mismos cues corrompidos y evalúa ambos con una energía XY común. Reporta
recuperación de atractores, residuo de fase, iteraciones y tiempo de pared.

Experimento causal pre/post de consolidación:

```powershell
cargo test --lib consolidation_basin_experiment -- --nocapture
cargo run --release --bin native_consolidation_basin_experiment
```

El protocolo conserva un snapshot antes de sueño, consolida una configuración
verificada y repite exactamente los mismos cues con 10–40 % de corrupción sobre
los paisajes pre y post. La exactitud es directa: el flip global Z₂ cuenta como
fallo (el reporte incluye la variante gauge-invariante sólo como diagnóstico).
El gate exige aumento de corrupción crítica, al menos 10 puntos porcentuales de
ganancia media en recuperación (`minimum_mean_success_gain`, en config) y
ninguna caída de exactitud por nivel. La corrida de referencia pasó 144/144
recuperaciones post frente a 0/144 pre, y el test se repite con ocho semillas
fijas. Es evidencia interna de deformación de cuenca, no de generalización
conceptual ni de energía física.

Validación cognitiva escalonada:

```powershell
cargo test --lib cognitive_generalization_benchmark -- --nocapture
cargo run --release --bin native_cognitive_generalization_benchmark
```

El protocolo entrena cuatro familias relacionales y separa cuatro niveles:
memoria exacta, variaciones de fase no vistas, composición A→B→C sin atajo
directo y transferencia a tres pares isomórficos con control sin simetría. La
corrida de 24 ensayos obtuvo 100 % en los cuatro niveles, en el control y en
abstención OOD. La órbita isomórfica se proporciona explícitamente: demuestra
transferencia estructural limitada, no descubrimiento autónomo de simetrías.

Validación adversarial y descubrimiento limitado de simetría:

```powershell
cargo test --lib advanced_cognitive_validation -- --nocapture
cargo run --release --bin native_advanced_cognitive_validation
```

El benchmark introduce ramificaciones A→B→C y A→D, selección dependiente de
fase, consultas equidistantes que deben causar abstención, tres escalas
topológicas y descubrimiento de una traslación cíclica de canales a partir de
ejemplos ruidosos con un outlier. En 36 ensayos pasó selección, trayectoria,
ambigüedad, orden energético, transferencia heldout y rechazo de estructuras
conflictivas. El grupo de transformaciones cíclicas sigue predefinido; el
desplazamiento y la órbita concreta no se proporcionan.

Descubrimiento de familia y selección por complejidad:

```powershell
cargo test --lib transformation_family_discovery -- --nocapture
cargo run --release --bin native_transformation_family_discovery
```

El selector contrasta traslaciones 2D, rotaciones, reflexiones, permutaciones y
composiciones. Su energía de hipótesis combina error mediano y penalización de
complejidad. El benchmark usa ejemplos ruidosos, un outlier, una permutación
memorizadora como competidor y patrones heldout. La familia y sus parámetros no
se proporcionan; el catálogo de cinco familias sí está definido previamente.

Motor híbrido sobre un único core CDT:

```powershell
cargo run --release --bin native_hybrid_phasor_cdt
```

`NativeHybridPhasorCdtEngine` monta dos capas sobre
`NativeThermoCdtSubstrate`. Durante wake, `infer_and_stage` busca atractores con
fasores y los conserva en una cola volátil sin modificar CDT. Sólo
`sleep_consolidate` revalida esa cola y transfiere los estados aprobados a
amplitudes, fases, pesos y estabilidad de las aristas CDT. La fase de sueño es
transaccional: ante un error restaura core, memoria y pendientes.

La prueba costosa de estabilidad se difiere a sleep para conservar la latencia
del solver fasorial standalone. Benchmark de paridad y overhead:

```powershell
cargo run --release --bin native_phasor_fusion_efficiency
```

Pruebas específicas:

```powershell
cargo test --lib native_phasor_thermodynamic_engine
```

## Motor unificado de espines y cognición

La arquitectura consolidada CDT–spin–RQM–EPR se ejecuta con:

```powershell
cargo run --release --bin native_unified_spin_cognitive
```

CDT mantiene la malla simplicial pyrochlore; la simetría guía la transferencia
de aprendizaje; el líquido de espines aporta estado cuántico y entrelazamiento;
RQM/EPR mantienen relaciones; la capa cognitiva compone únicamente conocimiento
que supera el gate conjunto.

El mismo core puede adjuntar y optimizar una referencia variacional
Jastrow/VMC mediante `refresh_variational_spin_liquid`. El ratio Metropolis usa
un recorrido híbrido calibrado: contiguo en N<16 y CSR por incidencias desde
N=16. El benchmark reproducible es:

```powershell
cargo run --release --bin native_vmc_ratio_benchmark
```

Diseño y límites: `docs/unified_spin_cognitive_engine.md`.

Entrenamiento sintético reanudable:

```powershell
$env:UNIFIED_TRAIN_HOURS="5"
cargo run --release --bin native_unified_infinite_trainer
```

Guarda `latest.json`, milestones, métricas JSONL y resumen en
`data/unified_infinite_training/`. Si existe un checkpoint compatible, continúa
desde el último batch.

### Datos y checkpoints no versionados

Los datasets, métricas y checkpoints generados no se incluyen en GitHub. Para
crearlos desde cero:

```powershell
$env:UNIFIED_TRAIN_HOURS="5"
cargo run --release --bin native_unified_infinite_trainer
```

Se generarán:

```text
data/unified_infinite_training/latest.json
data/unified_infinite_training/checkpoints/
data/unified_infinite_training/metrics.jsonl
data/unified_infinite_training/summary.json
```

Para regenerar el estado visual/cognitivo nativo:

```powershell
cargo run --release --bin native_cognitive_sleep_visualizer
```

Esto crea `data/native_cognitive_desktop/`. Para reanudar desde un artefacto
externo, cópialo a esas mismas rutas antes de ejecutar el entrenador.

## Entrenamiento principal

La aplicación comienza desde un sustrato limpio si no existe
`data/native_cognitive_desktop/latest.json`, ejecuta sueño infinito y guarda al
terminar cada fase:

```powershell
cargo run --release --bin native_cognitive_sleep_visualizer
```

Fases:

1. observación wake;
2. inducción automática de esquemas;
3. consolidación térmica;
4. exploración OOD;
5. validación, commit o rollback.

Persistencia:

```text
data/native_cognitive_desktop/latest.json
data/native_cognitive_desktop/checkpoints/*.cdt_native
data/native_cognitive_desktop/checkpoints/*.cognitive.json
```

Controles:

```text
Tab       2D / 3D
Espacio   pausa
E         mostrar relaciones
S         guardado manual
Esc       guardar y salir
```

## Integración continua y compilación

`.github/workflows/ci.yml` ejecuta en cada push: `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings` y `cargo test --release --lib`
(que incluye los gates científicos de regresión: cuenca de consolidación
multi-semilla, generalización, validación adversarial y descubrimiento de
familia).

`.cargo/config.toml` fija `target-cpu=native` para los bucles calientes; los
binarios resultantes son específicos de la CPU que compila.

Los cuatro benchmarks cognitivos reportan `wall_clock_seconds` en su JSON para
seguimiento de coste, no sólo de tasas.

## Transformer y pesos

Los pesos paginados, shards, catálogo y tokenizador se conservan en:

```text
data/native_tinyllama_paged_thermo/
data/native_gemma2_paged_thermo/
```

Reconstrucción o inspección GGUF:

```powershell
cargo run --release --bin native_gguf_paged_thermo -- --model tinyllama:1.1b-chat-v1-q4_0 --output data/native_tinyllama_paged_thermo --lazy
```

Periferia lingüística Rust:

```powershell
cargo run --release --bin native_hybrid_assistant
```

### Gemma 2 2B nativo

El Transformer Gemma 2 cuantizado, su tokenizador SentencePiece/Unigram y la
generación autoregresiva se ejecutan directamente con Candle/Rust. La aplicación
abre el GGUF local y no usa el proceso, API ni servidor de Ollama:

```powershell
cargo run --release --bin native_gemma2_chat
```

Comandos interactivos:

```text
/limpiar  borra el historial
/salir    termina
```

También admite una consulta no interactiva y límites configurables:

```powershell
cargo run --release --bin native_gemma2_chat -- --prompt "Explica la relatividad" --max-tokens 128 --context 2048
```

### Gemma 2 adaptativo con memoria termodinámica

El chat adaptativo captura resúmenes RMS por capa, trata cada bloque Transformer
como un supernodo, propone máscaras conservadoras y las compara contra el
forward completo. Una ruta solo se consolida cuando conserva el token principal,
supera el umbral de similitud de logits y pasa el gate spin exacto fuera de
línea. Si la confianza es insuficiente, limpia la KV cache y repite con todas las
capas.

```powershell
cargo run --release --bin native_gemma2_adaptive_chat
cargo run --release --bin native_gemma2_adaptive_chat -- --prompt "Explica la relatividad" --max-tokens 128
```

La memoria rápida se vacía al llenarse y también con `/sueño` o al cerrar el
chat. El estado persistente vive en `data/native_gemma2_adaptive/`: sustrato
termodinámico, registro de rutas y checkpoint versionado. Para ejecutar solo
consolidación, decaimiento selectivo y poda:

```powershell
cargo run --release --bin native_gemma2_adaptive_chat -- --sleep-only
```

El enrutamiento es deliberadamente seguro: un Gemma denso no garantiza que
omitir capas sea válido. Si ninguna máscara parcial conserva la salida, se usa
el modelo completo y no se registra un ahorro ficticio.

Busca el GGUF en `data/native_gemma2_paged_thermo/manifest.txt`, en
`ollama-models/` o en la ruta indicada mediante `--model`/`GEMMA2_GGUF`. Para
reconstruir el catálogo, tokenizador inspeccionable y manifiesto paginado:

```powershell
cargo run --release --bin native_gguf_paged_thermo -- --model gemma2:2b --output data/native_gemma2_paged_thermo --lazy
```

Entrenamiento infinito del sustrato CDT–líquido de espines–RQM/EPR, con Gemma 2
nativo como generador/evaluador lingüístico:

```powershell
cargo run --release --bin native_gemma2_spin_infinite_trainer
```

Los pesos GGUF permanecen congelados; se entrenan las relaciones cognitivas y
el estado del líquido de espines. El currículo avanza solamente al consolidar
cada etapa: acción sensorimotora, permanencia del objeto, imitación diferida,
predicción y error, atención preverbal, juego simbólico, abstracción,
etiquetado lingüístico y planificación ejecutiva. También mide integración
entre etapas, composición sin arista directa, transferencia, retención,
abstención OOD y entrelazamiento. Es un gate funcional de tareas, no evidencia
de consciencia ni cognición general.

Configuración y ejecución acotada:

```powershell
$env:GEMMA_SPIN_MAX_CYCLES="9"
$env:GEMMA_SPIN_CHECKPOINT_EVERY_CYCLES="2"
$env:GEMMA_SPIN_CHECKPOINT_EVERY_SECONDS="300"
cargo run --release --bin native_gemma2_spin_infinite_trainer
```

Sin `GEMMA_SPIN_MAX_CYCLES` ni `GEMMA_SPIN_TRAIN_HOURS`, el ciclo no termina.
Reanuda desde `data/gemma2_developmental_infinite_training/latest.json`; guarda
métricas en `metrics.jsonl`, hitos en `checkpoints/` y conserva por defecto los
24 más recientes. Otras variables: `GEMMA_SPIN_TEACHER_TOKENS`,
`GEMMA_SPIN_EXPOSURES`, `GEMMA_SPIN_VALIDATE_EVERY`,
`GEMMA_SPIN_MILESTONE_EVERY`, `GEMMA_SPIN_RETAIN_MILESTONES`,
`GEMMA_SPIN_MINIMUM_SEEN` y `GEMMA_SPIN_TRAIN_ROOT`.

Cuando las nueve etapas quedan consolidadas, el ciclo crea una zona lingüística
de planificación. Gemma 2 resuelve la referencia textual a un objeto etiquetado;
la red recupera el plan abstracto consolidado y lo vuelve a expresar con
etiquetas. Para consultarla después del entrenamiento:

```powershell
cargo run --release --bin native_gemma2_spin_infinite_trainer -- --plan "quiero alcanzar el muñeco oculto"
```

La salida separa `referencia_llm`, `objeto_red` y `plan_red`, permitiendo
auditar qué resolvió el LLM y qué secuencia provino del motor cognitivo.

Visualizador Transformer/CDT conservado:

```powershell
cargo run --release --bin native_thermo_visualizer
```

Los checkpoints, memorias, datasets, entrenadores y evaluadores del currículo
legacy fueron eliminados. Los estados persistentes vigentes son los generados
por `native_cognitive_sleep_visualizer`, `native_unified_infinite_trainer` y
`native_gemma2_spin_infinite_trainer`.
