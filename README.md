# CDT-RQM-EPR · Sistema Operativo Cognitivo

Motor termodinámico nativo en Rust. El Transformer se conserva como periferia
lingüística; CDT-RQM-EPR mantiene memoria, control, exploración, planificación y
sueño.

## Motor unificado de espines y cognición

La arquitectura consolidada CDT–spin–RQM–EPR se ejecuta con:

```powershell
cargo run --release --bin native_unified_spin_cognitive
```

CDT mantiene la malla simplicial pyrochlore; la simetría guía la transferencia
de aprendizaje; el líquido de espines aporta estado cuántico y entrelazamiento;
RQM/EPR mantienen relaciones; la capa cognitiva compone únicamente conocimiento
que supera el gate conjunto.

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
