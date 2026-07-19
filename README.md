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

Visualizador Transformer/CDT conservado:

```powershell
cargo run --release --bin native_thermo_visualizer
```

Los checkpoints, memorias, datasets, entrenadores y evaluadores del currículo
legacy fueron eliminados. El único estado de entrenamiento persistente vigente
es el generado por `native_cognitive_sleep_visualizer`.
