# Motor unificado CDT–spin–RQM–EPR–cognición

La entrada de producción es:

```powershell
cargo run --release --bin native_unified_spin_cognitive
```

Para adjuntar un snapshot DMRG de OxiCUDA:

```powershell
cargo run --release --bin native_unified_spin_cognitive -- --tensor-network
```

Para adjuntar también el scaffold 3D-PEPS:

```powershell
cargo run --release --bin native_unified_spin_cognitive -- --tensor-network --peps3d
```

Para ejecutar los tres backends tensoriales:

```powershell
cargo run --release --bin native_unified_spin_cognitive -- --tensor-network --peps3d --graph-tn
```

## Arquitectura consolidada

```text
CDT simplicial pyrochlore
  ├─ vértices, aristas, caras y tetraedros
  ├─ coordinación periódica y multiplicidad física
  └─ simetría topológica
              ↓
Líquido de espines
  ├─ Hamiltoniano XXZ
  ├─ amplitudes complejas y evolución local
  ├─ entropía de entrelazamiento
  └─ retroacción espín–malla
              ↓
RQM
  ├─ amplitud y fase relacional
  ├─ coherencia, incertidumbre y elegibilidad
  └─ error predictivo firmado
              ↓
EPR computacional
  ├─ enlaces por utilidad predictiva
  └─ contradicción ante errores altos
              ↓
Cognición abstracta
  ├─ composición de relaciones
  ├─ selección de rutas
  └─ abstención
```

La simetría no se identifica con cognición. Controla dónde se comparte el
aprendizaje y participa en el gate de consolidación. El contenido cognitivo
reside en relaciones aprendidas y en su composición.

## Gate de conocimiento

Una relación sólo entra en `ConsolidatedKnowledge` cuando se cumplen
simultáneamente:

1. topología CDT regular;
2. estado de espines con entropía mínima;
3. testigo de entrelazamiento si está requerido;
4. relación RQM estable, coherente y con bajo error predictivo.

Romper la coordinación de la malla bloquea la consolidación aunque RQM haya
recibido suficientes exposiciones.

## Componentes internos

- `unified_spin_cognitive_engine.rs`: orquestador único.
- `quantum_spin_thermodynamic_engine.rs`: backend exacto XXZ hasta 16 espines.
- `symmetry_guided_rqm_epr.rs`: campo relacional y capa cognitiva.
- `entanglement.rs`: EPR predictivo clásico.
- `symmetry_thermodynamic_substrate.rs`: geometría simplicial.
- `simplicial_thermodynamic_engine.rs`: Hodge y Regge.
- `variational_spin_liquid_vmc.rs`: backend VMC escalable experimental interno.
- `oxicuda-tn`: backend tensorial seleccionado para la siguiente migración.

Los módulos siguen separados para mantener pruebas y permitir sustituir el
backend de espines sin cambiar RQM/EPR/cognición.

## Backend tensorial OxiCUDA

`oxicuda-tn` 0.5.0 fue validado con DMRG sobre una cadena Heisenberg abierta de
12 sitios:

- energía DMRG: `−5.142090633`;
- referencia exacta: `−5.142090633`;
- norma final: `1`;
- entropía de entrelazamiento positiva.

`oxicuda_pyrochlore_backend` ya traduce cada enlace físico a tres términos:

```text
J/2 · S⁺ᵢS⁻ⱼ + J/2 · S⁻ᵢS⁺ⱼ + JΔ · SᶻᵢSᶻⱼ
```

El MPO usa una máquina de estados finita exacta, comparte tres canales por sitio
de apertura y conserva multienlaces periódicos. DMRG sobre el cluster
pyrochlore mínimo de cuatro espines reproduce la energía Lanczos `−3` a
tolerancia `10⁻⁶`.

`UnifiedSpinCognitiveEngine::refresh_tensor_network` adjunta el snapshot DMRG al
motor sin modificar RQM/EPR/cognición. El siguiente paso pasa a ser comprimir el
MPO para clusters mayores y evaluar el backend 3D-PEPS de OxiCUDA.

La ejecución unificada de ocho espines produce:

- energía DMRG pyrochlore: `−5.000000000`;
- dimensión MPO: `23 → 20` después de compresión SVD;
- dimensión MPS: `8`;
- entropía central: `1.386294361`;
- backend reportado: `ExactWithOxiCudaMpo`.

## Adaptador 3D-PEPS

`oxicuda_peps3d_backend` empaqueta las cuatro subredes de cada celda pyrochlore
en un motivo `2×2` de la grilla cúbica OBC de OxiCUDA.

Para ocho espines:

- grilla PEPS: `4×2×1`;
- norma del estado producto: `1`;
- enlaces representables como vecinos cúbicos: `13`;
- enlaces físicos todavía no locales: `11`;
- optimización del Hamiltoniano: no disponible en el scaffold 3D actual.

El adaptador valida almacenamiento, mapeo y métricas, pero no se presenta como
solución variacional. Los 11 enlaces no locales requieren puertas de largo
alcance, un grafo PEPS general o una nueva rutina de optimización 3D.

## Tensor network sobre el grafo pyrochlore

`pyrochlore_graph_tensor_network` asigna un índice virtual independiente a cada
uno de los 24 enlaces físicos. Cada espín es un tensor de grado seis más su
índice físico. La contracción usa el planificador greedy/einsum de OxiCUDA.

Validación GHZ de ocho espines:

- enlaces representados: `24/24`;
- tensores: `8`;
- pasos de contracción: `7`;
- coste estimado: `57,344`;
- norma: `1`;
- entropía de un espín: `ln(2)`;
- amplitudes no nulas: `2`;
- energía Heisenberg: `+6`.

La energía alta es correcta para el estado GHZ ferromagnético y confirma que la
red representa la topología, no que haya encontrado el ground state. La
optimización variacional directa de los tensores del grafo sigue pendiente.

## Comparación con legacy

`engine_comparison::compare_unified_against_legacy` ejecuta un protocolo pareado
de asociación, transferencia por órbita, composición, retención, OOD y lesión
topológica.

En 24 ensayos:

- asociación directa, composición, retención y OOD: empate `24/24`;
- transferencia a una órbita no observada: unificado `24/24`, legacy `0/24`;
- bloqueo de conocimiento con CDT lesionado: unificado `24/24`, legacy `0/24`;
- tiempo total: unificado `464.9 ms`, legacy `1.15 ms`.

El nuevo motor amplía capacidades y causalidad de consolidación, pero no mejora
latencia. El legacy se midió con pasos térmicos desactivados, mientras el nuevo
incluyó bootstrap y evolución exacta de ocho espines; el factor temporal no es
una comparación de algoritmos equivalentes.

## Entrenamiento infinito reanudable

`native_unified_infinite_trainer` genera batches numéricos sin lenguaje:

- 32 tareas composicionales `A→B→C` sin entrenar `A→C`;
- ocho tareas de transferencia por órbita;
- asociaciones aleatorias de interferencia;
- validación periódica de composición, OOD, simetría y entrelazamiento.

Variables:

```text
UNIFIED_TRAIN_HOURS
UNIFIED_TRAIN_BATCH_SIZE
UNIFIED_TRAIN_VALIDATE_EVERY
UNIFIED_TRAIN_CHECKPOINT_EVERY
UNIFIED_TRAIN_MILESTONE_EVERY
UNIFIED_TRAIN_HOMEOSTASIS_COOLING
UNIFIED_TRAIN_MAX_BATCHES
UNIFIED_TRAIN_ROOT
```

Persistencia:

```text
data/unified_infinite_training/latest.json
data/unified_infinite_training/checkpoints/batch-*.json
data/unified_infinite_training/metrics.jsonl
data/unified_infinite_training/summary.json
```

El checkpoint conserva amplitudes complejas, relaciones RQM, estado EPR,
conocimiento consolidado, RNG y contadores. La reanudación fue validada.

La corrida corta alcanzó el gate en el batch 14. Durante la primera escala
prolongada, pulsos sin enfriamiento degradaron el testigo de entrelazamiento
aunque composición y órbitas seguían correctas. Se añadió homeostasis adaptativa:
si pasan las capacidades abstractas pero falla el gate cuántico, el motor enfría
el cluster y vuelve a validar. Con 2.2 millones de ejemplos el gate se restauró.

## Límites físicos

El backend exacto valida clusters cuánticos frustrados y entrelazados, pero no
demuestra por sí solo una fase macroscópica de líquido de espines. Para esa
afirmación siguen siendo necesarios tamaños mayores, extrapolación del gap,
estructura dinámica, orden topológico y excitaciones fraccionalizadas.

RQM y EPR son nombres de arquitectura computacional. No implican comunicación
superlumínica ni hardware cuántico real.

## Validación

```powershell
cargo test --lib
cargo check --bins
```

La suite cubre:

- Hodge y cierre simplicial;
- simetría y reparación;
- pyrochlore periódico con coordinación seis;
- unitariedad y tiempo imaginario;
- entropía/testigos de entrelazamiento;
- Lanczos matrix-free;
- VMC y cota variacional;
- transferencia RQM por órbitas;
- EPR predictivo;
- composición cognitiva;
- gate unificado y bloqueo por topología rota.

## Referencias

- NetKet: <https://www.netket.org/>
- Carleo y Troyer, NQS: <https://doi.org/10.1126/science.aag2302>
- OxiCUDA tensor networks: <https://docs.rs/oxicuda-tn/>
- Regge: <https://doi.org/10.1007/BF02733251>
- Hodge discreto: <https://arxiv.org/abs/1105.2712>
