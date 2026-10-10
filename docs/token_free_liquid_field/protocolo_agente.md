# Protocolo de implementación — Token-Free Liquid Field (core Rust)

## Regla principal

Esta rama es un laboratorio aislado. El core de inferencia y sus benchmarks oficiales permanecen en Rust. Frameworks de redes neuronales se usan para entrenar prototipos, calcular gradientes y comparar funciones candidatas; no deben introducir Python/JAX/TensorFlow/PyTorch en el hot path de producción.

Rama: `exp/token-free-liquid-field`.

No reutilizar checkpoints, pesos entrenados, datasets sellados ni resultados favorables de `exp/field-autonomy-v4` o `exp/field-autonomy-next` para declarar resultados positivos. Puede reutilizarse infraestructura genérica documentando origen y versión.

## Estrategia recomendada

1. Mantener una implementación Rust de referencia para inferencia.
2. Construir prototipos entrenables en Keras 3 + JAX inicialmente.
3. Usar PyTorch o TensorFlow solo si una necesidad concreta justifica cambiar de backend; no entrenar la misma arquitectura en todos los frameworks sin motivo.
4. Exportar parámetros y definición de arquitectura con un manifiesto estable.
5. Portar la función de inferencia a Rust y comprobar equivalencia numérica.
6. Medir la latencia oficial dentro de Rust, por separado de encoder y decoder.
7. Mantener el entrenamiento fuera del hot path. Si después se investiga plasticidad online en Rust, debe ser una fase explícita y reproducible.

## Arquitectura objetivo

```text
texto → encoder lingüístico (opcional/frozen)
      → FieldEncoder → Ψ₀
      → Liquid Core Rust: Fθ(Ψ,c)
      → Ψ₁...Ψₙ
      → decoder independiente

sleep/entrenamiento:
experiencias → framework de entrenamiento o algoritmo Rust
             → actualización/consolidación
             → parámetros/estado versionado
             → exportación → runtime Rust
```

## Contrato de tipos del core

Definir tipos explícitos y versionados, por ejemplo:
- `FieldState`: vector/estado Ψ con dimensión declarada;
- `FieldContext`: solo señales estructuradas permitidas;
- `LiquidParams`: parámetros entrenados y versionados;
- `RolloutConfig`: máximo de pasos, tolerancias, reglas de parada;
- `RolloutTrace`: pasos, normas, energía, nodos activos y condición de terminación.

El core no debe aceptar:
- texto ni token IDs;
- tokenizer, logits o hidden states del LLM después de producir Ψ₀;
- conexión de red o llamadas al LLM;
- lookup de respuestas o índice directo input→target;
- acceso implícito a CDT/RQM/tabla/NN/attractor bank durante los experimentos que los prohíben.

## Módulos sugeridos

Mantener el código nuevo encapsulado inicialmente:
- `src/token_free_field.rs`: interfaz y tipos del core;
- `src/token_free_dynamics.rs`: variantes de transición;
- `src/bin/run_token_free_field.rs`: runner reproducible;
- `src/token_free_dataset.rs`: generación y splits;
- `src/token_free_metrics.rs`: métricas;
- `src/token_free_provenance.rs`: hashes y auditoría;
- `src/token_free_controls.rs`: controles;
- `training/token_free/`: scripts Python, configuración y entorno fijado;
- `artifacts/token_free/`: artefactos generados; respetar reglas existentes del repositorio sobre archivos grandes.

Antes de añadir módulos, inspeccionar el layout real de Cargo, dependencias, convenciones y módulos existentes. No copiar ciegamente estos nombres si la estructura real recomienda otra organización.

## Exportación y paridad

Todo artefacto entrenado en framework debe acompañarse de:
- arquitectura/version;
- nombres, formas, dtype y layout de parámetros;
- activaciones, normalización y orden de operaciones;
- seed, configuración, manifest y hashes;
- versión de framework/exportador;
- checksum del artefacto exportado;
- corpus de vectores de equivalencia y reporte Python↔Rust.

Protocolo de paridad:
1. congelar checkpoint entrenado y configuración;
2. generar vectores de entrada y estados intermedios de referencia fuera de TEST;
3. ejecutar esos mismos vectores en Rust;
4. calcular error absoluto/relativo máximo, media y p99 por capa/estado;
5. definir tolerancias antes de evaluar TEST;
6. probar decisiones discretas, top-k, umbrales y stopping;
7. registrar `PORT_PARITY_PASS` o `PORT_PARITY_FAIL`.

No declarar en Rust el rendimiento de un prototipo que no supera paridad. Las diferencias de solver, precisión, orden de suma y empates de top-k deben documentarse.

## Dataset y provenance

Debe haber `TRAIN / DEV / TEST / OOD / COMPOSITION / LONG_HORIZON` con manifest:
- versión del generador;
- seeds;
- SHA-256 por split;
- familias de transformación;
- cardinalidad/dimensiones;
- target y equivalentes canónicos excluidos;
- origen del encoder/checkpoint, si aplica.

No entrenar en TEST, ajustar stopping con TEST ni seleccionar seeds favorables. Los datos sintéticos deben probar reglas, composición y extrapolación, no solo interpolación de trayectorias vistas.

## Orden obligatorio

1. E45 bottleneck y contrato de estado.
2. E46 comparar MLP recurrente, Liquid y SSM con presupuesto comparable.
3. Validar exportación/paridad en Rust antes de optimizar.
4. E47 sparse; medir coste de selección incluido.
5. E48 stopping adaptativo.
6. E49 coarse-to-fine.
7. E50 beam geométrico.
8. E51 memoria asociativa con lookup como control.
9. E52 plasticidad.
10. E53 consolidación vs inferencia.
11. E54 LLM access kill test.
12. E55 encoder swap.
13. E56 scaling.
14. E57 comparación con v4 cuando v4 esté cerrada.

No saltar a consolidación o comparación final para evitar optimizar una interfaz o dinámica aún no validada.

## Métricas obligatorias

Calidad:
- accuracy, OOD, composition, long-horizon, abstention.

Dinámica:
- `||Psi||`, `||DeltaPsi||`, cosine, energía, distancia al manifold, error de transición, norma espectral/Jacobiano y amplificación de perturbaciones.

Eficiencia:
- latencia p50/p95/p99 del core Rust;
- latencia separada de encoder/decoder y end-to-end;
- pasos, nodos activos, memoria, operaciones aproximadas;
- coste de entrenamiento, exportación y consolidación;
- tamaño de parámetros y artefactos.

Auditoría:
- token IDs vistos por core;
- llamadas al LLM/tokenizer después del encoder;
- lookups de decoder, CDT, RQM, tablas, NN, attractor;
- target visto y equivalente visto;
- hashes de provenance;
- estado de paridad Python/Rust.

## Estados válidos

`NOT_RUN`, `SMOKE_ONLY`, `PASS`, `PARTIAL`, `FAIL`, `LEAKED`, `DATASET_INVALID`, `PORT_PARITY_FAIL`.

## Criterio de descubrimiento fuerte

Inferencia:
`Ψ₀ → dinámica Rust → estado nuevo → decoder`

con cero acceso a tokens/LLM después del encoder, target nunca almacenado, composición/OOD positivos, ventaja de coste frente a controles y replicación en múltiples seeds.

Consolidación:
`experiencias → consolidación → cambio persistente en Dθ → borrar episodios → resolver estados nuevos`.

El cambio debe persistir en un proceso nuevo y superar el control sin experiencia. No llamar a esto aprendizaje persistente si la respuesta se recupera desde una memoria externa.

## Anti-autoengaño

- No declarar cognición por velocidad.
- No declarar memoria por nearest-neighbor.
- No declarar aprendizaje por fitting.
- No declarar autonomía si el decoder/lookup recupera respuestas.
- No declarar token-free si hay llamadas al LLM después de Ψ₀.
- No comparar Python eager contra Rust optimizado como si fuera una comparación de arquitectura.
- No atribuir a Rust un resultado hasta validar paridad.
- No cambiar el modelo tras inspeccionar TEST.

## Cierre

La rama queda lista para comparación con v4 cuando E45–E56 tengan reportes reproducibles, se valide paridad de las variantes seleccionadas y E57 pueda ejecutarse sobre un dataset común sellado. El merge no es un objetivo automático.
