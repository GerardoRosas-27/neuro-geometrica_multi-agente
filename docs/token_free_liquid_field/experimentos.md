# Batería experimental E45–E57 — framework de entrenamiento + core Rust

Todas las pruebas deben ejecutarse con dos capas claramente diferenciadas: **entrenamiento experimental** (Keras 3/JAX por defecto; PyTorch/TensorFlow si se justifica) y **runtime de evaluación Rust**. El prototipo Python puede servir para desarrollar la función y calcular gradientes, pero la latencia oficial del core se mide en Rust. Los resultados Python solo son resultados del prototipo hasta que se exportan parámetros y se supera la prueba de equivalencia.

## Protocolo técnico común

Para cada arquitectura registrar:
- framework, versión, backend, compilador y dispositivo;
- seed, parámetros entrenables, pasos de optimización y presupuesto de cómputo;
- dimensiones de entrada/estado/salida, dtype, layout de tensores;
- pérdida, optimizador, learning rate, scheduler y early stopping;
- SHA-256 de dataset/splits/checkpoint/export;
- latencia de encoder, core Rust y decoder por separado;
- error numérico Python↔Rust: máximo, media, percentil 99;
- resultados en seen, OOD, composición y rollout;
- coste de entrenamiento y tamaño del artefacto exportado.

No comparar solo tiempos de Python eager con Rust optimizado. Los benchmarks de arquitectura deben comparar el runtime Rust de las implementaciones portadas, y reportar por separado los costes de entrenamiento.

## E45 — Bottleneck token-free

**Pregunta:** ¿cuánta dimensión necesita Ψ para conservar las variables relevantes para la dinámica?

Dimensiones: 16, 32, 64, 128, 256, 512.

Controles: hidden raw, proyección aleatoria, PCA/linear, FieldEncoder entrenado, bottleneck no lineal.

Entrenamiento: Keras/JAX puede optimizar FieldEncoder con pérdidas contrastivas/relacionales; congelar el encoder lingüístico para la primera prueba.

Aceptación: una dimensión compacta mantiene calidad semántica/relacional y reduce coste. Reconstrucción textual no basta y puede revelar fuga lexical.

## E46 — Liquid vs MLP recurrente vs SSM

Modelos:
A. MLP recurrente: Ψ'=MLP(Ψ,c)
B. dinámica líquida: Ψ'=Ψ+Δt·Fθ(Ψ,c)
C. SSM compacto/selective-state
D. dinámica lineal local.

Igualar aproximadamente parámetros, datos, seeds, presupuesto de entrenamiento y runtime Rust. La variante SSM es una referencia de comparación, no una presunción de victoria.

Métricas: error one-step, rollout h=1/2/4/8/16/32/64, OOD, estabilidad/Jacobiano, µs/query, memoria y coste de entrenamiento.

Aceptación: ventaja en al menos dos ejes independientes: generalización/rollout y eficiencia/estabilidad. La superioridad solo existe para la configuración medida.

## E47 — Sparse Field

k activo: 1%, 2%, 5%, 10%, 20%, 50%, 100%. Comparar top-k por magnitud, umbral adaptativo, routing aprendido y bloques locales.

Implementación: primero validar la máscara en framework; luego portar la misma semántica a Rust. Incluir coste de seleccionar nodos, no solo el coste de actualizarlos.

Aceptación: menor coste end-to-end del core Rust a calidad estadísticamente equivalente, sin inestabilidad ni explosión de pasos.

## E48 — Inferencia adaptativa / early stopping

Comparar pasos fijos 1/2/4/8/16/32/64 con parada por estabilidad:
`||Ψ(t+1)-Ψ(t)|| < ε`, más una condición calibrada de confianza/energía.

La regla de parada se ajusta en DEV y se congela antes de TEST.

Aceptación: baja latencia media sin degradación relevante en OOD, composición ni casos difíciles.

## E49 — Coarse-to-fine

Primero dinámica de baja dimensión/resolución; refinar solo ante incertidumbre alta. Comparar con full-resolution en el mismo runtime Rust.

Aceptación: menor latencia media con pérdida de calidad predefinida máxima de 1–2 puntos porcentuales, incluyendo OOD.

## E50 — Beam geométrico

Generar K estados futuros (K=1,2,4,8) y rankearlos en el espacio de campo por estabilidad, energía y consistencia. No decodificar cada rama para elegir la respuesta.

Aceptación: mejora composición/OOD frente a greedy y coste inferior a una búsqueda token-level comparable. Reportar coste de mantener ramas y no ocultar trabajo en el decoder.

## E51 — Memoria asociativa

Controles: sin memoria, nearest-neighbor, Hopfield moderno/sparse, atractores aprendidos. El target no puede almacenarse directamente.

Aceptación: la memoria cambia el estado o la trayectoria; no devuelve el target ni un índice input→target. Lookup se reporta como control, nunca como evidencia de dinámica.

## E52 — Plasticidad online

Probar actualización de θ o adaptadores mediante error predictivo, actualización local, low-rank o deformación energética. Definir claramente qué estado es persistente y qué se guarda.

Entrenamiento por lotes puede hacerse en JAX/Keras; si la actualización ocurre en Rust, demostrar equivalencia en un conjunto de operaciones y registrar su coste.

Aceptación: una experiencia mejora consultas futuras no vistas, con controles sin aprendizaje, experiencia aleatoria y etiquetas barajadas. Medir olvido e interferencia.

## E53 — Inferencia vs consolidación

Separar hot path y sleep path. Comparar:
1. Liquid sin aprendizaje;
2. plasticidad online;
3. consolidación por lotes;
4. Thermo CDT como control;
5. híbrido.

Métricas: µs/query en Rust, ms/experiencia, bytes/experiencia, coste de entrenamiento, retención, transferencia y coste por mejora.

Aceptación: mejora persistente sin que el hot path tenga que consultar continuamente el almacén de episodios.

## E54 — LLM access kill test

Después de producir Ψ₀, desconectar el LLM, bloquear tokenizer/logits/hidden states y ejecutar el rollout exclusivamente en Rust.

Gates:
- `llm_calls_after_encoder=0`
- `token_ids_seen_by_core=0`
- `tokenizer_calls_after_encoder=0`

Aceptación: rollout y evaluación pasan sin dependencia lingüística posterior a Ψ₀. La decodificación final debe quedar separada y auditada.

## E55 — Encoder swap

Encoders: Gemma hidden, otro encoder compatible, encoder semántico pequeño y control aleatorio. Congelar Dθ durante el test; documentar cualquier alineación aprendida.

Aceptación: medir cuánto sobrevive al cambio. Si el encoder requiere reentrenar la dinámica, registrar la dependencia; no declarar independencia.

## E56 — Escalabilidad

Dimensiones: 16, 32, 64, 128, 256, 512, 1024 según arquitectura. Medir FLOPs aproximados, p50/p95/p99 Rust, nodos activos, pasos, memoria, calidad, coste de entrenamiento y consolidación.

Aceptación: publicar curvas capacidad-coste y punto de ruptura. No extrapolar escalabilidad desde una única dimensión.

## E57 — Head-to-head con v4

Solo cuando v4 se haya cerrado y ambas ramas estén congeladas. Dataset común sellado, seeds emparejadas, decoder comparable, sin tuning posterior a TEST.

Competidores: v4 Dynamic Field, Liquid Rust denso, Liquid Rust sparse, Liquid con plasticidad/consolidación, baseline LLM, MLP recurrente y SSM.

Aceptar solo comparación reproducible con resultados positivos y negativos publicados. No fusionar por una mejora anecdótica.

## Exportación y equivalencia Python → Rust

Para cada modelo entrenado fuera de Rust:
1. exportar parámetros a un formato estable (por ejemplo, safetensors/NPZ para intercambio y un formato binario versionado del proyecto para runtime);
2. incluir manifiesto de nombres, formas, dtype, layout, activaciones y versión de arquitectura;
3. ejecutar un corpus fijo de vectores de entrada en framework y Rust;
4. medir error máximo, medio y p99 de cada estado intermedio y salida;
5. fijar tolerancias antes de ejecutar TEST;
6. verificar salidas discretas/decisiones cuando corresponda;
7. almacenar hashes y reporte de equivalencia.

Si una operación no tiene equivalencia exacta —por ejemplo, normalización, solver temporal, top-k con empates o reducción flotante—, definirla y testearla explícitamente. No modificar pesos para “arreglar” el resultado después de mirar TEST.

## Estados

Usar: `NOT_RUN`, `SMOKE_ONLY`, `PASS`, `PARTIAL`, `FAIL`, `LEAKED`, `DATASET_INVALID`, `PORT_PARITY_FAIL`.
