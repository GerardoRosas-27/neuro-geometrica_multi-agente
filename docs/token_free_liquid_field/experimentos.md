# Batería experimental E45–E57

## E45 — Bottleneck token-free

### Pregunta
¿Cuánta dimensión necesita Ψ para conservar la información útil para la dinámica?

### Condiciones
N = 16, 32, 64, 128, 256, 512.

### Controles
- raw hidden state;
- random projection;
- trained FieldEncoder;
- PCA/linear bottleneck;
- nonlinear bottleneck.

### Métricas
Accuracy, OOD, cosine, mutual-information proxy, reconstruction solo como diagnóstico, leakage lexical, latencia.

### Aceptación
Existe un punto de compresión donde la calidad semántica y relacional se mantiene mientras el coste cae sustancialmente frente al hidden state completo.

No se acepta reconstrucción textual como única evidencia: eso permitiría ocultar información lexical en Ψ.

---

## E46 — Liquid vs recurrent MLP vs SSM

### Pregunta
¿La dinámica líquida aporta una ventaja real frente a una función recurrente convencional?

### Modelos
A. `Ψ'=MLP(Ψ)`
B. `Ψ'=Liquid(Ψ)`
C. SSM/Mamba-like compact state model
D. linear local dynamics

### Igualación
Mismo dataset, dimensiones cercanas, presupuesto de parámetros y presupuesto de entrenamiento comparable.

### Métricas
Error one-step, rollout h=1/2/4/8/16/32/64, estabilidad, Jacobiano, µs/query, memoria.

### Aceptación
El ganador debe demostrar ventaja en al menos dos dimensiones independientes: calidad OOD/rollout y eficiencia/estabilidad. Una sola mejora de velocidad no basta.

---

## E47 — Sparse Field

### Hipótesis
Solo una pequeña fracción de nodos participa en cada transición.

### k
1%, 2%, 5%, 10%, 20%, 50%, 100%.

### Variantes
- top-k por magnitud;
- threshold adaptativo;
- routing aprendido;
- bloque local.

### Aceptación
Reducción significativa de operaciones y latencia sin degradación estadísticamente relevante de calidad y sin aumento patológico de pasos.

---

## E48 — Inference adaptive / early stopping

### Hipótesis
No todas las consultas requieren el mismo número de pasos.

Detener cuando:

`||Ψ_t+1-Ψ_t|| < ε`

más estabilidad de energía y confianza mínima.

### Prueba
Comparar pasos fijos 1/2/4/8/16/32/64 contra stopping adaptativo.

### Aceptación
Menor coste medio manteniendo calidad OOD y evitando terminación prematura en consultas difíciles.

---

## E49 — Coarse-to-fine

Primera fase:

`Ψ₀ → Ψ_coarse`

Segunda fase solo si incertidumbre alta:

`Ψ_coarse → Ψ_fine`.

### Aceptación
Menor latencia media que full-resolution con degradación máxima predefinida de 1–2 puntos porcentuales en accuracy/OOD.

---

## E50 — Geometric trajectory beam

En vez de explorar tokens, generar K futuros de campo:

`Ψ₀ → {Ψ₁^1,...,Ψ₁^K}`

Cada rama se puntúa por estabilidad, energía, plausibilidad dinámica y consistencia.

### K
1, 2, 4, 8.

### Prohibido
Decodificar cada rama para seleccionar la respuesta. El ranking debe realizarse en el espacio de campo.

### Aceptación
La búsqueda mejora composición/OOD frente a greedy con un coste menor que beam search token-level equivalente.

---

## E51 — Memoria asociativa de campo

### Objetivo
Determinar si una memoria asociativa puede deformar la dinámica sin convertirse en un lookup de respuestas.

### Controles
- no memory;
- nearest neighbor;
- modern Hopfield;
- sparse associative memory;
- learned field attractors.

### Test
Target nunca almacenado directamente.

### Aceptación
La memoria debe cambiar la trayectoria o basin, no devolver el target.

---

## E52 — Plasticidad online

### Objetivo
Aprender durante operación sin retraining global.

Actualización candidata:

`θ_{t+1}=θ_t + η·G(error, Ψ_t, Ψ̂_{t+1})`

Variantes:
- low-rank;
- local Hebbian;
- predictive-error;
- energy deformation;
- sparse adapter.

### Aceptación
Nueva experiencia produce una mejora medible en consultas posteriores y el cambio es reproducible.

---

## E53 — Inferencia vs consolidación

Separar dos costes:

`hot path: Ψ → Dθ → Ψ'`

`sleep path: experiences → consolidation → Δθ`

Comparar:

1. liquid inference + no learning;
2. liquid + online plasticity;
3. liquid + batch consolidation;
4. Thermo CDT;
5. híbrido.

### Métricas
µs/query, ms/experience, bytes/experience, energía/cambio, retención y transferencia.

### Aceptación
La arquitectura debe demostrar que la consolidación no contamina el hot path de forma desproporcionada.

---

## E54 — LLM access kill test

### Objetivo
Demostrar que después de Ψ₀ el LLM es innecesario.

### Procedimiento
1. producir Ψ₀;
2. matar/desconectar el LLM;
3. bloquear tokenizer, logits y acceso a hidden states;
4. ejecutar todo el rollout;
5. decodificar únicamente al final con un decoder independiente.

### Acceptance gate
`llm_calls_after_encoder = 0`.

Además:
`token_ids_seen_by_core = 0`.

---

## E55 — Encoder swap

Encoders:
- Gemma hidden;
- otro LLM compatible;
- encoder semántico pequeño;
- encoder aleatorio control.

El `Dθ` entrenado debe permanecer congelado.

### Aceptación
El cambio de encoder no destruye completamente la dinámica cuando los encoders están alineados en una interfaz de campo común. Si falla, se documenta dependencia del encoder.

---

## E56 — Scaling

N = 16, 32, 64, 128, 256, 512, 1024 nodos/variables según arquitectura.

Medir:
- FLOPs aproximados;
- µs/query;
- active nodes;
- steps;
- memory;
- accuracy;
- OOD;
- consolidation cost.

### Aceptación
La curva de coste debe ser explícita y compararse con el crecimiento de capacidad. No se aceptan afirmaciones de “escalabilidad” sin curva.

---

## E57 — Head-to-head con v4

Solo después de cerrar v4 y congelar ambos sistemas.

### Protocolos
- dataset común sellado;
- mismos seeds;
- mismo número de tareas;
- decoder independiente equivalente;
- sin tuning posterior al test.

### Competidores
1. v4 Dynamic Field;
2. Token-Free Liquid;
3. Token-Free Liquid Sparse;
4. Token-Free Liquid Plastic;
5. LLM baseline;
6. recurrent MLP;
7. SSM.

### Aceptación
Publicar tanto victorias como derrotas. La rama experimental no se fusiona por rendimiento anecdótico.
