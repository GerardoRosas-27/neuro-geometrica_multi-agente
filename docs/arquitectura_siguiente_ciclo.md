# Arquitectura del siguiente ciclo

**De atractor único a memoria continua**

**Fecha:** 31 de agosto de 2026
**Precedente:** [`revision_proyecto.md`](revision_proyecto.md) (P0–P6, 30 ago 2026)
**Checkout evaluado:** `main` en `4e050ac` (PR #7 `mejora/p0-p6-tesis-cuenca`)
**Autor:** ingeniero de la IA, evaluación directa (tests + binario canónico)

Este documento **sí** se apoya en corridas. No afirma consciencia, julios ni
cognición general. Propone la arquitectura que el código tiene que convertirse
en, no otro motor con otro nombre.

---

## 0. Resumen ejecutivo

P0–P6 recortaron el relato y construyeron el andamiaje (tesis de cuenca, holdout,
baselines, ΔR, escala, CI en dos jobs, trainer gated, Gemma fuera del preprint).
Eso es higiene. **No es todavía un sistema de memoria.**

Las pruebas de este ciclo dejan tres hechos:

1. **La consolidación de un patrón verificado deforma la cuenca.** El protocolo
   pre/post se reproduce: 0/144 → 144/144, ocho semillas, 128 nodos. Es el
   resultado principal y sigue en pie.
2. **Hopfield clásico, Hopfield moderno y Hebb de aristas también recuperan al
   100 % el mismo patrón.** El brazo fasorial *sin* consolidación queda en el
   azar. La ventaja medida es «escribir el patrón en las fases», no un
   mecanismo CDT distinto de Hebb.
3. **Consolidar un segundo patrón borra el primero: ΔR = −1,0.** El olvido no
   está acotado. El test científico *reporta* ΔR y acepta `catastrophic_forgetting`
   como éxito. El preprint §4 queda falsado en el único ensayo que existe.

Por tanto la tesis vigente describe un **slot de un atractor**. Un slot no es
una IA. El siguiente ciclo no añade física, ni otro trainer infinito, ni otro
binario unificado. Construye una memoria de *K* patrones con replay, un gate
de commit que puede rechazar, y una curva de capacidad comparable a Hopfield.

**Tesis v2 (única, falsable):**

> El sistema almacena *K* patrones verificados sobre el mismo sustrato fasorial.
> Consolidar el patrón *k* deja la retención de un conjunto retenido *A* en
> ΔR ≥ −ε. La curva *K(N, ρ)* se publica junto a Hopfield y Hebb sobre las
> mismas cues. Si ΔR < −ε o la curva no discrimina, el commit se rechaza y
> no hay resultado.

Criterio de hecho de este documento: un extraño ejecuta un comando y obtiene
una tabla *(K, método, ΔR, recuperación, tiempo)* donde al menos un método
pierde y el nuestro no satura al 100 % en todos los *K*.

---

## 1. Qué se ejecutó

Máquina: Windows, `cargo` release, `.cargo/config.toml` con `target-cpu=native`.
Los wall-clock no son comparables entre máquinas; las tasas sí.

| Comando | Resultado |
|---|---|
| `cargo test --release --lib scientific -- --nocapture` | **no compilaba** en el checkout: `results` se mueve antes de `std_accuracy`. Arreglo local de una línea en `evaluate_basin`. Tras el arreglo: 5/5 ok |
| `cargo test --release --lib -- --skip scientific` | 195 ok, 3 ignored (GGUF), 0 fallos |
| `cargo run --release --bin native_consolidation_basin_experiment` | `basin_expansion_pass` + tabla de baselines |
| `cargo run --release --bin native_gemma2_spin_infinite_trainer` | exit 1, mensaje de gate (correcto) |
| `cargo test --release --bin native_gemma2_spin_infinite_trainer infinite_trainer_stays_gated` | ok |
| Chat Gemma / trainer con modelo | **no** se reanudó. Opción C del preprint y P4 lo prohíben |

El preprint 0.6 (`docs/paper_inferencia_fasorial_consolidacion_cdt.md`) y
`docs/reproducibilidad.md` coinciden con los comandos. El binario canónico
reproduce la tabla 7.4 del manuscrito bit a bit en tasas; sólo cambia el
wall-clock.

---

## 2. Tablero P0–P6 contra los criterios de hecho

| Ítem | ¿Implementado? | Criterio de hecho | Veredicto de esta corrida |
|---|---|---|---|
| **P0** una tesis, un motor, un paper | README, preprint 0.6, `docs/archive.md`, 4 binarios de entrada, feature `research` en 3 huérfanos | un extraño corre un comando y obtiene el resultado | **Cumplido a medias.** El comando existe y pasa. VMC, PEPS, unificado, `cognitive_os`, generalización 100 % siguen compilando en el crate por defecto |
| **P1** romper el techo 100 % | holdout 48 nodos, `std_accuracy`, CI smoke vs scientific | una métrica principal deja de ser 1,0 *o* se explica el techo y se añade tarea más dura | **Techo del inyectado intacto (1,0 ± 0).** El holdout no inyectado sí baja a 0. El gate del holdout sólo exige `< 1,0`, así que 0,99 también pasaría |
| **P2** baselines externos | Hopfield, «moderno», fasorial pre, Hebb | tabla con ≥ 2 métodos ajenos | **Tabla sí; ventaja no.** Los dos Hopfield y Hebb empatan al 100 % con CDT post. Energía y saturación no son comparables entre métodos |
| **P3** Gemma o periferia o fuera | opción C en el preprint; system prompt en español; Dyamon fuera del historial de usuario | métrica de chat vs Gemma denso *o* el preprint no lo reclama | **Cumplido (opción C).** Test `system_prompt_keeps_dyamon_out_of_user_history` ok. No hay A/B de `wake_blend` |
| **P4** parar el infinito sin gate | `trainer_is_gated`; unbounded exige `GEMMA_SPIN_ALLOW_INFINITE=1` y checkpoint que pase | cada cifra del preprint se regenera con un comando | **Gate de arranque correcto.** `summary.json` (ciclo 491) está rancio respecto a `latest.json` (ciclo 1070, 9 lecciones vistas). El currículo local siguió girando después de P4 |
| **P5** higiene | `src/bin/archive/`, `.gitignore` de journals, `target-cpu=native` documentado | un binario por rol | **Cumplido en puntos de entrada.** El crate público sigue siendo un laboratorio de ~50 módulos |
| **P6** ΔR y escala | `run_bounded_forgetting`, `run_basin_scale_sweep` | ΔR medido; 128 en CI, 512/2048 fuera de smoke | **ΔR medido y falla: −1,0.** El test no es un gate. Escala 128 sigue en techo 100 %; no mide capacidad |

---

## 3. Resultados numéricos

### 3.1 Resultado principal (binario canónico, 32 nodos, 24 ensayos)

Semilla `0xBA51_CD72_2026`. `sleep_accepted=1`, `consolidated_edges=128`.
`wall_clock_seconds=0,0055`.

| Corrupción | Éxito pre | Éxito post | Exactitud pre | Exactitud post | Iter. pre | Iter. post |
|---:|---:|---:|---:|---:|---:|---:|
| 10 % | 0/24 | 24/24 | 0,497 ± 0,013 | 1,000 ± 0 | 76,8 | 26,1 |
| 20 % | 0/24 | 24/24 | 0,499 ± 0,015 | 1,000 ± 0 | 80,2 | 27,6 |
| 25 % | 0/24 | 24/24 | 0,500 ± 0,018 | 1,000 ± 0 | 79,8 | 26,3 |
| 30 % | 0/24 | 24/24 | 0,508 ± 0,021 | 1,000 ± 0 | 77,2 | 34,3 |
| 35 % | 0/24 | 24/24 | 0,500 ± 0,000 | 1,000 ± 0 | 72,6 | 35,1 |
| 40 % | 0/24 | 24/24 | 0,500 ± 0,018 | 1,000 ± 0 | 83,9 | 45,4 |

```text
rho_critica_pre=0.00
rho_critica_post=0.40
ganancia_media=1.00
decision=basin_expansion_pass
```

Reproduce la tabla 7.4 del preprint. El 100 % post del patrón inyectado es el
techo esperado de escribir un atractor. No se cita como cognición.

### 3.2 Multisemilla (6 ensayos/nivel, semillas `0xA11CE001`–`008`)

Ocho de ocho: `basin_expansion_pass`, ganancia 1,00, pre_crit=0, post_crit=0,40.
Exactitud post 1,000 ± 0 en todos los niveles. Exactitud pre ~0,46–0,56.
Wall-clock ~2–3 ms/semilla. El fixture no depende de una semilla accidental.
Tampoco discrimina: nunca deja de saturar.

### 3.3 Holdout (48 nodos, semilla `0xC0FF_EE42_2026`, ruido de grafo 15 % ± 0,40 rad)

| Patrón | Corrupción | Éxito | Exactitud |
|---|---:|---:|---:|
| inyectado | 20 % | 1,000 | 1,000 ± 0,000 |
| inyectado | 45 % | 0,625 | 0,812 ± 0,259 |
| inyectado | 55 % | 0,000 | 0,125 ± 0,232 |
| no inyectado | 20 % | 0,000 | 0,516 ± 0,041 |
| no inyectado | 45 % | 0,000 | 0,508 ± 0,029 |
| no inyectado | 55 % | 0,000 | 0,484 ± 0,041 |

```text
injected_mean_success=0.542 ± 0.505
non_injected_mean_success=0.000 ± 0.000
decision=holdout_discriminates
```

El no inyectado no satura: hay especificidad. El inyectado a 55 % cae *por
debajo del azar* (0,125). Eso es compatible con un flip global Z₂: la
corrupción supera mayoría, el cue empuja al anti-patrón, y la exactitud directa
lo cuenta como fallo (correcto según el protocolo). El holdout **mezcla** tres
efectos: patrón nunca escrito, ruido de grafo *después* del sueño, y cruce del
umbral de mayoría. No se pueden atribuir por separado.

### 3.4 Baselines (mismo fixture, 24 ensayos, presupuesto 300)

| Método | Ajeno | Éxito | Exactitud | Energía (del modelo) | Tiempo (s) |
|---|---|---:|---:|---:|---:|
| hopfield | sí | 1,000 | 1,000 | −496 | 3,9×10⁻⁴ |
| hopfield_moderno | sí | 1,000 | 1,000 | −0,47 | 1,2×10⁻⁵ |
| fasorial_sin_consolidacion_cdt | no | 0,000 | 0,501 | 1,13 | 3,1×10⁻³ |
| hebb_aristas | no | 1,000 | 1,000 | 4,1×10⁻⁵ | 1,8×10⁻³ |
| fasorial + sueño CDT (post) | no | 1,000 | 1,000 | ~2,5×10⁻⁵ | 5,5×10⁻³ (experimento entero) |

No hay ventaja de recuperación. Hay desventaja de tiempo. Las energías viven
en funcionales distintos y no se comparan. `mean_saturation` del brazo fasorial
copia `mean_accuracy`: la columna de saturación no mide saturación.

`hopfield_moderno` con un solo patrón es `sign(⟨ξ, q⟩)·ξ`. No es el Hopfield
exponencial de Ramsauer et al. (softmax sobre *varios* recuerdos). El nombre
sobra hasta que haya *K>1*.

### 3.5 Escala 128

```text
nodes=128 decision=basin_expansion_pass gain=1.0 wall=0.0017s
```

Un patrón, una corrupción (25 %), dos ensayos. El techo se mueve de 32 a 128
nodos. Eso no es escalabilidad de *capacidad* ni de *tiempo asintótico*.
512 y 2048 no corrieron (fuera de smoke, como documenta reproducibilidad).

### 3.6 Olvido acotado ΔR (preprint §4)

```text
retention_before=1.000
retention_after=0.000
delta_r=-1.000
epsilon=0.10
second_sleep_accepted=1
decision=catastrophic_forgetting
```

Consolidar B **borra** A. El segundo sueño se acepta. El test
`scientific_bounded_forgetting_reports_delta_r` da ok porque la aserción es
«la decisión es uno de cuatro strings», no «ΔR ≥ −ε».

Este es el resultado más importante del ciclo. Todo lo demás (cuenca, holdout,
baselines) es consistente con un único atractor escrito en las fases de arista.

### 3.7 Trainer y crate

- Arranque sin env: *entrenador gated…* (correcto).
- Unit test del predicado: ok.
- Artefacto local `data/gemma2_developmental_infinite_training/latest.json`:
  ciclo **1070**, 9 lecciones vistas, 9 aceptadas, 236 planes de lenguaje
  aceptados. `summary.json` sigue en ciclo **491**, `symbolic_accuracy=0`,
  `functional_cognition_gate=false`. El resumen publicado está desfasado del
  checkpoint.
- `functional_cognition_gate` **no incluye** `symbolic_accuracy`. Las etapas
  ≥ 5 son las simbólicas; el gate puede, en principio, pasar con simbólico a 0.
- Smoke: 195 tests en 0,37 s, incluidos los protocolos cognitivos al 100 %
  (`cognitive_generalization_benchmark`, `advanced_cognitive_validation`).
  Siguen en CI como si fueran evidencia.

---

## 4. Temas que salieron de las pruebas

Cada tema es un requisito de arquitectura, no un comentario.

### T1 — El 100 % post es Hebb, no un fenómeno raro

Hebb escribe `edge_phase = 0 o π` según el signo de los extremos. El sueño
del experimento escribe `arg(z_b) − arg(z_a)` con `learning_rate=1`. Con un
patrón completo anclado, son la misma operación. `cdt_consolidation_steps = 0`
en `training_engine`: **no hay paso CDT**. Pachner no se simula (el README
ya lo decía). El nombre «consolidación CDT» en este fixture es escritura
Hebbiana de fases con un gate de revalidación.

Arquitectura: o se enciende una dinámica de sustrato que Hebb no tiene, o se
deja de llamar CDT a esa escritura.

### T2 — Memoria de un slot

ΔR = −1 implica que el paisaje de fases no superpone recuerdos. El segundo
commit sustituye las fases del primero. Un sistema de dos tiempos (vigilia /
sueño) sin interferencia acotada es una cola, no una memoria.

Arquitectura: el commit tiene que ser una función de *(candidato, memoria
ya consolidada)*, no de *(candidato,)* solo.

### T3 — Los tests científicos no pueden perder de forma informativa

| Test | Qué afirma el nombre | Qué aserta de verdad |
|---|---|---|
| `scientific_holdout_*` | el holdout discrimina | `non_injected_mean_success < 1` |
| `scientific_basin_scale_*` | hay escala | `rows[0].nodes == 128` |
| `scientific_bounded_forgetting_*` | hay ΔR | `delta_r.is_finite()` y el string es conocido |
| `scientific_basin_baselines_*` | hay dos métodos ajenos | los nombres existen en la tabla |

Ninguno exige ventaja, ΔR ≥ −ε, ni que la escala conserve el gate. CI verde
no es evidencia. El gate científico tiene que poder ponerse rojo.

### T4 — Z₂ contra la exactitud directa

A 55 % de corrupción el cue ya no conserva la convención de signo
mayoritaria. El funcional es simétrico. Recuperar el anti-patrón es un mínimo
válido y un fallo métrico. Eso está bien como definición, mal como único
número publicado: hay que reportar juntas exactitud directa, gauge-invariante
y lado de mayoría del cue. Si no, el holdout a >50 % mide el umbral de
mayoría, no la cuenca.

### T5 — Confundir tamaño con capacidad

128 nodos × 1 patrón no pregunta cuántos atractores caben. La pregunta de
Hopfield es *K_max(N)* bajo una corrupción fija. Sin esa curva no hay
comparación con memoria asociativa, que es el baseline que P2 introdujo.

### T6 — Métricas de costes mezcladas

Tiempo, energía del funcional fasorial, energía de Hopfield (−½ ξᵀWξ) y
«saturación» (= exactitud) se imprimen en la misma tabla. Un revisor las
leerá como julios o como la misma U. Hay que separar: (a) recuperación,
(b) ΔR, (c) wall-clock, (d) energía *dentro de un mismo funcional*.

### T7 — El crate todavía es un laboratorio

Tres módulos bajo `research`. El resto (VMC, PEPS, OS cognitivo, logística,
familias de transformaciones, trainer unificado) sigue en `lib.rs`. Los
binarios se archivaron; los *claims* de test no. Cada test al 100 % en smoke
vuelve a vender cognición emergente en el log de CI.

### T8 — Compilación rota al añadir `std_accuracy`

P1 pidió intervalos. El parche recorre `results` por valor y luego calcula
la desviación. `cargo test --lib scientific` (el job de CI) no compilaba en
este checkout. Un gate científico que no compila no es un gate.

### T9 — Gemma ya no es el paper; el chat sigue siendo un modelo 2B con sesgo

Opción C es la decisión correcta. El system prompt ancla idioma e identidad.
Sigue sin haber métrica de que el motor *obligue* a abstenerse cuando no hay
atractor. El banner del chat lo dice; el decode no tiene un aborto duro si
Gemma ignora el ancla.

### T10 — El trainer gated no congela los artefactos locales

El predicado funciona. El directorio de entrenamiento local avanzó de 491 a
1070. `summary.json` no se reescribe al ritmo de `latest.json`. Publicar el
resumen rancio como estado actual viola P4 («no el número suelto»).

---

## 5. Qué sistema hay realmente

```text
cue binario verificado
        │
        ▼
  infer_and_stage          acoplamiento = 0 en adquisición
        │                  (el solver no descubre el patrón)
        ▼
  pending[0]               cola de un prototipo
        │
        ▼
  sleep_consolidate        revalida, writing rate=1
        │                  cdt_consolidation_steps = 0
        ▼
  edge_phase[e] ← arg(z_b)−arg(z_a)     ≡ Hebb de signo
        │
        ▼
  evaluate_basin           acoplamiento = 2, 300 iter Armijo
        │
        ├── patrón A: cuenca amplia (techo 100 %)
        └── patrón B después: A desaparece (ΔR = −1)
```

No hay superposición de recuerdos. No hay replay de A al escribir B. No hay
paso de triangulación. No hay periferia lingüística en el loop de cuenca.
Hay un minimizador de fasores bueno y un gate transaccional de sueño que
protege contra fallos de escritura, no contra interferencia.

Eso basta para el preprint de cuenca (un patrón, deformación causal). No
basta para una IA.

---

## 6. Arquitectura propuesta: CL-MPM

**Complementary Learning + Multi-Pattern Phasor Memory.**

No es un motor nuevo. Es el motor híbrido actual con un contrato de memoria
que hoy no existe. Tres planos, un bus de commit, ningún LLM en el claim.

### 6.1 Planos

```text
┌─────────────────────────────────────────────────────────────┐
│ Plano L  (fuera del claim)                                  │
│  Gemma 2 u otro compilador: texto ⇄ receta/cue.             │
│  Si el plano I no infiere, L se abstiene. Sin α en el paper.│
└───────────────────────────┬─────────────────────────────────┘
                            │ receta | abstención
┌───────────────────────────▼─────────────────────────────────┐
│ Plano I  vigilia                                            │
│  Campo de fasores. Minimiza F = U − T S. Cue = frontera.    │
│  Sale un prototipo + residuo + estabilidad. Nunca escribe   │
│  el paisaje persistido.                                     │
└───────────────────────────┬─────────────────────────────────┘
                            │ PendingAttractor
┌───────────────────────────▼─────────────────────────────────┐
│ Plano M  sueño / memoria lenta                              │
│  1. Replay intercalado del conjunto retenido A.             │
│  2. Candidato B se escribe en un *buffer de fases*.         │
│  3. Sonda ΔR sobre A y B en el buffer.                      │
│  4. Commit transaccional al core CDT o rollback.            │
│  5. Capacidad: rechazo si K supera el presupuesto o ΔR.     │
└─────────────────────────────────────────────────────────────┘
```

El plano M es el que falta. Hoy el sueño es el paso 2 sin 1, 3, 4 (ΔR) ni 5.

### 6.2 Representación: de un vector de fases a un código de *K* recuerdos

Tres diseños, en orden de riesgo. Se implementa **R1** primero. R2 y R3
sólo si R1 no alcanza ΔR ≥ −ε con *K* ≥ 4 en 32 nodos.

**R1. Superposición Hebbiana de fases (mínimo, comparable a Hopfield).**

Para cada arista *e=(i,j)* y patrones ξ¹…ξᴷ ∈ {±1}ᴺ:

```text
W_e = Σ_k ξ_i^k ξ_j^k          (acumula, no sustituye)
φ_e = arg(W_e)                 (0 o π si W_e real)
```

El commit de B **suma** el término de B, no asigna. Hopfield clásico ya hace
esto en pesos; el fasorial debe hacer lo análogo en `edge_phase` /
`edge_weight`. Es el único cambio que vuelve ΔR interpretable.

Invariante: `hebb_aristas` del baseline pasa a ser exactamente el acumulador
R1, no la asignación del patrón corriente. Hoy Hebb también sustituye.

**R2. Anclaje disperso por episodio (ya existe a medias).**

`anchored_consolidation` y `anchors[]` ya evitan escribir nodos no anclados.
Falta anclar *por patrón* y no reescribir anclas ajenas:

```text
rate(e, k) = lr · min(anchor_k(i), anchor_k(j)) · (1 − max_{m≠k} occupancy_m(e))
```

Un episodio no puede bajar la ocupación de otro por debajo de un suelo.
Eso es interferencia acotada a nivel de arista.

**R3. Módulos / códigos casi ortogonales.**

Reservar bloques de nodos o canales de fase para esquemas distintos, o
proyectar patrones a vectores con ⟨ξᵃ, ξᵇ⟩ ≈ 0. Sólo si R1+R2 no bastan.
No se llama «simetría de gauge» ni «CDT» a una máscara de índices.

### 6.3 Vigilia (plano I): sin cambios de tesis

Se conserva:

- acoplamiento nulo en *adquisición* del patrón a consolidar (aisla la
  escritura);
- acoplamiento idéntico y no nulo en evaluación pre/post;
- exactitud directa como métrica de gate, gauge-invariante como diagnóstico;
- Z₂ = fallo;
- Armijo, rollback transaccional, cola `pending`.

Se añade:

- la cue de evaluación de *A* nunca se usa como frontera durante el commit
  de *B* (ya es la política de `sleep_replay_boundary_gain`; mantenerla);
- un presupuesto de iteraciones *compartido* con los baselines.

### 6.4 Sueño: replay intercalado antes de escribir

Pseudocódigo del commit, el único algoritmo nuevo que importa:

```text
fn commit(candidato B, memoria M, retenidos A, ε):
    snapshot = clone(M.core)
    buffer   = M.core

    # 1. escribir B en el buffer (R1: suma, no asigna)
    apply_pattern(buffer, B)

    # 2. replay: para cada a en A, revalidar sin frontera
    for a in A:
        if not revalidate(buffer, a):
            restore(snapshot); return Reject("replay_failed")

    # 3. sonda ΔR (mismas cues, mismo solver, mismo presupuesto)
    R_before = mean_success(M.core, A)
    R_after  = mean_success(buffer, A)
    delta_r  = R_after - R_before
    if delta_r < -ε:
        restore(snapshot); return Reject("delta_r", delta_r)

    # 4. sonda de B: tiene que seguir siendo recuperable
    if mean_success(buffer, [B]) < θ_B:
        restore(snapshot); return Reject("candidate_not_retained")

    # 5. commit
    M.core = buffer
    M.patterns.push(B)
    return Accept(delta_r)
```

Hoy los pasos 2–4 no existen. `storage_delta_free_energy` mide novedad del
candidato contra prototipos fusionables, no retención de A. No sustituye ΔR.

El rollback ya clona el core. Reutilizarlo. No clonar el motor fasorial
entero: `recompile_from_core` ya es la vía barata.

### 6.5 Gate de commit: contratos numéricos versionados

Nada de literales invisibles. Todo en config, como `minimum_mean_success_gain`.

```text
BoundedMemoryConfig {
    epsilon: 0.10,                  // ΔR ≥ −ε
    theta_candidate: 0.80,          // B recuperable post-commit
    max_patterns: None,             // None = medir K_max; Some = techo duro
    replay_rounds: 1,
    evaluation: ConsolidationBasinConfig { trials, corruptions, seed, ... },
}
```

El test `scientific_bounded_forgetting` pasa a exigir
`decision == "bounded_forgetting_pass"` **después** de implementar R1.
Hasta entonces el test debe marcarse `#[ignore = "R1 pendiente"]` o
`should_panic` documentado. Un ok que esconde ΔR = −1 es peor que un fail.

### 6.6 Experimento principal del ciclo: curva de capacidad

Sustituye a «otro tamaño de un patrón» como siguiente figura del preprint.

Protocolo `run_capacity_curve`:

- N ∈ {32, 64, 128}
- K = 1, 2, 4, 8, … hasta fallo
- patrones equilibrados, semillas fuera del fixture de desarrollo
- corrupción ρ ∈ {0,20, 0,30}
- métodos: fasorial+commit R1, Hebb acumulativo, Hopfield clásico,
  Hopfield exponencial *de verdad* (softmax, β versionado, *K* prototipos)
- métricas por (N, K, ρ, método):
  - recuperación media ± std (no un solo 1,0)
  - ΔR del conjunto {1…K−1} al insertar K
  - wall-clock
  - energía *dentro del funcional de cada método*, nunca cruzada

**Pasa** si existe una región (N, K, ρ) donde el fasorial no es peor que
Hopfield en recuperación y cumple ΔR ≥ −ε. **No pasa** si sólo gana en
K=1, que es lo que hay hoy.

512/2048 se aplazan hasta que K>1 funcione en 32. Agrandar N con K=1
repite T5.

### 6.7 Periferia lingüística (se mantiene opción C)

El preprint no reclama Gemma. El chat es demo. El siguiente trabajo de L,
si se hace, es un **compilador con aborto**:

1. parsear receta o cue estructurada;
2. llamar al plano I;
3. verbalizar el resultado verificado;
4. si I se abstiene o el parseo falla, emitir la abstención y **no**
   continuar el decode libre.

Eso no entra en la tabla de cuenca. No hay A/B de `wake_blend` hasta que
exista un set fijo de prompts y un juez que no sea el propio 2B.

### 6.8 Módulos y API (sin quinto binario)

Un rol nuevo se cubre **ampliando** `native_consolidation_basin_experiment`,
que ya imprime cuenca + baselines. Añadir al mismo JSON:

```text
"capacity": { ... }
"bounded_forgetting": { ... }
```

Módulos:

| Pieza | Dónde | Notas |
|---|---|---|
| acumulador R1 | `native_hybrid_phasor_cdt_engine::apply_pattern_additive` | no un crate nuevo |
| commit con ΔR | `consolidation_basin_experiment::commit_with_retention` | reusa `evaluate_basin` |
| curva de capacidad | `basin_external_baselines::run_capacity_curve` | Hopfield ya está; hacerlo *K*-patrones |
| Hopfield exponencial | el mismo archivo | softmax sobre K, no `sign(overlap)` |
| saturación real | overlap absoluto ≠ exactitud | arreglar el bug de la tabla |
| feature `research` | más módulos, no menos | VMC, PEPS, `cognitive_os`, generalización 100 %, logística |

Binarios: siguen cuatro. El visualizador y el chat no tocan este ciclo.
El trainer permanece gated y **no** se reanuda.

### 6.9 Invariantes que el código debe cumplir

1. Escribir el patrón *k* no asigna `edge_phase`; acumula.
2. Todo commit toma snapshot y puede restaurar.
3. ΔR se mide con las mismas cues que la cuenca, no con una cue nueva.
4. Exactitud directa y gauge-invariante se publican juntas; el gate usa
   la directa.
5. Ninguna energía se compara entre funcionales.
6. Ningún test `scientific_*` aserta sólo que un string es conocido.
7. `cdt_consolidation_steps > 0` o se deja de etiquetar el brazo como CDT
   en la tabla publicada.
8. No se añaden nombres de física (Pachner, EPR, líquido de espines) a
   las APIs de R1.

---

## 7. Protocolo experimental P7–P12

Orden estricto. Cada ítem tiene comando, n, y un fail que CI debe poder dar.

### P7 — Arreglar el instrumento (esta semana)

- Compilación de `std_accuracy` (ya hecha en este checkout).
- `scientific_bounded_forgetting` deja de aceptar `catastrophic_forgetting`
  como ok, *o* se marca `ignore` con la razón R1.
- Holdout: asertar `non_injected_mean_success ≤ 0,25` **y**
  `injected_mean_success(ρ=0,20) ≥ 0,80`, no sólo `< 1`.
- Baselines: no copiar exactitud a saturación; documentar que
  `hopfield_moderno` es placeholder hasta K>1.
- Imprimir los reportes científicos (ya hecho: `eprintln` bajo `--nocapture`).

Fail: `cargo test --release --lib scientific` no compila o no imprime ΔR.

### P8 — R1 aditivo y ΔR como gate

Implementar acumulación. Correr:

```powershell
cargo test --release --lib scientific_bounded_forgetting -- --nocapture
```

Pasa: `decision=bounded_forgetting_pass`, `delta_r ≥ −0,10`,
`retention_after ≥ 0,80` en 32 nodos, 2 patrones, ρ=0,20, 8 ensayos.
Fail: ΔR < −ε. Entonces no se toca el preprint; se itera R1/R2.

### P9 — Curva de capacidad vs Hopfield/Hebb

```powershell
cargo run --release --bin native_consolidation_basin_experiment
```

(el mismo binario, JSON ampliado). N=32, K=1..8, ρ=0,20 y 0,30, 8 semillas
nuevas. Publicar media ± std.

Pasa: existe K≥2 donde el fasorial cumple ΔR y no pierde >10 puntos de
recuperación frente a Hopfield. Fail: sólo gana en K=1.

### P10 — Holdout desmezclado

Tres brazos, no uno:

1. patrón no inyectado, grafo *sin* ruido;
2. patrón inyectado, grafo *con* ruido;
3. corrupción 45/55 % con reporte de mayoría del cue y gauge-invariante.

Pasa: (1) no inyectado ~ azar, (2) inyectado no colapsa a 0 por 15 % de
ruido, (3) se ve el cruce de mayoría. Fail: si el «holdout» sólo era el
ruido o sólo era Z₂.

### P11 — Escala de capacidad, no de un patrón

Cuando P8–P9 pasen, repetir P9 en N=64 y N=128. 512/2048 quedan como
job nightly, no como smoke. El test CI de 128 nodos aserta
`decision` **y** que K_max ≥ 2, no sólo `rows[0].nodes==128`.

### P12 — Recorte real del crate

- `cognitive_os`, `cognitive_generalization_benchmark`,
  `advanced_cognitive_validation`, `cognitive_logistics`,
  `transformation_family_discovery` → `research` o fuera de smoke.
- VMC/PEPS/unificado: o `research`, o un párrafo en archive que deje de
  decir «siguen compilando porque el unificado los invoca» como si eso
  los justificara en el crate público.
- Un job CI `cargo test --release --lib --features research` semanal, no
  en cada push.

El paper de 8–12 páginas se reescribe **después** de P8–P9, no antes.
La tabla nueva es capacidad + ΔR. La tabla 7.4 de un patrón pasa a
«resultado previo, un slot».

---

## 8. Plan de 30 días (v2)

Sustituye el plan de `revision_proyecto.md` §7. Ese plan ya se ejecutó.

| Semana | Objetivo | Entregable que puede fallar |
|---|---|---|
| 1 | P7 instrumentos | CI scientific compila, imprime números, ΔR no se esconde |
| 2 | P8 R1 + commit con replay | test de ΔR rojo o verde de verdad; si rojo, no se finge |
| 3 | P9 curva K=1..8 vs Hopfield/Hebb | una tabla, un JSON del binario canónico |
| 4 | P10 holdout desmezclado + recorte `research` | preprint 0.7 con tesis v2 o con ΔR negativo publicado como resultado |

Si la semana 2 da ΔR < −ε con R1, la semana 3 es R2 (anclas por patrón),
no la curva de marketing. Si R1+R2 fallan, se publica el olvido catastrófico
como hallazgo del laboratorio y se detiene la analogía «memoria de dos
velocidades».

---

## 9. Qué no hacer

- No añadir un quinto binario «unificado» ni un motor con otro acrónimo.
- No reanudar `native_gemma2_spin_infinite_trainer` para ver si el ciclo 1070
  desbloquea lo simbólico. El gate está; el currículo sigue siendo el mismo
  techo sintético.
- No citar 100 % post-sueño, 100 % Hopfield o 100 % Hebb como ventaja.
- No correr 512/2048 nodos con K=1 y llamarlo escala.
- No comparar −496 de Hopfield con 10⁻⁵ del funcional fasorial.
- No llamar Hopfield moderno a `sign(overlap)` de un patrón.
- No devolver a Gemma al claim sin un set de prompts y un juez externo.
- No fabricar chip, consciencia ni SO cognitivo.
- No escribir Pachner, EPR o líquido de espines en la API de R1.
- No dejar un test científico que acepte el fallo que el preprint declara
  como predicción falsable.

---

## 10. Mapa de decisión

```text
¿ΔR ≥ −ε con K=2, R1, N=32?
    no ──► publicar olvido catastrófico; recortar el relato a «escritura
            de un atractor». Fin de la analogía CLS.
    sí
      │
      ▼
¿curva K(N) no peor que Hopfield en alguna región?
    no ──► el fasorial es un Hopfield más lento. Útil como solver de F,
            no como memoria. El paper se queda en cuenca de un patrón.
    sí
      │
      ▼
Holdout desmezclado + crate recortado + preprint 0.7 con tesis v2.
El chat sigue siendo demo. El trainer sigue gated.
```

Ese es el trabajo de un ingeniero que está construyendo una IA: no el
siguiente nombre físico, sino el primer sistema que **conserva** lo que
aprendió ayer.

---

## 11. Apéndice — cambios locales de esta evaluación

No forman parte de P7 completo; desbloquean la corrida y hacen visibles
los números:

1. `evaluate_basin`: iterar `&results` para poder calcular `std_accuracy`
   (el checkout en `4e050ac` no compilaba el job scientific).
2. `eprintln` de reportes en los tests `scientific_*` para que
   `--nocapture` de CI imprima media ± std, ΔR y la tabla de baselines.

Comandos de regeneración: los de [`reproducibilidad.md`](reproducibilidad.md),
más este documento para interpretarlos. Las cifras de las secciones 3.x
salen de las corridas del 31 ago 2026 en esta máquina.
