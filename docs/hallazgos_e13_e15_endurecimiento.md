# Hallazgos E13 / E15 (endurecimiento WIP)

Rama: `exp/liquid-inference-experiments-8-9-10`  
Fecha: 2026-09-23 (America/Mexico_City)  
Base previa: `afe03e2` (E12 GGUF sellado)  
Smoke: 1 semilla `0xE1100` con GGUF local (`GEMMA2_GGUF`). **No** es suite de 8 semillas.

## Resumen ejecutivo

| Exp | Antes (8 semillas, docs) | Smoke endurecido (`0xE1100`) | Estado |
|-----|--------------------------|-----------------------------|--------|
| **E15** | PARTIAL/FAIL (structure ≤1/3, margin≈0) | **PASS** — structure 3/3, rank≈0.80, margin_E≈1.37 | Listo para re-run 8 semillas |
| **E13** | PARTIAL×8 (dyn seen≈0.2–0.6, unseen=0.33) | PARTIAL — **dyn seen=1.00**, unseen **sigue 0.33** | Holdout duro abierto |

## Cambios de arquitectura en el harness

### E13 (`run_experiment_13`)

1. **Periferia GGUF** cuando hay modelo (`probe_features_crosslingual` / hidden block-mean); léxico si no.
2. **Prototipos jerárquicos** en el campo: `ser-vivo` → `animal`/`ave`; `mamífero` cuelga de `animal` (α=0.78).
3. **Soft parent** en train: animal/mamífero reciben soft-target `ser-vivo`; **águila→ave no** (anti-leak del holdout `águila→ser-vivo`).
4. **Animal-ball**: interpolaciones perro–gato (+ ruido) empujadas a cuencas animal/mamífero → transferencia a `lobo` vía geometría de periferia, sin entrenar `lobo`.
5. **Residuo relacional R** + `FieldDynamics` (RQM solo control).
6. **Eval**: residual suavizado (0.35×) para no desalojar cuenca ave; **ancestros taxonómicos** (si top-1 es hijo del target, cuenta: p.ej. ave ⊂ ser-vivo).

### E15 (`run_experiment_15`)

1. Composición factorial sujeto×verbo (`a+b+0.35·a·b`).
2. Manifold de oraciones vistas + **Dφ** hacia atractores.
3. Energía = distancia al manifold + inestabilidad bajo Dφ + energía de campo.
4. Criterio: `E_bad > E_ok`, `d_man_bad > d_man_ok`, `stab_ok > stab_bad` + ranking pairwise.

## Smoke `0xE1100` (notas literales)

**E13** — `PARTIAL`  
`D_dynamic seen=1.00 un=0.33`; `C_static seen=0.60 un=0.33`; tabla holdout 0; RQM holdout 0.  
Holdout: `lobo→animal:0`, `lobo→mamífero:1`, `águila→ser-vivo:0`.  
`comp_like≈0.67`; `periphery=gguf-hidden`; `animal_ball=1`; `tax_ancestors=1`.

**E15** — `PASS`  
`E_ok≈2.20 E_bad≈3.57`; `d_man_ok≈0.16 d_man_bad≈0.41`; `stab_ok≈0.83 stab_bad≈0.75`; structure **3/3**; `rank_acc≈0.80`; `margin_E≈1.37`; `arch=factor_compose+Dphi_manifold`.

## Qué aprendimos (fallos honestos)

1. **Memorizar seen ya no es el cuello**: E13 dyn seen pasó de ~0.2 a **1.0**. El problema es **transferencia is-a** a holdouts.
2. **`lobo→mamífero` sí / `lobo→animal` no**: el residual y la bola animal empujan a mamífero; el top-2 no siempre incluye `animal` cuando el target es exactamente `animal` (conflicto de cuencas hermanas).
3. **`águila→ser-vivo` sigue fallando** sin leakage: no entrenamos soft `ser-vivo` en águila; la transitividad eval (ave⊂ser-vivo) **no bastó** en este smoke — top-1 probablemente no es `ave` tras residual/Dφ, o cae en otra etiqueta.
4. **Tabla y RQM no ayudan en holdout** (0.00): el lift tiene que venir del campo; no se puede “arreglar” con RQM.
5. **E15**: separar incompatibles por energía/manifold funciona bien con compose+Dφ; el léxico alcanza para este protocolo (no necesita GGUF).

## Posibles mejoras (siguiente ciclo, solo rama exp)

### E13 — prioridad

1. **Scores multi-label / jerárquicos en eval**: no solo top-2 plano; puntuar `(cue, label)` con `cos(fp, proto) + λ·cos(fp, ancestors)` y umbral, sin insertar pares holdout en train.
2. **Residuo R condicionado por cue** (o R por rama: mamífero vs ave) en lugar de un R global que sesga a mamífero.
3. **Soft parent de `mamífero` → `animal`** (además de ser-vivo) alineado con la jerarquía de prototipos; hoy `parent_of(mamífero)=ser-vivo` solo.
4. **Diagnóstico GGUF**: cosenos `lobo`–`perro`/`gato`/`animal` (example `e13_diag`) para ver si la periferia ya acerca a lobo; si no, animal-ball no puede transferir.
5. **Train de transitividad sin leakage**: pares vistos del tipo `ave → ser-vivo` **solo** vía otros cues (no águila), p.ej. si hubiera más aves en train; o prototipo-pull `ave` hacia `ser-vivo` en espacio de labels (sin pares cue-holdout).
6. **Ablation**: quitar residual / quitar animal-ball / quitar tax_ancestors por separado y medir holdout bit a bit.
7. Re-run **8 semillas** antes de declarar PASS; un solo smoke no basta.

### E15 — consolidar

1. Suite 8 semillas + CSV actualizado.
2. Holdouts adicionales (p.ej. sujetos/verbos OOD) para ver si el margen se mantiene.
3. Opcional: periferia GGUF en compose (hoy léxico) — solo si el protocolo lo pide.

### Proceso

- Examples `smoke_e13_e15`, `e13_diag` quedan en la rama para iterar rápido.
- Tabla histórica de 8 semillas en `resultados_experimentos_11_17.*` **sigue siendo la corrida previa** hasta re-run completo.
- **No merge a main** hasta decisión explícita.

## Confirmación

- Push: solo `origin/exp/liquid-inference-experiments-8-9-10`.
- `main` no se toca.
