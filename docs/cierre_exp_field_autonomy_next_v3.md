# Cierre experimental — `exp/field-autonomy-next` (plan v3 / autonomía del campo)

**Rama:** `exp/field-autonomy-next`  
**Tip de evidencia (LONG v3.7 + confirmación stamp):** `33a35db0ccb77b429158b757e733e834a38b86df`  
**Contenido de palancas v3.7:** `0b000b113107325e349611f3db99168fbb418a6b`  
**Fecha de este cierre:** 2026-09-24 (America/Mexico_City)  
**Idioma:** español (términos técnicos en inglés cuando son identificadores del código)

> Este documento cierra el *ciclo experimental documentado* en la rama.  
> **No** es un merge a `main`. **No** inventa números: todos los veredictos
> citados salen de los CSV/MD listados al final.

---

## 1. Objetivo de la rama

Según [`plan_autonomia_campo_v3.md`](plan_autonomia_campo_v3.md):

La rama parte de `main` (no de pesos/datasets de ramas experimentales
anteriores) para una **segunda generación clean-room** que pregunte qué
capacidad permanece en el sustrato de campo cuando se eliminan mecanismos
de recuperación (RQM, tablas, NN, attractor banks, lookup CDT en eval).

Hipótesis a prueba:

> Experiencias previas pueden consolidarse en CDT como regularidades y
> modificar el aprendizaje de una dinámica del campo Dφ; después Dφ puede
> transformar estados nuevos y producir salidas que **nunca** fueron
> almacenadas como respuestas.

No cuenta como evidencia suficiente: encontrar un vecino, consultar una
tabla, recuperar de CDT/RQM o reproducir una trayectoria memorizada.

Módulo de ejecución: `src/field_autonomy_stage2_v2.rs`.  
Protocolo detallado: [`etapa_2_autonomia_campo_experimentos.md`](etapa_2_autonomia_campo_experimentos.md) §29+.

---

## 2. Protocolo clean-room

### Particiones

| Partición | Uso |
|---|---|
| **TRAIN** | aprendizaje |
| **DEV** | selección de hiperparámetros / early stopping |
| **TEST** | sellado **antes** del entrenamiento; no entra al pipeline |

Flujo obligatorio: `TRAIN → DEV → LOCK → TEST`. Tras mirar TEST no se
retunea; si se modifica algo → `TEST_INVALIDATED`.

### Seeds

| Rol | Rango | Notas |
|---|---|---|
| Desarrollo (DEV) | `0xA300–0xA307` (base) / LONG con `extra=true` → **`0xA300–0xA30F`** (16) | selección y LONG v3.x |
| Confirmación | **`0xB300–0xB30F`** (16) | held-out; hyperparams frozen desde `HyperparamLock::long()`; **sin retune** |

### Defaults FIELD_ONLY (eval)

- init aleatorio; CDT vacío al inicio del eval de lookup
- RQM / NN / table / attractor **OFF** en evaluación
- anti-leakage + anti-contaminación (`DATASET_INVALID` si falla el auditor)
- manifests con hash; leakage reportado por fila

Prohibiciones clean-room (extracto del plan): no reutilizar pesos
FieldEncoder/Dφ de E11–E17; no prototypes/targets de TEST; no ajustar
hiperparámetros mirando TEST; no CDT/RQM construidos con TEST.

---

## 3. Evolución de palancas v3.1 → v3.7

Ciclo de commits de palancas (rama únicamente; sin merge a `main`):

| Ver. | Commit | Palancas clave |
|---|---|---|
| **v3.1** | `01db3af` | static **sin** action (geom-only); soft-scale; unroll real E23; E22 compose vía PointDecoder mid |
| **v3.2** | `e265080` | features **Relative**/periódicas (desaturación static); MLP joint decoder; anneal h32 E23 |
| **v3.3** | `b612524` | hybrid soft-scale; E22 mid action-aware; horizontes mixtos E23; `dyn_updates=7` |
| **v3.4** | `09b83c1` | **dual FeatPath**: SoftScale (compose/rollout) vs Relative (desaturación static E18/E21/E24…) |
| **v3.5** | `8e96fb8` | E21 SoftScale train/dyn + Relative-only static probe; E22 compose-chain hop-2; nudge E23 |
| **v3.6** | `daa18dc` | E22 mid-refine + closed-loop hop-2; E23 mix v3.5 retenido |
| **v3.7** | `0b000b1` | E22 **mid-agree-gated** closed-loop + latent-T2 **hop\|shot**; E18/E24 dual-probe SoftScale+Relative (patrón E21); E21/E23 sin cambio |

### Levers que definen el cierre v3.7

1. **Dual FeatPath SoftScale / Relative** — SoftScale en train/dyn (E18/E21/E22/E23/E24); Relative-only como **static probe** (E18/E21/E22/E24) para desaturar el baseline estático sin colapsar el path dinámico.
2. **E22 compose / decoder** — mid-agree gate (evita envenenar con `mid_hat`); hop latent-T2 si el gate lo permite, si no single-shot; blend disagree `lin∩mlp`.
3. **E23 short + long** — mix compartido desde v3.5 (short TF+free-run + long TF); **sin** palanca nueva segura para h32/h64 en este ciclo.
4. **Static desaturation** — contraste geom-only + Relative probe; curriculum multi-familia; E21 dx denso.

Lock LONG v3.7: `enc_epochs=160 dyn_epochs=560 dyn_updates=7 train_n/dev_n/test_n=80/20/24`.

---

## 4. Resultados DEV finales (LONG v3.7, seeds `0xA300–0xA30F`)

Fuente: [`resultados_etapa_2_v3_long.md`](resultados_etapa_2_v3_long.md) /
[`resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv)  
Wall time ~1148 s · 240 filas · leakage>0 = **0** · `DATASET_INVALID` = **0**

### Gates principales (POS+STRONG / 16)

| Exp | Histograma (veredicto) | POS+STRONG |
|---|---|---|
| **E18** rule learning | NEG:1 NULL:1 POS:4 STRONG:10 | **14/16** |
| **E21** interp/extrap | NULL:1 PARTIAL:5 POS:1 STRONG:9 | **10/16** |
| **E22** composition RQM-off | NULL:2 PARTIAL:5 POS:8 STRONG:1 | **9/16** (PARTIAL+ **14/16**) |
| **E24** static vs dynamic | PARTIAL:2 POS:14 | **14/16** |
| **E23** rollout no-TF | PARTIAL:12 POS:2 STRONG:2 | **4/16** |

### Suite (resumen honesto)

| Exp | POS+STRONG | Nota |
|---|---|---|
| FASE_A cleanroom | 16/16 | scaffold FIELD_ONLY |
| E19 provenance | 15/16 | |
| E18C CDT C0–C6 | 1/16 | mayormente PARTIAL |
| E20 antimem | 0/16 | PARTIAL:16 |
| E25 CDT experience | 4/16 | |
| E26 ablation A–G | 0/16 | PARTIAL:16 |
| E27 transfer | 5/16 | |
| E28 forgetting | 0/16 | PARTIAL:16 |
| E29 causal | 0/16 | |
| E30 persistence | 0/16 | PARTIAL:16 |

Números de cuello (LONG md): E22 e2e mean **0.822**; E23 mean cos h1..h64 =
`[0.979, 0.95, 0.853, 0.684, 0.469, 0.325, 0.448]` — h32/h64 siguen abiertos.

---

## 5. Resultados confirmación `0xB300–0xB30F`

Fuente: [`resultados_etapa_2_v3_confirm.md`](resultados_etapa_2_v3_confirm.md) /
[`resultados_etapa_2_v3_confirm.csv`](resultados_etapa_2_v3_confirm.csv)  
Hyperparams = mismo `HyperparamLock::long()` congelado en DEV · **no retune** · ~1145 s · 240 filas

| Exp | Confirm hist | POS+STRONG |
|---|---|---|
| **E18** | POS:7 STRONG:9 | **16/16** |
| **E21** | PARTIAL:4 POS:6 STRONG:6 | **12/16** |
| **E22** | NULL:1 PARTIAL:3 POS:9 STRONG:3 | **12/16** |
| **E24** | POS:16 | **16/16** |
| E23 (no gate) | NULL:1 PARTIAL:12 POS:2 STRONG:1 | **3/16** |

**Confirmation: YES** — los cuatro gates (E18/E21/E22/E24) tienen mayoría
POS+STRONG en held-out `0xB300–0xB30F` (verificado fila a fila en el CSV).

---

## 6. Qué se considera cerrado vs abierto

### Cerrado en esta rama (ciclo v3.7 + confirmación)

- Protocolo clean-room v3 operable (TRAIN/DEV/TEST, lock, seeds A/B, FIELD_ONLY).
- Dual FeatPath SoftScale/Relative como patrón estable de desaturación.
- **Gates de confirmación** E18 / E21 / E22 / E24 en mayoría POS+STRONG
  (16/16, 12/16, 12/16, 16/16).
- Documentación reproducible: CSV+MD LONG y confirm, tip stampado.
- Decisión de **no** mergear a `main` hasta que gaps críticos se cierren o
  se redefina el criterio de producto.

### Abierto / gaps

1. **E23 h32/h64** — sigue abierto; sin palanca segura en este ciclo
   (intentos locales revirtieron o no mejoraron el cluster POS).
2. E22 residuales DEV (NULL en `0xA305`, `0xA30B`) y compose absoluto débil.
3. E18 residual DEV (1 NEG SoftScale collapse `0xA304`, 1 NULL).
4. Cadena central **experiencia→CDT→Dφ→salida nueva** (E25) y ablaciones
   (E26) siguen mayormente PARTIAL.
5. E20 / E28 / E29 / E30 — sin mayoría POS; no se reivindican.
6. E27 transfer — débil (5/16 confirm).
7. Escalado N, cross-lingual ampliado, cambio de LLM (fases 8–9 del plan)
   **no** ejecutados en este cierre.

---

## 7. Prohibiciones / no-claims

1. **No merge a `main`** desde este cierre. Push únicamente a
   `origin/exp/field-autonomy-next`.
2. **No evidencia falsa** — no citar PASS/STRONG que no estén en los CSV.
3. **No** afirmar consciencia, AGI, subjetividad ni “el campo ya es autónomo
   en producción”; el claim máximo permitido por el plan (§27–§28) exige
   checklist completo (leakage=0, RQM/CDT-retrieval/table/NN/attractor OFF,
   multi-seed, mejor que static/random, persistencia, intervención causal,
   transferencia) — **no todo está cerrado**.
4. Confirmación **no** autoriza retune; si se cambia arquitectura o lock
   tras ver `0xB*`, invalidar y generar nuevo TEST/seeds.
5. Resultados pre-cleanroom (`resultados_etapa_2_autonomia.*`, seeds
   `0xE1800..`) **no** se mezclan con v2/v3.
6. No reutilizar pesos ni datasets de E11–E17 ni de otras ramas exp.

---

## 8. Pointers a artefactos y tip SHA

| Artefacto | Ruta |
|---|---|
| Plan v3 | [`docs/plan_autonomia_campo_v3.md`](plan_autonomia_campo_v3.md) |
| Protocolo Etapa 2 §29+ | [`docs/etapa_2_autonomia_campo_experimentos.md`](etapa_2_autonomia_campo_experimentos.md) |
| LONG v3.7 (DEV) MD | [`docs/resultados_etapa_2_v3_long.md`](resultados_etapa_2_v3_long.md) |
| LONG v3.7 CSV | [`docs/resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv) |
| Confirmación MD | [`docs/resultados_etapa_2_v3_confirm.md`](resultados_etapa_2_v3_confirm.md) |
| Confirmación CSV | [`docs/resultados_etapa_2_v3_confirm.csv`](resultados_etapa_2_v3_confirm.csv) |
| Clean-Room v2 (histórico previo) | [`docs/resultados_etapa_2_cleanroom_v2.md`](resultados_etapa_2_cleanroom_v2.md) |
| Este cierre | [`docs/cierre_exp_field_autonomy_next_v3.md`](cierre_exp_field_autonomy_next_v3.md) |

**Tip SHA de evidencia al redactar este cierre:**  
`33a35db0ccb77b429158b757e733e834a38b86df`  
(`docs: stamp tip SHA for v3.7 LONG + confirmation results`)

Reproducir suite (referencia README / ejemplos Stage2 v2):

```bash
cargo run --release --example run_stage2_v2_dev
# confirmación: seeds 0xB300–0xB30F con HyperparamLock::long() frozen
```

---

## 9. Lectura en una frase

El ciclo v3.1→v3.7 en `exp/field-autonomy-next` deja **cerrados y
confirmados** (mayoría POS+STRONG en `0xB300–0xB30F`) los gates E18/E21/E22/E24
bajo clean-room FIELD_ONLY con dual FeatPath SoftScale/Relative; **sigue
abierto** el horizonte largo E23 (h32/h64) y la cadena experiencia-CDT /
transfer / forgetting / causal / persistencia — sin merge a `main` y sin
claims más allá de los CSV.
