# Experimento: campo sin tokens

**Fecha:** 13 de septiembre de 2026 (protocolo v2)  
**Rama:** `exp/campo-sin-tokens`  
**Módulo:** `src/field_substrate.rs`  
**Binario:** `native_field_substrate_experiment`  
**Qué es:** un sustrato `Ψ = (T, z, m, C)` — complejo simplicial 2D, fasores, máscara de consolidación, cavidades entre caras rígidas. Cero `vocab_size`. Cero next-token. T puede crecer (subdividir caras rígidas) y podarse (aristas libres no semilla).

## Qué no reclama

No consciencia. No ventaja frente a un transformer. No cifra del preprint de cuenca. No tok/s ni Gemma. Las cavidades no son un cálculo QFT de Casimir: `lambda_min` es el modo más bajo de L1 restringido al puente; `bridge_penalty` es una penalización de amplitud en esas aristas. El 2-esqueleto no se fusiona con `simplicial_thermodynamic_engine` (tetraedros 3D).

## Alcance de merge

Este experimento es el módulo, el binario, este doc y las menciones en README / reproducibilidad / archive. **No** depende de Layer Route Cache, skip de capas ni el grafo de 6 agentes (PR #9). Si la rama nació encima de `feature/optimizacion-velocidad-rutas`, un merge a `main` debe cherry-pickear solo estos archivos.

## Ciclo medido (comparación)

1. Escribe un patrón hemisferio **6+6**: radios y ecuador opuesto del polo norte (vértice 0) en fase 0; polo sur (vértice 1) en fase π.
2. Handshake si la correlación de fase ≥ 0,80; Hebb local en `L1`; cavidades entre caras rígidas.
3. Enmascara 4 aristas (dos norte, dos sur). Pone a cero **`z` y `z_past`** de esa región: el atractor no filtra la respuesta.
4. Relaja 80 pasos (Langevin + atención geométrica) y reconstruye.
5. Mismos cue y máscara para cuatro brazos: no hacer nada, promedio de vecinos visibles, Hopfield complejo (regla de proyección), Hebb/Jacobi sobre `L1`.

Un segundo protocolo guarda dos patrones (hemisferio + meridiano) en el mismo `L1` y puntúa cada uno.

## Comando

```powershell
cargo test --release --lib field_substrate -- --nocapture
cargo run --release --bin native_field_substrate_experiment
```

Dataset grande, un directorio por `dataset_id`, resume atómico:

```powershell
cargo run --release --bin native_field_substrate_experiment -- --dataset-size 4096 --dataset-id synthetic-4k
```

Artefactos en `data/field_substrate_training/<id>/`: `latest.json`, `checkpoints/step-*.json`, `metrics.jsonl`, `summary.json`. Ctrl+C persiste y sale. Un checkpoint de otro `dataset_id` o semilla se rechaza.

Semillas `0..7`, Xoshiro256**. Determinista.

## Resultados (comparación, 8 semillas)

Windows, `cargo test --release --lib field_substrate -- --nocapture`. MSE sobre 4 aristas ocultas. Baseline cero = 1 por construcción (fasores unitarios).

| seed | F antes | F después | HS | rígidas | cav | field | zero | neigh | hop | hebb |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 45,796 | 38,895 | 0,962 | 8 | 1 | 0,359 | 1,000 | 0,250 | 0,000 | 0,357 |
| 1 | 45,796 | 38,894 | 0,964 | 8 | 1 | 0,373 | 1,000 | 0,250 | 0,000 | 0,357 |
| 2 | 45,796 | 38,894 | 0,963 | 8 | 1 | 0,367 | 1,000 | 0,250 | 0,000 | 0,357 |
| 3 | 45,796 | 38,894 | 0,962 | 8 | 1 | 0,357 | 1,000 | 0,250 | 0,000 | 0,357 |
| 4 | 45,796 | 38,895 | 0,962 | 8 | 1 | 0,358 | 1,000 | 0,250 | 0,000 | 0,357 |
| 5 | 45,796 | 38,896 | 0,962 | 8 | 1 | 0,360 | 1,000 | 0,250 | 0,000 | 0,357 |
| 6 | 45,796 | 38,894 | 0,961 | 8 | 1 | 0,354 | 1,000 | 0,250 | 0,000 | 0,357 |
| 7 | 45,796 | 38,894 | 0,963 | 8 | 1 | 0,363 | 1,000 | 0,250 | 0,000 | 0,357 |

Media: **hopfield = 0,000**, **vecinos = 0,250**, **hebb = 0,357**, **field = 0,361**, **ceros = 1,000**. F cae ~45,80 → ~38,89. Handshake ~0,96. 8/12 aristas rígidas (las 4 ocultas no se reconsolidan). 1 cavidad entre las caras que siguen rígidas.

Ranking en K=1: Hopfield (patrón ±1) gana de calle; el promedio de vecinos visible es el segundo; field y Hebb de `L1` empatan cerca y ambos ganan a ceros. Eso es el resultado, no un recorte.

Dos patrones (hemisferio + meridiano) en el mismo `L1`: Hopfield recupera ambos (~0). El campo se queda en ~0,37 en el primero y **~1,17 en el segundo** (peor que ceros). El slot único de `z` no es una memoria de K.

## Tests

- los componentes del estado y las claves del checkpoint no son un vocabulario (no hay `token`/`vocab`; la dimensión es el número de aristas)
- el patrón hemisferio parte 6+6, no 4 radios vs 8 resto
- atención geométrica = 0 entre aristas que no comparten vértice
- handshake pone `m=1` solo si las fases traban
- la región oculta no queda en `z_past`
- relax no sube `F` (temperatura 0) sobre una perturbación del atractor
- T crece al subdividir caras rígidas y poda aristas libres no semilla
- field, vecinos, Hopfield y Hebb ganan al baseline de ceros
- dos patrones se puntúan sin filtrar la región oculta
- la tabla es determinista
- un dataset reanuda desde `latest.json` sin mezclar otro `dataset_id`

## Lectura honesta

El campo reconstruye mejor que dejar la región en cero. En el mismo cue, Hopfield clásico (proyección compleja, K=1) reconstruye perfecto y el promedio de vecinos gana al relajador. Eso discrimina: con un patrón ±1 escrito a mano, el campo no aporta ventaja frente a brazos triviales. T crece y se poda en el dataset; las cavidades son geometría discreta, no vacío de Casimir. No es un LLM sin tokens. El siguiente corte que aún puede fallar es K>1 con patrones que no sean ±1 alineados al eje real, o una máscara donde el promedio de vecinos no baste.
