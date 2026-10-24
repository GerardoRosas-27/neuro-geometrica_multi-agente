# ÓRBITA-Nova — Cómo lanzar el entrenamiento completo

**Preregistro**: [`PREREG.md`](PREREG.md) (commit `f64b289`).  
**Harness**: `orbita/src/nova_train.rs` + `orbita/scripts/nova_full.sh`.  
**Artefactos**: `artifacts/orbita/nova_full/`.

## Un comando (DEV → CONFIRM, sin reajuste)

```bash
# Desde la raíz del repo, rama exp/orbita-nova, working tree limpio recomendado.
# NO lanzar mientras Investigador IA / bytes / v6 usen el box.
bash orbita/scripts/nova_full.sh
```

Equivalente en background (recomendado en box compartido, tras OK de CPU):

```bash
setsid nohup bash orbita/scripts/nova_full.sh \
  >> artifacts/orbita/nova_full/nohup.out 2>&1 &
echo "pid $!"
```

El script:

1. Aborta si detecta `bytes_enc.py` / `bytes_exp.py` / `run_v6` / `run_predictive_field`  
   (override: `NOVA_IGNORE_BUSY=1`, estilo `V6_IGNORE_V5`).
2. Calibra **τ una sola vez** en H DEV familias **1–2** (16 seeds pooled, P95 dist. firma C4) → `tau.json`.  
   Si `tau.json` ya existe, **no** recalibra (PREREG §5).
3. Corre **DEV** `0xA400–0xA40F` (flujo 2400, C6 cada 50, checkpoints `{0,150,300,600,1200,2400}`).
4. Corre **CONFIRM** `0xB400–0xB40F` con el **mismo τ** y el mismo binario (sin reajuste).
5. Reanuda saltando seeds con `*.done`. Log: `artifacts/orbita/nova_full/nova_full.log`.

## Smoke corto (no satura CPU)

```bash
SMOKE=1 bash orbita/scripts/nova_full.sh
# o: SMOKE=1 N_STREAM=24 PHASE=dev bash orbita/scripts/nova_full.sh
```

## Qué implementa el harness (frente a stubs fase 0/1)

| pieza | estado |
|---|---|
| C6 biblioteca + consolidación N=50 | real (`OperatorLib::admit`, clustering greedy, no-daño) |
| Flujo 2400, mezcla 80/20, k∈{2,4,8} | `gen_stream` (XOR `0x4E0A_F100`) |
| Held-out H (176 ep.) | `gen_heldout` (XOR `0x4E0A_501D`) |
| τ calibración H DEV fam 1–2 | `--calibrate-tau` |
| Brazos | ÓRBITA, ÓRBITA+C3, LAT_EP, LAT, NOVA, NOVA−C3/C5/C6, NOVA_SHUF, MLP, ORACLE_EP |
| LAT hyps | λ₀=1e-4, λ_m=1e-3 (selected.json latente) |
| Núcleo ÓRBITA / Gemma / `run_v6` | no se tocan |

## Duración estimada

| tramo | estimación (1 hilo, nice 10) |
|---|---|
| Calibración τ | ~15–40 min |
| DEV 16×2400×brazos@ckpts | ~6–10 h |
| CONFIRM igual | ~6–10 h |
| **Total** | **~14–20 h CPU** (plan: ~1 día) |

Smoke (`N_STREAM=24`, 1 seed): minutos.

## Salidas por seed

`artifacts/orbita/nova_full/{dev,confirm}/metrics_0x….json` — métricas por checkpoint y brazo (`new_med_log`, `new_acc16`, `known_*`).  
Primaria PREREG: mediana `log(rel_err h16)` en nuevas {8,10,11}, k=4.

## Pendiente de lanzamiento

El job pesado **no** se arranca automáticamente. Esperar OK de CPU del Investigador IA (box libre) y entonces el comando de arriba.
