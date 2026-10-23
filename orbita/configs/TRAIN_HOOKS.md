# Hooks de train (Nova)

Estado post-PREREG (`f64b289`): **C6 + flujo cableados** en `nova_train.rs`.

| hook | símbolo | estado |
|---|---|---|
| CLI `--calibrate-tau` | `nova_train::calibrate_tau` | **activo** (H DEV fam 1–2) |
| CLI `--full` | `nova_train::run_seed_full` | **activo** |
| Script | `orbita/scripts/nova_full.sh` | DEV→CONFIRM, resume, busy-guard |
| Consolidación C6 | `OperatorLib::admit` + `consolidate` N=50 | **activo** |
| Compuerta C7 | `gate_nodamage` | activo |
| Ajuste C4 | `fit_gn` por episodio | activo (sin SGD) |

No reentrenar el núcleo ÓRBITA ni Gemma. No invocar `run_v6`. Ver `docs/orbita_nova/ENTRENAMIENTO.md`.
