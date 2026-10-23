# Hooks de train (Nova)

Estado fase 0: **stubs / no-op**.

| hook | símbolo | estado |
|---|---|---|
| CLI `--prepare-train` | escribe configs | listo |
| CLI `--train` | `train_stub` | no-op si `enabled=false` |
| Consolidación C6 | `OperatorLib::admit_stub` | rechaza siempre |
| Compuerta C7 | `gate_nodamage` | lógica real, sin train |
| Ajuste C4 | `fit_gn` por episodio | **activo** (sin SGD) |

No reentrenar el núcleo ÓRBITA ni Gemma. No invocar `run_v6`.
