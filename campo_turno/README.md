# campo_turno — campo de ESTADO por turno

El campo no aprende lenguaje: predice el estado `z` del siguiente turno assistant a partir de los `z` de turnos previos
(incluido el user actual). El LLM congelado es la periferia: codifica turnos (`turno_extract.py` + `turno_encoder.py`)
y decodifica vía un adaptador `z → P` soft-prompts (`turno_adapter.py`). El campo es el crate Rust `../gemma_campo` sin cambios.
Baselines: `turno_baselines.py`. Evaluación T1–T4: `turno_eval.py`. Preregistro y runbook: `docs/campo_estado_turno/`.
Smoke: `PY=python bash campo_turno/smoke.sh /tmp/ct_smoke conv.jsonl cpu`.
