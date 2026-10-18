# orbita/archive — retirado de la ruta activa (rama exp/orbita-yo)

Se conserva verbatim para reproducibilidad; **no se compila ni se ejecuta** desde la ruta activa. La historia completa está en
`exp/orbita-llm` (último commit antes del retiro: e2f6bb0).

| archivo | origen | por qué se retira |
|---|---|---|
| `src/cdt_subcommands.rs` | `cdt_feat`, `build_signal`, `cdt_causal` de `orbita/src/main.rs` | CDT no aportó (docs/resultados_cdt_sueno.md, docs/resultados_cdt_causal.md) |
| `src/mem.rs` | subcomando `mem` | memoria por capas E0–E5: H-E2 rechazada (docs/resultados_memoria_capas.md) |
| `llm/cdt.py`, `cdt_chain.sh`, `causal.py`, `causal_chain.sh`, `mem.py`, `mem_chain.sh`, `mem_chain2.sh` | `orbita/llm/` | idem |

Para re-ejecutar: `git checkout e2f6bb0` (rama exp/orbita-llm).

Dependencias de v4/CDT que **siguen** en la ruta activa (no son memoria CDT en inferencia):
- `orbita export`: los datos de entrenamiento de ÓRBITA-v1 incluyen replay del teacher consolidado (procedencia de los pesos ya entrenados).
- `orbita eval`/`fresh`: auditor v4 de fugas.
- `orbita llm-eval`: además de ÓRBITA calcula columnas diagnósticas `fam_hat_teacher`/`teacher_div16` (H-C2 de e2e). Se dejan para no
  romper `e2e.py`/`test_e2e.sh`; retirarlas es un cambio pendiente de aprobación.
