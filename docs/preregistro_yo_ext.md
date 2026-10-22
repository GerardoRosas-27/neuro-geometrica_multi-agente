# Prerregistro corto: entrenamiento extenso del sistema (D2_EXT + yo recalibrado), ÓRBITA-v1 congelada

Se lanza automáticamente **después** de `/workspace/orbita-data/yo/confirm.done` (evaluación del yo), independientemente de sus resultados.
Publicado antes de lanzar. (El archivo de textos de entrenamiento se generó antes del push: es determinista y no contiene resultados.)

- **Qué se entrena**: el encoder D2 (misma arquitectura, mismas pérdidas que `d2.py`) con **STEPS = 9000** (antes 3000) y un pool **ampliado**:
  pool2 original (128 contextos × es/en/fr × plantillas 0,1,4,5) + **256 contextos nuevos** (RNG 0xC0DE3, misma distribución TRAIN sin el hueco)
  × es/en/fr × plantillas 0,1,4,5 y **plantillas nuevas 10–11** (p en palabras/fracciones/grados/vueltas/radianes) + **idiomas nuevos pt/it**
  (plantillas 20–21, dígitos) = 7168 textos. Estadísticas de normalización: las de pool2 (sin cambio). Una D2_EXT por seed, ÓRBITA-v1 congelada
  (no hay justificación para reentrenarla). El yo/calibrador se recalibra sobre los rasgos de D2_EXT con los **mismos** datos de calibración
  (partición DEV, plantillas 8–9); Gemma y numnorm sin cambio. No se regenera con Gemma (H3/H4 no se repiten).
- **Split por plantilla**: entrenamiento 0,1,4,5,10,11,20,21; calibración 8–9; evaluación 6–7 (filas E2E_R de e2e). 0 queries TEST (contador v4).
- **Hipótesis** (δ = 0,05; IC95 bootstrap pareado por seed, 10 000, semilla 12345; DEV y luego CONFIRM; aceptar si IC95 > 0 en ambas):
  HX1 acc16 E2E_R EXT − BASE; HX2 familia correcta EXT − BASE; HX3 Brier del yo BASE − EXT (secundaria AURC). BASE = resultados de `docs/preregistro_yo.md`.
- Ejecución: `orbita/llm/yo_ext.py`, `yo_ext_chain.sh`; `setsid nohup nice -n 0`, log/.done en `/workspace/orbita-data/yo_ext/`
  (`pool2.tok.done`, `{ph}/<seed>.train.done`, `dev.done`, `confirm.done`), reanudable. Resultados en `docs/resultados_yo_ext.md`.
