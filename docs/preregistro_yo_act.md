# Prerregistro «el yo actúa»: re-observación adaptativa con ÓRBITA como verificador

Rama `exp/orbita-yo` (tras 7da2a42). Pila: Gemma → D2 (3000 pasos, pool2) + numnorm → ÓRBITA-v1 congelada + yo (calibrador de
`docs/preregistro_yo.md`). Publicado antes de lanzar. Única ejecución previa: una medición de tiempo de `orbita yo-act` sobre la seed 0xA400
(salida descartada a /dev/null, 1,3 s; sólo se leyó el contador de queries = 0).

## Datos (reutilizados, sin regenerar Gemma)
Rasgos y contextos de `/workspace/orbita-data/yo/{dev,confirm}/` (`ctx_*`, `yo_*`): evaluación = 96 filas E2E_R por seed (plantillas 6–7,
16 ejemplos TEST iterables como e2e; contador v4 = 0); calibración = 96 filas por seed de la partición DEV (plantillas 8–9).

## Acción
`orbita yo-act`: para cada fila, **BASE** = rollout de ÓRBITA desde x0 con ĉ. **OBSk** (k = 1, 2): se observan los k primeros estados
reales y1..yk, se re-identifica (familia g, p) por mínimo de Σ_{t≤k} rel_err(ÓRBITA^t(x0, c(g,p)), y_t) con g ∈ 8 familias y
p ∈ {p̂} ∪ malla 9×9 en [−1,1]² (ÓRBITA como verificador, como ORACLE_ORB), y el rollout sigue desde y_k con el contexto elegido.
Métricas: ok8, ok16 frente a la trayectoria real.

## Condiciones (por k; primaria k = 1, secundaria k = 2)
BASE; ALWAYS_OBS (todas las filas); YO_OBS (re-observa si p̂ < τ_YO); SURPRISE_OBS (si sorpresa > τ_S); RANDOM_OBS (misma tasa que YO_OBS
en esa seed; valor esperado sobre 200 sorteos, semilla 12345); ORACLE (sólo filas con BASE ok16 = 0).
Umbrales **por seed, sólo con las filas de calibración** (p̂ en calibración = calibrador ajustado en esas mismas filas): τ entre todos los
valores observados y los extremos, maximizando acc16_cal(política) − 0,05·tasa_cal (empates → menor tasa).
**Coste**: tasa de re-observación y nº medio de pasos reales observados para re-identificar (k × tasa). Nota declarada: YO/SURPRISE ya usan y1
para la sorpresa (coste de vigilancia 1 paso por fila, se reporta aparte). Curva acc16 vs tasa (descriptiva) ordenando por p̂, por sorpresa y al azar.

## Hipótesis (δ = 0,05; IC95 bootstrap pareado por seed, 10 000, semilla 12345; DEV y luego CONFIRM; aceptar si se cumple en ambas)
- **HA1**: acc16 YO_OBS − RANDOM_OBS > 0 (k = 1; secundaria k = 2; acc8 secundaria).
- **HA2**: acc16 YO_OBS − ALWAYS_OBS: IC95 inferior > −δ **y** tasa YO_OBS < 1 (IC95 superior < 1).
- **HA3** (reportada, sin criterio): acc16 YO_OBS − SURPRISE_OBS.
Ejecución: `orbita/llm/yo_act.py`, `yo_act_chain.sh`; `setsid nohup nice -n 0`, log/.done en `/workspace/orbita-data/yo_act/`; resultados en
`docs/resultados_yo_act.md`.
