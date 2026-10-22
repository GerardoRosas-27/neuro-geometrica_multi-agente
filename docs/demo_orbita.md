# Demo de la pila final ÓRBITA

Uso: `orbita/demo/README.md`. Código: `orbita/llm/demo.py`, subcomandos Rust `orbita demo-step` (rollout, sorpresa, σ, re-identificación) y
`orbita demo-rule` (sólo para simular observaciones), `orbita/demo/{smoke.sh,examples.sh}`. Salidas reales en `artifacts/orbita/demo/*.json`
(generadas 2026-10-18 ≈22:45 CST, seed 0xA400).

## Qué devuelve
familia y p leídos (D2 argmax + numnorm, o fallback D2), probabilidad del encoder, trayectoria h1..h16 de ÓRBITA desde x0; con `--obs k`:
familia/p re-identificados y trayectoria desde el último estado observado; el **yo** (confianza calibrada de acertar a h16 —el calibrador de
`docs/preregistro_yo.md`, ajustado en la partición DEV de esa seed; sin observación se ajusta sin el rasgo sorpresa—, sorpresa, σ, ratio de
norma); y una explicación de Gemma 2 2B Q4_K_M (greedy, prompt fijo en `demo.py`) a partir del yo. **Gemma sólo verbaliza al yo: no ve el mundo**
(resultado H4 de `docs/resultados_yo.md`).

## Ejemplos reales (`artifacts/orbita/demo/`)
| ejemplo | leído | re-identificado | confianza del yo |
|---|---|---|---|
| 01 es rotación (0,5; −0,2), k=1 | rotación (0,5; −0,2), numnorm | rotación (0,5; −0,2) | 0,1761 |
| 02 en reflexión (−0,3; 0,4), sin obs | reflexión (−0,3; 0,4) | — | 0,4805 |
| 03 fr espiral (nueva, fam 9), k=2 | **cizalla** (0,6; −0,5) | rotación (0,25; 0,5) | 0,0 |
| 04 pt rotación (0,3; 0,7), k=1 | rotación (0,3; 0,7) | rotación (0,25; 1,0) | 0,1456 |
| 05 it escalado (−0,4; 0,2), k=2 | **cizalla** (−0,4; 0,2) | escalado (−0,4; 0,2) | 0,0003 |

## Límites observados
- pt/it no están en el entrenamiento de D2: «scalatura» se leyó como cizalla (la re-observación lo corrigió); numnorm lee los dígitos.
- Una familia nueva (espiral) no tiene nombre en D2; la re-identificación sólo puede elegir entre las 8 familias v4.
- El yo es poco informativo (confianzas bajas incluso en aciertos) y Gemma a veces califica mal su propia cifra (ej. 02: «alta confianza del 48 %»).
