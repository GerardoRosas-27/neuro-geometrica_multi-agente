# Resultados — pruebas causales: ¿CDT-RQM-epr aporta REALMENTE a ÓRBITA?

Prerregistro `docs/preregistro_cdt_causal.md` (fb3a962, δ = 0,05), publicado antes de lanzar nada. Ejecución 2026-10-17
12:04 (CDMX), `setsid nohup nice -n 0`; log y `.done` en `/workspace/orbita-data/llm_cdt2/` (`dev.done`, `confirm.done`).
Artefactos: `artifacts/orbita/llm_cdt2/` (`summary_{dev,confirm}.md` con todas las tablas, `causal_<seed>.csv`,
`log_<seed>.txt`). Sin extracción nueva (reusa contextos E2E_R, plantillas 6–7). **0 queries** a TEST auditado en ambas
fases. Costo: ~5 s por fase en total (todo en `orbita cdt-causal`, sin entrenamiento).

## Veredicto: **CDT no aporta** (criterio prerregistrado no se cumple en ninguna variante)
| variante V | Δ acc16 = V − max(controles) DEV | CONFIRM | veredicto |
|---|---|---|---|
| OBS_DIR | −0,7148 [−0,7839, −0,6458] | −0,6882 [−0,7643, −0,6133] | Refutada |
| OBS_SLEEP | −0,8789 [−0,9271, −0,8255] | −0,8600 [−0,8978, −0,8210] | Refutada |
| CORR_DIR | −0,1816 [−0,2370, −0,1224] | −0,1634 [−0,2285, −0,0996] | Refutada |
| CORR_SLEEP | −0,0391 [−0,0697, −0,0104] | −0,0156 [−0,0430, +0,0111] | Inconclusa (no > 0) |
| OBSCORR_DIR | −0,9076 [−0,9427, −0,8711] | −0,8574 [−0,8984, −0,8158] | Refutada |
| OBSCORR_SLEEP | −0,9258 [−0,9570, −0,8906] | −0,8743 [−0,9154, −0,8346] | Refutada |

El criterio secundario (AUROC − max de controles) **no es evaluable**: varios controles (ORACLE_LS acierta el 100 %,
CORR_CORRUPT/SHUF el 0 %) no tienen las dos clases y su AUROC es indefinido, así que el máximo es NaN.

## CONFIRM, acc16 (IC95)
TRUE de la fase e2e 0,3164. BASE 0,2975 [0,2220, 0,3711]; OBS_DIR 0,3118 [0,2357, 0,3867]; **ORACLE_ORB 0,3118** (idéntica);
**ORACLE_LS 1,0000**; OBS_SLEEP 0,1400; OBS_SHUF 0,1628; OBS_CORRUPT 0,0065; CORR_DIR 0,1341; CORR_SLEEP 0,2819;
CORR_CORRUPT 0; CORR_SHUF 0; OBSCORR_DIR 0,1426; OBSCORR_SLEEP 0,1257.
Contrastes de lectura (CONFIRM): OBS_DIR − BASE +0,0143 [+0,0072, +0,0241]; OBS_DIR − ORACLE_ORB 0 [0, 0];
OBS_DIR − OBS_SHUF +0,1491; OBS_DIR − OBS_CORRUPT +0,3053. H-SD: OBS SLEEP − DIR −0,1719 [−0,2305, −0,1172] (sueño peor);
CORR SLEEP − DIR +0,1478 [+0,0898, +0,2077] (sueño mejor que directo, pero sin superar BASE); OBSCORR −0,0169
[−0,0365, +0,0059] (equivalentes dentro de δ). Poda en sueño: 384 → 240–288 números según seed. DEV da el mismo patrón.

## Lectura
- **Observar ayuda, pero no gracias al CDT.** Re-identificar la familia con 2 observaciones sube acc16 +0,014 sobre BASE
  y los controles SHUF/CORRUPT confirman que la señal consolidada sí contiene información de familia; pero ÓRBITA
  usada como su propio verificador con las mismas observaciones (ORACLE_ORB) logra exactamente lo mismo, y un ajuste afín
  por mínimos cuadrados sobre esas 6×2 parejas (ORACLE_LS) predice h16 perfectamente: las familias iterables de TEST
  (1, 2) son afines, así que 2 observaciones bastan sin ningún modelo. El aporte es «ver datos», no lo consolidado.
- **Corregir con el teacher perjudica.** Mezclar α = 0,5 con el teacher por paso baja acc16 (CORR_DIR 0,134 vs 0,298);
  el teacher tras sueño+poda daña menos (0,282) porque pierde reglas (familias podadas → sin corrección), no porque corrija mejor.
- **Sueño vs directo:** no hay una dirección consistente: el sueño empeora en OBS, mejora sobre directo en CORR y empata en OBSCORR.
- Limitaciones: K_OBS = 2 y α = 0,5 fijos (sin barrido); sólo las 16 entradas TEST iterables por seed (familias 1–2, afines).
