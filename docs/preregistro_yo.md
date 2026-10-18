# Prerregistro H1–H4: el «yo» de ÓRBITA y Gemma como observador de 2º orden

Rama `exp/orbita-yo` (desde 05e14c6). Plan aprobado: `docs/plan_yo_primer_segundo_orden.md`. Se publica (push sin force) **antes** de
ejecutar nada (incluida la extracción Gemma y la medición de coste de generación). Humo sólo con la seed 0x5EED (descartada).

## Datos (split por plantilla; 0 queries a TEST)
- Seeds DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F; ÓRBITA-v1 y D2 congelados (`w_<seed>.json`, `enc_D2_<seed>.pt`).
- **Evaluación** = mismas filas que e2e (925015f): por seed 16 ejemplos iterables (8 rotación + 8 reflexión, `llm_subset`) × 6 textos de
  robustez (es/en/fr × plantillas 6–7, p en palabras/fracciones/grados/vueltas/radianes), condición E2E_R. El contador de queries de v4
  debe dar 0 como en e2e.
- **Calibración (sólo partición DEV de v4)**: por seed los 8 primeros ejemplos de rotación y los 8 primeros de reflexión de `ds.dev`, × 6 textos
  con **plantillas nuevas 8–9** (es/en/fr; mismo esquema de formatos numéricos que 6–7). Nunca TRAIN/DEV de texto de evaluación.
- Objetivo de acierto: `ok16` (rel_err < 0,05 a h=16 frente a la trayectoria real), como en e2e.

## El yo (1er orden → auto-modelo)
Rasgos por fila, sólo con lo que ÓRBITA/encoder producen y una observación real y1 (paso 1):
1. `p_d2` = softmax máx. de los logits de familia de D2; 2. `margin_d2` = top1 − top2; 3. `parse_ok` (numnorm vs fallback D2);
4. `surprise` = log rel_err(ÓRBITA(x0, ĉ), y1); 5. `sigma` = log σ_máx del jacobiano de ÓRBITA en x0; 6. `normr` = log(|z16|/|x0|).
Calibrador: regresión logística L2 (λ = 1e-2, rasgos estandarizados en calibración) ajustada **por seed** en sus 96 filas de calibración →
p̂ = P(ok16). Si la calibración tiene una sola clase: p̂ = (n_ok+1)/(n+2) constante.
Baselines calibrados igual con un único rasgo: MARGIN (`margin_d2`) y SURPRISE (`surprise`); RANDOM = orden aleatorio.

## Observador de 2º orden (Gemma 2 2B it, GGUF Q4_K_M, llama.cpp b11259, greedy: temp 0, top-k 1, n_predict 40, seed 0)
Prompt fijo (plantilla de chat Gemma):
```
<start_of_turn>user
Eres el monitor de un modelo de predicción llamado ÓRBITA. {BLOQUE}
Responde exactamente con dos líneas:
Familia: <nombre de la transformación que ÓRBITA cree que es>
Confianza: <número entero de 0 a 100 de que ÓRBITA acierte>
<end_of_turn>
<start_of_turn>model
```
BLOQUE (YO) = «Este es su estado interno:» + líneas: familia elegida por el encoder: {nombre es}; probabilidad del encoder para esa familia:
{p_d2:.2f}; números leídos del texto: {sí | no, se usó la estimación del encoder}; sorpresa frente a la primera observación: {rel_err:.3f};
estabilidad del rollout (sigma máx. del jacobiano): {σ:.2f}; confianza estimada por el yo: {round(100·p̂)} %.
Controles: **SHUF** = BLOQUE de otra fila de la misma seed (desarreglo fijo, semilla 12345); **EMPTY** = mismas líneas con «desconocido»;
**NOYO** = BLOQUE = «No tienes acceso a su estado interno.»
Lector objetivo: familia = nombre (es) más largo contenido tras «Familia:»; confianza = primer entero tras «Confianza:» (recortado a 0–100).
**Coste**: se mide antes de escalar. Total completo = 32 seeds × 96 filas × 4 condiciones = 12 288 generaciones. Si la estimación supera 4 h,
se usa el **subconjunto prerregistrado**: por seed 1 texto por ejemplo (idioma = [es, en, fr][idx mod 3], plantilla 6) → 16 filas × 4 × 32 = 2048.
H1/H2 siempre con las 96 filas.

## Hipótesis (δ = 0,05; IC95 bootstrap pareado por seed, 10 000, semilla 12345; DEV y luego CONFIRM; se acepta si IC95 > 0 en ambas)
- **H1 calibración**: Brier: min(Brier MARGIN, Brier SURPRISE) − Brier YO > 0 (primaria); ECE (10 bins iguales) igual forma (secundaria).
- **H2 riesgo-cobertura**: AURC (área bajo riesgo vs cobertura al ordenar por p̂ descendente): min(AURC MARGIN, AURC SURPRISE, AURC RANDOM) − AURC YO > 0.
  AURC RANDOM = tasa de error global (esperanza).
- **H3 fidelidad del 2º orden**: (a) familia recuperada de la frase = familia del yo de esa fila: YO − max(SHUF, EMPTY, NOYO) > 0;
  (b) error de confianza |conf/100 − p̂|: min(SHUF, EMPTY, NOYO) − YO > 0. Frases sin campo legible cuentan como fallo / error 1.
- **H4 el 2º orden añade predicción**: AUROC(−conf Gemma → fallo ok16) − AUROC(−p̂ → fallo) > 0 (sólo filas generadas, condición YO).
  Si H4 falla, se declara que el 2º orden sólo verbaliza al yo. Seeds sin ambas clases se excluyen (se reporta n).
Equivalencia: IC95 ⊂ (−δ, δ).

## Ejecución
`setsid nohup nice -n 0`, log/`.done` por paso y seed en `/workspace/orbita-data/yo/`, reanudable; DEV antes que CONFIRM.
Resultados en `docs/resultados_yo.md` y `artifacts/orbita/yo/`. CDT diagnóstico de `llm-eval` no se usa.
