# Prerregistro corto: la pila final en otras familias (re-observación k = 1, 2)

Rama `exp/orbita-yo` (tras b8cd9b0). Publicado antes de evaluar. Pila: texto → Gemma (capa 13) → D2 (3000 pasos, pool2) + numnorm →
ÓRBITA-v1 congelada; re-observación fija k ∈ {1, 2} con re-identificación (g ∈ 8 familias v4, p ∈ {p̂} ∪ malla 9×9) por ÓRBITA como verificador
y rollout desde y_k (como `yo-act`). El yo no decide.

- **Conjunto** (`orbita fam-set`, RNG 0xFA4E, independiente de la seed y de v4; **no lee `ds.test`**): 16 ejemplos por familia, formas
  `make_object` (blob/círculo/triángulo), p ~ U[−1,1]². Familias: v4 iterables conocidas **1, 2** (referencia); v4 **no iterables en v4 (0, 3–7)**
  (ÓRBITA las vio a 1 paso en entrenamiento, nunca iteradas; 7 = deformación radial, no afín); **nuevas no afines 8** (giro ondulado = rotación +
  no linealidad suave) y **9** (espiral), idénticas a las del experimento mem.
- **Textos**: por ejemplo 3 textos (es/en/fr) con las plantillas de evaluación 6–7 de e2e (plantilla = 6 + idx mod 2; formatos numéricos de
  `e2e.fmts`). Nombres nuevos: es «giro ondulado»/«espiral», en «wavy turn»/«spiral», fr «virage ondulé»/«spirale». 480 textos, embebidos una vez.
- **Condiciones**: BASE (ĉ, sin re-observar), OBS1, OBS2, ORACLE_LS1/LS2 (afín por mínimos cuadrados con 1 o 2 pares observados, rollout desde
  y_k), TRUE (contexto verdadero; sólo familias 0–7). Métricas ok8, **ok16** (rel_err < 0,05).
- **Hipótesis** (δ = 0,05; IC95 bootstrap pareado por seed, 10 000, semilla 12345; 16 seeds DEV y luego CONFIRM; aceptar si IC95 > 0 en ambas):
  **HF1a** acc16 OBS1 − BASE > 0 en las nuevas (8, 9); **HF1b** igual en las v4 no iterables (0, 3–7); **HF2a/b** lo mismo con OBS2.
  Reportado sin criterio: brecha acc16 conocidas (1, 2) − nuevas para BASE/OBS1/OBS2; OBSk − ORACLE_LSk; TRUE.
- Nota: el conjunto de evaluación es el mismo en todas las seeds (cambian D2 y ÓRBITA), así que el bootstrap captura variación de modelos, no de datos.
- Ejecución: `orbita/llm/fam_eval.py`, `fam_chain.sh`; `setsid nohup nice -n 0`, log/.done en `/workspace/orbita-data/demo_fam/`;
  resultados `docs/resultados_otras_familias.md`.
