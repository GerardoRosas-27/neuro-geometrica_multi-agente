# PRERREGISTRO ÔÇö exp/orbita-vs-timesfm (├ôRBITA vs TimesFM 2.5 sobre latentes dinlat)

Escrito y empujado ANTES de ejecutar DEV. Rama base: `exp/orbita-dinamica-latente` @ 18ba4a4.

## Pregunta
┬┐├ôRBITA predice mejor la din├ímica de los estados latentes (Gemma capa 13 ÔåÆ PCA 16 ÔåÆ rotaci├│n fija) que el
modelo fundacional de series de tiempo TimesFM 2.5 usado zero-shot?

## Datos (sin cambios, sin re-embeber)
- Mismos episodios, embeddings y latentes de dinlat (`/workspace/orbita-data/dinlat/{dev,confirm}`): la PCA, Q_rand y
  los controles C1/C2 se reconstruyen con el mismo c├│digo/rng; el script ABORTA si PERS o AR2 recalculados no coinciden
  con `res_<seed>.json` de dinlat.
- Brazos ├ôRBITA/AR2/PERS: se reutilizan sus errores por episodio de dinlat (mismos modelos entrenados, mismos seeds).
- Seeds: 16 DEV (0xA400ÔÇô0xA40F) + 16 CONFIRM (0xB400ÔÇô0xB40F). 0 consultas a TEST.

## Brazos nuevos
- **TIMESFM**: `google/timesfm-2.5-200m-pytorch` (Apache-2.0), `timesfm[torch]`, CPU; `normalize_inputs=True`,
  `infer_is_positive=False`, `force_flip_invariance=True`, `fix_quantile_crossing=True`, max_context=64; pron├│stico puntual
  (media). Cada una de las 16 dimensiones latentes de cada episodio es una serie univariada independiente.
  Mismo contexto que ├ôRBITA: un paso desde s_0..s_t (t=2..7) ÔåÆ s_{t+1}; h3: contexto s_0..s_2 ÔåÆ s_3..s_5.
- **VAR2_OLS**: VAR(2) multivariado sin regularizaci├│n (AR2 de dinlat ya es VAR(2) con ridge ╬▒=1).

## M├®trica
G(A vs B) = log(MSE_B / MSE_A) por seed (MSE medio sobre episodios del split; >0 ÔçÆ A mejor).
IC95 bootstrap pareado por seed, 10000 r├®plicas, semilla 12345 (id├®ntico a dinlat/analyze.py). ╬┤ = 0.05.

## Hip├│tesis (split ho = held-out de familias de entrenamiento; tf = familias de transferencia)
- **H1 (primaria)**: ORB_best = ORB_SESGO (fijado de antemano: mejor brazo ├ôRBITA en DEV de dinlat; NOTA: usa etiquetas
  verdaderas de familia+p, ventaja que TimesFM no tiene). Aceptada si IC95_lo de G(ORB_SESGO vs TIMESFM, ho) > ╬┤.
- **H1b**: G(ORB_LIBRE vs TIMESFM, ho) IC95_lo > ╬┤ (comparaci├│n justa: ninguno ve etiquetas).
- **H2 (cordura)**: G(TIMESFM vs PERS, ho) IC95_lo > ╬┤.
- **H3**: G(ORB_LIBRE vs TIMESFM, tf) IC95_lo > ╬┤.
- Conclusi├│n inversa ("TimesFM mejor") solo si IC95_hi < ÔêÆ╬┤.
- Secundarios (sin correcci├│n, descriptivos): TIMESFM vs AR2 / VAR2_OLS (ho, tf); por grupo de familias
  (geom├®tricas = rotar, trasladar, escalar, cizalla; no geom├®tricas); horizonte h3 (rollout 3 pasos); controles C1
  permutado / C2 barajado (TIMESFM vs PERS, ORB_LIBRE vs TIMESFM).

## Gate DEV ÔåÆ CONFIRM
media G(ORB_SESGO vs TIMESFM, ho) > 0 **y** IC95_lo de H2 > 0. Si no pasa: parada, CONFIRM no se ejecuta.
CONFIRM requiere generar los embeddings Gemma y los brazos ├ôRBITA de los seeds 0xB4.. (no existen en cach├®;
dinlat par├│ en DEV) ÔÇö se ejecuta con `orbita/dinlat/chain.sh confirm` sin cambios de c├│digo y luego TimesFM.
En CONFIRM los criterios se eval├║an con los mismos umbrales; veredicto final = CONFIRM.

## Reglas
Sin tocar main, sin force push, sin n├║meros escritos a mano (RESULTADOS.md se genera desde analysis.json).
