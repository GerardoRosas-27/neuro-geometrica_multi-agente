# PRERREGISTRO — exp/orbita-vs-timesfm (ÓRBITA vs TimesFM 2.5 sobre latentes dinlat)

Escrito y empujado ANTES de ejecutar DEV. Rama base: `exp/orbita-dinamica-latente` @ 18ba4a4.

## Pregunta
¿ÓRBITA predice mejor la dinámica de los estados latentes (Gemma capa 13 → PCA 16 → rotación fija) que el
modelo fundacional de series de tiempo TimesFM 2.5 usado zero-shot?

## Datos (sin cambios, sin re-embeber)
- Mismos episodios, embeddings y latentes de dinlat (`/workspace/orbita-data/dinlat/{dev,confirm}`): la PCA, Q_rand y
  los controles C1/C2 se reconstruyen con el mismo código/rng; el script ABORTA si PERS o AR2 recalculados no coinciden
  con `res_<seed>.json` de dinlat.
- Brazos ÓRBITA/AR2/PERS: se reutilizan sus errores por episodio de dinlat (mismos modelos entrenados, mismos seeds).
- Seeds: 16 DEV (0xA400–0xA40F) + 16 CONFIRM (0xB400–0xB40F). 0 consultas a TEST.

## Brazos nuevos
- **TIMESFM**: `google/timesfm-2.5-200m-pytorch` (Apache-2.0), `timesfm[torch]`, CPU; `normalize_inputs=True`,
  `infer_is_positive=False`, `force_flip_invariance=True`, `fix_quantile_crossing=True`, max_context=64; pronóstico puntual
  (media). Cada una de las 16 dimensiones latentes de cada episodio es una serie univariada independiente.
  Mismo contexto que ÓRBITA: un paso desde s_0..s_t (t=2..7) → s_{t+1}; h3: contexto s_0..s_2 → s_3..s_5.
- **VAR2_OLS**: VAR(2) multivariado sin regularización (AR2 de dinlat ya es VAR(2) con ridge α=1).

## Métrica
G(A vs B) = log(MSE_B / MSE_A) por seed (MSE medio sobre episodios del split; >0 ⇒ A mejor).
IC95 bootstrap pareado por seed, 10000 réplicas, semilla 12345 (idéntico a dinlat/analyze.py). δ = 0.05.

## Hipótesis (split ho = held-out de familias de entrenamiento; tf = familias de transferencia)
- **H1 (primaria)**: ORB_best = ORB_SESGO (fijado de antemano: mejor brazo ÓRBITA en DEV de dinlat; NOTA: usa etiquetas
  verdaderas de familia+p, ventaja que TimesFM no tiene). Aceptada si IC95_lo de G(ORB_SESGO vs TIMESFM, ho) > δ.
- **H1b**: G(ORB_LIBRE vs TIMESFM, ho) IC95_lo > δ (comparación justa: ninguno ve etiquetas).
- **H2 (cordura)**: G(TIMESFM vs PERS, ho) IC95_lo > δ.
- **H3**: G(ORB_LIBRE vs TIMESFM, tf) IC95_lo > δ.
- Conclusión inversa ("TimesFM mejor") solo si IC95_hi < −δ.
- Secundarios (sin corrección, descriptivos): TIMESFM vs AR2 / VAR2_OLS (ho, tf); por grupo de familias
  (geométricas = rotar, trasladar, escalar, cizalla; no geométricas); horizonte h3 (rollout 3 pasos); controles C1
  permutado / C2 barajado (TIMESFM vs PERS, ORB_LIBRE vs TIMESFM).

## Gate DEV → CONFIRM
media G(ORB_SESGO vs TIMESFM, ho) > 0 **y** IC95_lo de H2 > 0. Si no pasa: parada, CONFIRM no se ejecuta.
CONFIRM requiere generar los embeddings Gemma y los brazos ÓRBITA de los seeds 0xB4.. (no existen en caché;
dinlat paró en DEV) — se ejecuta con `orbita/dinlat/chain.sh confirm` sin cambios de código y luego TimesFM.
En CONFIRM los criterios se evalúan con los mismos umbrales; veredicto final = CONFIRM.

## Reglas
Sin tocar main, sin force push, sin números escritos a mano (RESULTADOS.md se genera desde analysis.json).
