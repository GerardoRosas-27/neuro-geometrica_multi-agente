# Prerregistro — ÓRBITA-Bytes vs ÓRBITA (encoder Gemma + D2/D3-num)

Fecha: 2026-10-22 (America/Mexico_City). Rama `exp/orbita-bytes` (desde `main` `590dd93`). Diseño: Page «ÓRBITA-Bytes: red neuronal de bytes inspirada en Large Concept Models» (LCM 2412.08821, BLT 2412.09871, CALM 2510.27688).
Este documento y el código se commitean y empujan **antes de cualquier corrida** de entrenamiento/evaluación (sólo hubo humos de 60/300 pasos en `/tmp` para medir coste y depurar; ninguno de sus números se usa).

## Pregunta
¿Un encoder sin tokenizador (bytes UTF-8 → parches por entropía → VAE robusto por parche → concepto por frase → `c` de 10) es **no inferior** al encoder actual (Gemma 2 2B capa 13 → D2 + numnorm) para alimentar al **mismo** núcleo ÓRBITA-v1 congelado?

## Brazos (sólo cambia el encoder)
- **GEMMA** (base): `E2E_R` actual — tokens Gemma capa 13 cacheados → D2 de la seed (`llm_d2/{ph}/enc_D2_<seed>.pt`) → `e2e.build_c` (argmax familia + numnorm; fallback cabeza D2).
- **MAIN** (ÓRBITA-Bytes): `orbita/llm/bytes_enc.py` → mismo `e2e.build_c` (argmax familia + numnorm; fallback cabeza de parámetros de bytes).
- Núcleo: `/workspace/orbita-data/{dev,confirm}/w_<seed>.json`, idéntico en ambos brazos; binario `orbita` compilado desde esta rama (Rust sin cambios). No hay tokenizador en ninguna etapa del brazo MAIN; Gemma sólo es teacher fuera de línea (vector `u` de 128 de D2 sobre `pool2`), nunca en inferencia.
- Ablaciones (sólo DEV, exploratorias): NOVAE (sin VAE/KL/dropout latente), NOHASH (sin embeddings hash), NONCE (sin destilación), FIXPATCH (parches fijos de 5 bytes, sin LM de entropía).
- **H9 (energía en el corrector OT)**: no se ejecuta en esta cadena — exige re-entrenar el corrector Brenier y rompería la condición «mismo núcleo congelado en ambos brazos». Queda pendiente como experimento aparte.

## Arquitectura y entrenamiento (FIJOS)
- S0 LM de entropía: 2 capas Transformer causal, d=128, 4 cabezas; contexto = el texto (se reinicia por línea). θg por bisección para tamaño medio de parche 5 bytes en TRAIN; θr = 1,0 nat (inicio si H>θg o H−H₋₁>θr); tope 16 bytes por parche.
- S1 autoencoder: byte emb + hash n-gramas n=3..8 (RollPolyHash, 6 tablas × 8192 × 128); 1 capa con atención local causal (ventana 16); cross-attention parche←bytes (consulta = max-pool del parche); VAE por parche (MLP 128→256→2·32), β=0,001, KL clip por dimensión 0,25 nats, dropout latente 0,15, enmascarado 15 % de bytes de entrada; decoder local de 2 capas (cada byte ve sólo el latente de su parche y los bytes previos del parche). Congelado tras S1.
- S2 concepto: pooler 2 capas + atención con consulta aprendida → concepto 128; robust scaler (mediana/IQR de TRAIN) para el concepto exportado; cabezas 128→8 y 128→2 (tanh), straight-through one-hot como D2. Pérdida = CE familia + 10·MSE params + 0,5·InfoNCE(τ=0,1; proyección del concepto vs `u` de D2 Gemma del mismo texto) + 1,0·JEPA (predictor MLP → concepto EMA(0,99) de una paráfrasis del mismo contexto, stop-grad, 1−cos) + 1,0·trayectoria (rollout ÓRBITA congelada k∈{1,2,4,8}, idéntico a D2).
- AdamW β=(0,9; 0,95), ε=1e-8, wd 0,1, clip 1,0, lr 4e-4, warm-up 200, coseno; lote 32. CPU, 2 hilos por proceso, 4 seeds en paralelo.
- Datos de entrenamiento: `llm_d2/pool2.tsv` (los mismos 1536 textos es/en/fr, plantillas 0,1,4,5 con que se entrenó D2). Validación: contextos `pool{f}_14`, `pool{f}_15` (16 contextos, 192 textos), nunca entrenados (D2 sí vio esos 16 contextos: desventaja conservadora para bytes).

## Convergencia y checkpointing (prerregistrado)
- **Mejor validación desde el inicio** en cada etapa: evaluación cada 250 pasos; se guarda el estado del mejor valor y se restaura al final. Métrica: S0 CE de bytes val; S1 CE de reconstrucción val (μ, sin ruido); S2 CE familia + 10·MSE params val.
- Parada: sin mejora relativa > 1e-3 en 4 evaluaciones consecutivas, o tope S0 3000 / S1 4000 / S2 3000 pasos.
- Se reporta por seed paso del mejor checkpoint y si paró por paciencia (convergió) o por tope. Sin cambios de hiperparámetros entre DEV y CONFIRM.

## Conjuntos de evaluación (0 consultas a TEST de v4; manifiesto SHA-256 en `artifacts/orbita/bytes/manifest_sets.json`)
- **R**: `llm_e2e/{ph}/robust.tsv` (es/en/fr, plantillas 6–7 nunca vistas, números en palabras/fracciones/grados/vueltas/radianes), 96 textos por seed.
- **U**: pt/it/ru (ru en cirílico), 2 plantillas nuevas, números en dígitos con coma; mismos ejemplos que R. Ningún brazo vio esos idiomas en su entrenamiento supervisado (Gemma sí en preentrenamiento).
- **N**: R con ruido estilo BLT (aleatorio-mayúsculas, borrado 10 %, repetición 10 %, todo mayúsculas; uno por texto, rotando).
- **Sorpresa**: `learn/{ph}/texts.tsv` + `learn-eval` k=2 con τ y (D, λ) congelados de `learn/tau.json` y `learn/selected.json` (no se re-ajustan).
- Tokens Gemma para U y N: una sola instancia de `gemma-embed` a la vez.

## Fases
DEV seeds `0xA400–0xA40F` (MAIN + ablaciones) → CONFIRM seeds `0xB400–0xB40F` (sólo MAIN vs GEMMA), configuración congelada en este commit. 16 seeds por fase. IC95 bootstrap pareado por seed (10000 remuestreos, semilla 12345). Las conclusiones son las de CONFIRM; DEV es descriptivo. Si DEV revela un fallo de implementación, se detiene la cadena antes de CONFIRM y se documenta una enmienda con commit propio antes de reanudar.

## Hipótesis
Primarias (no inferioridad, δ = 0,05 absoluto, diferencia MAIN − GEMMA pareada por seed; no inferior si el límite inferior del IC95 > −δ, o para errores si el superior < +δ):
- **H1** lectura de familia: acc argmax familia en R.
- **H2** lectura de parámetros: mediana por seed de max|p̂−p| en R tras numnorm/fallback (menor es mejor).
- **H3** predicción: acc h16 de ÓRBITA en R.
- **H6** sorpresa: AUROC de `s_known` (nuevas 8–9 vs conocidas 0–7), k=2.
Un fallo de H3 o H6 bloquea cambiar el encoder por defecto aunque H5 se cumpla.

Secundarias:
- **H4** multilingüe: acc familia en U; MAIN superior si IC95 de (MAIN − GEMMA) > 0 (también por idioma, descriptivo).
- **H5** eficiencia: latencia mediana y memoria pico por consulta (32 consultas, proceso nuevo por consulta como en `demo.py`; GEMMA = `gemma-embed` + D2, BYTES = encoder); criterio razón de latencia GEMMA/BYTES ≥ 10 (IC95 bootstrap sobre consultas) y menor memoria pico.
- **H7** fidelidad: exactitud de bytes reconstruidos (decodificación voraz parche a parche desde μ, longitudes de parche conocidas) en R limpio; criterio límite inferior IC95 ≥ 0,99.
- **H8** ruido: caída de acc familia R→N; MAIN mejor si IC95 de (caída MAIN − caída GEMMA) < 0.
- Descriptivas: H3b acc h16 MAIN − 0,323 (referencia del Page), max|p̂−p| de la cabeza sin numnorm, AUROC residuo teacher, ablaciones.
- **H9** exploratoria: no ejecutada (ver arriba).

## Ejecución
Cadena reanudable `orbita/llm/bytes_chain.sh` (enlazada en `/workspace/orbita-data/bytes/chain.sh`, toma su propio `flock`), datos en `/workspace/orbita-data/bytes/`, resumen en `artifacts/orbita/bytes/summary_{dev,confirm}.md` (`bytes_exp.py analyze`). No se inventan números: todo valor reportado sale de esos archivos.
