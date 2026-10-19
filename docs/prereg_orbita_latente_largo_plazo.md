# Prerregistro: adaptación en el latente de Koopman con k > 2 (Exp. 1) y aprendizaje a largo plazo en un flujo largo (Exp. 2)

Rama `exp/orbita-yo` (tras e2ac4ab). Publicado y empujado **antes** de ejecutar. Única ejecución previa: prueba de regresión/tiempo del binario
sobre la seed 0xA400 del conjunto DEV antiguo (`/workspace/orbita-data/learn/dev`, ya publicado en e2ac4ab): se comprobó que las columnas
BASE/OBS/RES/RES_EP/ORACLE_EP(D=2, λ=1e-4)/LS reproducen exactamente las de e2ac4ab (160/160 filas) y se midió tiempo (k = 2: 2,8 s; k = 8: 14,8 s);
**no se miraron** las columnas de los métodos nuevos. Pila: texto (es/en/fr, números en palabras/fracciones/grados/vueltas/radianes) → Gemma 2 2B
capa 13 → D2 + numnorm → ĉ = (familia, p̂) → **ÓRBITA-v1 congelada** (pesos sólo leídos; sha256 antes/después). Código: `orbita/src/latent.rs`
(`lat-eval exp1|long`), `orbita/src/fam.rs` (`fam-set --rng --n`), `orbita/llm/latlp.py`, `latlp_chain.sh`. Datos en `/workspace/orbita-data/latlp/`.

## Datos (todos nuevos; ninguno lee `ds.test`; 0 queries v4 a TEST)
- Por fase, dos conjuntos `fam-set` (familias 0–9: v4 0–7 conocidas, 8 giro ondulado y 9 espiral nuevas no afines):
  **H** (16 ejemplos/familia = 160; DEV RNG 0xFA61, CONFIRM 0xFA63; 3 textos por ejemplo, como en e2ac4ab) y
  **S** (200 ejemplos/familia = 2000; DEV 0xFA60, CONFIRM 0xFA62; 1 texto por ejemplo: idioma = idx mod 3, plantilla 6 + idx mod 2).
- Seeds: DEV 0xA400–0xA40F (D2/ÓRBITA DEV), CONFIRM 0xB400–0xB40F. Un proceso Gemma a la vez; D2 por lotes.

## Módulos (núcleo congelado intacto)
- **LAT** (nuevo, en el latente de Koopman): por punto q, z = enc(q; c(g0, p̂)) ∈ R⁸ (observables congelados de ÓRBITA),
  u = K(c)·z + Aᵀ ξ(q), ξ = [z, monomios de q de grado 2, 1] (12), q' = Brenier(u)[0..2] (corrector congelado). A (12×8 = 96 parámetros) se
  ajusta con Gauss-Newton/Levenberg-Marquardt sobre la **pérdida multipaso** desde x0 (Σ_{t≤k} ‖(F^t(x0) − y_t)/‖y_t‖‖²) + λ‖A − A_prior‖².
- **RES**: adaptador residual + biblioteca de e2ac4ab, sin cambios (D = 2, λ = 1e-4); **RES_EP**: sólo episodio.
- **Biblioteca nueva (memoria episódica)**, arreglo de HL3: cada episodio aprendido guarda (g0, A, familia verdadera sólo para métricas). En un
  episodio nuevo se verifica cada entrada guardada con los k pasos observados (sorpresa S_e); el prior es la media ponderada
  (w = exp(−(S_e − S_min)/τ_k)) de las 8 mejores entradas con el mismo g0 que la mejor; luego ajuste MAP (λ_m) con los datos del episodio.
  Sustituye la regresión agrupada lineal en p̂ (dependencia en p no paramétrica y local, sin clústeres duros).
- Regla de decisión LAT: S_known (re-identificación de e2ac4ab, g ∈ 8, p ∈ {p̂} ∪ malla 9×9); S_lib = min S_e. Si S_known ≤ τ_k y
  S_known ≤ S_lib → conocido (= OBSk). Si no, si S_lib < S_known → prior de memoria + MAP (λ_m), se guarda. Si no → ajuste desde A = 0
  (g0 = g*, λ_0), se guarda. **LAT_EP**: sin memoria (conocido si S_known ≤ τ_k; si no, ajuste desde 0). **FINAL**: memoria congelada al final
  del flujo, misma regla sin guardar.
- **Oráculos justos por episodio** (mismos k pasos, sin núcleo): **ORACLE_EP** (q' = q + Wᵀ Φ, ridge a un paso, como e2ac4ab; D ∈ {2, 3},
  λ ∈ {1e-4 … 1}), **ORACLE_MS** (q' = q + Wᵀ φ_D(q), misma pérdida multipaso y optimizador que LAT; D ∈ {2, 3}, λ ∈ {1e-4 … 1e-1}).
  ORACLE_LS reportado.
- τ_k (k = 2, 4, 8): P95 de S_known en los episodios de familias 1–2 de **H DEV** (16 seeds).

## Exp. 1 — k ∈ {2, 4, 8} sobre H (flujo de 160 episodios por seed, orden e idioma por seed, memoria vacía al inicio)
Métrica primaria **acc16** (rel_err < 0,05 en t = 16; rollout desde y_k). Selección sólo en DEV con puntuación = media sobre k ∈ {2, 4, 8} de la
acc16 de los 160 episodios: LAT_EP elige λ_0 ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}; LAT elige (λ_0, λ_m) conjuntamente, λ_m ∈ {1e-3, 1e-2, 1e-1, 1, 10};
cada oráculo elige su (D, λ); empates → λ mayor. Se congela para CONFIRM y para el Exp. 2.
Hipótesis (por cada k ∈ {2, 4, 8}; nuevas 8–9 salvo indicación; δ = 0,05; IC95 bootstrap pareado por seed, 10 000, semilla 12345; aceptar si se
cumple en DEV **y** CONFIRM):
- **H1a** LAT_EP − RES_EP > 0 (el módulo latente multipaso supera al residual con los mismos datos). **H1b** LAT − RES > 0 (sistemas completos).
- **H2** LAT − LAT_EP > 0 (la memoria nueva ayuda; arreglo de HL3).
- **H3** LAT − OBS > 0.
- **H4** LAT − ORACLE_EP > −δ **y** LAT − ORACLE_MS > −δ (no inferioridad frente a ambos oráculos por episodio); superioridad reportada.
- **H5** (sin olvido) LAT − OBS y FINAL − OBS en conocidas 0–7 y en 1–2, IC95 inferior > −δ; sha256 de pesos idéntico.

## Exp. 2 — flujo largo (2000 episodios de S por seed; held-out = H de la misma fase)
- Orden de S por seed (`default_rng(seed)`), k de cada episodio del flujo uniforme en {2, 4, 8} (misma RNG); mezcla 80 % conocidas / 20 % nuevas
  (por construcción de S). Memoria LAT acumulada durante todo el flujo (hiperparámetros del Exp. 1).
- Puntos de control n ∈ {0, 125, 250, 500, 1000, 2000} episodios vistos: se evalúan los 160 episodios de H (un texto por ejemplo, idioma por seed)
  **con la memoria congelada** (no se guarda nada de H) a k_eval ∈ {2, 4, 8}. n = 0 equivale a LAT_EP. OBS y oráculos por episodio no dependen de n.
- Criterios de «aprende a largo plazo» (primario k_eval = 2, que mide transferencia a episodios con menos pasos; k = 4, 8 secundarios; nuevas
  8–9 en H; mismo bootstrap; DEV y CONFIRM):
  - **L1** pendiente por seed (MCO) de acc16 frente a log2 n, n ∈ {125, …, 2000}: IC95 > 0.
  - **L2** acc16(n = 2000) − acc16(n = 0) > 0.
  - **L3** acc16(n = 2000) − ORACLE_EP > 0 **y** acc16(n = 2000) − ORACLE_MS > 0 (supera a ambos oráculos por episodio con los mismos k pasos).
  - **L4** (sin olvido a lo largo del tiempo) conocidas 0–7 en H: acc16(2000) − OBS > −δ y acc16(2000) − acc16(125) > −δ.
  - **Aprende a largo plazo** ⇔ L1 ∧ L2 ∧ L3 ∧ L4 en k_eval = 2, en DEV y CONFIRM.
- Reportado sin criterio: curva completa por n y k_eval; curva precuencial en el flujo (acc16 nuevas por bloques de 250 episodios: LAT, LAT_EP,
  OBS); tamaño de la memoria; pureza de la recuperación (peso del prior en entradas de la misma familia verdadera) en H nuevas; tasas de acción.
- Ningún número se inventa: todo sale de `artifacts/orbita/latente_largo_plazo/`. Resultados en `docs/resultados_latente_largo_plazo.md`.
