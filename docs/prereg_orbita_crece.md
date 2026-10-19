# Prerregistro — ÓRBITA que crece: expertos añadidos por sorpresa persistente

Rama `exp/orbita-yo`. Parte de 9166f1f (`docs/resultados_latente_largo_plazo.md`). Allí vimos tres cosas. Los ajustes por episodio no extrapolan las familias nuevas no afines. La memoria LAT agrupa bien, pero no transfiere a familias nuevas. Las familias conocidas mejoran sin olvido.
Este documento se escribe y se sube **antes** de generar los datos de este experimento y antes de ejecutar nada con ellos.

## 1. Modelo GROW (código: `orbita/src/grow.rs`, subcomando `grow-eval`)
- **Base congelada.**
  - ÓRBITA-v1: núcleo Koopman + corrector ICNN.
  - Los pesos `w_{seed}.json` no cambian. El driver comprueba sha256 antes y después, y el binario hace un assert de igualdad.
- **Capacidad nueva = experto.**
  - Es un observable no lineal aprendido: MLP 4→32→32→2 (tanh, 1282 parámetros).
  - Entrada [x/2, y/2, p̂₁, p̂₂].
  - Salida: residuo sumado a la predicción de ÓRBITA con contexto c(g0, p̂). g0 es la familia identificada g* más frecuente en su grupo semilla.
  - Se aplica punto a punto sobre los 6 puntos.
- **Disparo.**
  - Un episodio con k pasos observados y S_known > τ_k que ningún experto explica (S ≤ ce·τ_k) entra en el *pool* (FIFO, 600).
  - La sorpresa es persistente cuando hay ≥ 15 episodios en el pool en el momento de una consolidación.
- **Consolidación**, cada 150 episodios del flujo, en este orden:
  1. **Repaso (rehearsal).** Cada experto que recibió episodios nuevos se reentrena 3000 pasos con su buffer reservorio (300 episodios). Eso incluye episodios antiguos, para protegerlo contra el olvido. Los expertos sin datos nuevos no se tocan, y ÓRBITA nunca se toca.
  2. **Presupuesto.** Como máximo E_MAX = 8 expertos (8·1282 = 10256 parámetros). Con el presupuesto lleno y el pool ≥ 15, se desaloja el experto más antiguo que no tenga ningún uso en el flujo desde su creación (edad ≥ 2 consolidaciones).
  3. **Crecimiento.** Hasta 3 intentos. En cada intento:
     - Semilla: el mayor grupo del pool con la misma g* (≥ 8, cada g* una vez).
     - Ajuste: 2/3 aleatorios del grupo, 8000 pasos de Adam (lr coseno 3e-3→1e-4, batch 128, pérdida de un paso con peso 1/‖y‖²), más 2 reajustes de 3000 pasos sobre la mitad mejor explicada.
     - Reclama los episodios del pool con S ≤ ce·τ_k.
     - Se acepta si reclama ≥ 8 y al menos 4 de ellos están **fuera de muestra** (no usados en el ajuste). Si se acepta, se reentrena con lo reclamado y esos episodios salen del pool.
  4. **Fusión/compactación.** Como mucho una por consolidación.
     - Precondición: ≥ 50 % de los 10 últimos episodios de un experto tienen S ≤ 2·ce·τ bajo el otro.
     - Se fusiona (1000 pasos sobre la unión) si la tasa de reclamación fusionada es ≥ 0,9 × la media de las originales.
- **Ruta (predicción)**, la misma en el flujo y en el held-out:
  - *known* si S_known ≤ τ_k y no hay experto con S menor;
  - si no, el mejor experto si su S ≤ ce·τ_k;
  - si no, *fallback* = LAT_EP (ajuste latente por episodio de 9166f1f).
- **Sólo se usan datos observados.** Por episodio se guardan y_1..y_k. La familia real sólo se usa para el oráculo FAMNET y para analizar.

## 2. Controles
- **OBS**: ÓRBITA con ĉ.
- **LAT_EP**: ajuste latente por episodio, λ0 = 1e-4.
- **LAT**: memoria latente de 9166f1f (λ0 = 1e-4, λm = 1e-3, fijados allí). Corre en el mismo flujo. Se evalúa en held-out en n = 0 y en n final.
- **ORACLE_EP** (D = 2, λ = 1e-4) y **ORACLE_MS** (D = 2, λ = 1e-4): fijados en 9166f1f, con los mismos k pasos observados del episodio.
- **MONO**, el control sin crecimiento con igual número de parámetros:
  - MLP (4+8)→93→93→2 = 10139 parámetros (≤ 10256), entrada + onehot(g*), base ÓRBITA c(g*, p̂).
  - Se entrena con **todos** los episodios disparados (S_known > τ_k), con reservorio de 1000, en las mismas consolidaciones (8000 pasos la primera vez, 3000 después).
  - Ruta idéntica a GROW, con MONO como único experto.
- **ORACLE_FAMNET** (sin criterio): un MLP 4→32→32→2 por familia **real** con base identidad. Se entrena con todos los episodios observados de esa familia (reservorio de 300). Es un techo con etiqueta privilegiada.

## 3. Datos (nuevos; RNG frescos)
- 12 familias.
  - 0–7: v4.
  - 8–9: las nuevas de 9166f1f.
  - **10 «pulso radial»** (es «un pulso radial», en «a radial pulse», fr «une pulsation radiale»): polar, r ← r·(1 + 0,15 p₁ sin 3θ), θ ← θ + 0,5 p₀.
  - **11 «cizalla saturada»** (es «una cizalla saturada», en «a saturated shear», fr «un cisaillement saturé»): u = x + 0,3 p₀ tanh(2y); (x, y) ← (u, y + 0,3 p₁ tanh(2u)).
  - Familias nuevas = 8–11 (no afines).
- `fam-set --nf 12`.
  - H (held-out): 16 por familia × 3 idiomas = 576 textos; 1 idioma por episodio y seed → 192 episodios.
  - S (flujo): 200 por familia = 2400 episodios, un idioma cada uno.
  - RNG DEV H 0xFA71 / S 0xFA70; CONFIRM H 0xFA73 / S 0xFA72.
- Pipeline: Gemma 2 2B (capa 13, embeddings por token) → D2 de cada seed (argmax de familia en 8) + parser numnorm p̂ → ĉ.
- Flujo de 2400 episodios en orden aleatorio por seed, con k uniforme en {2, 4, 8}.
- Puntos de control 0, 150, 300, 600, 1200, 2400. En cada uno se evalúa H con el estado **congelado**, para k_eval ∈ {2, 4, 8}.
- 16 seeds DEV (0xA400–0xA40F) y 16 CONFIRM (0xB400–0xB40F). 0 queries TEST.

## 4. Calibración y selección (sólo DEV)
- τ_k = P95 de S_known en H DEV, familias 1–2 (lat-eval exp1 con τ = 1e9), sobre los datos nuevos.
- Rejilla ce ∈ {1, 2, 4}.
  - Se elige el máximo de acc16 GROW en held-out DEV en n = 2400, promediando todas las familias y los k. En caso de empate, el menor ce.
  - CONFIRM se ejecuta una sola vez con el ce elegido.
- El resto de hiperparámetros quedan fijados en este documento.

## 5. Hipótesis y criterio «aprende lo nuevo a largo plazo»
- acc16: rel_err < 0,05 en t = 16, con rollout desde y_k.
- Media por seed sobre los episodios del grupo. Bootstrap pareado por seed (10000 réplicas, seed 12345), IC95. δ = 0,05.
- Primario: **k_eval = 2** (transferencia con pocos pasos observados), familias nuevas 8–11. También se informan k = 4 y 8.

| id | cantidad | criterio |
|---|---|---|
| G1 | pendiente de acc16 GROW (nuevas) vs log2 n, n ∈ {150, …, 2400} | IC95 > 0 |
| G2 | acc16 GROW nuevas n = 2400 − n = 0 | IC95 > 0 |
| G4 | acc16 GROW nuevas n = 2400 − ORACLE_EP y − ORACLE_MS | ambos IC95 > 0 |
| G6 | acc16 GROW conocidas 0–7 n = 2400 − OBS, y − n = 150 | ambos IC95 inferior > −δ |
| H_LAT | acc16 GROW − LAT nuevas n = 2400 | IC95 > 0 |
| H_MONO | acc16 GROW − MONO nuevas n = 2400 | IC95 > 0 |

- **«GROW aprende lo nuevo a largo plazo»** = G1 ∧ G2 ∧ G4 ∧ G6 en k_eval = 2, en DEV **y** en CONFIRM.
- H_LAT y H_MONO se valoran aparte. «El crecimiento aporta sobre un modelo fijo de igual tamaño» = H_MONO en DEV y CONFIRM.
- Se informan sin criterio:
  - ORACLE_FAMNET;
  - eventos de crecimiento, fallo, fusión y desalojo; nº de expertos y parámetros;
  - pureza de ruta (la familia mayoritaria del experto coincide con la real) y tasa de ruta a experto en nuevas y en conocidas;
  - curvas precuenciales en el flujo, en bloques de 300 (GROW, MONO, LAT, LAT_EP, OBS; y GROW − LAT_EP con k = 2).

## 6. Prueba de humo previa (declarada)
Antes de este documento, el código se depuró con datos **ya publicados** de 9166f1f (`latlp/dev`, seed 0xA400, 10 familias, flujo truncado a 150–600 episodios, 20 episodios H). No se tomaron decisiones mirando la precisión held-out. Se miró la sorpresa en muestra del ajuste y los eventos de crecimiento. Cambios que salieron de ahí:
- más pasos de Adam (1500 → 8000) con lr coseno y reinicio de momentos;
- semilla por grupo de g*;
- aceptación con reclamación fuera de muestra (≥ 4);
- desalojo por presupuesto;
- pool de 600.

En esa prueba, los expertos nuevos rara vez reclamaban episodios fuera de muestra de las familias 8–9. Sí lo hacían en la familia 7 (conocida con S alta). Ningún número de esa prueba se informa como resultado.

## 7. Ejecución
`orbita/llm/crece_chain.sh`, con setsid nohup y reanudable con ficheros .done. Un solo proceso Gemma. NPAR = 6. Artefactos en `artifacts/orbita/orbita_crece/`. Resultados en `docs/resultados_orbita_crece.md`.
