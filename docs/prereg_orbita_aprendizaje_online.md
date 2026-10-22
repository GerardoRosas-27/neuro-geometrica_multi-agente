# Prerregistro: ÓRBITA aprende dinámicas nuevas en línea (bucle predecir → verificar por sorpresa → aprender → seguir)

Rama `exp/orbita-yo` (tras 282f382). Publicado **antes** de ejecutar cualquier evaluación de este experimento (no hay ejecución previa:
ni piloto, ni extracción Gemma de los conjuntos nuevos). Pila: texto (es/en/fr, números en palabras/fracciones/grados/vueltas/radianes)
→ Gemma 2 2B capa 13 → D2 (3000 pasos, pool2) + numnorm → ĉ = (familia, p̂) → **ÓRBITA-v1 congelada** (núcleo Koopman + corrector
Brenier ICNN; sus pesos no se modifican: se verifica sha256 antes/después). El yo no decide (sólo informa; no interviene aquí).

## Motivación
`docs/resultados_otras_familias.md`: la re-observación (OBSk) mejora familias conocidas pero en familias nuevas no afines (8 giro
ondulado, 9 espiral) acc16 ≈ 0, porque ÓRBITA no tiene esa dinámica. La sorpresa (predicción vs observación) es buena señal de error.
Pieza que falta: que el sistema **aprenda en línea** un módulo de dinámica nuevo a partir de los k pasos observados, lo guarde en una
**biblioteca de familias aprendidas** reutilizable entre episodios, sin tocar el núcleo.

## Datos
- Dos conjuntos **nuevos** con `orbita fam-set --rng R` (mismo generador que otras familias: 16 ejemplos por familia 0–9, formas
  blob/círculo/triángulo, p ~ U[−1,1]²; no lee `ds.test`): **DEV** R = 0xFA50, **CONFIRM** R = 0xFA51. Textos como en otras familias
  (3 idiomas, plantillas 6–7, formatos numéricos de `e2e.fmts`, nombres nuevos para 8–9); 480 textos por conjunto, embebidos una vez
  (un solo proceso Gemma a la vez).
- **Episodio** = un ejemplo con un texto. Por seed, flujo de 160 episodios (cada ejemplo una vez; idioma = (índice del ejemplo + entero
  de la seed) mod 3; orden = permutación `numpy.default_rng(int(seed))`). 16 seeds DEV (0xA400–0xA40F, D2/ÓRBITA DEV) y 16 seeds CONFIRM
  (0xB400–0xB40F). Biblioteca vacía al inicio de cada seed.
- Grupos: **nuevas 8–9** (primario), **conocidas 1–2**, **v4 no iterables 0, 3–7**, **conocidas 0–7**.

## Bucle (k ∈ {1, 2} pasos observados; primario k = 2)
Sorpresa de un predictor F: S(F) = (1/k) Σ_{t≤k} min(rel_err(F^t(x0), y_t), 10).
1. **Predecir/verificar con lo conocido**: re-identificación de `fam-eval` (g ∈ 8 familias, p ∈ {p̂} ∪ malla 9×9) → S_known, (g*, p*).
   Cada módulo m de la biblioteca → S_m (antes de actualizarlo).
2. Si min(S_known, S_m) ≤ τ_k: si gana lo conocido → igual que OBSk; si gana un módulo m → se añaden los pares observados a su búfer,
   se reajusta y se usa.
3. Si no (sorpresa > τ_k): para cada m se reajusta con búfer ∪ episodio y se mide S' en el episodio; si min S' ≤ τ_k el episodio se une
   a ese módulo; si no, **se crea un módulo nuevo** (base g0 = g*) ajustado sólo con el episodio y se añade a la biblioteca.
4. Rollout desde y_k con el predictor elegido; métricas ok8, **ok16** (rel_err < 0,05 a t = 8, 16) frente a la trayectoria real.

**Módulo** m (adaptador residual tipo lifting de Koopman, núcleo congelado intacto): por punto q,
F_m(q) = ÓRBITA(q; c(g0_m, p̂)) + W_mᵀ Φ(q, p̂), Φ = monomios de (x, y) de grado ≤ D ⊗ [1, p̂1, p̂2]; W_m por ridge (λ) sobre todos los
pares observados (x_t → y_{t+1}, con estados observados) del búfer. Sólo usa estados observados (nunca t > k del episodio actual).

**Umbral** τ_k: percentil 95 de S_known en los episodios DEV de las familias conocidas 1–2 (16 seeds pooled), fijado antes de la malla.
**Hiperparámetros** D ∈ {2, 3}, λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}, elegidos **sólo en DEV** maximizando acc16 media (160 episodios,
media sobre seeds) a k = 2 (empates → D menor, λ mayor), por separado para LEARN y para ORACLE_FAM (ORACLE_EP usa su propia elección
con la misma regla). La elección se congela para k = 1 y para CONFIRM. Nota: DEV queda sesgado a favor por la selección; CONFIRM no.

## Condiciones
- **BASE** (ĉ desde x0), **OBSk** (re-observación sin aprendizaje; control principal), **LEARNk** (bucle completo con biblioteca).
- **LEARN_EPk** (ablación de biblioteca): mismo bucle pero el módulo se ajusta sólo con el episodio y se descarta.
- **Oráculos justos** (mismos datos observados, sin núcleo ÓRBITA, se aplican a todos los episodios): **ORACLE_EPk** F(q) = q + WᵀΦ(q, p̂)
  ajustado sólo con los k pasos del episodio; **ORACLE_FAMk** igual pero con búfer de todos los episodios previos **de la misma familia
  verdadera** + el actual (privilegio: conoce la agrupación). ORACLE_LSk (afín por episodio, como antes) reportado.
- **FINALk** (olvido): tras el flujo, biblioteca congelada; cada episodio se re-evalúa eligiendo argmin S entre conocido y módulos (sin
  actualizar ni crear).

## Hipótesis (δ = 0,05; IC95 bootstrap pareado por seed, 10 000 réplicas, semilla 12345; DEV y luego CONFIRM; aceptar si se cumple en ambas)
- **HL1** (aprende lo nuevo): acc16 LEARN2 − OBS2 en nuevas 8–9, IC95 inferior > 0. (HL1' secundaria: k = 1.)
- **HL2** (no degrada / no olvida): acc16 LEARN2 − OBS2 en conocidas 0–7 **y** en 1–2, IC95 inferior > −δ; ídem FINAL2 − OBS2.
  Además sha256 de los pesos ÓRBITA idéntico antes/después.
- **HL3** (biblioteca reutilizable): acc16 LEARN2 − LEARN_EP2 en nuevas 8–9, IC95 inferior > 0.
- **HL4** (frente al oráculo justo): acc16 LEARN2 − ORACLE_FAM2 en nuevas 8–9, IC95 inferior > −δ (no inferioridad). LEARN2 − ORACLE_EP2
  reportado (superioridad: IC95 > 0).
- Reportado sin criterio: acc8; k = 1 para todas; AUROC de S_known (nuevas vs conocidas 0–7 y vs 1–2); tasa de disparo por grupo; nº de
  módulos creados, tasa de reutilización (episodios nuevos asignados a un módulo existente) y pureza (familia mayoritaria por módulo);
  curva de aprendizaje (acc16 por orden de aparición de la familia en el flujo: 1–4, 5–8, 9–16); acc16 por idioma y por formato numérico
  (palabras vs resto); parse_fail; queries v4 a TEST (deben ser 0).
- No se inventa ningún número: todo valor reportado sale de `artifacts/orbita/aprendizaje_online/`.

## Ejecución
Código `orbita/src/learn.rs` (`learn-eval`), `orbita/llm/learn_eval.py`, `learn_chain.sh`; `setsid nohup`, reanudable con `.done` en
`/workspace/orbita-data/learn/`; resultados en `docs/resultados_aprendizaje_online.md`; 0 queries a TEST.
