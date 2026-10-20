# Prerregistro — Cognitive Transformer + ÓRBITA (atención estado↔memoria CDT-RQM)

**Rama:** `exp/cognitive-transformer-orbita` · **Documento de diseño:** `docs/cognitive_transformer_orbita.md` (3dfb95c) · **Fecha:** 2026-10-20 (CST, UTC−6).
Escrito y empujado **antes** de ejecutar DEV/CONFIRM. Antes de este documento sólo se corrió una prueba de humo (100 pasos de entrenamiento, seed 0xA400 DEV,
salida en `/tmp`, descartada) para medir tiempos (≈0,11 s/paso; ≈390 s/seed de evaluación) y comprobar que el código corre; no se usó para ninguna decisión.
Paridad comprobada antes de ejecutar: `rule()` Python vs `orbita demo-rule` (Rust) max |Δ| = 1,1e-16 en las 12 familias; ÓRBITA Python vs `orbita demo-step` 16 pasos max |Δ| = 4,1e-15.

## 1. Qué se implementa (fases del documento)
El documento (§15, §22, §25) pide, en orden: estado sintético sin lenguaje → ÓRBITA como baseline → atención sobre slots de estado → híbrido Attention→ÓRBITA →
memoria (retrieval de estados estructurados) → consolidación por error de predicción → periferia LLM → matriz de ablaciones. Implementación (`orbita/cto/cto.py`):

- **Slots de estado no lingüísticos:** el estado es el conjunto de 6 puntos del objeto; cada slot `s_i` es el latente ÓRBITA 8-d del punto (`z = [x, g(x,c)]`). Sin tokens lingüísticos.
- **Q desde ÓRBITA:** `q_i = f([z_i, ÓRBITA(z)_i[:2] − x_i, sig])` (estado, predicción ÓRBITA de un paso y firma del episodio).
- **K, V desde memoria:** cada entrada es una transición observada (slot, paso): clave `[z_j, sig_episodio]`, valor `[Δx_j, e_j = y_j − ÓRBITA(x_j), x_j]`.
  La memoria = transiciones observadas del episodio actual (k pasos) ∪ episodios recuperados del banco **CDT-RQM** (top-R = 4 por distancia de firma; `Retrieve(CDT-RQM, z_t)`).
- **Δz residual + ÓRBITA:** atención multi-cabeza (4×8), `z' = z + W_O A` (W_O inicializada a 0), `z_{t+1} = ÓRBITA(z', c)` con los pesos ÓRBITA-v1 publicados de cada seed **congelados** (sha256 verificado antes/después).
- **Consolidación CDT-RQM por error de predicción:** en el flujo S, un episodio se consolida (sus entradas y su firma) si la sorpresa ÓRBITA (rel_err medio en los pasos observados) > τ_k;
  capacidad 600 episodios, desalojo del de menor error. Los pesos no se actualizan en el flujo: lo nuevo sólo puede entrar por la memoria.
- **Periferia LLM:** el contexto `c` viene del encoder D2/numnorm sobre Gemma 2 2B (texto en es/en/fr), reutilizando el caché ya extraído en `exp/orbita-yo`
  (`/workspace/orbita-data/crece/{dev,confirm}/ctx/{H,S}_<seed>.csv`, mismos conjuntos H y S que «ÓRBITA que crece»). **No se ejecuta Gemma** en este experimento.
  El decoder estado→lenguaje no se implementa (fuera de alcance; limitación declarada).
- **Entrenamiento (sólo familias conocidas 0–7):** generador propio (RNG por seed, nunca H/S/TEST), k ∈ {2,4,8}, pérdida log1p(rel_err²) sobre el rollout desde x_k hasta el paso 16,
  4000 pasos Adam, batch 32, lr coseno 3e-3→3e-5. Con prob. 0,3 el contexto se sustituye por el de otra familia/parámetros (simula la periferia equivocándose, como ocurre con las familias nuevas).
  Durante el entrenamiento el banco CDT es un banco propio de 512 episodios de familias 0–7.

## 2. Brazos
| brazo | qué es | entrenado |
|---|---|---|
| ORB | ÓRBITA sola (pesos publicados, contexto Gemma), rollout desde x_k | no |
| **CT** | Cognitive Transformer completo: atención estado→(episodio + CDT-RQM) → Δz → ÓRBITA | sí |
| CT_NOCDT | mismos pesos que CT, banco CDT vacío (sólo memoria del episodio) | — |
| CT_RAND | mismos pesos, memoria (claves, valores y firma) sustituida por ruido gaussiano con media/desv. de la memoria real | — |
| CT_SHUF | mismos pesos, memoria (episodio + recuperación CDT + firma) de otro episodio de H (desarreglo) | — |
| SELF | atención sin memoria CDT ni transiciones: auto-atención sobre los slots de la historia de estados observados x_0..x_k (+ posición temporal), Δz → ÓRBITA | sí |
| MLP | control de parámetros emparejados (4755 vs 4744 de CT): Δz = MLP([q_i, media de claves, media de valores]) con la misma memoria que CT, sin selección por atención | sí |
| NOORB | misma atención y memoria que CT pero **sin ÓRBITA**: dinámica libre aprendida `x + g(x,c)` (5442 parámetros) | sí |
| ORACLE_EP | oráculo justo por episodio: ajuste afín 2×3 compartido por puntos con los mismos k pasos observados (como `ls_fit`) | — |
| ORACLE_KNN | oráculo justo con **los mismos datos de memoria** que CT (episodio + CDT recuperada): ÓRBITA + regresión de núcleo gaussiano en x sobre los residuos e_j (h = 0,3 fijado a priori; atención no aprendida) | — |

## 3. Evaluación
- Conjunto H (192 episodios, 16 por familia 0–11, fijo por fase; `crece/{ph}/H_set.csv`), contexto Gemma por seed; k ∈ {2,4,8}; se observan x_1..x_k y se predice hasta el paso absoluto 16.
- Flujo S (2400 episodios, k del caché) sólo para consolidación; H se evalúa con el banco congelado en n ∈ {0, 150, 600, 2400}.
- **acc16** = rel_err(x̂_16, x_16) < 0,05 (igual que `v4_metrics::correct`). Secundarias: acc a h8, h32, h64, h128, h256 (horizontes absolutos, verdad por la regla) y rel_err16 mediana.
- **Sorpresa / AUROC** (nuevas 8–11 = positivo vs conocidas 0–7): ÓRBITA (rel_err medio en pasos observados) y CT (rel_err del último paso observado prediciéndolo con memoria de los pasos anteriores + CDT).
- **Olvido:** acc16 conocidas CT n = 2400 − n = 0.
- **Coste:** parámetros entrenables, tiempo de entrenamiento y de evaluación por seed.
- Semillas: DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F` (16 cada una; cada seed usa sus pesos ÓRBITA publicados y su contexto Gemma). 0 queries a TEST (no se genera ni se lee ninguna partición TEST).
- Estadística: media por seed; diferencias pareadas por seed; IC95 bootstrap (10 000 réplicas, semilla 12345); δ = 0,05.
- τ_k = P95 de la sorpresa ÓRBITA en H DEV, familias 1–2, 16 seeds (misma regla que `crece`); es la única calibración y sólo usa DEV y ÓRBITA. No hay rejilla de hiperparámetros.

## 4. Hipótesis (primario k = 2, n = 2400, acc16; k = 4, 8 secundarios)
Las hipótesis del documento (§17) hechas comprobables:
| id | hipótesis | cantidad | aceptación |
|---|---|---|---|
| H1 | la atención estado↔memoria mejora la predicción de lo conocido | CT − ORB, conocidas 0–7 | IC95 > 0 |
| H2 | … y de lo nuevo (doc §17 H4: la memoria ayuda cuando el futuro depende de información no contenida en el estado local) | CT − ORB, nuevas 8–11 | IC95 > 0 |
| H3 | la consolidación CDT-RQM por error aprende familias nuevas | CT n=2400 − CT n=0, nuevas (≡ CT − CT_NOCDT) | IC95 > 0 |
| H4 | la mejora depende del contenido de la memoria (doc §14 memoria aleatoria) | CT − CT_SHUF y CT − CT_RAND, en nuevas y en conocidas | los 4 IC95 > 0 |
| H5 | la memoria de transiciones/CDT aporta sobre auto-atención en la historia de estados | CT − SELF, nuevas y conocidas | ambos IC95 > 0 |
| H6 | la selección por atención aporta sobre un MLP de igual tamaño con la misma memoria (doc §17 H1, riesgo 2 y 6) | CT − MLP, nuevas y conocidas | ambos IC95 > 0 |
| H7 | complementariedad atención + ÓRBITA (doc §17 H3) | CT − NOORB, nuevas y conocidas | ambos IC95 > 0 |
| H8 | supera al oráculo justo con los mismos datos (doc riesgo 5) | CT − ORACLE_EP y CT − ORACLE_KNN, nuevas | ambos IC95 > 0 |
| H9 | sin olvido | CT n=2400 − n=0, conocidas | IC95 inferior > −δ |
| H10 | la memoria hace menos sorprendente lo nuevo | AUROC_CT(n=2400) − AUROC_CT(n=0) | IC95 < 0 |

Regla: una hipótesis se **acepta** sólo si el criterio se cumple en DEV **y** en CONFIRM. Se etiqueta **refutada** si en CONFIRM el IC95 queda entero del lado contrario
(< 0, o en H9 superior < −δ; en H10 > 0) e **inconclusa** en otro caso. Criterio del documento §24 («Attention+ÓRBITA > Attention ≥≈ ÓRBITA > baseline») se informa
con CT − NOORB (H7), NOORB − ORB y CT − MLP; **éxito de la rama** = H1 ∧ H7 ∧ H6 en conocidas o nuevas, reproducido en DEV y CONFIRM.
Secundarios (k = 4, 8; horizontes h8–h256; subgrupos 8–9, 10–11, 1–2) se reportan sin criterio.

## 5. Desviaciones conocidas respecto al documento
- No se replica la receta de entrenamiento de ÓRBITA (256 000 pasos): ÓRBITA se reutiliza congelada; los módulos nuevos usan 4000 pasos (presupuesto de CPU compartida).
- Las entidades no interactúan en estas familias (la regla actúa por punto): la relación que la atención puede explotar es estado↔memoria, no punto↔punto. Riesgo 2 del documento.
- Fase 4 (LLM) sólo como encoder ya extraído; sin decoder; sin paráfrasis nuevas (H5 del documento, independencia lingüística, no se prueba aquí).

## 6. Ejecución
`orbita/cto/chain.sh` lanzado con `setsid nohup`, reanudable (`.done`, salidas atómicas), 4 trabajos en paralelo con `nice 10` (CPU compartida con `exp/orbita-tf`).
Salidas: `/workspace/orbita-data/cto/`; tablas: `artifacts/orbita/cognitive_transformer/summary_{dev,confirm}.md`; resultados: `docs/resultados_cognitive_transformer_orbita.md`.
