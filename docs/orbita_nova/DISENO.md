# ÓRBITA-Nova — Diseño (rama `exp/orbita-nova`, base `main` 590dd93)

Estado: **harness train listo** (2026-10-22). PREREG [`PREREG.md`](PREREG.md) `f64b289`; C6 real + flujo 2400 en `nova_train.rs` / `nova_full.sh` (ver [`ENTRENAMIENTO.md`](ENTRENAMIENTO.md)). Fase 0–1 PASS. Job 2400×seeds **pendiente** OK de CPU. Sin bytes; sin merge main.

## 0. Propuesta original (conversación compartida, resumida fielmente)
Fuente: https://chatgpt.com/share/6adaadd5-f6f8-83e8-a1f7-235313e9fc87 (leída del JSON embebido del HTML).
- Conservar el núcleo ÓRBITA y cambiar *cómo aprende dinámicas desconocidas*. No volver a Transformer convencional ni a CDT en inferencia.
- Pila: (1) periferia de entrada (texto/bytes/números) → (2) encoder de estado «independiente del idioma» → (3) núcleo dinámico adaptativo =
  ÓRBITA base + **adaptador de innovación** (aprende de errores) + **estimador de estado y confianza** (decide corregir/observar/actualizar)
  → (4) **aprendizaje en línea** gradual sin sobrescribir el núcleo.
- Ciclo ante una sorpresa: error predicho−observado → ¿ruido o dinámica nueva? → **corrección de bajo rango del operador** → verificar en
  observaciones posteriores → conservar sólo si pasa estabilidad y generalización.
- Descarta GROW como mecanismo principal. Experimento decisivo: ÓRBITA vs NOVA en conocidas (no perder), nuevas (pocas observaciones),
  horizontes 16–64, ruido (no aprender falsas dinámicas), idiomas (mismo estado). Mismas seeds, presupuesto y DEV/CONFIRM.
- Mencionaba «escalas temporales duales» e «innovaciones» (actualizar estado y parámetros). No pudo leer ÓRBITA-Bytes y no la usa.

## 1. Diagnóstico de partida (de `main`)
| hecho documentado | consecuencia para Nova |
|---|---|
| LAT/LAT_EP ya es una corrección latente aditiva `u = K(c)z + Aᵀξ(q)` ajustada por GN con prior (`latent.rs`) | la «corrección de bajo rango del operador» de la propuesta **ya existe en forma casi igual**; repetirla sin más no es nuevo |
| LAT: sin olvido en conocidas; **no** aprende nuevas a largo plazo; memoria crece sin compactar (~1100 entradas) | falta *consolidación* (pasar de episodio a conocimiento estable), no adaptación rápida |
| GROW: pureza alta al enrutar, pero 43–44 intentos fallidos/seed; agrupa por g* de D2 (mezcla nuevas y conocidas); ORACLE_FAMNET 0,29 en nuevas | el cuello de botella es **agrupar por dinámica observada** y aceptar por mejora en datos retenidos, no la capacidad |
| Cognitive Transformer: memoria consolidada por sorpresa + recuperación top-R sin umbral → colapso a 0; entrenamiento 4000 pasos empeoró incluso en distribución | toda memoria necesita **compuerta de no-daño** y debe evaluarse con banco de la misma distribución que el uso |
| TF-E: KL/energía libre en entrenamiento destruye h16; TF-PE (gradiente sobre latente) mejor en nuevas (exploratorio) | adaptación iterativa sobre latente es prometedora; **no** usar objetivo variacional |
| attn-gate: compuerta casi nunca cambia nada; corrector no converge en todas las seeds | criterio de convergencia previo obligatorio (fase 0) |
| Nuevas k=2 acc16 ≈ 0 en todos → hipótesis inconclusas por suelo | métrica primaria **continua** además de acc16 |
| Sorpresa AUROC 0,96–0,99; observación > «yo» | la sorpresa sirve como detector, pero la decisión debe basarse en verificación empírica |

## 2. Arquitectura Nova
Análogo cognitivo (como hipótesis de trabajo, no como afirmación): memoria de trabajo = estado/innovación del episodio; memoria episódica
= buffer de episodios sorprendentes; memoria a largo plazo = biblioteca de operadores consolidados; núcleo ÓRBITA = conocimiento semántico fijo.

### Componentes
| # | componente | reutiliza | nuevo |
|---|---|---|---|
| C1 | Periferia + encoder (Gemma capa 13 + D2/D3-num + numnorm) | íntegro (`gemma_embed.rs`, `d2/d3/numnorm.py`) | nada en fases 1–3; bytes/idiomas sólo en fase 5 |
| C2 | Núcleo ÓRBITA-v1 congelado (Koopman + Brenier) | íntegro, sha256 verificado | — |
| C3 | **Filtro de innovación** (memoria de trabajo): estado latente z con covarianza P, innovación ν = y − ŷ, ganancia tipo Kalman extendido usando el Jacobiano de ÓRBITA | `yo.rs` (σ_jacobiano), sorpresa de `learn.rs` | sí: corrige *estado* antes de tocar parámetros; separa ruido (ν dentro de S=HPHᵀ+R) de dinámica nueva (ν persistente) |
| C4 | **Adaptador rápido de bajo rango** (por episodio): ΔK = U Vᵀ, rango r∈{1,2}, en el espacio latente de 8 dims; ajuste GN/LM multipaso con prior | `fit_gn`, estructura LAT | rango explícito (LAT es denso M×nxi) y prior = operador recuperado de C6 |
| C5 | **Árbitro de verificación** (estimador de confianza): mantiene la adaptación sólo si mejora el error en pasos *retenidos* del mismo episodio (split k_fit / k_val) y no aumenta ‖·‖ espectral > 1+ε | idea de OBS/re-observación | sí: decisión por validación, no por umbral de sorpresa |
| C6 | **Biblioteca de operadores consolidados** (largo plazo): clusters por *firma de dinámica observada* (coeficientes ΔK y residuo), no por g*; un ΔK medio por cluster; recuperación con **umbral** y opción «ninguno» | memoria episódica LAT, lección GROW §Lectura 4 | sí: consolidación «offline» (sueño) en lotes; admisión de un cluster sólo si mejora en episodios retenidos y no daña conocidas |
| C7 | Compuerta de no-daño global | lección CT §7.1 | sí: si el mejor candidato no supera a ÓRBITA+C3 en validación, salida = ÓRBITA |

### Flujo de datos (un episodio)
```mermaid
flowchart TD
  T[Texto / números] --> E[C1 Encoder D3-num -> familia, p, contexto c]
  E --> O[C2 ÓRBITA congelada: z_t+1 = K(c) z_t]
  X[Observaciones y_1..y_k] --> F[C3 Filtro de innovación: nu, P, corrige estado]
  O --> F
  F -->|nu dentro de S: ruido| OUT[Predicción h1..h64]
  F -->|nu persistente: dinámica nueva| R[C6 Recuperar operador por firma, umbral]
  R --> A[C4 Ajuste rápido rango r con prior recuperado, k_fit]
  A --> V{C5 Mejora en k_val y estable?}
  V -->|no| G[C7 Compuerta: ÓRBITA+C3]
  V -->|sí| OUT
  G --> OUT
  A --> B[(Buffer episódico)]
  B -->|sueño por lotes| C[C6 Consolidación: cluster + validación retenida + test de no-daño]
  C --> R
```

### Escalas temporales
- rápida (paso): C3 actualiza estado; media (episodio): C4 ajusta ΔK; lenta (cada N episodios): C6 consolida. Núcleo C2: nunca.

## 3. Cómo evita las fallas documentadas
1. **Colapso CT**: recuperación con umbral + opción nula + C7; la memoria nunca entra sin pasar validación retenida; evaluación con banco
   generado por el mismo flujo que el test.
2. **Entrenamiento que empeora**: no hay red nueva entrenada por SGD a través de rollouts largos; C3–C6 son estimación cerrada/GN de pocos
   parámetros (r·2·8 ≤ 32). Fase 0 exige que C4 supere a ÓRBITA en su propia distribución antes de seguir.
3. **GROW**: se agrupa por dinámica observada y se acepta por mejora retenida (exactamente la recomendación del doc de GROW).
4. **TF-E**: sin KL/energía libre; sí inferencia iterativa sobre latente (H4b fue aceptada).
5. **Suelo en nuevas**: métrica primaria continua (log rel_err h16) + acc16 secundaria; k ∈ {2,4,8}, primario k=4.
6. **attn-gate**: criterios de convergencia C1–C4 antes de DEV.

## 4. Dónde difiero de la propuesta y por qué
| propuesta | Nova | motivo |
|---|---|---|
| «Corrección de bajo rango» como innovación principal | se mantiene, pero se reconoce que LAT ya lo hace; el aporte nuevo está en C3 (separar estado/parámetros), C5 (verificación retenida) y C6 (consolidación) | LAT ya está en main y no aprende nuevas a largo plazo: el déficit es consolidar, no adaptar |
| Estimador de confianza «decide» | decide la **validación empírica**; la sorpresa/«yo» sólo informa | resultados_yo: el yo calibra pero no decide; observación > yo |
| Encoder independiente del idioma / bytes en el núcleo | aplazado a fase 5 y fuera del criterio principal | sin decoder ni dataset multilingüe nuevo; mezclarlo confunde la atribución |
| Horizontes 16–64 «reducir colapso» | h64 secundaria | ÓRBITA colapsa a 64 por diseño del núcleo congelado; Nova no lo toca |
| Comparar sólo ÓRBITA vs NOVA | + MLP de igual presupuesto, SHUFFLED (biblioteca con etiquetas/firmas permutadas), LAT_EP, ORACLE_FAMNET (techo) | lecciones: sin controles el resultado no es atribuible |
| Aprendizaje en línea continuo | consolidación por lotes («sueño») | evita deriva y permite test de no-daño antes de admitir |
