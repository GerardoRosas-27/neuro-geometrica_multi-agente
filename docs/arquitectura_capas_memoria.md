# Arquitectura por capas con memoria de largo plazo: LLM → encoder → ÓRBITA ⇄ CDT-RQM-epr (sueño) → campo → LLM

Rama `exp/orbita-llm` (tras 02ce4ed). **Sólo investigación y diseño**: no se entrenó ni se ejecutó nada nuevo. Las cifras
propias que aparecen se citan de documentos ya publicados en esta rama; de la literatura sólo se resume lo que dicen las
fuentes enlazadas (verificadas por URL el 2026-10-17).

## 1. Revisión de literatura (fuentes verificadas)

| línea | trabajo | qué aporta (según la fuente) | URL |
|---|---|---|---|
| CLS | McClelland, McNaughton & O'Reilly (1995), *Psychological Review* 102(3):419–457 | dos sistemas: hipocampo (rápido, episódico) y neocorteza (lento, estructurado, aprendizaje intercalado) | doi:10.1037/0033-295x.102.3.419 (verificado vía la lista de referencias de Kumaran 2016; psycnet bloquea la descarga) |
| CLS actualizada | Kumaran, Hassabis & McClelland (2016), *Trends Cogn. Sci.* 20(7):512–534 | el replay puede ponderarse por recompensa/novedad; la neocorteza aprende rápido lo consistente con su estructura; relevancia para agentes artificiales | https://doi.org/10.1016/j.tics.2016.05.004 · https://pubmed.ncbi.nlm.nih.gov/27315762/ |
| wake-sleep | Hinton, Dayan, Frey & Neal (1995), *Science* 268:1158 | fase vigilia (reconocimiento → ajusta generativo) y fase sueño (generativo → ajusta reconocimiento) | https://www.science.org/doi/10.1126/science.7761831 |
| replay generativo | Shin, Lee, Kim & Kim (2017), NeurIPS, «Deep Generative Replay» | generador + solver: pseudo-datos de tareas pasadas intercalados con la tarea nueva, sin guardar datos crudos | https://arxiv.org/abs/1705.08690 |
| sueño en RNA | Tadros, Krishnan, Ramyaa & Bazhenov (2022), *Nat. Commun.*, SRC | fase offline con plasticidad hebbiana y entrada ruidosa; recupera tareas olvidadas; aumenta la esparsidad y **poda** conexiones excesivas | https://www.nature.com/articles/s41467-022-34938-7 |
| vigilia-sueño + memoria semiparamétrica | Liu et al. (2025), BrainCL, arXiv:2504.14727 | memoria de trabajo episódica en vigilia; en sueño se selecciona qué pasa a largo plazo y se re-entrena el modelo con replay interno | https://arxiv.org/abs/2504.14727 |
| memoria LLM (OS) | Packer et al. (2023), MemGPT, arXiv:2310.08560 | contexto virtual con paginación entre niveles de memoria | https://research.memgpt.ai/ |
| memoria episódica LLM | Das et al. (2024), Larimar, ICML 2024 | memoria episódica distribuida, actualizaciones one-shot y olvido selectivo sin re-entrenar | https://arxiv.org/abs/2403.11901 |
| memoria a largo plazo por sorpresa | Behrouz, Zhong & Mirrokni, Titans, arXiv:2501.00663 (NeurIPS 2025) | memoria neuronal que aprende en test; la **sorpresa** (gradiente de la pérdida asociativa) decide qué memorizar; decaimiento como olvido; atención = corto plazo | https://arxiv.org/abs/2501.00663 |
| recuperación | Borgeaud et al. (2021), RETRO, arXiv:2112.04426 | LM condicionado por fragmentos recuperados de una base de 2 billones de tokens | https://arxiv.org/abs/2112.04426 |
| segmentación por sorpresa | Fountas et al., EM-LLM, arXiv:2407.09450 (ICLR 2025) | límites de evento por sorpresa bayesiana (NLL del siguiente token) refinados por grafo; recuperación por similitud + contigüidad | https://doi.org/10.48550/arxiv.2407.09450 |
| replay por sorpresa en LLM | Hazard et al. (2025), SuRe, arXiv:2511.22367 | buffer con las secuencias de mayor NLL + adaptadores LoRA rápido/lento fusionados por EMA | https://arxiv.org/abs/2511.22367 |
| posición | Pink et al. (2025), arXiv:2502.06975 | memoria episódica externa + **consolidación periódica** en parámetros; «cuándo consolidar» es pregunta abierta (RQ5) | https://arxiv.org/abs/2502.06975 |
| world models | Zhou et al., DINO-WM, arXiv:2411.04983 | dinámica aprendida sobre features DINOv2 congeladas; planificación zero-shot | https://arxiv.org/abs/2411.04983 |
| JEPA + LLM | Assran et al. (2025), V-JEPA 2, arXiv:2506.09985 | predictor de representaciones de video; alineado con un LLM para QA de video; variante con acciones para planificación robótica | https://arxiv.org/abs/2506.09985 |

### Qué es conocido y qué sería nuevo (veredicto honesto)
- **Conocido:** separar memoria rápida y lenta (CLS), consolidar offline con replay generativo (Shin 2017; `pseudo_experiences`
  de v4 hace lo mismo), el sueño con poda/esparsificación (Tadros 2022), los ciclos vigilia-sueño en aprendizaje continual
  (BrainCL), la memoria a largo plazo para LLM (MemGPT, Larimar, RETRO, Titans), disparar la memorización/consolidación por
  sorpresa (Titans, EM-LLM, SuRe) y los world models sobre representaciones congeladas con salida a lenguaje (DINO-WM, V-JEPA 2).
  La arquitectura propuesta, en abstracto, **no es nueva**: es una instancia de CLS + replay generativo + sorpresa.
- **Posiblemente nuevo (no encontré un trabajo que lo combine; la búsqueda no es exhaustiva):** (i) que la memoria rápida
  sea un **modelo de dinámica geométrica** (ÓRBITA: núcleo Koopman + corrector) alimentado por un LLM vía encoder restringido
  (familia discreta + p numérico), (ii) que la memoria lenta sea **simbólica/interpretable** (CDT-RQM-epr: una regla ridge
  por familia, máscara *trusted*, tamaño medible en números), (iii) el disparo del sueño por **saturación** de ese modelo de
  dinámica (no por NLL de tokens) y (iv) la vuelta de la memoria consolidada a lenguaje vía el códec de fasores. Lo nuevo
  sería la combinación y el protocolo de control causal (NOCDT/SHUF/CORRUPT/oráculo-justo), no ninguna pieza aislada.
- **Advertencia empírica propia:** en esta rama CDT no aportó a la inferencia de ÓRBITA en ningún diseño probado
  (e328ff2, 02ce4ed): el aporte venía de reentrenar la cabeza o de **ver datos** (ORACLE_ORB ≡ OBS_DIR; ORACLE_LS acierta
  el 100 % a h16 en las familias iterables de TEST, que son afines). La arquitectura sólo tiene sentido si se prueba en un
  régimen donde la memoria **deba** aportar (§4).

## 2. «Saturación» de ÓRBITA: definición medible
Lo ya medido condiciona la definición:
- El residuo teacher-vs-ÓRBITA **no predice fallos** (H-C2b inconclusa en e2e: AUROC − 0,5 = +0,0080 [−0,0895, +0,1104];
  H-C1 invertida/inconclusa en D1/D2). → **No usar** el acuerdo teacher-ÓRBITA como señal de saturación.
- El teacher reconoce la familia sobre la predicción de ÓRBITA muy por encima del azar (H-C2a: +0,79/+0,81 sobre 1/8).
- Con observaciones reales, el residuo de ÓRBITA contra lo observado (score de ORACLE_ORB) sí separa fallos:
  AUROC 0,8649 [0,8141, 0,9163] en CONFIRM (02ce4ed). → la señal útil es **sorpresa contra la observación**.
- El encoder no satura por formato: parse_fail = 0 con dígitos, palabras, fracciones y unidades (925015f).

Definición propuesta (se calcula online, sin mirar TEST para ajustar):
1. **Sorpresa por episodio** S_e = (1/K) Σ_{t≤K} rel_err(ÓRBITA(y_{t−1}, ĉ), y_t) sobre las K primeras observaciones.
2. **Tasa de sorpresa** ρ_W = fracción de los últimos W episodios con S_e > τ_S (τ_S = cuantil 95 de S_e en DEV de vigilia).
3. **Novedad de familia** ν: el encoder da una familia que no está en la memoria, o ningún candidato (ÓRBITA ni teacher)
   explica las observaciones: min_g r_g > τ_S.
4. **Capacidad**: tamaño del buffer episódico de vigilia N_wake (número de episodios guardados) frente a un tope N_max.
**ÓRBITA está saturada** si ρ_W > ρ* **o** N_wake ≥ N_max **o** se acumulan ≥ m episodios con ν. Las constantes
(W, ρ*, N_max, m) se fijan en el prerregistro de cada experimento, no aquí.

## 3. Diseño por capas
```mermaid
flowchart LR
  U[Usuario: texto es/en/fr] --> G[Gemma 2 2B, capa 13 por token<br/>(congelado)]
  G -->|H: T×2304 f32| E[Encoder D2 (familia, argmax)<br/>+ numnorm (p1,p2)]
  E -->|ĉ ∈ R^10 = one-hot(8) ++ p| O[ÓRBITA-v1<br/>memoria de trabajo / inferencia rápida]
  X[Estado x0 y observaciones y1..yK] --> O
  O -->|trayectoria z1..zH| R[Respuesta numérica]
  O -->|episodio (x, ĉ, y_obs, S_e)| B[(Buffer episódico de vigilia<br/>CdtStore)]
  B -->|saturación: ρ_W, ν, N_wake| S{{SUEÑO}}
  S -->|collect_experience → consolidate(Full)<br/>family_fit_residual → trusted (poda)<br/>pseudo_experiences → re-consolidar<br/>borrar CDT| M[(CDT-RQM-epr: LearningSignal<br/>FamilyRule por familia, 48 números)]
  M -->|replay generativo trusted| O2[ÓRBITA reentrenada (opcional)<br/>replay intercalado]
  M -->|consulta: familia, rango p, residuo, trusted| F[Códec de campo<br/>LinguisticFieldCodec / FieldEncoder → fasores]
  F -->|banco de frases / plantilla| G2[Gemma como intérprete]
  G2 --> V[Respuesta verbal]
```

### Interfaces (qué pasa entre capas)
| de → a | estructura | notas |
|---|---|---|
| Gemma → encoder | `sequence_hidden` capa 13, T×2304 f32 (`gemma-embed --per-token --exit-after 12`) | existente |
| encoder → ÓRBITA | ĉ ∈ R¹⁰ (one-hot 8 + p1, p2 en [−1, 1]) | existente (e2e, `e2e.py run`) |
| ÓRBITA → buffer | `Example`/`Experience` {x, contexto, y observada, S_e} | `collect_experience` ya construye `CdtStore` desde `Example` |
| buffer → sueño | `CdtStore` | `sleep_consolidate` (field_v4), `family_fit_residual`, `TRUST_TAU`, `pseudo_experiences` (field_autonomy_v4) |
| sueño → memoria | `LearningSignal` {rules: FamilyRule{coef_x, coef_y, p_min/p_max, centro, radio}}, `n_numbers()` | existente; tamaño tras poda medido (384 → 240–288 números, 02ce4ed) |
| memoria → ÓRBITA | replay de confianza (`replay_sample` con máscara) para reentrenar ÓRBITA en sueño | mecanismo §16 de v4 (`train_trusted_on`) aplicado a ÓRBITA en lugar de Dφ: **nuevo, sin probar** |
| memoria → códec de campo | vector de rasgos por regla (familia, coeficientes normalizados, rango p, trusted) → `FieldEncoder::encode(features) -> Vec<Phasor>` | `FieldEncoder` y `LinguisticFieldCodec` existen en `origin/exp/encoder-linguistico` / `exp/encoder-campo` (a0eff2a), no en `exp/orbita-llm`, y dependen de `field_substrate` (no presente en esta rama) |
| campo → texto | `translate_field_to_text`: vecino más cercano en un **banco de frases** por coincidencia de fase | existente; el LLM **elige** una frase del banco, no genera libremente |

### Ciclo vigilia-sueño
1. **Vigilia:** texto → ĉ → ÓRBITA predice; cuando llegan observaciones se calcula S_e y el episodio entra al buffer.
   Si S_e es alto, la respuesta se marca de baja confianza (lo único que la evidencia actual permite afirmar: la sorpresa contra
   la observación separa fallos; ORACLE_ORB AUROC ≈ 0,86).
2. **Disparo:** criterio de saturación §2.
3. **Sueño:** `consolidate(Full)` del buffer → máscara trusted (poda de reglas con residuo ≥ τ) → borrado del buffer →
   `pseudo_experiences` desde la señal podada → re-consolidación; opcionalmente reentrenar ÓRBITA con replay intercalado
   (pseudo-episodios de familias viejas + nuevas, como Shin 2017 / §16).
4. **Respuesta verbal:** a una pregunta («¿qué transformación es esta?», «¿qué aprendiste?»), la memoria aporta una
   estructura cerrada {familia, rango de p, confianza trusted, tamaño} → códec de campo → frase del banco; el LLM sólo
   parafrasea la plantilla rellenada (no inventa números: los números salen de la estructura, como hace numnorm al leer).

### Riesgos
- **Olvido catastrófico de ÓRBITA** si se reentrena en sueño sólo con lo nuevo → replay intercalado obligatorio y métrica de
  retención en familias viejas (hay que medirlo).
- **Poda que borra conocimiento útil:** la poda por `TRUST_TAU` eliminó familias iterables en algunas seeds (02ce4ed:
  OBS_SLEEP peor que OBS_DIR) → registrar qué se poda y medir su efecto.
- **Alucinación del LLM:** con `translate_field_to_text` el LLM queda limitado a un banco cerrado; si se usa generación libre,
  exigir que cada número de la respuesta salga literal de la estructura (verificable con numnorm).
- **Señal de saturación espuria:** el residuo teacher-ÓRBITA no sirve (H-C2b); sólo sorpresa contra observación.
- **Memoria sin valor añadido:** si las familias son afines, un ajuste por mínimos cuadrados con 2 observaciones basta
  (ORACLE_LS 100 %): la memoria sólo puede aportar cuando la observación sola no alcanza (familias no afines, pocas
  observaciones, ruido, retención a largo plazo).
- **Integración del códec:** `field_substrate` no está en esta rama; traerlo es trabajo de ingeniería con riesgo de romper el
  aislamiento de v4.

## 4. Plan de experimentos falsables escalonados
Para todos: δ = 0,05; IC95 bootstrap pareado (16 seeds DEV y CONFIRM, 10 000 remuestreos); prerregistro push antes de
lanzar; split por plantilla; 0 queries a TEST; controles NOCDT, SHUF, CORRUPT y **oráculo-justo** con la misma información
(ORACLE_ORB, ORACLE_LS y, para retención, «buffer crudo» = replay de episodios reales sin consolidar).

| fase | pregunta | diseño mínimo | criterio (CONFIRM y DEV) |
|---|---|---|---|
| E0 | ¿La saturación §2 detecta fallos y novedad? | ÓRBITA-v1 sobre familias vistas + familias nuevas introducidas a mitad del flujo | AUROC(S_e → fallo h16) − 0,5 IC95 > 0 y detección de familia nueva antes de N episodios; control: residuo teacher-ÓRBITA |
| E1 | Tarea continual donde la memoria **deba** aportar | flujo de tareas: familias A (vistas) → B (nuevas **no afines**, p.ej. deformación radial/compuestas, K=1 observación y ruido) → vuelta a A | métrica de retención: acc16 en A tras B; acc16 en B |
| E2 | ¿Consolidar + sueño mejora retención frente a no tener memoria? | SUEÑO (sleep_consolidate + poda + replay a ÓRBITA) vs NOCDT (fine-tune secuencial) vs SHUF/CORRUPT vs buffer crudo (oráculo-justo de capacidad) | «memoria aporta» si SUEÑO − max(NOCDT, SHUF, CORRUPT, buffer crudo con el **mismo** presupuesto de números) IC95 > 0 en acc16 retención |
| E3 | ¿Disparar por saturación es mejor que dormir a intervalos fijos? | sueño disparado por §2 vs sueño cada N episodios (mismo número total de sueños) | Δ acc16 promedio sobre el flujo IC95 > 0 |
| E4 | ¿La respuesta verbal es fiel a la memoria? | preguntas en es/en/fr sobre lo aprendido; respuesta vía códec de campo + banco de frases vs LLM libre con la estructura en el prompt | tasa de números/familias correctas (verificables con numnorm) y tasa de afirmaciones no soportadas; control: estructura barajada |
| E5 | ¿Poda útil o dañina? | sueño con poda trusted vs sin poda vs poda aleatoria del mismo tamaño | retención y tamaño (números) |

Orden: E0 → E1/E2 (núcleo de la hipótesis) → E3 → E5 → E4 (la salida verbal sólo tiene sentido si la memoria aporta).
Si E2 no supera al buffer crudo con el mismo presupuesto, la conclusión será que la consolidación no añade nada sobre guardar
episodios, y el resto del plan pierde su razón de ser.

## 5. Preguntas para el usuario
- ¿Qué familias nuevas no afines usamos en E1 (p.ej. combinaciones de las 8 actuales, deformaciones no lineales nuevas)?
  Requiere ampliar el generador **fuera** de v4 (en `orbita/`).
- ¿Se permite reentrenar ÓRBITA durante el sueño (rompe «ÓRBITA-v1 congelada»)?
- ¿Traemos `field_substrate`/`LinguisticFieldCodec` a `exp/orbita-llm` o el códec se reimplementa mínimo en `orbita/`?
