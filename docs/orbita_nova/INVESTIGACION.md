# ÓRBITA-Nova — Fundamentos y riesgos

Referencias con identificador; son trabajos publicados conocidos. **Pendiente**: revisar cada cita contra la fuente antes de citarla en un informe externo.

## Fundamentos
| idea en Nova | base | referencia |
|---|---|---|
| Núcleo lineal en espacio levantado (C2) | operador de Koopman aprendido con autoencoder | Lusch, Kutz, Brunton (2018) *Deep learning for universal linear embeddings of nonlinear dynamics*, Nature Commun. 9:4950, arXiv:1712.09707 |
| Identificación de operadores con pocos datos | DMD / SINDy | Schmid (2010) J. Fluid Mech. 656; Brunton, Proctor, Kutz (2016) PNAS 113(15), arXiv:1509.03580 |
| Corrector Brenier/ICNN | redes convexas en la entrada; mapas de transporte óptimo | Amos, Xu, Kolter (2017) ICML, arXiv:1609.07152; Makkuva et al. (2020) ICML, arXiv:1908.10962 |
| Filtro de innovación (C3) | filtro de Kalman / EKF; test de consistencia de innovación (NIS) | Kalman (1960) J. Basic Eng. 82; Bar-Shalom, Li, Kirubarajan (2001) *Estimation with Applications to Tracking and Navigation*, Wiley |
| Error de predicción como señal de aprendizaje | codificación predictiva | Rao & Ballard (1999) Nat. Neurosci. 2:79–87; Friston (2010) Nat. Rev. Neurosci. 11:127–138 |
| Adaptador de bajo rango (C4) | LoRA | Hu et al. (2021) arXiv:2106.09685 |
| Pesos rápidos (escala de episodio) | fast weights | Ba, Hinton et al. (2016) NeurIPS, arXiv:1610.06258 |
| Consolidación lenta (C6), memoria corto/largo plazo | sistemas complementarios de aprendizaje (hipocampo/neocórtex), replay | McClelland, McNaughton, O'Reilly (1995) Psychol. Rev. 102(3); Kumaran, Hassabis, McClelland (2016) Trends Cogn. Sci. 20(7) |
| No-olvido | consolidación elástica | Kirkpatrick et al. (2017) PNAS 114(13), arXiv:1612.00796 |
| Memoria de trabajo limitada | modelo multicomponente | Baddeley & Hitch (1974); Baddeley (2000) Trends Cogn. Sci. 4(11) |
| Encoder de conceptos independiente del idioma (fase 5) | Large Concept Models (mencionado en la conversación) | LCM team, Meta (2024) arXiv:2412.08821 |

## Riesgos
1. **Redundancia con LAT**: C4 puede ser LAT con otra parametrización → efecto nulo. Mitigación: brazo LAT_EP obligatorio; el aporte debe venir de C3/C5/C6 (ablaciones).
2. **Suelo en nuevas** (acc16 ≈ 0): métrica continua primaria; si ORACLE_FAMNET/ORACLE_EP también están en suelo en una familia, esa familia se excluye *antes* de DEV.
3. **Clustering por firma inestable** con k=2 (pocos datos): primario k=4; k=2 secundario.
4. **Envenenamiento de memoria** (lección CT): umbral + C7 + test de no-daño; reportar tasa de recuperación y pureza.
5. **Validación retenida intra-episodio** con k pequeño reduce datos de ajuste: k_fit/k_val fijado en prerregistro (k=4 → 3/1; k=8 → 6/2).
6. **Sobreinterpretación cognitiva**: las analogías memoria de trabajo/largo plazo son marco de diseño; ningún resultado se reportará como «cognición humana».
7. **Encoder Gemma** rebaja a ÓRBITA (0,273 vs 0,359 con contexto verdadero); se reporta con contexto verdadero y con Gemma.
8. **Cómputo**: GN con Jacobiano por diferencias finitas escala ~O(np²·nr); con r≤2 es barato, pero la consolidación con validación sobre conocidas puede dominar.
