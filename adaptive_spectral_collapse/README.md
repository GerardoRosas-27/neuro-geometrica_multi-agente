# `adaptive_spectral_collapse` (Rama aislada `exp/adaptive-spectral-collapse`)

Cuarto experimento aislado de la serie **Wave Inference → Collapse → Adaptive State Learning** (Rust puro con **0 dependencias externas** + orquestación, bootstrap pareado y SVG en Python de biblioteca estándar).

Responde directamente a la evaluación crítica de las tres ramas previas:
1. **Entrada desconocida:** $K_{\text{base}} = 24 \to K_{\text{tot}} = 28$ conceptos arbitrarios opacos en $\mathbb{R}^{16}$ (sobrecompletos, sin primos ni ceros de Riemann).
2. **Aprendizaje autónomo del campo:** `adaptive_spectral` parte de frecuencias aleatorias y aprende conjuntamente su codificador de fase anti-colisiones y sus modos espectrales 1D+3D solo desde la experiencia (`TRAIN_PAIRS`, 40% de los pares).
3. **Generalización estricta OOD y comparación justa:** evaluación sobre **166 pares nunca vistos (60%)** + **4 conceptos completamente nuevos** (`few-shot`), comparando contra **Riemann fijo, Fourier, Dyson GUE, Poisson, QMC Halton y un MLP convencional** de capacidad (`1,118` vs. `962` params) y cómputo equivalentes.

- **Plan preregistrado:** [`docs/adaptive_spectral_collapse_v1/PLAN.md`](../docs/adaptive_spectral_collapse_v1/PLAN.md)
- **Resultados CONFIRM:** [`docs/adaptive_spectral_collapse_v1/RESULTADOS.md`](../docs/adaptive_spectral_collapse_v1/RESULTADOS.md)
- **Auditoría:** [`docs/adaptive_spectral_collapse_v1/AUDIT.md`](../docs/adaptive_spectral_collapse_v1/AUDIT.md)
