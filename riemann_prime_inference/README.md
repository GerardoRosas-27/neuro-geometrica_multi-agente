# `riemann_prime_inference` (Rama aislada `exp/riemann-prime-inference`)

Crate autónomo en Rust puro (**0 dependencias externas**) + orquestación/análisis/SVG en Python de biblioteca estándar para evaluar la **inferencia retrodictiva y bidireccional mediante colapso de ondas Pasado × Futuro en el patrón de los primos de Riemann** frente a **colapso aleatorio**, **Dyson GUE sin primos** y **Fourier lineal aditivo**.

- **Plan preregistrado:** [`docs/riemann_prime_inference_v1/PLAN.md`](../docs/riemann_prime_inference_v1/PLAN.md)
- **Resultados CONFIRM:** [`docs/riemann_prime_inference_v1/RESULTADOS.md`](../docs/riemann_prime_inference_v1/RESULTADOS.md)
- **Auditoría:** [`docs/riemann_prime_inference_v1/AUDIT.md`](../docs/riemann_prime_inference_v1/AUDIT.md)

## Ejecución reproducible

```bash
cd riemann_prime_inference
cargo test
cargo test --release
cargo build --release
python3 run.py DEV --workers 2 --skip-build
python3 freeze_and_verify.py freeze
python3 run.py CONFIRM --workers 2 --skip-build
python3 freeze_and_verify.py replay
python3 summarize.py
python3 render_plots.py
python3 freeze_and_verify.py audit
```
