# `riemann_collapse` — Colapso de Onda Pasada y Onda Futura hacia Primos de Riemann

Experimento científico aislado en rama independiente (`exp/riemann-wave-collapse`), en **Rust puro sin dependencias externas** + orquestación, bootstrap pareado y graficación SVG en Python con biblioteca estándar.

- **Plan preregistrado:** [`docs/riemann_collapse_v1/PLAN.md`](../docs/riemann_collapse_v1/PLAN.md)
- **Resultados CONFIRM y autocrítica honesta:** [`docs/riemann_collapse_v1/RESULTADOS.md`](../docs/riemann_collapse_v1/RESULTADOS.md)
- **Auditoría de integridad:** [`docs/riemann_collapse_v1/AUDIT.md`](../docs/riemann_collapse_v1/AUDIT.md)

## Reproducción rápida

```bash
cd riemann_collapse
cargo test && cargo test --release && cargo build --release
./target/release/riemann-collapse-v1 find-config 0xB800
python3 freeze_and_verify.py replay
python3 summarize.py
python3 render_plots.py
python3 freeze_and_verify.py audit
```
