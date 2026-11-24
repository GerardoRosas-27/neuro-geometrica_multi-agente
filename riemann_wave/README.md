# `riemann_wave` — Frecuencia de Ondas de Riemann como Medio de Inferencia

Experimento científico aislado en rama independiente (`exp/riemann-wave-inference`), implementado en **Rust puro sin dependencias externas** + orquestación estadística en Python con biblioteca estándar únicamente.

- **Propuesta archivada:** [`docs/riemann_wave_v1/PROPUESTA_GEMINI.md`](../docs/riemann_wave_v1/PROPUESTA_GEMINI.md)
- **Plan preregistrado:** [`docs/riemann_wave_v1/PLAN.md`](../docs/riemann_wave_v1/PLAN.md)
- **Resultados CONFIRM y autocrítica honesta:** [`docs/riemann_wave_v1/RESULTADOS.md`](../docs/riemann_wave_v1/RESULTADOS.md)
- **Auditoría de integridad:** [`docs/riemann_wave_v1/AUDIT.md`](../docs/riemann_wave_v1/AUDIT.md)

## Cómo compilar y reproducir

```bash
cd riemann_wave

# 1. Pruebas unitarias (debug y release)
cargo test
cargo test --release

# 2. Compilar binario optimizado y verificar los primeros 128 ceros exactos de zeta(1/2 + i*gamma)
cargo build --release
./target/release/riemann-wave-v1 verify-zeros 128

# 3. Correr DEV (16 semillas 0xA700..=0xA70F), congelar SHA-256, correr CONFIRM (0xB700..=0xB70F)
python3 run.py DEV --workers 2
python3 freeze_and_verify.py freeze
python3 run.py CONFIRM --workers 2 --skip-build

# 4. Verificar replay bit-a-bit en 0xB700, resumir bootstrap pareado (10,000 remuestreos), graficar y auditar
python3 freeze_and_verify.py replay
python3 summarize.py
python3 render_plots.py
python3 freeze_and_verify.py audit
```
