# ÓRBITA-Nova — Resultados fase 0 (factibilidad)

**Fecha**: 2026-10-22 (America/Mexico_City, UTC−6)  
**Rama**: `exp/orbita-nova`  
**Veredicto**: **PASS** — C4 supera a ÓRBITA en **16/16** seeds (umbral ≥14/16).  
NIS C3: cobertura media **0,933** (objetivo 0,95 ± 0,05) → calibrado.

## Protocolo (preregistro de fase 0, sin train)

| ítem | valor |
|---|---|
| Distribución | episodios **generados** fam 0–7 (contexto verdadero; no H / no Gemma / no bytes) |
| Seeds | DEV `0xA400`–`0xA40F` (pesos congelados `artifacts/orbita/dev/weights_*.json`) |
| n / k / rank / λ / σ_noise | 64 / 4 / 2 / 1e-6 / 0,01 |
| Métrica primaria | mediana por seed de `log(rel_err h16)` (menor = mejor) |
| Baseline | ÓRBITA free-run h16 |
| C4 | ΔK = UVᵀ (r=2) en latente M=8; GN/LM por episodio; init LoRA U~ε, V=0 |
| Nova | C3 (filtro) + C4 + C5 (val retenida) + C7 (compuerta); C6 stub |
| Train | **no**; `nova-eval --train` es no-op; configs en `orbita/configs/` |

Núcleo ÓRBITA: sha/contenido verificado igual antes/después (`assert` en `nova-eval`). 0 queries a TEST.

## Resumen

| brazo | wins vs ÓRBITA | mediana Δ log rel_err (media seeds) | acc16 media |
|---|---|---|---|
| ÓRBITA | — | 0 | 0,337 |
| C4 | **16/16** | **−1,32** | 0,637 |
| Nova (C3–C5+C7) | **16/16** | **−1,32** | 0,635 |

NIS cobertura media 0,933 (rango por seed ≈ 0,883–0,977).

## Tabla por seed

| seed | Δ_C4 | ÓRBITA med log | C4 med log | ÓRBITA acc16 | C4 acc16 | NIS cov | accept |
|---|---|---|---|---|---|---|---|
| 0xA400 | −1,710 | −2,578 | −4,287 | 0,391 | 0,656 | 0,938 | 1,00 |
| 0xA401 | −1,083 | −1,980 | −3,063 | 0,203 | 0,516 | 0,945 | 1,00 |
| 0xA402 | −1,046 | −2,535 | −3,580 | 0,297 | 0,578 | 0,953 | 0,97 |
| 0xA403 | −1,547 | −2,552 | −4,099 | 0,328 | 0,734 | 0,941 | 0,97 |
| 0xA404 | −1,496 | −2,184 | −3,680 | 0,188 | 0,609 | 0,961 | 0,95 |
| 0xA405 | −1,401 | −2,548 | −3,950 | 0,375 | 0,641 | 0,914 | 0,92 |
| 0xA406 | −1,755 | −2,825 | −4,580 | 0,438 | 0,734 | 0,914 | 0,97 |
| 0xA407 | −1,378 | −2,927 | −4,305 | 0,484 | 0,734 | 0,941 | 0,97 |
| 0xA408 | −1,081 | −2,691 | −3,772 | 0,312 | 0,609 | 0,883 | 0,95 |
| 0xA409 | −0,922 | −2,537 | −3,459 | 0,297 | 0,562 | 0,969 | 0,98 |
| 0xA40A | −1,594 | −2,607 | −4,202 | 0,438 | 0,641 | 0,902 | 0,94 |
| 0xA40B | −1,085 | −2,435 | −3,519 | 0,250 | 0,578 | 0,922 | 0,98 |
| 0xA40C | −0,763 | −2,487 | −3,250 | 0,281 | 0,562 | 0,898 | 0,97 |
| 0xA40D | −1,807 | −2,605 | −4,412 | 0,406 | 0,719 | 0,957 | 0,98 |
| 0xA40E | −1,005 | −2,525 | −3,529 | 0,328 | 0,594 | 0,977 | 1,00 |
| 0xA40F | −1,434 | −2,588 | −4,022 | 0,391 | 0,719 | 0,906 | 0,98 |

Artefactos: `artifacts/orbita/nova_fase0/` (`summary.csv`, `FASE0_SUMMARY.json`, `fase0_0x*.json`, `run.log`).

## Cómo reproducir

```bash
cargo build --release --manifest-path orbita/Cargo.toml
BIN=orbita/target/release/orbita N=64 K=4 RANK=2 LAM=1e-6 NOISE=0.01 \
  bash orbita/scripts/nova_phase0.sh
# un seed:
orbita/target/release/orbita nova-eval \
  --weights artifacts/orbita/dev/weights_0xA400.json --seed 0xA400 \
  --n 64 --k 4 --rank 2 --lam 1e-6 --noise 0.01
# stubs de train (no-op):
orbita/target/release/orbita nova-eval --prepare-train --out orbita/configs
orbita/target/release/orbita nova-eval --train --config orbita/configs/nova_train.toml
```

## Qué quedó listo para entrenar (sin activar)

- `orbita/configs/nova_train.toml` (`enabled = false`)
- `orbita/configs/TRAIN_HOOKS.md`
- CLI `nova-eval --prepare-train` / `--train` → `train_stub` no-op
- C6 `OperatorLib::admit_stub` (rechaza consolidación hasta cablear sueño)
- C7 `gate_nodamage` operativo

**Fuera de fase 0**: bytes/idiomas (fase 5), `run_v6`, reentrenar Gemma/núcleo, DEV/CONFIRM N1–N6.

## Lectura

1. El cuello de botella de init (U,V)=(0,0) con Jacobiano nulo se resolvió con init estilo LoRA (U~ε, V=0); sin eso C4 no se movía.
2. En distribución propia, C4 reduce claramente el error continuo h16 y sube acc16 (~0,34 → ~0,64).
3. C5/C7 aceptan el adaptador en ~95–100 % de episodios aquí (hay margen real de mejora vs base); en nuevas/ruido se espera más rechazo — a medir en fases siguientes.
4. Puerta de fase 0 cumplida → se puede pasar a fase 1 (techo ORACLE) cuando se decida; **aún no** hay preregistro congelado ni train.
