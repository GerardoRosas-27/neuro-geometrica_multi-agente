# ÓRBITA-Nova — Resultados fase 1 (techo / suelo)

**Fecha**: 2026-10-22 (America/Mexico_City, UTC−6)  
**Rama**: `exp/orbita-nova`  
**Veredicto**: techo medido; **incluir 8, 10, 11** en preregistro DEV primario; **excluir 9 (espiral)** del primario (suelo blando a k=4).

## Protocolo

| ítem | valor |
|---|---|
| Distribución | episodios **generados** fam 8–11 (`fam::rule`; no H / no Gemma / no bytes) |
| Seeds | DEV `0xA400`–`0xA40F` (pesos congelados `artifacts/orbita/dev/weights_*.json`) |
| n / k / rank / λ_C4 | 16 por familia / {2,4,8} / 2 / 1e-6 |
| ORACLE_EP | `Module(None, deg=2, λ=1e-4)` residual Φ, ajuste **sólo** k pasos del episodio |
| C4_EP | ΔK=UVᵀ rango 2, GN/LM por episodio (techo del adaptador Nova; sin C5/C7) |
| ORACLE_FAMNET | mismo `Module`, **un** módulo por familia verdadera (etiqueta privilegiada), todos los episodios de esa fam |
| ÓRBITA | reid g∈0..7 con p̂=p verdadero → free-run h16 desde x0 |
| Métrica | mediana `log(rel_err h16)`; acc16 (umbral 0,05) |
| Train | **no**; sin bytes; núcleo sha verificado |

Nombres: **8** giro ondulado, **9** espiral, **10** pulso radial, **11** cizalla saturada.

### Criterios de suelo (documentados)

1. **Suelo duro** (por seed, k=4): `best_acc16 ≤ 0,02` **y** `best_delta_log ≥ −0,1` (ningún oráculo mejora el continuo). Excluir familia si ≥8/16 seeds en suelo duro **y** `best_acc16_mean ≤ 0,02`.
2. **Suelo blando**: a k=4, `best_acc16_mean < 0,05` y `best_delta_log_mean ≥ 0`, **y** k=8 no rescata (`best_acc < 0,05` y `bestΔlog > −0,2`) → excluir del **primario** DEV (secundaria opcional).

`best_*` = mejor entre ORACLE_EP, C4_EP y ORACLE_FAMNET.

## Resumen k=4 (primario; media 16 seeds)

| fam | nombre | ÓRBITA acc16 | OEP acc16 | C4 acc16 | FAMNET acc16 | best acc16 | best Δlog | floor seeds | decisión |
|---|---|---|---|---|---|---|---|---|---|
| 8 | giro ondulado | 0,016 | 0,066 | 0,047 | 0,000 | **0,070** | **−0,286** | 4/16 | **incluir** |
| 9 | espiral | 0,000 | 0,031 | 0,023 | 0,000 | 0,031 | +0,044 | 7/16 | **excluir primario** |
| 10 | pulso radial | 0,074 | **0,129** | 0,086 | 0,000 | **0,137** | +0,428† | 2/16 | **incluir** |
| 11 | cizalla saturada | 0,000 | 0,035 | 0,004 | 0,000 | 0,035 | +0,008 | 6/16 | **incluir** (techo a k=8) |

† Fam 10: el continuo mediano de ÓRBITA es ya bajo (med log ≈ −1,65); los oráculos suben **acc16** pero no el mediano log a k=4. A k=8, OEP Δlog ≈ −0,32 y acc16 ≈ 0,28 → techo real.

### k=2 y k=8 (best acc16 / best Δlog, media seeds)

| fam | k=2 acc | k=2 Δlog | k=8 acc | k=8 Δlog |
|---|---|---|---|---|
| 8 | 0,059 | −0,283 | 0,098 | −0,234 |
| 9 | 0,027 | +0,000 | 0,047 | −0,119 |
| 10 | 0,117 | +0,524 | **0,277** | **−0,319** |
| 11 | 0,023 | +0,206 | 0,047 | **−0,632** |

## Lectura

1. **Ninguna familia está en suelo duro** → no hay exclusión automática por la puerta estricta.
2. **Fam 9 (espiral)** es la única en **suelo blando** a k=4: oráculos no mejoran el continuo y acc16 ≈ 0,03. Recomendación: **fuera del primario** N1–N6 (evitar inconclusas por suelo, lección CT).
3. **Fam 8**: C4 es el techo útil (Δc4 ≈ −0,21 en log); coherente con fase 0.
4. **Fam 10**: mayor techo acc16 (OEP); a k=8 el continuo también mejora.
5. **Fam 11**: débil a k=4; a k=8 el continuo cae fuerte (bestΔ ≈ −0,63) → hay techo si hay más observaciones; mantener en primario con k∈{2,4,8} secundarios.
6. **ORACLE_FAMNET (Module Φ) ≈ 0** en todas: el techo de consolidación ridge-Φ con etiqueta perfecta **no** basta. El techo operativo para Nova es **ORACLE_EP / C4_EP**. Un FAMNET tipo MLP (GROW) no se reimplementó aquí (sin train largo); si el preregistro exige ese brazo, cablearlo en fase 2/3 con presupuesto acotado.

## k_fit / k_val (fijar en preregistro)

| k | k_fit / k_val |
|---|---|
| 2 | 1 / 1 |
| 4 | **3 / 1** (primario) |
| 8 | 6 / 2 |

## Recomendación para preregistro DEV

- **Familias nuevas primarias**: **{8, 10, 11}**
- **Excluir del primario**: **{9}** (reportar en secundaria / sensibilidad si se desea)
- **k primario**: 4; secundarios 2 y 8
- **Brazos techo en DEV**: ORACLE_EP (Module) + C4_EP; ORACLE_FAMNET Module como control de consolidación (expectativa baja) o sustituir por MLP privilegiado si se quiere empatar GROW
- **Siguiente**: fase 2 `PREREG.md` congelado **antes** de generar datos DEV

## Cómo reproducir

```bash
cargo build --release --manifest-path orbita/Cargo.toml
BIN=orbita/target/release/orbita NPF=16 \
  bash orbita/scripts/nova_phase1.sh
# un seed:
orbita/target/release/orbita nova-eval --ceiling \
  --weights artifacts/orbita/dev/weights_0xA400.json --seed 0xA400 \
  --n-per-fam 16 --rank 2 --lam 1e-6
```

Artefactos: `artifacts/orbita/nova_fase1/` (`summary.csv`, `FASE1_SUMMARY.json`, `fase1_0x*.json`).

## Fuera de fase 1

Bytes/idiomas, train/SGD, `run_v6`, tocar `main`, preregistro congelado, flujo 2400 episodios.
