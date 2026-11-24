# Plan Preregistrado — Carga de Adaptive Spectral Collapse en ÓRBITA + Tamiz Espectral de Números Primos (`exp/orbita-adaptive-spectral`)

**Escrito y comprometido en git antes de implementar cualquier código de integración en `orbita/`.**

- **Rama aislada:** `exp/orbita-adaptive-spectral` (creada desde `main@120e1e8`; **sin fusionar** con `main` ni tocar `aura-engine`, `src/v4_*.rs` ni ninguna de las 4 ramas experimentales previas).
- **Núcleo ÓRBITA congelado:** se cargan los pesos publicados oficiales `artifacts/orbita/{dev,confirm}/weights_0x*.json` (`2,602` parámetros) y se verifica invariancia exacta (`d_after.raw == d.raw`) en cada semilla.
- **Trazabilidad explícita de las 5 ramas independientes:**
  1. `exp/riemann-wave-inference` (`363d1ed`) → `riemann_wave/`, `docs/riemann_wave_v1/`
  2. `exp/riemann-wave-collapse` (`b36cf96`) → `riemann_collapse/`, `docs/riemann_collapse_v1/`
  3. `exp/riemann-prime-inference` (`06cb22c`) → `riemann_prime_inference/`, `docs/riemann_prime_inference_v1/`
  4. `exp/adaptive-spectral-collapse` (`662c26c`) → `adaptive_spectral_collapse/`, `docs/adaptive_spectral_collapse_v1/`
  5. **`exp/orbita-adaptive-spectral`** *(este experimento)* → `orbita/src/adaptive_spectral.rs` (+ hook CLI en `orbita/src/main.rs`), `docs/orbita_adaptive_spectral_v1/`

---

## 1. Preguntas Científicas

1. **¿Mejora ÓRBITA al cargar nuestro mecanismo entrenado de `Adaptive Spectral Collapse` (0 primos, 0 ceros de Riemann)?**
   Se somete a prueba sobre **tres bancos reales de ÓRBITA**:
   - **Suite O1 (Protocolo Oficial §16 de ÓRBITA):** `TEST` sellado (`256` ejemplos en 8 familias, incluyendo `interp_hole` $p_1 \in (0.10, 0.45)$) y rollout free-run `E33` (`h = 1, 2, 4, 8, 16, 32, 64` en `rotate` + `reflect`, `gate E33`, auditor `leaked = 0`, `queries = 0`, proceso nuevo `fresh`). El adaptador se entrena **estrictamente sobre `s.ds.train`** (0 acceso a `dev`/`test`, 0 consultas al teacher CDT).
   - **Suite O2 (Abducción de Contexto Desconocido y Composición de Reglas en ÓRBITA):** recuperación de reglas desconocidas (`0..11`) y de **40 pares compuestos inéditos (60.6%)** desde firmas geométricas opacas $\mathbf{s}(\mathbf{x}_0, \mathbf{y}_1) \in \mathbb{R}^{16}$ mediante colapso ondulatorio Pasado × Futuro (0 tabla de pares, 0 primos, 0 $\zeta(s)$).
   - **Suite O3 (Aprendizaje Continuo *Few-Shot* de las 4 Familias Nuevas No Lineales `8..11`):** el problema abierto donde `ORBITA_BASE` (`0.0%`), `OBS` (`0.1%`), `GROW` (`2.25%`), `TFE` (`~2%`) y `LAT` (`4.4%`) fallaron en ÓRBITA (`8: wavy_rot`, `9: spiral`, `10: trefoil`, `11: coupled_tanh`), evaluando en *held-out* (`acc_h1`, `acc_h8`, `acc_h16`) sin olvidar las 8 familias conocidas `0..7`.
2. **Tamiz Espectral de Números Primos (*"¿qué inferencia se esconde si solo extraes las frecuencias exactas de los números primos?"*):**
   - **Suite O4:** una vez entrenado completamente `ORBITA_ASC` (sin haber visto jamás un primo ni un cero de Riemann), congelamos su campo de inferencia y extraemos **exclusivamente** las frecuencias exactas de números primos (`prime_harmonic_exact`: $p \in \{2,3,5,7,11,13,17,19,23,29\}$; `riemann_prime_dual`: ceros exactos $\{\gamma_1,\dots,\gamma_{10}\}$; `log_prime_exact`: $\{\ln p\}$), comparando contra controles de **idéntica cardinalidad ($M_s = 10$)** (`composite_only`, `even_only`, `random_matched`) y contra `full_adaptive` para medir con exactitud qué patrón geométrico, de simetría impar y de supresión de ecos fantasma (coprimalidad $\gcd(p_i, p_j) = 1$) sobrevive y qué se pierde "del otro lado" de la inferencia.

---

## 2. Arquitectura Integrada y Paridad de Brazos

### A. Adaptador de Campo Espectral Adaptativo sobre ÓRBITA (`ORBITA_ASC`)
Para cada punto 2D $\mathbf{q} = (x, y)$ bajo contexto $\mathbf{c} = (f, p_1, p_2)$ con vector de fase $\mathbf{u} = [x, y, p_1, p_2] \in \mathbb{R}^4$:
$$\hat{\mathbf{q}}'_{\text{ASC}}(\mathbf{q}, \mathbf{c}) = \text{Orbita}_{\text{frozen}}(\mathbf{q}, \mathbf{c}) + \mathbf{b}_f + \sum_{m=1}^{M} \mathbf{a}_{f,m}\cos\!\big(\mathbf{k}_{f,m}\cdot \mathbf{u} + \phi_{f,m}\big), \qquad M = 16$$
- Inicializado con $\mathbf{a}_{f,m} = \mathbf{0}, \mathbf{b}_f = \mathbf{0}$ (en paso 0, `ORBITA_ASC ≡ ORBITA_BASE` bit a bit) y $\mathbf{k}_{f,m}$ aleatorios uniformes (Poisson; **0 primos, 0 ceros de Riemann**).
- Entrenado con Adam sobre error relativo en `s.ds.train` + regularización de banda suave (para interpolar limpiamente a través de `interp_hole`) + repulsión espectral entre normas $\|\mathbf{k}_{f,m}\|$.
- **Parámetros (`O1`):** `2,602` (ÓRBITA congelada) + `8 × (16 × 7 + 2) = 912` → **`3,514` parámetros totales**.

### B. Brazos Comparados (Paridad Paramétrica 1:1)
1. **`ORBITA_BASE`** (`2,602` params): pesos oficiales publicados de ÓRBITA-v1 congelados.
2. **`ORBITA_ASC`** (`3,514` params en `O1`): ÓRBITA + Campo Espectral Adaptativo + Cavidad de Colapso 3D (0 primos, 0 $\zeta(s)$).
3. **`ORBITA_RIEMANN_FIXED`** (`3,514` params): idéntica arquitectura con frecuencias congeladas a los ceros exactos de Riemann $\gamma_m$.
4. **`ORBITA_FOURIER_FIXED`** (`3,514` params): idéntica arquitectura con armónicos regulares de Fourier congelados.
5. **`ORBITA_POISSON_FIXED`** (`3,514` params): idéntica arquitectura con frecuencias aleatorias uniformes (Poisson) congeladas.
6. **`ORBITA_MLP_RES`** (`3,514` params): ÓRBITA congelada + MLP residual convencional por familia (`4 -> 16 (tanh) -> 2`, `114` params/familia = `912` params → paridad exacta `3,514` params) entrenado con idéntico presupuesto Adam.
7. En **`O3`** (aprendizaje continuo de familias nuevas `8..11`) se incluyen además los comparadores nativos reales de ÓRBITA: **`ORBITA_OBS`** (`fam::reid`) y **`ORBITA_LAT`** (adaptador latente polinomial de `orbita/src/latent.rs`).

---

## 3. Partición `DEV` / `CONFIRM`, Freeze y Los 8 Criterios Preregistrados

- **`DEV`:** 16 semillas oficiales de ÓRBITA `0xA400..=0xA40F` (`artifacts/orbita/dev/weights_0xA4*.json`).
- **Congelamiento pre-CONFIRM:** `docs/orbita_adaptive_spectral_v1/FROZEN.json` (hashes SHA-256 de código, binario `orbita` y `PLAN.md`).
- **`CONFIRM`:** 16 semillas oficiales de ÓRBITA `0xB400..=0xB40F` (`artifacts/orbita/confirm/weights_0xB4*.json`).
- **Replay bit-exacto:** `0xB400` · **Bootstrap pareado:** 10,000 remuestreos (`seed = 12345`), IC 95%.

| ID | Suite | Hipótesis / Criterio Preregistrado | Regla de Aceptación (`CONFIRM`, $N = 16$) |
|---|---|---|---|
| **`H_O1a`** | O1 | Mejora en §16 1-paso sin regresión ni fugas (`TEST` 256) | `CI95_hi(rel1_asc - rel1_base) < -0.0005`, `CI95_lo(acc1_asc - acc1_base) > -0.005`, `16/16 CLEAN`, `0 queries` |
| **`H_O1b`** | O1 | Mejora real en horizonte largo E33 (`h16`) de ÓRBITA | `CI95_lo(h16_asc - h16_base) > 0.03` y gate E33 `16/16 PASS` |
| **`H_O1c`** | O1 | Ventaja del espectro adaptativo frente a espectros fijos en E33 `h16` | `CI95_lo(h16_asc - h16_fourier) > 0.02` y `CI95_lo(h16_asc - h16_poisson) > 0.02` |
| **`H_O2a`** | O2 | Abducción composicional de 40 pares inéditos de reglas ÓRBITA sin tabla de pares | `CI95_lo(acc_asc - acc_poisson) > 0.05`, `CI95_lo(acc_asc - acc_fourier) > 0.05` y `CI95_lo(acc_asc - acc_mlp) > 0.05` |
| **`H_O3a`** | O3 | Aprendizaje continuo de las 4 familias nuevas (`8..11`) superando a ÓRBITA-LAT sin olvidar conocidas (`0..7`) | `CI95_lo(new_h16_asc - new_h16_lat) > 0.10` y `CI95_lo(known_h16_asc - known_h16_base) > -0.02` |
| **`H_O3b`** | O3 | Ventaja del espectro adaptativo frente a espectros fijos en familias nuevas (`8..11`, `acc_h16`) | `CI95_lo(new_h16_asc - new_h16_fourier) > 0.03` y `CI95_lo(new_h16_asc - new_h16_poisson) > 0.03` |
| **`H_O4a`** | O4b | **Tamiz primo en colapso Pasado × Futuro:** las frecuencias exactas primas suprimen ecos fantasma (coprimalidad $\gcd(p_i,p_j)=1$) frente a frecuencias compuestas de igual tamaño ($M_s=10$) | `CI95_lo(psr_prime - psr_comp) > 0.05` y `CI95_lo(acc_prime - acc_comp) > 0.03` |
| **`H_O4b`** | O4a | **Anatomía honesta del corte primo en dinámica ÓRBITA:** el patrón primo aísla una firma de paridad impar superior a `composite_only`/`even_only` (`CI95_lo(odd_prime - odd_comp) > 0.15`), y se cuantifica honestamente la pérdida frente al espectro completo `full_adaptive` (`CI95_hi(acc_prime - acc_full) < 0`) | Ambas desigualdades satisfechas en `CONFIRM` |

---

## 4. Aislamiento y Verificación

- Ningún archivo de `src/v4_*.rs`, `aura-engine/`, `orbita/src/model.rs` ni `main` será modificado.
- Verificación de invariancia bit a bit de los pesos congelados de ÓRBITA (`d_after.raw == d.raw`), verificación de `0` consultas a `TEST` (`QueryCounters::total() == 0`), proceso nuevo (`fresh_ok = true`), pruebas unitarias de gradientes analíticos vs. diferencias finitas (`< 1e-5`), y auditoría automatizada `AUDIT.md`.
