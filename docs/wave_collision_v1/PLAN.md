# Plan Preregistrado — Red de Colisión de Ondas Auto-Calibrables (`768` Ondas, Cero Matrices Densas en Núcleo, `exp/wave-collision-autocal`)

- **Rama aislada:** `exp/wave-collision-autocal` (creada directamente desde `main@120e1e8`; **cero conexión** con ÓRBITA `orbita/`, CDT-RQM `src/`, `aura-engine/` ni `Cargo.toml` raíz).
- **Crate independiente:** `wave_collision/` · **Documentación y evidencia:** `docs/wave_collision_v1/`.
- **Salto conceptual respecto a `exp/spectral-transfer-discovery`:**
  1. **Incremento drástico de ondas (`16×`):** pasamos de `48` modos con una gran matriz densa `W_pair` (`1,536` pesos lineales) a **`768` ondas activas** (**`512` ondas primarias en `256` colisiones Pasado × Futuro + `256` ondas secundarias de interferencia onda-onda**) con **CERO matriz densa `W_pair` en el núcleo**.
  2. **Las frecuencias y fases SON los parámetros de la red:** el núcleo (`3,328` parámetros = `2,304` frecuencias + `1,024` fases/ángulos de mezcla) aprende directamente calibrando las frecuencias al chocar ondas incidentes del pasado contra ondas del futuro y luego haciendo chocar las ondas resultantes entre sí en una segunda etapa resonante.
  3. **Auto-calibración resonante en transferencia entre dominios (`A → B` y `A → C`):** al transferir a un dominio nuevo (*few-shot*), las **`2,816` frecuencias y fases primarias aprendidas en A permanecen `100%` congeladas** (verificado bit a bit con checksum IEEE-754) y se activan solo **`48` perillas de auto-calibración resonante** (`16` escalas de longitud de onda por banco + `16` desplazamientos de fase por banco + `16` ganancias resonantes de etapa 2), comparadas en paridad exacta `1:1` contra **`transfer_mlp_A` (+48 parámetros FiLM de escala/desplazamiento)** y también contra transferencia `100%` rígida (`0` perillas, `transfer_frozen_A`).

---

## 1. Arquitectura Física: `WaveCollisionCore` (`768` Ondas, `3,328` Parámetros = `100%` Frecuencias y Fases)

Para cualquier par de estados latentes \(\mathbf{u} = \mathbf{E}_{\text{dom}}(\mathbf{x}_a) \in \mathbb{R}^4\) y \(\mathbf{v} = \mathbf{E}_{\text{dom}}(\mathbf{x}_b) \in \mathbb{R}^4\):

### Etapa 1 — Colisión Primaria de `512` Ondas (`G = 16` bancos × `K = 16` pares colisionantes = `256` colisiones)
Cada colisión \(m = 16g + k \in \{0..255\}\) posee dos ondas con frecuencias auto-calibrables \(\boldsymbol{\omega}_m^{(1)}, \boldsymbol{\omega}_m^{(2)} \in \mathbb{R}^4\), fases \(\phi_m^{(1)}, \phi_m^{(2)} \in \mathbb{R}\) y ángulo de mezcla interferométrica \(\beta_m \in \mathbb{R}\):
- **Colisión conjugada + heterodina Pasado × Futuro** (con perillas de auto-calibración de cavidad \(s_g\) de escala de frecuencia y \(\delta_g\) de fase, fijadas a \(s_g = 1, \delta_g = 0\) en Dominio A y `transfer_frozen_A`):
  $$C_m(\mathbf{u}, \mathbf{v}) = \cos(\beta_m)\,\cos\!\bigl(s_g\,\boldsymbol{\omega}_m^{(1)}\cdot(\mathbf{v}-\mathbf{u}) + \phi_m^{(1)} + \delta_g\bigr) \;+\; \sin(\beta_m)\,\cos\!\bigl(s_g\,\boldsymbol{\omega}_m^{(1)}\cdot\mathbf{u} + \phi_m^{(1)}\bigr)\,\cos\!\bigl(s_g\,\boldsymbol{\omega}_m^{(2)}\cdot\mathbf{v} + \phi_m^{(2)} + \delta_g\bigr)$$
- **Campo estacionario del banco \(g \in \{0..15\}\):**
  $$h_g(\mathbf{u}, \mathbf{v}) = \frac{1}{\sqrt{16}} \sum_{k=0}^{15} C_{16g+k}(\mathbf{u}, \mathbf{v})$$

### Etapa 2 — Colisión Secundaria Onda-Onda (`256` Ondas Moduladas en Fase)
Las `16` ondas estacionarias \(h_0..h_{15}\) chocan entre sí mediante modulación interferométrica de fase con `256` frecuencias de acoplamiento \(\kappa_{g,j}\) y `256` fases \(\psi_{g,j}\):
$$z_g(\mathbf{u}, \mathbf{v}) = h_g(\mathbf{u}, \mathbf{v}) + \frac{\rho_g}{\sqrt{16}} \sum_{j=0}^{15} \cos\!\bigl(\kappa_{g,j}\, h_j(\mathbf{u}, \mathbf{v}) + \psi_{g,j}\bigr)$$
donde \(\rho_g = 1\) en Dominio A (y es una de las `16` ganancias resonantes auto-calibrables en `transfer_autocal_A`, sumando `16 + 16 + 16 = 48` perillas).

### Paridad de Parámetros y Justicia Experimental
- **`WaveCollisionCore`:** `1,024` (\(\boldsymbol{\omega}^{(1)}\)) + `1,024` (\(\boldsymbol{\omega}^{(2)}\)) + `768` (\(\phi^{(1)}, \phi^{(2)}, \beta\)) + `256` (\(\kappa\)) + `256` (\(\psi\)) = **`3,328` parámetros (`100%` frecuencias y fases, `0` matrices densas)**.
- **`MlpCore` (control neuronal con `+15%` más capacidad):** \(8 \xrightarrow{\tanh} 64 \xrightarrow{\tanh} 40 \xrightarrow{\text{lin}} 16\) = **`3,832` parámetros**.
- **Paridad exacta en transferencia auto-calibrable:**
  - Dominio B (`obs_dim=12`, `reg=3`, `cls=4`): `171` (encoder + head) + `48` (auto-calibración resonante en ondas **o** `48` parámetros FiLM de escala/desplazamiento en `MlpCore`: `16` escala + `16` bias en `z_rel` + `16` escala en entrada `h2`/proyección) = **`219` parámetros adaptables exactos en ambos**.
  - Dominio C (`obs_dim=16`, `reg=0`, `cls=4`): `136` (encoder + head) + `48` = **`184` parámetros adaptables exactos en ambos**.
- **Guardia anti-Riemann (test unitario `#[should_panic]`):** el modo `Adaptive` se inicializa con ruido puro y entra en `panic!` si se intenta inyectar cualquier cero de Riemann o frecuencia prima. Para el control `RiemannFixed`, se calculan los **256 primeros ceros reales de la función zeta de Riemann** \(\gamma_1..\gamma_{256}\) mediante la función \(Z(t)\) de Riemann-Siegel + refinamiento de secante verificado a `< 1e-9` contra las tablas de Odlyzko.

---

## 2. Los Tres Dominios Independientes y Suites Experimentales (`T1..T4`)

Mismos tres dominios independientes sin entidades ni dimensiones compartidas:
1. **Dominio A (Fuente, \(\mathbb{R}^{20}\)):** Ecosistema causal-relacional 4D-isotrópico (`24` entidades, `220` train / `332` inéditos, `120` consultas 2-saltos).
2. **Dominio B (Objetivo 1, \(\mathbb{R}^{12}\)):** Osciladores físicos no lineales acoplados continuos (`18` osciladores, trayectorias + cinemática; \(N_{\text{few}} \in \{24, 48, 96\}\), evaluado en `306 - N_few` pares inéditos).
3. **Dominio C (Objetivo 2, \(\mathbb{R}^{16}\)):** Grafo relacional discreto \(\mathbb{Z}_5 \times \{0..3\}\) (`20` nodos, `4` relaciones algebraicas; \(N_{\text{few}} \in \{30, 60, 120\}\), evaluado en `380 - N_few` arcos inéditos + consistencia algebraica + abducción *zero-shot*).

### Suites
- **Suite T1 (Dominio A):** `adaptive_wave` (`768` ondas, `0` Riemann) vs `riemann_fixed`, `fourier_fixed`, `poisson_fixed`, `dyson_fixed` (frecuencias \(\boldsymbol{\omega}^{(1)}, \boldsymbol{\omega}^{(2)}, \kappa\) congeladas, fases libres) y `mlp_conventional` (`3,832` params).
- **Suite T2 (Disección Causal en Dominio A):** `full_adaptive`, `ablate_phase_scramble`, `train_zero_phase`, `ablate_pairing_additive` (sin colisión multiplicativa), `ablate_pairing_permuted` (emparejamiento roto \(\pi(m)\neq m\)), `ablate_no_stage2` (sin colisión secundaria onda-onda), `ablate_freq_riemann`, `ablate_freq_fourier`, `ablate_freq_poisson`.
- **Suite T3 (Transferencia `A → B`, \(N_{\text{few}} \in \{24, 48, 96\}\)):** `transfer_autocal_A` (`+48` perillas resonantes, `2,816` ondas primarias + `512` secundarias `100%` congeladas), `transfer_frozen_A` (`0` perillas, `100%` congelado), `riemann_A`, `fourier_A`, `poisson_A`, `dyson_A`, `transfer_mlp_A` (`+48` perillas FiLM idénticas), `neg_ctrl_noise_A`, `neg_ctrl_phase_scramble`, `untrained_poisson`, `scratch_wave`, `scratch_mlp`.
- **Suite T4 (Transferencia `A → C`, \(N_{\text{few}} \in \{30, 60, 120\}\)):** mismos 12 brazos exactos.

---

## 3. Criterios Preregistrados Falsables (`H_W1..H_W8`)

Semillas deterministas: **16 `DEV` (`0xF100..0xF10F`)** para calibración; congelación obligatoria (`FROZEN.json` SHA-256) antes de **16 `CONFIRM` (`0xF200..0xF20F`)**. Bootstrap pareado `10,000` remuestreos (`seed = 12345`), IC 95%.

| ID | Suite | Regla Preregistrada Exacta (`CONFIRM`, *verbatim*) |
|---|---|---|
| **`H_W1`** | T1 | `CI95_lo(nmi_adapt) > 0.35`, `CI95_lo(acc_adapt - acc_riemann) > +0.04`, `CI95_lo(acc_adapt - acc_poisson) > +0.05`, `CI95_hi(rmse_adapt - rmse_riemann) < -0.05` y `CI95_lo(<r>_final - <r>_init) > +0.10` (`0` Riemann). |
| **`H_W2`** | T1 | `CI95_lo(abd_adapt - abd_mlp) > +0.15`, `CI95_hi(rmse_adapt - rmse_mlp) < -0.08`, `CI95_lo(acc_adapt - acc_mlp) > +0.04` y `CI95_lo(cfr_adapt) >= 0.90`. |
| **`H_W3`** | T2 | `CI95_lo(acc_full - acc_phase_scramble) > +0.20`, `CI95_lo(acc_full - acc_additive) > +0.20`, `CI95_lo(acc_full - acc_permuted) > +0.20` y `CI95_lo(acc_full - acc_no_stage2) > +0.02`. |
| **`H_W4`** | T3 (`N=48`) | `transfer_autocal_A` vence a `scratch_wave` (`CI95_lo(acc) > +0.04`, `CI95_hi(rmse) < -0.04`), a `neg_ctrl_noise_A` (`> +0.06`), a `neg_ctrl_phase_scramble` (`> +0.06`) y a `untrained_poisson` (`> +0.08`). |
| **`H_W5`** | T3 (`N=48`) | `transfer_autocal_A` vence a `transfer_mlp_A` (`CI95_lo(acc) > +0.06`, `CI95_hi(rmse) < -0.06`), a `riemann_A` (`> +0.03`) y a `poisson_A` (`> +0.04`). |
| **`H_W6`** | T4 (`N=60`) | `transfer_autocal_A` vence a `scratch_wave` (`CI95_lo(acc) > +0.03`), a `neg_ctrl_noise_A` (`> +0.06`), a `neg_ctrl_phase_scramble` (`> +0.06`) y a `untrained_poisson` (`> +0.05`). |
| **`H_W7`** | T4 (`N=60`) | `transfer_autocal_A` vence a `transfer_mlp_A` (`CI95_lo(acc) > +0.05`, `CI95_lo(cons) > +0.05`), a `riemann_A` (`> +0.03`) y a `untrained_poisson` en abducción (`> +0.10`). |
| **`H_W8`** | T3 & T4 (ultra) | En `N=24` (B) y `N=30` (C), `transfer_autocal_A` supera tanto a `scratch_wave` como a `scratch_mlp` con `CI95_lo > +0.05` en las 4 comparaciones pareadas. |
