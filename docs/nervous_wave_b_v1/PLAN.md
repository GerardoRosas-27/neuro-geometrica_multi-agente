# Prueba B — Topología Sistema Nervioso Escalada con Inhibición Lateral por Coherencia de Fase (Kuramoto) y Auto-Calibración Resonante (`exp/nervous-wave-coherence-b`)

## 1. Motivación directa desde la Prueba A (`exp/nervous-wave-topology`)

La Prueba A (`N=64, E=256, T=8`, `0/6 PASS` bajo criterios estrictos) estableció tres hechos empíricos claros y expuso tres cuellos de botella concretos:

1. **Confirmado en Prueba A:** propagación causal estricta (`E_hub(t=0)=0.0000`, `t_arr=2.00` vs `grid 3.17`, `locSNR=100.0%`), causalidad decisiva de la fase de onda (`+9.81 pp [+6.74, +13.03]` acc y `+11.94 pp [+9.58, +14.46]` abducción en `16/16` semillas frente a `no_delay`), resiliencia estructural superior ante lesiones axonales `15%/30%` y ruido de fase frente a rejilla local (`+3.24 pp` / `+3.17 pp` / `+3.39 pp`) y frente a grafo aleatorio bajo lesión severa `30%` (`+1.90 pp [+0.29, +3.53]`), más transferencia con sinapsis congeladas `A → C` superior al control de sinapsis permutadas (`+2.95 pp [+0.23, +5.82]`).
2. **Limitación 1 (Shunting escalar neutro ante ruido de fase):** en Prueba A, `nervous_system` vs `nervous_no_inhibition` dio `-0.04 pp` bajo ruido de fase $\sigma=0.50$, porque el ruido de fase unitario rota ángulos sin apenas alterar $|\psi|^2$.
3. **Limitación 2 (Capacidad estrecha a `E=256`):** la brecha frente a `all_to_all_flat` (`E=4,096`, `16×` sinapsis) fue `-1.69 pp` en Dominio A (`64.46%` vs `66.15%`) y `-9.57 pp` en Dominio C (`67.38%` vs `76.95%`).
4. **Limitación 3 (Transferencia sin resorte resonante ni dominio continuo `A → B`):** la transferencia `A → C` con sinapsis congeladas alcanzó `52.54%` con adaptación Adam plana y no evaluó transferencia continua `A → B`.

## 2. Arquitectura de la Prueba B (`nervous_wave_b`, crate 100% aislado)

### 2.1 Topología escalada (`N = 128` neuronas, `E = 1,024` sinapsis dirigidas, `T = 6` pasos)
Densidad $1024 / 128^2 = 6.25\%$ (`16×` más dispersa que una matriz densa `128×128`, y `4×` menos sinapsis que el control denso `flat_dense_4096` de `N=64, E=4,096`):

- **Receptores sensoriales bipolares (`32` nodos):** `0..16` (hemisferio Pasado $a$) y `16..32` (hemisferio Futuro $b$).
- **Ganglios resonantes modulares (`64` nodos = `8` ganglios × `8` neuronas):** `G0..G3` (`32..64`, Pasado) y `G4..G7` (`64..96`, Futuro).
- **Hubs asociativos convergentes de línea media (`32` nodos):** `96..128`.
- **Presupuesto exacto de `E = 1,024` sinapsis (`in_deg = 8` promedio):**
  - `256` aferentes sensoriales (`16` Pasado → `G0..G3`, `8` aristas/receptor; `16` Futuro → `G4..G7`, `8` aristas/receptor).
  - `384` sinapsis intraganglionares resonantes (`8` ganglios × `8` neuronas × `6` desplazamientos `{1,2,3,5,6,7}` sin romper ningún ciclo).
  - `128` tractos comisurales mielinizados de largo alcance (`16` pares interhemisféricos `{G0..G3} × {G4..G7}` × `4` ida + `4` vuelta).
  - `256` proyecciones convergentes a los `32` hubs (`8` aristas por hub, `1` desde cada ganglio `G0..G7`).
- **Lectura (readout) igualada a `64` dimensiones en los 6 brazos:** promedio temporal en $t \in \{4, 5, 6\}$ sobre los `32` hubs de $[\operatorname{Re}(\psi_h), |\psi_h|^2 - 0.5]$.

### 2.2 Inhibición lateral selectiva por coherencia de fase (Dispersión incoherente de Kuramoto)
Para cada neurona $i$ y paso físico $t \in \{0,\dots,T-1\}$ con factor de escala sináptica $\eta_{\text{syn}} = 1/\sqrt{\overline{d}_{\text{in}}}$ (`1/sqrt(8)` en brazos dispersos y `1/sqrt(64)` en `flat_dense_4096` para invarianza de escala 100% justa):

$$I_i(t) = \sum_{e=(j\to i)} g_e e^{\mathrm{i}\theta_e}\psi_j(t), \qquad S_i(t) = \sum_{e=(j\to i)} |g_e|\sqrt{|\psi_j(t)|^2 + \epsilon_s}, \qquad g_e = \tanh(\alpha_e),\; \epsilon_s = 10^{-8}$$

$$D_i^{\text{incoh}}(t) = \eta_{\text{syn}}^2 \max\!\big(0,\; S_i(t)^2 - |I_i(t)|^2\big) \ge 0$$

$$s_i(t) = \gamma_i e^{\mathrm{i}(\omega_i + \Delta\omega_i^{\text{myelin}})}\psi_i(t) + \eta_{\text{syn}} I_i(t) + u_i(t)$$

$$\psi_i(t+1) = \frac{s_i(t)}{\sqrt{1 + \beta_{\text{sat}}|s_i(t)|^2 + \lambda_{\text{coh}}\,\sigma(\tau_i)\,D_i^{\text{incoh}}(t)}}$$

con $\beta_{\text{sat}} = 0.25$, $\lambda_{\text{coh}} = 0.45$, $\tau_i$ entrenable por neurona (inicializado en `0.0`), y gradiente analítico BPTT exacto verificado contra diferencias finitas (`< 1e-5`). Por desigualdad triangular, $D_i^{\text{incoh}} = 0$ si y sólo si todas las ondas entrantes arriban en fase coherente; cuando el ruido de fase desincroniza las llegadas, $D_i^{\text{incoh}}$ crece y atenúa selectivamente la señal incoherente.

### 2.3 Los 6 brazos experimentales (comparación 100% justa)
1. `nervous_coherence_b` (`N=128, E=1,024`, topología nerviosa completa + compuerta de coherencia Kuramoto).
2. `nervous_no_coherence_gate` (`N=128, E=1,024`, misma topología nerviosa con $\lambda_{\text{coh}} = 0$ — ablación pura de coherencia).
3. `nervous_no_delay` (`N=128, E=1,024`, misma topología nerviosa con $\theta_e = 0, \omega_i = 0, \lambda_{\text{coh}} = 0$ — ablación de fase).
4. `grid_local_128` (`N=128, E=1,024`, rejilla local Moore `8×16`, `0` axones largos, CON idéntica compuerta de coherencia Kuramoto).
5. `random_er_128` (`N=128, E=1,024`, grafo dirigido aleatorio con grado entrante pareado, CON idéntica compuerta de coherencia Kuramoto).
6. `flat_dense_4096` (`N=64, E=4,096`, `4×` MÁS sinapsis, CON idéntica compuerta de coherencia Kuramoto e idéntico readout `64`-dim).

## 3. Suites de evaluación (4 suites por semilla)

- **Suite `B1` (Física de propagación y coherencia Kuramoto):** `E_hub(t=0)`, `E_hub(t=1)`, tiempo medio de arribo `t_arr`, energía final `E_fin`, aislamiento espacial `localization_snr`, coincidencia pasado-futuro `coincidence`, y parámetro de orden de coherencia Kuramoto medio en hubs $R_{\text{coh}}^{\text{hub}} = \langle |I_h| / (S_h + 10^{-6}) \rangle$.
- **Suite `B2` (Inferencia en Dominio A $\mathbb{R}^{20}$ y Dominio C $\mathbb{Z}_5 \times \{0..3\}$):** `acc`, `rmse`, `abduction_top1`, `two_hop_acc` en Dominio A (`220` train / `332` test) y `acc`, `consistency` en Dominio C (`120` train / `260` test).
- **Suite `B3` (Resiliencia sin reentrenamiento en Dominio A):** lesiones axonales aleatorias `15%` y `30%` (`6` máscaras por semilla) y ruido de fase $\sigma = 0.50$ en sinapsis (`6` realizaciones por semilla).
- **Suite `B4` (Transferencia con sinapsis congeladas bit-a-bit + Auto-calibración resonante por resorte):**
  - Congela al 100% todas las sinapsis $(\alpha_e, \theta_e)$ y parámetros base $(\omega_i, \gamma_i, \tau_i)$ (checksum IEEE-754 verificado antes y después).
  - Calibra únicamente `128` perillas de mielina nodal (`myelin_omega`) + `32` perillas sensoriales (`sensory_scale`, `sensory_phase`) con dinámica de resorte armónico amortiguado en dos fases (`warmup` de readout + calibración resonante conjunta), aplicada simétricamente a los 5 brazos de transferencia (`nervous_transfer`, `grid_transfer`, `random_transfer`, `flat_transfer`, `ctrl_permuted_synapses`).
  - Evalúa dos dominios independientes nunca vistos durante el entrenamiento base:
    - **`A → B`**: osciladores acoplados continuos $\mathbb{R}^{12}$ (`N_train = 48`).
    - **`A → C`**: grafo relacional discreto $\mathbb{Z}_5 \times \{0..3\}$ (`N_train = 60`).

## 4. Hipótesis prerregistradas (`H_B1..H_B6`) — Criterio ciego `CONFIRM` (`N = 16` semillas `0xB200..0xB20F`)

Todos los intervalos de confianza al 95% se calculan mediante bootstrap pareado (`10,000` remuestreos, `seed = 12345`).

- **`H_B1` (Causalidad física, aislamiento y coherencia):** en Suite `B1`, `nervous_coherence_b` cumple causalidad exacta (`E_hub(t=0) == 0.0` y `E_hub(t=1) == 0.0`), arribo más rápido que `grid_local_128` (`mean t_arr <= 2.5` vs `grid >= 3.0`), `localization_snr >= 0.85`, coherencia Kuramoto `R_coh_hub > 0.25`, y coincidencia pasado-futuro estrictamente superior a `grid_local_128` (`CI_lo > 0`).
- **`H_B2` (Causalidad de fase y paridad dispersa `4×`):** en Suite `B2` (Dominio A), `nervous_coherence_b` supera a `nervous_no_delay` en `>= +5.0 pp` en `acc` (`CI_lo > 0`) y `>= +6.0 pp` en `abduction` (`CI_lo > 0`), y alcanza al menos `97%` de la precisión de `flat_dense_4096` (`diff >= -2.0 pp`) usando `4×` menos sinapsis.
- **`H_B3` (Ventaja causal de la inhibición por coherencia ante ruido de fase):** en Suite `B3` (ruido de fase $\sigma = 0.50$, cero reentrenamiento), `nervous_coherence_b` supera a `nervous_no_coherence_gate` (`CI_lo > 0`) Y supera a `grid_local_128` en `>= +2.0 pp` (`CI_lo > 0`).
- **`H_B4` (Resiliencia estructural ante lesiones axonales):** en Suite `B3` (cero reentrenamiento), `nervous_coherence_b` supera a `grid_local_128` tanto en lesiones `15%` como en `30%` (`CI_lo > 0` en ambas) Y supera a `random_er_128` en lesiones `30%` (`CI_lo > 0`).
- **`H_B5` (Transferencia resonante con sinapsis congeladas `A → C`, `N=60`):** en Suite `B4` (`A → C`, checksum bit-a-bit verificado), `nervous_transfer` supera a `ctrl_permuted_synapses` (`CI_lo > 0`) Y supera la línea base de la Prueba A alcanzando `mean acc >= 55.0%`.
- **`H_B6` (Transferencia resonante con sinapsis congeladas `A → B`, `N=48`):** en Suite `B4` (`A → B`, checksum bit-a-bit verificado), `nervous_transfer` supera a `ctrl_permuted_synapses` (`CI_lo > 0`) Y supera a `grid_transfer` (`CI_lo > 0`).

## 5. Protocolo de integridad y aislamiento

1. Rama aislada `exp/nervous-wave-coherence-b` desde `main@120e1e8`; cero cambios en `main`, `orbita/`, `src/`, `aura-engine/` ni `Cargo.toml` raíz.
2. `PLAN.md` se commitea **antes** de escribir el código del experimento.
3. Desarrollo y ajuste únicamente en `DEV` (`16` semillas `0xB100..0xB10F`).
4. Congelamiento SHA-256 en `FROZEN.json` y commit previo a `CONFIRM`. Prohibido modificar código, scripts o `PLAN.md` tras el congelamiento.
5. Ejecución ciega `CONFIRM` (`16` semillas `0xB200..0xB20F`), replay bit-a-bit (`0xB200`), resumen estadístico, `4` figuras SVG, reporte honesto `RESULTADOS.md` (incluyendo tabla comparativa directa Prueba A vs Prueba B) y auditoría automática `AUDIT.md` (`AUDIT PASS 8/8`).
