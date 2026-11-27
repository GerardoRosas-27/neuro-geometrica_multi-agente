# Plan Preregistrado — Prueba A: Validación de la Topología Tipo "Sistema Nervioso" para Propagación de Ondas (`exp/nervous-wave-topology`)

- **Rama aislada:** `exp/nervous-wave-topology` (creada directamente desde `main@120e1e8`; **cero cambios** en `orbita/`, `src/`, `aura-engine/` ni `Cargo.toml` raíz).
- **Crate independiente:** `nervous_wave/` · **Evidencia y reportes:** `docs/nervous_wave_v1/`.
- **Pregunta científica central (Prueba A):** ¿Una topología neuro-anatómica dispersa (*sistema nervioso*: receptores sensoriales bipolares → ganglios modulares con inhibición lateral → tractos axonales largos mielinizados → nodos asociativos convergentes) produce propagación de ondas físicamente superior, mayor eficiencia sináptica (`16×` menos conexiones que un grafo plano denso), mayor robustez ante lesiones axonales/ruido y mejor inferencia relacional que retículas puramente locales, grafos aleatorios de igual presupuesto o acoplamiento global plano?

---

## 1. Dinámica Física: Ecuación de Onda Compleja Amortiguada sobre Grafo Dirigido

Todos los brazos topológicos ejecutan **exactamente la misma ecuación de propagación de ondas complejas** durante \(T = 8\) pasos temporales discretos sobre \(N = 64\) osciladores nodales \(\psi_i(t) \in \mathbb{C}\):

1. **Inyección sensorial bipolar en \(t = 0\) (vacío exacto en nodos internos):**
   - Receptores del Pasado \(i \in \{0..7\}\): \(\psi_i(0) = \exp\!\bigl(\mathrm{i}(\mathbf{w}_i^{\text{in}}\cdot \mathbf{u} + \phi_i^{\text{in}})\bigr)\).
   - Receptores del Futuro \(i \in \{8..15\}\): \(\psi_i(0) = \exp\!\bigl(\mathrm{i}(\mathbf{w}_i^{\text{in}}\cdot \mathbf{v} + \phi_i^{\text{in}})\bigr)\).
   - Interneuronas ganglionares (\(16..47\)) y hubs asociativos (\(48..63\)): \(\psi_i(0) = 0 + 0\,\mathrm{i}\).
2. **Propagación axonal con velocidad/retardo de fase, inhibición divisiva (*shunting*) y superposición compleja (\(t = 0..7\)):**
   $$I_i(t) = \sum_{e=(j\to i),\,\text{sign}_e>0} g_e\, e^{\mathrm{i}\theta_e}\, \psi_j(t), \qquad E_i^{\text{inh}}(t) = \sum_{e=(j\to i),\,\text{sign}_e<0} |g_e|\, |\psi_j(t)|^2$$
   donde \(g_e = \text{sign}_e\cdot \text{softplus}(\alpha_e)/\sqrt{\max(1, d_{\text{in}}(i))}\); las interneuronas inhibitorias (\(\text{sign}_e = -1\)) ejercen **inhibición lateral divisiva (*shunting* GABAérgico)** sobre la energía ganglionar local sin destruir la fase coherente, y \(\theta_e \in \mathbb{R}\) codifica el retardo de fase axonal (en `nervous_no_delay`, \(\theta_e \equiv 0\) y \(\omega_i \equiv 0\)).
3. **Actualización resonante somática con saturación preservadora de fase e inhibición divisiva (energía acotada):**
   $$s_i(t) = \gamma_i\, e^{\mathrm{i}\omega_i}\,\psi_i(t) + I_i(t) + \eta_i(t), \qquad \psi_i(t+1) = \frac{s_i(t)}{\sqrt{1 + \beta_{\text{sat}}\, |s_i(t)|^2 + \lambda_{\text{shunt}}\, E_i^{\text{inh}}(t)}}$$
   con \(\gamma_i = \sigma(\tilde{\gamma}_i)\in(0,1)\), \(\beta_{\text{sat}} = 0.50\), \(\lambda_{\text{shunt}} = 0.22\), e inyección sensorial sostenida suave \(\eta_i(t) = 0.35\,\psi_i(0)\) en receptores durante \(t \in \{0, 1\}\).
4. **Lectura interferométrica en los `16` hubs asociativos convergentes (\(h \in \{48..63\}\)):**
   Promediada en la ventana de colisión estacionaria \(t \in \{6, 7, 8\}\), cada hub produce 2 observables físicos —cuadratura real \(\operatorname{Re}(\psi_h)\) e intensidad interferométrica centrada \(|\psi_h|^2 - 0.5\) (que contiene el término cruzado de colisión \(2\operatorname{Re}(\psi_h^{\text{pasado}}\psi_h^{\text{futuro}*})\))— formando \(\mathbf{z}_{\text{rel}} \in \mathbb{R}^{32}\). Retropropagación exacta en el tiempo (BPTT analítico verificado a `< 1e-5` por diferencias finitas).

---

## 2. Las 6 Topologías Comparadas (Paridad Exacta `N = 64`, `E = 256` en Todos los Grafos Dispersos)

| Brazo | Nodos \(N\) | Aristas \(E\) | Estructura Topológica y Física |
|---|---:|---:|---|
| **`nervous_system`** | `64` | **`256`** | `64` aferentes bipolares (Pasado→G0/G1, Futuro→G2/G3) + `96` microcircuitos intra-ganglionares (`4` ganglios × `24`, con `2` interneuronas inhibitorias por ganglio, \(\text{sign}=-1\)) + `32` tractos comisurales largos mielinizados (`G0/G1 ↔ G2/G3`) + `64` proyecciones convergentes a `16` hubs (`2` izq. + `2` der. por hub). |
| `grid_local_lattice` | `64` | **`256`** | Retícula 2D `8×8` puramente local (vecinos N/S/E/O, fronteras verticales abiertas entre polo Pasado fila 0 y polo Futuro fila 7; hubs en filas centrales 3–4; **0 axones largos**). |
| `random_erdos_renyi` | `64` | **`256`** | Grafo dirigido aleatorio disperso (`4` aristas salientes/nodo sin auto-bucles ni modularidad; misma densidad y conteo exacto de parámetros). |
| `all_to_all_flat` | `64` | **`4,096`** | Acoplamiento global denso `64×64` (**`16×` más conexiones**, sin localidad ni jerarquía espacial). |
| `nervous_no_delay` | `64` | **`256`** | Ablación física: topología `nervous_system` exacta pero sin retardo de fase ni oscilación (\(\theta_e \equiv 0, \omega_i \equiv 0\), propagación puramente real). |
| `nervous_no_inhibition` | `64` | **`256`** | Ablación física: topología `nervous_system` exacta pero con todas las interneuronas excitatorias (\(\text{sign}_e = +1\), sin inhibición lateral). |

---

## 3. Suites de Validación (`P1..P4`) y Criterios Preregistrados Falsables (`H_N1..H_N6`)

Semillas deterministas: **16 `DEV` (`0xA100..0xA10F`)** para calibración → congelación obligatoria (`FROZEN.json` SHA-256) → **16 `CONFIRM` ciegas (`0xA200..0xA20F`)** + replay exacto bit a bit en `0xA200`. Bootstrap pareado `10,000` remuestreos (`seed = 12345`), IC 95%.

- **Suite P1 (Propagación Física de Ondas):** inyección de pulsos Pasado, Futuro y simultáneo Pasado+Futuro sobre redes entrenadas; mide causalidad en \(t=0\), tiempo de llegada al hub \(t_{\text{arr}}\), velocidad efectiva, energía estacionaria, dispersión/fantasmas fuera de ruta e índice de coincidencia interferométrica en hubs \(\mathcal{C}_{\text{col}} = \frac{\|\,|\psi_{\text{P+F}}|^2 - |\psi_{\text{P}}|^2 - |\psi_{\text{F}}|^2\,\|_2}{\|\,|\psi_{\text{P}}|^2 + |\psi_{\text{F}}|^2\,\|_2 + \varepsilon}\).
- **Suite P2 (Inferencia Relacional y Composición):** Dominio A (\(\mathbb{R}^{20}\), 220 train / 332 inéditos + abducción + 2-saltos) y Dominio C (\(\mathbb{Z}_5\times\{0..3\}\), \(\mathbb{R}^{16}\), 120 train / 260 inéditos + consistencia algebraica) entrenando las 6 topologías desde cero en igualdad de condiciones.
- **Suite P3 (Resiliencia Neurológica sin Reentrenamiento):** evaluación en Dominio A bajo lesiones axonales aleatorias (`15%` y `30%` de aristas seccionadas, promedio de `8` máscaras deterministas por semilla) y ruido sensorial aditivo (\(\sigma = 0.50\)).
- **Suite P4 (Transferencia Cruzada por Mielinización `A → C`, \(N_{\text{few}} = 60\)):** congelación bit a bit (IEEE-754) de todas las sinapsis \((\alpha_e, \theta_e)\) aprendidas en Dominio A, adaptando solo la interfaz y `64` perillas nodales de mielinización/fase \((\Delta\omega_i, \Delta\gamma_i)\) comparando `nervous_system`, `grid_local_lattice`, `random_erdos_renyi`, `all_to_all_flat` y control negativo de sinapsis permutadas.

| ID | Suite | Regla Preregistrada Exacta (`CONFIRM`, *verbatim*) |
|---|---|---|
| **`H_N1`** | P1 | Causalidad exacta `E_hub(t=0) == 0`, llegada rápida `mean(t_arr_nervous) <= 2.5 < mean(t_arr_grid)`, energía acotada `0.15 <= mean(E_final) <= 1.25`, y mayor colisión interferométrica `CI95_lo(coinc_nervous - coinc_grid) > +0.05` y `CI95_lo(coinc_nervous - coinc_random) > +0.05`. |
| **`H_N2`** | P2 (Dom A) | A igual presupuesto `E = 256`, `nervous_system` supera a `grid_local_lattice` (`CI95_lo(acc) > +0.03`, `CI95_hi(rmse) < -0.04`) y a `random_erdos_renyi` (`CI95_lo(acc) > +0.03`, `CI95_hi(rmse) < -0.04`). |
| **`H_N3`** | P2 (Dom A & C) | Con `16×` menos sinapsis que `all_to_all_flat` (`256` vs `4,096`), `nervous_system` mantiene paridad o superioridad: `CI95_lo(acc_A - flat_A) > -0.02`, `CI95_hi(rmse_A - flat_A) < +0.02` y `CI95_lo(acc_C - flat_C) > -0.02` (`> 14×` eficiencia por sinapsis). |
| **`H_N4`** | P2 & P3 (Abl.) | Sobre la misma topología nerviosa (`E = 256`), eliminar retardo de fase (`nervous_no_delay`) cae significativamente (`CI95_lo(acc) > +0.05`, `CI95_lo(abd) > +0.08`), y eliminar inhibición lateral (`nervous_no_inhibition`) pierde robustez ante ruido (`CI95_lo(noise_acc) > +0.02`). |
| **`H_N5`** | P3 (Lesión) | Sin reentrenamiento, `nervous_system` supera a `grid` y `random` bajo lesión axonal `15%` (`CI95_lo > +0.03` contra ambos), a `random` bajo lesión `30%` (`CI95_lo > +0.03`), y bajo ruido sensorial \(\sigma = 0.50\) supera a `grid` y `all_to_all_flat` (`CI95_lo > +0.02` contra ambos). |
| **`H_N6`** | P4 (`A → C`) | Con `256` sinapsis congeladas bit a bit + `64` perillas nodales de mielina (`N_few = 60`), `nervous_transfer` vence a `grid_transfer` (`CI95_lo > +0.03`), a `random_transfer` (`CI95_lo > +0.03`) y al control de sinapsis permutadas (`CI95_lo > +0.05`). |
