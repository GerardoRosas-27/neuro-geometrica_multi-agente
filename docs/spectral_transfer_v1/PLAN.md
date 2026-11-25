# Plan Preregistrado — Aprendizaje Autónomo de Estructuras Espectrales y Transferencia Entre Dominios (`exp/spectral-transfer-discovery`)

- **Rama aislada:** `exp/spectral-transfer-discovery` (creada directamente desde `main@120e1e8`).
- **Aislamiento estricto:** **CERO conexión** con ÓRBITA (`orbita/`), CDT-RQM (`src/`), `aura-engine/` ni con `Cargo.toml` raíz. Todo el código vive en el crate independiente `spectral_transfer/` y toda la documentación/evidencia en `docs/spectral_transfer_v1/`.
- **Motivación experimental:** Antes de adoptar la interferencia espectral como mecanismo cognitivo general, debemos demostrar de forma aislada y falsable que un campo interferométrico puede:
  1. **Organizar información y descubrir relaciones latentes** desde observaciones opacas sin depender de ceros de Riemann predefinidos ni de números primos;
  2. **Separar causalmente** qué emerge de las **frecuencias** ($\boldsymbol{\Omega}$), qué emerge de las **fases** ($\boldsymbol{\phi}$) y qué emerge de los **emparejamientos interferométricos** ($\pi$ / producto conjugado y suma de fase);
  3. **Transferir con el núcleo espectral 100% congelado** la estructura espectral aprendida en un **Dominio Fuente A** hacia **dos Dominios Objetivo completamente distintos** (**Dominio B**: osciladores físicos no lineales acoplados continuos; **Dominio C**: grafo relacional discreto y álgebra cíclica/jerárquica) en régimen *few-shot*, superando el aprendizaje desde cero, espectros matemáticos fijos (Riemann, Fourier, Poisson, Dyson), transferencia neuronal convencional (`MLP` con paridad paramétrica) y controles negativos estrictos.

---

## 1. Arquitectura Universal del Campo Interferométrico (`SpectralField`)

El sistema se descompone limpiamente en tres módulos independientes para permitir **transferencia estricta con núcleo congelado**:

1. **Interfaz Sensorial Específica de Dominio (`DomainEncoder`):** mapa lineal ligero $\mathbf{E}_{\text{dom}}: \mathbb{R}^{D_{\text{dom}}} \to \mathbb{R}^d$ ($d = 4$, con normalización suave por radio $\mathbf{u} = \mathbf{W}_E \mathbf{x} + \mathbf{b}_E$). Es el **único** adaptador de entrada que se ajusta en dominios objetivo.
2. **Núcleo Espectral Universal (`SpectralCore`, $d = 4$, $M = 48$ modos, `1,936` parámetros, agnóstico al dominio):**
   - Frecuencias aprendidas $\boldsymbol{\Omega} \in \mathbb{R}^{48 \times 4}$ (`192` params).
   - Fases coherentes aprendidas $\boldsymbol{\phi}_{\text{diff}}, \boldsymbol{\phi}_{\text{sum}} \in \mathbb{R}^{48}$ (`96` params).
   - Envolvente de amplitudes positivas $\mathbf{a}_{\text{diff}} = \operatorname{softplus}(\boldsymbol{\alpha}_{\text{diff}})$, $\mathbf{a}_{\text{sum}} = \operatorname{softplus}(\boldsymbol{\alpha}_{\text{sum}})$ (`96` params).
   - **Emparejamiento interferométrico de estados** $(\mathbf{u}, \mathbf{v}) \in \mathbb{R}^4 \times \mathbb{R}^4$:
     $$I_m^{\text{diff}}(\mathbf{u}, \mathbf{v}) = a_m^{\text{diff}} \cos\!\bigl(\boldsymbol{\omega}_m \cdot (\mathbf{v} - \mathbf{u}) + \phi_m^{\text{diff}}\bigr), \qquad I_m^{\text{sum}}(\mathbf{u}, \mathbf{v}) = a_m^{\text{sum}} \cos\!\bigl(\boldsymbol{\omega}_m \cdot (\mathbf{u} + \mathbf{v}) + \phi_m^{\text{sum}}\bigr)$$
     - El canal `diff` (producto conjugado $\psi_m(\mathbf{u}) \psi_m^*(\mathbf{v})$) codifica desplazamientos relacionales: $\phi_m^{\text{diff}} \approx 0$ extrae simetría par (afinidad, sincronización), $\phi_m^{\text{diff}} \approx -\pi/2$ extrae antisimetría impar (causalidad dirigida $a \to b$, flujo de energía) y fases intermedias codifican relaciones quirales/cíclicas.
     - El canal `sum` (producto de enlace $\psi_m(\mathbf{u}) \psi_m(\mathbf{v})$) codifica composición simétrica y enlace de pares.
   - **Operador de lectura interferométrica:** proyección $\mathbf{z}_{\text{rel}}(\mathbf{u}, \mathbf{v}) = \tanh\!\bigl(\mathbf{W}_{\text{pair}}\,[\mathbf{I}^{\text{diff}};\, \mathbf{I}^{\text{sum}}] + \mathbf{b}_{\text{pair}}\bigr) \in \mathbb{R}^{16}$ (`16 × 96 + 16 = 1,552` params).
   - **Operador de Colapso Ondulatorio Pasado × Futuro (abducción sin tabla de pares):** dada una firma interferométrica objetivo $\mathbf{I}_{\text{obs}}$ (o una relación deseada $\mathbf{z}^*$) y el estado conocido $\mathbf{u}_a$, la amplitud de colapso sobre cualquier candidato $\mathbf{u}_k$ evalúa la coincidencia coherente de fase en el interferograma:
     $$\mathcal{C}(k \mid \mathbf{u}_a, \mathbf{z}^*) = \frac{\langle \mathbf{I}(\mathbf{u}_a, \mathbf{u}_k),\, \mathbf{I}^* \rangle}{\|\mathbf{I}(\mathbf{u}_a, \mathbf{u}_k)\|_2 \, \|\mathbf{I}^*\|_2}$$
     (para abducción directa por interferograma observado) y $\mathcal{C}_{\text{rel}}(k \mid \mathbf{u}_a, \mathbf{z}^*) = \langle \mathbf{z}_{\text{rel}}(\mathbf{u}_a, \mathbf{u}_k), \mathbf{z}^* \rangle / (\|\mathbf{z}_{\text{rel}}\|\|\mathbf{z}^*\|)$ (para abducción estructural condicionada a relación).
3. **Cabezal Lineal Específico de Dominio (`DomainHead`):** $\mathbf{H}_{\text{dom}}: \mathbb{R}^{16} \to \mathbb{R}^{O_{\text{dom}}}$ (predicción continua + logits de clase relacional).

### Autoorganización Espectral (0 Ceros de Riemann, 0 Primos)
Durante el entrenamiento en el Dominio Fuente A, el núcleo adaptativo optimiza:
$$\mathcal{L}_A = \mathcal{L}_{\text{tarea}} + \lambda_{\text{repel}}\,\mathcal{L}_{\text{repel}}(\boldsymbol{\Omega}) + \lambda_{\text{alias}}\,\mathcal{L}_{\text{anti-alias}}(\boldsymbol{\Omega}) + \lambda_{\text{coh}}\,\mathcal{L}_{\text{gram}}(\mathbf{I}) + \lambda_{\text{bw}}\,\mathcal{L}_{\text{bw}}(\boldsymbol{\Omega})$$
- $\mathcal{L}_{\text{repel}}$: repulsión local de Riesz entre vectores de onda $\boldsymbol{\omega}_m, \boldsymbol{\omega}_{m'}$ y entre sus radios ordenados $\|\boldsymbol{\omega}_m\|$, induciendo separación espectral sin colisiones desde experiencia.
- $\mathcal{L}_{\text{anti-alias}}$: penaliza resonancias armónicas exactas $\boldsymbol{\omega}_m \approx 2\boldsymbol{\omega}_{m'}$ que generarían ecos fantasma.
- $\mathcal{L}_{\text{gram}}$: reduce la coherencia mutua fuera de la diagonal entre modos activos sobre el lote.
- **Guardia estricta (verificada por test unitario):** el modo `Adaptive` inicializa $\boldsymbol{\Omega}$ puramente con ruido gaussiano/uniforme de la semilla y **entra en `panic!`** si se intenta inyectar cualquier cero de Riemann o frecuencia prima.

### Baseline Neuronal con Paridad Exacta (`MlpCore`, `2,008` params en núcleo)
Para garantizar una comparación justa de representación y transferencia:
- Usa exactamente el **mismo** `DomainEncoder` ($\mathbb{R}^{D} \to \mathbb{R}^4$) y el **mismo** `DomainHead` ($\mathbb{R}^{16} \to \mathbb{R}^{O}$).
- Su núcleo (`MlpCore`) toma la concatenación $[\mathbf{u}; \mathbf{v}] \in \mathbb{R}^8$ y aplica $8 \xrightarrow{\tanh} 48 \xrightarrow{\tanh} 24 \xrightarrow{\tanh} 16$ (`432 + 1,176 + 400 = 2,008` parámetros — `+3.7%` más capacidad que `SpectralCore` de `1,936` params, a favor del MLP).
- Durante la transferencia (`transfer_mlp_A_to_B` y `transfer_mlp_A_to_C`), los `2,008` parámetros de `MlpCore` se **congelan 100%** exactamente igual que `SpectralCore`, entrenando el mismo número exacto de parámetros adaptables (`171` en Dominio B; `136` en Dominio C).

---

## 2. Los Tres Dominios Independientes

### Dominio A (Fuente — Entrenamiento desde cero): Ecosistema Causal-Relacional Abstracto ($\mathbb{R}^{20}$)
- $K_A = 24$ entidades latentes en $[-1.5, 1.5]^4$, observadas solo mediante proyección sensorial no lineal opaca y ruidosa $\mathbf{x}_k = \tanh(\mathbf{M}_A \mathbf{s}_k + \mathbf{b}_A) + \mathcal{N}(0, 0.03^2) \in \mathbb{R}^{20}$.
- Entre cada par ordenado $(a, b)$ ($552$ pares totales), la ley física oculta genera un vector relacional continuo $\mathbf{y}_{ab} \in \mathbb{R}^4$ que combina 4 simetrías distintas:
  1. $y_0$: resonancia simétrica par en $\Delta = \mathbf{s}_b - \mathbf{s}_a$;
  2. $y_1$: precedencia causal antisimétrica impar en $\Delta$ (cambia de signo al invertir $(b, a)$);
  3. $y_2$: acoplamiento quiral/cíclico no simétrico ($\sin(s_{a,0}s_{b,1} - s_{a,1}s_{b,0}) + 0.5\sin(s_{b,2}-s_{a,2})$);
  4. $y_3$: enlace composicional dependiente de $\mathbf{s}_a + \mathbf{s}_b$;
  junto con una etiqueta latente de tipo relacional $r(a, b) \in \{0, 1, 2, 3\}$ (que NO se usa en la evaluación de descubrimiento no supervisado).
- **Partición:** `220` pares de entrenamiento (`40%`) y `332` pares completamente inéditos (`60%`), más `120` consultas composicionales de 2 saltos `(a, b), (b, c) → (a, c)` nunca vistas.

### Dominio B (Objetivo 1 — Transferencia Cruzada): Osciladores Físicos No Lineales Acoplados ($\mathbb{R}^{12}$, *Few-Shot*)
- **Cero solapamiento con A:** otra dimensión sensorial ($\mathbb{R}^{12}$ vs $\mathbb{R}^{20}$), otras entidades ($N_B = 18$ osciladores dinámicos con estado intrínseco de fase, frecuencia natural, amplitud y amortiguamiento no lineal), observados a través de ventanas cortas de trayectoria física (posición, velocidad, curvatura y energía espectral en $\mathbb{R}^{12}$ con ruido).
- Cada par $(i, j)$ ($306$ pares totales) produce una respuesta dinámica acoplada $\mathbf{f}_{ij} \in \mathbb{R}^3$ (`flujo_neto_energía`, `coherencia_sincronía`, `amplitud_batimiento`) y un régimen físico de acoplamiento $c(i, j) \in \{0:\text{desacoplado},\; 1:\text{sincronía difusiva},\; 2:\text{forzamiento dirigido } i\to j,\; 3:\text{batimiento resonante frustrado}\}$.
- **Régimen *few-shot*:** presupuesto principal $N_{\text{few}} = 48$ pares (`15.7%`), evaluado sobre los `258` pares físicos inéditos (`84.3%`). Curva de eficiencia muestral en $N_{\text{few}} \in \{24, 48, 96\}$.
- **Parámetros adaptables en transferencia congelada:** `12×4 + 4` (`52`, encoder) `+` `16×7 + 7` (`119`, head) `= 171` parámetros totales.

### Dominio C (Objetivo 2 — Transferencia Cruzada): Grafo Relacional Discreto y Álgebra Modular/Jerárquica ($\mathbb{R}^{16}$, *Few-Shot*)
- **Cero solapamiento con A y B:** estructura combinatoria discreta de $N_C = 20$ nodos organizados como producto directo de un **anillo cíclico modular** $\mathbb{Z}_5$ y una **jerarquía estricta de 4 niveles** ($\{0, 1, 2, 3\}$), observados únicamente mediante embeddings continuos ruidosos opacos en $\mathbb{R}^{16}$ (sin IDs discretos).
- Cuatro relaciones estructurales discretas sobre los `380` arcos ordenados $(u, v)$:
  - `0`: Mismo nivel o equivalencia (simétrica);
  - `1`: Dominancia jerárquica estricta `tier(u) < tier(v)` (antisimétrica y transitiva);
  - `2`: Sucesor cíclico `(ring(v) - ring(u)) mod 5 == 1` (dirigida no transitiva);
  - `3`: Dominancia inversa / ortogonal.
- **Régimen *few-shot*:** presupuesto principal $N_{\text{few}} = 60$ arcos (`15.8%`), evaluado sobre los `320` arcos inéditos (`84.2%`). Curva de eficiencia muestral en $N_{\text{few}} \in \{30, 60, 120\}$.
- **Métricas adicionales de álgebra relacional:** consistencia transitiva y antisimétrica sobre tripletas inéditas, `NMI` de descubrimiento no supervisado de relaciones sobre $\mathbf{z}_{\text{rel}}$ y abducción *zero-shot* del nodo destino mediante colapso ondulatorio.
- **Parámetros adaptables en transferencia congelada:** `16×4 + 4` (`68`, encoder) `+` `16×4 + 4` (`68`, head) `= 136` parámetros totales.

---

## 3. Suites Experimentales y Brazos de Control

1. **Suite T1 (Dominio A — Organización, Descubrimiento No Supervisado e Inferencia):**
   - Brazos: `adaptive_spectral`, `riemann_fixed`, `fourier_fixed`, `poisson_fixed`, `dyson_fixed`, `mlp_conventional`.
2. **Suite T2 (Dominio A — Disección Causal de Frecuencias, Fases y Emparejamientos):**
   - Brazos: `full_adaptive`, `ablate_phase_scramble` (fases aleatorizadas en inferencia), `train_zero_phase` (entrenado con $\boldsymbol{\phi} \equiv 0$), `ablate_pairing_additive` (superposición aditiva $\Phi(\mathbf{u})+\Phi(\mathbf{v})$ sin producto interferométrico), `ablate_pairing_permuted` (emparejamiento cruzado $\pi(m) \neq m$ rompiendo la conjugación), `ablate_freq_riemann`, `ablate_freq_fourier`, `ablate_freq_poisson`.
3. **Suite T3 (Transferencia `A → B` — Osciladores Físicos Continuos, $N_{\text{few}} \in \{24, 48, 96\}$):**
   - Brazos con núcleo congelado (`171` params adaptables):
     - `transfer_spectral_A` (núcleo adaptativo entrenado en A, 100% congelado)
     - `riemann_A`, `fourier_A`, `poisson_A`, `dyson_A` (frecuencias fijas + fases y $\mathbf{W}_{\text{pair}}$ entrenados en A, luego 100% congelados — control máximo de justicia)
     - `transfer_mlp_A` (`MlpCore` de `2,008` params entrenado en A, 100% congelado)
     - `neg_ctrl_noise_A` (idéntico a `transfer_spectral_A` pero entrenado sobre etiquetas/vectores permutados aleatoriamente en A, 100% congelado)
     - `neg_ctrl_phase_scramble` (`transfer_spectral_A` con fases $\boldsymbol{\phi}_A$ aleatorizadas al congelar)
     - `untrained_poisson` (núcleo espectral sin entrenar en A, 100% congelado — control de características aleatorias de Fourier)
   - Brazos entrenados desde cero sobre B (`~2,107` / `~2,179` params): `scratch_spectral`, `scratch_mlp`.
4. **Suite T4 (Transferencia `A → C` — Grafo Relacional Discreto y Álgebra, $N_{\text{few}} \in \{30, 60, 120\}$):**
   - Mismos 11 brazos exactos con núcleo congelado (`136` params adaptables) frente a `scratch_spectral` y `scratch_mlp`.

---

## 4. Criterios Preregistrados Falsables (`H_T1..H_T8`)

Semillas deterministas: **16 `DEV` (`0xD100..0xD10F`)** para calibración y validación de ingeniería; congelación obligatoria (`FROZEN.json` con hashes SHA-256) antes de ejecutar **16 `CONFIRM` (`0xE100..0xE10F`)** una sola vez. Estadística: bootstrap pareado de `10,000` remuestreos (`seed = 12345`), IC 95%.

| ID | Suite | Criterio Preregistrado Exacto (evaluado *verbatim* en `CONFIRM`) |
|---|---|---|
| **`H_T1`** | T1 | `CI95_lo(nmi_unsup_adapt) > 0.35`, `CI95_lo(acc_adapt - acc_riemann) > +0.02`, `CI95_lo(acc_adapt - acc_poisson) > +0.04` y `CI95_lo(<r>_final - <r>_init) > +0.08` (con `0` ceros de Riemann). |
| **`H_T2`** | T1 | `CI95_lo(abd_adapt - abd_mlp) > +0.10`, `CI95_hi(rmse_adapt - rmse_mlp) < -0.03` y `CI95_lo(cfr_adapt) >= 0.90`. |
| **`H_T3`** | T2 | `CI95_lo(acc_full - acc_phase_scramble) > +0.15`, `CI95_lo(acc_full - acc_pairing_additive) > +0.20` y `CI95_lo(acc_full - acc_pairing_permuted) > +0.20`. |
| **`H_T4`** | T3 (`N=48`) | Transferencia positiva y real en B: `CI95_lo(acc_tr - acc_scratch_spec) > +0.05`, `CI95_hi(rmse_tr - rmse_scratch_spec) < -0.04`, `CI95_lo(acc_tr - acc_noise_A) > +0.05` y `CI95_lo(acc_tr - acc_untrained) > +0.08`. |
| **`H_T5`** | T3 (`N=48`) | Ventaja espectral frente a MLP y espectros fijos en B: `CI95_lo(acc_tr - acc_transfer_mlp) > +0.04`, `CI95_hi(rmse_tr - rmse_transfer_mlp) < -0.02`, `CI95_lo(acc_tr - acc_riemann_A) > +0.02` y `CI95_lo(acc_tr - acc_poisson_A) > +0.03`. |
| **`H_T6`** | T4 (`N=60`) | Transferencia positiva y real en C: `CI95_lo(acc_tr - acc_scratch_spec) > +0.05`, `CI95_lo(acc_tr - acc_noise_A) > +0.05`, `CI95_lo(acc_tr - acc_phase_scramble) > +0.10` y `CI95_lo(nmi_tr - nmi_untrained) > +0.08`. |
| **`H_T7`** | T4 (`N=60`) | Ventaja espectral y álgebra relacional en C: `CI95_lo(acc_tr - acc_transfer_mlp) > +0.04`, `CI95_lo(acc_tr - acc_riemann_A) > +0.02`, `CI95_lo(cons_tr - cons_transfer_mlp) > +0.03` y `CI95_lo(abd_tr - abd_untrained) > +0.10`. |
| **`H_T8`** | T3 & T4 (ultra few-shot) | En $N_{\text{few}} = 24$ (B) y $N_{\text{few}} = 30$ (C), `transfer_spectral_A` supera tanto a `scratch_spectral` como a `scratch_mlp` con `CI95_lo > +0.06` en las 4 comparaciones pareadas. |

---

## 5. Verificación Técnica Obligatoria

1. **Pruebas unitarias Rust `--release`:**
   - Verificación de gradientes analíticos vs. diferencias finitas (`< 1e-5`) en `SpectralCore`, `MlpCore`, `DomainEncoder`, `DomainHead` y regularizadores espectrales.
   - Verificación bit a bit de que el núcleo transferido (`SpectralCore` y `MlpCore`) **no cambia ni un solo bit** durante toda la adaptación a Dominio B y Dominio C.
   - Guardia `#[should_panic]` asegurando que `Adaptive` rechaza ceros de Riemann y frecuencias primas.
   - Exactitud de ceros de Riemann (`< 1e-9`) para el brazo de control `riemann_fixed` / `riemann_A`.
2. **Replay bit-exacto:** re-ejecutar `0xE100` independientemente y comprobar coincidencia exacta byte a byte en todos los CSVs (`T1..T4` + traza de espectros/colapso).
3. **Auditoría automática (`AUDIT.md`):** hashes SHA-256 de `FROZEN.json`, integridad de todos los CSVs `DEV` y `CONFIRM` (`0` NaN/Inf), cero diffs en `orbita/`, `src/`, `aura-engine/` y `Cargo.toml` raíz, y aislamiento estricto de rama.
