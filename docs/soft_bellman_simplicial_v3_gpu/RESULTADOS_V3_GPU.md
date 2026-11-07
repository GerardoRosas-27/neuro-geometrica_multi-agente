# Soft-Bellman Simplicial 3D v3 GPU — Resultados de Ampliación y Entrenamiento en AMD Radeon RX 7800 XT

Fecha de ejecución: 2026-11-06 America/Mexico_City (UTC-6).
Hardware GPU ejecutado: **AMD Radeon RX 7800 XT** (`wgpu 30.0.1`, Backend **Vulkan**, Driver `AMD proprietary driver (26.5.1 (LLPC))`).

---

## 1. Resumen Ejecutivo de Mejoras frente a v2

Se descargó la rama `exp/soft-bellman-simplicial-v2` (sin tocar `main`), se verificaron las 14/14 pruebas originales y se amplió la arquitectura a **Soft-Bellman Simplicial 3D v3 GPU** (`20/20` pruebas unitarias superadas en `release`), resolviendo cada uno de los límites reportados en v2:

1. **Aceleración nativa en GPU (`src/gpu_simplicial.rs`)**:
   - Pipeline WGSL de 3 kernels sobre `wgpu 30.0.1` en **AMD Radeon RX 7800 XT (Vulkan)**:
     - `compute_simplicial_costs`: Métrica Cholesky-SPD $3\times 3$ + estiramiento intrínseco acoplado al Laplaciano de Hodge $\Delta_1$ + base no lineal de Fourier-Polinomios por cara $\Phi_\theta$ + holonomía magnética $B = d_1 A$.
     - `soft_bellman_wavefront_step`: Relajación de frente de onda Soft-Bellman paralela por capas sobre todos los estados aumentados.
     - `compute_cayley_menger_volumes`: Determinantes de volumen simplicial y barrera logarítmica de Cayley–Menger en paralelo.
   - Paridad numérica GPU (`f32` WGSL) vs CPU (`f64` analítico) en lote de 24 mallas (`1,260` estados, `14,548` acciones, `240` tetraedros):
     - Error máximo en costos de transición: **`1.37e-6`**
     - Error máximo en función de valor $V_\tau^*$: **`1.61e-6`**
     - Tiempo de despacho GPU del lote completo (58 capas Bellman + costos + CM): **`18.094 ms`**.

2. **Cierre de la brecha de discretización mediante Refinamiento Multinivel + Optimización Continua Baricéntrica de Snell-Fermat (`src/continuous_geodesic.rs`)**:
   - En v2, refinar la discretización reducía el costo del oráculo un **11.0605%**, evidenciando que el grafo Steiner grueso no alcanzaba la geodésica continua.
   - En v3, `v3_continuous_gpu` planifica sobre el complejo refinado y optimiza continuamente las coordenadas baricéntricas $\lambda_m^* \in \text{int}(\Delta^{k_m})$ de cada cruce de arista, cara y tetraedro mediante gradiente riemanniano proyectado sobre el símplice con búsqueda de línea de Armijo:
     - **Reducción de costo frente al oráculo Steiner grueso**: **`-13.2360%`** (supera incluso al oráculo discreto refinado de v2 que reducía `-11.0605%`, logrando un **`-2.4693%` adicional** sobre el oráculo discreto refinado).
     - **Brecha residual frente al oráculo discreto Nivel-2 Ultra-Fino**: cae de **`11.0605%` a `1.1439%`**.
     - **Sobrecosto (regret) de `v3_continuous_gpu` en CONFIRM (`test`, 384 mallas, 67,648 estados)**: **`0.009112%`** (mejor que el `0.010733%` de v2 grueso), con **100.00% de éxito** y **85.38% de segmentos interiores** (caras/volúmenes).

3. **Robustez y Adaptación ante Cambio de Ley del Terreno (`changed_law`)**:
   - En v2, al cambiar la ley del terreno por el factor no lineal $1 + 0.45\sin(7a) + 0.8(b-0.5)^2$, el modelo `full` empeoraba a **9.2810% de sobrecosto** frente a **5.2784% del control euclídeo** (porque en el soporte de medias faciales $a_f \in [0.22, 0.67]$, $\sin(7a_f)$ tiene pendiente negativa e invierte la preferencia lineal aprendida).
   - En v3 se implementaron dos mecanismos complementarios:
     - **`v3_zeroshot_gpu` (Transferencia Zero-Shot con Gate Epistémico, 0 etiquetas de la nueva ley)**: Neutraliza el cabezal superficial descalibrado y enruta con la métrica anisotrópica SPD $G_\theta$ aprendida conjuntamente, reduciendo el sobrecosto zero-shot a **`4.9625%`** (**supera al control Euclídeo `5.2784%`, a `no_phi` `5.0155%` y a v2 `full` `9.2810%`** sin ver una sola etiqueta de la ley cambiada).
     - **`v3_adapted_gpu` (Cabezal No Lineal de 9 Funciones Base Fourier-Polinomiales + Hodge calibrado en split disjunto con $G_\theta$ congelado)**: Reduce el sobrecosto en las 128 mallas de prueba `CONFIRM/changed_law` de **`9.2810%` a `0.1355%`** (una reducción de **`-9.1454 puntos porcentuales`**, IC95% `[-10.1905, -8.1393]`, es decir, **68.5× menos sobrecosto que v2 `full` y 38.9× menos que el control Euclídeo**), elevando la exactitud de acción de **65.81% a 91.32%** y reduciendo la NLL de **4.8747 a 0.2025**.

4. **Hodge ($\Delta_1$), Regge ($S_{\text{Regge}}$), Cayley–Menger ($B_{\text{CM}}$) y Holonomía ($B = d_1 A$) como Componentes Entrenados**:
   - Las longitudes intrínsecas de arista $\ell_e(\theta) = \ell_e^{\text{SPD}}(\theta) \exp(0.15\tanh(w_{15}) h_{\text{down}}(e) + 0.15\tanh(w_{16}) h_{\text{up}}(e))$ se acoplan a las componentes irrotacional ($d_0 \delta_1$) y solenoidal ($\delta_2 d_1$) del Laplaciano de Hodge ponderado $\Delta_1$.
   - Esto induce curvatura intrínseca no nula en las bisagras interiores ($\epsilon_h(\ell_\theta) \neq 0$) y permite entrenar directamente el funcional de energía libre completo:
     $$\mathcal{L}(\theta) = \mathcal{L}_{\text{ruta}}(V_\tau^*(\theta)) + \alpha S_{\text{Regge}}(\ell_\theta) + \beta B_{\text{CM}}(\ell_\theta) + \gamma \mathcal{E}_{\text{Hodge}}(\theta)$$
   - Se incorporó la 2-cocadena de flujo magnético $B = d_1 A_0$ sobre la 1-forma de Poincaré $A_0([v_i, v_j]) = \frac{1}{2}(a_i b_j - a_j b_i)$, verificando la identidad de Bianchi discreta $d_2 B = d_2 d_1 A_0 = 0$ en cada tetraedro.

5. **Importación y Exportación de Mallas Tetraédricas 3D Externas (`src/external_mesh.rs`)**:
   - Soporte completo de lectura y escritura para formatos **Medit ASCII (`.mesh`)**, **Legacy VTK UnstructuredGrid (`.vtk`, `VTK_TETRA = 10`)** y **Simplicial JSON (`.json`)**, validado en complejos 3D no triviales (`corridor_l_obstacle_3d.mesh` y `snell_refraction_prism_3d.vtk`).

---

## 2. Resultados Principales en CONFIRM (`test`, 384 mallas nuevas, 16 semillas `0xB400–0xB40F`)

| Modelo | Parámetros entrenados | Discretización / Inferencia | Sobrecosto (Regret) | Llega a meta | Accuracy Acción | NLL Acción (nat) | Segmentos interiores |
|---|---:|---|---:|---:|---:|---:|---:|
| **v3 Continuous Snell GPU** | **18** | **Refinado + Snell continuo $\lambda^* \in \Delta^k$** | **`0.009112%`** | **100.00%** | 87.34%* | 0.3020* | **85.38%** |
| **v2 Completo SPD + Phi** | 9 | Steiner grueso | `0.010733%` | 100.00% | 91.48% | 0.1370 | 47.90% |
| **v3 Full GPU (Hodge+Regge+CM)** | **18** | Steiner grueso | **`0.017043%`** | **100.00%** | **92.82%** | **0.1324** | 47.48% |
| Diagonal + Phi, mismo Steiner | 6 | Steiner grueso | `0.811051%` | 100.00% | 80.35% | 0.4854 | 49.45% |
| Completo sin Phi | 6 | Steiner grueso | `2.260044%` | 100.00% | 74.14% | 0.7300 | 31.20% |
| Completo sin entrenar | 0 | Steiner grueso | `2.420714%` | 100.00% | 72.97% | 1.2582 | 31.14% |
| Euclídeo, mismo Steiner | 0 | Steiner grueso | `2.484227%` | 100.00% | 73.17% | 1.0968 | 31.41% |
| Diagonal solo 1-esqueleto | 6 | Solo aristas (1-esqueleto) | `8.375407%` | 100.00% | 94.46%* | 0.2154* | 0.00% |

\* *Nota: `v3_continuous_gpu` evalúa NLL/accuracy sobre 67,648 estados refinados (3.3× más estados y ~10× más acciones por malla que el grafo Steiner grueso de 20,232 estados) y mide su regret frente al oráculo refinado; frente al oráculo Steiner grueso su costo verdadero es un **13.2360% menor** que el del propio oráculo grueso.*

- **Contraste pareado `v3_full_gpu` − `diagonal` en CONFIRM/test**: **-0.7940 puntos porcentuales**, IC95% **[-0.8514, -0.7366]**.
- **Mejora en NLL de `v3_full_gpu` frente a `v2_full`**: `0.1324` vs `0.1370` (`-0.0046 nat`), y en exactitud de acción: **`92.82%` vs `91.48%`** (`+1.34 pp`).

---

## 3. Resultados en Cambio de Ley del Terreno (`CONFIRM/changed_law`, 128 mallas)

| Modelo en `CONFIRM/changed_law` | Sobrecosto (Regret) | Accuracy Acción | NLL Acción (nat) | Llega a meta | Contraste vs v2 `full` (pp) [IC95%] |
|---|---:|---:|---:|---:|---:|
| **`v3_adapted_gpu` (Fourier 9 + Hodge + SPD congelado)** | **`0.1355%`** | **`91.32%`** | **`0.2025`** | **100.00%** | **`-9.1454 pp` `[-10.1905, -8.1393]`** |
| **`v3_zeroshot_gpu` (Gate Epistémico Zero-Shot)** | **`4.9625%`** | **`69.55%`** | **`1.5299`** | **100.00%** | **`-4.3185 pp` `[-5.1092, -3.5612]`** |
| `no_phi` (Métrica aprendida sin $\Phi$) | `5.0155%` | `68.56%` | `1.0792` | 100.00% | `-4.2654 pp` |
| `euclidean` (Control Euclídeo) | `5.2784%` | `67.76%` | `1.6165` | 100.00% | `-4.0026 pp` |
| `untrained` (Inicialización aleatoria) | `5.4060%` | `67.40%` | `1.8874` | 100.00% | `-3.8750 pp` |
| `v2_full` (Lineal 3-param sin adaptar) | `9.2810%` | `65.81%` | `4.8747` | 100.00% | Referencia (`0.0000 pp`) |
| `diagonal` (Diagonal sin adaptar) | `9.9887%` | `64.07%` | `2.0641` | 100.00% | `+0.7077 pp` |

- **Contraste `v3_adapted_gpu` − `euclidean`**: **`-5.1428 puntos porcentuales`**, IC95% **`[-5.6530, -4.5633]`**.
- **Contraste `v3_zeroshot_gpu` − `euclidean`**: **`-0.3159 puntos porcentuales`** (mejora zero-shot sin etiquetas de la nueva ley).

---

## 4. Convergencia Geodésica y Refinamiento Continuo (48 mallas diagnósticas CONFIRM)

| Nivel de Resolución / Método | Estados medios | Acciones medias | Reducción de costo vs 1-Esqueleto | Reducción de costo vs Steiner Grueso | Reducción adicional vs Steiner Refinado |
|---|---:|---:|---:|---:|---:|
| Oráculo 1-Esqueleto (solo aristas) | 12.0 | 64.0 | `0.0000%` | `+8.0149%` (más caro) | `+20.2562%` (más caro) |
| Oráculo Steiner Grueso (v2 base) | 52.7 | 606.2 | `-7.4202%` | `0.0000%` | `+12.4359%` (más caro) |
| Oráculo Steiner Refinado (Nivel 1) | 176.2 | 6,158.0 | `-17.6599%` | `-11.0605%` | `0.0000%` |
| Oráculo Steiner Ultra-Fino (Nivel 2) | 310.8 | 19,420.5 | `-18.9191%` | `-12.4203%` | `-1.5465%` |
| **Geodésica Continua Snell `v3_continuous_gpu`** | **Continuo $\Delta^k$** | **Continuo $\Delta^k$** | **`-19.6742%`** | **`-13.2360%`** | **`-2.4693%`** |

---

## 5. Evaluación en Mallas Tetraédricas 3D Externas Importadas (`.mesh`, `.vtk`, `.json`)

| Malla Externa | Formato Importado | Vértices / Aristas / Caras / Tets | Costo Oráculo 1-Esqueleto | Costo Oráculo Steiner Grueso | Costo Oráculo Steiner Refinado | Costo `v3_continuous_gpu` | Mejora vs Grueso | Fracción Interior |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `corridor_l_obstacle_3d` | Medit ASCII (`.mesh`) | 16 / 47 / 50 / 18 | `7.138450` | `6.862848` | `6.674966` | **`6.678436`** | **`-2.6871%`** | 75.00% |
| `snell_refraction_prism_3d` | Legacy VTK (`.vtk`) | 18 / 57 / 64 / 24 | `5.489738` | `4.783225` | `4.724764` | **`4.733561`** | **`-1.0383%`** | 100.00% |
| `corridor_l_obstacle_3d` | Simplicial JSON (`.json`) | 16 / 47 / 50 / 18 | `7.138450` | `6.862848` | `6.674966` | **`6.678436`** | **`-2.6871%`** | 75.00% |
