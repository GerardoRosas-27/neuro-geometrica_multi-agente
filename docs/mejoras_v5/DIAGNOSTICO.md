# CONSTELACIÓN v5 — Diagnóstico cuantitativo post-v4, `cst-vs-orbita`, `constelacion-v4-mejoras` y `dinlat`

Todo el diagnóstico se fundamenta exclusivamente en los artefactos ya cerrados del repositorio (`artifacts/constelacion_v4/`, `artifacts/cst_vs_orbita/`, `docs/constelacion_v4_mejoras/tuning/`, `docs/dinamica_latente/datos/` y `docs/orbita_vs_timesfm/`). No se ejecutó ninguna consulta nueva sobre `TEST` (contador de consultas `TEST = 0`).

---

## 1. Estado de partida tras las cuatro ramas posteriores a CONSTELACIÓN v4

| Rama @ commit | Experimento | Hallazgo principal | Cuello de botella descubierto |
|---|---|---|---|
| `exp/constelacion-v4` @ `fd80320` | F2 (DEV + CONFIRM, 16+16 semillas) | Gana a ÓRBITA-v1 congelada en nuevas (`−0,076` DEV / `−0,107` CONFIRM), no-daño `0,000` (`acc16 = 0,877`) y E33 `16/16`. | `acc16` en familias nuevas sigue baja (`0,069` / `0,066`); atribución a codificación predictiva (PC) es **nula** (`CST − fixed_w = +0,000002 / +0,00005`). |
| `exp/cst-vs-orbita` @ `c1b1f54` | Comparación homologada con presupuesto igualado (`122 880` transiciones) vs `ÓRBITA-eq+LAT` | CST gana H1 (`−0,6902` CONFIRM), H2 (`+0,3359`) y H3 (`+0,4771`). | En modo con contexto sin prefijo ($k=0$): `1-paso acc` de CST es `0,0422 / 0,0417` (= persistencia) y `E33-ctx acc16` cae a `0,0762 / 0,0029` (`0,0107 / 0,0137` incluso con `cst_prior`). Además, `orbita_v1` original alcanza `−0,9134 / −0,9257` en `new_med_log` cuando se evalúa con `reid(k=4)`. |
| `exp/constelacion-v4-mejoras` @ `0967539` | Tuning M1 (sorpresa+momento), M2 (regla local ES), M3 (MLP rápido) en `0xC420–0xC423` | Paso 0 bit a bit PASA (`1985bdbad5eee41f`). M1 mejora ligeramente el criterio de tuning (`−10,134` vs `−10,121`). | **M2 fracasa en meta-val** (`ganancia = 0,00 ≤ 0,02 log`; cualquier $\theta \neq \theta_0$ degrada de `5,36` a `≤ 1,6`). **M3 degrada severamente las familias conocidas** en todas las configuraciones de la rejilla (`−5,76` a `−4,33` vs `−10,12` de v4). |
| `exp/orbita-dinamica-latente` @ `18ba4a4` y `exp/orbita-vs-timesfm` @ `e56ccd6` | Dinámica en latentes Gemma capa 13 (`PCA 16`) vs `AR2` y `TimesFM 2.5` | `ORB_SESGO` supera a `TimesFM` (`+0,3296`), pero usa etiquetas verdaderas. | `ORB_LIBRE` (sin etiquetas) **pierde contra `AR2`** (`H1 = −0,0647 [−0,1040, −0,0262]`, no pasa gate DEV→CONFIRM) y pierde contra `TimesFM` en familias geométricas (`−0,4207`). |

---

## 2. Cuello 1: Por qué $W^F$ pierde $0{,}35\text{–}0{,}45\text{ log}$ frente al oráculo afín en-contexto y por qué M3 destruye las familias conocidas

### 2.1 Evidencia empírica en tuning (`cst3-oracle` vs `v3` vs `v4` vs `M3`)

En `docs/constelacion_v3/DIAGNOSTICO.md` y `docs/constelacion_v4_mejoras/tuning/tune_summary.json`, sobre las mismas semillas de tuning (`0xC410–0xC413` / `0xC420–0xC423`):

| Método | Parámetros rápidos por episodio | `med_log` conocidas | `med_log` nuevas | Observación |
|---|---|---|---|---|
| Persistencia ($\hat{x}(t+1)=x(t)$) | 0 | `−0,19 a −0,24` | `+0,11 a +0,23` | Línea base trivial |
| **Oráculo afín in-context (`oracle_icl`, $\lambda=10^{-6}$)** | $2 \times 3 = 6$ | **`−12,30`** | **`−1,34 a −1,17`** | Ridge sobre los $K=4$ cuadros ($24$ muestras) con $\phi_{\text{aff}}(x) = [x_1, x_2, 1]$ |
| CST v3 ($t_f=32$, NLMS por cuadro sobre $\phi \in \mathbb{R}^{131}$) | $2 \times 131 = 262$ | `−9,33 / −9,40` | `−0,888 / −0,909` | Pierde $\approx 2{,}9\text{ nats}$ en conocidas y $\approx 0{,}35\text{ nats}$ en nuevas frente a `oracle_icl` |
| CST v4 ($t_f=128$, NLMS por cuadro sobre $\phi \in \mathbb{R}^{131}$) | $2 \times 131 = 262$ | `−9,96 / −10,06` | `−0,875 / −0,900` | Empeora nuevas respecto a v3 en `+0,013` DEV / `+0,009` CONFIRM (`IC95` excluye 0) |
| CST v4 + **M3** ($H=32, \eta_V=0{,}1, \eta_U=0{,}05, s_3=32$) | $262 + 4288 = 4550$ | **`−5,76`** (peor caso `−4,33`) | — (no pasa tuning) | Destruye $+4{,}36\text{ nats}$ en conocidas respecto a v4 |

### 2.2 Causa matemática exacta en `orbita/src/constelacion4/mod.rs` y `mejoras.rs`

1. **Estructura de las familias conocidas (`src/v4_dataset.rs`, l. 92–124):**
   De las 8 familias conocidas ($0..7$), **7 familias ($0..6$, es decir $7/8 = 87{,}5\%$)** son transformaciones **exactamente afines** en $\mathbb{R}^2$ (`translate`, `rotate`, `reflect`, `scale`, `shear`, `affine`, `compose_rot_scale`):
   $$\Delta x_p(t) = x_p(t) - x_p(t-1) = A_{\text{ep}}\,x_p(t-1) + b_{\text{ep}}, \quad A_{\text{ep}} \in \mathbb{R}^{2\times 2},\; b_{\text{ep}} \in \mathbb{R}^2.$$
   Sólo la familia 7 (`radial_warp`, $1/8 = 12{,}5\%$) y las familias nuevas $\{8, 10, 11\}$ contienen términos no lineales ($\sin, \tanh, r\cdot\sin(3\theta)$). Por eso el `known_acc16` de CST v4 se clava exactamente en $7/8 = 0{,}875$ (`0,8770` en DEV y CONFIRM).

2. **Olvido intra-episodio tipo Kaczmarz en `head_update` (`mod.rs`, l. 385–410):**
   En cada cuadro observado $t \in \{1, 2, 3, 4\}$, `head_update` ejecuta $t_f = 128$ iteraciones de NLMS usando **únicamente los $N_p = 6$ puntos del cuadro actual $t$**:
   $$W^F \leftarrow W^F + \frac{\eta_F}{\|\Phi_t\|_F^2} \bigl(\Delta X_t - (W^P + W^F)\Phi_t\bigr)\Phi_t^\top, \quad \Phi_t \in \mathbb{R}^{131 \times 6}.$$
   Como $\text{rango}(\Phi_t) \le 6 \ll d_\phi = 131$, hacer $t_f = 128$ pasos con $\eta_F = 1{,}0$ proyecta $W^F$ ortogonalmente sobre la variedad afín de dimensión 6 del cuadro $t$, **sobrescribiendo parcialmente la información de los cuadros anteriores $1, \dots, t-1$** (proyección alternante de Kaczmarz subdeterminada) en lugar de resolver el sistema conjunto de los $K=4$ cuadros ($4 \times 6 = 24$ ecuaciones en el prefijo $\mathcal{B}_{1:4}$).

3. **Contaminación por el transitorio de $\mu_1(t)$ en el vector de rasgos $\phi(t)$:**
   El vector presináptico es $\phi(t) = \bigl[x(t);\; \frac{0{,}3}{\sqrt{128}}F_1(\mu_1(t));\; 1\bigr] \in \mathbb{R}^{131}$. Durante el prefijo $t=1..4$, $\mu_1(t)$ depende de la historia previa vía el prior temporal $\Pi^\tau_1(\mu_1(t) - F_1(\mu_1(t-1)))$. Al repartir norma de $W^F$ sobre las 128 coordenadas de $F_1(\mu_1(t))$ por mínima norma euclídea en $\mathbb{R}^{131}$, cualquier deriva de $\mu_1(t)$ durante el rollout imaginado $h = 5..16$ se amplifica linealmente en $\hat{x}(t+1)$.

4. **Por qué M3 destruye las familias conocidas (`mejoras.rs`, `m3_step`, l. 432–445):**
   M3 suma $\Delta\hat{x}_{\text{M3}} = V\tanh(U\phi + b)$ y actualiza $(V, U, b)$ en cada cuadro $t$ sin separar primero la componente afín $[x_1, x_2, 1]$. Con $H=32$ unidades ocultas, $V \in \mathbb{R}^{2\times 32}$ absorbe parte de la transformación afín del cuadro $t$ a través de la sigmoide $\tanh(\cdot)$; cuando el rollout en $h=5..16$ se desplaza a coordenadas fuera de la envolvente convexa de los 6 puntos del cuadro $t$, la saturación de $\tanh(\cdot)$ introduce un sesgo sistemático que destruye la exactitud en las 7 familias afines (`−10,12` $\to$ `−5,76`).

---

## 3. Cuello 2: Por qué la Codificación Predictiva (PC) tiene atribución cero en v3 y v4

En `docs/constelacion_v4/RESULTADOS.md`, los controles preregistrados de atribución dan:
- `CST − fixed_w`: `+0,000002 [−0,00003, +0,00003]` (DEV) / `+0,00005 [+0,00002, +0,00009]` (CONFIRM)
- `CST − no_err`: `+0,0025 [+0,0005, +0,0048]` (DEV) / `+0,0008 [−0,0002, +0,0018]` (CONFIRM)
- `CST − t1`: `−0,00007` (DEV) / `−0,000001` (CONFIRM)
- `CST − no_sleep`: `−0,00003` (DEV) / `−0,00001` (CONFIRM)
- `CST − one_layer`: `+0,00016` (DEV) / `+0,00005` (CONFIRM)

Las tres causas estructurales en `orbita/src/constelacion4/mod.rs` son:

1. **Inferencia espacial punto a punto sin transición (`step_ex`, l. 358–361):**
   `infer_point` se invoca independientemente para cada punto $p \in \{0..5\}$ sobre su coordenada instantánea $x_p(t) \in \mathbb{R}^2$. Un único punto $(x_1, x_2) \in \mathbb{R}^2$ sin su desplazamiento $\Delta x_p(t) = x_p(t) - x_p(t-1)$ ni relación con los otros $5$ puntos es estadísticamente independiente de la familia $f$ y de los parámetros $p = (p_0, p_1)$. Por tanto, $\mu_1(t)$ sólo puede actuar como un codificador estático de posición $(x_1, x_2)$.
2. **Desconexión entre el error temporal $e^P$ y la jerarquía PC (`infer_point`, l. 280–308):**
   La energía $E$ minimizada por $\mu_{1..L}$ sólo contiene los errores generativos espaciales $\varepsilon_{l-1} = \mu_{l-1} - W_l F_l(\mu_l)$ y el prior fijo $\mu_l - F_l(\mu_l(t-1))$. El error de predicción de la cabeza temporal $e^P(t) = x(t) - \hat{x}(t)$ (`eprior`) **no participa en la inferencia de $\mu_1$ ni en la plasticidad de $W_1$**.
3. **Cancelación en media de $W^P$ global (`head_update`, l. 390–399):**
   Los parámetros de cada episodio se muestrean uniformemente $p \sim \mathcal{U}([-1, 1]^2)$ (`make_ep`, l. 47). Para traslaciones, rotaciones, cizallas y warps impares en $p$, $\mathbb{E}_{p}[\Delta x \mid x] \approx 0$. Una única matriz lenta $W^P$ compartida sin condicionamiento por episodio converge necesariamente a un promedio cercano a cero.

---

## 4. Cuello 3: Por qué CST cae a `0,0137` en `E33-ctx` y a `0,0417` en `1-paso acc` (`cst-vs-orbita`)

En `artifacts/cst_vs_orbita/RESULTADOS.md` (CONFIRM):
- `1-paso acc` (con contexto verdadero $c$, desde $x_0$): `orbita_v1 = 0,9819`, `orbita_eq = 0,6179`, mientras que `cst = 0,0417` (persistencia `0,0420`).
- `E33-ctx acc16` (rollout desde $x_0$ con contexto $c$, sin observar $x_1..x_4$): `orbita_v1 = 0,3232`, `cst = 0,0029`, y **`cst_prior = 0,0137`**.

¿Por qué `cst_prior` falla en `E33-ctx` (`0,0137`) si tiene acceso a ÓRBITA en $h=1$?
Porque en `orbita/src/constelacion4/cvo.rs` (y `e33_net2` en `mod.rs`, l. 574–582), `cst_prior` únicamente sobrescribe `preds[0]` con `d.step(&e.input, &e.context)`, **pero en `E33-ctx` ($k=0$) el campo rápido $W^F$ nunca se actualiza con la transición predicha por el prior $(x_0 \to \hat{x}_{\text{ORB}}(1))$**. Al quedar $W^F = 0$ durante todo el rollout $h=2..16$, la red revierte a persistencia inmediatamente en $h=2$.

---

## 5. Cuello 4: Por qué `ORB_LIBRE` pierde contra `AR2` en `dinlat` (`−0,0647`) y contra `TimesFM` en geométricas (`−0,4207`)

En `docs/dinamica_latente/RESULTADOS.md` y `docs/orbita_vs_timesfm/RESULTADOS.md` (DEV, 16 semillas):
- `H1 ORB_LIBRE vs AR2 (ho)`: `−0,0647 [−0,1040, −0,0262]` (`AR2` supera significativamente a `ORB_LIBRE`).
- `S ORB_LIBRE vs AR2 (ho, geométricas)`: `−0,1764 [−0,2566, −0,1035]`.
- `S ORB_LIBRE vs TIMESFM (ho, geométricas)`: `−0,4207 [−0,4995, −0,3515]`.
- `S AR2 vs VAR2_OLS (ho)`: `+0,0166 [+0,0133, +0,0198]`.

Causas:
1. Cada semilla en `dinlat` tiene sólo $48$ episodios de entrenamiento $\times 8$ transiciones $= 384$ muestras en $\mathbb{R}^{16}$. `AR2` resuelve un ridge global cerrado ($\alpha=1$) sobre $[s_t, s_{t-1}, 1] \in \mathbb{R}^{33}$ ($33 \times 16 = 528$ parámetros con solución analítica óptima), mientras que `ORB_LIBRE` entrena por SGD un lift no lineal $g(s, c) \in \mathbb{R}^{16}$, un codificador de historia $c = \text{MLP}([s_\tau, s_{\tau-1}])$, una mezcla de matrices densas $K(c) = K_0 + \sum_k \text{mix}_k(c) K_k \in \mathbb{R}^{32 \times 32}$ y un corrector Brenier, sobreajustando las 384 muestras.
2. En el espacio latente de Gemma (capa 13 proyectada por PCA a $\mathbb{R}^{16}$), la dependencia de segundo orden $s_{t-1}$ captura directamente la velocidad latente $\Delta s_t = s_t - s_{t-1}$, mientras que `ORB_LIBRE` aplica $K(c)$ sólo al estado actual levantado $z_t = [s_t; g(s_t, c)]$ y comprime la historia en un vector $c$ de baja capacidad entrenado con pocos datos.
