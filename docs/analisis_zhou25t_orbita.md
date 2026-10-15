# Análisis de Zhou et al. (ICML 2025, PMLR v267) y su aplicabilidad a ÓRBITA

**Fecha:** 2026-10-15 (CDMX). **Tipo:** sólo análisis y documentación; no se entrenó ni se cambió código. Ninguna cifra de este documento es nueva: las del paper se citan de sus tablas; las de ÓRBITA provienen de documentos ya comprometidos en `exp/orbita`.

## 0. Verificación bibliográfica

| Campo | Valor verificado |
|---|---|
| Página | https://proceedings.mlr.press/v267/zhou25t.html |
| PDF | https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhou25t/zhou25t.pdf (21 págs.; leído completo con `pdftotext -layout`) |
| Título | **DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning** |
| Autores | Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto (NYU Courant; LeCun también Meta AI) |
| Venue | Proceedings of the 42nd ICML, PMLR 267, pp. 79115–79135, 2025 |

**Advertencia de alcance.** El paper es de *world models visuales para planificación con acciones*. No trata operadores de Koopman, semigrupos, transporte óptimo/Brenier, ICNN, ni verificación. Por tanto su transferencia a ÓRBITA es **metodológica** (cómo entrenar y evaluar un modelo de dinámica latente), no de componentes matemáticos.

## 1. Resumen fiel del paper

- **Problema (§1, §3 "Overview").** Aprender modelos del mundo *task-agnostic* a partir de trayectorias offline, en un POMDP $(\mathcal O,\mathcal A,p)$ con $p(o_{t+1}\mid o_{\le t},a_{\le t})$, y usarlos en test para alcanzar una observación objetivo $o_g$ sin recompensas, demostraciones ni modelos inversos.
- **Modelo (§3.1).** Tres piezas:
  $z_t\sim\mathrm{enc}_\theta(z_t\mid o_t)$, $z_{t+1}\sim p_\theta(z_{t+1}\mid z_{t-H:t},a_{t-H:t})$, y un decodificador opcional $\hat o_t\sim q_\theta(o_t\mid z_t)$.
  - **Observación (§3.1.1):** DINOv2 *congelado*; $z_t\in\mathbb R^{N\times E}$ (patches; A.7: $14\times14$ patches, $E=384$).
  - **Transición (§3.1.2):** ViT sin tokenización, decoder-only con **máscara causal a nivel de frame** (cada patch de $z_t$ atiende a todos los patches de $z_{t-H:t-1}$); acción codificada por MLP $\phi$ y concatenada a cada patch. ~19 M parámetros (A.7).
  - **Pérdida (Ec. 1), teacher forcing sobre segmentos de longitud $H+1$:**
    $$\mathcal L_{\text{pred}}=\big\|p_\theta(\mathrm{enc}_\theta(o_{t-H:t}),\phi(a_{t-H:t}))-\mathrm{enc}_\theta(o_{t+1})\big\|^2 .$$
    Entrenamiento enteramente en el espacio latente, sin reconstrucción de píxeles.
  - **Decodificador (§3.1.3, Ec. 2):** $\mathcal L_{\text{rec}}=\|q_\theta(z_t)-o_t\|^2$, entrenado **por separado**; propagar esta pérdida al predictor empeora (A.4.2, Tabla 9: SR 0.92 sin vs 0.80 con).
- **Planificación (§3.2, A.5).** Costo terminal $C=\|\hat z_T-z_g\|^2$, $\hat z_t=p(\hat z_{t-1},a_{t-1})$, $\hat z_0=\mathrm{enc}(o_0)$, $z_g=\mathrm{enc}(o_g)$ (Ecs. 3–5). Optimización con **CEM** (muestrear $N$ secuencias, quedarse con las top-$K$, reajustar media/covarianza) dentro de **MPC** (ejecutar las primeras $k$ acciones y re-planificar). Alternativa por gradiente (Ecs. 6–8, $a_t\leftarrow a_t-\eta\,\partial C/\partial a_t$). Tabla 10: MPC > CEM open-loop > GD (p. ej. PointMaze 0.98 / 0.8 / 0.22).
- **Resultados.** Tabla 1 (SR / Chamfer en 6 entornos vs IRIS, DreamerV3, TD-MPC2; p. ej. PushT 0.90 vs 0.30 de DreamerV3); Tabla 2 y 3 (encoders: patches DINOv2 > CLS, R3M, ResNet, MAE; *linear probe* de features a estado); Tabla 4 (configuraciones no vistas: WallRandom 0.82, PushObj 0.34); Tablas 5–6 (LPIPS/SSIM de rollouts decodificados); Tabla 7 (escalado con datos en PushT: SR 0.08 con 200 trayectorias → 0.92 con 18 500); Tabla 8 (máscara causal: sin máscara el SR cae 0.76→0.08 al crecer $H$ de 1 a 3, con máscara sube a 0.92).
- **Detalles (A.7, Tablas 12–13).** *Frameskip* 5 (1 en Rope/Granular) "para que la tarea de predicción sea significativa"; $H\in\{1,3\}$; AdamW, batch 32, 100 épocas.
- **Limitaciones declaradas (§5).** Requiere cobertura suficiente estado-acción; requiere acciones verdaderas; planificación plana (no jerárquica). §3.2 deja como trabajo futuro regularizaciones de entrenamiento/planificación para que GD funcione.

## 2. Situación de ÓRBITA tomada como punto de partida (ya documentada)

Núcleo espectral $K_h=e^{h(S-D)}$ con propiedad de semigrupo $K_{a+b}=K_aK_b$; corrector de Brenier (ICNN, $r=2$, $\eta=0.5$) aplicado recurrentemente; pérdida multi-horizonte $k\in\{1,2,4,8\}$ con pesos $\gamma^{\cdot}$ / $e^{-r/\tau}$ y $\lambda_H=0.01$; 2602 parámetros. Resultados: E33 16/16 seeds con región ≥ 8; h16 0.313 (DEV) / 0.323 (CONFIRM); empate/inconcluso frente a v4 E3 según el criterio prerregistrado. Experimento D: error relativo crece más lento que E3 en h128/h256 con IC95 < 0 en ambas fases, pero acc ≈ 0 para todos desde h ≈ 96. Verificador: no implementado. Ablaciones A (16/16 DEV), B1 y C: parciales, detenidas.

## 3. Mejoras propuestas (de mayor a menor prioridad)

Costos estimados por analogía con los tiempos medidos de ÓRBITA en esta máquina (~1 h por ronda de 8 seeds sin carga; 4 rondas = DEV+CONFIRM). Son estimaciones, no mediciones.

### M1. Diagnóstico *linear probe* del latente (de §4.4, Tabla 3) — prioridad ALTA
Idea del paper: la calidad de una representación para dinámica se mide con una sonda lineal a las variables de estado. En ÓRBITA, medir qué tan bien $z=[q;g(q,c)]\in\mathbb R^8$ y, sobre todo, $z$ tras $h$ pasos del núcleo, codifican el estado verdadero:
$$W^\star=\arg\min_W \sum_i\big\|W\,K_h z_i - x^{\text{true}}_{i,h}\big\|^2+\alpha\|W\|^2,\qquad \mathcal E_{\text{probe}}(h)=\text{MSE de validación}.$$
Si $\mathcal E_{\text{probe}}(h)$ se mantiene bajo hasta $h\approx 96$ mientras la acc del decodificado cae, el fallo largo está en la **lectura**/corrector y no en el núcleo; si sube, está en el núcleo. *Implementación:* script de análisis sobre pesos existentes (subcomando `orbita probe` o Python), sin entrenar. *Costo:* minutos. *Riesgo:* bajo (sólo diagnóstico; no puede alterar resultados prerregistrados). Ayuda a explicar D y a decidir entre M2 y M3.

### M2. "Frameskip"/horizontes de entrenamiento más largos explotando el semigrupo (de A.7 "frameskip" y Ec. 1) — prioridad ALTA
El paper entrena el predictor a saltos de 5 pasos para que la señal no sea trivial. En ÓRBITA el análogo natural es ampliar el conjunto de horizontes de la pérdida, lo cual es barato gracias a $K_{2k}=K_k^2$ (cuadrados repetidos: $O(\log k)$ productos $8\times8$):
$$\mathcal L_{\text{MH}}=\sum_{k\in\mathcal K}w_k\;\frac{\big\|\mathrm{dec}\big(\mathcal C^{(k)}(K_k z)\big)-y_k\big\|^2}{\|y_k\|^2},\qquad \mathcal K=\{1,2,4,8,16,32\},\; w_k\propto e^{-k/\tau}.$$
*Implementación:* exportar desde el harness v4 objetivos de replay a $k=16,32$ (hoy `replay.bin` sólo trae $k\in\{1,2,4,8\}$) — exige tocar el exportador de ÓRBITA, no el código de v4; la pérdida en `train_orbita.py` ya itera `KS`. *Costo:* moderado (re-exportar datos y 4 rondas de entrenamiento; el paso del núcleo es barato pero el corrector recurrente se aplica $k$ veces, así que el costo por lote crece ~linealmente con $\max\mathcal K$). *Riesgo:* medio — los objetivos del teacher a $k$ grande tienen error acumulado (teacher acc 0.57 ya en h1 en el TEST iterado), por lo que se podría aprender el error del teacher; debe usarse sólo sobre la máscara *trusted*. Es la mejora más directa para el hallazgo "acc ≈ 0 desde h ≈ 96".

### M3. Desacoplar la reconstrucción del aprendizaje de la dinámica (de §3.1.3 y A.4.2, Tabla 9) — prioridad MEDIA
El paper encuentra que retropropagar la pérdida del decodificador al predictor empeora. En ÓRBITA, la pérdida multi-horizonte ya compara en latente (`tgt = encode(y_k)` bajo `no_grad`), pero el término de un paso $\ell_1$ se mide tras corrector+decodificación. Propuesta: término de consistencia latente puro para $k>1$ con objetivo detenido, y que el corrector reciba gradiente sólo del término de un paso:
$$\mathcal L=\ell_1\big(\mathcal C(K_1z),y_1\big)+\lambda_H\sum_{k\in\mathcal K} w_k\,\big\|\,\mathrm{sg}[\mathcal C]\!\circ K_k z-\mathrm{sg}[\mathrm{enc}(y_k)]\big\|^2 ,$$
($\mathrm{sg}$ = stop-gradient). *Implementación:* un `detach` sobre los parámetros del corrector en el bucle multi-horizonte. *Costo:* igual al entrenamiento actual. *Riesgo:* medio; puede reducir la acc a h1. Se apoya también en la ablación C (sin corrector) cuando termine.

### M4. Verificador como búsqueda en tiempo de inferencia tipo CEM (de §3.2, A.5.1, Tabla 10) — prioridad MEDIA
ÓRBITA no tiene acciones, así que no hay planificación; pero el *verificador no implementado* puede formularse como selección/búsqueda en test con el propio modelo como juez, que es exactamente el mecanismo de CEM: proponer candidatos, puntuarlos con un costo en latente, quedarse con los mejores. Costo propuesto, usando el semigrupo como chequeo interno sin etiquetas:
$$C(\tilde z)=\big\|K_{2h}\tilde z-K_h\,\mathcal C(K_h\tilde z)\big\|^2+\mu\,\big\|\mathcal C(\tilde z)-\tilde z\big\|^2,$$
y CEM sobre perturbaciones $\tilde z=z+\epsilon$ (o sobre los contextos $c$ inferidos), aceptando sólo si $C$ baja. El paper muestra CEM > GD para optimizar sobre su modelo (Tabla 10), lo que sugiere preferir búsqueda por muestreo a gradiente si se implementa. *Costo:* sólo inferencia, $N\times$iteraciones evaluaciones por paso (barato a 2602 parámetros). *Riesgo:* alto de "trampa": no debe usar el TEST ni el teacher; debe auditarse con el `Auditor` (0 queries) y prerregistrarse. Además, la idea de usar la consistencia de semigrupo como costo es **nuestra**, no del paper.

### M5. Re-anclaje tipo MPC sólo como diagnóstico (de A.5.1 g y Tabla 10) — prioridad BAJA
El paper gana mucho re-planificando (MPC 0.98 vs CEM open-loop 0.8 en PointMaze). E33 mide *free-run* por diseño, así que re-anclar con estados verdaderos no es admisible como resultado. Útil sólo como cota superior diagnóstica: acc con re-anclaje cada $m$ pasos para separar error de modelo de acumulación. *Costo:* inferencia. *Riesgo:* confundirse con el resultado principal; reportarlo siempre aparte.

### M6. Escalado con datos (de §4.8, Tabla 7) — prioridad BAJA (fuera del protocolo)
El paper ve fuerte mejora con más trayectorias. En ÓRBITA el presupuesto de interacción está fijado por el protocolo v4 (misma auditoría y presupuesto que E3); aumentar datos rompería la comparabilidad. Sólo como estudio separado y etiquetado como desviación.

## 4. Hipótesis falsables

Criterio común (igual al protocolo v4/ÓRBITA): seeds DEV 0xA400–0xA40F y CONFIRM 0xB400–0xB40F; mismo presupuesto y auditoría (0 queries al TEST); IC95 bootstrap pareado (10 000 remuestreos, semilla 12345); prerregistro comprometido antes de DEV; sin reajuste con CONFIRM. Una hipótesis se **acepta** sólo si el IC95 tiene el signo indicado en DEV **y** en CONFIRM; se **refuta** si en CONFIRM el IC95 tiene el signo opuesto o el efecto no aparece en DEV.

| # | Hipótesis | Experimento mínimo | Métrica | Criterio |
|---|---|---|---|---|
| H1 (M1) | La pérdida de acc a h ≥ 96 se debe a la lectura/corrector, no al núcleo $K_h$ | Sonda lineal ridge sobre $K_hz$ de los pesos ORBITA-v1 existentes, h ∈ {1,…,256}, ajustada en TRAIN y medida en TEST | $\mathcal E_{\text{probe}}(h)$ vs rel_err del modelo | Se sostiene si, en ambas fases, $\mathcal E_{\text{probe}}(128)<$ rel_err ORBITA-v1(128) con IC95 de la diferencia < 0; refutada si ≥ |
| H2 (M2) | Entrenar con $\mathcal K$ hasta 32 extiende la región útil | ORBITA-v1 + $\mathcal K=\{1,\dots,32\}$, 16 seeds/fase | acc h32, h64; región E33 | Δacc(h64) vs ORBITA-v1 IC95 > 0 y vs E3 IC95 > 0 en DEV y CONFIRM; sin pérdida de acc h1 > 0.02 (IC95 completo por debajo) |
| H3 (M3) | Desacoplar el corrector de la pérdida multi-horizonte reduce el crecimiento del error | ORBITA-v1 con stop-gradient del corrector en $k>1$ | pendiente de log rel_err en h ∈ [16,256]; acc h16 | pendiente menor que ORBITA-v1 (IC95 < 0) en ambas fases y acc h16 no inferior a E3 (IC95 inferior > −0.02) |
| H4 (M4) | Un verificador CEM con costo de consistencia de semigrupo mejora la acc sin consultar el TEST | Inferencia sobre pesos ORBITA-v1, N=64, 5 iteraciones, top-8; 0 queries auditadas | acc h16, h32; colapso | Δacc(h16) vs ORBITA-v1 IC95 > 0 en ambas fases y vs E3 IC95 > 0; colapso no mayor |
| H5 (M5, diagnóstico) | Con re-anclaje cada m=8 pasos, la acc a h128 es > 0 | Rollout con re-anclaje (sólo diagnóstico) | acc h128 | Sólo describe; no entra en la comparación v6 |

H1, H4 y H5 no requieren entrenar (pesos existentes). H2 y H3 requieren 4 rondas de entrenamiento cada una y deben esperar a cerrar las ablaciones A/B1/C pendientes.

## 5. Qué NO aplica y por qué

- **Encoder preentrenado congelado DINOv2 y patches ViT (§3.1.1, Tablas 2–3).** ÓRBITA no recibe imágenes; su entrada son puntos 2D con contexto $c\in\mathbb R^{10}$. No existe un encoder preentrenado relevante, y un ViT de ~19 M parámetros es cuatro órdenes de magnitud mayor que ÓRBITA (2602), incompatible con la comparación en igualdad de condiciones con v4. Sí se toma la *idea de evaluación* (sonda lineal, M1).
- **Máscara causal e historia $H>1$ (§3.1.2, Tabla 8).** La dinámica de E33 (rotar+reflejar) es markoviana en el estado; no hay velocidades ocultas que la historia ayude a estimar. Añadir historia aumentaría parámetros y riesgo de fuga sin motivación física. El hallazgo de la Tabla 8 (sin máscara el modelo "hace trampa" mirando el futuro) sí refuerza la regla ya vigente de no exponer objetivos futuros en entrenamiento.
- **Condicionamiento por acciones y planificación hacia objetivos (§3.2).** E33 no tiene acciones ni objetivos; la métrica es predicción free-run. Por eso CEM/MPC sólo se reutiliza como mecanismo de búsqueda para el verificador (M4) o como diagnóstico (M5).
- **Decodificador a píxeles, LPIPS/SSIM (§4.7, Tablas 5–6), comparación con AVDC (§4.6).** No hay píxeles; las métricas de ÓRBITA son acc < 5 % y error relativo.
- **Teacher forcing puro (Ec. 1).** ÓRBITA ya entrena con objetivos multi-horizonte iterando su propio núcleo, más próximo al uso en free-run; volver a teacher forcing de un paso sería un retroceso.
- **Más datos (§4.8).** Rompe el presupuesto fijado del protocolo (ver M6).
- **Componentes matemáticos.** El paper no contiene Koopman, semigrupos, Brenier/ICNN ni verificadores; nada de eso puede atribuírsele.

## 6. Recomendación
Orden sugerido: (1) terminar las ablaciones A/B1/C ya prerregistradas; (2) H1 (sonda lineal, sin entrenar) para localizar el fallo a h ≥ 96; (3) según H1, H2 (si el núcleo degrada) o H3 (si degrada la lectura/corrector); (4) H4 como primer verificador, prerregistrado y auditado.
