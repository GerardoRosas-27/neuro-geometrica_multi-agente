# PLAN_KAGGLE — entrenar un BQ10 «más grande» en Kaggle

Rama `exp/kaggle-bq10-grande` (desde `main`). Nada se ha ejecutado aún en Kaggle (no hay credenciales). 0 consultas TEST.

## 1. Qué se entrena hoy (BQ10 en main)
| pieza | qué es | dónde | ¿se entrena? |
|---|---|---|---|
| Red CONSTELACIÓN v4 | jerarquía de ÓRBITAS-neurona `layers=[16,8,4]`, `m=8` → dims 2→128→64→32; Koopman rot-escala fijo + plasticidad local (W de capas, Π precisiones, τ/poda, sueño CDT) | `orbita/src/constelacion4/mod.rs` | sí, **por semilla**, online (`V4_T_F=128` it. NLMS/frame, flujo A `N_A=3072`, flujo B `N_NEW=N_KNOWN_B=192`); ~370–390 s/semilla |
| Cabeza temporal v3/v4 | x̂(t+1)=x(t)+(W^P+W^F)φ; W^P lento (entre episodios), W^F rápido (por episodio) | idem | W^P lento ≈ no aporta a la primaria (campo rápido solo = v4) |
| ÓRBITA (prior E33) | x̂(1)=ÓRBITA(x0,c) cuando hay contexto y W^F vacío | `artifacts/orbita/{dev,confirm}/weights_*.json` | **congelada** (pesos por semilla precomputados) |
| ATLAS-FW CORE (M1+M2+M6) + M7 | ridge exacto **por episodio** sobre biblioteca polinómica (`N_LIB=17`, grado ≤3, `d=148` features), meta-selección (λ∈{1e-3,1e-2,1e-1}, thr) con `N_FIT=192/N_VAL=96`; M7 = Tikhonov por grado λ_j=λ·da^(g−1), da=10 | `constelacion4/atlas.rs` | ajuste cerrado por episodio (k=4 cuadros observados), ~12–14 s/brazo |

Coste medido (DEV, `artifacts/bio_quantum_v10/bq_dev/metrics_*.json`): **~450–475 s/semilla** de 1 hilo (v4_train ≈ 388 s + brazos v4 4.5 s, CORE/BQ10/BQR10 ≈ 12 s, ALL ≈ 42 s). Memoria: decenas de MB por proceso (estado dims ≤128, d=148). 32 semillas (DEV+CONFIRM) ≈ 4.2 h-CPU → **~1.1 h de pared con 4 núcleos**. Es Rust CPU f64, single-thread por semilla, paralelo por semillas (`xargs -P`), y `bq10_phase.sh` ya es reanudable (`*.done`).

Hallazgos que condicionan el escalado:
- El aprendizaje lento de v4 casi no aporta a la primaria: escalar sólo la red v4 (más neuronas/capas/epochs) probablemente **no** mueve `new_med_log`.
- M7 gana por **reducir sobreajuste** con k=4 (pocos datos por episodio, d=148). Un ajuste por episodio con más features sin más regularización empeorará; la capacidad útil está en *qué* biblioteca/prior se usa y cuánta información se transfiere entre episodios.

## 2. Kaggle (consultado 2026-11-03)
Fuente oficial: https://www.kaggle.com/docs/notebooks y https://www.kaggle.com/docs/efficient-gpu-usage (resumen 2026: https://fastero.com/blog/kaggle-notebooks).
- CPU: **4 núcleos, 30 GB RAM**; GPU P100 (16 GB) o **T4×2** con 4 núcleos/29 GB; TPU v3-8 (96 núcleos/330 GB, sesión 9 h).
- Sesión: **12 h** CPU/GPU, 9 h TPU; interactiva muere a 20 min inactiva → usar «Save & Run All» / `kaggle kernels push` (batch).
- Disco persistente **/kaggle/working 20 GB** (se guarda como salida de la versión); resto es scratch.
- GPU: cuota ~**30 h/semana** (variable, ver ajustes de cuenta); TPU ~20 h/sem; CPU sin cuota publicada. Puede haber cola.
- Internet y GPU requieren **teléfono verificado**. Secrets: Add-ons → Secrets, `kaggle_secrets.UserSecretsClient().get_secret(...)`.
- API CLI: `pip install kaggle`, token `~/.kaggle/kaggle.json` (o `KAGGLE_USERNAME/KAGGLE_KEY`); `kaggle kernels push -p kaggle/`, `kaggle kernels status/output`, `kaggle datasets create|version -p dir` para artefactos.
- Conclusión: **4 núcleos CPU = la mitad que la caja actual (8)**. Kaggle no da más CPU salvo TPU-VM (96 núcleos, sin GPU, 9 h): ahí el arnés Rust escala ~24× por semillas. GPU sólo compensa si portamos a JAX/PyTorch las partes que escalan (ver §4).

## 3. Qué es «modelo más grande» (hipótesis medibles)
Primaria siempre `new_med_log` (h16, familias nuevas 8/10/11, k=4); co-primaria vs BQ10; no-daño known_acc16, E33 16/16, 0 NaN.
- **H1 (red v4 ancha)** `layers=[64,32,16]`, `m=16` (dims 2→1024→512→256), `N_A×4`. Predicción preregistrada: Δ vs BQ10 |Δ|<δ (nulo), por el hallazgo del campo lento. Sirve de control de «tamaño sin efecto».
- **H2 (biblioteca grande + M7)** grado ≤5 / features de Fourier-RBF (`d` 148→~600) con λ_j por grado (M7 extendido) y CV meta. Predicción: mejora si la regularización compensa; riesgo de sobreajuste.
- **H3 (prior aprendido entre episodios)** ridge bayesiano: media previa de los coeficientes de la biblioteca aprendida sobre flujo A/B de entrenamiento (no TEST); este es el «aprendizaje lento útil» que v4 no da. Predicción: mayor ganancia con k=4.
- **H4 (más datos meta)** `N_FIT/N_VAL` 192/96 → 1536/768 y episodios del flujo B ×8.
- Barridos secundarios (solo TUNE): k∈{2,4,8} reportado aparte (primaria sigue k=4), horizonte 16 (fijo) y 32/64 como exploratorio, nº familias nuevas fijo (8,10,11) — añadir familias cambia la métrica, no se hace en confirmatorio.
- Cada H es un flag nuevo con **off ⇒ bit-exacto a BQ10** (test como `off_is_v4_bit_exact`).

## 4. ¿GPU? Port parcial
Las partes que escalan con H2–H4 son álgebra densa: construir Φ (episodios×features), resolver ridge batched (d×d), rollouts h16 en lote. Port JAX/PyTorch (`kaggle/bq10_jax/`, futuro) en f64 (T4 f64 es lento ~1/32; usar f32 con verificación contra Rust ≤1e-6 rel o P100 con f64 decente). Red v4 online (secuencial, pequeña) **se queda en Rust CPU**. Decisión: sólo portar si el piloto H2/H3 en CPU supera 8 h/fase.

## 5. Protocolo (idéntico, preregistrar antes de TUNE)
1. **Paso 0**: `cst4-run --phase smoke --seed 0xC420 --arms cst` → hash `1985bdbad5eee41f` en la CPU de Kaggle (si difiere: no comparable, parar). Más hash BQ10 de una semilla DEV vs `artifacts/bio_quantum_v10/bq_dev`.
2. PREREG `docs/kaggle/PREREG_GRANDE.md` commit+push **antes** de TUNE; regla de selección congelada (estilo `bq10_tune_pick.py`) → `frozen.json`.
3. TUNE sólo 0xC4B0–0xC4B3. DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F, pesos ÓRBITA congelados.
4. Bootstrap pareado por semilla 10 000, seed 12345, IC95 percentil; éxito IC sup<0 y |Δ|≥0.05 en DEV y CONFIRM; no-daño known_acc16 IC inf>−0.05, E33 16/16, 0 NaN; 0 TEST. Negativos se reportan.

## 6. Fases y tiempos (4 núcleos Kaggle CPU)
| fase | contenido | estimación pared | sesiones |
|---|---|---|---|
| F0 setup+Paso0 | rustup + build (~3–5 min) + smoke | ~15 min | 1 |
| F1 réplica BQ10 | DEV 16 semillas brazos v4,ALL,CORE,BQ10 (≈475 s c/u) | ~35 min (+CONFIRM 35) | 1 |
| F2 piloto | H1–H4 a escala ×2 en TUNE (4 semillas) | 1–3 h | 1 |
| F3 escalado | mejor H a escala ×4–×8 en TUNE; medir s/semilla, RAM | 2–8 h | 1–2 |
| F4 DEV+CONFIRM | 32 semillas, brazo congelado + v4 + BQ10 | H1: v4 ×~16 coste ⇒ ~2 h/sem → ~16 h (2 sesiones); H2–H4: ~1–3 h | 1–3 |
Total ≈ 2–4 sesiones de 12 h CPU (sin cuota publicada) o TPU-VM (96 núcleos, 20 h/sem) para H1. GPU sólo si se porta (≤30 h/sem).

## 7. Checkpoints y artefactos
- Todo en `/kaggle/working/ckpt/` (artifacts por semilla + `.done`, binario cacheado por sha, `git_sha.txt`, `status_*.json`).
- Reanudar: publicar la salida como Dataset (`kaggle datasets version -p ckpt`) o añadir la salida del kernel previo como input; `bq10_train.py` copia `/kaggle/input/*/ckpt` y `bq10_phase.sh` salta semillas hechas. `BUDGET_H=11` corta antes del límite de 12 h.
- Opcional: push de `artifacts/` a una rama (`PUSH_BRANCH`) con `GITHUB_TOKEN` en Kaggle Secrets (fine-grained, sólo contents:write de este repo). Nunca en código ni en logs.

## 8. Riesgos
- Hash Paso 0 distinto por CPU/libm (Skylake/AMD variable) → comparar dentro de Kaggle siempre (v4 y BQ10 re-ejecutados allí, no números de la caja).
- 4 núcleos < 8 de la caja: Kaggle no acelera la réplica; su valor es capacidad paralela gratuita/TPU-VM y RAM 30 GB.
- Cola/cambios de cuota; sesión interactiva 20 min inactiva; salida >20 GB (no ocurre: <100 MB).
- Sobreajuste con modelo grande y k=4 (H2): mitigado por M7 extendido y CV meta.
- Fuga: no usar TEST; prior H3 sólo de flujos de entrenamiento.

## 9. Uso
```bash
pip install kaggle; # ~/.kaggle/kaggle.json (chmod 600)
sed -i 's/KAGGLE_USER/<usuario>/' kaggle/kernel-metadata.json
kaggle datasets create -p <dir_vacio_con_dataset-metadata.json>   # bq10-ckpt (una vez)
kaggle kernels push -p kaggle/      # PHASE por defecto paso0; editar env en bq10_train.py o variables
kaggle kernels status <usuario>/bq10-grande; kaggle kernels output <usuario>/bq10-grande -p out/
```
Del usuario se necesita: cuenta Kaggle con **teléfono verificado** (internet, y GPU si se usa), **kaggle.json / API token**, y opcionalmente un **GitHub fine-grained token** guardado como Secret `GITHUB_TOKEN`.
