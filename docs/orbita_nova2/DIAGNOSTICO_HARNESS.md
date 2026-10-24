# Diagnóstico del harness Nova (`orbita/src/nova_train.rs` @ 04dd2e1)

Fecha: 2026-10-24 (America/Mexico_City, UTC−6).

## 1. NOVA−C3 idéntico a NOVA (16/16) — **bug**
`eval_arm` agrupa `Arm::Nova | Arm::NovaNoC3 | …` en una rama que define `use_c5` y `use_lib`, pero **no existe `use_c3`**: ningún brazo NOVA llama a `InnovFilter`; todos hacen `err_h16` desde x0. El PREREG f64b289 define NOVA = C3+C4+C5+C6+C7, así que es una omisión de cableado, no diseño. (`stream_step` tampoco usa C3, pero el flujo sólo aprende C4/C6, así que no afecta a la biblioteca.)
**Arreglo (Nova-2)**: NOVA filtra el estado con C3 en los k pasos observados y hace free-run con el operador elegido 16−k pasos (mismo esquema que ÓRBITA+C3). NOVA−C3 = código antiguo.
Verificación (smoke A400, 4 ep/fam): NOVA ≠ NOVA−C3 (p.ej. k=2 nuevas −1,128 vs −1,075).

## 2. LAT_EP ≡ ÓRBITA (16/16) — **bug**
LAT_EP/LAT deciden "conocido" si S_known ≤ τ, y reciben `tau` = τ_C6 = 0,93866, que es una distancia L2 entre firmas C4. S_known es rel_err medio en k pasos (P95 en conocidas ≈ 0,015–0,037 según `artifacts/orbita/latente_largo_plazo/tau.json`). Con 0,94 casi todo episodio se clasifica como conocido → LAT_EP devuelve ÓRBITA; LAT casi igual (15/16). Error de unidades al reutilizar `tau`.
**Arreglo**: τ_LAT(k) congelado del ciclo latente (k=2 0,0146557; k=4 0,0220203; k=8 0,0371425). Verificación (smoke): LAT_EP ya no coincide con ÓRBITA (k=4 nuevas −0,09 vs −0,89: el adaptador denso por episodio sin árbitro sobreajusta y empeora, lo que lo hace un control realista y no un clon de ÓRBITA).

## 3. Otras observaciones (sin arreglo en exp1, documentadas)
- **Fuga en C5 de NOVA**: C4 se ajusta con los k pasos y `arbitrate` valida sobre los pasos k_fit..k ya vistos. Es la motivación directa de NOVA_BAYES (ajuste sólo con k_fit, validación predictiva en k_val). NOVA se mantiene tal cual para que el contraste sea limpio.
- **ORACLE_EP con peor mediana log que NOVA pese a mejor acc16**: `Module` residual deg=2 ajustado por episodio da distribución bimodal (muy bien o divergente, rel_err topado en 10) → mediana mala con acc alto. No es un bug de la métrica pero el "techo" está mal especificado para log-mediana; ORACLE_EP no entra en exp1.
