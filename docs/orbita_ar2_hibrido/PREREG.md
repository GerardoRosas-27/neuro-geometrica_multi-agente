# PRERREGISTRO — exp/orbita-ar2-hibrido

Escrito y empujado ANTES de ejecutar DEV. Base: `exp/orbita-vs-timesfm` @ 2d9ff63. Código: `orbita/hibrido/`.

## Pregunta
¿Una combinación ÓRBITA×AR2 con compuerta dependiente del contexto (solo información pasada) predice el siguiente
latente mejor que cada experto por separado?

## Datos
Idénticos a dinlat/vstimesfm (episodios, embeddings Gemma-2-2B capa 13, PCA 16 ajustada solo con train del seed,
Q_rand). 16 DEV 0xA400–0xA40F, 16 CONFIRM 0xB400–0xB40F. **0 consultas a TEST.** Abortar si los baselines
recalculados no coinciden con `res_<seed>.json` de dinlat.

## Brazos (definición completa en PLAN.md)
Primario: **HIB_GATE** = compuerta softmax lineal sobre {AR2, ORB_LIBRE}, características = errores pasados.
Secundarios: HIB_HEDGE, HIB_RES (AR2 + ÓRBITA sobre residuos), HIB_GATE_S (incluye ORB_SESGO, usa etiquetas).
Baselines: AR2, ORB_LIBRE, ORB_SESGO, PERS, MIX50 (control 50/50), ORACLE_FAM (techo, no modelo).
BEST_IND = por seed, min(MSE_AR2, MSE_ORB_LIBRE) en el split evaluado (elección a posteriori ⇒ conservador).

## Métrica
G(A vs B) = log(MSE_B/MSE_A) por seed (MSE medio de episodios del split, error de un paso promediado en
t=2..7). IC95 bootstrap pareado por seed, 10000 réplicas, semilla 12345. δ = 0.05. >0 ⇒ A mejor.

## Hipótesis (decididas en CONFIRM; ho = held-out de familias de train, tf = familias de transferencia)
- **H1 (primaria)**: G(HIB_GATE vs BEST_IND, ho) IC95_lo > δ.
- **H2**: G(HIB_GATE vs AR2, ho) IC95_lo > δ.
- **H3 (transferencia)**: G(HIB_GATE vs AR2, tf) IC95_lo > δ.
- **H4 (la compuerta aporta sobre promedio fijo)**: G(HIB_GATE vs MIX50, ho) IC95_lo > δ.
- Conclusión inversa (híbrido peor) solo si IC95_hi < −δ; en otro caso «no concluyente».
- Sin corrección por multiplicidad en secundarios (descriptivos): vs ORB_LIBRE (ho/tf), por grupo geométricas/no
  geométricas, HIB_HEDGE, HIB_RES (ho/tf), MIX50 vs AR2, HIB_GATE_S vs ORB_SESGO (ho/tf), brecha ORACLE_FAM vs
  HIB_GATE, réplica ORB_LIBRE vs AR2.

## Controles anti-fuga / validez
- **V0 (código)**: características de la compuerta en t dependen solo de s_0..s_t (test automático que perturba
  s_{t+1..T}); compuerta y η ajustados solo con predicciones cross-fit K=4 de train; ho/tf nunca se usan para ajustar.
- **V1**: G(HIB_GATE vs AR2, C1 permutado) IC95_hi < δ.
- **V2**: G(HIB_GATE vs AR2, C2 barajado) IC95_hi < δ.
- Si V1 o V2 fallan, H1–H4 se reportan como inválidas aunque pasen.

## Protocolo
DEV (16 seeds) para depurar; hiperparámetros congelados al terminar DEV (K=4, compuerta lineal, GATE_STEPS=1500,
wd 1e-3, grilla Hedge, STEPS/LR/WD de dinlat). Paso a CONFIRM si media G(HIB_GATE vs AR2, ho) en DEV > 0.
Seeds fallidos no se relanzan; se reportan. Cualquier cambio tras ver DEV se documenta como desviación.
