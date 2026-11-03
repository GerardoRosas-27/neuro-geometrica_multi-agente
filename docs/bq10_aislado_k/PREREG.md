# PREREG — BQ10 aislado de la red v4 + barrido del número de observaciones contextuales k

Rama `exp/bq10-aislado-k` desde `origin/main` be725f2 (BQ10 integrado). Una sola cadena; **no se integra ningún mecanismo nuevo de Bio-Quantum**. Escrito y publicado **antes** de correr DEV/CONFIRM.

## Brazos
| brazo | definición | qué usa de v4 |
|---|---|---|
| `v4` | CONSTELACIÓN v4 entrenada (`Mech::default()`) | todo |
| `ALL` | ATLAS-FW M1–M6 (sólo k=4) | φ_v4, W^P, M6 |
| `CORE` | ATLAS M1+M2+M6 | φ_v4, W^P, M6 |
| `BQ10` | CORE + M7 DA-STR (da=10), frozen `docs/bio_quantum_v10/frozen.json` | φ_v4, W^P, M6 |
| **`BQ10_nov4`** (nuevo) | M1+M2+M7 (da=10), `nov4=true` | **nada** |
| **`ATLAS_nov4`** (nuevo) | M1+M2, `nov4=true` (control pareado de M7 sin v4) | **nada** |
| `orbita` | ÓRBITA-v1 congelada (reid con k) | — |
| `persistence` | x̂16 = x(k) | — |

**Interruptor `Mech.nov4`** (orbita/src/constelacion4/atlas.rs). Con `nov4=false` el código de BQ10/CORE/ALL/v4 es el mismo camino que en main (verificado en Paso 0 bit a bit). Con `nov4=true`: ψ_p(t)=LIB_S·g(x_p(t)) (sólo la biblioteca M1, 17 funciones; **sin φ_v4**), predicción base x̂ = x (**sin W^P de v4**), **sin compuerta M6**, Θ por ridge/STLSQ (M2) con reescalado DA-STR (M7). **No se usa la red v4 en absoluto** (ni estados, ni entrenamiento): el estado por punto es la última posición observada x(t) (lo que v4 guarda en `PState.xl`); h≤k precuencial, h>k rollout reinyectado desde x(k). Test `nov4_no_net_and_k4_default`: predicciones idénticas con red entrenada y red sin entrenar. Lo único compartido con v4 es el arnés (generador de episodios, métricas, E33 con la cabeza 1-paso ÓRBITA de `Cfg::v4().orb_prior`, que no requiere entrenamiento).
La red v4 se sigue entrenando en cada semilla sólo para los brazos que la usan; el ahorro de tiempo se reporta como `train_s + wall(BQ10)` vs `wall(BQ10_nov4)`.

**Hiperparámetros:** ninguno nuevo. λ, umbral se eligen por semilla con la **misma regla** que BQ10 (rejilla congelada LAMS={1e-3,1e-2,1e-1} × THRS={0,0.05}, objetivo 1 paso en k+1 sobre los mismos N_VAL=96 episodios de validación de `meta_eps`, sin TEST). da=10 congelado. No se usa 0xC4xx para elegir nada.

## Barrido k
k ∈ {2, 4, 8, 12} (el generador tiene HORIZON=16 cuadros y la métrica es h16; k=16 sería observar el objetivo, así que se ajusta a 12). Para cada k se corren v4, CORE, BQ10, BQ10_nov4, ATLAS_nov4, ÓRBITA y persistencia; ALL sólo en k=4. La regla de selección de λ/umbral usa el mismo k. k=4 reproduce exactamente BQ10.

## Paso 0 (aborta si falla)
1. `cst4-run --phase smoke --seed 0xC420 --arms cst` → hash v4 **1985bdbad5eee41f**.
2. `atlas-run --phase bq_tune --seed 0xC4B0` (pesos dev 0xA400) → BQ10 new_med_log k=4 **== −1.4055906062744339** bit a bit (y `arms_k["4"]` idéntico).

## Protocolo
DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F (pesos ÓRBITA dev/confirm). Bootstrap pareado por semilla, 10000 réplicas, seed 12345, IC95 percentil, δ=0.05. 0 consultas TEST (assert). Métrica: new_med_log (mediana log rel. err. h16, familias nuevas); no-harm: known_acc16 (IC inf de Δ > −δ) y E33 16/16, NaN 0.

## Hipótesis y criterios (deben cumplirse en DEV **y** CONFIRM)
**A (primaria, k=4).**
- A1 *v4 innecesario*: Δ(BQ10_nov4 − BQ10) new_med_log con IC sup < +δ (no inferioridad) **y** no-harm de BQ10_nov4 vs BQ10 (known_acc16 IC inf > −δ, E33 16/16, NaN 0). A1b (superioridad): IC sup < 0 y |Δ| ≥ δ.
- A2 *M7 ayuda sin v4*: Δ(BQ10_nov4 − ATLAS_nov4) IC sup < 0 y |Δ| ≥ δ.
- A3 (secundaria): BQ10_nov4 − CORE IC sup < 0 y |Δ| ≥ δ. Se reportan además vs ALL, v4, ÓRBITA, persistencia.
**B (tendencia).** Predicción (hipótesis de reducción de varianza): Δ_k(BQ10 − CORE) se **atenúa** (menos negativo) al subir k. Estadístico: pendiente OLS por semilla de Δ_k contra log2 k, media con IC bootstrap. Confirmada si IC inf de la pendiente > 0 **y** las medias de Δ_k son estrictamente crecientes en k. Refutada si IC sup < 0 en CONFIRM. Igual para el par sin v4 BQ10_nov4 − ATLAS_nov4 (secundaria).

Los negativos se reportan tal cual; no hay re-tuning tras ver DEV.

## Ejecución
`nohup orbita/scripts/bq10k_chain.sh >> artifacts/bq10_aislado_k/chain.log 2>&1 &` (reanudable; .done por semilla, .phase_done por fase; máx 6 jobs, nice 10). Al terminar: `docs/bq10_aislado_k/RESULTADOS.md` (scripts/analyze_bq10k.py), commit+push, marcador `artifacts/bq10_aislado_k/CHAIN_DONE`.
