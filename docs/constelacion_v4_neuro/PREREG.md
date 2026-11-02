# CONSTELACIÓN v4-NEURO — Preregistro

Este documento se fija con commit y push **antes** de cualquier ejecución DEV.

## Semillas
- **Fase 1:** DEV 0xA460–0xA46F, CONFIRM 0xB460–0xB46F.
- **Fase 2:** DEV 0xA470–0xA47F, CONFIRM 0xB470–0xB47F.
- Todas son semillas nuevas, sin uso previo.

## Código y reglas fijas
- Código congelado en este commit: `orbita/src/constelacion4/neuro.rs` (más los ganchos en `constelacion4/mod.rs`), comando `cst4n-run`.
- Rama `exp/constelacion-v4-neuro`, creada desde `exp/constelacion-v4` (fd80320). No se toca main.
- `cst4n-run` aplica tres restricciones:
  - rechaza flags extra en dev/confirm;
  - asserta los rangos de semilla;
  - exige 0 consultas TEST (assert).
- NaN cuenta como fallo.
- No hay reajuste entre DEV y CONFIRM. Las constantes de los mecanismos están fijadas en `neuro.rs` (abajo).

## Mecanismos
Cada mecanismo es un flag en `Cfg.nm`. Con todos en off, el resultado es v4 exacta.

- **M1. Disparo gatillado por sorpresa.**
  - En el primer paso de inferencia de cada cuadro, cada ÓRBITA-neurona calcula su sorpresa. La sorpresa es el error de predicción que le llega, con precisión: media sobre su bloque de (−Jᵀ(C⊙W)ᵀε_abajo + ε_propio)², evaluada en μ = prior.
  - Si la sorpresa supera θ/g, la neurona dispara: actualiza μ y difunde.
  - Si no la supera, μ queda en su prior y las demás usan su predicción cacheada F(prior). Esto ahorra cómputo.
  - θ es homeostático por neurona (sólo en entrenamiento): log θ += 0.05·(disparó − 0.5).
  - θ se inicializa con la primera sorpresa observada.
- **M2. Refractario.** Tras disparar, la neurona no dispara durante K_REF = 1 cuadro.
- **M3. Plasticidad de tres factores sobre W (A = neurona presináptica en la capa l+1, B = unidad de la capa l).**
  - La actualización PC de v4 se multiplica por la compuerta elig_A · fire_B · [e²_B < EMA_0.1(e²_B)].
  - elig_A: A disparó en este cuadro o en el anterior.
  - fire_B: B disparó en este cuadro.
  - Tercer factor: el error de B bajó.
  - **Anti-colapso:** la compuerta está en [0,1] y sólo filtra la regla PC de v4. No crea dirección nueva, a diferencia de la regla local de v2 que colapsó a ~0. El colapso se mide igual que en v4.
- **M4. Debilitamiento y poda.**
  - Crédito por conexión = Σ |ΔW| · (+1 si el error de B bajó; −1 si no), en ventanas de 500 pasos.
  - Crédito ≤ 0 ⇒ W ×0.5. Dos ventanas seguidas con crédito ≤ 0 ⇒ poda.
  - Rebrote: si en una fila la EMA corta del error es > 2× la EMA larga (0.001), se reactiva la conexión inactiva con mayor actividad, con W = 0.
- **M5. Consolidación.**
  - Una conexión con 4 ventanas seguidas de crédito > 0 pasa a memoria lenta: tasa ×0.1, exenta de M4.
- **M6. Neuromodulación global.**
  - g = clip(√(E0/EMA_0.01(E0)), 0.5, 2).
  - La tasa de W y de la cabeza se multiplica por g. El umbral de M1 se divide por g.

## Verificación off ≡ v4
Hecha antes del preregistro. Artefactos en `artifacts/constelacion_v4_neuro/`:
- **`off_eq_v4_check.json`:** en semilla smoke 0xC4E0, con na=48 y nb=8, el brazo v4 de `cst4n-run` es igual a `cst4esc-run --arms v4` (exp/constelacion-v4-escalada, ya verificado ≡ `cst4-run` de v4). Coinciden el hash de pesos, pre, held, collapse, E33, AUROC, eta_scale, poda y sueños: todo `true`.
- **`paso0_smoke_0xC4E0.json`:**
  - (a) Cfg::v4 tiene nm = off;
  - (b) red clonada tras el flujo A con NState reiniciado ≡ ejecución en línea recta, bit a bit;
  - (d) todos los brazos sin NaN.

## Fase 1: reutilizar entrenamiento
- Por semilla, v4 se entrena una sola vez en el flujo A (N_A = 3072), exactamente como v4. Se mide pre.
- Cada brazo clona esa red ya entrenada (pesos v4 reutilizados, sin re-entrenar desde cero) y activa sus mecanismos desde el flujo B (192+192). Después vienen las evaluaciones H, colapso y E33.
- Brazos: v4, M1, M2, M3, M4, M5, M6, ALL.
- ÓRBITA-v1 congelada: la semilla 0x{A,B}4{6,7}i usa `artifacts/orbita/{dev,confirm}/weights_0x{A,B}40i.json`. Es la misma desviación declarada en v4-escalada: no se re-entrena.

## Métricas
- **Primaria:** new_med_log (mediana de log rel_err h16 en familias nuevas, k = 4). Contraste V − v4.
- **No-daño:** known_acc16 (final − pre, y V − v4) y E33 (pass ≥ 15/16, acc h1).
- **Se reportan además:**
  - n_params y conexiones activas;
  - fracción de neuronas que disparan (flujo B y evaluación) como proxy de cómputo;
  - AUROC de sorpresa (h1_log, flujo B);
  - referencia ÓRBITA-v1;
  - M4 (debilitadas, podadas, rebrotes) y fracción consolidada M5.

## Estadística
- Bootstrap pareado por semilla, 10000 réplicas, semilla 12345, IC percentil, δ = 0.05.
- **Hipótesis principal:** ALL (IC95).
- M1..M6 en "V − v4" primaria usan IC Bonferroni ×7 (99.29%). El resto va con IC95.

## Hipótesis (V ∈ {M1..M6, ALL})
- **H1:** IC sup (V − v4) en new_med_log < 0, en DEV y en CONFIRM.
- **H2:** IC inf (final − pre) en known_acc16 > −δ, y IC inf (V − v4) en known_acc16 > −δ.
- **H3:** IC sup (V − ÓRBITA) en new_med_log < 0.
- **H4:** E33 ≥ 15/16 y IC inf (V − v4) en acc h1 > −δ.
- **Secundaria:** IC inf (V − v4) en AUROC > −δ.

## Gate DEV → CONFIRM
Idéntico a v4-escalada. V pasa si cumple todo lo siguiente:
- 16/16 semillas, 0 NaN, 0 colapso, 0 TEST;
- media (V − v4) en primaria < 0;
- H2 en DEV;
- E33 ≥ 15/16.

En CONFIRM corren v4 más los brazos que pasan.

## Pasar la fase 1
V "pasa" si cumple H1, H2 y E33 ≥ 15/16, limpio, en DEV y CONFIRM.

"Candidata a main" exige además H3, h1 no inferior y sorpresa no inferior.

## Fase 2: desde cero
Sólo se ejecuta si algo pasa la fase 1.
- **Mecanismos de NEU:**
  - si ALL pasa: M1..M6;
  - si no: la unión de los Mi que pasan.
  - Se deciden automáticamente por `analyze_constelacion_v4_neuro.py p1 dev confirm --mech`.
- **Brazos:** v4 y NEU, los dos entrenados desde cero en el mismo flujo A+B, con los mecanismos activos desde el inicio.
- **Semillas:** 0xA47x y 0xB47x.
- Mismas métricas, gate y criterios, con NEU como principal (IC95).

## Expectativa declarada (sin números)
- En v4, eta_scale suele terminar muy reducido (los sueños CDT lo dividen a la mitad), así que en la fase 1 los mecanismos de plasticidad (M3–M6) pueden tener efecto pequeño.
- M1/M2 cambian la inferencia y reducen cómputo, pero podrían perjudicar la primaria.
- Es plausible un resultado nulo o negativo.
