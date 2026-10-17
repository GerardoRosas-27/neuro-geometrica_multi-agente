# Prerregistro único E0–E5: memoria por capas (ÓRBITA-v1 congelada ⇄ CDT-RQM-epr con sueño)

Basado en `docs/arquitectura_capas_memoria.md` (7caad73) y en las decisiones del usuario (2026-10-17 12:55 CST):
(1) familias nuevas no afines = rotación + no linealidad suave y deformación espiral, generadas en `orbita/` sin tocar v4;
(2) ÓRBITA-v1 **congelada** (no se reentrena en sueño); (3) códec de campo mínimo reimplementado en `orbita/`.
Este documento se publica (push sin force) **antes** de ejecutar nada; el código se escribe y prueba después sólo con
la seed de humo 0x5EED (fuera de DEV/CONFIRM), cuyos resultados no se usan.

## Datos
- Seeds: DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F (16 c/u). Pesos ÓRBITA-v1 congelados `/workspace/orbita-data/{dev,confirm}/w_<seed>.json`.
- Generador v4 `generate_and_seal(seed, standard(6))` sin modificar. **Nunca se lee `ds.test`** (0 queries a TEST).
- Familias A = las 8 de v4. Vigilia A = `ds.train`. Evaluación A = `ds.dev` (partición DEV).
- Familias B (nuevas, definidas en `orbita/src/mem.rs`), parámetros p ~ U[−1,1]² con RNG propio:
  - **B1 rotación + no linealidad suave**: (u,v) = R(0,6·p1)·q; salida (u + 0,25·p2·sin(2v), v + 0,25·p2·sin(2u)).
  - **B2 espiral**: r = |q|, θ = 0,4·p1 + 0,8·p2·(r − 0,7); salida e^{0,1·p2}·R(θ)·q.
  - Contexto que recibe ÓRBITA para B (lo que diría el encoder: familia base + p): B1 → one-hot 1 (rotación), B2 → one-hot 6
    (rotación+escala). **Supuesto declarado**: no se ejecuta Gemma en esta tarea (el encoder ya validado en e2e reconoce las
    familias A; la etapa numérica no tiene texto, así que «split por plantilla» sólo aplica a E4).
  - Vigilia B: 1024 episodios (512 por familia, intercalados), formas = entradas de `ds.train`. Evaluación B: 256 por familia,
    formas = entradas de `ds.dev`.
- Episodio: (x0, familia del encoder f, p, una observación y1 = regla(x0)); objetivo y_h = regla^h(x0).
  Métrica `ok_h` = rel_err < 0,05 (`v4_metrics::correct`). **Primaria acc4**; secundaria acc1.
- Tarea continual: A (vigilia+sueño) → B (vigilia con sueños disparados) → evaluación en A (retención) y B.
  **Puntuación primaria por seed: S = ½·(acc4 en A_eval + acc4 en B_eval)**.

## Memoria y sueño (mecanismos v4 existentes, sin modificarlos)
- Sueño A: `collect_experience(train, Real)` → `consolidate(Full)` → `family_fit_residual` → **poda**: se descartan las reglas con
  residuo ≥ `TRUST_TAU` (=0,01) → se borra el almacén.
- Umbral de novedad por familia base f: τ_f = max(0,05; 2 × cuantil-95 del rel_err de un paso, en la vigilia A, del predictor
  activo de f (regla de memoria si existe, si no ÓRBITA)).
- Vigilia B: para cada episodio, puntuación de sorpresa r = rel_err(predictor activo de f (x0), y1). novel ⇔ r > τ_f.
  Un episodio novel se guarda en el buffer del **slot nuevo** f′ (uno por familia base); si el slot f′ ya existe y explica
  la observación (r′ ≤ τ_f) no es novel.
- **Saturación** (dispara sueño): ventana W = 64 episodios con tasa de novedad ρ_W > 0,25 y buffer ≥ 64, **o** buffer ≥ N_max = 256.
  Se duerme también al final de la vigilia B si el buffer no está vacío.
- Sueño B: por slot, experiencias = `pseudo_experiences(señal previa del slot, n_previo)` ∪ buffer → `consolidate` → poda
  (según variante) → borrar buffer. ÓRBITA no cambia.
- Enrutamiento en evaluación (idéntico para todas las variantes con memoria y RAW_BUFFER): regla A de f si existe y r ≤ τ_f;
  si no, slot f′ si existe; si no, ÓRBITA. Rollout = aplicar el predictor elegido h veces.

## Condiciones
- Variantes de memoria (todas mecanismos v4): **CDT** = `consolidate(Full)` + poda trusted; **CDT_NP** = sin poda;
  **CDT_GEN** = `consolidate_generic` (RFF ridge, sin poda).
- Controles: **NOCDT** (ÓRBITA sola); **SHUF** (CDT con y barajadas dentro de familia: `ExpMode::ShuffledOrder` en A,
  equivalente en B); **CORRUPT** (`consolidate(Corrupt)`, coeficientes permutados); **ORACLE_ORB** (ÓRBITA con la familia de las
  8 que mejor explica y1, p dado); **ORACLE_LS** (ajuste afín por mínimos cuadrados con los 6 puntos de (x0,y1), iterado);
  **RAW_BUFFER_V** (para cada variante V: reservorio uniforme de episodios crudos de toda la vigilia con el **mismo presupuesto de
  números** que la memoria final de V; 26 números por episodio: x0, y1, p; predicción = `nearest_answer` de v4
  (vecino por l2(x)²+4·l2(c)² dentro del mismo slot, desplazamiento aplicado), mismo enrutamiento).

## Hipótesis y criterios (δ = 0,05; IC95 bootstrap pareado por seed, 10 000 remuestreos, semilla 12345)
- **E0 saturación/novedad**: (a) AUROC(r → fallo acc4 de NOCDT) − 0,5, IC95 > 0 sobre A_eval ∪ B_eval;
  (b) descriptivo: TPR de novedad en vigilia B y FPR en A_eval, con IC95.
- **E1 validez de la tarea**: B debe exigir memoria: IC95 superior de ORACLE_LS acc4(B) < 0,9 **y** NOCDT acc4(B) < NOCDT acc4(A)
  (IC95 de la diferencia < 0). Si falla, la tarea no discrimina: se documenta y se para.
- **E2 (kill criterion)**: para V ∈ {CDT, CDT_NP, CDT_GEN}: S(V) − S(RAW_BUFFER_V). **Kill** si ninguna variante tiene IC95 > 0
  en DEV **y** en CONFIRM. Si se cumple el kill: se documenta y **no** se ejecutan E3–E5.
  Hipótesis principal: S(V) − max(NOCDT, SHUF, CORRUPT, ORACLE_ORB, ORACLE_LS, RAW_BUFFER_V) IC95 > 0 en ambas fases.
  Equivalencia: IC95 ⊂ (−δ, δ).
- **E3** (sólo si no hay kill): sueño disparado por saturación vs sueño a intervalos fijos con el mismo número de sueños
  (variante ganadora de E2): ΔS IC95 > 0.
- **E4** (sólo si no hay kill) códec CDT→campo→texto. **El traductor es un banco de frases** (`translate_field_to_text`
  reimplementado: vecino por coherencia de fase; no hay generación libre). Rasgos = one-hot de slot (10) ⊕ p; FieldEncoder
  mínimo (12 aristas, matriz compleja fija, fasores unitarios). Banco: plantillas es/en/fr «familia/family/famille NOMBRE,
  p1 = V, p2 = W» con V,W en malla de 0,25; plantillas pares para DEV, impares para CONFIRM (split por plantilla).
  Métricas objetivas: familia recuperada (palabra clave) y |Δp| recuperado con `numnorm` desde la frase elegida; controles:
  memoria SHUF y banco permutado. Criterio: familia − max(controles) IC95 > 0.
- **E5** (sólo si no hay kill): poda trusted vs sin poda vs poda aleatoria del mismo número de reglas: ΔS IC95.

## Ejecución
`setsid nohup nice -n 0`, log y `.done` por seed/fase en `/workspace/orbita-data/mem/`, reanudable; DEV y luego CONFIRM.
Resultados: `docs/resultados_memoria_capas.md`, artefactos `artifacts/orbita/mem/`.
