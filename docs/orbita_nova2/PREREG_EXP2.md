# ÓRBITA-Nova-2 · Experimento 2 — Base compartida A,B + código c∈ℝ² (preregistro)

Rama `exp/orbita-nova2` (desde `51cf683`). Aprobado por Bruster (Nova-2 exp2; propuesta 2 de la página «papers de vanguardia y propuestas»: GEPS arXiv:2410.23889 + Sparse Memory Finetuning arXiv:2510.15103). Fecha: 2026-10-24 (America/Mexico_City, UTC−6).
Este documento se congela con su commit **antes** de generar cualquier dato DEV/CONFIRM del exp2. No modifica PREREG_EXP1 ni PREREG de Nova.

Regla dura: entre DEV y CONFIRM no se cambia código, hiperparámetros, umbrales, familias, métricas ni criterios. Sin calibración nueva.

## 0. Base

Se conservan los arreglos del exp1 (`DIAGNOSTICO_HARNESS.md`): **C3 activo** dentro de NOVA* (filtro en k pasos observados + free-run 16−k) y **LAT_EP con τ_LAT(k)** propio (0,0146557 / 0,0220203 / 0,0371425; λ0=1e-4). τ_C6 = 0,9386641928884383 (Nova DEV, congelado). Núcleo ÓRBITA congelado (pesos `artifacts/orbita/{dev,confirm}/weights_0x*.json`).

## 1. Hipótesis

Motivación (no probada): en NOVA, C4 ajusta por episodio U,V ∈ ℝ^{8×2} (32 parámetros) con k∈{2,4,8} pasos; con k pequeño sobreajusta. GEPS sugiere que una base compartida entre entornos más un código pequeño por entorno generaliza mejor con pocas observaciones; Sparse Memory Finetuning sugiere que consolidar sólo lo específico de dinámicas nuevas reduce olvido.

- **E1 (primaria)**: NOVA_BASE reduce el error en familias nuevas {8,10,11} frente a NOVA (base corregida) con **k=2**.
- **E2 (secundarias)**: lo mismo con k=4 y k=8 (por separado; sin corrección múltiple; no vetan E1).
- **E3 (no-daño)**: NOVA_BASE no pierde más de δ=0,05 de acc16 en familias conocidas 0–7 frente a NOVA, k=2/4/8.
- **E4 (aprende con experiencia)**: NOVA_BASE(n=2400) − NOVA_BASE(n=0) < 0 en nuevas, log, k=2 (reportada; no veta).
- **E5 (sparse)**: NOVA_BASE − NOVA_BASE−sparse en conocidas acc16 y nuevas log, k=2/4/8 (reportada; no veta).

## 2. Brazo nuevo NOVA_BASE (definición congelada)

ΔK = A·diag(c)·Bᵀ, A,B ∈ ℝ^{8×2} compartidos (columnas ortonormales), c ∈ ℝ² por episodio. Se implementa como U = A·diag(c), V = B, reutilizando `lr_step`, `arbitrate`, `gate_nodamage` de NOVA (rango 2).

**Pool**: cada episodio del flujo cuyo ΔK de NOVA (C4 GN/LM con prior C6, `fit_c4`) es aceptado por C5 + no-daño (exactamente los ítems que entran al buffer de NOVA) se añade, con su firma (θ0) y la etiqueta known (fam<8), a un pool acumulativo (no se vacía).

**Consolidación de la base** (en el mismo calendario que C6: cada N=50 episodios y en cada checkpoint):
- Selección (versión *sparse*, = NOVA_BASE): clustering greedy del pool completo por firma con τ_C6 (mismo algoritmo que `consolidate`); se usan sólo los ítems de clusters con fracción de conocidas < 0,5. Versión NOVA_BASE−sparse: todos los ítems del pool.
- Con los ΔK_i (8×8) seleccionados: A = 2 autovectores principales de Σ ΔK_i ΔK_iᵀ, B = 2 autovectores principales de Σ ΔK_iᵀ ΔK_i (iteración de potencia con deflación, 300 iter., inicio determinista); signo fijado para que la mayor componente en módulo sea positiva. Código c_i = diag(Aᵀ ΔK_i B); c̄ = media de c_i.
- Si no hay ítems seleccionados (o antes de la primera consolidación): no hay base → el brazo devuelve ÓRBITA+C3 (no-adaptación, C7).

**Por episodio**: c = MAP GN/LM (`fit_gn`, 12 iter.) sobre los k pasos con prior c̄ y λ = 1e-6 (= λ_C4), inicio en c̄; C5 = `arbitrate` en (k_fit,k_val) con ε=0,5 + `gate_nodamage`, idéntico protocolo a NOVA (incluida su fuga conocida: así la única diferencia frente a NOVA es la parametrización/base). Acepta → free-run con ΔK = A diag(c) Bᵀ; si no → ÓRBITA. Ambos con C3 como NOVA.

## 3. Datos, semillas y métrica

Idénticos a exp1: `gen_stream` n=2400 (80 % conocidas 0–7 / 20 % nuevas {8,10,11}, k∼U{2,4,8}), H = 16 episodios/familia, checkpoints n ∈ {0,150,300,600,1200,2400}.
Semillas: DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F`.
**Métrica primaria**: por seed, mediana de `log(rel_err h16)` sobre episodios H de nuevas {8,10,11}, n=2400; menor = mejor. Secundarias: acc16 (rel_err<0,05) conocidas/nuevas, known_med_log, curva n, tamaño del pool y fracción seleccionada.

## 4. Brazos

ÓRBITA, ÓRBITA+C3, LAT_EP, NOVA, NOVA−C3 (como exp1), **NOVA_BASE** (§2, sparse), **NOVA_BASE−sparse** (§2, sin selección). Los brazos bayesianos del exp1 no se corren.

## 5. Análisis y criterios

Contraste = media sobre 16 seeds de la diferencia pareada por seed; IC95 bootstrap pareado percentil, 10 000 réplicas, semilla 12345 (`scripts/analyze_nova2_exp2.py`). δ = 0,05.

- **E1**: NOVA_BASE − NOVA, nuevas, log, k=2: **aceptada** si IC95 sup < 0 en DEV **y** CONFIRM; **refutada** si en CONFIRM IC95 inf > 0; inconclusa en otro caso.
- **E2**: igual con k=4, k=8.
- **E3**: NOVA_BASE − NOVA, conocidas acc16, k=2/4/8: IC95 inf > −δ en ambas fases.
- **E4, E5**, controles NOVA − NOVA−C3, NOVA − LAT_EP, NOVA − ÓRBITA: reportados.

**Éxito del exp2** ⇔ E1 aceptada ∧ E3 aceptada. `RESULTADOS_EXP2.md` se publica en cualquier caso (negativos incluidos).

## 6. Ejecución

Binario `orbita nova2-exp2`; script `orbita/scripts/nova2_exp2.sh` (setsid nohup, `nice 10`, JOBS=7, reanudable por `.done` por seed). Salida `artifacts/orbita/nova2_exp2/{smoke,dev,confirm}/`. Orden: smoke (no analizado) → DEV → CONFIRM, mismo binario y commit. Sin `run_v6`, sin tocar `main`, sin force push.
