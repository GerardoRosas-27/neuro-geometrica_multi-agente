# Resultados — Opción A (D3-num) vs Opción B (ÓRBITA-robusta)

Prerregistro: `docs/preregistro_d3_vs_robusta.md` (commit 1c14c8e, δ = 0,05), comprometido antes de cualquier resultado. Base: D2 (`docs/resultados_encoder_d2.md`, f7d1385). Ejecución 2026-10-16/17 (CDMX) con `chain.sh` (`setsid nohup`, nice 0). Logs y archivos `.done` en `/workspace/orbita-data/llm_d3/` (`dev.done`, `confirm.done`). Artefactos en `artifacts/orbita/llm_d3/`: `summary_{dev,confirm}.md` (idénticos a `analyze_{dev,confirm}.log`), `chain.log`, y por fase `evalA_<seed>.csv`, `evalB_<seed>.csv`, `queries{A,B}_<seed>.txt` y `B_eval/` (protocolo §16 de ÓRBITA-B). Nada cambió entre DEV y CONFIRM. **Auditoría: 0 queries** en ambas fases; **parse_fail = 0** en ambas fases (el parser encontró dos números en todos los textos).

**Coste real (fine-tune de B, 64 000 pasos, 8 seeds en paralelo, 1 hilo cada una):** DEV de 23:01:27 a 23:35:44, media 1027 s por seed (máx. 1674 s); CONFIRM de 23:36:00 a 00:40:55, media 1946 s por seed (máx. 2390 s). Más que los ~6,4 min estimados en el prerregistro (CPU compartida entre 8 procesos). La evaluación de cada fase tardó menos de 20 s.

## Cifras clave (media [IC95], 16 seeds)

| cantidad | DEV | CONFIRM |
|---|---|---|
| v1 + TRUE acc h16 | 0,2969 [0,2305, 0,3633] | 0,3164 [0,2344, 0,4023] |
| v1 + RAND acc h16 | 0,0286 [0,0208, 0,0371] | 0,0260 [0,0182, 0,0339] |
| v1 + D2 acc h16 | 0,1784 [0,1322, 0,2259] | 0,1543 [0,1061, 0,2051] |
| v1 + D1_POOL2 acc h16 | 0,1882 [0,1445, 0,2363] | 0,1882 [0,1582, 0,2194] |
| **v1 + D3NUM acc h16 (A)** | **0,2591 [0,1960, 0,3229]** | **0,2865 [0,2077, 0,3646]** |
| v1 + D3NUM acc h1 / h8 / h32 | 0,9147 / 0,6126 / 0,0664 | 0,8887 / 0,6016 / 0,0801 |
| v1 + TRUE acc h1 / h8 / h32 | 0,9844 / 0,6797 / 0,0898 | 0,9883 / 0,6836 / 0,0703 |
| B + TRUE acc h1 / h8 / h16 | 0,5664 / 0,0078 / 0,0000 | 0,5234 / 0,0195 / 0,0039 |
| **B + D2 acc h16 (B)** | **0,0000 [0,0000, 0,0000]** | **0,0039 [0,0013, 0,0072]** |
| B + D2 acc h1 / h8 | 0,2656 / 0,0072 | 0,2103 / 0,0098 |
| B + D1_POOL2 acc h16 | 0,0000 | 0,0000 |
| max\|p̂−p\| mediano D3NUM | 0,0036 [0,0034, 0,0037] | 0,0034 [0,0032, 0,0036] |
| B: acc a 1 paso TEST completo (§16, c real) | 0,8211 [0,8067, 0,8359] | 0,8176 [0,8096, 0,8262] |
| B: fracción de seeds con GATE E33 PASS | 0,0000 | 0,0000 |

## Contrastes prerregistrados (diferencia pareada, media [IC95])

| hipótesis | DEV | CONFIRM | veredicto |
|---|---|---|---|
| **H-A1** acc16 D3NUM − D2 (v1), IC95 > 0 | +0,0807 [+0,0417, +0,1224] | +0,1322 [+0,0658, +0,1986] | **Aceptada** (IC95 > 0 en ambas fases) |
| **H-A2** acc16 D3NUM − TRUE (v1), inf > −0,05 | −0,0378 [−0,0697, −0,0033] | −0,0299 [−0,0638, +0,0007] | **Inconclusa** (el límite inferior queda por debajo de −0,05 en ambas fases; no se refuta porque el superior en CONFIRM es +0,0007, no < −0,05) |
| **H-B1** acc16 B+D2 − v1+D2, IC95 > 0 | −0,1784 [−0,2259, −0,1322] | −0,1504 [−0,2018, −0,1016] | **Refutada** (IC95 < 0 en CONFIRM) |
| **H-B2** acc16 B+TRUE − v1+TRUE, inf > −0,05 | −0,2969 [−0,3633, −0,2305] | −0,3125 [−0,3945, −0,2305] | **Refutada** (superior −0,2305 < −0,05 en CONFIRM) |
| **H-B3** acc16 B+D2 − v1+TRUE, inf > −0,05 | −0,2969 [−0,3633, −0,2305] | −0,3125 [−0,3965, −0,2298] | **Refutada** (superior −0,2298 < −0,05 en CONFIRM) |

**H-C1 (descriptivo):**
- A: familia inferida por el teacher − 1/8 = +0,8027 [+0,7246, +0,8704] (DEV) y +0,7786 [+0,7227, +0,8288] (CONFIRM); AUROC − 0,5 = −0,0014 [−0,1252, +0,1240] (DEV, n = 15) y +0,0287 [−0,0727, +0,1338] (CONFIRM). La familia se recupera muy por encima del azar; la divergencia del teacher no predice fallos a h16 (IC95 cruza 0).
- B: familia − 1/8 = +0,7962 [+0,7181, +0,8639] (DEV) y +0,7682 [+0,7096, +0,8203] (CONFIRM); AUROC no evaluable en DEV (n = 0, porque B falla todos los casos a h16) y +0,0330 [−0,2634, +0,3295] en CONFIRM con solo n = 5 seeds evaluables.

## Regla de selección (aplicada tal como se prerregistró)

1. **Elegibilidad:** H-B2 queda refutada en CONFIRM (B pierde 0,3125 de acc16 con c real, mucho más que δ), así que **B no es elegible**.
2. **Primaria** (informativa, ya que B quedó fuera): acc16(A) − acc16(B) en CONFIRM = +0,2826 [+0,2044, +0,3600], IC95 > 0, que también daría la victoria a A. La secundaria (acc8: +0,5918 [+0,5234, +0,6589]) apunta igual.

**Ganadora: Opción A (D3-num)**: ÓRBITA-v1 congelada, familia del argmax de D2 y p leído por el parser de números.

## Lectura

- D3-num casi cierra la brecha con el contexto real: el error mediano en p cae de ~0,15 (D2) a ~0,003, y acc16 sube a 0,2865 en CONFIRM frente a 0,3164 con TRUE. La diferencia restante (−0,03) es compatible con un empate dentro de δ pero el IC95 no basta para afirmarlo (H-A2 inconclusa). Lo que queda probablemente viene de los errores de familia del argmax de D2.
- El fine-tune con ruido en p (B) degradó la dinámica iterada de ÓRBITA: con c real cae de 0,6836 a 0,0195 a h8 y a ~0 a h16, y a 1 paso sobre TEST completo se queda en 0,82 sin pasar el gate E33 en ninguna seed. Robustecer así el modelo rompe la iteración en lugar de tolerar el ruido del encoder.
- Advertencia: la lectura numérica de A depende de que los textos (plantillas 2–3) contengan los valores de p explícitos; con parse_fail = 0 eso se cumplió aquí, pero no dice nada de textos donde p no aparece como número.
