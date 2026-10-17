# Resultados E0–E5: memoria por capas (ÓRBITA-v1 congelada ⇄ CDT-RQM-epr con sueño)

Prerregistro: `docs/preregistro_memoria_capas.md` (4282693, publicado 12:59 CST antes de escribir/ejecutar código; humo sólo con
seed 0x5EED, descartada). Código: `orbita/src/mem.rs`, `orbita/llm/mem.py`, `mem_chain.sh` (E0–E2), `mem_chain2.sh` (E3–E5).
Tablas completas: `artifacts/orbita/mem/analyze_{dev,confirm}.log`, `analyze2_{dev,confirm}.log`, logs por seed `artifacts/orbita/mem/{dev,confirm}/`.
16 seeds DEV + 16 CONFIRM; IC95 bootstrap pareado (10 000, semilla 12345). **`ds.test` no se lee en ningún punto** (0 queries a TEST);
el contador interno de v4 registra lecturas del almacén CDT/replay propias del sueño (≈9–12 mil por seed, no son TEST).
Coste real: ≈12 s/seed (etapa 1) y <1 s/seed (etapa 2), 1 núcleo; ejecutado 13:01–13:15 CST. Gemma no se ejecutó (supuesto
declarado: el encoder entrega familia base + p).

## Veredicto
- **Kill criterion NO se cumple**: la consolidación supera al buffer crudo con el mismo presupuesto en las tres variantes y en ambas fases
  (CONFIRM: CDT − RAW +0,1034 [+0,0796, +0,1274]; CDT_NP +0,0961 [+0,0601, +0,1317]; CDT_GEN +0,3490 [+0,3217, +0,3751]; DEV igual signo).
  Por eso se ejecutaron E3–E5 según el prerregistro. **Advertencia**: RAW_BUFFER usa el predictor `nearest_answer` de v4 (desplazamiento
  del vecino), que es débil para rotaciones; el kill test es por tanto poco exigente.
- **Hipótesis principal E2 rechazada**: ninguna memoria supera al mejor control; todas quedan por **debajo** de ORACLE_LS y de NOCDT
  (CONFIRM: S CDT − max = −0,1033 [−0,1113, −0,0958]; CDT − NOCDT −0,0115 [−0,0153, −0,0078]). La memoria no aporta en esta tarea.
- CDT_GEN es la única que **aprende B** (acc4 B 0,3307 DEV / 0,3002 CONFIRM vs NOCDT 0,024 y ORACLE_LS 0,157) pero a costa de
  **olvido de A** (acc4 A 0,457 / 0,449 vs NOCDT 0,851): interferencia en el campo genérico compartido.
- CDT (consolidación por reglas + poda trusted) **poda todas las reglas B** (rulesB = 0: las reglas afines no ajustan B con residuo < 0,01),
  así que en B cae a ÓRBITA; CDT_NP retiene reglas B pero aproximadas (acc4 B 0,041 / 0,037) y empeora A.

## E0 saturación/novedad
AUROC(r → fallo acc4 de NOCDT) − 0,5 = +0,4983 [+0,4972, +0,4991] DEV, +0,4993 [+0,4990, +0,4995] CONFIRM (criterio cumplido).
Interpretación honesta: es casi trivial, porque NOCDT falla casi siempre en B y r es alto en B: la señal separa sobre todo A de B.
Novedad en vigilia B (CDT): ≈930/1024 episodios marcados nuevos; FPR en A_eval ≈ 0,01–0,03 (por seed en los logs).

## E1 validez de la tarea
ORACLE_LS acc4(B) = 0,1570 [0,1489, 0,1659] DEV / 0,1593 [0,1530, 0,1658] CONFIRM (< 0,9 ✓); NOCDT B − A = −0,8267 / −0,8277 (✓).
B no es resoluble por ajuste afín ni por ÓRBITA: la tarea sí exige memoria no afín.

## E2 (S = ½·(acc4 A + acc4 B))
| condición | S DEV | S CONFIRM |
|---|---|---|
| NOCDT | 0,4375 [0,4300, 0,4446] | 0,4371 [0,4297, 0,4438] |
| ORACLE_LS | 0,5306 [0,5266, 0,5344] | 0,5289 [0,5242, 0,5344] |
| ORACLE_ORB | 0,4385 | 0,4392 |
| SHUF = CORRUPT | 0,4375 | 0,4371 |
| CDT | 0,4269 [0,4201, 0,4336] | 0,4256 [0,4174, 0,4334] |
| CDT_NP | 0,3665 | 0,3678 |
| CDT_GEN | 0,3939 [0,3725, 0,4152] | 0,3747 [0,3503, 0,3972] |
| RAW_CDT / RAW_CDT_NP / RAW_CDT_GEN | 0,3437 / 0,2738 / 0,0287 | 0,3222 / 0,2717 / 0,0257 |
SHUF y CORRUPT quedan sin reglas tras la poda trusted y coinciden con NOCDT.

## E3 disparo por saturación vs intervalos fijos (variante CDT; elegida por tener la mayor S entre memorias, el prerregistro decía «ganadora» y no hubo ganadora)
ΔS = 0,0000 [0,0000, 0,0000] en ambas fases: **nulo trivial**, porque la memoria B se poda entera en cualquier horario de sueño.

## E4 CDT → códec de campo → texto (**el traductor es un banco de frases**, no generación libre)
FieldEncoder mínimo (12 aristas, fasores unitarios) en `mem.py`; banco de 2430 frases, plantillas pares en DEV e impares en CONFIRM; el
lector (palabras clave + `numnorm`) interpreta bien las 2430 frases de cada banco (0 errores).
Familia recuperada: CODEC 0,1579 = CODEC_SHUF 0,1579; CODEC_BANKPERM 0,0981 (DEV) / 0,0946 (CONFIRM). |Δp|: CODEC 0,0695 / 0,0701,
BANKPERM 0,7065 / 0,7074. **E4: CODEC − max(controles) = 0,0000 [0,0000, 0,0000] → no se cumple.** El códec transmite fielmente lo que
la memoria tiene (p bien recuperado), pero la memoria CDT no contiene slot B, así que la familia B nunca sale; 0,1579 ≈ fracción de episodios A
(96 de 608) en la evaluación.

## E5 poda
Poda trusted − sin poda: +0,0603 [+0,0557, +0,0657] DEV, +0,0578 [+0,0481, +0,0673] CONFIRM. Poda trusted − poda aleatoria del mismo
tamaño: +0,0535 [+0,0423, +0,0637] / +0,0539 [+0,0451, +0,0630]. La poda trusted es mejor que no podar o podar al azar, pero sólo porque
deja paso a ÓRBITA: CDT sigue por debajo de NOCDT.

## Conclusión
La arquitectura, con los mecanismos existentes, no aporta: las reglas afines por familia no pueden representar las familias no afines y la poda
las elimina; la consolidación genérica sí aprende B pero olvida A. Siguiente paso razonable (no ejecutado): slots genéricos **separados** por familia
nueva (sin campo compartido), y un RAW_BUFFER con un predictor más fuerte para que el kill test sea exigente.
