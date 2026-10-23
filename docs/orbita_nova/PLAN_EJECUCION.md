# ÓRBITA-Nova — Plan de ejecución y preregistro (borrador)

Este borrador **no es el preregistro congelado**. Antes de DEV se congela en `docs/orbita_nova/PREREG.md` con commit/push, y después no se ajusta nada.
Seeds: DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F` (n=16). Bootstrap pareado por seed, 10 000 réplicas, semilla 12345, IC95, δ=0,05. 0 queries a TEST.
Pesos ÓRBITA congelados (sha256 antes/después). Conocidas 0–7, nuevas 8–11.

## Métricas
- **Primaria**: Δ log(rel_err h16) mediano por episodio (continua, menor es mejor), nuevas 8–11, k=4.
- Secundarias: acc16 (umbral 0,05), rel_err h32/h64, k=2 y 8, AUROC sorpresa, tasa de recuperación, pureza de cluster, tamaño de biblioteca, coste (s/episodio).
- Olvido: acc16 y log rel_err en conocidas 0–7 a n=0 vs n=2400.

## Brazos
ÓRBITA (OBS), ÓRBITA+C3 (sólo filtro), LAT_EP, LAT, NOVA (completo), NOVA−C3, NOVA−C5, NOVA−C6 (sin biblioteca), NOVA_SHUF (biblioteca con firmas permutadas entre clusters),
MLP (mismo nº de parámetros adaptables, misma inferencia GN), ORACLE_EP, ORACLE_FAMNET (techo).

## Hipótesis (aceptada si se cumple en DEV **y** CONFIRM; refutada si en CONFIRM IC95 entero del lado contrario; inconclusa en otro caso)
- N1: NOVA − LAT_EP, nuevas, primaria: IC95 < 0 (mejora).
- N2: NOVA n=2400 − NOVA n=0, nuevas: IC95 < 0 (aprende con experiencia = largo plazo).
- N3: no-daño: NOVA − ÓRBITA conocidas acc16, inf. IC95 > −δ; y NOVA n=2400 − n=150 conocidas inf. > −δ.
- N4: NOVA − NOVA_SHUF y NOVA − MLP, nuevas: IC95 < 0 (la estructura importa).
- N5: ablaciones NOVA − NOVA−C5 y NOVA − NOVA−C6: IC95 < 0.
- N6 (ruido): con ruido gaussiano σ∈{0,01;0,05}, NOVA no peor que ÓRBITA+C3 en conocidas (inf. > −δ).
- **Éxito de la rama** = N1 ∧ N2 ∧ N3 ∧ N4. **Fracaso** = N1 o N3 refutada, o N2 no aceptada (no hay aprendizaje a largo plazo). Se reporta igual en ambos casos.

## Fases
| fase | entregable | puerta para avanzar | estimación |
|---|---|---|---|
| 0. Factibilidad | `orbita/src/nova.rs` (`nova-eval`) con C3+C4+C5; prueba en episodios generados (no H) | C4 mejora a ÓRBITA en su propia distribución en ≥14/16 seeds; NIS de C3 calibrado (cobertura 95 % ±5) | 2–3 días dev; CPU < 1 h |
| 1. Suelo | tabla ORACLE_EP/FAMNET por familia nueva y k | excluir familias donde el techo está en suelo; fijar k_fit/k_val, umbrales | ½ día; < 1 h |
| 2. Preregistro | `PREREG.md` congelado (métricas, brazos, τ, r, umbral de recuperación, N_consolidación, mezcla del flujo) | commit empujado antes de generar datos DEV | ½ día |
| 3. DEV | flujo 2400 episodios × 16 seeds × brazos; análisis | sólo se calibran τ/umbral en H DEV familias 1–2 según prereg; sin tocar código después | ~6–10 h CPU (lat-eval ≈ 45 s/768 ep. por k; consolidación extra) |
| 4. CONFIRM | misma cadena, sin cambios | — | ~6–10 h CPU |
| 5. (opcional) Periferia | paráfrasis multilingües / bytes → mismo estado (distancia de contexto) | sólo si 4 tiene éxito; preregistro aparte | 2–3 días |
| 6. Informe | `docs/orbita_nova/RESULTADOS.md` (o `docs/lecciones/` si falla) | — | ½ día |

Total: ~1,5–2 semanas calendario; cómputo ~1 día CPU (sin GPU; Gemma sólo para contextos ya cacheados de experimentos previos si se reutilizan).
Cadena `setsid nohup` reanudable con `.done`, `nice 10`, sin compartir CPU con otras ramas (lección CT: tiempos contaminados).


## Estado fase 0 (actualizado)
- **PASS** 16/16 seeds (C4 vs ÓRBITA, mediana log rel_err h16); NIS media 0,933.
- Detalle: [`FASE0_RESULTADOS.md`](FASE0_RESULTADOS.md). Artefactos: `artifacts/orbita/nova_fase0/`.
- Siguiente puerta: fase 1 (ORACLE_EP/FAMNET) cuando se autorice; preregistro (fase 2) aún no congelado.
