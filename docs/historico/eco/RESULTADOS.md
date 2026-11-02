# ECO — Resultados (DEV + CONFIRM, sin reajuste)

Análisis: `python3 scripts/analyze_eco.py` → `artifacts/eco/analysis.json`. 16+16 semillas, bootstrap pareado 10000, semilla 12345, δ=0.05. Asserts: 0 consultas TEST y paridad `eco_zero`≡ÓRBITA+LAT (0 discrepancias) en las 32 semillas. ~178 s/semilla; ~9 min por fase con 7 jobs.

| Criterio | DEV media [IC95] | CONFIRM media [IC95] | Veredicto |
|---|---|---|---|
| Primaria: AUC log-error nuevas, ECO − ÓRBITA+LAT (<0 = mejor) | 0.0000 [0, 0] | 0.0000 [0, 0] | **NO CONCLUYENTE — H1 no aceptada** (ECO idéntico a ÓRBITA+LAT) |
| No-daño: acc16 conocidas, ECO − ÓRBITA | +0.076 [0.058, 0.095] | +0.088 [0.076, 0.101] | OK |
| No-daño: acc16 conocidas, ECO − ÓRBITA+LAT | 0 [0, 0] | 0 [0, 0] | OK |
| Secundaria E33 h16: ECO − ÓRBITA | +0.152 [0.103, 0.205] | +0.190 [0.148, 0.231] | OK (PASS 16/16 ambos brazos) |

Controles (AUC nuevas, ECO − control): ÓRBITA sola +0.530 [0.453, 0.609] DEV / +0.567 [0.502, 0.642] CONFIRM (ÓRBITA sola tiene menor log-error AUC en nuevas que ÓRBITA+LAT); sin-sueño, CDT barajado, sólo-éxitos, E2, M3: 0 exacto; sin-rollback: −8e-5 [−2.4e-4, 0] DEV, 0 CONFIRM.

## Negativo honesto
El corrector Δ **nunca actuó**: 0/256 sueños aceptados en DEV+CONFIRM y uso de Δ en nuevas = 0. De 264 sueños (incl. smoke), 262 no redujeron la pérdida de puerta del stream y empeoraron la de conocidas más de +0.05; 2 bajaron la del stream pero dañaron conocidas. El árbitro además nunca prefirió Δ. Por tanto todas las ventajas observadas (no-daño, E33) se deben a LAT, no a ECO; los controles no pueden discriminar porque ningún brazo con sueño consolida nada útil.
Causas plausibles (no probadas): presupuesto de sueño 2000 pasos (vs 32000×8 en E3), pocos episodios confiables por sueño (~6), puerta estricta. Cualquier cambio sería un experimento nuevo con preregistro nuevo; aquí no se reajustó nada.
