# ECO — Preregistro

**Sistema.** ÓRBITA-v1 congelado predice ŷ y sorpresa s. Corrector Dφ (receta v4 §16 E3, gated_h, hidden 136) con c=[ŷ, s, ŷ−q] → y=ŷ+Δ. Consolidación en "sueño" CDT enmascarada (episodios con error de ajuste del maestro ≥1% no se consolidan; réplica 0.5 de maestro confiable, TRUST_TAU 0.01). Cada 24 episodios un sueño (8 sueños, 2000 pasos, batch 32, lr 3e-3→3e-5 coseno). Checkpoint previo a cada sueño; puerta no-daño (pérdida de puerta del stream baja Y pérdida de familias conocidas ≤ +0.05), si no → rollback (hash verificado). LAT (λ0=1e-4, umbral τ_LAT=0.0220) con árbitro: Δ sólo se usa si es estrictamente mejor en los k=4 pasos observados; retroalimenta a ÓRBITA.

**Stream.** 48 episodios por familia nueva {8,10,11} + 48 conocidos, barajados (192).

**Primaria.** Área prequencial bajo la curva de error (log) en familias nuevas, ECO vs ÓRBITA+LAT. H1 aceptada si IC95 superior de (ECO−ÓRBITA+LAT) < 0 en DEV **y** CONFIRM; refutada si IC95 inferior > 0 en CONFIRM; si no, no concluyente.
**Secundaria.** E33 (rotar/reflejar, h=16): ECO ≥ ÓRBITA, IC95 inferior de la diferencia > −δ.
**No-daño (veto).** acc16 held-out conocidas: ECO−ÓRBITA y ECO−ÓRBITA+LAT con IC95 inferior > −δ.
**Controles.** ÓRBITA sola, sin sueño, sin rollback, CDT barajado, sólo éxitos, réplica completa sin máscara (E2), MLP misma receta (M3).
**Estadística.** Bootstrap pareado por semilla, 10000 réplicas, semilla 12345, δ=0.05 (`scripts/analyze_eco.py`).
**Reglas.** 0 consultas TEST (assert). DEV 0xA400–0xA40F, luego CONFIRM 0xB400–0xB40F sin reajuste. Smoke sólo en semillas fuera de esos rangos.

**Desviaciones declaradas.** (1) Presupuesto de sueño 2000 pasos vs 32000×8 de E3. (2) El contexto usa p̂ de reid (estela usaba e.p verdadero). (3) E33 lee el iterable TEST v4 como hace `orbita eval` (rotación/reflexión, sin contar como consulta de selección).
