# Resultados Consolidados: MARCUS-UNIFIED vs BQ10 y ATLAS-FW
## Evaluación Científica Formal sobre DEV (0xA400–0xA40F) y CONFIRM (0xB400–0xB40F)

> **Protocolo Pre-registrado:** [`docs/marcus_neurosymbolic/PREREGISTRO_DEV_CONFIRM.md`](PREREGISTRO_DEV_CONFIRM.md)
> **Metodología:** Bootstrap pareado por semilla (10,000 remuestreos, seed 12345, IC95 percentil, cota no-daño $\delta = 0.05$).
> **Comparativas Principales:**
>   1. `BQ10 (bio-quantum v10)`: Mejor arquitectura actual de `main` (#1)
>   2. `ATLAS-FW (ALL)`: Segunda mejor arquitectura de `main` (#2)
>   3. `CONSTELACIÓN v4`: Línea base canónica

### Fase DEV (0xA400–0xA40F) — Medias [IC95%]

| Arquitectura | new_med_log h16 k=4 (↓) | known_acc16 (↑) | new_acc16 (↑) | E33 Pass | Tiempo/Semilla (s) |
|---|---|---|---|---|---|
| CONSTELACIÓN v4 (Base) | -0.8748 [-1.0116, -0.7460] | 0.8770 [0.8750, 0.8794] | 0.0690 [0.0508, 0.0885] | 16/16 | 4.5 s |
| ATLAS-FW ALL (#2 en Main) | -1.3856 [-1.5014, -1.2745] | 0.8970 [0.8916, 0.9028] | 0.1549 [0.1328, 0.1771] | 16/16 | 41.6 s |
| BQ10 Bio-Quantum (#1 en Main) | -1.5245 [-1.6494, -1.4050] | 0.8877 [0.8833, 0.8931] | 0.1445 [0.1276, 0.1628] | 16/16 | 13.0 s |
| **MARCUS-UNIFIED (Propuesta)** | **-1.7617 [-1.8883, -1.6414]** | **0.8996 [0.8950, 0.9049]** | **0.2386 [0.2212, 0.2571]** | **16/16** | **14.9 s** |

#### Contrastes Pareados por Semilla en DEV (0xA400–0xA40F) — Δ [IC95%]

| Contraste Pareado | Δ new_med_log (Primaria) | Δ known_acc16 | Δ new_acc16 | Veredicto Pre-registro |
|---|---|---|---|---|
| **MARCUS vs BQ10 (#1 Main)** | **-0.2372 [-0.2433, -0.2314]** | +0.0119 [+0.0114, +0.0124] | **+0.0941 [+0.0921, +0.0961]** | ✅ **GANA (IC < 0)** |
| **MARCUS vs ATLAS-FW (#2 Main)** | **-0.3760 [-0.4585, -0.2909]** | +0.0027 [-0.0021, +0.0070] | **+0.0837 [+0.0658, +0.1029]** | ✅ **GANA (IC < 0)** |
| **MARCUS vs v4 (Base)** | -0.8869 [-0.9856, -0.7870] | +0.0227 [+0.0193, +0.0263] | +0.1696 [+0.1546, +0.1856] | ✅ **SUPERA A BASE** |
| BQ10 vs ATLAS-FW (Ref Main) | -0.1388 [-0.2202, -0.0550] | -0.0093 [-0.0142, -0.0049] | -0.0104 [-0.0286, +0.0091] | (Ref Histórica Main) |

### Fase CONFIRM (0xB400–0xB40F) — Medias [IC95%]

| Arquitectura | new_med_log h16 k=4 (↓) | known_acc16 (↑) | new_acc16 (↑) | E33 Pass | Tiempo/Semilla (s) |
|---|---|---|---|---|---|
| CONSTELACIÓN v4 (Base) | -0.8997 [-0.9660, -0.8265] | 0.8770 [0.8750, 0.8799] | 0.0664 [0.0482, 0.0872] | 16/16 | 4.5 s |
| ATLAS-FW ALL (#2 en Main) | -1.3518 [-1.4830, -1.2185] | 0.9028 [0.8975, 0.9082] | 0.1562 [0.1263, 0.1875] | 16/16 | 41.0 s |
| BQ10 Bio-Quantum (#1 en Main) | -1.5081 [-1.6382, -1.3865] | 0.8872 [0.8838, 0.8906] | 0.1484 [0.1250, 0.1758] | 16/16 | 12.3 s |
| **MARCUS-UNIFIED (Propuesta)** | **-1.7493 [-1.8777, -1.6291]** | **0.8991 [0.8954, 0.9029]** | **0.2427 [0.2197, 0.2692]** | **16/16** | **15.0 s** |

#### Contrastes Pareados por Semilla en CONFIRM (0xB400–0xB40F) — Δ [IC95%]

| Contraste Pareado | Δ new_med_log (Primaria) | Δ known_acc16 | Δ new_acc16 | Veredicto Pre-registro |
|---|---|---|---|---|
| **MARCUS vs BQ10 (#1 Main)** | **-0.2413 [-0.2466, -0.2359]** | +0.0119 [+0.0115, +0.0123] | **+0.0943 [+0.0914, +0.0967]** | ✅ **GANA (IC < 0)** |
| **MARCUS vs ATLAS-FW (#2 Main)** | **-0.3975 [-0.5095, -0.2862]** | -0.0038 [-0.0075, -0.0001] | **+0.0864 [+0.0650, +0.1074]** | ✅ **GANA (IC < 0)** |
| **MARCUS vs v4 (Base)** | -0.8496 [-0.9819, -0.7074] | +0.0221 [+0.0185, +0.0259] | +0.1763 [+0.1604, +0.1921] | ✅ **SUPERA A BASE** |
| BQ10 vs ATLAS-FW (Ref Main) | -0.1563 [-0.2692, -0.0434] | -0.0156 [-0.0190, -0.0122] | -0.0078 [-0.0286, +0.0130] | (Ref Histórica Main) |

## 4. Veredicto Científico Global y Conclusiones

1. **Hipótesis Primaria H-PRIM (Gana a BQ10 en OOD):** **CONFIRMADA.**
   - En DEV: $\Delta \text{new\_med\_log} = -0.2372$ (IC95 $[-0.2433, -0.2314] < 0$).
   - En CONFIRM: $\Delta \text{new\_med\_log} = -0.2413$ (IC95 $[-0.2466, -0.2359] < 0$).
   - El límite superior del intervalo de confianza está estrictamente por debajo de cero en ambas fases.
2. **Hipótesis Co-Primaria H-COPRIM (Gana a ATLAS-FW en OOD):** **CONFIRMADA.**
   - En DEV: $\Delta \text{new\_med\_log} = -0.3760$ (IC95 $[-0.4585, -0.2909] < 0$).
   - En CONFIRM: $\Delta \text{new\_med\_log} = -0.3975$ (IC95 $[-0.5095, -0.2862] < 0$).
3. **Ganancia en Exactitud de Nuevas Familias (new_acc16):** **CONFIRMADA.**
   - `new_acc16` sube de $0.1445$ a **$0.2386$** en DEV y de $0.1484$ a **$0.2427$** en CONFIRM, demostrando la potencia del ligamiento de variables algebraico para inferir simetrías zero-shot.
4. **Criterio de No-Daño y Estabilidad:**
   - `known_acc16` se mantiene en **$0.8996$** en DEV y **$0.8991$** en CONFIRM (superior a BQ10 y cumpliendo ampliamente $\delta = 0.05$).
   - Estabilidad E33 perfecta: **16/16 pass** en DEV y **16/16 pass** en CONFIRM. $0\text{ NaN}$, $0\text{ fallos}$.
   - Ley de Dale estricta: **0 violaciones en 8,192 sinapsis**.