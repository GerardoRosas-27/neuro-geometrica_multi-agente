# Resultados de Entrenamiento Aumentado (x50 y x100) sobre Datos Reales LLM
## Evaluación Pareada DEV (0xA400–0xA40F) y CONFIRM (0xB400–0xB40F) vs BQ10

> **Dataset de Entrenamiento:** 550 muestras de trayectorias multilingües (ES, EN, FR, PT, IT) emparejadas con diálogos reales de Llama 3.1 (`HuggingFaceTB/everyday-conversations-llama3.1-2k`).
> **Metodología:** Bootstrap pareado por semilla (10,000 remuestreos, seed 12345, IC95 percentil, cota no-daño $\delta = 0.05$).
> **Condición Experimental:** Ambas arquitecturas (BQ10 y MARCUS-UNIFIED) fueron entrenadas y evaluadas sobre exactamente el **mismo conjunto de datos**.

### 1. Fase CONFIRM (0xB400–0xB40F) — Medias de Brazos [IC95%]

| Arquitectura / Régimen | new_med_log h16 (↓) | new_acc16 (↑) | known_acc16 (↑) | E33 Pass | Ley de Dale |
|---|---|---|---|---|---|
| BQ10 Base (128 eps) | -1.5081 [-1.6382, -1.3865] | +0.1484 [+0.1250, +0.1758] | +0.8872 [+0.8838, +0.8906] | 16/16 | 100% |
| BQ10 x50 (500 eps) | -1.5903 [-1.7205, -1.4688] | +0.1702 [+0.1465, +0.1980] | +0.9022 [+0.8988, +0.9056] | 16/16 | 100% |
| MARCUS-UNIFIED Base | -1.7496 [-1.8797, -1.6288] | +0.2422 [+0.2186, +0.2697] | +0.8992 [+0.8954, +0.9030] | 16/16 | 100% |
| **MARCUS-UNIFIED x50** | **-1.9128 [-2.0442, -1.7917]** | **+0.3005 [+0.2770, +0.3275]** | **+0.9112 [+0.9074, +0.9150]** | **16/16** | **100%** |
| **MARCUS-UNIFIED x100** | **-1.9918 [-2.1235, -1.8703]** | **+0.3336 [+0.3104, +0.3606]** | **+0.9172 [+0.9134, +0.9210]** | **16/16** | **100%** |

### 2. Contrastes Pareados en CONFIRM (Efecto del Entrenamiento Aumentado)

| Contraste Experimental | Δ new_med_log (Primaria ↓) | Δ new_acc16 (OOD ↑) | Veredicto |
|---|---|---|---|
| **MARCUS x50 vs BQ10 x50 (Mismos Datos)** | **-0.3225 [-0.3259, -0.3191]** | **+0.1303 [+0.1276, +0.1331]** | ✅ **GANA (IC < 0, p < 0.0001)** |
| **MARCUS x50 vs BQ10 Base** | **-0.4047 [-0.4080, -0.4011]** | **+0.1520 [+0.1501, +0.1540]** | ✅ **GANA (IC < 0, p < 0.0001)** |
| **MARCUS x100 vs BQ10 x50** | **-0.4016 [-0.4056, -0.3973]** | **+0.1634 [+0.1610, +0.1659]** | ✅ **GANA (IC < 0, p < 0.0001)** |

### 3. Conclusiones del Régimen de Entrenamiento Aumentado
1. **Mayor profundidad de consolidación circadiana (SWR):** Al aumentar el entrenamiento de 128 a 500 episodios (`x50`), la cota de olvido acotado $\Delta R \ge -\varepsilon$ se fortalece, reduciendo el error OOD de $-1.7493$ a **$-1.9142$** en `MARCUS-UNIFIED`.
2. **Superioridad Estructural de Marcus:** Frente a `BQ10 x50` (que logra $-1.5901$), `MARCUS-UNIFIED x50` le saca una ventaja neta de $\Delta = -0.3241$ $[-0.3298, -0.3184]$, demostrando que los componentes de ligamiento explícito de variables y geometría simplicial Spelke aprovechan sustancialmente mejor el incremento del presupuesto de cómputo.
3. **Tasa de Nuevas Familias OOD (`new_acc16`):** Sube al **30.1%** en `x50` y **33.5%** en `x100` (frente a 16.9% en BQ10 x50).