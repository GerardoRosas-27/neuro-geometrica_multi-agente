# Pre-registro Científico Formal: Evaluación DEV & CONFIRM
## Comparativa Pareada de MARCUS-UNIFIED vs. BQ10 y ATLAS-FW

> **Fecha de Pre-registro:** Noviembre 2026  
> **Arquitectura Evaluada:** `MARCUS-UNIFIED` (Crate `marcus_neurosymbolic`, rama `exp/marcus-neurosymbolic-architecture`)  
> **Líneas de Referencia en Main:**
>   1. **`BQ10 (bio-quantum v10)`** (Arquitectura Ganadora #1 de `main`, PR #48 / `integra/bio-quantum-v10-main`)
>   2. **`ATLAS-FW (ALL)`** (Arquitectura Ganadora #2 de `main`, PR #47 / `release/atlas-fw`)
>   3. **`CONSTELACIÓN v4`** (Línea Base canónica del sustrato v4, `fd80320`)
> **Diseño Experimental:** Evaluación pareada semilla por semilla con bootstrap de 10,000 remuestreos, intervalo de confianza percentil al 95% (IC95), umbral de no-daño $\delta = 0.05$.

---

## 1. Declaración de Hipótesis Pre-registradas

### Hipótesis Primaria H-PRIM: Superioridad en Error OOD Fuera de Distribución (`new_med_log`)
* **Formulación:** La integración de ligamiento algebraico de variables (Marcus 2001) y la barrera métrica de Cayley-Menger (Soft-Bellman Simplicial) sobre la base BQ10 reduce el error relativo mediano en escala logarítmica sobre familias nunca vistas a horizonte 16 ($k=4$).
* **Criterio Matemático:**
  $$\Delta_{\text{prim}} = \text{new\_med\_log}(\text{MARCUS-UNIFIED}) - \text{new\_med\_log}(\text{BQ10}) < 0$$
  con el límite superior del intervalo de confianza al 95% estrictamente negativo:
  $$\text{IC95}_{\text{sup}}(\Delta_{\text{prim}}) < 0.0 \quad \text{en DEV y en CONFIRM}$$
* **Criterio de Falsificación:** Si $\text{IC95}_{\text{sup}}(\Delta_{\text{prim}}) \ge 0.0$ en la fase ciega CONFIRM, la hipótesis primaria queda falsificada.

---

### Hipótesis Co-Primaria H-COPRIM: Superioridad frente a ATLAS-FW (`ALL`)
* **Formulación:** MARCUS-UNIFIED supera a la segunda mejor arquitectura de `main` (ATLAS-FW) en error OOD:
  $$\Delta_{\text{coprim}} = \text{new\_med\_log}(\text{MARCUS-UNIFIED}) - \text{new\_med\_log}(\text{ALL}) < 0$$
  con $\text{IC95}_{\text{sup}}(\Delta_{\text{coprim}}) < 0.0$ en DEV y CONFIRM.

---

### Hipótesis Secundaria H-ACC: Ganancia en Exactitud de Familias Nuevas (`new_acc16`)
* **Formulación:** La extracción explícita de transformaciones invariantes y ligamiento rol-rellenador sin diafonía aumenta la exactitud de predicción sobre trayectorias de familias nuevas:
  $$\Delta_{\text{acc}} = \text{new\_acc16}(\text{MARCUS-UNIFIED}) - \text{new\_acc16}(\text{BQ10}) > 0$$
  con $\text{IC95}_{\text{inf}}(\Delta_{\text{acc}}) > 0.0$.

---

### Hipótesis de No-Daño H-STAB: Preservación de Familias Conocidas y Estabilidad
* **No-Daño en Familias Conocidas:** La exactitud en familias conocidas no debe degradarse más allá de la cota de tolerancia $\delta = 0.05$:
  $$\Delta_{\text{known}} = \text{known\_acc16}(\text{MARCUS-UNIFIED}) - \text{known\_acc16}(\text{BQ10}) \ge -0.05$$
* **Estabilidad Dinámica E33:** Ausencia total de colapso de covarianza o divergencia numérica:
  $$\text{E33\_pass} = 16/16 \quad \text{en DEV y CONFIRM}$$
* **Integridad Numérica:** $0 \text{ NaN}$, $0 \text{ fallos de aserción}$, $0 \text{ consultas de test}$ (`test_queries = 0`).
* **Ley de Dale Estricta:** $0 \text{ violaciones}$ a la segregación $W^E \ge 0, W^I \ge 0$ en las 8,192 sinapsis bio-cuánticas.

---

## 2. Conjuntos de Semillas y Partición Canónica

Para garantizar comparabilidad absoluta y pareada contra los registros oficiales de `main`:

1. **Fase DEV (Desarrollo y Calibración):**
   - 16 semillas deterministas: `0xA400, 0xA401, 0xA402, 0xA403, 0xA404, 0xA405, 0xA406, 0xA407, 0xA408, 0xA409, 0xA40A, 0xA40B, 0xA40C, 0xA40D, 0xA40E, 0xA40F`.
2. **Fase CONFIRM (Evaluación Ciega sin Tuning de Hiperparámetros):**
   - 16 semillas deterministas disjuntas: `0xB400, 0xB401, 0xB402, 0xB403, 0xB404, 0xB405, 0xB406, 0xB407, 0xB408, 0xB409, 0xB40A, 0xB40B, 0xB40C, 0xB40D, 0xB40E, 0xB40F`.

---

## 3. Protocolo de Evaluación y Cálculo Estadístico

1. Cargar las métricas empíricas oficiales pareadas de cada semilla almacenadas en `artifacts/bio_quantum_v10/{bq_dev,bq_confirm}/metrics_0x*.json`.
2. Ejecutar la inferencia de MARCUS-UNIFIED sobre cada semilla utilizando los mismos embeddings y split de familias.
3. Para cada contraste pareado $\Delta_s = M_s(\text{MARCUS}) - M_s(\text{REF})$:
   - Generar 10,000 remuestreos bootstrap con reemplazo (semilla del generador `12345`).
   - Calcular media muestral y cuantiles $q_{0.025}$ y $q_{0.975}$ del estimador bootstrap.
4. Generar el reporte formal en `docs/marcus_neurosymbolic/RESULTADOS_DEV_CONFIRM.md`.
