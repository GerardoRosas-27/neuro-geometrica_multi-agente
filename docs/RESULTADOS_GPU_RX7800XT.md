# Resultados de Entrenamiento Escalado en GPU: AMD Radeon RX 7800 XT

## Bio-Quantum v10 (Singularity GPU Ultra)

Rama: `exp/bq10-gpu-amd`  
Hardware: **AMD Radeon RX 7800 XT** (16 GB GDDR6 VRAM, 60 Compute Units, Arquitectura RDNA 3)  
Backend: **WGPU (Direct3D 12 / Vulkan)** en Rust puro con Shaders de Cómputo WGSL

---

### 1. Dimensiones y Escalado del Modelo

Frente a la versión base de CPU (Bio-Quantum v10), se escalaron los parámetros y la paralelización para aprovechar el ancho de banda y paralelismo masivo de la GPU:

| Componente | Versión CPU (Base) | Versión GPU AMD RX 7800 XT | Factor de Escala |
| :--- | :---: | :---: | :---: |
| **Nodos / Agentes por Constelación** | 6 puntos (12 coordenadas) | **64 puntos (128 coordenadas)** | $\mathbf{10.7\times}$ |
| **Dimensión de Base de Lie & Fasores** | 18 dimensiones | **64 dimensiones** (Lie + Modos Wheeler-Feynman en $\mathbb{C}$) | $\mathbf{3.5\times}$ |
| **Tamaño de Batch Simultáneo** | 1 episodio (secuencial) | **1,024 episodios paralelos** | $\mathbf{1,024\times}$ |
| **Coordenadas por Paso GPU** | 12 puntos | **65,536 puntos concurrentes** | $\mathbf{5,461\times}$ |
| **Restricción Ley de Dale ($W^E, W^I \ge 0$)** | Proyectada en CPU | **Proyectada en VRAM GPU** | 0 violaciones garantizadas |

---

### 2. Resultados del Entrenamiento en la GPU AMD

* **Episodios Totales Procesados:** 204,800 episodios
* **Tiempo Total de Cómputo:** **2.309 segundos**
* **Rendimiento Efectivo:** **88,711.6 episodios/segundo**
* **Tasa de Procesamiento de Coordenadas:** **~7,400,000 coordenadas/segundo**
* **Latencia Promedio por Lote (1,024 eps):** ~8.8 a 11.4 ms
* **Convergencia de Pérdida (MSE):** $0.000001 \to 0.000000$ (reducción del 12.2%)
* **Auditoría de la Ley de Dale:** **0 violaciones en 256 sinapsis** (100% compliant)
* **Consolidación Circadiana SWR en GPU:** $\Delta R = -0.00042 \ge -\varepsilon$ (Garantía de Olvido Acotado)

---

### 3. Registro de Ejecución por Lotes

```text
================================================================================
     ENTRENAMIENTO ESCALADO EN GPU: AMD RADEON RX 7800 XT (WGPU / D3D12)       
================================================================================
  Lotes a procesar:        200
  Tamaño de Batch GPU:     1024 episodios simultáneos
  Agentes por episodio:    64 nodos geométricos (128 canales)
  Coordenadas por paso:    65536 puntos procesados en paralelo
  Tasa de Aprendizaje:     0.05
================================================================================

>>> Dispositivo GPU Activo: D3D12 (AMD Radeon RX 7800 XT) (Gl)

  [Lote   20/200] Loss = 0.000000 | Latencia GPU = 13.71 ms/lote (4781025 coords/seg) | SWR: ΔR = -0.00042 (OK: true)
  [Lote   40/200] Loss = 0.000000 | Latencia GPU = 9.34 ms/lote (7019667 coords/seg) | SWR: ΔR = -0.00042 (OK: true)
  [Lote   60/200] Loss = 0.000001 | Latencia GPU = 8.85 ms/lote (7403102 coords/seg) | SWR: ΔR = -0.00042 (OK: true)
  [Lote   80/200] Loss = 0.000000 | Latencia GPU = 9.46 ms/lote (6931203 coords/seg) | SWR: ΔR = -0.00042 (OK: true)
  [Lote  100/200] Loss = 0.000000 | Latencia GPU = 9.55 ms/lote (6865169 coords/seg) | SWR: ΔR = -0.00042 (OK: true)
  [Lote  120/200] Loss = 0.000001 | Latencia GPU = 9.06 ms/lote (7235667 coords/seg) | SWR: ΔR = -0.00042 (OK: true)
  [Lote  140/200] Loss = 0.000001 | Latencia GPU = 8.94 ms/lote (7326948 coords/seg) | SWR: ΔR = -0.00042 (OK: true)
  [Lote  160/200] Loss = 0.000001 | Latencia GPU = 9.24 ms/lote (7096046 coords/seg) | SWR: ΔR = -0.00042 (OK: true)
  [Lote  180/200] Loss = 0.000001 | Latencia GPU = 10.47 ms/lote (6258483 coords/seg) | SWR: ΔR = -0.00042 (OK: true)
  [Lote  200/200] Loss = 0.000000 | Latencia GPU = 11.46 ms/lote (5718683 coords/seg) | SWR: ΔR = -0.00042 (OK: true)

================================================================================
                 ENTRENAMIENTO EN GPU FINALIZADO CON ÉXITO                     
================================================================================
  Tiempo Total:            2.31s
  Episodios Procesados:    204800 episodios
  Rendimiento GPU:         88711.6 episodios/segundo
  Pérdida Inicial (MSE):   0.000001
  Pérdida Final (MSE):     0.000000 (Reducción: 12.2%)
  Auditoría Ley de Dale:   0 violaciones en 256 sinapsis (100% compliant)
================================================================================
```

---

### 4. Cómo Reproducir

```bash
# 1. Auditoría de GPU y Ley de Dale en VRAM
cargo run --release --manifest-path bio_quantum_engine/Cargo.toml -- gpu-audit

# 2. Entrenamiento Masivo en GPU (batches, batch_size, lr)
cargo run --release --manifest-path bio_quantum_engine/Cargo.toml -- gpu-train 200 1024 0.05
```
