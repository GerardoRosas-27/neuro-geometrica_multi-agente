# Encoder-sonda hacia el sustrato de campo

**Fecha:** 13 de septiembre de 2026  
**Rama:** `exp/campo-sin-tokens` (encoder traído de `exp/encoder-campo`)  
**Módulo:** `src/field_encoder.rs`

## Decisión

- **No hay Llama 2** en este repo (cero hits). No se finge.
- Las **capas desacopladas** que sí existen son Gemma 2 2B (26 capas, LRC / T2.1). El encoder reutiliza esa geometría como **rejilla de escritura**, no como modelo.
- El LLM no es el sustrato. Escribe un patrón en `z` y se calla.

Máscara por defecto: apaga las 5 capas caras de T2.1 (`2, 3, 4, 5, 17`). 21/26 capas escriben el campo.

Atractores de cluster (v2, geométricos): hemisferio 6+6, meridiano, polos/ecuador. No el corte `e < 4`.

## Cómo aprende (sin next-token)

1. Gemma 2 congelada (GGUF si está) tokeniza y entrega RMS de 26 capas con máscara T2.1. Sin GGUF, misma forma 26-d de huella. **Los n-gramas no son la capa lingüística.**
2. Mapa lineal `26 RMS → 12 fasores`. Gemma no recibe gradiente.
3. Regla delta hacia el atractor del cluster. Cada 4 pasos, EP.
4. Decoder rígido: `z` → banco {frío, calor, animales}. No decodifica palabras.
5. La región oculta se apaga en `z` **y** `z_past`. Cero cross-entropy. Los tokens no entran en `FieldState`.

## Comando

```powershell
cargo test --release --lib field_encoder -- --nocapture
cargo run --release --bin native_field_substrate_experiment -- --no-train
```

Progreso por dataset: `data/field_substrate_training/encoder-corpus/` (`encoder-latest.json`, `checkpoints/encoder-step-*.json`). Resume atómico. Ctrl+C persiste y sale.

```powershell
cargo run --release --bin native_field_substrate_experiment -- --encoder-steps 480 --encoder-dir data/field_substrate_training/encoder-corpus
```

## Resultados (Gemma 2 2B GGUF congelada, 120 pasos, semilla `0xE4C0`, release)

Capa lingüística = hidden del último token (T2.1). Mapa lineal → 12 fasores. Decoder rígido al banco {frío, calor, animales}.

| métrica | antes | después |
|---|---:|---:|
| similitud mismo cluster | 0,764 | **0,838** |
| similitud otro cluster | 0,773 | **0,072** |
| margen (mismo − otro) | −0,009 | **+0,767** |
| decoder banco (train) | 0,33 | **1,00** |
| decoder holdout | 0,67 | 0,33 |
| recon (`z` y `z_past` tapados) | 0,886 | 0,942 |

El n-gram/hash **no basta**: decoder train 0,33→0,56 y holdout plano. Gemma congelada sí: el banco de entrenamiento pasa de azar a acierto perfecto. El holdout no generaliza (el mapa lineal memoriza las 9 frases). Completar región oculta sigue sin mejorar. 9/9 tests verdes (el de GGUF se salta si no hay modelo).

## Qué no reclama

No GGUF, no RMS reales de Gemma, no Llama 2, no ventaja vs transformer, no preprint. El decoder rígido → palabras queda como sonda (`read_rigid_signature`); no se entrena un vocabulario.
