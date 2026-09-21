# Encoder-sonda hacia el sustrato de campo

**Fecha:** 13 de septiembre de 2026  
**Rama:** `exp/encoder-campo` (parte de `exp/campo-sin-tokens`)  
**Módulo:** `src/field_encoder.rs`

## Decisión

- **No hay Llama 2** en este repo (cero hits). No se finge.
- Las **capas desacopladas** que sí existen son Gemma 2 2B (26 capas, LRC / T2.1). El encoder reutiliza esa geometría como **rejilla de escritura**, no como modelo.
- Se **reconstruye** el encoder en rama aparte: el LLM no es el sustrato. Escribe un patrón en `z` y se calla.

Máscara por defecto: apaga las 5 capas caras de T2.1 (`2, 3, 4, 5, 17`). 21/26 capas escriben el campo.

## Cómo aprende (sin next-token)

1. Features = sonda de 26 capas (huella determinista del texto; con GGUF serían RMS reales) + 16 n-gramas. Los n-gramas son papel, no ondas.
2. Mapa lineal `features → 12 fasores` (una arista del octaedro).
3. Regla delta local hacia el atractor de campo del cluster. De vez en cuando, Equilibrium Propagation: el `z` relajado enseña al mapa.
4. Cero cross-entropy, cero `vocab_size`.

## Comando

```bash
cargo test --release --lib field_encoder -- --nocapture
```

## Resultados (13 sep 2026, rustc 1.98.1, release, 120 pasos, semilla `0xE4C0`)

| métrica | antes | después |
|---|---:|---:|
| similitud mismo cluster | 0,810 | **0,882** |
| similitud otro cluster | 0,844 | **0,356** |
| margen (mismo − otro) | −0,034 | **+0,526** |
| recon enmascarando `z` y `z_past` | 1,369 | 2,079 |
| pérdida de campo L | 1,419 | 2,129 |

El encoder **sí habla geometría**: agrupa frío / calor / animales en `z`. La predicción de región oculta *sin* copiar la respuesta en `z_past` no mejora (el Hopfield local no completa fases de alta frecuencia). Eso es honesto, no un LLM.

## Qué no reclama

No GGUF, no RMS reales de Gemma, no Llama 2, no ventaja vs transformer, no preprint. El decoder rígido → palabras queda como sonda (`read_rigid_signature`); no se entrena un vocabulario.
