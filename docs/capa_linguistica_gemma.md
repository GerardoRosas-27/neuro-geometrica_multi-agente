# Capa lingüística: Gemma congelada → campo (sin tokens al núcleo)

**Fecha:** 14 de septiembre de 2026  
**Rama:** `exp/encoder-linguistico`  
**Módulos:** `src/field_linguistic_layer.rs`, `src/field_gemma_probe.rs`

## Regla dura

El sustrato `Ψ = (T, z, m, C)` **nunca** guarda tokens.  
Pila:

```text
texto → [Gemma congelada / sonda] → LinguisticPacket
      → into_field_features() tira los IDs
      → projector entrenable → z
      → FieldState
```

Lectura (periferia):

```text
z rígido → nearest neighbor en banco de frases → texto al usuario
```

## Por qué no bastaba el encoder n-gram

El mapa lineal sobre bigramas/capas-sintéticas separaba clusters (`margen +0,53`), pero no es lingüística fluida. Aquí Gemma (o una sonda con la misma forma: 26 RMS + hidden + stem-bag) es **una capa más del encoder**, no el modelo.

## Qué se entrena

- **Congelado:** tokenizer + capas Gemma / léxico de forma Gemma + máscara T2.1 (apagadas 2, 3, 4, 5, 17).
- **Entrenable:** solo el projector `features → 12 fasores` (regla delta hacia atractores de campo).

Cero next-token loss. Cero `vocab_size` en el núcleo.

## Resultados (sonda `gemma-shaped-lexicon`, sin GGUF, 160 pasos, semilla `0x61A`)

| métrica | antes | después |
|---|---:|---:|
| acierto de cluster en `z` | 0,33 | **1,00** |
| decode campo → frase (mismo cluster) | 1,00 | **1,00** |
| sim mismo / otro | — | **0,98 / 0,02** |

Test `tokens_die_at_the_probe_boundary`: los IDs existen en el packet y desaparecen al escribir `Ψ`.

## Gemma GGUF real

`FrozenGemma2Probe::try_open` carga el GGUF (`GEMMA2_GGUF` / Ollama store), hace `forward_with_mask` con traza + `sequence_hidden`, arma el mismo `LinguisticPacket`. Sin GGUF el test se salta con mensaje; no inventa RMS.

```bash
cargo test --release --lib field_linguistic -- --nocapture
cargo test --release --lib field_gemma_probe -- --nocapture   # opcional GGUF
```

## Qué no reclama

No Llama 2 (no está en el repo). No fine-tune de Gemma. No tokens dentro de `FieldState`. No ventaja vs transformer en el preprint.
