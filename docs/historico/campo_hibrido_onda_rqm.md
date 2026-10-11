# Campo sin tokens + híbrido onda/RQM (LLM periferia)

**Rama:** `exp/campo-hibrido-onda` (sobre `exp/encoder-linguistico`)  
**Módulo:** `src/field_hybrid_infer.rs`

## Arquitectura

```
texto ──► LinguisticPacket.into_field_features() ──► concept_id
              (tira token IDs)                         │
                                                      ▼
                                            HybridWaveRqm
                                         ┌─────────┴─────────┐
                                    cue frío            cue entrenado
                                         │                   │
                                  WavePredictCore    NativeThermoRqm
                                         └─────────┬─────────┘
                                                   ▼
                              write fasores → FieldState (Ψ)
                              handshake + Hebb (CDT geométrico)
                                                   ▼
                              LabelDecoder / Gemma  (solo decode)
```

- **Sustrato:** `FieldState` — cero tokens (`has_token_ids() == false`).
- **Inferencia nueva:** líquido de ondas (interferencia pasado×futuro).
- **Inferencia entrenada:** RQM nativo (destilado tras la primera onda).
- **LLM:** periferia de encode (firewall) y decode; no consolida el núcleo.

## Resultados (`cargo test --release --lib field_hybrid_infer`)

| Test | Resultado |
|------|-----------|
| text→hybrid sin tokens en campo | OK, hs=1.0 |
| cold WaveNew → warm RqmTrained | OK |
| consolidación fasorial | OK |
| decode periférico | OK |

5/5 tests.

## Nota (2026-09-21): routing supersedido

El **camino caliente de query ya no enruta warm → RQM**.

Fuente de verdad actual: **`src/liquid_cdt_memory.rs`** + `docs/arquitectura_liquido_cdt_memoria.md`.

- Inferencia (frío y caliente) = **solo líquido** (`WavePredictCore`).
- Memoria consolidada = **Thermo CDT** tras `sleep_consolidate`.
- **RQM** = pegamento relacional cue→label **únicamente durante el sueño**, no como router de inferencia.

Este documento describe el híbrido onda/RQM en query como diseño de la pila campo; ese routing warm-RQM queda **supersedido** para el núcleo de inferencia del POC.
