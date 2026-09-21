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
