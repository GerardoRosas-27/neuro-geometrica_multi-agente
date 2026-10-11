# E43/E44 — smoke

- git: `bbba6e2a0bb40aaba1c2e4e1a9309f91c51fab11`
- emb_gemma sha256: `9f91f68a010ff8aee83adabfe8fc08e27ce48e95f57c8dd3fea4444f7cce4ece`
- emb_qwen sha256: `4d48a877d9ebeb37c3df7c173e3c8cf093ae1a8c71931fee084cfd904927048f`

| exp | PASS | FAIL | n |
|---|---|---|---|
| E43 | 1 | 0 | 1 |
| E44 | 0 | 1 | 1 |

- 0x5A00 E43 PASS — gemma dphi es/en/fr/ja=0.11/0.00/0.11/0.25 random=0.00/0.00/0.00/0.00 static=0.00/0.00/0.00/0.00 wins=2/3 decoder_lookup=closed_vocab
- 0x5A00 E44 FAIL — pooled novel acc: gemma=0.097 gemma_anchor_aligned=0.097 qwen_aligned=0.065 random_aligned=0.065
