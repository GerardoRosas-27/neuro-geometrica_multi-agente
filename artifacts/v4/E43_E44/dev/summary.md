# E43/E44 — dev

- git: `dd28f9f4d9c83e4856a19579f36d33a3e0eadc38`
- emb_gemma sha256: `9f91f68a010ff8aee83adabfe8fc08e27ce48e95f57c8dd3fea4444f7cce4ece`
- emb_qwen sha256: `4d48a877d9ebeb37c3df7c173e3c8cf093ae1a8c71931fee084cfd904927048f`

| exp | PASS | FAIL | n |
|---|---|---|---|
| E43 | 2 | 14 | 16 |
| E44 | 3 | 13 | 16 |

- 0xA400 E43 FAIL — gemma dphi es/en/fr/ja=0.11/0.00/0.00/0.12 random=0.00/0.00/0.00/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xA400 E44 FAIL — pooled novel acc: gemma=0.059 gemma_anchor_aligned=0.088 qwen_aligned=0.088 random_aligned=0.147
- 0xA401 E43 PASS — gemma dphi es/en/fr/ja=0.22/0.11/0.00/0.14 random=0.00/0.00/0.00/0.00 static=0.00/0.00/0.00/0.00 wins=2/3 decoder_lookup=closed_vocab
- 0xA401 E44 FAIL — pooled novel acc: gemma=0.118 gemma_anchor_aligned=0.029 qwen_aligned=0.000 random_aligned=0.029
- 0xA402 E43 FAIL — gemma dphi es/en/fr/ja=0.11/0.11/0.11/0.00 random=0.33/0.00/0.11/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xA402 E44 FAIL — pooled novel acc: gemma=0.094 gemma_anchor_aligned=0.125 qwen_aligned=0.000 random_aligned=0.094
- 0xA403 E43 FAIL — gemma dphi es/en/fr/ja=0.00/0.00/0.00/0.00 random=0.00/0.00/0.11/0.00 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xA403 E44 FAIL — pooled novel acc: gemma=0.000 gemma_anchor_aligned=0.061 qwen_aligned=0.000 random_aligned=0.091
- 0xA404 E43 PASS — gemma dphi es/en/fr/ja=0.33/0.11/0.00/0.14 random=0.11/0.00/0.00/0.00 static=0.00/0.00/0.00/0.00 wins=2/3 decoder_lookup=closed_vocab
- 0xA404 E44 FAIL — pooled novel acc: gemma=0.147 gemma_anchor_aligned=0.088 qwen_aligned=0.029 random_aligned=0.118
- 0xA405 E43 FAIL — gemma dphi es/en/fr/ja=0.11/0.00/0.11/0.00 random=0.11/0.00/0.11/0.00 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xA405 E44 FAIL — pooled novel acc: gemma=0.061 gemma_anchor_aligned=0.061 qwen_aligned=0.000 random_aligned=0.000
- 0xA406 E43 FAIL — gemma dphi es/en/fr/ja=0.22/0.00/0.00/0.00 random=0.00/0.00/0.22/0.00 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xA406 E44 FAIL — pooled novel acc: gemma=0.061 gemma_anchor_aligned=0.030 qwen_aligned=0.061 random_aligned=0.030
- 0xA407 E43 FAIL — gemma dphi es/en/fr/ja=0.22/0.00/0.00/0.17 random=0.00/0.00/0.00/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xA407 E44 FAIL — pooled novel acc: gemma=0.091 gemma_anchor_aligned=0.061 qwen_aligned=0.030 random_aligned=0.121
- 0xA408 E43 FAIL — gemma dphi es/en/fr/ja=0.11/0.00/0.00/0.14 random=0.11/0.00/0.11/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xA408 E44 FAIL — pooled novel acc: gemma=0.059 gemma_anchor_aligned=0.029 qwen_aligned=0.029 random_aligned=0.147
- 0xA409 E43 FAIL — gemma dphi es/en/fr/ja=0.44/0.00/0.00/0.00 random=0.22/0.00/0.11/0.29 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xA409 E44 FAIL — pooled novel acc: gemma=0.118 gemma_anchor_aligned=0.088 qwen_aligned=0.000 random_aligned=0.088
- 0xA40A E43 FAIL — gemma dphi es/en/fr/ja=0.22/0.00/0.00/0.00 random=0.00/0.00/0.00/0.14 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xA40A E44 FAIL — pooled novel acc: gemma=0.059 gemma_anchor_aligned=0.029 qwen_aligned=0.029 random_aligned=0.029
- 0xA40B E43 FAIL — gemma dphi es/en/fr/ja=0.22/0.11/0.25/0.00 random=0.22/0.11/0.00/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xA40B E44 FAIL — pooled novel acc: gemma=0.152 gemma_anchor_aligned=0.152 qwen_aligned=0.000 random_aligned=0.121
- 0xA40C E43 FAIL — gemma dphi es/en/fr/ja=0.00/0.11/0.22/0.14 random=0.00/0.11/0.00/0.14 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xA40C E44 PASS — pooled novel acc: gemma=0.118 gemma_anchor_aligned=0.059 qwen_aligned=0.088 random_aligned=0.029
- 0xA40D E43 FAIL — gemma dphi es/en/fr/ja=0.33/0.00/0.00/0.17 random=0.00/0.11/0.00/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xA40D E44 PASS — pooled novel acc: gemma=0.125 gemma_anchor_aligned=0.125 qwen_aligned=0.125 random_aligned=0.062
- 0xA40E E43 FAIL — gemma dphi es/en/fr/ja=0.22/0.11/0.22/0.00 random=0.11/0.11/0.11/0.29 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xA40E E44 PASS — pooled novel acc: gemma=0.147 gemma_anchor_aligned=0.088 qwen_aligned=0.088 random_aligned=0.000
- 0xA40F E43 FAIL — gemma dphi es/en/fr/ja=0.22/0.00/0.11/0.20 random=0.00/0.11/0.11/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xA40F E44 FAIL — pooled novel acc: gemma=0.125 gemma_anchor_aligned=0.094 qwen_aligned=0.031 random_aligned=0.062
