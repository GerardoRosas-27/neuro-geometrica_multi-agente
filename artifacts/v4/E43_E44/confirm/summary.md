# E43/E44 — confirm

- git: `dd28f9f4d9c83e4856a19579f36d33a3e0eadc38`
- emb_gemma sha256: `9f91f68a010ff8aee83adabfe8fc08e27ce48e95f57c8dd3fea4444f7cce4ece`
- emb_qwen sha256: `4d48a877d9ebeb37c3df7c173e3c8cf093ae1a8c71931fee084cfd904927048f`

| exp | PASS | FAIL | n |
|---|---|---|---|
| E43 | 3 | 13 | 16 |
| E44 | 3 | 13 | 16 |

- 0xB400 E43 PASS — gemma dphi es/en/fr/ja=0.11/0.11/0.12/0.14 random=0.44/0.00/0.12/0.00 static=0.00/0.00/0.00/0.00 wins=2/3 decoder_lookup=closed_vocab
- 0xB400 E44 FAIL — pooled novel acc: gemma=0.121 gemma_anchor_aligned=0.030 qwen_aligned=0.061 random_aligned=0.030
- 0xB401 E43 FAIL — gemma dphi es/en/fr/ja=0.22/0.00/0.11/0.00 random=0.22/0.00/0.00/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xB401 E44 PASS — pooled novel acc: gemma=0.088 gemma_anchor_aligned=0.118 qwen_aligned=0.088 random_aligned=0.000
- 0xB402 E43 PASS — gemma dphi es/en/fr/ja=0.22/0.11/0.11/0.33 random=0.11/0.00/0.11/0.00 static=0.00/0.00/0.00/0.00 wins=2/3 decoder_lookup=closed_vocab
- 0xB402 E44 PASS — pooled novel acc: gemma=0.167 gemma_anchor_aligned=0.100 qwen_aligned=0.167 random_aligned=0.067
- 0xB403 E43 FAIL — gemma dphi es/en/fr/ja=0.33/0.00/0.11/0.00 random=0.11/0.22/0.00/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xB403 E44 FAIL — pooled novel acc: gemma=0.129 gemma_anchor_aligned=0.129 qwen_aligned=0.000 random_aligned=0.032
- 0xB404 E43 FAIL — gemma dphi es/en/fr/ja=0.11/0.00/0.11/0.17 random=0.22/0.11/0.22/0.17 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xB404 E44 FAIL — pooled novel acc: gemma=0.091 gemma_anchor_aligned=0.121 qwen_aligned=0.061 random_aligned=0.030
- 0xB405 E43 FAIL — gemma dphi es/en/fr/ja=0.22/0.00/0.22/0.00 random=0.22/0.11/0.11/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xB405 E44 FAIL — pooled novel acc: gemma=0.129 gemma_anchor_aligned=0.065 qwen_aligned=0.065 random_aligned=0.065
- 0xB406 E43 FAIL — gemma dphi es/en/fr/ja=0.11/0.00/0.11/0.00 random=0.33/0.00/0.22/0.00 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xB406 E44 FAIL — pooled novel acc: gemma=0.062 gemma_anchor_aligned=0.031 qwen_aligned=0.031 random_aligned=0.094
- 0xB407 E43 FAIL — gemma dphi es/en/fr/ja=0.44/0.00/0.11/0.00 random=0.00/0.22/0.44/0.17 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xB407 E44 FAIL — pooled novel acc: gemma=0.152 gemma_anchor_aligned=0.091 qwen_aligned=0.030 random_aligned=0.091
- 0xB408 E43 FAIL — gemma dphi es/en/fr/ja=0.00/0.00/0.00/0.00 random=0.22/0.00/0.12/0.00 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xB408 E44 FAIL — pooled novel acc: gemma=0.000 gemma_anchor_aligned=0.000 qwen_aligned=0.031 random_aligned=0.062
- 0xB409 E43 FAIL — gemma dphi es/en/fr/ja=0.11/0.11/0.00/0.00 random=0.11/0.11/0.11/0.00 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xB409 E44 FAIL — pooled novel acc: gemma=0.061 gemma_anchor_aligned=0.091 qwen_aligned=0.061 random_aligned=0.061
- 0xB40A E43 FAIL — gemma dphi es/en/fr/ja=0.00/0.11/0.12/0.14 random=0.33/0.11/0.00/0.14 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xB40A E44 FAIL — pooled novel acc: gemma=0.091 gemma_anchor_aligned=0.030 qwen_aligned=0.091 random_aligned=0.061
- 0xB40B E43 FAIL — gemma dphi es/en/fr/ja=0.22/0.00/0.11/0.14 random=0.11/0.22/0.44/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xB40B E44 FAIL — pooled novel acc: gemma=0.118 gemma_anchor_aligned=0.088 qwen_aligned=0.118 random_aligned=0.118
- 0xB40C E43 FAIL — gemma dphi es/en/fr/ja=0.33/0.00/0.00/0.00 random=0.11/0.00/0.11/0.00 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xB40C E44 FAIL — pooled novel acc: gemma=0.091 gemma_anchor_aligned=0.091 qwen_aligned=0.030 random_aligned=0.091
- 0xB40D E43 PASS — gemma dphi es/en/fr/ja=0.00/0.11/0.12/0.00 random=0.11/0.00/0.00/0.20 static=0.00/0.00/0.00/0.00 wins=2/3 decoder_lookup=closed_vocab
- 0xB40D E44 FAIL — pooled novel acc: gemma=0.065 gemma_anchor_aligned=0.129 qwen_aligned=0.032 random_aligned=0.000
- 0xB40E E43 FAIL — gemma dphi es/en/fr/ja=0.00/0.00/0.00/0.00 random=0.11/0.11/0.00/0.00 static=0.00/0.00/0.00/0.00 wins=0/3 decoder_lookup=closed_vocab
- 0xB40E E44 FAIL — pooled novel acc: gemma=0.000 gemma_anchor_aligned=0.000 qwen_aligned=0.029 random_aligned=0.086
- 0xB40F E43 FAIL — gemma dphi es/en/fr/ja=0.22/0.33/0.00/0.00 random=0.11/0.00/0.22/0.00 static=0.00/0.00/0.00/0.00 wins=1/3 decoder_lookup=closed_vocab
- 0xB40F E44 PASS — pooled novel acc: gemma=0.156 gemma_anchor_aligned=0.125 qwen_aligned=0.188 random_aligned=0.031
