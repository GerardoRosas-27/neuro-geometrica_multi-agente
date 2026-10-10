# v4 smoke summary

- git SHA: `5d08d64a340a48b7682d51d144cb381d033e6ef6` (dirty=false)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 1
- manifests: `dataset_manifest.jsonl` (sha256 of file: `efd9eb4384f1e47cc9aea5d539b6d65559534f5e6f1bbadd6690cb7c4328c745`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| DIST_C | 1 | 0 | 1 |

## Per-seed gate notes

### DIST_C

- 0x5A00 **PASS** — H=64 rel_err1 C0=0.0781 C1=0.0458 M1=0.0467 acc C0=0.2891 C1=0.7891 M1=0.7734 fresh_C1=0.7891(q=0) clean=true

