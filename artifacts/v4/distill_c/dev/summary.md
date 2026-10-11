# v4 dev summary

- git SHA: `5d08d64a340a48b7682d51d144cb381d033e6ef6` (dirty=false)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 16
- manifests: `dataset_manifest.jsonl` (sha256 of file: `7b70a417efe5313d6238dc01cdcd645faeeead7cd32b0bc0975cf80e14a92aa3`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| DIST_C | 16 | 0 | 16 |

## Per-seed gate notes

### DIST_C

- 0xA400 **PASS** — H=64 rel_err1 C0=0.0638 C1=0.0338 M1=0.0337 acc C0=0.3984 C1=0.7969 M1=0.8047 fresh_C1=0.7969(q=0) clean=true
- 0xA401 **PASS** — H=64 rel_err1 C0=0.0673 C1=0.0382 M1=0.0374 acc C0=0.3711 C1=0.7969 M1=0.7891 fresh_C1=0.7969(q=0) clean=true
- 0xA402 **PASS** — H=64 rel_err1 C0=0.0698 C1=0.0413 M1=0.0418 acc C0=0.3672 C1=0.8047 M1=0.8203 fresh_C1=0.8047(q=0) clean=true
- 0xA403 **PASS** — H=64 rel_err1 C0=0.0625 C1=0.0416 M1=0.0397 acc C0=0.5039 C1=0.7930 M1=0.8008 fresh_C1=0.7930(q=0) clean=true
- 0xA404 **PASS** — H=64 rel_err1 C0=0.0714 C1=0.0438 M1=0.0423 acc C0=0.4492 C1=0.7891 M1=0.8086 fresh_C1=0.7891(q=0) clean=true
- 0xA405 **PASS** — H=64 rel_err1 C0=0.0679 C1=0.0389 M1=0.0367 acc C0=0.3906 C1=0.8047 M1=0.8047 fresh_C1=0.8047(q=0) clean=true
- 0xA406 **PASS** — H=64 rel_err1 C0=0.0754 C1=0.0332 M1=0.0338 acc C0=0.2227 C1=0.8320 M1=0.8203 fresh_C1=0.8320(q=0) clean=true
- 0xA407 **PASS** — H=64 rel_err1 C0=0.0727 C1=0.0348 M1=0.0337 acc C0=0.3672 C1=0.8281 M1=0.8359 fresh_C1=0.8281(q=0) clean=true
- 0xA408 **PASS** — H=64 rel_err1 C0=0.0754 C1=0.0367 M1=0.0360 acc C0=0.2461 C1=0.8242 M1=0.8242 fresh_C1=0.8242(q=0) clean=true
- 0xA409 **PASS** — H=64 rel_err1 C0=0.0696 C1=0.0395 M1=0.0377 acc C0=0.4023 C1=0.8047 M1=0.7812 fresh_C1=0.8047(q=0) clean=true
- 0xA40A **PASS** — H=64 rel_err1 C0=0.0723 C1=0.0386 M1=0.0369 acc C0=0.2422 C1=0.8125 M1=0.8164 fresh_C1=0.8125(q=0) clean=true
- 0xA40B **PASS** — H=64 rel_err1 C0=0.0632 C1=0.0315 M1=0.0329 acc C0=0.3984 C1=0.8398 M1=0.8438 fresh_C1=0.8398(q=0) clean=true
- 0xA40C **PASS** — H=64 rel_err1 C0=0.0736 C1=0.0351 M1=0.0361 acc C0=0.2500 C1=0.7617 M1=0.7773 fresh_C1=0.7617(q=0) clean=true
- 0xA40D **PASS** — H=64 rel_err1 C0=0.0779 C1=0.0364 M1=0.0368 acc C0=0.2266 C1=0.8125 M1=0.7812 fresh_C1=0.8125(q=0) clean=true
- 0xA40E **PASS** — H=64 rel_err1 C0=0.0790 C1=0.0353 M1=0.0364 acc C0=0.1875 C1=0.7969 M1=0.7852 fresh_C1=0.7969(q=0) clean=true
- 0xA40F **PASS** — H=64 rel_err1 C0=0.0657 C1=0.0390 M1=0.0385 acc C0=0.3945 C1=0.8242 M1=0.8320 fresh_C1=0.8242(q=0) clean=true

