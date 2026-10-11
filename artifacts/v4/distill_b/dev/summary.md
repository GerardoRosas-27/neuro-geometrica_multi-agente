# v4 dev summary

- git SHA: `03335f9862a20e1b8e508d4e1c9a082d8484173e` (dirty=false)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 16
- manifests: `dataset_manifest.jsonl` (sha256 of file: `7b70a417efe5313d6238dc01cdcd645faeeead7cd32b0bc0975cf80e14a92aa3`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| DIST_B | 12 | 4 | 16 |

## Per-seed gate notes

### DIST_B

- 0xA400 **PASS** — acc_H64=0.3945 acc_H136=0.4844 acc_H272=0.4766 acc_H544=0.4570 T=0.8125 clean=true
- 0xA401 **FAIL** — acc_H64=0.4258 acc_H136=0.3594 acc_H272=0.4336 acc_H544=0.2734 T=0.8281 clean=true
- 0xA402 **FAIL** — acc_H64=0.4414 acc_H136=0.1914 acc_H272=0.3359 acc_H544=0.3164 T=0.8203 clean=true
- 0xA403 **FAIL** — acc_H64=0.4453 acc_H136=0.4688 acc_H272=0.4258 acc_H544=0.4609 T=0.8320 clean=true
- 0xA404 **PASS** — acc_H64=0.3906 acc_H136=0.4961 acc_H272=0.1836 acc_H544=0.0977 T=0.8320 clean=true
- 0xA405 **PASS** — acc_H64=0.4023 acc_H136=0.4180 acc_H272=0.5156 acc_H544=0.3125 T=0.8438 clean=true
- 0xA406 **PASS** — acc_H64=0.2070 acc_H136=0.3789 acc_H272=0.2578 acc_H544=0.4492 T=0.8438 clean=true
- 0xA407 **PASS** — acc_H64=0.3125 acc_H136=0.2617 acc_H272=0.4062 acc_H544=0.4336 T=0.8320 clean=true
- 0xA408 **PASS** — acc_H64=0.2656 acc_H136=0.4336 acc_H272=0.3594 acc_H544=0.2969 T=0.8164 clean=true
- 0xA409 **PASS** — acc_H64=0.2695 acc_H136=0.4102 acc_H272=0.3672 acc_H544=0.1992 T=0.8242 clean=true
- 0xA40A **PASS** — acc_H64=0.2266 acc_H136=0.3555 acc_H272=0.1445 acc_H544=0.1797 T=0.8047 clean=true
- 0xA40B **FAIL** — acc_H64=0.4297 acc_H136=0.4219 acc_H272=0.3203 acc_H544=0.4258 T=0.8398 clean=true
- 0xA40C **PASS** — acc_H64=0.2383 acc_H136=0.4023 acc_H272=0.5742 acc_H544=0.5000 T=0.8320 clean=true
- 0xA40D **PASS** — acc_H64=0.2773 acc_H136=0.3438 acc_H272=0.2539 acc_H544=0.2773 T=0.8320 clean=true
- 0xA40E **PASS** — acc_H64=0.2031 acc_H136=0.5430 acc_H272=0.2227 acc_H544=0.1367 T=0.8242 clean=true
- 0xA40F **PASS** — acc_H64=0.4180 acc_H136=0.4102 acc_H272=0.5352 acc_H544=0.3555 T=0.8477 clean=true

