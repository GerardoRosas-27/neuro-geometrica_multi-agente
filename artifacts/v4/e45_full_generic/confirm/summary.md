# v4 confirm summary

- git SHA: `0b6acd3ecb7a954a0a0d53316736faf9aa8d58a9` (dirty=true)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 16
- manifests: `dataset_manifest.jsonl` (sha256 of file: `89c0207ff0e479c2aa0e37446cbbb20eb2ffcb5bf41dacb11f42555d52d8188e`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| E45 | 0 | 16 | 16 |
| E45a | 0 | 16 | 16 |
| E45b | 0 | 16 | 16 |
| E45c | 3 | 13 | 16 |

## Per-seed gate notes

### E45

- 0xB400 **FAIL** — L1=0.039 L2=0.000 T1=0.547 H=0.180 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xB401 **FAIL** — L1=0.055 L2=0.000 T1=0.504 H=0.293 static=0.035 L2_rollback_exact=true all_queries_zero=true
- 0xB402 **FAIL** — L1=0.031 L2=0.000 T1=0.559 H=0.098 static=0.039 L2_rollback_exact=true all_queries_zero=true
- 0xB403 **FAIL** — L1=0.059 L2=0.004 T1=0.484 H=0.285 static=0.016 L2_rollback_exact=true all_queries_zero=true
- 0xB404 **FAIL** — L1=0.066 L2=0.000 T1=0.531 H=0.102 static=0.023 L2_rollback_exact=true all_queries_zero=true
- 0xB405 **FAIL** — L1=0.055 L2=0.004 T1=0.531 H=0.137 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xB406 **FAIL** — L1=0.059 L2=0.000 T1=0.496 H=0.234 static=0.031 L2_rollback_exact=true all_queries_zero=true
- 0xB407 **FAIL** — L1=0.027 L2=0.004 T1=0.359 H=0.098 static=0.055 L2_rollback_exact=true all_queries_zero=true
- 0xB408 **FAIL** — L1=0.117 L2=0.000 T1=0.531 H=0.234 static=0.051 L2_rollback_exact=true all_queries_zero=true
- 0xB409 **FAIL** — L1=0.047 L2=0.000 T1=0.477 H=0.145 static=0.039 L2_rollback_exact=true all_queries_zero=true
- 0xB40A **FAIL** — L1=0.035 L2=0.000 T1=0.496 H=0.180 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xB40B **FAIL** — L1=0.047 L2=0.000 T1=0.430 H=0.070 static=0.055 L2_rollback_exact=true all_queries_zero=true
- 0xB40C **FAIL** — L1=0.055 L2=0.004 T1=0.445 H=0.160 static=0.031 L2_rollback_exact=true all_queries_zero=true
- 0xB40D **FAIL** — L1=0.094 L2=0.004 T1=0.559 H=0.133 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xB40E **FAIL** — L1=0.141 L2=0.000 T1=0.594 H=0.223 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xB40F **FAIL** — L1=0.047 L2=0.000 T1=0.523 H=0.184 static=0.023 L2_rollback_exact=true all_queries_zero=true

### E45a

- 0xB400 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.115 acc_adapt=0.000 bin0:acc1=0.19,adapt=0.00,n=8.00 bin1:acc1=0.07,adapt=0.00,n=8.00 bin2:acc1=0.05,adapt=0.00,n=8.00]
- 0xB401 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.448 acc_adapt=0.000 bin0:acc1=0.69,adapt=0.00,n=8.00 bin1:acc1=0.31,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB402 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.104 acc_adapt=0.000 bin0:acc1=0.15,adapt=0.00,n=8.00 bin1:acc1=0.11,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB403 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.260 acc_adapt=0.000 bin0:acc1=0.51,adapt=0.00,n=8.00 bin1:acc1=0.12,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB404 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.125 acc_adapt=0.000 bin0:acc1=0.19,adapt=0.00,n=8.00 bin1:acc1=0.07,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB405 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.229 acc_adapt=0.000 bin0:acc1=0.43,adapt=0.00,n=8.00 bin1:acc1=0.10,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB406 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.333 acc_adapt=0.000 bin0:acc1=0.55,adapt=0.00,n=8.00 bin1:acc1=0.29,adapt=0.00,n=8.00 bin2:acc1=0.06,adapt=0.00,n=8.00]
- 0xB407 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.156 acc_adapt=0.000 bin0:acc1=0.20,adapt=0.00,n=8.00 bin1:acc1=0.17,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB408 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.281 acc_adapt=0.000 bin0:acc1=0.40,adapt=0.00,n=8.00 bin1:acc1=0.23,adapt=0.00,n=8.00 bin2:acc1=0.08,adapt=0.00,n=8.00]
- 0xB409 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.167 acc_adapt=0.000 bin0:acc1=0.31,adapt=0.00,n=8.00 bin1:acc1=0.10,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB40A **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.073 acc_adapt=0.000 bin0:acc1=0.09,adapt=0.00,n=8.00 bin1:acc1=0.08,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB40B **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.052 acc_adapt=0.000 bin0:acc1=0.08,adapt=0.00,n=8.00 bin1:acc1=0.04,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB40C **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.156 acc_adapt=0.000 bin0:acc1=0.27,adapt=0.00,n=8.00 bin1:acc1=0.13,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB40D **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.146 acc_adapt=0.000 bin0:acc1=0.17,adapt=0.00,n=8.00 bin1:acc1=0.14,adapt=0.00,n=8.00 bin2:acc1=0.12,adapt=0.00,n=8.00]
- 0xB40E **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.240 acc_adapt=0.000 bin0:acc1=0.39,adapt=0.00,n=8.00 bin1:acc1=0.19,adapt=0.00,n=8.00 bin2:acc1=0.06,adapt=0.00,n=8.00]
- 0xB40F **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.229 acc_adapt=0.000 bin0:acc1=0.43,adapt=0.00,n=8.00 bin1:acc1=0.14,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]

### E45b

- 0xB400 **FAIL** — L1=0.039 L2=0.000 L2g=0.070 L3=0.016 T1=0.547 H=0.180
- 0xB401 **FAIL** — L1=0.055 L2=0.000 L2g=0.059 L3=0.008 T1=0.504 H=0.293
- 0xB402 **FAIL** — L1=0.031 L2=0.000 L2g=0.070 L3=0.027 T1=0.559 H=0.098
- 0xB403 **FAIL** — L1=0.059 L2=0.004 L2g=0.031 L3=0.008 T1=0.484 H=0.285
- 0xB404 **FAIL** — L1=0.066 L2=0.000 L2g=0.031 L3=0.016 T1=0.531 H=0.102
- 0xB405 **FAIL** — L1=0.055 L2=0.004 L2g=0.059 L3=0.012 T1=0.531 H=0.137
- 0xB406 **FAIL** — L1=0.059 L2=0.000 L2g=0.047 L3=0.016 T1=0.496 H=0.234
- 0xB407 **FAIL** — L1=0.027 L2=0.004 L2g=0.031 L3=0.008 T1=0.359 H=0.098
- 0xB408 **FAIL** — L1=0.117 L2=0.000 L2g=0.047 L3=0.004 T1=0.531 H=0.234
- 0xB409 **FAIL** — L1=0.047 L2=0.000 L2g=0.027 L3=0.004 T1=0.477 H=0.145
- 0xB40A **FAIL** — L1=0.035 L2=0.000 L2g=0.035 L3=0.000 T1=0.496 H=0.180
- 0xB40B **FAIL** — L1=0.047 L2=0.000 L2g=0.055 L3=0.004 T1=0.430 H=0.070
- 0xB40C **FAIL** — L1=0.055 L2=0.004 L2g=0.047 L3=0.008 T1=0.445 H=0.160
- 0xB40D **FAIL** — L1=0.094 L2=0.004 L2g=0.047 L3=0.016 T1=0.559 H=0.133
- 0xB40E **FAIL** — L1=0.141 L2=0.000 L2g=0.062 L3=0.012 T1=0.594 H=0.223
- 0xB40F **FAIL** — L1=0.047 L2=0.000 L2g=0.074 L3=0.000 T1=0.523 H=0.184

### E45c

- 0xB400 **FAIL** — L1:chain=0.000,direct=0.125 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.312,direct=0.562 H:chain=0.000,direct=0.219 STATIC:chain=0.031,direct=0.031
- 0xB401 **PASS** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=0.344,direct=0.312 H:chain=0.281,direct=0.156 STATIC:chain=0.000,direct=0.000
- 0xB402 **FAIL** — L1:chain=0.000,direct=0.000 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.125,direct=0.531 H:chain=0.000,direct=0.062 STATIC:chain=0.000,direct=0.000
- 0xB403 **PASS** — L1:chain=0.000,direct=0.062 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.406,direct=0.719 H:chain=0.219,direct=0.312 STATIC:chain=0.000,direct=0.000
- 0xB404 **FAIL** — L1:chain=0.000,direct=0.062 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.219,direct=0.750 H:chain=0.031,direct=0.094 STATIC:chain=0.000,direct=0.000
- 0xB405 **FAIL** — L1:chain=0.000,direct=0.188 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.000,direct=0.656 H:chain=0.000,direct=0.062 STATIC:chain=0.000,direct=0.000
- 0xB406 **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=0.031,direct=0.594 H:chain=0.000,direct=0.219 STATIC:chain=0.000,direct=0.000
- 0xB407 **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.031 T1:chain=0.188,direct=0.625 H:chain=0.000,direct=0.000 STATIC:chain=0.000,direct=0.000
- 0xB408 **FAIL** — L1:chain=0.000,direct=0.156 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=0.344,direct=0.625 H:chain=0.000,direct=0.125 STATIC:chain=0.000,direct=0.000
- 0xB409 **FAIL** — L1:chain=0.000,direct=0.188 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.281,direct=0.625 H:chain=0.000,direct=0.219 STATIC:chain=0.000,direct=0.000
- 0xB40A **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.031,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.094,direct=0.625 H:chain=0.000,direct=0.156 STATIC:chain=0.000,direct=0.000
- 0xB40B **FAIL** — L1:chain=0.000,direct=0.062 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.188,direct=0.406 H:chain=0.000,direct=0.094 STATIC:chain=0.000,direct=0.000
- 0xB40C **PASS** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.094,direct=0.500 H:chain=0.125,direct=0.219 STATIC:chain=0.031,direct=0.031
- 0xB40D **FAIL** — L1:chain=0.000,direct=0.156 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.656,direct=0.594 H:chain=0.000,direct=0.125 STATIC:chain=0.000,direct=0.000
- 0xB40E **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=0.344,direct=0.625 H:chain=0.031,direct=0.250 STATIC:chain=0.031,direct=0.031
- 0xB40F **FAIL** — L1:chain=0.000,direct=0.062 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.125,direct=0.750 H:chain=0.062,direct=0.125 STATIC:chain=0.031,direct=0.031

