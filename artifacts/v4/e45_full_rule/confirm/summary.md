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
| E45c | 7 | 9 | 16 |

## Per-seed gate notes

### E45

- 0xB400 **FAIL** — L1=0.039 L2=0.000 T1=0.832 H=0.156 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xB401 **FAIL** — L1=0.055 L2=0.000 T1=0.793 H=0.309 static=0.035 L2_rollback_exact=true all_queries_zero=true
- 0xB402 **FAIL** — L1=0.031 L2=0.000 T1=0.766 H=0.293 static=0.039 L2_rollback_exact=true all_queries_zero=true
- 0xB403 **FAIL** — L1=0.059 L2=0.004 T1=0.789 H=0.215 static=0.016 L2_rollback_exact=true all_queries_zero=true
- 0xB404 **FAIL** — L1=0.066 L2=0.000 T1=0.785 H=0.305 static=0.023 L2_rollback_exact=true all_queries_zero=true
- 0xB405 **FAIL** — L1=0.055 L2=0.004 T1=0.809 H=0.117 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xB406 **FAIL** — L1=0.059 L2=0.000 T1=0.805 H=0.211 static=0.031 L2_rollback_exact=true all_queries_zero=true
- 0xB407 **FAIL** — L1=0.027 L2=0.004 T1=0.816 H=0.227 static=0.055 L2_rollback_exact=true all_queries_zero=true
- 0xB408 **FAIL** — L1=0.117 L2=0.000 T1=0.836 H=0.430 static=0.051 L2_rollback_exact=true all_queries_zero=true
- 0xB409 **FAIL** — L1=0.047 L2=0.000 T1=0.773 H=0.238 static=0.039 L2_rollback_exact=true all_queries_zero=true
- 0xB40A **FAIL** — L1=0.035 L2=0.000 T1=0.805 H=0.176 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xB40B **FAIL** — L1=0.047 L2=0.000 T1=0.820 H=0.246 static=0.055 L2_rollback_exact=true all_queries_zero=true
- 0xB40C **FAIL** — L1=0.055 L2=0.004 T1=0.840 H=0.230 static=0.031 L2_rollback_exact=true all_queries_zero=true
- 0xB40D **FAIL** — L1=0.094 L2=0.004 T1=0.766 H=0.277 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xB40E **FAIL** — L1=0.141 L2=0.000 T1=0.801 H=0.328 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xB40F **FAIL** — L1=0.047 L2=0.000 T1=0.797 H=0.262 static=0.023 L2_rollback_exact=true all_queries_zero=true

### E45a

- 0xB400 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.219 acc_adapt=0.000 bin0:acc1=0.30,adapt=0.00,n=8.00 bin1:acc1=0.23,adapt=0.00,n=8.00 bin2:acc1=0.05,adapt=0.00,n=8.00]
- 0xB401 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.479 acc_adapt=0.000 bin0:acc1=0.62,adapt=0.00,n=8.00 bin1:acc1=0.44,adapt=0.00,n=8.00 bin2:acc1=0.08,adapt=0.00,n=8.00]
- 0xB402 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.427 acc_adapt=0.000 bin0:acc1=0.71,adapt=0.00,n=8.00 bin1:acc1=0.30,adapt=0.00,n=8.00 bin2:acc1=0.19,adapt=0.00,n=8.00]
- 0xB403 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.115 acc_adapt=0.000 bin0:acc1=0.14,adapt=0.00,n=8.00 bin1:acc1=0.08,adapt=0.00,n=8.00 bin2:acc1=0.20,adapt=0.00,n=8.00]
- 0xB404 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.396 acc_adapt=0.000 bin0:acc1=0.60,adapt=0.00,n=8.00 bin1:acc1=0.23,adapt=0.00,n=8.00 bin2:acc1=0.11,adapt=0.00,n=8.00]
- 0xB405 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.104 acc_adapt=0.000 bin0:acc1=0.10,adapt=0.00,n=8.00 bin1:acc1=0.10,adapt=0.00,n=8.00 bin2:acc1=0.14,adapt=0.00,n=8.00]
- 0xB406 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.292 acc_adapt=0.000 bin0:acc1=0.48,adapt=0.00,n=8.00 bin1:acc1=0.27,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB407 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.333 acc_adapt=0.000 bin0:acc1=0.53,adapt=0.00,n=8.00 bin1:acc1=0.22,adapt=0.00,n=8.00 bin2:acc1=0.13,adapt=0.00,n=8.00]
- 0xB408 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.552 acc_adapt=0.000 bin0:acc1=0.88,adapt=0.00,n=8.00 bin1:acc1=0.41,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB409 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.260 acc_adapt=0.000 bin0:acc1=0.37,adapt=0.00,n=8.00 bin1:acc1=0.22,adapt=0.00,n=8.00 bin2:acc1=0.09,adapt=0.00,n=8.00]
- 0xB40A **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.177 acc_adapt=0.000 bin0:acc1=0.30,adapt=0.00,n=8.00 bin1:acc1=0.08,adapt=0.00,n=8.00 bin2:acc1=0.07,adapt=0.00,n=8.00]
- 0xB40B **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.333 acc_adapt=0.000 bin0:acc1=0.65,adapt=0.00,n=8.00 bin1:acc1=0.17,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB40C **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.292 acc_adapt=0.000 bin0:acc1=0.39,adapt=0.00,n=8.00 bin1:acc1=0.33,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB40D **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.312 acc_adapt=0.000 bin0:acc1=0.64,adapt=0.00,n=8.00 bin1:acc1=0.16,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB40E **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.365 acc_adapt=0.000 bin0:acc1=0.69,adapt=0.00,n=8.00 bin1:acc1=0.23,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xB40F **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.396 acc_adapt=0.000 bin0:acc1=0.77,adapt=0.00,n=8.00 bin1:acc1=0.20,adapt=0.00,n=8.00 bin2:acc1=0.09,adapt=0.00,n=8.00]

### E45b

- 0xB400 **FAIL** — L1=0.039 L2=0.000 L2g=0.070 L3=0.016 T1=0.832 H=0.156
- 0xB401 **FAIL** — L1=0.055 L2=0.000 L2g=0.059 L3=0.008 T1=0.793 H=0.309
- 0xB402 **FAIL** — L1=0.031 L2=0.000 L2g=0.070 L3=0.027 T1=0.766 H=0.293
- 0xB403 **FAIL** — L1=0.059 L2=0.004 L2g=0.031 L3=0.008 T1=0.789 H=0.215
- 0xB404 **FAIL** — L1=0.066 L2=0.000 L2g=0.031 L3=0.016 T1=0.785 H=0.305
- 0xB405 **FAIL** — L1=0.055 L2=0.004 L2g=0.059 L3=0.012 T1=0.809 H=0.117
- 0xB406 **FAIL** — L1=0.059 L2=0.000 L2g=0.047 L3=0.016 T1=0.805 H=0.211
- 0xB407 **FAIL** — L1=0.027 L2=0.004 L2g=0.031 L3=0.008 T1=0.816 H=0.227
- 0xB408 **FAIL** — L1=0.117 L2=0.000 L2g=0.047 L3=0.004 T1=0.836 H=0.430
- 0xB409 **FAIL** — L1=0.047 L2=0.000 L2g=0.027 L3=0.004 T1=0.773 H=0.238
- 0xB40A **FAIL** — L1=0.035 L2=0.000 L2g=0.035 L3=0.000 T1=0.805 H=0.176
- 0xB40B **FAIL** — L1=0.047 L2=0.000 L2g=0.055 L3=0.004 T1=0.820 H=0.246
- 0xB40C **FAIL** — L1=0.055 L2=0.004 L2g=0.047 L3=0.008 T1=0.840 H=0.230
- 0xB40D **FAIL** — L1=0.094 L2=0.004 L2g=0.047 L3=0.016 T1=0.766 H=0.277
- 0xB40E **FAIL** — L1=0.141 L2=0.000 L2g=0.062 L3=0.012 T1=0.801 H=0.328
- 0xB40F **FAIL** — L1=0.047 L2=0.000 L2g=0.074 L3=0.000 T1=0.797 H=0.262

### E45c

- 0xB400 **FAIL** — L1:chain=0.000,direct=0.125 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.969 H:chain=0.000,direct=0.125 STATIC:chain=0.031,direct=0.031
- 0xB401 **PASS** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.062,direct=0.125 STATIC:chain=0.000,direct=0.000
- 0xB402 **PASS** — L1:chain=0.000,direct=0.000 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.969 H:chain=0.156,direct=0.250 STATIC:chain=0.000,direct=0.000
- 0xB403 **FAIL** — L1:chain=0.000,direct=0.062 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.969 H:chain=0.031,direct=0.219 STATIC:chain=0.000,direct=0.000
- 0xB404 **FAIL** — L1:chain=0.000,direct=0.062 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.750 H:chain=0.031,direct=0.094 STATIC:chain=0.000,direct=0.000
- 0xB405 **PASS** — L1:chain=0.000,direct=0.188 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.062,direct=0.125 STATIC:chain=0.000,direct=0.000
- 0xB406 **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.938 H:chain=0.000,direct=0.500 STATIC:chain=0.000,direct=0.000
- 0xB407 **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.031 T1:chain=0.906,direct=1.000 H:chain=0.031,direct=0.125 STATIC:chain=0.000,direct=0.000
- 0xB408 **PASS** — L1:chain=0.000,direct=0.156 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.250,direct=0.156 STATIC:chain=0.000,direct=0.000
- 0xB409 **FAIL** — L1:chain=0.000,direct=0.188 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.000,direct=0.344 STATIC:chain=0.000,direct=0.000
- 0xB40A **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.031,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.000,direct=0.125 STATIC:chain=0.000,direct=0.000
- 0xB40B **PASS** — L1:chain=0.000,direct=0.062 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.094,direct=0.219 STATIC:chain=0.000,direct=0.000
- 0xB40C **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.000,direct=0.062 STATIC:chain=0.031,direct=0.031
- 0xB40D **FAIL** — L1:chain=0.000,direct=0.156 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.031,direct=0.188 STATIC:chain=0.000,direct=0.000
- 0xB40E **PASS** — L1:chain=0.000,direct=0.094 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.219,direct=0.406 STATIC:chain=0.031,direct=0.031
- 0xB40F **PASS** — L1:chain=0.000,direct=0.062 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.969 H:chain=0.094,direct=0.000 STATIC:chain=0.031,direct=0.031

