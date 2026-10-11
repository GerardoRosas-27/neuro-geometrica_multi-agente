# v4 dev summary

- git SHA: `0b6acd3ecb7a954a0a0d53316736faf9aa8d58a9` (dirty=false)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 16
- manifests: `dataset_manifest.jsonl` (sha256 of file: `7b70a417efe5313d6238dc01cdcd645faeeead7cd32b0bc0975cf80e14a92aa3`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| E45 | 0 | 16 | 16 |
| E45a | 0 | 16 | 16 |
| E45b | 0 | 16 | 16 |
| E45c | 3 | 13 | 16 |

## Per-seed gate notes

### E45

- 0xA400 **FAIL** — L1=0.109 L2=0.000 T1=0.812 H=0.160 static=0.039 L2_rollback_exact=true all_queries_zero=true
- 0xA401 **FAIL** — L1=0.078 L2=0.000 T1=0.805 H=0.273 static=0.062 L2_rollback_exact=true all_queries_zero=true
- 0xA402 **FAIL** — L1=0.074 L2=0.000 T1=0.844 H=0.301 static=0.035 L2_rollback_exact=true all_queries_zero=true
- 0xA403 **FAIL** — L1=0.066 L2=0.004 T1=0.789 H=0.250 static=0.047 L2_rollback_exact=true all_queries_zero=true
- 0xA404 **FAIL** — L1=0.062 L2=0.000 T1=0.785 H=0.215 static=0.051 L2_rollback_exact=true all_queries_zero=true
- 0xA405 **FAIL** — L1=0.027 L2=0.000 T1=0.773 H=0.152 static=0.012 L2_rollback_exact=true all_queries_zero=true
- 0xA406 **FAIL** — L1=0.023 L2=0.000 T1=0.816 H=0.285 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xA407 **FAIL** — L1=0.098 L2=0.004 T1=0.820 H=0.262 static=0.070 L2_rollback_exact=true all_queries_zero=true
- 0xA408 **FAIL** — L1=0.082 L2=0.000 T1=0.820 H=0.191 static=0.023 L2_rollback_exact=true all_queries_zero=true
- 0xA409 **FAIL** — L1=0.109 L2=0.000 T1=0.848 H=0.332 static=0.055 L2_rollback_exact=true all_queries_zero=true
- 0xA40A **FAIL** — L1=0.055 L2=0.000 T1=0.758 H=0.285 static=0.039 L2_rollback_exact=true all_queries_zero=true
- 0xA40B **FAIL** — L1=0.051 L2=0.000 T1=0.820 H=0.266 static=0.051 L2_rollback_exact=true all_queries_zero=true
- 0xA40C **FAIL** — L1=0.051 L2=0.000 T1=0.801 H=0.184 static=0.031 L2_rollback_exact=true all_queries_zero=true
- 0xA40D **FAIL** — L1=0.074 L2=0.000 T1=0.820 H=0.234 static=0.055 L2_rollback_exact=true all_queries_zero=true
- 0xA40E **FAIL** — L1=0.074 L2=0.000 T1=0.777 H=0.188 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xA40F **FAIL** — L1=0.031 L2=0.000 T1=0.840 H=0.230 static=0.020 L2_rollback_exact=true all_queries_zero=true

### E45a

- 0xA400 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.240 acc_adapt=0.000 bin0:acc1=0.30,adapt=0.00,n=8.00 bin1:acc1=0.26,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA401 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.250 acc_adapt=0.000 bin0:acc1=0.39,adapt=0.00,n=8.00 bin1:acc1=0.23,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA402 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.479 acc_adapt=0.000 bin0:acc1=0.68,adapt=0.00,n=8.00 bin1:acc1=0.39,adapt=0.00,n=8.00 bin2:acc1=0.09,adapt=0.00,n=8.00]
- 0xA403 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.250 acc_adapt=0.000 bin0:acc1=0.32,adapt=0.00,n=8.00 bin1:acc1=0.25,adapt=0.00,n=8.00 bin2:acc1=0.07,adapt=0.00,n=8.00]
- 0xA404 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.292 acc_adapt=0.000 bin0:acc1=0.36,adapt=0.00,n=8.00 bin1:acc1=0.30,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA405 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.198 acc_adapt=0.000 bin0:acc1=0.37,adapt=0.00,n=7.89 bin1:acc1=0.12,adapt=0.00,n=7.98 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA406 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.344 acc_adapt=0.000 bin0:acc1=0.68,adapt=0.00,n=8.00 bin1:acc1=0.16,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA407 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.406 acc_adapt=0.000 bin0:acc1=0.68,adapt=0.00,n=8.00 bin1:acc1=0.25,adapt=0.00,n=8.00 bin2:acc1=0.10,adapt=0.00,n=8.00]
- 0xA408 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.177 acc_adapt=0.000 bin0:acc1=0.33,adapt=0.00,n=8.00 bin1:acc1=0.10,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA409 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.448 acc_adapt=0.000 bin0:acc1=0.64,adapt=0.00,n=8.00 bin1:acc1=0.36,adapt=0.00,n=8.00 bin2:acc1=0.15,adapt=0.00,n=8.00]
- 0xA40A **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.302 acc_adapt=0.000 bin0:acc1=0.54,adapt=0.00,n=8.00 bin1:acc1=0.12,adapt=0.00,n=8.00 bin2:acc1=0.17,adapt=0.00,n=8.00]
- 0xA40B **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.427 acc_adapt=0.000 bin0:acc1=0.63,adapt=0.00,n=8.00 bin1:acc1=0.36,adapt=0.00,n=8.00 bin2:acc1=0.07,adapt=0.00,n=8.00]
- 0xA40C **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.312 acc_adapt=0.000 bin0:acc1=0.37,adapt=0.00,n=8.00 bin1:acc1=0.30,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA40D **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.250 acc_adapt=0.000 bin0:acc1=0.39,adapt=0.00,n=8.00 bin1:acc1=0.17,adapt=0.00,n=8.00 bin2:acc1=0.20,adapt=0.00,n=8.00]
- 0xA40E **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.333 acc_adapt=0.010 bin0:acc1=0.44,adapt=0.00,n=8.00 bin1:acc1=0.40,adapt=0.03,n=7.88 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA40F **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.354 acc_adapt=0.000 bin0:acc1=0.57,adapt=0.00,n=8.00 bin1:acc1=0.24,adapt=0.00,n=8.00 bin2:acc1=0.15,adapt=0.00,n=8.00]

### E45b

- 0xA400 **FAIL** — L1=0.109 L2=0.000 L2g=0.043 L3=0.012 T1=0.812 H=0.160
- 0xA401 **FAIL** — L1=0.078 L2=0.000 L2g=0.062 L3=0.004 T1=0.805 H=0.273
- 0xA402 **FAIL** — L1=0.074 L2=0.000 L2g=0.074 L3=0.023 T1=0.844 H=0.301
- 0xA403 **FAIL** — L1=0.066 L2=0.004 L2g=0.062 L3=0.008 T1=0.789 H=0.250
- 0xA404 **FAIL** — L1=0.062 L2=0.000 L2g=0.059 L3=0.000 T1=0.785 H=0.215
- 0xA405 **FAIL** — L1=0.027 L2=0.000 L2g=0.039 L3=0.000 T1=0.773 H=0.152
- 0xA406 **FAIL** — L1=0.023 L2=0.000 L2g=0.074 L3=0.004 T1=0.816 H=0.285
- 0xA407 **FAIL** — L1=0.098 L2=0.004 L2g=0.020 L3=0.027 T1=0.820 H=0.262
- 0xA408 **FAIL** — L1=0.082 L2=0.000 L2g=0.078 L3=0.004 T1=0.820 H=0.191
- 0xA409 **FAIL** — L1=0.109 L2=0.000 L2g=0.043 L3=0.000 T1=0.848 H=0.332
- 0xA40A **FAIL** — L1=0.055 L2=0.000 L2g=0.074 L3=0.004 T1=0.758 H=0.285
- 0xA40B **FAIL** — L1=0.051 L2=0.000 L2g=0.039 L3=0.008 T1=0.820 H=0.266
- 0xA40C **FAIL** — L1=0.051 L2=0.000 L2g=0.070 L3=0.004 T1=0.801 H=0.184
- 0xA40D **FAIL** — L1=0.074 L2=0.000 L2g=0.098 L3=0.023 T1=0.820 H=0.234
- 0xA40E **FAIL** — L1=0.074 L2=0.000 L2g=0.098 L3=0.008 T1=0.777 H=0.188
- 0xA40F **FAIL** — L1=0.031 L2=0.000 L2g=0.062 L3=0.008 T1=0.840 H=0.230

### E45c

- 0xA400 **FAIL** — L1:chain=0.000,direct=0.188 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.000,direct=0.156 STATIC:chain=0.000,direct=0.000
- 0xA401 **FAIL** — L1:chain=0.000,direct=0.156 L2g:chain=0.000,direct=0.062 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.969 H:chain=0.031,direct=0.188 STATIC:chain=0.000,direct=0.000
- 0xA402 **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.031,direct=0.125 STATIC:chain=0.000,direct=0.000
- 0xA403 **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.000,direct=0.156 STATIC:chain=0.000,direct=0.000
- 0xA404 **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.906 H:chain=0.000,direct=0.094 STATIC:chain=0.000,direct=0.000
- 0xA405 **FAIL** — L1:chain=0.000,direct=0.000 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.031,direct=0.031 STATIC:chain=0.000,direct=0.000
- 0xA406 **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.938 H:chain=0.000,direct=0.125 STATIC:chain=0.031,direct=0.031
- 0xA407 **PASS** — L1:chain=0.000,direct=0.156 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.969 H:chain=0.094,direct=0.312 STATIC:chain=0.000,direct=0.000
- 0xA408 **FAIL** — L1:chain=0.000,direct=0.250 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.000,direct=0.219 STATIC:chain=0.000,direct=0.000
- 0xA409 **PASS** — L1:chain=0.000,direct=0.250 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.344,direct=0.500 STATIC:chain=0.031,direct=0.031
- 0xA40A **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.906 H:chain=0.000,direct=0.156 STATIC:chain=0.000,direct=0.000
- 0xA40B **PASS** — L1:chain=0.000,direct=0.125 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.812 H:chain=0.188,direct=0.156 STATIC:chain=0.000,direct=0.000
- 0xA40C **FAIL** — L1:chain=0.000,direct=0.125 L2g:chain=0.031,direct=0.062 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.000,direct=0.125 STATIC:chain=0.000,direct=0.000
- 0xA40D **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.062,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=0.875 H:chain=0.031,direct=0.125 STATIC:chain=0.031,direct=0.031
- 0xA40E **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.000,direct=0.031 STATIC:chain=0.000,direct=0.000
- 0xA40F **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=1.000,direct=1.000 H:chain=0.031,direct=0.125 STATIC:chain=0.000,direct=0.000

