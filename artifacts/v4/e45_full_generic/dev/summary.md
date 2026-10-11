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
| E45c | 2 | 14 | 16 |

## Per-seed gate notes

### E45

- 0xA400 **FAIL** — L1=0.109 L2=0.000 T1=0.441 H=0.133 static=0.039 L2_rollback_exact=true all_queries_zero=true
- 0xA401 **FAIL** — L1=0.078 L2=0.000 T1=0.574 H=0.359 static=0.062 L2_rollback_exact=true all_queries_zero=true
- 0xA402 **FAIL** — L1=0.074 L2=0.000 T1=0.477 H=0.207 static=0.035 L2_rollback_exact=true all_queries_zero=true
- 0xA403 **FAIL** — L1=0.066 L2=0.004 T1=0.457 H=0.109 static=0.047 L2_rollback_exact=true all_queries_zero=true
- 0xA404 **FAIL** — L1=0.062 L2=0.000 T1=0.453 H=0.164 static=0.051 L2_rollback_exact=true all_queries_zero=true
- 0xA405 **FAIL** — L1=0.027 L2=0.000 T1=0.398 H=0.094 static=0.012 L2_rollback_exact=true all_queries_zero=true
- 0xA406 **FAIL** — L1=0.023 L2=0.000 T1=0.559 H=0.203 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xA407 **FAIL** — L1=0.098 L2=0.004 T1=0.512 H=0.207 static=0.070 L2_rollback_exact=true all_queries_zero=true
- 0xA408 **FAIL** — L1=0.082 L2=0.000 T1=0.527 H=0.176 static=0.023 L2_rollback_exact=true all_queries_zero=true
- 0xA409 **FAIL** — L1=0.109 L2=0.000 T1=0.488 H=0.195 static=0.055 L2_rollback_exact=true all_queries_zero=true
- 0xA40A **FAIL** — L1=0.055 L2=0.000 T1=0.430 H=0.168 static=0.039 L2_rollback_exact=true all_queries_zero=true
- 0xA40B **FAIL** — L1=0.051 L2=0.000 T1=0.449 H=0.164 static=0.051 L2_rollback_exact=true all_queries_zero=true
- 0xA40C **FAIL** — L1=0.051 L2=0.000 T1=0.516 H=0.125 static=0.031 L2_rollback_exact=true all_queries_zero=true
- 0xA40D **FAIL** — L1=0.074 L2=0.000 T1=0.523 H=0.145 static=0.055 L2_rollback_exact=true all_queries_zero=true
- 0xA40E **FAIL** — L1=0.074 L2=0.000 T1=0.434 H=0.141 static=0.043 L2_rollback_exact=true all_queries_zero=true
- 0xA40F **FAIL** — L1=0.031 L2=0.000 T1=0.469 H=0.199 static=0.020 L2_rollback_exact=true all_queries_zero=true

### E45a

- 0xA400 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.073 acc_adapt=0.000 bin0:acc1=0.08,adapt=0.00,n=8.00 bin1:acc1=0.09,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA401 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.417 acc_adapt=0.000 bin0:acc1=0.76,adapt=0.00,n=8.00 bin1:acc1=0.31,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA402 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.156 acc_adapt=0.000 bin0:acc1=0.20,adapt=0.00,n=8.00 bin1:acc1=0.16,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA403 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.135 acc_adapt=0.000 bin0:acc1=0.30,adapt=0.00,n=8.00 bin1:acc1=0.05,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA404 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.156 acc_adapt=0.000 bin0:acc1=0.25,adapt=0.00,n=8.00 bin1:acc1=0.12,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA405 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.073 acc_adapt=0.000 bin0:acc1=0.14,adapt=0.00,n=8.00 bin1:acc1=0.04,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA406 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.240 acc_adapt=0.000 bin0:acc1=0.41,adapt=0.00,n=8.00 bin1:acc1=0.16,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA407 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.125 acc_adapt=0.000 bin0:acc1=0.16,adapt=0.00,n=8.00 bin1:acc1=0.10,adapt=0.00,n=8.00 bin2:acc1=0.10,adapt=0.00,n=8.00]
- 0xA408 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.188 acc_adapt=0.000 bin0:acc1=0.30,adapt=0.00,n=8.00 bin1:acc1=0.15,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA409 **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.219 acc_adapt=0.000 bin0:acc1=0.33,adapt=0.00,n=8.00 bin1:acc1=0.18,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA40A **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.177 acc_adapt=0.000 bin0:acc1=0.34,adapt=0.00,n=8.00 bin1:acc1=0.05,adapt=0.00,n=8.00 bin2:acc1=0.08,adapt=0.00,n=8.00]
- 0xA40B **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.156 acc_adapt=0.000 bin0:acc1=0.29,adapt=0.00,n=8.00 bin1:acc1=0.09,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA40C **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.083 acc_adapt=0.000 bin0:acc1=0.12,adapt=0.00,n=8.00 bin1:acc1=0.07,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA40D **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.156 acc_adapt=0.000 bin0:acc1=0.27,adapt=0.00,n=8.00 bin1:acc1=0.09,adapt=0.00,n=8.00 bin2:acc1=0.10,adapt=0.00,n=8.00]
- 0xA40E **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.042 acc_adapt=0.000 bin0:acc1=0.08,adapt=0.00,n=8.00 bin1:acc1=0.03,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00]
- 0xA40F **FAIL** — L1[acc1=0.000 acc_adapt=0.000 bin0:acc1=0.00,adapt=0.00,n=8.00 bin1:acc1=0.00,adapt=0.00,n=8.00 bin2:acc1=0.00,adapt=0.00,n=8.00] H[acc1=0.250 acc_adapt=0.000 bin0:acc1=0.49,adapt=0.00,n=8.00 bin1:acc1=0.11,adapt=0.00,n=8.00 bin2:acc1=0.08,adapt=0.00,n=8.00]

### E45b

- 0xA400 **FAIL** — L1=0.109 L2=0.000 L2g=0.043 L3=0.012 T1=0.441 H=0.133
- 0xA401 **FAIL** — L1=0.078 L2=0.000 L2g=0.062 L3=0.004 T1=0.574 H=0.359
- 0xA402 **FAIL** — L1=0.074 L2=0.000 L2g=0.074 L3=0.023 T1=0.477 H=0.207
- 0xA403 **FAIL** — L1=0.066 L2=0.004 L2g=0.062 L3=0.008 T1=0.457 H=0.109
- 0xA404 **FAIL** — L1=0.062 L2=0.000 L2g=0.059 L3=0.000 T1=0.453 H=0.164
- 0xA405 **FAIL** — L1=0.027 L2=0.000 L2g=0.039 L3=0.000 T1=0.398 H=0.094
- 0xA406 **FAIL** — L1=0.023 L2=0.000 L2g=0.074 L3=0.004 T1=0.559 H=0.203
- 0xA407 **FAIL** — L1=0.098 L2=0.004 L2g=0.020 L3=0.027 T1=0.512 H=0.207
- 0xA408 **FAIL** — L1=0.082 L2=0.000 L2g=0.078 L3=0.004 T1=0.527 H=0.176
- 0xA409 **FAIL** — L1=0.109 L2=0.000 L2g=0.043 L3=0.000 T1=0.488 H=0.195
- 0xA40A **FAIL** — L1=0.055 L2=0.000 L2g=0.074 L3=0.004 T1=0.430 H=0.168
- 0xA40B **FAIL** — L1=0.051 L2=0.000 L2g=0.039 L3=0.008 T1=0.449 H=0.164
- 0xA40C **FAIL** — L1=0.051 L2=0.000 L2g=0.070 L3=0.004 T1=0.516 H=0.125
- 0xA40D **FAIL** — L1=0.074 L2=0.000 L2g=0.098 L3=0.023 T1=0.523 H=0.145
- 0xA40E **FAIL** — L1=0.074 L2=0.000 L2g=0.098 L3=0.008 T1=0.434 H=0.141
- 0xA40F **FAIL** — L1=0.031 L2=0.000 L2g=0.062 L3=0.008 T1=0.469 H=0.199

### E45c

- 0xA400 **FAIL** — L1:chain=0.000,direct=0.188 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.156,direct=0.625 H:chain=0.000,direct=0.062 STATIC:chain=0.000,direct=0.000
- 0xA401 **PASS** — L1:chain=0.000,direct=0.156 L2g:chain=0.000,direct=0.062 L3:chain=0.000,direct=0.000 T1:chain=0.438,direct=0.469 H:chain=0.156,direct=0.094 STATIC:chain=0.000,direct=0.000
- 0xA402 **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.000,direct=0.625 H:chain=0.000,direct=0.156 STATIC:chain=0.000,direct=0.000
- 0xA403 **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.094,direct=0.469 H:chain=0.000,direct=0.062 STATIC:chain=0.000,direct=0.000
- 0xA404 **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=0.156,direct=0.625 H:chain=0.031,direct=0.094 STATIC:chain=0.000,direct=0.000
- 0xA405 **FAIL** — L1:chain=0.000,direct=0.000 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.219,direct=0.312 H:chain=0.000,direct=0.031 STATIC:chain=0.000,direct=0.000
- 0xA406 **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.125,direct=0.812 H:chain=0.000,direct=0.125 STATIC:chain=0.031,direct=0.031
- 0xA407 **FAIL** — L1:chain=0.000,direct=0.156 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.312,direct=0.562 H:chain=0.000,direct=0.469 STATIC:chain=0.000,direct=0.000
- 0xA408 **FAIL** — L1:chain=0.000,direct=0.250 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.375,direct=0.875 H:chain=0.000,direct=0.312 STATIC:chain=0.000,direct=0.000
- 0xA409 **FAIL** — L1:chain=0.000,direct=0.250 L2g:chain=0.000,direct=0.031 L3:chain=0.000,direct=0.000 T1:chain=0.250,direct=0.500 H:chain=0.000,direct=0.219 STATIC:chain=0.031,direct=0.031
- 0xA40A **FAIL** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.000,direct=0.500 H:chain=0.000,direct=0.062 STATIC:chain=0.000,direct=0.000
- 0xA40B **FAIL** — L1:chain=0.000,direct=0.125 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.281,direct=0.594 H:chain=0.000,direct=0.156 STATIC:chain=0.000,direct=0.000
- 0xA40C **FAIL** — L1:chain=0.000,direct=0.125 L2g:chain=0.031,direct=0.062 L3:chain=0.000,direct=0.000 T1:chain=0.062,direct=0.625 H:chain=0.000,direct=0.219 STATIC:chain=0.000,direct=0.000
- 0xA40D **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.062,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.125,direct=0.594 H:chain=0.031,direct=0.094 STATIC:chain=0.031,direct=0.031
- 0xA40E **FAIL** — L1:chain=0.000,direct=0.094 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.125,direct=0.344 H:chain=0.000,direct=0.188 STATIC:chain=0.000,direct=0.000
- 0xA40F **PASS** — L1:chain=0.000,direct=0.031 L2g:chain=0.000,direct=0.000 L3:chain=0.000,direct=0.000 T1:chain=0.156,direct=0.594 H:chain=0.094,direct=0.156 STATIC:chain=0.000,direct=0.000

