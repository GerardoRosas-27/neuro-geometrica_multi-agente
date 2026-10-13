# v6 smoke summary

- git SHA: `0c4136ce960bd86838247769476847852713e67b` (dirty=true)
- steps: 200
- seeds: 1

| Gate | PASS | FAIL | n |
|---|---|---|---|
| R_E3_mlp | 0 | 1 | 1 |
| R_E3_shuffled | 0 | 1 | 1 |
| R_E3_v4 | 0 | 1 | 1 |
| R_E3_v4_V1reproj | 0 | 1 | 1 |
| R_H1_bias | 0 | 1 | 1 |
| R_H2_mh_eqc | 0 | 1 | 1 |
| R_H2_prm_eqc | 0 | 1 | 1 |
| R_H3_x0.5 | 0 | 1 | 1 |
| R_H3_x1 | 0 | 1 | 1 |
| R_H3_x2 | 0 | 1 | 1 |
| R_H3_x4 | 0 | 1 | 1 |
| R_H4_sobolev | 0 | 1 | 1 |
| R_H4_sobolev_mlp | 0 | 1 | 1 |
| R_H5_eta1 | 0 | 1 | 1 |
| R_H5_eta2 | 0 | 1 | 1 |
| R_H5_eta4 | 0 | 1 | 1 |
| R_H5_eta8 | 0 | 1 | 1 |
| R_H5_soft | 0 | 1 | 1 |
| R_H6_semigroup | 0 | 1 | 1 |
| R_H7_normal | 0 | 1 | 1 |
| R_V6_all | 0 | 1 | 1 |
| R_V6_all_H2 | 0 | 1 | 1 |
| R_V6_all_mlp | 0 | 1 | 1 |
| R_V6_shuffled | 0 | 1 | 1 |

## Per-seed gate notes

### R_E3_mlp

- 0x5A60 **FAIL** — R_E3_mlp pending=false region=0 acc1=0.0000 rel_err1=0.3174 acc_h16=0.0000 rel_h8=1.0814 pert_h8=0.2140 bias_norm=-0.22228 bias_ang=-0.45240 bias_rel=0.50148 sigma_gap=0.1567 dstep_vs_E3=0.03499 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.23 h1:acc=0.0000,rel=0.5097,pert=0.7589 h2:acc=0.0000,rel=0.5472,pert=0.5982 h4:acc=0.0000,rel=0.8105,pert=0.4028 h8:acc=0.0000,rel=1.0814,pert=0.2140 h16:acc=0.0000,rel=1.2057,pert=0.0782 h32:acc=0.0000,rel=1.2295,pert=0.0126 h64:acc=0.0000,rel=1.2373,pert=0.0004

### R_E3_shuffled

- 0x5A60 **FAIL** — R_E3_shuffled pending=false region=0 acc1=0.0000 rel_err1=1.0102 acc_h16=0.0000 rel_h8=1.0172 pert_h8=0.0000 bias_norm=-0.74243 bias_ang=0.36736 bias_rel=0.98809 sigma_gap=0.5622 dstep_vs_E3=0.95706 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.22 h1:acc=0.0000,rel=0.9904,pert=0.2506 h2:acc=0.0000,rel=0.9911,pert=0.0494 h4:acc=0.0000,rel=1.0181,pert=0.0022 h8:acc=0.0000,rel=1.0172,pert=0.0000 h16:acc=0.0000,rel=1.0082,pert=0.0000 h32:acc=0.0000,rel=1.0210,pert=0.0000 h64:acc=0.0000,rel=1.0187,pert=0.0000

### R_E3_v4

- 0x5A60 **FAIL** — R_E3_v4 pending=false region=0 acc1=0.0000 rel_err1=0.3143 acc_h16=0.0000 rel_h8=1.0670 pert_h8=0.2022 bias_norm=-0.22403 bias_ang=-0.44915 bias_rel=0.50009 sigma_gap=0.1374 dstep_vs_E3=0.00000 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.23 h1:acc=0.0000,rel=0.5076,pert=0.7766 h2:acc=0.0000,rel=0.5459,pert=0.6071 h4:acc=0.0000,rel=0.8076,pert=0.4020 h8:acc=0.0000,rel=1.0670,pert=0.2022 h16:acc=0.0000,rel=1.1637,pert=0.0775 h32:acc=0.0000,rel=1.1999,pert=0.0098 h64:acc=0.0000,rel=1.1663,pert=0.0002

### R_E3_v4_V1reproj

- 0x5A60 **FAIL** — R_E3_v4_V1reproj pending=false region=0 acc1=0.0000 rel_err1=0.3143 acc_h16=0.0000 rel_h8=1.0316 pert_h8=0.0461 bias_norm=-0.22403 bias_ang=-0.44915 bias_rel=0.50009 sigma_gap=0.1374 dstep_vs_E3=0.00000 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.23 h1:acc=0.0000,rel=0.4974,pert=0.3706 h2:acc=0.0000,rel=0.5292,pert=0.2485 h4:acc=0.0000,rel=0.7888,pert=0.1174 h8:acc=0.0000,rel=1.0316,pert=0.0461 h16:acc=0.0000,rel=1.1363,pert=0.0151 h32:acc=0.0000,rel=1.2016,pert=0.0034 h64:acc=0.0000,rel=1.1714,pert=0.0001

### R_H1_bias

- 0x5A60 **FAIL** — R_H1_bias pending=false region=0 acc1=0.0000 rel_err1=0.3166 acc_h16=0.0000 rel_h8=1.0758 pert_h8=0.2002 bias_norm=-0.22462 bias_ang=-0.44960 bias_rel=0.50100 sigma_gap=0.1445 dstep_vs_E3=0.00812 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.29 h1:acc=0.0000,rel=0.5087,pert=0.7721 h2:acc=0.0000,rel=0.5487,pert=0.6063 h4:acc=0.0000,rel=0.8131,pert=0.4002 h8:acc=0.0000,rel=1.0758,pert=0.2002 h16:acc=0.0000,rel=1.1853,pert=0.0766 h32:acc=0.0000,rel=1.2474,pert=0.0095 h64:acc=0.0000,rel=1.4805,pert=0.0002

### R_H2_mh_eqc

- 0x5A60 **FAIL** — R_H2_mh_eqc pending=true region=0 acc1=0.0000 rel_err1=0.5689 acc_h16=0.0000 rel_h8=1.1302 pert_h8=0.0804 bias_norm=-0.27247 bias_ang=0.04330 bias_rel=0.92627 sigma_gap=0.1240 dstep_vs_E3=0.55594 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.20 h1:acc=0.0000,rel=0.9238,pert=0.7267 h2:acc=0.0000,rel=0.5646,pert=0.5563 h4:acc=0.0000,rel=0.7998,pert=0.3323 h8:acc=0.0000,rel=1.1302,pert=0.0804 h16:acc=0.0000,rel=2.2923,pert=0.0050 h32:acc=0.0000,rel=3.4099,pert=0.0000 h64:acc=0.0000,rel=3.4044,pert=0.0000

### R_H2_prm_eqc

- 0x5A60 **FAIL** — R_H2_prm_eqc pending=true region=0 acc1=0.0000 rel_err1=0.3620 acc_h16=0.0000 rel_h8=1.2015 pert_h8=0.1210 bias_norm=-0.31005 bias_ang=-0.37898 bias_rel=0.57072 sigma_gap=0.1902 dstep_vs_E3=0.18986 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.14 h1:acc=0.0000,rel=0.5744,pert=0.6955 h2:acc=0.0000,rel=0.6892,pert=0.5255 h4:acc=0.0000,rel=0.9327,pert=0.3066 h8:acc=0.0000,rel=1.2015,pert=0.1210 h16:acc=0.0000,rel=1.4020,pert=0.0240 h32:acc=0.0000,rel=1.5980,pert=0.0012 h64:acc=0.0000,rel=1.6282,pert=0.0000

### R_H3_x0.5

- 0x5A60 **FAIL** — R_H3_x0.5 pending=false region=0 acc1=0.0000 rel_err1=0.3061 acc_h16=0.0000 rel_h8=1.0459 pert_h8=0.2225 bias_norm=-0.22967 bias_ang=-0.39906 bias_rel=0.47923 sigma_gap=0.1222 dstep_vs_E3=0.04832 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.22 h1:acc=0.0000,rel=0.4860,pert=0.7948 h2:acc=0.0000,rel=0.5344,pert=0.6170 h4:acc=0.0000,rel=0.7898,pert=0.4216 h8:acc=0.0000,rel=1.0459,pert=0.2225 h16:acc=0.0000,rel=1.1490,pert=0.0751 h32:acc=0.0000,rel=1.1587,pert=0.0115 h64:acc=0.0000,rel=1.1953,pert=0.0003

### R_H3_x1

- 0x5A60 **FAIL** — R_H3_x1 pending=false region=0 acc1=0.0039 rel_err1=0.3081 acc_h16=0.0000 rel_h8=1.0797 pert_h8=0.1963 bias_norm=-0.23532 bias_ang=-0.43555 bias_rel=0.49613 sigma_gap=0.1365 dstep_vs_E3=0.04380 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0039 fresh_q=0 train_s=0.23 h1:acc=0.0000,rel=0.5015,pert=0.7757 h2:acc=0.0000,rel=0.5571,pert=0.5954 h4:acc=0.0000,rel=0.8227,pert=0.3920 h8:acc=0.0000,rel=1.0797,pert=0.1963 h16:acc=0.0000,rel=1.1966,pert=0.0717 h32:acc=0.0000,rel=1.3681,pert=0.0111 h64:acc=0.0000,rel=1.7435,pert=0.0003

### R_H3_x2

- 0x5A60 **FAIL** — R_H3_x2 pending=false region=0 acc1=0.0000 rel_err1=0.3043 acc_h16=0.0000 rel_h8=1.0335 pert_h8=0.2164 bias_norm=-0.22294 bias_ang=-0.41732 bias_rel=0.48168 sigma_gap=0.1094 dstep_vs_E3=0.04763 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.24 h1:acc=0.0000,rel=0.4859,pert=0.7966 h2:acc=0.0000,rel=0.5212,pert=0.6229 h4:acc=0.0000,rel=0.7787,pert=0.4199 h8:acc=0.0000,rel=1.0335,pert=0.2164 h16:acc=0.0000,rel=1.1126,pert=0.0724 h32:acc=0.0000,rel=1.1231,pert=0.0108 h64:acc=0.0000,rel=1.2035,pert=0.0002

### R_H3_x4

- 0x5A60 **FAIL** — R_H3_x4 pending=false region=0 acc1=0.0000 rel_err1=0.3037 acc_h16=0.0000 rel_h8=1.0518 pert_h8=0.1899 bias_norm=-0.24124 bias_ang=-0.38276 bias_rel=0.47674 sigma_gap=0.1363 dstep_vs_E3=0.04425 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.25 h1:acc=0.0000,rel=0.4818,pert=0.7753 h2:acc=0.0000,rel=0.5462,pert=0.5896 h4:acc=0.0000,rel=0.8020,pert=0.3838 h8:acc=0.0000,rel=1.0518,pert=0.1899 h16:acc=0.0000,rel=1.1315,pert=0.0698 h32:acc=0.0000,rel=1.1287,pert=0.0111 h64:acc=0.0000,rel=1.1011,pert=0.0002

### R_H4_sobolev

- 0x5A60 **FAIL** — R_H4_sobolev pending=false region=0 acc1=0.0000 rel_err1=0.3129 acc_h16=0.0000 rel_h8=1.0586 pert_h8=0.2075 bias_norm=-0.22223 bias_ang=-0.45144 bias_rel=0.49981 sigma_gap=0.1347 dstep_vs_E3=0.00688 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.47 h1:acc=0.0000,rel=0.5069,pert=0.7825 h2:acc=0.0000,rel=0.5410,pert=0.6106 h4:acc=0.0000,rel=0.8005,pert=0.4073 h8:acc=0.0000,rel=1.0586,pert=0.2075 h16:acc=0.0000,rel=1.1556,pert=0.0788 h32:acc=0.0000,rel=1.1968,pert=0.0104 h64:acc=0.0000,rel=1.1437,pert=0.0002

### R_H4_sobolev_mlp

- 0x5A60 **FAIL** — R_H4_sobolev_mlp pending=false region=0 acc1=0.0000 rel_err1=0.3163 acc_h16=0.0000 rel_h8=1.0750 pert_h8=0.2172 bias_norm=-0.21993 bias_ang=-0.45547 bias_rel=0.50155 sigma_gap=0.1577 dstep_vs_E3=0.03570 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.47 h1:acc=0.0000,rel=0.5095,pert=0.7633 h2:acc=0.0000,rel=0.5428,pert=0.5998 h4:acc=0.0000,rel=0.8043,pert=0.4055 h8:acc=0.0000,rel=1.0750,pert=0.2172 h16:acc=0.0000,rel=1.2012,pert=0.0801 h32:acc=0.0000,rel=1.2256,pert=0.0132 h64:acc=0.0000,rel=1.2147,pert=0.0004

### R_H5_eta1

- 0x5A60 **FAIL** — R_H5_eta1 pending=false region=0 acc1=0.0000 rel_err1=0.3143 acc_h16=0.0000 rel_h8=1.0672 pert_h8=0.2024 bias_norm=-0.22402 bias_ang=-0.44923 bias_rel=0.50012 sigma_gap=0.1373 dstep_vs_E3=0.00049 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.22 h1:acc=0.0000,rel=0.5076,pert=0.7765 h2:acc=0.0000,rel=0.5460,pert=0.6073 h4:acc=0.0000,rel=0.8077,pert=0.4022 h8:acc=0.0000,rel=1.0672,pert=0.2024 h16:acc=0.0000,rel=1.1640,pert=0.0773 h32:acc=0.0000,rel=1.2004,pert=0.0097 h64:acc=0.0000,rel=1.1672,pert=0.0002

### R_H5_eta2

- 0x5A60 **FAIL** — R_H5_eta2 pending=false region=0 acc1=0.0000 rel_err1=0.3143 acc_h16=0.0000 rel_h8=1.0671 pert_h8=0.2023 bias_norm=-0.22405 bias_ang=-0.44928 bias_rel=0.50013 sigma_gap=0.1374 dstep_vs_E3=0.00061 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.23 h1:acc=0.0000,rel=0.5076,pert=0.7765 h2:acc=0.0000,rel=0.5460,pert=0.6073 h4:acc=0.0000,rel=0.8077,pert=0.4022 h8:acc=0.0000,rel=1.0671,pert=0.2023 h16:acc=0.0000,rel=1.1639,pert=0.0771 h32:acc=0.0000,rel=1.2006,pert=0.0097 h64:acc=0.0000,rel=1.1669,pert=0.0002

### R_H5_eta4

- 0x5A60 **FAIL** — R_H5_eta4 pending=false region=0 acc1=0.0000 rel_err1=0.3143 acc_h16=0.0000 rel_h8=1.0672 pert_h8=0.2021 bias_norm=-0.22408 bias_ang=-0.44932 bias_rel=0.50013 sigma_gap=0.1377 dstep_vs_E3=0.00069 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.21 h1:acc=0.0000,rel=0.5076,pert=0.7763 h2:acc=0.0000,rel=0.5461,pert=0.6072 h4:acc=0.0000,rel=0.8078,pert=0.4020 h8:acc=0.0000,rel=1.0672,pert=0.2021 h16:acc=0.0000,rel=1.1637,pert=0.0770 h32:acc=0.0000,rel=1.1998,pert=0.0096 h64:acc=0.0000,rel=1.1662,pert=0.0002

### R_H5_eta8

- 0x5A60 **FAIL** — R_H5_eta8 pending=false region=0 acc1=0.0000 rel_err1=0.3142 acc_h16=0.0000 rel_h8=1.0672 pert_h8=0.2016 bias_norm=-0.22431 bias_ang=-0.44940 bias_rel=0.50019 sigma_gap=0.1379 dstep_vs_E3=0.00123 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.25 h1:acc=0.0000,rel=0.5076,pert=0.7762 h2:acc=0.0000,rel=0.5461,pert=0.6070 h4:acc=0.0000,rel=0.8079,pert=0.4017 h8:acc=0.0000,rel=1.0672,pert=0.2016 h16:acc=0.0000,rel=1.1633,pert=0.0760 h32:acc=0.0000,rel=1.1987,pert=0.0094 h64:acc=0.0000,rel=1.1657,pert=0.0002

### R_H5_soft

- 0x5A60 **FAIL** — R_H5_soft pending=false region=0 acc1=0.0039 rel_err1=0.3134 acc_h16=0.0000 rel_h8=1.1193 pert_h8=0.2425 bias_norm=-0.22644 bias_ang=-0.44915 bias_rel=0.50857 sigma_gap=0.1411 dstep_vs_E3=0.05998 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0039 fresh_q=0 train_s=0.22 h1:acc=0.0000,rel=0.5174,pert=0.7828 h2:acc=0.0000,rel=0.5664,pert=0.6325 h4:acc=0.0000,rel=0.8527,pert=0.4432 h8:acc=0.0000,rel=1.1193,pert=0.2425 h16:acc=0.0000,rel=1.2255,pert=0.0818 h32:acc=0.0000,rel=1.5307,pert=0.0087 h64:acc=0.0000,rel=2.0551,pert=0.0001

### R_H6_semigroup

- 0x5A60 **FAIL** — R_H6_semigroup pending=false region=0 acc1=0.0000 rel_err1=0.3158 acc_h16=0.0000 rel_h8=1.0665 pert_h8=0.1523 bias_norm=-0.27864 bias_ang=-0.44811 bias_rel=0.52114 sigma_gap=0.1725 dstep_vs_E3=0.08091 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.74 h1:acc=0.0000,rel=0.5282,pert=0.7273 h2:acc=0.0000,rel=0.6104,pert=0.5569 h4:acc=0.0000,rel=0.8383,pert=0.3453 h8:acc=0.0000,rel=1.0665,pert=0.1523 h16:acc=0.0000,rel=1.1339,pert=0.0497 h32:acc=0.0000,rel=1.1499,pert=0.0068 h64:acc=0.0000,rel=1.1041,pert=0.0001

### R_H7_normal

- 0x5A60 **FAIL** — R_H7_normal pending=false region=0 acc1=0.0000 rel_err1=0.3143 acc_h16=0.0000 rel_h8=1.0665 pert_h8=0.2017 bias_norm=-0.22423 bias_ang=-0.44916 bias_rel=0.50015 sigma_gap=0.1372 dstep_vs_E3=0.00045 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.45 h1:acc=0.0000,rel=0.5076,pert=0.7764 h2:acc=0.0000,rel=0.5460,pert=0.6068 h4:acc=0.0000,rel=0.8075,pert=0.4015 h8:acc=0.0000,rel=1.0665,pert=0.2017 h16:acc=0.0000,rel=1.1625,pert=0.0772 h32:acc=0.0000,rel=1.1983,pert=0.0097 h64:acc=0.0000,rel=1.1625,pert=0.0002

### R_V6_all

- 0x5A60 **FAIL** — R_V6_all pending=false region=0 acc1=0.0039 rel_err1=0.3200 acc_h16=0.0000 rel_h8=1.1297 pert_h8=0.1496 bias_norm=-0.29729 bias_ang=-0.42331 bias_rel=0.53537 sigma_gap=0.1894 dstep_vs_E3=0.11715 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0039 fresh_q=0 train_s=1.36 h1:acc=0.0000,rel=0.5430,pert=0.7049 h2:acc=0.0000,rel=0.6470,pert=0.5519 h4:acc=0.0000,rel=0.8900,pert=0.3435 h8:acc=0.0000,rel=1.1297,pert=0.1496 h16:acc=0.0000,rel=1.2270,pert=0.0513 h32:acc=0.0000,rel=1.5388,pert=0.0057 h64:acc=0.0000,rel=2.2492,pert=0.0001

### R_V6_all_H2

- 0x5A60 **FAIL** — R_V6_all_H2 pending=true region=0 acc1=0.0000 rel_err1=0.3377 acc_h16=0.0000 rel_h8=1.5041 pert_h8=0.1278 bias_norm=-0.25618 bias_ang=-0.30905 bias_rel=0.49766 sigma_gap=0.1719 dstep_vs_E3=0.16685 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.70 h1:acc=0.0000,rel=0.5040,pert=0.7256 h2:acc=0.0000,rel=0.6210,pert=0.5138 h4:acc=0.0000,rel=0.9422,pert=0.2984 h8:acc=0.0000,rel=1.5041,pert=0.1278 h16:acc=0.0000,rel=3.0280,pert=0.0391 h32:acc=0.0000,rel=3.7890,pert=0.0005 h64:acc=0.0000,rel=4.2494,pert=0.0000

### R_V6_all_mlp

- 0x5A60 **FAIL** — R_V6_all_mlp pending=false region=0 acc1=0.0000 rel_err1=0.3148 acc_h16=0.0000 rel_h8=1.1487 pert_h8=0.2263 bias_norm=-0.22612 bias_ang=-0.43642 bias_rel=0.50226 sigma_gap=0.1813 dstep_vs_E3=0.06966 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=0.75 h1:acc=0.0000,rel=0.5101,pert=0.7653 h2:acc=0.0000,rel=0.5623,pert=0.6140 h4:acc=0.0000,rel=0.8548,pert=0.4230 h8:acc=0.0000,rel=1.1487,pert=0.2263 h16:acc=0.0000,rel=1.2806,pert=0.0744 h32:acc=0.0000,rel=1.3182,pert=0.0084 h64:acc=0.0000,rel=1.3394,pert=0.0001

### R_V6_shuffled

- 0x5A60 **FAIL** — R_V6_shuffled pending=false region=0 acc1=0.0000 rel_err1=0.9034 acc_h16=0.0000 rel_h8=1.0232 pert_h8=0.0000 bias_norm=-0.79399 bias_ang=0.48999 bias_rel=0.95197 sigma_gap=0.7316 dstep_vs_E3=0.81875 clean=true leak=0 q_test=0 fresh_ok=true fresh_acc=0.0000 fresh_q=0 train_s=1.20 h1:acc=0.0000,rel=0.9556,pert=0.1735 h2:acc=0.0000,rel=1.0102,pert=0.0235 h4:acc=0.0000,rel=1.0261,pert=0.0004 h8:acc=0.0000,rel=1.0232,pert=0.0000 h16:acc=0.0000,rel=1.0148,pert=0.0000 h32:acc=0.0000,rel=1.0263,pert=0.0000 h64:acc=0.0000,rel=1.0231,pert=0.0000

