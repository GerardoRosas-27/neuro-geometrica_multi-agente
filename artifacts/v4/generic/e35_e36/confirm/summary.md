# v4 confirm summary

- git SHA: `9cb7f5d2b71cea33bf8b75b1978a71122a2a6659` (dirty=true)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 16
- manifests: `dataset_manifest.jsonl` (sha256 of file: `89c0207ff0e479c2aa0e37446cbbb20eb2ffcb5bf41dacb11f42555d52d8188e`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| E35 | 12 | 4 | 16 |
| E36 | 12 | 4 | 16 |

## Per-seed gate notes

### E35

- 0xB400 **FAIL** — A=0.2969 B=0.3086 diff=0.0117 relerrA=0.0747 relerrB=0.0705 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB401 **PASS** — A=0.3008 B=0.4414 diff=0.1406 relerrA=0.0779 relerrB=0.0619 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB402 **PASS** — A=0.1914 B=0.5039 diff=0.3125 relerrA=0.0857 relerrB=0.0635 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB403 **PASS** — A=0.3398 B=0.4609 diff=0.1211 relerrA=0.0690 relerrB=0.0618 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB404 **PASS** — A=0.2383 B=0.3477 diff=0.1094 relerrA=0.0799 relerrB=0.0693 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB405 **PASS** — A=0.2383 B=0.4648 diff=0.2266 relerrA=0.0804 relerrB=0.0650 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB406 **PASS** — A=0.2695 B=0.4102 diff=0.1406 relerrA=0.0696 relerrB=0.0624 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB407 **FAIL** — A=0.3047 B=0.3086 diff=0.0039 relerrA=0.0686 relerrB=0.0649 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB408 **PASS** — A=0.2188 B=0.4609 diff=0.2422 relerrA=0.0856 relerrB=0.0609 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB409 **PASS** — A=0.2383 B=0.3984 diff=0.1602 relerrA=0.0782 relerrB=0.0723 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40A **PASS** — A=0.1953 B=0.4688 diff=0.2734 relerrA=0.0884 relerrB=0.0638 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40B **PASS** — A=0.1562 B=0.2383 diff=0.0820 relerrA=0.0895 relerrB=0.0725 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40C **FAIL** — A=0.3633 B=0.3906 diff=0.0273 relerrA=0.0668 relerrB=0.0615 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40D **FAIL** — A=0.1992 B=0.2070 diff=0.0078 relerrA=0.0793 relerrB=0.0801 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40E **PASS** — A=0.2227 B=0.4453 diff=0.2227 relerrA=0.0783 relerrB=0.0617 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40F **PASS** — A=0.1953 B=0.3633 diff=0.1680 relerrA=0.0779 relerrB=0.0654 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false

### E36

- 0xB400 **FAIL** — C=0.309 B_raw=0.273 A=0.297 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.016 G_shuffled_order=0.000
- 0xB401 **PASS** — C=0.441 B_raw=0.219 A=0.301 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.023 G_shuffled_order=0.000
- 0xB402 **PASS** — C=0.504 B_raw=0.223 A=0.191 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.012 G_shuffled_order=0.000
- 0xB403 **PASS** — C=0.461 B_raw=0.211 A=0.340 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.012 G_shuffled_order=0.000
- 0xB404 **PASS** — C=0.348 B_raw=0.266 A=0.238 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.023 G_shuffled_order=0.000
- 0xB405 **PASS** — C=0.465 B_raw=0.355 A=0.238 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.020 G_shuffled_order=0.000
- 0xB406 **PASS** — C=0.410 B_raw=0.195 A=0.270 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.012 G_shuffled_order=0.000
- 0xB407 **PASS** — C=0.309 B_raw=0.223 A=0.305 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.016 G_shuffled_order=0.000
- 0xB408 **PASS** — C=0.461 B_raw=0.223 A=0.219 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.008 G_shuffled_order=0.000
- 0xB409 **FAIL** — C=0.398 B_raw=0.352 A=0.238 D_corrupt_cdt=0.008 E_irrelevant_cdt=0.000 F_stats_no_topology=0.008 G_shuffled_order=0.000
- 0xB40A **PASS** — C=0.469 B_raw=0.133 A=0.195 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.016 G_shuffled_order=0.000
- 0xB40B **PASS** — C=0.238 B_raw=0.164 A=0.156 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.012 G_shuffled_order=0.000
- 0xB40C **FAIL** — C=0.391 B_raw=0.395 A=0.363 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.016 G_shuffled_order=0.000
- 0xB40D **FAIL** — C=0.207 B_raw=0.375 A=0.199 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.000 G_shuffled_order=0.000
- 0xB40E **PASS** — C=0.445 B_raw=0.309 A=0.223 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.016 G_shuffled_order=0.000
- 0xB40F **PASS** — C=0.363 B_raw=0.309 A=0.195 D_corrupt_cdt=0.008 E_irrelevant_cdt=0.000 F_stats_no_topology=0.004 G_shuffled_order=0.000

