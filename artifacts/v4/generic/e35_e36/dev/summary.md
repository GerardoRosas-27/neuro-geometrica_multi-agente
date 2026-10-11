# v4 dev summary

- git SHA: `9cb7f5d2b71cea33bf8b75b1978a71122a2a6659` (dirty=false)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 16
- manifests: `dataset_manifest.jsonl` (sha256 of file: `7b70a417efe5313d6238dc01cdcd645faeeead7cd32b0bc0975cf80e14a92aa3`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| E35 | 14 | 2 | 16 |
| E36 | 10 | 6 | 16 |

## Per-seed gate notes

### E35

- 0xA400 **PASS** — A=0.1875 B=0.2695 diff=0.0820 relerrA=0.0817 relerrB=0.0692 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA401 **PASS** — A=0.2695 B=0.4023 diff=0.1328 relerrA=0.0790 relerrB=0.0649 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA402 **PASS** — A=0.2109 B=0.3047 diff=0.0938 relerrA=0.0793 relerrB=0.0727 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA403 **PASS** — A=0.1523 B=0.3945 diff=0.2422 relerrA=0.0866 relerrB=0.0697 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA404 **FAIL** — A=0.2734 B=0.2344 diff=-0.0391 relerrA=0.0790 relerrB=0.0752 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA405 **PASS** — A=0.2227 B=0.3555 diff=0.1328 relerrA=0.0793 relerrB=0.0662 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA406 **FAIL** — A=0.2539 B=0.2344 diff=-0.0195 relerrA=0.0714 relerrB=0.0693 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA407 **PASS** — A=0.1836 B=0.2617 diff=0.0781 relerrA=0.0808 relerrB=0.0693 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA408 **PASS** — A=0.1328 B=0.3672 diff=0.2344 relerrA=0.0868 relerrB=0.0637 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA409 **PASS** — A=0.2148 B=0.2734 diff=0.0586 relerrA=0.0779 relerrB=0.0703 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40A **PASS** — A=0.2461 B=0.4766 diff=0.2305 relerrA=0.0784 relerrB=0.0601 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40B **PASS** — A=0.3086 B=0.4023 diff=0.0938 relerrA=0.0667 relerrB=0.0645 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40C **PASS** — A=0.2383 B=0.4531 diff=0.2148 relerrA=0.0834 relerrB=0.0650 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40D **PASS** — A=0.2812 B=0.3828 diff=0.1016 relerrA=0.0793 relerrB=0.0657 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40E **PASS** — A=0.2852 B=0.4922 diff=0.2070 relerrA=0.0737 relerrB=0.0604 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40F **PASS** — A=0.2422 B=0.4492 diff=0.2070 relerrA=0.0802 relerrB=0.0616 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false

### E36

- 0xA400 **FAIL** — C=0.270 B_raw=0.305 A=0.188 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.008 G_shuffled_order=0.000
- 0xA401 **PASS** — C=0.402 B_raw=0.180 A=0.270 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.020 G_shuffled_order=0.000
- 0xA402 **FAIL** — C=0.305 B_raw=0.336 A=0.211 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.004 G_shuffled_order=0.000
- 0xA403 **PASS** — C=0.395 B_raw=0.234 A=0.152 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.031 G_shuffled_order=0.000
- 0xA404 **FAIL** — C=0.234 B_raw=0.289 A=0.273 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.020 G_shuffled_order=0.000
- 0xA405 **FAIL** — C=0.355 B_raw=0.332 A=0.223 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.016 G_shuffled_order=0.000
- 0xA406 **FAIL** — C=0.234 B_raw=0.328 A=0.254 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.004 G_shuffled_order=0.000
- 0xA407 **FAIL** — C=0.262 B_raw=0.281 A=0.184 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.004 G_shuffled_order=0.000
- 0xA408 **PASS** — C=0.367 B_raw=0.191 A=0.133 D_corrupt_cdt=0.008 E_irrelevant_cdt=0.000 F_stats_no_topology=0.004 G_shuffled_order=0.000
- 0xA409 **PASS** — C=0.273 B_raw=0.199 A=0.215 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.000 G_shuffled_order=0.000
- 0xA40A **PASS** — C=0.477 B_raw=0.211 A=0.246 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.004 G_shuffled_order=0.000
- 0xA40B **PASS** — C=0.402 B_raw=0.324 A=0.309 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.012 G_shuffled_order=0.000
- 0xA40C **PASS** — C=0.453 B_raw=0.266 A=0.238 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.051 G_shuffled_order=0.000
- 0xA40D **PASS** — C=0.383 B_raw=0.254 A=0.281 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.027 G_shuffled_order=0.000
- 0xA40E **PASS** — C=0.492 B_raw=0.234 A=0.285 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.020 G_shuffled_order=0.000
- 0xA40F **PASS** — C=0.449 B_raw=0.246 A=0.242 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.008 G_shuffled_order=0.000

