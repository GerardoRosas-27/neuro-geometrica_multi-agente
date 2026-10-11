# v4 dev summary

- git SHA: `5c558fefd8b21476dbcce94d32728891ae25571a` (dirty=false)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 16
- manifests: `dataset_manifest.jsonl` (sha256 of file: `7b70a417efe5313d6238dc01cdcd645faeeead7cd32b0bc0975cf80e14a92aa3`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| E31 | 16 | 0 | 16 |
| E32 | 16 | 0 | 16 |
| E33 | 0 | 16 | 16 |
| E34 | 5 | 11 | 16 |
| E35 | 11 | 5 | 16 |
| E36 | 10 | 6 | 16 |
| E37 | 11 | 5 | 16 |
| E38 | 11 | 5 | 16 |
| E39 | 13 | 3 | 16 |
| E40 | 16 | 0 | 16 |
| E41 | 16 | 0 | 16 |
| E42 | 4 | 12 | 16 |

## Per-seed gate notes

### E31

- 0xA400 **PASS** — families_won=6 [translate rotate scale shear affine compose_rot_scale] dphi_acc=0.188 best_ctrl_acc=0.059
- 0xA401 **PASS** — families_won=6 [translate rotate scale shear affine compose_rot_scale] dphi_acc=0.270 best_ctrl_acc=0.039
- 0xA402 **PASS** — families_won=7 [translate reflect scale shear affine compose_rot_scale radial_warp] dphi_acc=0.211 best_ctrl_acc=0.043
- 0xA403 **PASS** — families_won=5 [translate scale shear affine compose_rot_scale] dphi_acc=0.152 best_ctrl_acc=0.047
- 0xA404 **PASS** — families_won=7 [translate rotate reflect scale shear affine compose_rot_scale] dphi_acc=0.273 best_ctrl_acc=0.039
- 0xA405 **PASS** — families_won=7 [translate rotate scale shear affine compose_rot_scale radial_warp] dphi_acc=0.223 best_ctrl_acc=0.020
- 0xA406 **PASS** — families_won=7 [translate rotate reflect scale shear affine compose_rot_scale] dphi_acc=0.254 best_ctrl_acc=0.016
- 0xA407 **PASS** — families_won=5 [scale shear affine compose_rot_scale radial_warp] dphi_acc=0.184 best_ctrl_acc=0.051
- 0xA408 **PASS** — families_won=5 [translate scale shear affine compose_rot_scale] dphi_acc=0.133 best_ctrl_acc=0.035
- 0xA409 **PASS** — families_won=6 [translate rotate scale shear affine compose_rot_scale] dphi_acc=0.215 best_ctrl_acc=0.066
- 0xA40A **PASS** — families_won=6 [translate rotate scale shear affine compose_rot_scale] dphi_acc=0.246 best_ctrl_acc=0.035
- 0xA40B **PASS** — families_won=7 [translate rotate reflect scale shear affine compose_rot_scale] dphi_acc=0.309 best_ctrl_acc=0.031
- 0xA40C **PASS** — families_won=5 [translate rotate shear affine compose_rot_scale] dphi_acc=0.238 best_ctrl_acc=0.055
- 0xA40D **PASS** — families_won=7 [translate rotate reflect scale shear affine compose_rot_scale] dphi_acc=0.281 best_ctrl_acc=0.047
- 0xA40E **PASS** — families_won=6 [translate rotate scale shear affine compose_rot_scale] dphi_acc=0.285 best_ctrl_acc=0.035
- 0xA40F **PASS** — families_won=6 [translate rotate scale shear affine compose_rot_scale] dphi_acc=0.242 best_ctrl_acc=0.035

### E32

- 0xA400 **PASS** — fams_sigma_within10%=4 frac_steps_sigma>1=1.000 translate:σD=1.102/σT=1.000 rotate:σD=1.094/σT=1.000 reflect:σD=1.128/σT=1.000 scale:σD=1.189/σT=1.087 shear:σD=1.191/σT=1.115 affine:σD=1.202/σT=1.093 compose_rot_scale:σD=1.135/σT=1.018 radial_warp:σD=1.244/σT=1.224
- 0xA401 **PASS** — fams_sigma_within10%=3 frac_steps_sigma>1=0.993 translate:σD=1.113/σT=1.000 rotate:σD=1.124/σT=1.000 reflect:σD=1.124/σT=1.000 scale:σD=1.177/σT=1.132 shear:σD=1.185/σT=1.112 affine:σD=1.187/σT=1.082 compose_rot_scale:σD=0.995/σT=0.878 radial_warp:σD=1.129/σT=1.259
- 0xA402 **PASS** — fams_sigma_within10%=5 frac_steps_sigma>1=1.000 translate:σD=1.094/σT=1.000 rotate:σD=1.118/σT=1.000 reflect:σD=1.112/σT=1.000 scale:σD=1.218/σT=1.160 shear:σD=1.209/σT=1.142 affine:σD=1.248/σT=1.112 compose_rot_scale:σD=1.121/σT=1.075 radial_warp:σD=1.147/σT=1.179
- 0xA403 **PASS** — fams_sigma_within10%=4 frac_steps_sigma>1=0.941 translate:σD=1.077/σT=1.000 rotate:σD=1.109/σT=1.000 reflect:σD=1.159/σT=1.000 scale:σD=1.191/σT=1.110 shear:σD=1.163/σT=1.094 affine:σD=1.216/σT=1.093 compose_rot_scale:σD=1.122/σT=0.930 radial_warp:σD=1.137/σT=1.187
- 0xA404 **PASS** — fams_sigma_within10%=5 frac_steps_sigma>1=0.995 translate:σD=1.100/σT=1.000 rotate:σD=1.130/σT=1.000 reflect:σD=1.092/σT=1.000 scale:σD=1.173/σT=1.088 shear:σD=1.157/σT=1.109 affine:σD=1.205/σT=1.081 compose_rot_scale:σD=1.078/σT=0.957 radial_warp:σD=1.135/σT=1.203
- 0xA405 **PASS** — fams_sigma_within10%=7 frac_steps_sigma>1=0.938 translate:σD=1.080/σT=1.000 rotate:σD=1.080/σT=1.000 reflect:σD=1.131/σT=1.000 scale:σD=1.214/σT=1.131 shear:σD=1.228/σT=1.158 affine:σD=1.177/σT=1.076 compose_rot_scale:σD=1.094/σT=1.008 radial_warp:σD=1.159/σT=1.200
- 0xA406 **PASS** — fams_sigma_within10%=7 frac_steps_sigma>1=0.981 translate:σD=1.099/σT=1.000 rotate:σD=1.072/σT=1.000 reflect:σD=1.128/σT=1.000 scale:σD=1.206/σT=1.129 shear:σD=1.135/σT=1.098 affine:σD=1.195/σT=1.089 compose_rot_scale:σD=1.102/σT=1.030 radial_warp:σD=1.138/σT=1.225
- 0xA407 **PASS** — fams_sigma_within10%=7 frac_steps_sigma>1=0.997 translate:σD=1.088/σT=1.000 rotate:σD=1.059/σT=1.000 reflect:σD=1.179/σT=1.000 scale:σD=1.167/σT=1.135 shear:σD=1.144/σT=1.094 affine:σD=1.167/σT=1.079 compose_rot_scale:σD=1.116/σT=1.088 radial_warp:σD=1.110/σT=1.238
- 0xA408 **PASS** — fams_sigma_within10%=7 frac_steps_sigma>1=0.957 translate:σD=1.065/σT=1.000 rotate:σD=1.076/σT=1.000 reflect:σD=1.121/σT=1.000 scale:σD=1.127/σT=1.105 shear:σD=1.197/σT=1.150 affine:σD=1.141/σT=1.069 compose_rot_scale:σD=1.107/σT=1.012 radial_warp:σD=1.172/σT=1.236
- 0xA409 **PASS** — fams_sigma_within10%=6 frac_steps_sigma>1=0.999 translate:σD=1.088/σT=1.000 rotate:σD=1.098/σT=1.000 reflect:σD=1.125/σT=1.000 scale:σD=1.220/σT=1.101 shear:σD=1.210/σT=1.101 affine:σD=1.224/σT=1.075 compose_rot_scale:σD=1.041/σT=0.934 radial_warp:σD=1.152/σT=1.224
- 0xA40A **PASS** — fams_sigma_within10%=8 frac_steps_sigma>1=1.000 translate:σD=1.075/σT=1.000 rotate:σD=1.064/σT=1.000 reflect:σD=1.090/σT=1.000 scale:σD=1.206/σT=1.141 shear:σD=1.179/σT=1.104 affine:σD=1.173/σT=1.058 compose_rot_scale:σD=1.122/σT=0.986 radial_warp:σD=1.191/σT=1.230
- 0xA40B **PASS** — fams_sigma_within10%=6 frac_steps_sigma>1=1.000 translate:σD=1.104/σT=1.000 rotate:σD=1.097/σT=1.000 reflect:σD=1.141/σT=1.000 scale:σD=1.156/σT=1.101 shear:σD=1.178/σT=1.167 affine:σD=1.210/σT=1.086 compose_rot_scale:σD=1.058/σT=1.003 radial_warp:σD=1.163/σT=1.217
- 0xA40C **PASS** — fams_sigma_within10%=6 frac_steps_sigma>1=0.953 translate:σD=1.103/σT=1.000 rotate:σD=1.088/σT=1.000 reflect:σD=1.173/σT=1.000 scale:σD=1.237/σT=1.123 shear:σD=1.154/σT=1.122 affine:σD=1.161/σT=1.064 compose_rot_scale:σD=1.128/σT=1.045 radial_warp:σD=1.155/σT=1.284
- 0xA40D **PASS** — fams_sigma_within10%=6 frac_steps_sigma>1=0.994 translate:σD=1.080/σT=1.000 rotate:σD=1.091/σT=1.000 reflect:σD=1.105/σT=1.000 scale:σD=1.151/σT=1.099 shear:σD=1.187/σT=1.103 affine:σD=1.234/σT=1.102 compose_rot_scale:σD=1.097/σT=1.010 radial_warp:σD=1.151/σT=1.189
- 0xA40E **PASS** — fams_sigma_within10%=6 frac_steps_sigma>1=0.996 translate:σD=1.074/σT=1.000 rotate:σD=1.103/σT=1.000 reflect:σD=1.093/σT=1.000 scale:σD=1.238/σT=1.144 shear:σD=1.179/σT=1.091 affine:σD=1.192/σT=1.105 compose_rot_scale:σD=1.043/σT=0.964 radial_warp:σD=1.159/σT=1.204
- 0xA40F **PASS** — fams_sigma_within10%=4 frac_steps_sigma>1=0.983 translate:σD=1.117/σT=1.000 rotate:σD=1.106/σT=1.000 reflect:σD=1.074/σT=1.000 scale:σD=1.168/σT=1.085 shear:σD=1.179/σT=1.100 affine:σD=1.186/σT=1.084 compose_rot_scale:σD=1.100/σT=0.974 radial_warp:σD=1.139/σT=1.177

### E33

- 0xA400 **FAIL** — stability_region_h=0 iterable/all h1:0.09/0.19 h2:0.00/0.01 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.01 h64:0.00/0.01
- 0xA401 **FAIL** — stability_region_h=0 iterable/all h1:0.11/0.27 h2:0.02/0.02 h4:0.00/0.00 h8:0.00/0.01 h16:0.00/0.01 h32:0.00/0.01 h64:0.00/0.01
- 0xA402 **FAIL** — stability_region_h=0 iterable/all h1:0.09/0.21 h2:0.00/0.02 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA403 **FAIL** — stability_region_h=0 iterable/all h1:0.02/0.15 h2:0.00/0.00 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA404 **FAIL** — stability_region_h=0 iterable/all h1:0.09/0.27 h2:0.00/0.03 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA405 **FAIL** — stability_region_h=0 iterable/all h1:0.14/0.22 h2:0.00/0.02 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA406 **FAIL** — stability_region_h=0 iterable/all h1:0.25/0.25 h2:0.00/0.01 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA407 **FAIL** — stability_region_h=0 iterable/all h1:0.11/0.18 h2:0.02/0.00 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA408 **FAIL** — stability_region_h=0 iterable/all h1:0.09/0.13 h2:0.00/0.00 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA409 **FAIL** — stability_region_h=0 iterable/all h1:0.20/0.21 h2:0.00/0.01 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.01
- 0xA40A **FAIL** — stability_region_h=0 iterable/all h1:0.11/0.25 h2:0.00/0.04 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.01 h64:0.00/0.01
- 0xA40B **FAIL** — stability_region_h=0 iterable/all h1:0.25/0.31 h2:0.00/0.03 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA40C **FAIL** — stability_region_h=0 iterable/all h1:0.12/0.24 h2:0.00/0.01 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA40D **FAIL** — stability_region_h=0 iterable/all h1:0.09/0.28 h2:0.00/0.00 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA40E **FAIL** — stability_region_h=0 iterable/all h1:0.11/0.29 h2:0.05/0.04 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xA40F **FAIL** — stability_region_h=0 iterable/all h1:0.14/0.24 h2:0.00/0.02 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.01 h64:0.00/0.01

### E34

- 0xA400 **FAIL** — dphi_h1=0.188 mlp_h1=0.203 dphi_h4=0.000 mlp_h4=0.000 static=0.059 params=962/962
- 0xA401 **FAIL** — dphi_h1=0.270 mlp_h1=0.266 dphi_h4=0.000 mlp_h4=0.000 static=0.039 params=962/962
- 0xA402 **FAIL** — dphi_h1=0.211 mlp_h1=0.254 dphi_h4=0.000 mlp_h4=0.000 static=0.043 params=962/962
- 0xA403 **FAIL** — dphi_h1=0.152 mlp_h1=0.246 dphi_h4=0.000 mlp_h4=0.000 static=0.047 params=962/962
- 0xA404 **FAIL** — dphi_h1=0.273 mlp_h1=0.277 dphi_h4=0.000 mlp_h4=0.000 static=0.039 params=962/962
- 0xA405 **FAIL** — dphi_h1=0.223 mlp_h1=0.211 dphi_h4=0.000 mlp_h4=0.000 static=0.020 params=962/962
- 0xA406 **FAIL** — dphi_h1=0.254 mlp_h1=0.297 dphi_h4=0.000 mlp_h4=0.000 static=0.016 params=962/962
- 0xA407 **PASS** — dphi_h1=0.184 mlp_h1=0.098 dphi_h4=0.000 mlp_h4=0.000 static=0.051 params=962/962
- 0xA408 **FAIL** — dphi_h1=0.133 mlp_h1=0.121 dphi_h4=0.000 mlp_h4=0.000 static=0.035 params=962/962
- 0xA409 **PASS** — dphi_h1=0.215 mlp_h1=0.082 dphi_h4=0.000 mlp_h4=0.000 static=0.066 params=962/962
- 0xA40A **FAIL** — dphi_h1=0.246 mlp_h1=0.316 dphi_h4=0.000 mlp_h4=0.000 static=0.035 params=962/962
- 0xA40B **FAIL** — dphi_h1=0.309 mlp_h1=0.281 dphi_h4=0.000 mlp_h4=0.000 static=0.031 params=962/962
- 0xA40C **PASS** — dphi_h1=0.238 mlp_h1=0.137 dphi_h4=0.000 mlp_h4=0.000 static=0.055 params=962/962
- 0xA40D **PASS** — dphi_h1=0.281 mlp_h1=0.207 dphi_h4=0.000 mlp_h4=0.000 static=0.047 params=962/962
- 0xA40E **PASS** — dphi_h1=0.285 mlp_h1=0.211 dphi_h4=0.000 mlp_h4=0.000 static=0.035 params=962/962
- 0xA40F **FAIL** — dphi_h1=0.242 mlp_h1=0.266 dphi_h4=0.000 mlp_h4=0.000 static=0.035 params=962/962

### E35

- 0xA400 **PASS** — A=0.1875 B=0.3984 diff=0.2109 relerrA=0.0817 relerrB=0.0638 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA401 **PASS** — A=0.2695 B=0.3711 diff=0.1016 relerrA=0.0790 relerrB=0.0673 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA402 **PASS** — A=0.2109 B=0.3672 diff=0.1562 relerrA=0.0793 relerrB=0.0698 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA403 **PASS** — A=0.1523 B=0.5039 diff=0.3516 relerrA=0.0866 relerrB=0.0625 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA404 **PASS** — A=0.2734 B=0.4492 diff=0.1758 relerrA=0.0790 relerrB=0.0714 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA405 **PASS** — A=0.2227 B=0.3906 diff=0.1680 relerrA=0.0793 relerrB=0.0679 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA406 **FAIL** — A=0.2539 B=0.2227 diff=-0.0312 relerrA=0.0714 relerrB=0.0754 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA407 **PASS** — A=0.1836 B=0.3672 diff=0.1836 relerrA=0.0808 relerrB=0.0727 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA408 **PASS** — A=0.1328 B=0.2461 diff=0.1133 relerrA=0.0868 relerrB=0.0754 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA409 **PASS** — A=0.2148 B=0.4023 diff=0.1875 relerrA=0.0779 relerrB=0.0696 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40A **FAIL** — A=0.2461 B=0.2422 diff=-0.0039 relerrA=0.0784 relerrB=0.0723 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40B **PASS** — A=0.3086 B=0.3984 diff=0.0898 relerrA=0.0667 relerrB=0.0632 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40C **FAIL** — A=0.2383 B=0.2500 diff=0.0117 relerrA=0.0834 relerrB=0.0736 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40D **FAIL** — A=0.2812 B=0.2266 diff=-0.0547 relerrA=0.0793 relerrB=0.0779 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40E **FAIL** — A=0.2852 B=0.1875 diff=-0.0977 relerrA=0.0737 relerrB=0.0790 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xA40F **PASS** — A=0.2422 B=0.3945 diff=0.1523 relerrA=0.0802 relerrB=0.0657 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false

### E36

- 0xA400 **PASS** — C=0.398 B_raw=0.305 A=0.188 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.102 G_shuffled_order=0.000
- 0xA401 **PASS** — C=0.371 B_raw=0.180 A=0.270 D_corrupt_cdt=0.008 E_irrelevant_cdt=0.004 F_stats_no_topology=0.016 G_shuffled_order=0.000
- 0xA402 **FAIL** — C=0.367 B_raw=0.336 A=0.211 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.035 G_shuffled_order=0.000
- 0xA403 **PASS** — C=0.504 B_raw=0.234 A=0.152 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.102 G_shuffled_order=0.000
- 0xA404 **PASS** — C=0.449 B_raw=0.289 A=0.273 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.004 F_stats_no_topology=0.066 G_shuffled_order=0.000
- 0xA405 **PASS** — C=0.391 B_raw=0.332 A=0.223 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.059 G_shuffled_order=0.000
- 0xA406 **FAIL** — C=0.223 B_raw=0.328 A=0.254 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.008 G_shuffled_order=0.000
- 0xA407 **PASS** — C=0.367 B_raw=0.281 A=0.184 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.023 G_shuffled_order=0.000
- 0xA408 **PASS** — C=0.246 B_raw=0.191 A=0.133 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.016 G_shuffled_order=0.000
- 0xA409 **PASS** — C=0.402 B_raw=0.199 A=0.215 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.027 G_shuffled_order=0.000
- 0xA40A **FAIL** — C=0.242 B_raw=0.211 A=0.246 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.027 G_shuffled_order=0.000
- 0xA40B **PASS** — C=0.398 B_raw=0.324 A=0.309 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.109 G_shuffled_order=0.000
- 0xA40C **FAIL** — C=0.250 B_raw=0.266 A=0.238 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.121 G_shuffled_order=0.000
- 0xA40D **FAIL** — C=0.227 B_raw=0.254 A=0.281 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.105 G_shuffled_order=0.000
- 0xA40E **FAIL** — C=0.188 B_raw=0.234 A=0.285 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.055 G_shuffled_order=0.000
- 0xA40F **PASS** — C=0.395 B_raw=0.246 A=0.242 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.035 G_shuffled_order=0.000

### E37

- 0xA400 **PASS** — pre_delete=0.3984 post_delete_fresh_process=0.3984 A=0.1875 fresh_process_queries=0 ckpt_sha=bfec0525c5ee8a1299b21ecf9687fe9cf78493d9545331be51f9f85ccd25b218
- 0xA401 **PASS** — pre_delete=0.3711 post_delete_fresh_process=0.3711 A=0.2695 fresh_process_queries=0 ckpt_sha=13bb3481dee9f052a45a03f73543774e64d8e7b45f821fa88afbca2a3be6e65d
- 0xA402 **PASS** — pre_delete=0.3672 post_delete_fresh_process=0.3672 A=0.2109 fresh_process_queries=0 ckpt_sha=bf341581419a766a49d454d8c9ad93ef18576efec5f39ccd2b21756ca9254952
- 0xA403 **PASS** — pre_delete=0.5039 post_delete_fresh_process=0.5039 A=0.1523 fresh_process_queries=0 ckpt_sha=bd5a8ea921c5e1b0e26d7dc781b12a7c35530b1259f7a5cb986a860916016249
- 0xA404 **PASS** — pre_delete=0.4492 post_delete_fresh_process=0.4492 A=0.2734 fresh_process_queries=0 ckpt_sha=49e4b584e4568ed02e3464d269a88a46cf1a51d1e38c948b726db359f4c2780d
- 0xA405 **PASS** — pre_delete=0.3906 post_delete_fresh_process=0.3906 A=0.2227 fresh_process_queries=0 ckpt_sha=ea7f1f1ba6d5f1b9373ff6ecb55b426022615c283814dd8f418837621156f599
- 0xA406 **FAIL** — pre_delete=0.2227 post_delete_fresh_process=0.2227 A=0.2539 fresh_process_queries=0 ckpt_sha=f8c86912bdc61b1cbf16df87683e0f10aad79fed26e086752e966109258c1608
- 0xA407 **PASS** — pre_delete=0.3672 post_delete_fresh_process=0.3672 A=0.1836 fresh_process_queries=0 ckpt_sha=94897d6396ad9547b4307e514ba4bbd70b78b787e78dc6fc6179ec2e129a1d55
- 0xA408 **PASS** — pre_delete=0.2461 post_delete_fresh_process=0.2461 A=0.1328 fresh_process_queries=0 ckpt_sha=2ceffd45f72cf43300ce82ca42a36bc1ae02beb3da91240b78c1720330524386
- 0xA409 **PASS** — pre_delete=0.4023 post_delete_fresh_process=0.4023 A=0.2148 fresh_process_queries=0 ckpt_sha=dccea18d7979080721deb9d6e62f093a12b441664a9379a2504e8f8a9af53a1c
- 0xA40A **FAIL** — pre_delete=0.2422 post_delete_fresh_process=0.2422 A=0.2461 fresh_process_queries=0 ckpt_sha=3ef582395ce756473d9db63974e8323dc266e8aab422be60b952a993dab08208
- 0xA40B **PASS** — pre_delete=0.3984 post_delete_fresh_process=0.3984 A=0.3086 fresh_process_queries=0 ckpt_sha=217f6fe566a394e5917826b150790c3f1756408b2e5ab6d11db86372672200c0
- 0xA40C **FAIL** — pre_delete=0.2500 post_delete_fresh_process=0.2500 A=0.2383 fresh_process_queries=0 ckpt_sha=ebc9f805ddceded731d1306752d04b9b017182b61343b07ca699b4d0ebbe5a81
- 0xA40D **FAIL** — pre_delete=0.2266 post_delete_fresh_process=0.2266 A=0.2812 fresh_process_queries=0 ckpt_sha=75ff9ddafa937023b807995694f73dddc62e50bd0e3daf8c3e907ef9157dd0c1
- 0xA40E **FAIL** — pre_delete=0.1875 post_delete_fresh_process=0.1875 A=0.2852 fresh_process_queries=0 ckpt_sha=966af65e941033e66207beedde07caacb2ad0cdcc4e4169a5097340917882d3a
- 0xA40F **PASS** — pre_delete=0.3945 post_delete_fresh_process=0.3945 A=0.2422 fresh_process_queries=0 ckpt_sha=500579315435df02a9e58929c0ee04208351c3f28fccde74c9fe3b97b1f15dc1

### E38

- 0xA400 **PASS** — P0=0.3984 P0_newtest=0.4453 A_newtest=0.2109 P1=0.3984(q=0) P2=P0(identity encoder) P3_lookup=0.0156(cdt_q=256) A=0.1875
- 0xA401 **PASS** — P0=0.3711 P0_newtest=0.3984 A_newtest=0.3398 P1=0.3711(q=0) P2=P0(identity encoder) P3_lookup=0.0078(cdt_q=256) A=0.2695
- 0xA402 **PASS** — P0=0.3672 P0_newtest=0.4062 A_newtest=0.1875 P1=0.3672(q=0) P2=P0(identity encoder) P3_lookup=0.0117(cdt_q=256) A=0.2109
- 0xA403 **PASS** — P0=0.5039 P0_newtest=0.5078 A_newtest=0.1719 P1=0.5039(q=0) P2=P0(identity encoder) P3_lookup=0.0117(cdt_q=256) A=0.1523
- 0xA404 **PASS** — P0=0.4492 P0_newtest=0.3945 A_newtest=0.2383 P1=0.4492(q=0) P2=P0(identity encoder) P3_lookup=0.0273(cdt_q=256) A=0.2734
- 0xA405 **PASS** — P0=0.3906 P0_newtest=0.3516 A_newtest=0.2109 P1=0.3906(q=0) P2=P0(identity encoder) P3_lookup=0.0156(cdt_q=256) A=0.2227
- 0xA406 **FAIL** — P0=0.2227 P0_newtest=0.1914 A_newtest=0.2344 P1=0.2227(q=0) P2=P0(identity encoder) P3_lookup=0.0078(cdt_q=256) A=0.2539
- 0xA407 **PASS** — P0=0.3672 P0_newtest=0.3828 A_newtest=0.1797 P1=0.3672(q=0) P2=P0(identity encoder) P3_lookup=0.0195(cdt_q=256) A=0.1836
- 0xA408 **PASS** — P0=0.2461 P0_newtest=0.3359 A_newtest=0.1328 P1=0.2461(q=0) P2=P0(identity encoder) P3_lookup=0.0078(cdt_q=256) A=0.1328
- 0xA409 **PASS** — P0=0.4023 P0_newtest=0.3555 A_newtest=0.1836 P1=0.4023(q=0) P2=P0(identity encoder) P3_lookup=0.0117(cdt_q=256) A=0.2148
- 0xA40A **FAIL** — P0=0.2422 P0_newtest=0.2500 A_newtest=0.2539 P1=0.2422(q=0) P2=P0(identity encoder) P3_lookup=0.0078(cdt_q=256) A=0.2461
- 0xA40B **PASS** — P0=0.3984 P0_newtest=0.4297 A_newtest=0.3633 P1=0.3984(q=0) P2=P0(identity encoder) P3_lookup=0.0078(cdt_q=256) A=0.3086
- 0xA40C **FAIL** — P0=0.2500 P0_newtest=0.2344 A_newtest=0.2812 P1=0.2500(q=0) P2=P0(identity encoder) P3_lookup=0.0039(cdt_q=256) A=0.2383
- 0xA40D **FAIL** — P0=0.2266 P0_newtest=0.1836 A_newtest=0.2734 P1=0.2266(q=0) P2=P0(identity encoder) P3_lookup=0.0039(cdt_q=256) A=0.2812
- 0xA40E **FAIL** — P0=0.1875 P0_newtest=0.1758 A_newtest=0.2891 P1=0.1875(q=0) P2=P0(identity encoder) P3_lookup=0.0117(cdt_q=256) A=0.2852
- 0xA40F **PASS** — P0=0.3945 P0_newtest=0.4219 A_newtest=0.2500 P1=0.3945(q=0) P2=P0(identity encoder) P3_lookup=0.0156(cdt_q=256) A=0.2422

### E39

- 0xA400 **PASS** — A=0.406 B=0.457 static=0.035
- 0xA401 **FAIL** — A=0.215 B=0.250 static=0.031
- 0xA402 **PASS** — A=0.199 B=0.387 static=0.047
- 0xA403 **FAIL** — A=0.375 B=0.398 static=0.035
- 0xA404 **PASS** — A=0.172 B=0.414 static=0.043
- 0xA405 **PASS** — A=0.195 B=0.418 static=0.055
- 0xA406 **PASS** — A=0.223 B=0.367 static=0.039
- 0xA407 **PASS** — A=0.254 B=0.359 static=0.066
- 0xA408 **PASS** — A=0.266 B=0.406 static=0.043
- 0xA409 **PASS** — A=0.188 B=0.316 static=0.051
- 0xA40A **PASS** — A=0.199 B=0.250 static=0.043
- 0xA40B **PASS** — A=0.188 B=0.359 static=0.031
- 0xA40C **PASS** — A=0.305 B=0.426 static=0.039
- 0xA40D **PASS** — A=0.160 B=0.289 static=0.043
- 0xA40E **FAIL** — A=0.195 B=0.191 static=0.035
- 0xA40F **PASS** — A=0.188 B=0.453 static=0.039

### E40

- 0xA400 **PASS** — final A=0.148 B=0.625 D=0.062 forgetting A=0.844 B=0.323 D=0.302
- 0xA401 **PASS** — final A=0.180 B=0.781 D=0.055 forgetting A=0.917 B=0.094 D=0.323
- 0xA402 **PASS** — final A=0.203 B=0.641 D=0.039 forgetting A=0.938 B=0.385 D=0.333
- 0xA403 **PASS** — final A=0.156 B=0.711 D=0.055 forgetting A=0.938 B=0.177 D=0.323
- 0xA404 **PASS** — final A=0.141 B=0.672 D=0.078 forgetting A=0.833 B=0.240 D=0.375
- 0xA405 **PASS** — final A=0.156 B=0.625 D=0.094 forgetting A=0.906 B=0.281 D=0.333
- 0xA406 **PASS** — final A=0.219 B=0.680 D=0.008 forgetting A=0.896 B=0.240 D=0.323
- 0xA407 **PASS** — final A=0.172 B=0.750 D=0.055 forgetting A=0.958 B=0.156 D=0.354
- 0xA408 **PASS** — final A=0.148 B=0.789 D=0.078 forgetting A=0.906 B=0.146 D=0.312
- 0xA409 **PASS** — final A=0.188 B=0.680 D=0.047 forgetting A=0.917 B=0.208 D=0.344
- 0xA40A **PASS** — final A=0.188 B=0.594 D=0.102 forgetting A=0.979 B=0.302 D=0.344
- 0xA40B **PASS** — final A=0.164 B=0.484 D=0.070 forgetting A=0.896 B=0.406 D=0.396
- 0xA40C **PASS** — final A=0.078 B=0.766 D=0.023 forgetting A=0.875 B=0.094 D=0.344
- 0xA40D **PASS** — final A=0.180 B=0.742 D=0.055 forgetting A=0.865 B=0.073 D=0.354
- 0xA40E **PASS** — final A=0.180 B=0.672 D=0.086 forgetting A=0.948 B=0.312 D=0.365
- 0xA40F **PASS** — final A=0.156 B=0.656 D=0.047 forgetting A=0.917 B=0.219 D=0.365

### E41

- 0xA400 **PASS** — base_rot=0.625 base_translate=0.469 structured: redirect=0.594 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA401 **PASS** — base_rot=0.406 base_translate=0.656 structured: redirect=0.625 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA402 **PASS** — base_rot=0.469 base_translate=0.438 structured: redirect=0.438 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA403 **PASS** — base_rot=0.562 base_translate=0.812 structured: redirect=0.688 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA404 **PASS** — base_rot=0.500 base_translate=0.594 structured: redirect=0.625 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA405 **PASS** — base_rot=0.750 base_translate=0.375 structured: redirect=0.375 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA406 **PASS** — base_rot=0.125 base_translate=0.344 structured: redirect=0.281 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA407 **PASS** — base_rot=0.438 base_translate=0.594 structured: redirect=0.594 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA408 **PASS** — base_rot=0.656 base_translate=0.375 structured: redirect=0.406 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA409 **PASS** — base_rot=0.438 base_translate=0.594 structured: redirect=0.688 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA40A **PASS** — base_rot=0.062 base_translate=0.719 structured: redirect=0.719 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA40B **PASS** — base_rot=0.562 base_translate=0.500 structured: redirect=0.719 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA40C **PASS** — base_rot=0.312 base_translate=0.719 structured: redirect=0.594 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA40D **PASS** — base_rot=0.531 base_translate=0.469 structured: redirect=0.531 rot_acc=0.031 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA40E **PASS** — base_rot=0.219 base_translate=0.344 structured: redirect=0.406 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xA40F **PASS** — base_rot=0.688 base_translate=0.469 structured: redirect=0.375 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true

### E42

- 0xA400 **FAIL** — N8:A=0.363,B=0.227 N16:A=0.328,B=0.480 N32:A=0.309,B=0.336 N64:A=0.273,B=0.441 N128:A=0.289,B=0.414
- 0xA401 **FAIL** — N8:A=0.117,B=0.234 N16:A=0.266,B=0.301 N32:A=0.164,B=0.559 N64:A=0.379,B=0.473 N128:A=0.285,B=0.453
- 0xA402 **FAIL** — N8:A=0.176,B=0.340 N16:A=0.230,B=0.207 N32:A=0.227,B=0.414 N64:A=0.371,B=0.172 N128:A=0.215,B=0.441
- 0xA403 **FAIL** — N8:A=0.125,B=0.332 N16:A=0.238,B=0.309 N32:A=0.234,B=0.391 N64:A=0.238,B=0.426 N128:A=0.250,B=0.168
- 0xA404 **FAIL** — N8:A=0.258,B=0.281 N16:A=0.176,B=0.324 N32:A=0.270,B=0.484 N64:A=0.340,B=0.535 N128:A=0.488,B=0.605
- 0xA405 **PASS** — N8:A=0.129,B=0.438 N16:A=0.297,B=0.426 N32:A=0.305,B=0.426 N64:A=0.387,B=0.457 N128:A=0.309,B=0.383
- 0xA406 **FAIL** — N8:A=0.211,B=0.141 N16:A=0.367,B=0.445 N32:A=0.375,B=0.414 N64:A=0.273,B=0.602 N128:A=0.363,B=0.242
- 0xA407 **FAIL** — N8:A=0.254,B=0.422 N16:A=0.312,B=0.285 N32:A=0.098,B=0.488 N64:A=0.145,B=0.508 N128:A=0.359,B=0.500
- 0xA408 **FAIL** — N8:A=0.191,B=0.258 N16:A=0.352,B=0.352 N32:A=0.156,B=0.570 N64:A=0.344,B=0.469 N128:A=0.266,B=0.543
- 0xA409 **FAIL** — N8:A=0.199,B=0.344 N16:A=0.312,B=0.332 N32:A=0.227,B=0.492 N64:A=0.246,B=0.566 N128:A=0.238,B=0.523
- 0xA40A **PASS** — N8:A=0.109,B=0.348 N16:A=0.242,B=0.422 N32:A=0.227,B=0.504 N64:A=0.191,B=0.414 N128:A=0.203,B=0.359
- 0xA40B **FAIL** — N8:A=0.211,B=0.203 N16:A=0.289,B=0.348 N32:A=0.336,B=0.340 N64:A=0.242,B=0.449 N128:A=0.168,B=0.469
- 0xA40C **FAIL** — N8:A=0.160,B=0.223 N16:A=0.305,B=0.344 N32:A=0.238,B=0.316 N64:A=0.270,B=0.402 N128:A=0.273,B=0.441
- 0xA40D **PASS** — N8:A=0.102,B=0.238 N16:A=0.246,B=0.414 N32:A=0.289,B=0.352 N64:A=0.164,B=0.395 N128:A=0.340,B=0.598
- 0xA40E **PASS** — N8:A=0.188,B=0.348 N16:A=0.277,B=0.418 N32:A=0.316,B=0.484 N64:A=0.352,B=0.453 N128:A=0.246,B=0.312
- 0xA40F **FAIL** — N8:A=0.238,B=0.277 N16:A=0.242,B=0.336 N32:A=0.168,B=0.340 N64:A=0.234,B=0.449 N128:A=0.328,B=0.422

