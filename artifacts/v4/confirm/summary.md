# v4 confirm summary

- git SHA: `456c77d3cf38daf0df81842a665f31f21e8ef4cd` (dirty=false)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 16
- manifests: `dataset_manifest.jsonl` (sha256 of file: `89c0207ff0e479c2aa0e37446cbbb20eb2ffcb5bf41dacb11f42555d52d8188e`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| E31 | 16 | 0 | 16 |
| E32 | 16 | 0 | 16 |
| E33 | 0 | 16 | 16 |
| E34 | 6 | 10 | 16 |
| E35 | 10 | 6 | 16 |
| E36 | 13 | 3 | 16 |
| E37 | 10 | 6 | 16 |
| E38 | 10 | 6 | 16 |
| E39 | 10 | 6 | 16 |
| E40 | 16 | 0 | 16 |
| E41 | 16 | 0 | 16 |
| E42 | 5 | 11 | 16 |

## Per-seed gate notes

### E31

- 0xB400 **PASS** — families_won=6 [translate rotate scale shear affine compose_rot_scale] dphi_acc=0.297 best_ctrl_acc=0.047
- 0xB401 **PASS** — families_won=7 [translate rotate scale shear affine compose_rot_scale radial_warp] dphi_acc=0.301 best_ctrl_acc=0.027
- 0xB402 **PASS** — families_won=5 [translate rotate scale shear compose_rot_scale] dphi_acc=0.191 best_ctrl_acc=0.051
- 0xB403 **PASS** — families_won=7 [translate rotate reflect scale shear affine compose_rot_scale] dphi_acc=0.340 best_ctrl_acc=0.066
- 0xB404 **PASS** — families_won=6 [translate rotate scale shear affine compose_rot_scale] dphi_acc=0.238 best_ctrl_acc=0.031
- 0xB405 **PASS** — families_won=5 [translate scale affine compose_rot_scale radial_warp] dphi_acc=0.238 best_ctrl_acc=0.027
- 0xB406 **PASS** — families_won=6 [translate rotate scale shear affine compose_rot_scale] dphi_acc=0.270 best_ctrl_acc=0.055
- 0xB407 **PASS** — families_won=6 [translate scale shear affine compose_rot_scale radial_warp] dphi_acc=0.305 best_ctrl_acc=0.020
- 0xB408 **PASS** — families_won=7 [translate rotate scale shear affine compose_rot_scale radial_warp] dphi_acc=0.219 best_ctrl_acc=0.043
- 0xB409 **PASS** — families_won=7 [translate rotate scale shear affine compose_rot_scale radial_warp] dphi_acc=0.238 best_ctrl_acc=0.043
- 0xB40A **PASS** — families_won=5 [translate scale shear affine radial_warp] dphi_acc=0.195 best_ctrl_acc=0.031
- 0xB40B **PASS** — families_won=4 [translate scale shear compose_rot_scale] dphi_acc=0.156 best_ctrl_acc=0.059
- 0xB40C **PASS** — families_won=7 [translate rotate scale shear affine compose_rot_scale radial_warp] dphi_acc=0.363 best_ctrl_acc=0.043
- 0xB40D **PASS** — families_won=6 [translate reflect scale shear affine compose_rot_scale] dphi_acc=0.199 best_ctrl_acc=0.043
- 0xB40E **PASS** — families_won=7 [translate rotate scale shear affine compose_rot_scale radial_warp] dphi_acc=0.223 best_ctrl_acc=0.020
- 0xB40F **PASS** — families_won=4 [translate scale affine compose_rot_scale] dphi_acc=0.195 best_ctrl_acc=0.070

### E32

- 0xB400 **PASS** — fams_sigma_within10%=7 frac_steps_sigma>1=0.970 translate:σD=1.068/σT=1.000 rotate:σD=1.051/σT=1.000 reflect:σD=1.140/σT=1.000 scale:σD=1.210/σT=1.155 shear:σD=1.225/σT=1.155 affine:σD=1.151/σT=1.072 compose_rot_scale:σD=1.132/σT=1.053 radial_warp:σD=1.199/σT=1.203
- 0xB401 **PASS** — fams_sigma_within10%=5 frac_steps_sigma>1=0.975 translate:σD=1.076/σT=1.000 rotate:σD=1.085/σT=1.000 reflect:σD=1.121/σT=1.000 scale:σD=1.153/σT=1.095 shear:σD=1.194/σT=1.143 affine:σD=1.213/σT=1.076 compose_rot_scale:σD=1.044/σT=0.916 radial_warp:σD=1.164/σT=1.185
- 0xB402 **PASS** — fams_sigma_within10%=7 frac_steps_sigma>1=0.944 translate:σD=1.081/σT=1.000 rotate:σD=1.088/σT=1.000 reflect:σD=1.115/σT=1.000 scale:σD=1.196/σT=1.123 shear:σD=1.145/σT=1.074 affine:σD=1.151/σT=1.079 compose_rot_scale:σD=1.080/σT=1.037 radial_warp:σD=1.140/σT=1.203
- 0xB403 **PASS** — fams_sigma_within10%=7 frac_steps_sigma>1=1.000 translate:σD=1.110/σT=1.000 rotate:σD=1.091/σT=1.000 reflect:σD=1.096/σT=1.000 scale:σD=1.202/σT=1.100 shear:σD=1.196/σT=1.120 affine:σD=1.171/σT=1.080 compose_rot_scale:σD=1.148/σT=1.086 radial_warp:σD=1.163/σT=1.260
- 0xB404 **PASS** — fams_sigma_within10%=5 frac_steps_sigma>1=0.969 translate:σD=1.102/σT=1.000 rotate:σD=1.109/σT=1.000 reflect:σD=1.122/σT=1.000 scale:σD=1.199/σT=1.130 shear:σD=1.226/σT=1.156 affine:σD=1.195/σT=1.093 compose_rot_scale:σD=1.052/σT=0.966 radial_warp:σD=1.164/σT=1.194
- 0xB405 **PASS** — fams_sigma_within10%=6 frac_steps_sigma>1=0.976 translate:σD=1.075/σT=1.000 rotate:σD=1.083/σT=1.000 reflect:σD=1.108/σT=1.000 scale:σD=1.164/σT=1.145 shear:σD=1.185/σT=1.172 affine:σD=1.159/σT=1.090 compose_rot_scale:σD=1.142/σT=1.030 radial_warp:σD=1.110/σT=1.213
- 0xB406 **PASS** — fams_sigma_within10%=7 frac_steps_sigma>1=1.000 translate:σD=1.097/σT=1.000 rotate:σD=1.088/σT=1.000 reflect:σD=1.126/σT=1.000 scale:σD=1.169/σT=1.115 shear:σD=1.201/σT=1.122 affine:σD=1.174/σT=1.090 compose_rot_scale:σD=1.129/σT=1.023 radial_warp:σD=1.203/σT=1.200
- 0xB407 **PASS** — fams_sigma_within10%=4 frac_steps_sigma>1=0.969 translate:σD=1.104/σT=1.000 rotate:σD=1.078/σT=1.000 reflect:σD=1.118/σT=1.000 scale:σD=1.185/σT=1.111 shear:σD=1.197/σT=1.131 affine:σD=1.193/σT=1.073 compose_rot_scale:σD=1.116/σT=0.967 radial_warp:σD=1.187/σT=1.235
- 0xB408 **PASS** — fams_sigma_within10%=6 frac_steps_sigma>1=0.986 translate:σD=1.087/σT=1.000 rotate:σD=1.066/σT=1.000 reflect:σD=1.206/σT=1.000 scale:σD=1.122/σT=1.097 shear:σD=1.162/σT=1.114 affine:σD=1.173/σT=1.098 compose_rot_scale:σD=1.099/σT=0.956 radial_warp:σD=1.143/σT=1.222
- 0xB409 **PASS** — fams_sigma_within10%=5 frac_steps_sigma>1=0.976 translate:σD=1.102/σT=1.000 rotate:σD=1.070/σT=1.000 reflect:σD=1.132/σT=1.000 scale:σD=1.191/σT=1.100 shear:σD=1.158/σT=1.093 affine:σD=1.191/σT=1.072 compose_rot_scale:σD=1.202/σT=1.101 radial_warp:σD=1.180/σT=1.221
- 0xB40A **PASS** — fams_sigma_within10%=5 frac_steps_sigma>1=0.972 translate:σD=1.119/σT=1.000 rotate:σD=1.136/σT=1.000 reflect:σD=1.086/σT=1.000 scale:σD=1.199/σT=1.109 shear:σD=1.175/σT=1.118 affine:σD=1.226/σT=1.094 compose_rot_scale:σD=1.144/σT=1.061 radial_warp:σD=1.169/σT=1.228
- 0xB40B **PASS** — fams_sigma_within10%=7 frac_steps_sigma>1=0.962 translate:σD=1.064/σT=1.000 rotate:σD=1.050/σT=1.000 reflect:σD=1.107/σT=1.000 scale:σD=1.187/σT=1.115 shear:σD=1.157/σT=1.104 affine:σD=1.117/σT=1.069 compose_rot_scale:σD=0.966/σT=0.900 radial_warp:σD=1.131/σT=1.254
- 0xB40C **PASS** — fams_sigma_within10%=7 frac_steps_sigma>1=0.981 translate:σD=1.081/σT=1.000 rotate:σD=1.085/σT=1.000 reflect:σD=1.122/σT=1.000 scale:σD=1.216/σT=1.150 shear:σD=1.184/σT=1.108 affine:σD=1.185/σT=1.085 compose_rot_scale:σD=1.039/σT=0.932 radial_warp:σD=1.166/σT=1.212
- 0xB40D **PASS** — fams_sigma_within10%=6 frac_steps_sigma>1=0.982 translate:σD=1.094/σT=1.000 rotate:σD=1.135/σT=1.000 reflect:σD=1.161/σT=1.000 scale:σD=1.232/σT=1.214 shear:σD=1.241/σT=1.139 affine:σD=1.201/σT=1.082 compose_rot_scale:σD=1.050/σT=1.005 radial_warp:σD=1.179/σT=1.195
- 0xB40E **PASS** — fams_sigma_within10%=6 frac_steps_sigma>1=0.934 translate:σD=1.090/σT=1.000 rotate:σD=1.112/σT=1.000 reflect:σD=1.144/σT=1.000 scale:σD=1.194/σT=1.117 shear:σD=1.187/σT=1.109 affine:σD=1.171/σT=1.055 compose_rot_scale:σD=1.141/σT=1.038 radial_warp:σD=1.187/σT=1.243
- 0xB40F **PASS** — fams_sigma_within10%=6 frac_steps_sigma>1=0.999 translate:σD=1.119/σT=1.000 rotate:σD=1.075/σT=1.000 reflect:σD=1.102/σT=1.000 scale:σD=1.182/σT=1.103 shear:σD=1.153/σT=1.100 affine:σD=1.132/σT=1.051 compose_rot_scale:σD=1.151/σT=1.052 radial_warp:σD=1.129/σT=1.205

### E33

- 0xB400 **FAIL** — stability_region_h=0 iterable/all h1:0.09/0.30 h2:0.00/0.04 h4:0.00/0.00 h8:0.00/0.01 h16:0.00/0.01 h32:0.00/0.01 h64:0.00/0.01
- 0xB401 **FAIL** — stability_region_h=0 iterable/all h1:0.19/0.30 h2:0.02/0.05 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB402 **FAIL** — stability_region_h=0 iterable/all h1:0.20/0.19 h2:0.03/0.02 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB403 **FAIL** — stability_region_h=0 iterable/all h1:0.33/0.34 h2:0.05/0.03 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB404 **FAIL** — stability_region_h=0 iterable/all h1:0.09/0.24 h2:0.00/0.02 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB405 **FAIL** — stability_region_h=0 iterable/all h1:0.03/0.24 h2:0.00/0.02 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.01
- 0xB406 **FAIL** — stability_region_h=0 iterable/all h1:0.12/0.27 h2:0.00/0.02 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB407 **FAIL** — stability_region_h=0 iterable/all h1:0.06/0.30 h2:0.00/0.03 h4:0.00/0.00 h8:0.00/0.01 h16:0.00/0.01 h32:0.00/0.00 h64:0.00/0.00
- 0xB408 **FAIL** — stability_region_h=0 iterable/all h1:0.17/0.22 h2:0.00/0.01 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB409 **FAIL** — stability_region_h=0 iterable/all h1:0.20/0.24 h2:0.03/0.02 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB40A **FAIL** — stability_region_h=0 iterable/all h1:0.05/0.20 h2:0.00/0.02 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB40B **FAIL** — stability_region_h=0 iterable/all h1:0.06/0.16 h2:0.00/0.00 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB40C **FAIL** — stability_region_h=0 iterable/all h1:0.19/0.36 h2:0.02/0.03 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB40D **FAIL** — stability_region_h=0 iterable/all h1:0.09/0.20 h2:0.02/0.02 h4:0.02/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00
- 0xB40E **FAIL** — stability_region_h=0 iterable/all h1:0.09/0.22 h2:0.00/0.01 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.01
- 0xB40F **FAIL** — stability_region_h=0 iterable/all h1:0.03/0.20 h2:0.00/0.01 h4:0.00/0.00 h8:0.00/0.00 h16:0.00/0.00 h32:0.00/0.00 h64:0.00/0.00

### E34

- 0xB400 **PASS** — dphi_h1=0.297 mlp_h1=0.184 dphi_h4=0.000 mlp_h4=0.000 static=0.047 params=962/962
- 0xB401 **PASS** — dphi_h1=0.301 mlp_h1=0.246 dphi_h4=0.000 mlp_h4=0.000 static=0.027 params=962/962
- 0xB402 **FAIL** — dphi_h1=0.191 mlp_h1=0.312 dphi_h4=0.000 mlp_h4=0.000 static=0.051 params=962/962
- 0xB403 **PASS** — dphi_h1=0.340 mlp_h1=0.195 dphi_h4=0.000 mlp_h4=0.000 static=0.066 params=962/962
- 0xB404 **FAIL** — dphi_h1=0.238 mlp_h1=0.266 dphi_h4=0.000 mlp_h4=0.000 static=0.031 params=962/962
- 0xB405 **FAIL** — dphi_h1=0.238 mlp_h1=0.191 dphi_h4=0.000 mlp_h4=0.000 static=0.023 params=962/962
- 0xB406 **PASS** — dphi_h1=0.270 mlp_h1=0.199 dphi_h4=0.000 mlp_h4=0.000 static=0.055 params=962/962
- 0xB407 **PASS** — dphi_h1=0.305 mlp_h1=0.234 dphi_h4=0.000 mlp_h4=0.000 static=0.020 params=962/962
- 0xB408 **FAIL** — dphi_h1=0.219 mlp_h1=0.234 dphi_h4=0.000 mlp_h4=0.000 static=0.043 params=962/962
- 0xB409 **FAIL** — dphi_h1=0.238 mlp_h1=0.203 dphi_h4=0.000 mlp_h4=0.000 static=0.043 params=962/962
- 0xB40A **FAIL** — dphi_h1=0.195 mlp_h1=0.156 dphi_h4=0.000 mlp_h4=0.000 static=0.031 params=962/962
- 0xB40B **FAIL** — dphi_h1=0.156 mlp_h1=0.160 dphi_h4=0.000 mlp_h4=0.000 static=0.059 params=962/962
- 0xB40C **PASS** — dphi_h1=0.363 mlp_h1=0.285 dphi_h4=0.000 mlp_h4=0.000 static=0.043 params=962/962
- 0xB40D **FAIL** — dphi_h1=0.199 mlp_h1=0.262 dphi_h4=0.016 mlp_h4=0.000 static=0.043 params=962/962
- 0xB40E **FAIL** — dphi_h1=0.223 mlp_h1=0.367 dphi_h4=0.000 mlp_h4=0.000 static=0.020 params=962/962
- 0xB40F **FAIL** — dphi_h1=0.195 mlp_h1=0.211 dphi_h4=0.000 mlp_h4=0.000 static=0.070 params=962/962

### E35

- 0xB400 **PASS** — A=0.2969 B=0.4727 diff=0.1758 relerrA=0.0747 relerrB=0.0631 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB401 **PASS** — A=0.3008 B=0.4492 diff=0.1484 relerrA=0.0779 relerrB=0.0670 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB402 **PASS** — A=0.1914 B=0.3867 diff=0.1953 relerrA=0.0857 relerrB=0.0691 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB403 **FAIL** — A=0.3398 B=0.2617 diff=-0.0781 relerrA=0.0690 relerrB=0.0728 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB404 **FAIL** — A=0.2383 B=0.2695 diff=0.0312 relerrA=0.0799 relerrB=0.0755 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB405 **PASS** — A=0.2383 B=0.4727 diff=0.2344 relerrA=0.0804 relerrB=0.0622 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB406 **FAIL** — A=0.2695 B=0.2734 diff=0.0039 relerrA=0.0696 relerrB=0.0709 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB407 **FAIL** — A=0.3047 B=0.3125 diff=0.0078 relerrA=0.0686 relerrB=0.0695 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB408 **PASS** — A=0.2188 B=0.4492 diff=0.2305 relerrA=0.0856 relerrB=0.0680 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB409 **PASS** — A=0.2383 B=0.5117 diff=0.2734 relerrA=0.0782 relerrB=0.0610 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40A **PASS** — A=0.1953 B=0.4492 diff=0.2539 relerrA=0.0884 relerrB=0.0633 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40B **PASS** — A=0.1562 B=0.3047 diff=0.1484 relerrA=0.0895 relerrB=0.0776 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40C **FAIL** — A=0.3633 B=0.3984 diff=0.0352 relerrA=0.0668 relerrB=0.0656 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40D **PASS** — A=0.1992 B=0.4414 diff=0.2422 relerrA=0.0793 relerrB=0.0616 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40E **FAIL** — A=0.2227 B=0.1758 diff=-0.0469 relerrA=0.0783 relerrB=0.0869 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false
- 0xB40F **PASS** — A=0.1953 B=0.4102 diff=0.2148 relerrA=0.0779 relerrB=0.0694 cdt_q=0 rqm_q=0 table_q=0 nn_q=0 attr_q=0 direct_q=0 leaked=0 target_seen_cdt=false target_equivalent_seen=false

### E36

- 0xB400 **PASS** — C=0.473 B_raw=0.273 A=0.297 D_corrupt_cdt=0.008 E_irrelevant_cdt=0.000 F_stats_no_topology=0.047 G_shuffled_order=0.000
- 0xB401 **PASS** — C=0.449 B_raw=0.219 A=0.301 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.066 G_shuffled_order=0.000
- 0xB402 **PASS** — C=0.387 B_raw=0.223 A=0.191 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.004 F_stats_no_topology=0.027 G_shuffled_order=0.000
- 0xB403 **PASS** — C=0.262 B_raw=0.211 A=0.340 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.051 G_shuffled_order=0.000
- 0xB404 **FAIL** — C=0.270 B_raw=0.266 A=0.238 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.004 F_stats_no_topology=0.059 G_shuffled_order=0.000
- 0xB405 **PASS** — C=0.473 B_raw=0.355 A=0.238 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.102 G_shuffled_order=0.000
- 0xB406 **PASS** — C=0.273 B_raw=0.195 A=0.270 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.066 G_shuffled_order=0.000
- 0xB407 **PASS** — C=0.312 B_raw=0.223 A=0.305 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.082 G_shuffled_order=0.000
- 0xB408 **PASS** — C=0.449 B_raw=0.223 A=0.219 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.027 G_shuffled_order=0.000
- 0xB409 **PASS** — C=0.512 B_raw=0.352 A=0.238 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.102 G_shuffled_order=0.000
- 0xB40A **PASS** — C=0.449 B_raw=0.133 A=0.195 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.070 G_shuffled_order=0.000
- 0xB40B **PASS** — C=0.305 B_raw=0.164 A=0.156 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.027 G_shuffled_order=0.000
- 0xB40C **FAIL** — C=0.398 B_raw=0.395 A=0.363 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.016 G_shuffled_order=0.000
- 0xB40D **PASS** — C=0.441 B_raw=0.375 A=0.199 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.059 G_shuffled_order=0.000
- 0xB40E **FAIL** — C=0.176 B_raw=0.309 A=0.223 D_corrupt_cdt=0.004 E_irrelevant_cdt=0.000 F_stats_no_topology=0.047 G_shuffled_order=0.000
- 0xB40F **PASS** — C=0.410 B_raw=0.309 A=0.195 D_corrupt_cdt=0.000 E_irrelevant_cdt=0.000 F_stats_no_topology=0.074 G_shuffled_order=0.000

### E37

- 0xB400 **PASS** — pre_delete=0.4727 post_delete_fresh_process=0.4727 A=0.2969 fresh_process_queries=0 ckpt_sha=5d8918f5038927a86cc3138ab8e2209bcc8df11f897a6478b891e545820a757b
- 0xB401 **PASS** — pre_delete=0.4492 post_delete_fresh_process=0.4492 A=0.3008 fresh_process_queries=0 ckpt_sha=170b6625cb1909e97af185ca87495760fdf3edbc665e0846434d3c529566af35
- 0xB402 **PASS** — pre_delete=0.3867 post_delete_fresh_process=0.3867 A=0.1914 fresh_process_queries=0 ckpt_sha=3f7d3830de103ec4c23f2fd1d9bb3d29383518591bac980d920406e92bd0888d
- 0xB403 **FAIL** — pre_delete=0.2617 post_delete_fresh_process=0.2617 A=0.3398 fresh_process_queries=0 ckpt_sha=7caaba3cd67ea120e8b1b14f75f9298a3c9c5579f161f1e7e78adbce6da1f2c2
- 0xB404 **FAIL** — pre_delete=0.2695 post_delete_fresh_process=0.2695 A=0.2383 fresh_process_queries=0 ckpt_sha=944bb23c7f61d7540918087fe11d8549cab20f1ce6f44953e919cdb63f6a52ee
- 0xB405 **PASS** — pre_delete=0.4727 post_delete_fresh_process=0.4727 A=0.2383 fresh_process_queries=0 ckpt_sha=17b9017a45393dd0b5ad4c8609c3b6fe7258efe42e481c75b90a9b881caf6785
- 0xB406 **FAIL** — pre_delete=0.2734 post_delete_fresh_process=0.2734 A=0.2695 fresh_process_queries=0 ckpt_sha=36017778a4d9293388f38c6c87ccaacb4cb02fcd7bd82fbc94dff9aca79b12a6
- 0xB407 **FAIL** — pre_delete=0.3125 post_delete_fresh_process=0.3125 A=0.3047 fresh_process_queries=0 ckpt_sha=7114115a256a7e3a55ff75a50f3a94d82a35508062cd31362f95cc7a2f1ed054
- 0xB408 **PASS** — pre_delete=0.4492 post_delete_fresh_process=0.4492 A=0.2188 fresh_process_queries=0 ckpt_sha=390ecab4b7947c3fb8dd6b7e78d3e850798352ca78568d9b6d19139db4a1f1b0
- 0xB409 **PASS** — pre_delete=0.5117 post_delete_fresh_process=0.5117 A=0.2383 fresh_process_queries=0 ckpt_sha=5215333dbe3700441ffc5000367a8ca7c8895fa2e9f2422c31fa48c183ac4d74
- 0xB40A **PASS** — pre_delete=0.4492 post_delete_fresh_process=0.4492 A=0.1953 fresh_process_queries=0 ckpt_sha=ae870b623b3de22a82de00c90e7d2661b550a02a81f3e570679b72ea6227e0da
- 0xB40B **PASS** — pre_delete=0.3047 post_delete_fresh_process=0.3047 A=0.1562 fresh_process_queries=0 ckpt_sha=4b711f931a73797d797577d7889b6725f3c72db106dcec1b7207127d94387711
- 0xB40C **FAIL** — pre_delete=0.3984 post_delete_fresh_process=0.3984 A=0.3633 fresh_process_queries=0 ckpt_sha=dab0292a207f2417d39ea3ad72607125d7baeb6e48086252a187ac4904deccb0
- 0xB40D **PASS** — pre_delete=0.4414 post_delete_fresh_process=0.4414 A=0.1992 fresh_process_queries=0 ckpt_sha=429f0985f9507172295eb04ac1c731186cd685e5d8ce78b9461339319054f95c
- 0xB40E **FAIL** — pre_delete=0.1758 post_delete_fresh_process=0.1758 A=0.2227 fresh_process_queries=0 ckpt_sha=3e9341038b25976e267c34f78bbe04f535a26558b9a4404369bb4476a8ab64e1
- 0xB40F **PASS** — pre_delete=0.4102 post_delete_fresh_process=0.4102 A=0.1953 fresh_process_queries=0 ckpt_sha=6be351a07ab96558ebfc1dd88b8a5fb1e64d0e1f2061b75477f1c72981c69f2e

### E38

- 0xB400 **PASS** — P0=0.4727 P0_newtest=0.4766 A_newtest=0.3359 P1=0.4727(q=0) P2=P0(identity encoder) P3_lookup=0.0117(cdt_q=256) A=0.2969
- 0xB401 **PASS** — P0=0.4492 P0_newtest=0.4414 A_newtest=0.2383 P1=0.4492(q=0) P2=P0(identity encoder) P3_lookup=0.0117(cdt_q=256) A=0.3008
- 0xB402 **PASS** — P0=0.3867 P0_newtest=0.3750 A_newtest=0.1797 P1=0.3867(q=0) P2=P0(identity encoder) P3_lookup=0.0117(cdt_q=256) A=0.1914
- 0xB403 **FAIL** — P0=0.2617 P0_newtest=0.2852 A_newtest=0.3516 P1=0.2617(q=0) P2=P0(identity encoder) P3_lookup=0.0312(cdt_q=256) A=0.3398
- 0xB404 **FAIL** — P0=0.2695 P0_newtest=0.2656 A_newtest=0.2500 P1=0.2695(q=0) P2=P0(identity encoder) P3_lookup=0.0078(cdt_q=256) A=0.2383
- 0xB405 **PASS** — P0=0.4727 P0_newtest=0.4688 A_newtest=0.2031 P1=0.4727(q=0) P2=P0(identity encoder) P3_lookup=0.0273(cdt_q=256) A=0.2383
- 0xB406 **FAIL** — P0=0.2734 P0_newtest=0.2656 A_newtest=0.2891 P1=0.2734(q=0) P2=P0(identity encoder) P3_lookup=0.0117(cdt_q=256) A=0.2695
- 0xB407 **FAIL** — P0=0.3125 P0_newtest=0.3203 A_newtest=0.4141 P1=0.3125(q=0) P2=P0(identity encoder) P3_lookup=0.0000(cdt_q=256) A=0.3047
- 0xB408 **PASS** — P0=0.4492 P0_newtest=0.4414 A_newtest=0.2266 P1=0.4492(q=0) P2=P0(identity encoder) P3_lookup=0.0156(cdt_q=256) A=0.2188
- 0xB409 **PASS** — P0=0.5117 P0_newtest=0.4375 A_newtest=0.1914 P1=0.5117(q=0) P2=P0(identity encoder) P3_lookup=0.0117(cdt_q=256) A=0.2383
- 0xB40A **PASS** — P0=0.4492 P0_newtest=0.5156 A_newtest=0.2031 P1=0.4492(q=0) P2=P0(identity encoder) P3_lookup=0.0039(cdt_q=256) A=0.1953
- 0xB40B **PASS** — P0=0.3047 P0_newtest=0.3125 A_newtest=0.1055 P1=0.3047(q=0) P2=P0(identity encoder) P3_lookup=0.0117(cdt_q=256) A=0.1562
- 0xB40C **FAIL** — P0=0.3984 P0_newtest=0.3633 A_newtest=0.3203 P1=0.3984(q=0) P2=P0(identity encoder) P3_lookup=0.0352(cdt_q=256) A=0.3633
- 0xB40D **PASS** — P0=0.4414 P0_newtest=0.4492 A_newtest=0.2773 P1=0.4414(q=0) P2=P0(identity encoder) P3_lookup=0.0195(cdt_q=256) A=0.1992
- 0xB40E **FAIL** — P0=0.1758 P0_newtest=0.1797 A_newtest=0.2461 P1=0.1758(q=0) P2=P0(identity encoder) P3_lookup=0.0195(cdt_q=256) A=0.2227
- 0xB40F **PASS** — P0=0.4102 P0_newtest=0.3711 A_newtest=0.2500 P1=0.4102(q=0) P2=P0(identity encoder) P3_lookup=0.0156(cdt_q=256) A=0.1953

### E39

- 0xB400 **PASS** — A=0.188 B=0.270 static=0.047
- 0xB401 **PASS** — A=0.195 B=0.375 static=0.031
- 0xB402 **FAIL** — A=0.242 B=0.285 static=0.039
- 0xB403 **PASS** — A=0.219 B=0.359 static=0.039
- 0xB404 **PASS** — A=0.184 B=0.328 static=0.062
- 0xB405 **FAIL** — A=0.305 B=0.227 static=0.039
- 0xB406 **PASS** — A=0.340 B=0.414 static=0.055
- 0xB407 **PASS** — A=0.250 B=0.336 static=0.039
- 0xB408 **FAIL** — A=0.273 B=0.266 static=0.035
- 0xB409 **PASS** — A=0.277 B=0.383 static=0.070
- 0xB40A **FAIL** — A=0.188 B=0.199 static=0.031
- 0xB40B **PASS** — A=0.184 B=0.383 static=0.043
- 0xB40C **PASS** — A=0.277 B=0.398 static=0.039
- 0xB40D **FAIL** — A=0.285 B=0.262 static=0.035
- 0xB40E **FAIL** — A=0.391 B=0.316 static=0.031
- 0xB40F **PASS** — A=0.273 B=0.387 static=0.043

### E40

- 0xB400 **PASS** — final A=0.188 B=0.688 D=0.047 forgetting A=0.823 B=0.250 D=0.302
- 0xB401 **PASS** — final A=0.109 B=0.656 D=0.055 forgetting A=0.948 B=0.260 D=0.354
- 0xB402 **PASS** — final A=0.195 B=0.336 D=0.094 forgetting A=0.823 B=0.625 D=0.271
- 0xB403 **PASS** — final A=0.156 B=0.742 D=0.070 forgetting A=0.917 B=0.146 D=0.333
- 0xB404 **PASS** — final A=0.164 B=0.531 D=0.070 forgetting A=0.938 B=0.427 D=0.302
- 0xB405 **PASS** — final A=0.141 B=0.719 D=0.055 forgetting A=0.948 B=0.188 D=0.354
- 0xB406 **PASS** — final A=0.180 B=0.703 D=0.094 forgetting A=0.885 B=0.198 D=0.292
- 0xB407 **PASS** — final A=0.156 B=0.570 D=0.078 forgetting A=0.958 B=0.260 D=0.354
- 0xB408 **PASS** — final A=0.188 B=0.695 D=0.055 forgetting A=0.885 B=0.219 D=0.344
- 0xB409 **PASS** — final A=0.195 B=0.711 D=0.039 forgetting A=0.917 B=0.260 D=0.344
- 0xB40A **PASS** — final A=0.125 B=0.727 D=0.062 forgetting A=0.906 B=0.125 D=0.333
- 0xB40B **PASS** — final A=0.117 B=0.539 D=0.086 forgetting A=0.938 B=0.458 D=0.312
- 0xB40C **PASS** — final A=0.203 B=0.836 D=0.102 forgetting A=0.906 B=0.042 D=0.312
- 0xB40D **PASS** — final A=0.180 B=0.680 D=0.094 forgetting A=0.781 B=0.052 D=0.177
- 0xB40E **PASS** — final A=0.164 B=0.594 D=0.086 forgetting A=0.917 B=0.250 D=0.396
- 0xB40F **PASS** — final A=0.156 B=0.664 D=0.102 forgetting A=0.948 B=0.198 D=0.333

### E41

- 0xB400 **PASS** — base_rot=0.406 base_translate=0.906 structured: redirect=0.812 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB401 **PASS** — base_rot=0.625 base_translate=0.906 structured: redirect=0.688 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB402 **PASS** — base_rot=0.219 base_translate=0.531 structured: redirect=0.719 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB403 **PASS** — base_rot=0.406 base_translate=0.438 structured: redirect=0.406 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB404 **PASS** — base_rot=0.125 base_translate=0.406 structured: redirect=0.406 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB405 **PASS** — base_rot=0.406 base_translate=0.781 structured: redirect=0.750 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB406 **PASS** — base_rot=0.281 base_translate=0.469 structured: redirect=0.719 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB407 **PASS** — base_rot=0.094 base_translate=0.438 structured: redirect=0.344 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB408 **PASS** — base_rot=0.688 base_translate=0.531 structured: redirect=0.500 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB409 **PASS** — base_rot=0.594 base_translate=0.625 structured: redirect=0.688 rot_acc=0.031 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB40A **PASS** — base_rot=0.625 base_translate=0.656 structured: redirect=0.750 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB40B **PASS** — base_rot=0.500 base_translate=0.562 structured: redirect=0.531 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB40C **PASS** — base_rot=0.500 base_translate=0.812 structured: redirect=0.750 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB40D **PASS** — base_rot=0.344 base_translate=0.656 structured: redirect=0.562 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB40E **PASS** — base_rot=0.000 base_translate=0.406 structured: redirect=0.375 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true
- 0xB40F **PASS** — base_rot=0.375 base_translate=0.438 structured: redirect=0.344 rot_acc=0.000 | random(|Δ|F equal): redirect=0.000 rot_acc=0.000 | rollback_exact=true

### E42

- 0xB400 **FAIL** — N8:A=0.266,B=0.398 N16:A=0.262,B=0.227 N32:A=0.215,B=0.348 N64:A=0.301,B=0.316 N128:A=0.238,B=0.422
- 0xB401 **PASS** — N8:A=0.242,B=0.398 N16:A=0.215,B=0.387 N32:A=0.188,B=0.488 N64:A=0.363,B=0.535 N128:A=0.195,B=0.402
- 0xB402 **FAIL** — N8:A=0.176,B=0.195 N16:A=0.312,B=0.461 N32:A=0.293,B=0.199 N64:A=0.312,B=0.348 N128:A=0.289,B=0.195
- 0xB403 **FAIL** — N8:A=0.273,B=0.227 N16:A=0.102,B=0.379 N32:A=0.258,B=0.340 N64:A=0.297,B=0.363 N128:A=0.219,B=0.250
- 0xB404 **FAIL** — N8:A=0.203,B=0.285 N16:A=0.375,B=0.383 N32:A=0.418,B=0.469 N64:A=0.195,B=0.297 N128:A=0.250,B=0.336
- 0xB405 **PASS** — N8:A=0.262,B=0.449 N16:A=0.195,B=0.469 N32:A=0.344,B=0.434 N64:A=0.293,B=0.520 N128:A=0.219,B=0.469
- 0xB406 **FAIL** — N8:A=0.234,B=0.375 N16:A=0.219,B=0.371 N32:A=0.258,B=0.355 N64:A=0.309,B=0.152 N128:A=0.176,B=0.535
- 0xB407 **FAIL** — N8:A=0.258,B=0.223 N16:A=0.297,B=0.414 N32:A=0.262,B=0.418 N64:A=0.195,B=0.328 N128:A=0.211,B=0.461
- 0xB408 **PASS** — N8:A=0.273,B=0.352 N16:A=0.277,B=0.367 N32:A=0.176,B=0.332 N64:A=0.230,B=0.328 N128:A=0.203,B=0.594
- 0xB409 **FAIL** — N8:A=0.145,B=0.270 N16:A=0.375,B=0.332 N32:A=0.266,B=0.230 N64:A=0.188,B=0.285 N128:A=0.309,B=0.387
- 0xB40A **PASS** — N8:A=0.254,B=0.379 N16:A=0.230,B=0.348 N32:A=0.266,B=0.504 N64:A=0.219,B=0.355 N128:A=0.184,B=0.316
- 0xB40B **FAIL** — N8:A=0.285,B=0.316 N16:A=0.188,B=0.496 N32:A=0.129,B=0.266 N64:A=0.281,B=0.410 N128:A=0.234,B=0.336
- 0xB40C **FAIL** — N8:A=0.262,B=0.453 N16:A=0.211,B=0.398 N32:A=0.203,B=0.469 N64:A=0.352,B=0.312 N128:A=0.203,B=0.289
- 0xB40D **FAIL** — N8:A=0.293,B=0.246 N16:A=0.262,B=0.309 N32:A=0.266,B=0.352 N64:A=0.203,B=0.574 N128:A=0.164,B=0.344
- 0xB40E **FAIL** — N8:A=0.172,B=0.328 N16:A=0.293,B=0.434 N32:A=0.230,B=0.383 N64:A=0.371,B=0.371 N128:A=0.219,B=0.352
- 0xB40F **PASS** — N8:A=0.191,B=0.305 N16:A=0.250,B=0.453 N32:A=0.266,B=0.500 N64:A=0.258,B=0.445 N128:A=0.270,B=0.332

