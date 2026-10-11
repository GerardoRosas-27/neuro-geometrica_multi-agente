# v5 smoke summary

- git SHA: `37e256b695865f177bf091e4d77b86698268d0d2` (dirty=false)
- steps: 200
- seeds: 1

| Gate | PASS | FAIL | n |
|---|---|---|---|
| R_P2_gated | 0 | 1 | 1 |
| R_P2_mlp | 0 | 1 | 1 |
| R_P2_shuffled | 0 | 1 | 1 |

## Per-seed gate notes

### R_P2_gated

- 0x5A00 **FAIL** — R_P2_gated region=0 rel_err1=0.4855 clean=true fresh_ok=true fresh_acc=0.0078(q=0) leaks=0 h1:acc=0.000(rot=0.00,refl=0.00),rel=0.889,norm=0.832,pert=0.859 h2:acc=0.000(rot=0.00,refl=0.00),rel=0.410,norm=0.713,pert=0.761 h4:acc=0.000(rot=0.00,refl=0.00),rel=0.600,norm=0.583,pert=0.614 h8:acc=0.000(rot=0.00,refl=0.00),rel=0.743,norm=0.527,pert=0.417 h16:acc=0.000(rot=0.00,refl=0.00),rel=0.857,norm=0.430,pert=0.204 h32:acc=0.000(rot=0.00,refl=0.00),rel=1.134,norm=0.353,pert=0.075 h64:acc=0.000(rot=0.00,refl=0.00),rel=1.364,norm=0.312,pert=0.017

### R_P2_mlp

- 0x5A00 **FAIL** — R_P2_mlp region=0 rel_err1=0.4408 clean=true fresh_ok=true fresh_acc=0.0078(q=0) leaks=0 h1:acc=0.000(rot=0.00,refl=0.00),rel=0.910,norm=0.795,pert=0.738 h2:acc=0.000(rot=0.00,refl=0.00),rel=0.394,norm=0.669,pert=0.584 h4:acc=0.000(rot=0.00,refl=0.00),rel=0.561,norm=0.526,pert=0.416 h8:acc=0.000(rot=0.00,refl=0.00),rel=0.685,norm=0.416,pert=0.279 h16:acc=0.000(rot=0.00,refl=0.00),rel=0.764,norm=0.374,pert=0.172 h32:acc=0.000(rot=0.00,refl=0.00),rel=0.881,norm=0.352,pert=0.109 h64:acc=0.000(rot=0.00,refl=0.00),rel=0.977,norm=0.383,pert=0.010

### R_P2_shuffled

- 0x5A00 **FAIL** — R_P2_shuffled region=0 rel_err1=0.9364 clean=true fresh_ok=true fresh_acc=0.0078(q=0) leaks=0 h1:acc=0.000(rot=0.00,refl=0.00),rel=0.897,norm=0.210,pert=0.202 h2:acc=0.000(rot=0.00,refl=0.00),rel=0.991,norm=0.091,pert=0.026 h4:acc=0.000(rot=0.00,refl=0.00),rel=1.003,norm=0.085,pert=0.001 h8:acc=0.000(rot=0.00,refl=0.00),rel=1.009,norm=0.085,pert=0.000 h16:acc=0.000(rot=0.00,refl=0.00),rel=1.003,norm=0.085,pert=0.000 h32:acc=0.000(rot=0.00,refl=0.00),rel=1.005,norm=0.085,pert=0.000 h64:acc=0.000(rot=0.00,refl=0.00),rel=0.999,norm=0.085,pert=0.000

