# v4 smoke summary

- git SHA: `db27d89f78d9e7c8dfd125d4d9b128a5f11cf5bb` (dirty=false)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 1
- manifests: `dataset_manifest.jsonl` (sha256 of file: `efd9eb4384f1e47cc9aea5d539b6d65559534f5e6f1bbadd6690cb7c4328c745`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| DIST_D | 0 | 5 | 5 |

## Per-seed gate notes

### DIST_D

- 0x5A00 **FAIL** — V0_pure_C1 region=1 sigma_med=1.0237 rel_err1_all=0.0458/0.0467/0.0781 clean=true h1:acc=0.531,rel=0.087,norm=0.991,energy=0.983,manif=0.494,pert_growth=0.988 h2:acc=0.328,rel=0.110,norm=0.980,energy=0.961,manif=0.042,pert_growth=0.977 h4:acc=0.047,rel=0.191,norm=0.973,energy=0.946,manif=0.074,pert_growth=0.953 h8:acc=0.000,rel=0.314,norm=0.944,energy=0.892,manif=0.133,pert_growth=0.914 h16:acc=0.000,rel=0.484,norm=0.879,energy=0.773,manif=0.203,pert_growth=0.866 h32:acc=0.000,rel=0.730,norm=0.849,energy=0.720,manif=0.335,pert_growth=0.823 h64:acc=0.000,rel=1.074,norm=0.820,energy=0.672,manif=0.472,pert_growth=0.615
- 0x5A00 **FAIL** — V1_reproj_C1 region=1 sigma_med=1.0244 rel_err1_all=0.0458/0.0467/0.0781 clean=true h1:acc=0.547,rel=0.081,norm=0.991,energy=0.983,manif=0.468,pert_growth=0.489 h2:acc=0.453,rel=0.099,norm=0.980,energy=0.960,manif=0.000,pert_growth=0.479 h4:acc=0.188,rel=0.172,norm=0.969,energy=0.939,manif=0.000,pert_growth=0.459 h8:acc=0.031,rel=0.283,norm=0.928,energy=0.862,manif=0.000,pert_growth=0.430 h16:acc=0.000,rel=0.431,norm=0.860,energy=0.739,manif=0.000,pert_growth=0.394 h32:acc=0.000,rel=0.638,norm=0.817,energy=0.668,manif=0.000,pert_growth=0.362 h64:acc=0.000,rel=0.897,norm=0.812,energy=0.660,manif=0.000,pert_growth=0.302
- 0x5A00 **FAIL** — V3_reproj_spectral_C1 region=0 sigma_med=1.0215 rel_err1_all=0.0458/0.0467/0.0781 clean=true h1:acc=0.484,rel=0.089,norm=0.969,energy=0.939,manif=0.468,pert_growth=0.489 h2:acc=0.359,rel=0.119,norm=0.953,energy=0.908,manif=0.000,pert_growth=0.472 h4:acc=0.109,rel=0.205,norm=0.932,energy=0.869,manif=0.000,pert_growth=0.443 h8:acc=0.000,rel=0.334,norm=0.891,energy=0.795,manif=0.000,pert_growth=0.383 h16:acc=0.000,rel=0.510,norm=0.820,energy=0.673,manif=0.000,pert_growth=0.335 h32:acc=0.000,rel=0.766,norm=0.763,energy=0.583,manif=0.000,pert_growth=0.303 h64:acc=0.000,rel=1.059,norm=0.726,energy=0.528,manif=0.000,pert_growth=0.221
- 0x5A00 **FAIL** — V0_pure_M1_mlp region=1 sigma_med=1.0196 rel_err1_all=0.0458/0.0467/0.0781 clean=true h1:acc=0.516,rel=0.088,norm=0.992,energy=0.984,manif=0.486,pert_growth=0.988 h2:acc=0.359,rel=0.109,norm=0.985,energy=0.971,manif=0.041,pert_growth=0.978 h4:acc=0.172,rel=0.189,norm=0.971,energy=0.942,manif=0.075,pert_growth=0.956 h8:acc=0.078,rel=0.303,norm=0.954,energy=0.910,manif=0.138,pert_growth=0.930 h16:acc=0.016,rel=0.454,norm=0.918,energy=0.843,manif=0.233,pert_growth=0.925 h32:acc=0.031,rel=0.639,norm=0.889,energy=0.791,manif=0.355,pert_growth=0.895 h64:acc=0.000,rel=0.850,norm=0.843,energy=0.710,manif=0.514,pert_growth=0.744
- 0x5A00 **FAIL** — V0_pure_C0_lock region=0 sigma_med=1.0356 rel_err1_all=0.0458/0.0467/0.0781 clean=true h1:acc=0.312,rel=0.105,norm=0.975,energy=0.951,manif=0.508,pert_growth=0.965 h2:acc=0.062,rel=0.147,norm=0.956,energy=0.915,manif=0.069,pert_growth=0.925 h4:acc=0.000,rel=0.252,norm=0.930,energy=0.866,manif=0.122,pert_growth=0.880 h8:acc=0.000,rel=0.410,norm=0.890,energy=0.793,manif=0.213,pert_growth=0.845 h16:acc=0.000,rel=0.631,norm=0.846,energy=0.716,manif=0.337,pert_growth=0.797 h32:acc=0.000,rel=0.907,norm=0.852,energy=0.727,manif=0.450,pert_growth=0.684 h64:acc=0.000,rel=1.094,norm=0.875,energy=0.769,manif=0.582,pert_growth=0.401

