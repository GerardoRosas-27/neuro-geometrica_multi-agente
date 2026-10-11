# v4 dev summary

- git SHA: `86a750a56ee4b639bac9c908684867461a24fe64` (dirty=false)
- hp lock sha256: `9578f730b57cb0f311628743620f632e10a5f02009406ad70555147854624647`
- seeds: 16
- manifests: `dataset_manifest.jsonl` (sha256 of file: `7b70a417efe5313d6238dc01cdcd645faeeead7cd32b0bc0975cf80e14a92aa3`)

| Exp | PASS | FAIL | n |
|---|---|---|---|
| DIST_A | 3 | 13 | 16 |

## Per-seed gate notes

### DIST_A

- 0xA400 **FAIL** — T=0.8125 A=0.1875 B=0.3984 D1=0.3945 M1=0.5391 D4=0.5117 relerr_B=0.0638 relerr_D1=0.0651 relerr_T=0.0236 fresh_D1=0.3945(q=0,pid=622524) clean=true ckpt=958663ab0a54
- 0xA401 **PASS** — T=0.8281 A=0.2695 B=0.3711 D1=0.4258 M1=0.2930 D4=0.6211 relerr_B=0.0673 relerr_D1=0.0682 relerr_T=0.0289 fresh_D1=0.4258(q=0,pid=622602) clean=true ckpt=c512519a2a14
- 0xA402 **PASS** — T=0.8203 A=0.2109 B=0.3672 D1=0.4414 M1=0.2734 D4=0.7383 relerr_B=0.0698 relerr_D1=0.0690 relerr_T=0.0312 fresh_D1=0.4414(q=0,pid=622597) clean=true ckpt=15d4e001f293
- 0xA403 **FAIL** — T=0.8320 A=0.1523 B=0.5039 D1=0.4453 M1=0.4922 D4=0.6172 relerr_B=0.0625 relerr_D1=0.0656 relerr_T=0.0302 fresh_D1=0.4453(q=0,pid=622604) clean=true ckpt=2c5e79ae4169
- 0xA404 **FAIL** — T=0.8320 A=0.2734 B=0.4492 D1=0.3906 M1=0.3633 D4=0.4688 relerr_B=0.0714 relerr_D1=0.0735 relerr_T=0.0340 fresh_D1=0.3906(q=0,pid=622526) clean=true ckpt=0af8735c0149
- 0xA405 **FAIL** — T=0.8438 A=0.2227 B=0.3906 D1=0.4023 M1=0.2891 D4=0.4492 relerr_B=0.0679 relerr_D1=0.0672 relerr_T=0.0267 fresh_D1=0.4023(q=0,pid=622594) clean=true ckpt=60022e520090
- 0xA406 **FAIL** — T=0.8438 A=0.2539 B=0.2227 D1=0.2070 M1=0.3320 D4=0.7500 relerr_B=0.0754 relerr_D1=0.0792 relerr_T=0.0233 fresh_D1=0.2070(q=0,pid=622528) clean=true ckpt=40efaa83789b
- 0xA407 **FAIL** — T=0.8320 A=0.1836 B=0.3672 D1=0.3125 M1=0.2422 D4=0.6289 relerr_B=0.0727 relerr_D1=0.0781 relerr_T=0.0260 fresh_D1=0.3125(q=0,pid=622603) clean=true ckpt=814d7264915c
- 0xA408 **FAIL** — T=0.8164 A=0.1328 B=0.2461 D1=0.2656 M1=0.3281 D4=0.6992 relerr_B=0.0754 relerr_D1=0.0761 relerr_T=0.0259 fresh_D1=0.2656(q=0,pid=622527) clean=true ckpt=c8dd57979923
- 0xA409 **FAIL** — T=0.8242 A=0.2148 B=0.4023 D1=0.2695 M1=0.3398 D4=0.5781 relerr_B=0.0696 relerr_D1=0.0768 relerr_T=0.0298 fresh_D1=0.2695(q=0,pid=622595) clean=true ckpt=309932c86909
- 0xA40A **FAIL** — T=0.8047 A=0.2461 B=0.2422 D1=0.2266 M1=0.1914 D4=0.4570 relerr_B=0.0723 relerr_D1=0.0748 relerr_T=0.0290 fresh_D1=0.2266(q=0,pid=622521) clean=true ckpt=5c0d2dd39baa
- 0xA40B **FAIL** — T=0.8398 A=0.3086 B=0.3984 D1=0.4297 M1=0.1758 D4=0.4375 relerr_B=0.0632 relerr_D1=0.0636 relerr_T=0.0220 fresh_D1=0.4297(q=0,pid=622592) clean=true ckpt=ea25795fa8cd
- 0xA40C **FAIL** — T=0.8320 A=0.2383 B=0.2500 D1=0.2383 M1=0.2461 D4=0.7070 relerr_B=0.0736 relerr_D1=0.0762 relerr_T=0.0241 fresh_D1=0.2383(q=0,pid=622523) clean=true ckpt=a06d0b793c76
- 0xA40D **PASS** — T=0.8320 A=0.2812 B=0.2266 D1=0.2773 M1=0.2773 D4=0.6016 relerr_B=0.0779 relerr_D1=0.0795 relerr_T=0.0273 fresh_D1=0.2773(q=0,pid=622525) clean=true ckpt=9e821bc2c85a
- 0xA40E **FAIL** — T=0.8242 A=0.2852 B=0.1875 D1=0.2031 M1=0.2070 D4=0.7344 relerr_B=0.0790 relerr_D1=0.0808 relerr_T=0.0284 fresh_D1=0.2031(q=0,pid=622522) clean=true ckpt=d3610a3dec09
- 0xA40F **FAIL** — T=0.8477 A=0.2422 B=0.3945 D1=0.4180 M1=0.5000 D4=0.7539 relerr_B=0.0657 relerr_D1=0.0658 relerr_T=0.0277 fresh_D1=0.4180(q=0,pid=622593) clean=true ckpt=901d9f46080f

