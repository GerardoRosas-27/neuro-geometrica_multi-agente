# Resultados — ÓRBITA + corrector de atención gateado (rama exp/orbita-attn-gate)

**Estado: cerrada sin CONFIRM por decisión del usuario 2026-10-22.**

- Solo DEV. Los resultados de abajo son **exploratorios, no un veredicto**: no se ejecutó la fase CONFIRM.
- Consultas a TEST: **0**.
- Fuente: `artifacts/orbita/attn_gate/analyze_dev.md` (copiado literalmente abajo), con curvas `curve_dev_0xA40*.csv`, `per_seed_dev.json`, `tau.json`. Artefactos DEV pre-enmienda en `artifacts/orbita/attn_gate/dev_pre_enmienda/` (commit fe5650d).

## Historial de convergencia del corrector

1. Corrida original, 20000 pasos: C2 falló en las seeds 0xA409 y 0xA40E (commit fe5650d).
2. Enmienda 8d9c1c2 (corrector 40000 pasos, coseno a 1e-5; sin otro cambio experimental): C2 pasa en todas las seeds; C1 falla en 0xA403 (val final 0.0844 vs límite 0.0837). Por tanto la convergencia sigue fallando según el prerregistro.

## Resultados DEV (k = 2, 4, 8) — copia de analyze_dev.md

Comprobaciones: discrepancias OBS Python vs Rust y LAT_EP vs artefacto dcaadad = 0 (debe ser 0). τ: {"k2_q0.90": 0.08803585189900001, "k2_q0.95": 0.10128905386999998, "k2_q0.99": 0.12421525018200003, "k4_q0.90": 0.12492585092, "k4_q0.95": 0.15068405667999998, "k4_q0.99": 0.19177612447800005, "k8_q0.90": 0.19027896746, "k8_q0.95": 0.24799849591999995, "k8_q0.99": 0.32440083432500005}

## k = 2

| id | A − B | grupo | media [IC95] | umbral (inf. IC95 >) | cumple |
|---|---|---|---|---|---|
| H1 | GATE − LAT_EP | nuevas | -0.0242 [-0.0283, -0.0203] | +0.00 | opuesto |
| H2 | GATE − ORBITA | conocidas | -0.0024 [-0.0032, -0.0016] | -0.05 | sí |
| H3 | GATE − ALWAYS | conocidas | +0.1022 [+0.0874, +0.1180] | +0.00 | sí |
| H4 | GATE − TF-PE | conocidas | -0.2147 [-0.2429, -0.1855] | +0.00 | opuesto |
| S1 | GATE − ORBITA | nuevas | -0.0002 [-0.0007, +0.0000] | +0.00 | no |
| S2 | ALWAYS − LAT_EP | nuevas | -0.0266 [-0.0305, -0.0232] | +0.00 | opuesto |
| S3 | GATE − TF-PE | nuevas | -0.0471 [-0.0515, -0.0425] | +0.00 | opuesto |
| S4 | GATE − LAT_EP | conocidas | -0.2654 [-0.2759, -0.2544] | +0.00 | opuesto |
| S5 | ALWAYS − ORBITA | nuevas | -0.0027 [-0.0049, -0.0002] | +0.00 | opuesto |
| S6 | ALWAYS − ORBITA | conocidas | -0.1046 [-0.1204, -0.0901] | -0.05 | opuesto |

| brazo | nuevas acc16 | conocidas acc16 |
|---|---|---|
| ORBITA | 0.0137 [0.0117, 0.0156] | 0.1587 [0.1453, 0.1718] |
| TF-PE | 0.0605 [0.0559, 0.0652] | 0.3710 [0.3481, 0.3926] |
| GATE | 0.0134 [0.0115, 0.0154] | 0.1562 [0.1423, 0.1696] |
| GATE_q0.90 | 0.0129 [0.0112, 0.0146] | 0.1555 [0.1416, 0.1686] |
| GATE_q0.99 | 0.0137 [0.0117, 0.0156] | 0.1586 [0.1450, 0.1718] |
| ALWAYS | 0.0110 [0.0085, 0.0137] | 0.0541 [0.0466, 0.0620] |
| LAT_EP | 0.0376 [0.0342, 0.0415] | 0.4216 [0.4052, 0.4381] |

| tasa de disparo | nuevas | conocidas |
|---|---|---|
| GATE_q0.90 | 0.4688 [0.4658, 0.4717] | 0.0948 [0.0927, 0.0969] |
| GATE_q0.95 | 0.3738 [0.3708, 0.3770] | 0.0529 [0.0510, 0.0547] |
| GATE_q0.99 | 0.2473 [0.2454, 0.2493] | 0.0159 [0.0149, 0.0171] |

## k = 4

| id | A − B | grupo | media [IC95] | umbral (inf. IC95 >) | cumple |
|---|---|---|---|---|---|
| H1 | GATE − LAT_EP | nuevas | -0.0403 [-0.0461, -0.0344] | +0.00 | opuesto |
| H2 | GATE − ORBITA | conocidas | +0.0004 [+0.0000, +0.0007] | -0.05 | sí |
| H3 | GATE − ALWAYS | conocidas | +0.0917 [+0.0751, +0.1091] | +0.00 | sí |
| H4 | GATE − TF-PE | conocidas | -0.2217 [-0.2484, -0.1936] | +0.00 | opuesto |
| S1 | GATE − ORBITA | nuevas | -0.0005 [-0.0012, +0.0000] | +0.00 | no |
| S2 | ALWAYS − LAT_EP | nuevas | -0.0437 [-0.0496, -0.0381] | +0.00 | opuesto |
| S3 | GATE − TF-PE | nuevas | -0.0505 [-0.0557, -0.0457] | +0.00 | opuesto |
| S4 | GATE − LAT_EP | conocidas | -0.3678 [-0.3805, -0.3542] | +0.00 | opuesto |
| S5 | ALWAYS − ORBITA | nuevas | -0.0039 [-0.0088, +0.0010] | +0.00 | no |
| S6 | ALWAYS − ORBITA | conocidas | -0.0913 [-0.1089, -0.0747] | -0.05 | opuesto |

| brazo | nuevas acc16 | conocidas acc16 |
|---|---|---|
| ORBITA | 0.0278 [0.0251, 0.0305] | 0.1835 [0.1700, 0.1960] |
| TF-PE | 0.0779 [0.0732, 0.0825] | 0.4055 [0.3827, 0.4271] |
| GATE | 0.0273 [0.0247, 0.0300] | 0.1838 [0.1703, 0.1964] |
| GATE_q0.90 | 0.0234 [0.0205, 0.0266] | 0.1837 [0.1703, 0.1963] |
| GATE_q0.99 | 0.0278 [0.0251, 0.0305] | 0.1836 [0.1702, 0.1962] |
| ALWAYS | 0.0239 [0.0198, 0.0281] | 0.0922 [0.0807, 0.1041] |
| LAT_EP | 0.0676 [0.0630, 0.0720] | 0.5516 [0.5342, 0.5681] |

| tasa de disparo | nuevas | conocidas |
|---|---|---|
| GATE_q0.90 | 0.5078 [0.5042, 0.5117] | 0.1030 [0.0997, 0.1063] |
| GATE_q0.95 | 0.3489 [0.3462, 0.3513] | 0.0479 [0.0458, 0.0502] |
| GATE_q0.99 | 0.2239 [0.2219, 0.2258] | 0.0144 [0.0129, 0.0159] |

## k = 8

| id | A − B | grupo | media [IC95] | umbral (inf. IC95 >) | cumple |
|---|---|---|---|---|---|
| H1 | GATE − LAT_EP | nuevas | -0.0764 [-0.0830, -0.0698] | +0.00 | opuesto |
| H2 | GATE − ORBITA | conocidas | +0.0002 [+0.0000, +0.0006] | -0.05 | sí |
| H3 | GATE − ALWAYS | conocidas | -0.0144 [-0.0350, +0.0065] | +0.00 | no |
| H4 | GATE − TF-PE | conocidas | -0.2272 [-0.2501, -0.2028] | +0.00 | opuesto |
| S1 | GATE − ORBITA | nuevas | +0.0000 [+0.0000, +0.0000] | +0.00 | no |
| S2 | ALWAYS − LAT_EP | nuevas | -0.0369 [-0.0461, -0.0276] | +0.00 | opuesto |
| S3 | GATE − TF-PE | nuevas | -0.0911 [-0.0974, -0.0840] | +0.00 | opuesto |
| S4 | GATE − LAT_EP | conocidas | -0.4949 [-0.5112, -0.4805] | +0.00 | opuesto |
| S5 | ALWAYS − ORBITA | nuevas | +0.0396 [+0.0315, +0.0474] | +0.00 | sí |
| S6 | ALWAYS − ORBITA | conocidas | +0.0146 [-0.0061, +0.0352] | -0.05 | sí |

| brazo | nuevas acc16 | conocidas acc16 |
|---|---|---|
| ORBITA | 0.0305 [0.0278, 0.0330] | 0.2410 [0.2311, 0.2506] |
| TF-PE | 0.1216 [0.1150, 0.1277] | 0.4684 [0.4486, 0.4863] |
| GATE | 0.0305 [0.0278, 0.0330] | 0.2412 [0.2313, 0.2509] |
| GATE_q0.90 | 0.0305 [0.0278, 0.0330] | 0.2423 [0.2323, 0.2521] |
| GATE_q0.99 | 0.0305 [0.0278, 0.0330] | 0.2410 [0.2311, 0.2506] |
| ALWAYS | 0.0701 [0.0635, 0.0769] | 0.2556 [0.2340, 0.2772] |
| LAT_EP | 0.1069 [0.1003, 0.1133] | 0.7361 [0.7197, 0.7521] |

| tasa de disparo | nuevas | conocidas |
|---|---|---|
| GATE_q0.90 | 0.4624 [0.4592, 0.4656] | 0.1007 [0.0966, 0.1049] |
| GATE_q0.95 | 0.3118 [0.3091, 0.3142] | 0.0518 [0.0493, 0.0542] |
| GATE_q0.99 | 0.1897 [0.1882, 0.1912] | 0.0133 [0.0116, 0.0150] |

## Convergencia del corrector (criterios C1–C4 del prerregistro)

| seed | val(0) | val mín | val final | train último 10 % | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|
| 0xA400 | 0.9265 | 0.0476 | 0.0478 | 0.0367 | ✔ | ✔ | ✔ | ✔ |
| 0xA401 | 0.6333 | 0.0621 | 0.0637 | 0.0362 | ✔ | ✔ | ✔ | ✔ |
| 0xA402 | 0.4220 | 0.0399 | 0.0418 | 0.0338 | ✔ | ✔ | ✔ | ✔ |
| 0xA403 | 0.4884 | 0.0797 | 0.0844 | 0.0402 | ✘ | ✔ | ✔ | ✔ |
| 0xA404 | 0.6416 | 0.0519 | 0.0525 | 0.0412 | ✔ | ✔ | ✔ | ✔ |
| 0xA405 | 0.4195 | 0.0622 | 0.0633 | 0.0390 | ✔ | ✔ | ✔ | ✔ |
| 0xA406 | 0.5532 | 0.0775 | 0.0775 | 0.0425 | ✔ | ✔ | ✔ | ✔ |
| 0xA407 | 0.3578 | 0.0525 | 0.0535 | 0.0421 | ✔ | ✔ | ✔ | ✔ |
| 0xA408 | 0.7166 | 0.0772 | 0.0789 | 0.0450 | ✔ | ✔ | ✔ | ✔ |
| 0xA409 | 0.5421 | 0.0706 | 0.0733 | 0.0471 | ✔ | ✔ | ✔ | ✔ |
| 0xA40A | 0.2601 | 0.0543 | 0.0543 | 0.0464 | ✔ | ✔ | ✔ | ✔ |
| 0xA40B | 0.6223 | 0.0571 | 0.0592 | 0.0435 | ✔ | ✔ | ✔ | ✔ |
| 0xA40C | 0.7696 | 0.0370 | 0.0379 | 0.0371 | ✔ | ✔ | ✔ | ✔ |
| 0xA40D | 0.4124 | 0.0740 | 0.0756 | 0.0455 | ✔ | ✔ | ✔ | ✔ |
| 0xA40E | 0.6019 | 0.0760 | 0.0768 | 0.0431 | ✔ | ✔ | ✔ | ✔ |
| 0xA40F | 0.3454 | 0.0412 | 0.0418 | 0.0421 | ✔ | ✔ | ✔ | ✔ |

**Convergencia: FALLA en alguna seed**
