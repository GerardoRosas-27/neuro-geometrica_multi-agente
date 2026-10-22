# ÓRBITA en `main`

`main` pasa de v4 a **ÓRBITA**. v4 queda archivado en el tag **`v4-final`** (`49c00a9`, el HEAD de `main` antes de este cambio). No se borró historia.
Fuente del código ÓRBITA: `exp/orbita-yo` en **`9166f1f`** (último commit antes de GROW). No entran GROW (`53e7f96`/`0b6f12a`), TFE, Cognitive Transformer, attn-gate ni el CDT/memoria archivado (`orbita/archive`).

## Qué incluye

| pieza | dónde | estado |
|---|---|---|
| ÓRBITA-v1: núcleo Koopman congelado + corrector Brenier (ICNN) | `orbita/src/model.rs`, `orbita/src/main.rs` | funciona (gana a v4 en CONFIRM) |
| Encoder D2/D3-num: Gemma 2 2B Q4_K_M (capa 13) + D2 + `numnorm` | `orbita/src/gemma_embed.rs`, `orbita/llm/{d2,d3,numnorm,e2e,pipeline}.py` | funciona |
| Sorpresa (predicción vs observación) | `orbita/src/{yo,learn,latent}.rs` | AUROC 0,96–0,99 |
| Integración de la observación (re-observación k=1–2 + re-identificación) | `orbita/src/fam.rs`, `orbita/llm/demo.py` | funciona; mejor que el «yo» |
| «yo» sólo como informante (confianza/sorpresa, no decide) | `orbita/src/yo.rs`, `orbita/llm/yo.py` | informa |
| Adaptador latente LAT / LAT_EP (aprendizaje acumulativo sin olvido en familias conocidas) | `orbita/src/latent.rs` (`lat-eval`), `orbita/llm/latlp.py` | funciona en conocidas |
| Sustrato v4 (dataset sellado, métricas, procedencia) | `src/v4_*.rs`, `src/field_autonomy_v4.rs` | lo usa ÓRBITA como dependencia (`cdt_rqm_epr`) y lo usa la GUI |
| GUI | `src/web/`, `src/bin/agentic_web.rs`, `web/static/` | pestaña **ÓRBITA** (`/api/orbita/*`, `src/web/orbita_job.rs`, `web/static/orbita.js`); chat/entrenamiento/sueño/Spider siguen sobre v4. Ver `docs/ORBITA_GUI.md` |

Retirado de `main` (sigue en `v4-final`): `src/bin/run_field_autonomy_v4.rs`, `examples/v4_*`, `scripts/v4_*`, `training/v4_psi64/`.

## Cómo correr

Requisitos: pesos por seed en `/workspace/orbita-data/{dev,confirm}/w_<seed>.json`, calibración del yo en `/workspace/orbita-data/yo/`,
Gemma en `/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf`, venv `/workspace/orbita-venv`. Son los valores por defecto; se cambian con `ORBITA_DATA`, `GEMMA_GGUF_PATH`, `ORBITA_BIN`, `ORBITA_LLAMACPP` (ver `docs/ORBITA_GUI.md`).

```
cd orbita && cargo build --release            # binario orbita/target/release/orbita
cd .. && sh orbita/demo/smoke.sh              # humo (R y PY se pueden sobrescribir por entorno)
PYTHONPATH=orbita /workspace/orbita-venv/bin/python orbita/llm/demo.py --text "Aplica una rotación con parámetros 0,5 y -0,2." --obs 1 --simulate 1,0.5,-0.2 --json
PYTHONPATH=orbita /workspace/orbita-venv/bin/python orbita/llm/test_numnorm.py
```
Más detalle: `orbita/demo/README.md`, `docs/demo_orbita.md`.

## Puntos de entrada para la GUI

1. **Pila completa (recomendado)**: `python orbita/llm/demo.py --text ... [--obs k --observed "12 números" | --simulate fam,p1,p2] [--x0 ...] [--seed 0xA400] [--no-explain] --json`.
   Salida JSON: `texto, seed, familia_leida, p_leido, p_fuente, prob_encoder, trayectoria_h1_h16, yo{confianza_acierto_h16, rasgos, sorpresa_vs_obs, sigma_jacobiano, ratio_norma_h16}, re_observacion{k, familia_reidentificada, p_reidentificado, error_ajuste, trayectoria_h1_h16}` (+ explicación de Gemma si no se pasa `--no-explain`).
2. **Binario Rust** `orbita/target/release/orbita`:
   - `gemma-embed --model <gguf> --in <txt> --out <tok> --exit-after 12 --per-token` (encoder)
   - `demo-step --weights w.json --c <contexto> --x0 <12 números> [--obs "y1|y2"]` (paso ÓRBITA + re-identificación, JSON)
   - `demo-rule --fam f --p p1,p2 --x ...` (simulador de la familia verdadera)
   - `lat-eval` (adaptador latente LAT/LAT_EP), `learn-eval`, `yo-feat`, `fam-eval`, `llm-eval`, `eval` (harness §16).
3. No hay API en biblioteca para ÓRBITA: el crate `orbita` es un binario aparte (no está en el workspace raíz). La GUI tendrá que invocar el binario/Python por proceso o exponer `orbita/src/model.rs` como lib.

## Resultados clave (CONFIRM, seeds `0xB400–0xB40F`, n = 16; copiados de los docs)

- **ÓRBITA vs v4** (`docs/resultados_orbita.md`, prerregistro `f5d7781`): gate E33 ÓRBITA 16/16 vs v4_E3 9/16; acc h16 0.323 vs 0.217; ORBITA − v4_E3 acc h16 +0.1065 [+0.0664, +0.1504] (15/16 seeds).
- **Encoder D3-num** (`docs/resultados_d3_vs_robusta.md`, `951fbf7`): v1 + D3NUM acc h16 0,2865 [0,2077, 0,3646] (TRUE 0,3164); max|p̂−p| mediano 0,0034.
- **E2E texto → ÓRBITA** (`docs/resultados_integracion_e2e.md`, `925015f`): E2E_R acc h16 0,2975 [0,2220, 0,3711]; parse_fail 0 de 1536; 0 queries.
- **Observación > yo**: re-observación OBS2 − BASE en v4 no iterables +0,0655 [+0,0484, +0,0842] (`docs/resultados_otras_familias.md`, `282f382`); el yo calibra (AUROC −p̂ → fallo 0,8476 [0,7972, 0,8960]) pero no decide (`docs/resultados_yo.md`, `14d68e3`).
- **Sorpresa** (`docs/resultados_aprendizaje_online.md`, `e2ac4ab`): AUROC de S_known nuevas vs conocidas 0–7: 0,9868 (DEV) / 0,9706 (CONFIRM); `docs/resultados_latente_largo_plazo.md` (`9166f1f`): 0,989–0,990 (DEV), 0,958–0,966 (CONFIRM).
- **Aprendizaje online** (`e2ac4ab`): HL2 sin olvido aceptada; LEARN2 − OBS2 conocidas 1–2 +0,0918 [+0,0703, +0,1133]; en nuevas 8–9 efecto mínimo (+0,0371).
- **Latente LAT** (`9166f1f`): H5 (sin olvido) aceptada en k = 2, 4, 8 y ambas fases; L4 memoria larga en conocidas 0–7 aceptada. **No aprende familias nuevas a largo plazo.**

## Ramas de experimentos

| rama / commit | resultado | doc |
|---|---|---|
| `exp/orbita` (`1bc624f`) | base ÓRBITA-v1 + ablaciones | `docs/resultados_orbita*.md`, `docs/resultados_ablaciones_orbita.md` |
| `exp/orbita-yo` `d391036`/`e2ac4ab` | aprendizaje online | `docs/resultados_aprendizaje_online.md` |
| `exp/orbita-yo` `9166f1f` | latente largo plazo | `docs/resultados_latente_largo_plazo.md` |
| `exp/orbita-yo` `0b6f12a` | **falló**: GROW (G1/G4/H_LAT refutadas) | `docs/lecciones/resultados_orbita_crece.md` |
| `exp/orbita-tf` `dcaadad` | **falló**: TFE no supera a ÓRBITA | `docs/lecciones/resultados_orbita_transformer_fe.md` |
| `exp/cognitive-transformer-orbita` `15a9524` | **falló**: colapso por memoria CDT | `docs/lecciones/resultados_cognitive_transformer_orbita.md` |
| `exp/orbita-attn-gate` `9db8466` | **cerrada** sin CONFIRM | `docs/lecciones/resultados_orbita_attn_gate.md` |
| CDT/memoria (`02ce4ed`, `e328ff2`, `e2f6bb0`) | CDT no aporta | `docs/resultados_cdt_*.md`, `docs/resultados_memoria_capas.md` |

## Para Arquitecto-IA (incompatibilidades)

- La GUI (`src/web/*`) sigue llamando a `field_v4`, `spider_field`, `prompt_spider`, `v4_controls`; no conoce ÓRBITA. Esos módulos se dejan en `src/` para que compile.
- ÓRBITA es un crate aparte (`orbita/`, con `[workspace]` propio) que depende de la lib raíz; `cargo build` en la raíz no lo compila y la CI actual no lo prueba.
- Rutas absolutas `/workspace/...` en `orbita/llm/*.py` (pesos, Gemma, venv). Los pesos no están en git. El Dockerfile/Railway no incluye Python, pesos ni el binario `orbita`.
- La GUI trabaja con prompts de texto libre; ÓRBITA espera texto que describa una transformación (familia + 2 parámetros) y un estado de 12 números (6 puntos 2D).
- LAT/LAT_EP sólo existe dentro de `lat-eval`/`latlp.py` (evaluación); falta un servicio con estado persistente para usarlo en línea.
