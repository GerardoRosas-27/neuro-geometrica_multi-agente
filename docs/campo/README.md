# Campo sobre LLM congelado (Qwen3-VL-4B)

Dos caminos que predicen un **campo de estado** a partir de las activaciones de
un LLM congelado. El tip integrado es `exp/campo-estado-turno` @ `a2c9c9f`
(5 nov 2026, 06:42 hora de México, UTC-6), que **contiene** la línea principal
`exp/gemma-campo-chat` @ `a420311` (5 nov 2026, 02:05 hora de México, UTC-6).

**Línea principal (para seguir trabajando):** `gemma_campo` — campo a nivel
*token*. **Camino alterno:** campo de estado *por turno* (`campo_turno/`).

Ambos usaron el mismo JSONL (SHA-256
`2e9ca4ca11abb7cb36f12645926c7d87ef6cb54020a33b19d460b0f1a85e6dbe`) y el
mismo LLM `qwen/qwen3-vl-4b` (capa 13, H=2560, `d_field=64`) en una AMD
Radeon RX 7800 XT. **No son comparables entre sí** (unidad token vs turno,
splits 60 vs 200 conversaciones de test, NLL distinta) ni con ATLAS-FW / BQ10
canónico (arnés CST4 de ÓRBITA). Esas tablas no cambian.

Los pesos grandes (`encoder.npz`, `model.bin`, `adapter.pt`, `*_h.npy`, el
JSONL) **no** están en el repo (están fuera a propósito; ninguno superaba un
límite de 50 MB en git: simplemente no se versionaron). Sí están hashes,
logs, configs y `eval_*.json`.

---

## Línea principal: gemma_campo (campo a nivel token)

Qué es: crate Rust `gemma_campo/` (ridge M7 + SGD WGSL sin carreras + NLMS
rápido, wgpu/Vulkan) más Python en `gemma_campo/python/`. El LLM está
congelado; el campo predice `z_t` (PCA 2560→64 de la capa 13) y, en el eval
de `a420311`, se combina la predicción contextual con la innovación causal
del embedding del token actual `e(x_t) W_res` y solo se sustituye el
subespacio contextual `V_ctx` (`k_skip=16`).

- Preregistro: [`docs/gemma_campo_chat/PLAN.md`](../gemma_campo_chat/PLAN.md)
  (congelado en `82271cb`). Runbook:
  [`docs/gemma_campo_chat/RUNBOOK_ENTRENAMIENTO_INFINITO.md`](../gemma_campo_chat/RUNBOOK_ENTRENAMIENTO_INFINITO.md)
  (`abf0f18`). README de una pasada: [`docs/gemma_campo_chat/README.md`](../gemma_campo_chat/README.md).
- Código del run: commit `4987f56` (anotado en
  [`run_20261104/commit.txt`](../gemma_campo_chat/run_20261104/commit.txt));
  el tip `a420311` añade `e(x_t)` / `V_ctx` y actualiza
  [`run_20261104/RESULTADOS.md`](../gemma_campo_chat/run_20261104/RESULTADOS.md)
  y [`eval_test.json`](../gemma_campo_chat/run_20261104/eval_test.json).
- Split del run: 1 800 conv train (9 shards de 200) + held-out fijo del
  shard 0 (60 val / 9 600 tokens, 60 test / 9 985 tokens). Permutación y
  bootstrap con semilla 12345. B=10 000.
- Modelo del run (no Gemma 2): `D:\agentesIA\qwen3_vl_4b_hf`
  (`qwen/qwen3-vl-4b`), L=13. Hardware: RX 7800 XT, campo por Vulkan/wgpu y
  LLM por DirectML (`privateuseone:0`, `torch==2.4.1`,
  `torch-directml==0.2.5.dev240914`, `transformers==4.57.6`). Congelación del
  entorno: [`run_20261104/env_freeze.txt`](../gemma_campo_chat/run_20261104/env_freeze.txt).
- Dataset: `HuggingFaceTB/everyday-conversations-llama3.1-2k` (`train_sft`),
  vía `gemma_campo/python/fetch_jsonl.py`. SHA del JSONL en
  [`run_20261104/data_sha.txt`](../gemma_campo_chat/run_20261104/data_sha.txt).
- Hashes (archivos no versionados): `encoder.npz`
  `41e6c15b3ecc807ab55fbef1b74b6de0586740b9eb7adb80f7e44197b5633152`;
  `model.bin` `d8fae45bc96b6ea237e0edffbd679a6d02b67ad6abe8507a1434ba10e255734a`;
  `token_res.npz` `8e686916d7bc631f1612c594e730ec672a4945ce39eb2cb90c5cd9102899a6c8`
  ([`sha.txt`](../gemma_campo_chat/run_20261104/sha.txt)).

### Resultados (de `eval_test.json`, n=60)

| k / contraste | campo | repeat | campo−repeat [IC95] | P(Δ≤0) |
|---|---:|---:|---|---:|
| 1 con `e(x_t)` | 0.197222 | 1.238695 | −1.041472 [−1.063950, −1.016694] | 1.0 |
| 1 solo contexto (sin `e(x_t)`) | 0.417240 | 1.238695 | −0.821455 [−0.841033, −0.801295] | 1.0 |
| 2 | 0.567080 | 1.486495 | −0.919416 [−0.943512, −0.896036] | 1.0 |
| 4 | 0.691082 | 1.615537 | −0.924456 [−0.946453, −0.901925] | 1.0 |
| 8 | 0.793802 | 1.686998 | −0.893195 [−0.913777, −0.871005] | 1.0 |
| campo−lenta (k=1) | 0.197222 | 0.418979 (lenta) | −0.221757 [−0.227630, −0.214936] | 1.0 |

NLL/token (`k_skip=16` + `e(x_t)`):

| Condición | NLL/token | PPL |
|---|---:|---:|
| gemma (LLM intacto) | 1.7145656049251556 | 5.554262244080218 |
| oracle (PCA contextual) | 1.7150465667247772 | 5.556934274565487 |
| campo | 1.7339503705501556 | 5.662980651931948 |
| repeat-last | 3.049496765931447 | 21.104721135082237 |

| Δ | media [IC95] | P(Δ≤0) |
|---|---|---:|
| campo−oracle | +0.01890380382537842 [0.010692580093940099, 0.027486240863800047] | 0.0 |
| campo−gemma | +0.019384765625 [0.011221353212992351, 0.027918350050846735] | 0.0 |
| campo−repeat | −1.3155463953812918 [−1.389042289654414, −1.2422269453605017] | 1.0 |

Antes de `e(x_t)` y del recorte a `V_ctx`, campo−oracle era +2.944636
[2.853318, 3.037305] (`RESULTADOS.md`).

Hipótesis (veredicto en `RESULTADOS.md`): **H1, H1k (k=2,4,8), H2, H3, OLV,
EST, REP = éxito.** H2: extremo superior del IC de campo−oracle = 0.027486 ≤
0.05. Olvido shard 1: 0.403230 / 0.402799 = 1.00107 ≤ 1.05
([`olvido_final.txt`](../gemma_campo_chat/run_20261104/olvido_final.txt)).
REP: `rep_predict_byte_identical=true` ([`rep_check.json`](../gemma_campo_chat/run_20261104/rep_check.json)).
Identidad: `sanity_identity_maxdiff = 0.0` (límite 0.05).

`g5_eval_test.json` es un eval auxiliar (n=5) usado sobre todo para la
identidad; **no** es el veredicto de `a420311`.

Limitaciones: un solo run / una semilla de bootstrap (12345); test de n=60
ya consultado; el campo no añade capacidad de lenguaje (capas 1–36 y LM head
siguen siendo del LLM); campo queda *peor* que el modelo intacto por +0.0194
nats/token (dentro de H2). Selección de η=0.03, γ=1, `k_skip=16` y
`λ_tok=5.0` en val.

### Reproducción (línea principal)

Hardware del run: Windows, AMD Radeon RX 7800 XT (16 GB). Campo: Vulkan/wgpu.
LLM: DirectML (`--device dml`, no ROCm). Python 3.10+,
`pip install -r gemma_campo/python/requirements-rocm.txt` más
`torch==2.4.1` y `torch-directml==0.2.5.dev240914` (el runbook describe ROCm
`cuda` como alternativa). Rust estable.

El PLAN presume `google/gemma-2-2b` con `HF_TOKEN` (licencia en Hugging Face;
el token solo del entorno, nunca en archivos). El run real usó
`MODEL=D:/agentesIA/qwen3_vl_4b_hf` (`qwen/qwen3-vl-4b`, `LAYER=13`) porque
no había `HF_TOKEN` para Gemma. Para repetir el run reportado:

```bash
# datos (fuera del repo)
python gemma_campo/python/fetch_jsonl.py --out $RUN/all.jsonl
# default: HuggingFaceTB/everyday-conversations-llama3.1-2k, split train_sft
sha256sum $RUN/all.jsonl
# debe ser 2e9ca4ca11abb7cb36f12645926c7d87ef6cb54020a33b19d460b0f1a85e6dbe

cd gemma_campo
cargo build --release
cargo test --release
./target/release/gemma_campo validate-wgsl
./target/release/gemma_campo check-gpu
./target/release/gemma_campo check-gpu --n 2048 --d 64 --ctx 4 --steps 3

export PY=python
export EXTRACT_ARGS="--jsonl $RUN/all.jsonl --layer 13 --max-tokens 384 --dtype float16 --device dml --model D:/agentesIA/qwen3_vl_4b_hf"
export SHARD_SIZE=200 N_VAL=60 N_TEST=60 D_FIELD=64 LR0=0.05 LR_T=20 EPOCHS=2 BATCH=1024 REPLAY=2 KEEP_H=2 DEVICE=gpu THREADS=6
bash gemma_campo/run_stream.sh   # 9 shards en el run 20261104

F=$RUN/final
cp $RUN/ckpt/last_good/model.* $F/
./gemma_campo/target/release/gemma_campo predict --data $RUN/shards/0/z --model $F --split test --kmax 8 --stride 4
python gemma_campo/python/parity.py --zdir $RUN/shards/0/z --model $F
python gemma_campo/python/eval.py --hdir $RUN/shards/0/h --zdir $RUN/shards/0/z --model-dir $F --device dml --out $F/eval
```

Secuencia canónica del runbook (Gemma 2 + ROCm) en
`docs/gemma_campo_chat/RUNBOOK_ENTRENAMIENTO_INFINITO.md` §7. Duración del
bucle de shards del run: 31 min 26 s (4 nov 2026, 23:51:10 a 5 nov 2026,
00:22:36 hora de México, UTC-6).

---

## Camino alterno: campo de estado por turno

Qué es: `campo_turno/` extrae un embedding por *turno* (pool mean de la capa
13), proyecta a z (d=64) y entrena el mismo crate `gemma_campo` más un
adaptador soft-prompt (P=4) que no toca el LLM. Sirve para preguntar si el
estado de la conversación ayuda a generar la siguiente respuesta del asistente.

- Preregistro: [`docs/campo_estado_turno/PLAN.md`](../campo_estado_turno/PLAN.md)
  (`84f19f6`). Runbook: [`docs/campo_estado_turno/RUNBOOK.md`](../campo_estado_turno/RUNBOOK.md).
  Código: `317c38a`. Resultados:
  [`run_20261105/RESULTADOS.md`](../campo_estado_turno/run_20261105/RESULTADOS.md)
  = [`eval_test.json`](../campo_estado_turno/run_20261105/eval_test.json).
- Split: train 1600 conv / 12 385 turnos; val 200 / 1 565; test 200 / 1 548
  (NLL en 759 turnos de asistente). `--n-conv 2000` sobre el mismo JSONL.
- Mismo modelo Qwen3-VL-4B, L=13, pool=mean, float16, `--device dml`.
  Entorno: [`run_20261105/env_freeze.txt`](../campo_estado_turno/run_20261105/env_freeze.txt)
  (el archivo en git contiene un byte nulo; salió de `pip freeze` en Windows).
- Hashes: encoder.npz `8f5813d05fb95cdc6f437fe376127494b88b933ac241ae8d4d449be5a26b3fa0`;
  model.bin `13d2dca20170d895c427d13adb2c64d917f4e808ab9c707b4c49ada222dd6cdf`;
  adapter.pt `9501be7da8cf688ba59848f1c6fe528a77fa88d999bb6ca7ea4ccc75067db479`.
  LLM SHA antes y después:
  `78685cdf6081c3802b007200ddbba61ae24eb29c7ca25416baeacdefef7484ee` (intacto).

### Resultados (test, n=200, B=10 000, semilla 12345)

z-MSE de turnos assistant: campo 0.3685527244210243; repeat 1.0917954957485199;
repeat_asst 1.1434330561757087; mean 0.9872847780585289;
ridge 0.4062537436932325; mlp 0.38687286220490935.

| Contraste | media [IC95] | ¿pasa? |
|---|---|---|
| T1 campo−repeat | −0.7232427713274956 [−0.7550630325078964, −0.692163755280897] | sí |
| T1 campo−mean | −0.6187320536375046 [−0.6475693066623062, −0.5908669548947364] | sí |
| T2 campo−mlp | −0.018320137783885002 [−0.024368760069832206, −0.012354790948331359] (umbral 0.05·mlp = 0.01934364311024547) | sí |

NLL/token de la respuesta real: none 1.9457939405404034 (PPL 6.999186587659266);
campo 0.6475953880107791 (1.9109402301150247); true 0.6301269838271273;
shuf 0.9346708592687366; mlp 0.6469799832911522.

| Contraste | media [IC95] | ¿pasa? |
|---|---|---|
| T3 campo−none | −1.2981985525296238 [−1.3309107688946231, −1.2654467306231854] | sí |
| T4 campo−shuf | −0.2870754712579576 [−0.3118449081435307, −0.2625430659422041] | sí |
| campo−true | +0.01746840418365173 [0.012052836236839417, 0.023955998619875577] | (info) |
| mlp−campo | −0.0006154047196268981 [−0.003264976345001743, 0.0019708417281849473] | empate (IC cruza 0) |

No-daño: `identidad_ok=true` (máx \|Δ\|=0.0 < 0.05), LLM intacto.

Limitaciones: un solo run; test ya consultado. T3 vs `none` infla el estilo
del adaptador: `shuf` ya baja la NLL de 1.9458 a 0.9347; el estado propio
aporta −0.2871 frente a `shuf`. En decodificación, campo ≈ mlp. El NLMS
elegido tuvo η=0 (sin adaptación online). Codificador: var. explicada
0.7814175983846907 (peor que el PCA token-level de la línea principal,
0.998367060873689).

### Reproducción (camino alterno)

Datos grandes fuera del repo (`/c/ct_work` en el run). Mismos requisitos de
GPU/Python/Rust. Comandos del
[`RUNBOOK.md`](../campo_estado_turno/RUNBOOK.md), con el modelo que sí se usó:

```bash
export RUN=$PWD/docs/campo_estado_turno/run_$(date +%Y%m%d); mkdir -p $RUN
export W=/c/ct_work; mkdir -p $W
export PY=python MODEL=D:/agentesIA/qwen3_vl_4b_hf LAYER=13 DEV=dml DT=float16
pip freeze > $RUN/env_freeze.txt

(cd gemma_campo && cargo build --release && cargo test --release)
gemma_campo/target/release/gemma_campo validate-wgsl
gemma_campo/target/release/gemma_campo check-gpu
gemma_campo/target/release/gemma_campo check-gpu --n 2048 --d 64 --ctx 4
(cd campo_turno && $PY -m unittest discover -s tests -v)

$PY gemma_campo/python/fetch_jsonl.py --out $W/all.jsonl
sha256sum $W/all.jsonl   # 2e9ca4ca11abb7cb36f12645926c7d87ef6cb54020a33b19d460b0f1a85e6dbe

cd campo_turno
$PY turno_extract.py --model $MODEL --layer $LAYER --pool mean --jsonl $W/all.jsonl --n-conv 2000 \
    --max-turn-tokens 128 --dtype $DT --device $DEV --out $W/h
$PY turno_encoder.py --hdir $W/h --out $W/z --d-field 64
BIN=../gemma_campo/target/release/gemma_campo
$BIN train --data $W/z --out $W/m --device gpu --ctx 4 --quad 1 --epochs 20 --batch 256 --lr 0.01 --threads 8
$BIN predict --data $W/z --model $W/m --split val  --kmax 1 --stride 1
$BIN predict --data $W/z --model $W/m --split test --kmax 1 --stride 1
$PY turno_baselines.py --zdir $W/z --out $W/b --ctx 4
$PY turno_adapter.py --model $MODEL --dtype $DT --device $DEV --hdir $W/h --zdir $W/z --out $W/a \
    --P 4 --epochs 2 --lr 1e-3 --accum 8 --noise 0.1 --ctx-tokens 384 --loss-scale 128
$PY turno_eval.py --split test --model $MODEL --dtype $DT --device $DEV --hdir $W/h --zdir $W/z \
    --field-dir $W/m --base-dir $W/b --adapter-dir $W/a --out $W/eval --n-samples 5 --max-new 60
```

---

## Parche externo de DirectML (`sitecustomize.py`)

**No está en ninguna rama del repo** (búsqueda de `sitecustomize.py` y
`disable_tiled_resources` en todo el historial: solo se menciona en
`docs/campo_estado_turno/run_20261105/RESULTADOS.md`). El run por turno lo
activó poniendo en `PYTHONPATH` un `sitecustomize.py` *fuera del repositorio*:

```python
import torch_directml
torch_directml.disable_tiled_resources(True)
```

Hace falta para evitar la degradación de tablas de páginas D3D12 Tiled
Resources en `torch-directml` tras cientos de pasadas con longitudes
dinámicas. Sin ese parche, extracciones largas en DirectML pueden fallar o
volverse muy lentas. No formó parte del código versionado a propósito
(«sin tocar código del repo»).

---

## Qué no se versionó (necesario para bit-exacto, no para reentrenar)

- `all.jsonl` (se descarga; hay que comprobar el SHA).
- `encoder.npz`, `model.bin`, `adapter.pt`, `token_res.npz`, estados `*_h.npy`
  (hashes arriba; viven en el disco del run, p. ej. `/c/ct_work` o `$RUN/shards`).
- Pesos del LLM (`D:/agentesIA/qwen3_vl_4b_hf` o `google/gemma-2-2b` + `HF_TOKEN`).
- El `sitecustomize.py` de DirectML.

Reentrenar con los comandos de arriba reproduce el *protocolo*; igualar al
bit los JSON de test exige además esos artefactos y el mismo parche/entorno.
