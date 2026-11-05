# RUNBOOK — campo de estado por turno (AMD RX 7800 XT, Windows)

> Para el agente que ejecuta. Preregistro: [`PLAN.md`](PLAN.md) (congelado en el commit `84f19f6`). NO cambiar hipótesis,
> umbrales, semillas ni splits. Test se evalúa UNA vez (paso 7). Si algo falla, anotar y parar; no "arreglar" mirando test.
> Nunca tocar `main` ni `run_v6`; nunca `push --force`. Los resultados van en `docs/campo_estado_turno/run_<AAAAMMDD>/`.

## 0. Entorno (igual que el run 20261104)
- Windows + Git Bash (todos los comandos son bash). GPU: AMD Radeon RX 7800 XT.
- Python 3.10+ con `torch==2.4.1` + `torch-directml==0.2.5.dev240914` (funcionó en el run anterior; dispositivo `--device dml`)
  y `pip install -r gemma_campo/python/requirements-rocm.txt` (transformers 4.57.6). ROCm es opcional (`--device cuda` si `torch.version.hip`).
- Rust estable (`cargo`) para el crate `gemma_campo` (wgpu → Vulkan en la RX 7800 XT).
- `HF_TOKEN` en el entorno si se usa `google/gemma-2-2b` (licencia). Nunca escribirlo en archivos.

```bash
git fetch origin && git switch exp/campo-estado-turno && git rev-parse HEAD
export RUN=$PWD/docs/campo_estado_turno/run_$(date +%Y%m%d); mkdir -p $RUN; git rev-parse HEAD > $RUN/commit.txt
export W=/c/ct_work; mkdir -p $W            # datos grandes FUERA del repo
export PY=python MODEL=google/gemma-2-2b LAYER=13 DEV=dml DT=float16
# alternativa usada en el run anterior: MODEL=qwen/qwen3-vl-4b LAYER=13 (anotarlo en RESULTADOS)
pip freeze > $RUN/env_freeze.txt
python -c "import torch,torch_directml;print(torch.__version__, torch_directml.device_name(0))" >> $RUN/env_freeze.txt
```

## 1. Fase 0 (puertas; si alguna falla, PARAR)
```bash
(cd gemma_campo && cargo build --release && cargo test --release 2>&1 | tee $RUN/fase0.txt)
gemma_campo/target/release/gemma_campo validate-wgsl | tee -a $RUN/fase0.txt
gemma_campo/target/release/gemma_campo check-gpu | tee -a $RUN/fase0.txt               # CHECK-GPU OK
gemma_campo/target/release/gemma_campo check-gpu --n 2048 --d 64 --ctx 4 | tee -a $RUN/fase0.txt
(cd campo_turno && $PY -m unittest discover -s tests -v) 2>&1 | tee -a $RUN/fase0.txt   # 7 tests OK
$PY gemma_campo/python/fetch_jsonl.py --out $W/all.jsonl
head -60 $W/all.jsonl > $W/smoke.jsonl
PY=$PY bash campo_turno/smoke.sh $W/smoke $W/smoke.jsonl gpu 2>&1 | tail -40 | tee $RUN/smoke.txt
```
`sha256sum $W/all.jsonl > $RUN/data_sha.txt`.

## 2. Extraer estados por turno (LLM congelado)
```bash
cd campo_turno
$PY turno_extract.py --model $MODEL --layer $LAYER --pool mean --jsonl $W/all.jsonl --n-conv 2000 \
    --max-turn-tokens 128 --dtype $DT --device $DEV --out $W/h 2>&1 | tee $RUN/extract_log.txt
```
Esperado: ~1600/200/200 conversaciones. (Opción preregistrada secundaria: `--pool last` en `$W/h_last`, informar aparte.)

## 3. Codificador (train) → z, d=64
```bash
$PY turno_encoder.py --hdir $W/h --out $W/z --d-field 64 | tee $RUN/encoder_log.txt
cp $W/z/encoder_info.json $RUN/; sha256sum $W/z/encoder.npz > $RUN/sha.txt
```

## 4. Campo (Rust, GPU) y baselines
```bash
BIN=../gemma_campo/target/release/gemma_campo
$BIN train --data $W/z --out $W/m --device gpu --ctx 4 --quad 1 --epochs 20 --batch 256 --lr 0.01 --threads 8
$BIN predict --data $W/z --model $W/m --split val  --kmax 1 --stride 1
$BIN predict --data $W/z --model $W/m --split test --kmax 1 --stride 1   # solo escribe predicciones; NO mirar métricas aún
cp $W/m/train_log.txt $RUN/field_train_log.txt; cp $W/m/model.json $RUN/; sha256sum $W/m/model.bin >> $RUN/sha.txt
$PY turno_baselines.py --zdir $W/z --out $W/b --ctx 4 | tee $RUN/baselines_log.txt
```
Si `train` con GPU falla (código 3) usar `--device cpu` y anotarlo.

## 5. Adaptador z → soft-prompt (solo adaptador; LLM congelado)
```bash
$PY turno_adapter.py --model $MODEL --dtype $DT --device $DEV --hdir $W/h --zdir $W/z --out $W/a \
    --P 4 --epochs 2 --lr 1e-3 --accum 8 --noise 0.1 --ctx-tokens 384 --loss-scale 128 2>&1 | tee $RUN/adapter_log.txt
cp $W/a/adapter_info.json $RUN/; sha256sum $W/a/adapter.pt >> $RUN/sha.txt
```
- Debe terminar con `LLM intacto=True` (si no, el script aborta: PARAR).
- `--loss-scale 128` solo con `float16` (evita underflow de gradientes). Si la pérdida sale `nan`, repetir con `--dtype float32`
  (gemma-2-2b ~10 GB en fp32 cabe en 16 GB con contexto 384) y anotarlo.
- Si una época tarda > 2 h, usar `--max-train 3000` y anotarlo (permitido: no toca test).

## 6. Validación (opcional, sin mirar test)
```bash
$PY turno_eval.py --split val --model $MODEL --dtype $DT --device $DEV --hdir $W/h --zdir $W/z --field-dir $W/m --base-dir $W/b \
    --adapter-dir $W/a --out $W/eval_val | tee $RUN/eval_val_log.txt
```

## 7. TEST (una sola vez)
```bash
$PY turno_eval.py --split test --model $MODEL --dtype $DT --device $DEV --hdir $W/h --zdir $W/z --field-dir $W/m --base-dir $W/b \
    --adapter-dir $W/a --out $W/eval --n-samples 5 --max-new 60 | tee $RUN/eval_test_log.txt
cp $W/eval/eval_test.json $RUN/
```
## 8. Informe
Copiar `RESULTADOS_PLANTILLA.md` a `$RUN/RESULTADOS.md` y rellenar SOLO con números de los archivos de `$RUN`.
`git add docs/campo_estado_turno/run_*` (no añadir `$W`), commit `run <fecha>: resultados campo estado por turno`, `git push origin exp/campo-estado-turno` (sin `--force`).
