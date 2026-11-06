# gemma_campo_chat: Gemma 2 2B congelado → campo predice estados futuros → Gemma decodifica

Preregistro, hipótesis y criterios: [PLAN.md](PLAN.md). Entrenamiento continuo con Gemma real en GPU: [RUNBOOK_ENTRENAMIENTO_INFINITO.md](RUNBOOK_ENTRENAMIENTO_INFINITO.md) (plantilla: [RESULTADOS_PLANTILLA.md](RESULTADOS_PLANTILLA.md)). Código: `gemma_campo/` (crate Rust, wgpu/WGSL) y `gemma_campo/python/`.

## Requisitos (PC de Bruster, GPU AMD)
* Rust estable y drivers Vulkan (Linux: mesa RADV; Windows: driver Adrenalin). wgpu elige el adaptador "AMD/Radeon".
* Python 3.10+ con `torch`, `transformers>=4.44`, `numpy` y `datasets` (solo si no usas `--jsonl`). Gemma corre en torch (CPU, ROCm o CUDA). La GPU AMD vía wgpu solo entrena el campo.
* Acceso a `google/gemma-2-2b`: acepta la licencia en Hugging Face y exporta tu token. **No se guarda en ningún archivo.**
  ```bash
  export HF_TOKEN=hf_xxx          # PowerShell: $env:HF_TOKEN="hf_xxx"
  ```

## Comandos exactos
```bash
git fetch origin && git checkout exp/gemma-campo-chat
cd gemma_campo
cargo build --release
cargo test --release                       # 7 tests, incluido gpu_igual_a_cpu (en tu AMD)
./target/release/gemma_campo validate-wgsl
./target/release/gemma_campo check-gpu     # debe decir "adaptador: AMD Radeon ..." y "CHECK-GPU OK"

cd python
D=$HOME/gc_run
# 1) estados ocultos capa 13 (600 conv, 480/60/60 por conversación)
python extract.py --layer 13 --n-conv 600 --max-tokens 384 --out $D/h          # añade --device cuda/--dtype bfloat16 si tienes ROCm
# 2) codificador PCA 2304->64 + decodificador lineal (solo train)
python encoder.py --hdir $D/h --out $D/z --d-field 64
# 3) entrenamiento del campo en la GPU AMD (ridge M7 en val + SGD WGSL + rápida en val)
../target/release/gemma_campo train --data $D/z --out $D/m --device gpu --ctx 4 --quad 1 --epochs 5 --batch 1024 --lr 0.05 --mom 0.9 --threads 6
# 4) predicciones y z-MSE en TEST (una sola vez)
../target/release/gemma_campo predict --data $D/z --model $D/m --split test --kmax 8 --stride 4
python parity.py --zdir $D/z --model $D/m                    # numpy == Rust
# 5) evaluación: perplejidad con sustitución + controles + bootstrap 10000/12345
python eval.py --hdir $D/h --zdir $D/z --model-dir $D/m --out $D/eval
# 6) chat
python chat_demo.py --hdir $D/h --zdir $D/z --model-dir $D/m --mode campo --prompt "Hi! Any tips for a rainy weekend?"
python chat_demo.py --hdir $D/h --zdir $D/z --model-dir $D/m --mode gemma --prompt "Hi! Any tips for a rainy weekend?"
```
Salidas:
* `$D/m/train_log.txt`: la rejilla ridge (train y val), la pérdida train/val por época y la rejilla de la parte rápida.
* `$D/m/ckpt_epoch*.bin` y `model.bin/json`: los checkpoints y el modelo elegido.
* `$D/eval/eval_test.json`: los resultados de test.

Antes del paso 4, mira `train_log.txt` y `encoder_info.json` y aplica los criterios de parada de PLAN.md (sobre val).

**Si `lr 0.05` diverge** ("pérdida no finita"), baja a `--lr 0.01`. Elígelo solo mirando val.

## Formatos
* `<split>_h.npy`: f16 [N, 2304].
* `<split>_tokens.json`: lista de `{conv, ids}`.
* `<split>_z.f32` + `<split>_z.json` (`{d, lengths}`): z blanqueado.
* `model.bin`: f32 W[F×d], con F = 1 + C·d + d(d+1)/2.
* `test_pred1.f32`: la fila t es la predicción causal de z_t. La fila 0 es z_0 y se excluye.

## Qué se verificó aquí (sin GPU AMD ni pesos Gemma) — SMOKE, no resultados
* `cargo build`, `cargo test` (6 unitarios y 1 GPU) y `validate-wgsl` (naga) pasan.
* `check-gpu` y `gpu_igual_a_cpu` se ejecutaron **de verdad** en wgpu sobre llvmpipe (backend GL de Mesa, en CPU). La pérdida GPU coincide con la CPU con un error relativo ≤3e-8 y |ΔW| ≤ 7.5e-9. Además es determinista.
* El entrenamiento con `--device gpu` y con `--device cpu` produce la misma curva de pérdida (6 cifras).
* Tubería completa (`smoke.sh`) con un Gemma2 **diminuto aleatorio** (4 capas, H=64, tokenizador de bytes) y 60 conversaciones reales del dataset por defecto. Log en `smoke/`. Paridad numpy==Rust con un error de 2.8e-8. La sustitución identidad por el h guardado en f16 da un |Δlogits| máximo de 1.8e-4.
* Las cifras de `smoke/eval_test_SMOKE.json` vienen de un modelo aleatorio y **no significan nada** sobre Gemma.
