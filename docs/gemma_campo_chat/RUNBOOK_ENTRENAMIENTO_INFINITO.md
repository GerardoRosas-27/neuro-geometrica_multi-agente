# RUNBOOK: entrenamiento continuo ("infinito") con Gemma 2 2B real, todo en GPU

Este runbook es para el agente que ejecute el experimento en el PC de Bruster (GPU AMD). La cadena es:
Gemma 2 2B congelado (PyTorch ROCm) → estados de la capa 13 → codificador PCA congelado → **campo** (wgpu/Vulkan) → decodificador lineal → capas 14–26 de Gemma y LM head.

Base: rama `exp/gemma-campo-chat`. Preregistro original: [PLAN.md](PLAN.md). Instrucciones de una sola pasada: [README.md](README.md). Plantilla de resultados: [RESULTADOS_PLANTILLA.md](RESULTADOS_PLANTILLA.md).

> **Este documento NO contiene resultados.** Todo lo que se ejecutó al escribirlo fue un SMOKE con un Gemma2 diminuto aleatorio (ver §8).

---
## 0. Reglas para el agente ejecutor (obligatorias)
1. Crea **tu propia rama** desde esta: `git checkout -b exp/gemma-campo-infinito origin/exp/gemma-campo-chat`.
   - No toques `main`, `run_v6` ni `exp/bq10-gpu-amd`.
   - **Nunca** hagas `push --force`.
   - Antes de cada commit, haz `git pull --no-rebase`. Si el remoto se movió, haz merge, no rebase.
2. Reporta **solo números reales** salidos de los logs y JSON de este pipeline. Los negativos se informan igual que los positivos.
   - Prohibido inyectar respuestas, prohibidos valores de retorno fijos y prohibido reajustar hiperparámetros mirando **test**.
3. `HF_TOKEN` se lee **solo** del entorno. Nunca se escribe en archivos, logs ni commits. Antes de cada commit ejecuta `git grep -nE "hf_[A-Za-z0-9]{20,}"`; tiene que salir vacío.
4. Antes de arrancar, congela este runbook y `PLAN.md`: haz commit y anota el hash en `RESULTADOS.md`. Las secciones §5 y §6 no se cambian después de ver datos de Gemma.
5. Qué se sube al repo:
   - Sí: logs (`stream_log.jsonl`, `train_log.txt` de los shards, `eval_test.json`) y `RESULTADOS.md`, en `docs/gemma_campo_chat/run_<fecha>/`.
   - No: pesos grandes ni `*_h.npy`, que están en `.gitignore` o se dejan fuera. El hash SHA-256 de `encoder.npz` y del checkpoint final sí va al informe.

---
## 1. Prerrequisitos y entorno
| Componente | Requisito | Comprobación |
|---|---|---|
| GPU | AMD (p.ej. RX 7800 XT, 16 GB) | `rocm-smi`; `vulkaninfo --summary` debe listar la GPU AMD (RADV o AMDVLK) |
| Vulkan (campo) | Driver Mesa RADV (Linux) o Adrenalin (Windows) | `gemma_campo check-gpu` debe imprimir `adaptador: AMD Radeon ... (Vulkan)` |
| ROCm + PyTorch (Gemma) | ROCm ≥ 6.x soportado por tu GPU; torch del índice ROCm | `python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` debe dar `True` y el nombre AMD |
| Rust | estable (probado: rustc 1.99.1) | `cargo --version` |
| Python | 3.10+; `pip install -r gemma_campo/python/requirements-rocm.txt` (versiones fijadas) | `pip freeze > $RUN/env_freeze.txt` |
| Gemma | licencia aceptada en huggingface.co/google/gemma-2-2b; `export HF_TOKEN=...` | `python -c "import os;assert os.environ.get('HF_TOKEN')"` |

En ROCm, PyTorch expone la GPU AMD como `cuda`, así que `--device cuda` es correcto.

**Presupuestos**
* **VRAM:**
  - Gemma 2 2B en bf16: unos 5,2 GB de pesos, más las activaciones (≤384 tokens): < 8 GB.
  - Campo con d=64: W y V suman unos 1,2 MB y un batch de φ (1024×2337 f32) ocupa unos 9,6 MB. Es despreciable.
  - Las dos cargas caben a la vez en 16 GB, pero corren en procesos distintos (extracción y entrenamiento se alternan).
* **Disco** (estimado, no medido):
  - Caché HF de Gemma: unos 10 GB.
  - Estados ocultos: 2304 × 2 B = 4,6 KB por token. Un shard de 200 conv × unos 300 tok son 60k tok, es decir, unos 280 MB por shard en `*_h.npy`.
  - z (64 f32): unos 15 MB por shard.
  - Con `KEEP_H=2` se rota `h` y z se conserva siempre. 1000 shards ocupan unos 15 GB de z más unos 0,6 GB de h.
  - Checkpoints: unos 0,6 MB cada uno.
* **Semillas:**
  - Permutación de datos y bootstrap: 12345.
  - SGD del shard i: 12345+i.
  - `chat_demo`: `--seed 12345`.

---
## 2. Fase 0: puertas (si alguna falla, se ABORTA y se reporta; no se sigue)
```bash
cd gemma_campo
cargo build --release
cargo test --release                         # 7 tests: 6 unitarios + gpu_igual_a_cpu
./target/release/gemma_campo validate-wgsl   # "WGSL OK (naga)"
./target/release/gemma_campo check-gpu       # "adaptador: AMD..." + "CHECK-GPU OK" (rel ≤1e-4, |ΔW|≤1e-4, determinista=true)
./target/release/gemma_campo check-gpu --n 2048 --d 64 --ctx 4 --steps 3   # a escala real de d (F=2337)
python python/fetch_jsonl.py --max-rows 60 --out /tmp/smoke.jsonl
PY=python bash smoke.sh /tmp/gc_smoke /tmp/smoke.jsonl gpu        # termina en "[SMOKE ...] [campo] Assistant:..." y "PARIDAD OK"
```
Reglas de aborto de la Fase 0:
* `check-gpu` dice `FALLA`, o el adaptador no es AMD/Vulkan → no se entrena en GPU. Se reporta el adaptador y el error. Usar `--device cpu` es una vía lenta y válida, pero hay que declararla en el informe.
* `parity.py` falla → el gemelo numpy y Rust no coinciden, así que el chat no es fiable. Se para.
* Hay que pasar una **prueba de Gemma real** antes de la Fase 1:
  - `extract.py --stream --shard 0` sobre 5 conversaciones.
  - `eval.py` sobre esas 5, con `sanity_identity_maxdiff` < 0,05. Son logits de Gemma cuando la capa 13 se sustituye por su propio h guardado en f16. Si sale mayor, el hook o el dtype están mal y se para.

---
## 3. Flujo de datos para entrenamiento continuo
* **Fuentes:**
  - `fetch_jsonl.py` descarga cualquier dataset de diálogos de HF con un campo `messages` (role/content).
  - Por defecto: `HuggingFaceTB/everyday-conversations-llama3.1-2k`, unas 2,2k conversaciones.
  - Para "infinito" se concatenan más fuentes públicas en un solo JSONL (`cat a.jsonl b.jsonl > all.jsonl`) **antes** de arrancar. Cada fuente se anota en `RESULTADOS.md`.
  - Si una fuente no tiene `messages`, hay que convertirla a ese formato con un script propio y commitearlo.
* **Dedup:** `extract.py` calcula el SHA-256 del texto formateado de cada conversación (16 hex) y se queda con la primera aparición de cada hash.
* **Splits fijos** (`--stream`):
  - Sobre los conversaciones únicas se hace una permutación con semilla 12345.
  - Las primeras `N_TEST` son **test** y las siguientes `N_VAL` son **val**. Están presentes solo en `shards/0`, nunca se entrenan y nunca se rotan.
  - El resto es el flujo de train. El shard i son las conversaciones `[i·S, (i+1)·S)`.
  - **Importante:** el JSONL no se cambia durante el run. Añadir fuentes cambia la permutación, así que eso es un **run nuevo**.
* **Contaminación:** `check_splits.py` corre antes de cada shard. Exige val ∩ test = ∅, (val ∪ test) ∩ train_i = ∅ y que ninguna conversación se repita entre shards. Si falla, el supervisor sale con código 2.
* **Extracción en GPU:** `--dtype bfloat16 --device cuda`, capa 13. Se guarda en f16 (`*_h.npy`).
* **Codificador CONGELADO:**
  - La PCA se ajusta **una sola vez** sobre el train del shard 0 (`encoder.py`). Los shards siguientes usan `encoder.py --frozen shards/0/z` y solo transforman, sin reajustar.
  - El SHA-256 de `encoder.npz` se registra en `encoder_info.json` de cada shard. Tiene que ser el mismo en todos.
  - **Re-base** (prohibido dentro de un run). Si `train_recon_rel_mse_centrado` del shard 0 es > 0,5, o el de shards nuevos supera al del shard 0 en más de un 20 %, se considera deriva. Entonces se para y se abre un run nuevo con d_field mayor o más datos en el shard 0. Un run nuevo implica un checkpoint nuevo desde cero (W vive en la base anterior) y una entrada en `RESULTADOS.md`. No se mezclan métricas de bases distintas.

---
## 4. Bucle de entrenamiento continuo (`gemma_campo/run_stream.sh`)
Por cada shard i, el supervisor hace lo siguiente. Es idempotente: si se relanza, reanuda en `next_shard`.
1. Extrae el shard (si falta `h/.done`). Cuando el flujo se agota, `extract.py` sale con "flujo agotado" y el supervisor registra `fin`.
2. Codifica: PCA en el shard 0 y `--frozen` en el resto. Marca `z/.done`.
3. Ejecuta el chequeo de contaminación.
4. **Entrena en GPU** con `gemma_campo train`:
   - `--init-model ckpt/last_good` para continuar. Así no hay rejilla ridge y se conservan cfg y λ del shard 0.
   - `--val-data shards/0/z`, que es el val fijo.
   - `--replay`: shard 1 como ancla temprana más los `REPLAY` shards más recientes, contra el olvido.
   - `--epochs EPOCHS` y semilla 12345+i.
   - Se guarda un checkpoint por época (`runs/i.tryN/ckpt_epoch*.bin`) y el mejor según val (`model.bin`).
   - η y γ del campo rápido se reeligen en val en cada shard. Es una selección repetida sobre val, así que val queda algo optimista. **Test no se toca.**
5. **LR:** `lr_i = LR0 / sqrt(1 + i/LR_T)`, con LR0=0,05 y LR_T=20 por defecto.
6. **Divergencia:** si el entrenador encuentra una pérdida no finita, sale con código 4. Entonces el supervisor registra `divergencia_o_error` y reintenta desde `last_good` con lr/2, hasta `MAX_RETRY` veces. Si sigue fallando, registra `abortado_sin_recuperar` y sale con 4: es una **divergencia no recuperada**.
   - Comprobado en el smoke con LR0=1e7: hubo 3 reintentos y luego abortó.
7. Si sale bien: `ckpt/shard_i/` pasa a ser `ckpt/last_good`. Se mide el **olvido** como el z-MSE 1 paso del campo en el train del shard 1 con el modelo actual (`predict --split train`).
8. Añade una línea JSONL a `stream_log.jsonl` con:
   - `shard`, `try` y `lr`
   - `val_dz_mse_lenta` (pérdida val real de la parte lenta) y `val_z_mse_campo` (lenta + rápida, val)
   - `olvido_shard1_z_mse`, `datos` y `t`
   - Además, `runs/i.tryN/train_log.txt` contiene la pérdida train y val real **por época**.
9. Rota el disco: borra `*_h.npy` de los shards más viejos que `KEEP_H`. El shard 0 no se borra nunca.

**Supervisor ante caídas** (systemd, tmux o bucle). Re-ejecutar el script es seguro:
```bash
while true; do RUN=$RUN bash gemma_campo/run_stream.sh; rc=$?; [ $rc -eq 0 ] && break; [ $rc -eq 2 -o $rc -eq 4 ] && break; sleep 60; done
```
Con códigos 2 (contaminación) o 4 (divergencia no recuperada) **no** se reintenta; hay que intervenir.

**Chequeo cada hora:** `RUN=$RUN bash gemma_campo/health.sh` (por ejemplo `0 * * * *` en cron, con la salida añadida a `$RUN/health.log`). Muestra:
* los últimos eventos y la edad del log (alerta si es mayor que 3 veces la duración típica de un shard)
* el número de divergencias y si hubo aborto
* el disco, `rocm-smi` y si el supervisor sigue vivo

**Perillas de escala** (lo que se cambia y lo que cuesta):
| Perilla | Cómo | Nota |
|---|---|---|
| Tamaño de shard / épocas / batch | `SHARD_SIZE`, `EPOCHS`, `BATCH` | `BATCH` ≤ ~8192 por el límite de buffer de φ (B·F·4 bytes) |
| Replay | `REPLAY` | más replay = menos olvido y más tiempo por shard (el train crece lineal) |
| d_field 64→96 | `D_FIELD=96` en run nuevo | F=1+384+4656=5041. La matriz de Gram es F² f64, unos 200 MB, aceptable. |
| d_field 128–256 | **hace falta cambiar código** | Con `--quad 1`, F² hace inviable la Gram (d=256 da unos 9 GB). Opciones: (a) `--quad 0` (F=1+C·d); (b) términos cuadráticos solo de las primeras k componentes (por implementar en `features.rs`); (c) quitar la Gram en modo `--init-model` y calcular val con forward directo. El crate hoy la calcula aunque no haya rejilla. Coste: `--ridge-max` muestras de train. |
| Más capas | por implementar | `extract.py` guarda una sola capa. Para varias: concatenar `hidden_states[l]` en H. El decodificador sigue apuntando a la capa L de sustitución. |
| Multi-GPU | no soportado | haría falta paralelismo de datos con all-reduce de gradientes. El kernel actual es por dispositivo. Alternativa simple: varios runs independientes (semillas o hiperparámetros) en GPUs distintas. |
| Velocidad | `THREADS`, `RIDGE_MAX` | φ se construye en CPU. La Gram de val/train corre en CPU en cada shard. |

---
## 5. Protocolo de evaluación final (PRERREGISTRADO; congelar antes del run)
Se usa el **último checkpoint bueno**, una sola vez, sobre **test** (shard 0):
```bash
F=$RUN/final; mkdir -p $F && cp $RUN/ckpt/last_good/model.* $F/
gemma_campo/target/release/gemma_campo predict --data $RUN/shards/0/z --model $F --split test --kmax 8 --stride 4
python gemma_campo/python/parity.py --zdir $RUN/shards/0/z --model $F
python gemma_campo/python/eval.py --hdir $RUN/shards/0/h --zdir $RUN/shards/0/z --model-dir $F --device cuda --out $F/eval
```
**Métricas:**
* **z-MSE** a 1 paso y a k=1..8 pasos. Se compara campo frente a repeat-last y campo frente a lenta.
* **ΔNLL/token** (perplejidad) del campo frente a: Gemma sola, repeat-last D(E(h_{t−1})) y oracle D(E(h_t)), que es el cuello PCA. También oracle−gemma.
* **Curva de olvido:** `olvido_shard1_z_mse` en función del shard. Además, al final, `predict --split train` sobre shards 1, 2 y 3 con el checkpoint final, comparado con el checkpoint tomado justo después de cada uno.
* **Muestras de generación:** `chat_demo.py` en modo `campo` y en modo `gemma` con 5 prompts fijos (§7), `--temperature 0`.
* **Estadística:** bootstrap pareado por conversación, 10000 remuestreos, semilla 12345, IC95 %. Lo implementa `eval.py`.

**Criterios de ÉXITO.** Todos deben cumplirse para declarar "éxito":
| ID | Criterio | Fuente |
|---|---|---|
| H1 | z-MSE 1 paso, campo − repeat < 0 con el IC95 entero < 0 | `eval_test.json: zmse_1paso.campo-repeat.ic95[1] < 0` |
| H1k | lo mismo para k=2, 4 y 8 | `zmse_k[k].campo-repeat.ic95[1] < 0` |
| H2 | ΔNLL campo − oracle ≤ **0,05 nat/token** (el extremo superior del IC95 ≤ 0,05) | `delta_nll.campo-oracle.ic95[1] ≤ 0.05` |
| H3 | ΔNLL campo − repeat < 0 (IC95 entero < 0) | `delta_nll.campo-repeat` |
| OLV | sin olvido: el z-MSE final del shard 1 es ≤ 1,05 × el mínimo de `olvido_shard1_z_mse` en todo el run | `stream_log.jsonl` + predict final |
| EST | estabilidad: cero eventos `abortado_sin_recuperar` | `stream_log.jsonl` |
| REP | reproducibilidad (ver abajo) | ver abajo |

**Reproducibilidad (REP):**
* Ejecutar dos veces `predict` y `eval.py` sobre `$F` tiene que dar el mismo `test_zmse.json`, byte a byte (Rust en CPU es determinista). En `eval.py`, las NLL tienen que coincidir dentro de 1e-4 (GPU ROCm).
* Además, reentrena el último shard desde el checkpoint anterior con la misma semilla y el mismo lr. Las pérdidas por época de `train_log.txt` tienen que coincidir dentro de una diferencia relativa de 1e-6 (las reducciones de la GPU son deterministas).

**FALLO.** Basta con uno de estos para declarar "fallo" en esa hipótesis:
* el IC95 de H1 está entero por encima de 0 (el campo es peor que repeat)
* una divergencia no recuperada
* contaminación detectada
* REP no se cumple

**INCONCLUSO:** el IC95 cruza 0 (H1 o H3), o se cumple H1 pero no hay datos suficientes para H2 (n_conv test < 30). Se informa como inconcluso, sin redondear a éxito.

**Qué puede y qué NO puede afirmar el experimento**
* **Puede:** que un sustrato de campo (ridge, SGD y NLMS) predice la trayectoria de estados de la capa 13 en z mejor o peor que repetir el último estado; cuánta perplejidad cuesta sustituir los estados de Gemma por los del campo; si el entrenamiento continuo es estable y sin olvido.
* **No puede:** decir que el campo "añade capacidad de lenguaje".
  - Las capas 1–13, las 14–26 y el LM head son de Gemma congelado.
  - El campo predice h_t **sin ver el token t**, así que se espera ΔNLL campo−gemma > 0. **Lo esperable es que H2 falle**, y si falla se informa como negativo.
  - Una mejora en z-MSE no implica mejor texto.
  - Tampoco cabe comparar con orbita, BQ10, v4 o ATLAS: no es este protocolo.

---
## 6. Informe y commits
Copia [RESULTADOS_PLANTILLA.md](RESULTADOS_PLANTILLA.md) a `docs/gemma_campo_chat/run_<fecha>/RESULTADOS.md`. Rellénalo **solo** con valores copiados de `eval_test.json`, `stream_log.jsonl` y `train_log.txt`, e incluye las rutas.

Haz commit de logs y JSON (no de `*_h.npy` ni de pesos > 10 MB) en tu rama:
```bash
git pull --no-rebase && git add docs/gemma_campo_chat/run_<fecha> && git commit -m "run <fecha>: ..." && git push
```
Nada de `push --force` ni de tocar `main`.

---
## 7. Secuencia exacta de extremo a extremo
```bash
# --- entorno ---
git fetch origin && git checkout -b exp/gemma-campo-infinito origin/exp/gemma-campo-chat
export HF_TOKEN=...                                   # nunca en archivos
export RUN=$HOME/gc_infinito_$(date +%Y%m%d); mkdir -p $RUN
pip install -r gemma_campo/python/requirements-rocm.txt   # + torch ROCm (ver archivo)
pip freeze > $RUN/env_freeze.txt; git rev-parse HEAD > $RUN/commit.txt
# --- Fase 0 (§2) ---
(cd gemma_campo && cargo build --release && cargo test --release && ./target/release/gemma_campo check-gpu && ./target/release/gemma_campo check-gpu --n 2048 --d 64 --ctx 4 --steps 3) | tee $RUN/fase0.txt
# --- datos ---
python gemma_campo/python/fetch_jsonl.py --out $RUN/all.jsonl          # + otras fuentes concatenadas, ANTES de empezar
sha256sum $RUN/all.jsonl > $RUN/data_sha.txt
# --- prueba Gemma real en 5 conv (sanity hook) ---
head -n 20 $RUN/all.jsonl > /tmp/g5.jsonl
(cd gemma_campo/python && python extract.py --stream --shard 0 --shard-size 10 --n-val 5 --n-test 5 --jsonl /tmp/g5.jsonl --layer 13 --dtype bfloat16 --device cuda --out /tmp/g5/h \
 && python encoder.py --hdir /tmp/g5/h --out /tmp/g5/z --d-field 8 \
 && ../target/release/gemma_campo train --data /tmp/g5/z --out /tmp/g5/m --device gpu --epochs 1 --batch 256 --quad 0 \
 && ../target/release/gemma_campo predict --data /tmp/g5/z --model /tmp/g5/m --split test \
 && python eval.py --hdir /tmp/g5/h --zdir /tmp/g5/z --model-dir /tmp/g5/m --device cuda --out /tmp/g5/eval)   # revisar sanity_identity_maxdiff < 0.05
# --- run continuo (supervisor) ---
export PY=python EXTRACT_ARGS="--jsonl $RUN/all.jsonl --layer 13 --max-tokens 384 --dtype bfloat16 --device cuda"
export SHARD_SIZE=200 N_VAL=60 N_TEST=60 D_FIELD=64 LR0=0.05 LR_T=20 EPOCHS=2 BATCH=1024 REPLAY=2 KEEP_H=2 DEVICE=gpu THREADS=6
while true; do bash gemma_campo/run_stream.sh; rc=$?; [ $rc -eq 0 -o $rc -eq 2 -o $rc -eq 4 ] && break; sleep 60; done 2>&1 | tee -a $RUN/supervisor.log
# (cron horario)  0 * * * * RUN=$RUN bash /ruta/gemma_campo/health.sh >> $RUN/health.log 2>&1
# --- evaluación final (§5), UNA vez ---
F=$RUN/final; mkdir -p $F && cp $RUN/ckpt/last_good/model.* $F/ && sha256sum $F/model.bin $RUN/shards/0/z/encoder.npz > $F/sha.txt
gemma_campo/target/release/gemma_campo predict --data $RUN/shards/0/z --model $F --split test --kmax 8 --stride 4
python gemma_campo/python/parity.py --zdir $RUN/shards/0/z --model $F
python gemma_campo/python/eval.py --hdir $RUN/shards/0/h --zdir $RUN/shards/0/z --model-dir $F --device cuda --out $F/eval
for k in 1 2 3; do gemma_campo/target/release/gemma_campo predict --data $RUN/shards/$k/z --model $F --split train --kmax 1 | tee -a $F/olvido_final.txt; done
for p in "Hi! Any tips for a rainy weekend?" "Can you explain how rainbows form?" "I need help planning a birthday party." "What should I cook tonight?" "How do I start learning guitar?"; do
  for mo in campo gemma; do python gemma_campo/python/chat_demo.py --hdir $RUN/shards/0/h --zdir $RUN/shards/0/z --model-dir $F --mode $mo --device cuda --prompt "$p"; done; done | tee $F/muestras.txt
# --- REP ---
cp $F/test_zmse.json /tmp/a.json && gemma_campo/target/release/gemma_campo predict --data $RUN/shards/0/z --model $F --split test --kmax 8 --stride 4 && cmp /tmp/a.json $F/test_zmse.json && echo REP_OK
```
Al ejecutar `predict --split train` sobre un shard antiguo hace falta su `train_z.f32`, que se conserva. Escribe `train_pred1.f32` y `train_zmse.json` en `$F` (se sobrescriben en cada k): lo que vale es la línea impresa en `olvido_final.txt`.

**Para la comparación de olvido** (z-MSE de los shards 2 y 3 justo después de entrenarlos), usa `ckpt/shard_k`:
```bash
gemma_campo/target/release/gemma_campo predict --data $RUN/shards/$k/z --model $RUN/ckpt/shard_$k --split train --kmax 1
```
Cuidado: escribe dentro de `ckpt/shard_k`. Es inocuo porque son archivos de predicción.

### Checklist
- [ ] Rama propia creada; hash del runbook y PLAN congelados anotado
- [ ] Fase 0: 7 tests OK, check-gpu OK en AMD/Vulkan (dos tamaños), smoke OK, sanity del hook de Gemma real < 0,05
- [ ] `env_freeze.txt`, `commit.txt`, `data_sha.txt` guardados; HF_TOKEN fuera de todo archivo
- [ ] Shard 0: `encoder_info.json` (var. explicada, recon) revisado contra los criterios de re-base; SHA del encoder anotado
- [ ] `check_splits.py` OK en todos los shards
- [ ] Supervisor + health horario activos; divergencias registradas y resueltas (o aborto reportado)
- [ ] Evaluación final ejecutada UNA vez; paridad OK; REP OK
- [ ] RESULTADOS.md rellenado con números reales, H1/H1k/H2/H3/OLV/EST/REP marcados como éxito, fallo o inconcluso
- [ ] Commit + push sin force; sin pesos grandes ni token

---
## 8. Qué se verificó al escribir este runbook (SMOKE, no resultados)
El SMOKE usó un Gemma2 diminuto aleatorio y 60 conversaciones reales. wgpu corrió sobre llvmpipe/GL en CPU.
* `run_stream.sh` completó 4 shards (12 conversaciones cada uno, 6+6 held-out). Funcionaron el replay, la reanudación y el `fin` al agotarse el flujo.
* `check_splits.py` dio OK en cada shard.
* Con LR0=1e7 se registraron 3 reintentos con lr/2 y después `abortado_sin_recuperar`, con salida 4.
* `health.sh` funciona.
* `fetch_jsonl.py` descargó 250 conversaciones.
* No se ejecutó nada con Gemma 2 2B real ni en la GPU AMD.
