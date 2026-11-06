#!/usr/bin/env bash
# Supervisor de entrenamiento continuo (shards en flujo). Idempotente: re-ejecutarlo reanuda donde quedó.
# uso: RUN=/ruta/run bash gemma_campo/run_stream.sh            (variables abajo)
set -uo pipefail
cd "$(dirname "$0")"; ROOT=$PWD
RUN=${RUN:?define RUN=directorio del run}
PY=${PY:-python}
EXTRACT_ARGS=${EXTRACT_ARGS:-"--layer 13 --max-tokens 384 --dtype bfloat16"}   # añade --jsonl X o --dataset Y; --tiny-random solo smoke
SHARD_SIZE=${SHARD_SIZE:-200}; N_VAL=${N_VAL:-60}; N_TEST=${N_TEST:-60}
D_FIELD=${D_FIELD:-64}; MAX_SHARDS=${MAX_SHARDS:-1000000}
LR0=${LR0:-0.05}; LR_T=${LR_T:-20}; MOM=${MOM:-0.9}; EPOCHS=${EPOCHS:-2}; BATCH=${BATCH:-1024}
REPLAY=${REPLAY:-2}            # nº de shards anteriores mezclados (los REPLAY más recientes + shard 1 como ancla)
KEEP_H=${KEEP_H:-2}            # shards con *_h.npy conservados (rotación de disco); z se conserva siempre
DEVICE=${DEVICE:-gpu}; THREADS=${THREADS:-6}; MAX_RETRY=${MAX_RETRY:-3}
BIN=$ROOT/target/release/gemma_campo
mkdir -p $RUN/shards $RUN/ckpt $RUN/runs
LOG=$RUN/stream_log.jsonl
jlog() { echo "$1" >> $LOG; echo "$1"; }
[ -x $BIN ] || cargo build --release || exit 1
i=$(cat $RUN/next_shard 2>/dev/null || echo 0)
while [ $i -lt $MAX_SHARDS ]; do
  S=$RUN/shards/$i
  if [ ! -f $S/h/.done ]; then
    (cd python && $PY extract.py --stream --shard $i --shard-size $SHARD_SIZE --n-val $N_VAL --n-test $N_TEST $EXTRACT_ARGS --out $S/h)
    rc=$?
    if [ $rc -ne 0 ]; then
      jlog "{\"evento\":\"extract_fallo_o_flujo_agotado\",\"shard\":$i,\"rc\":$rc}"; break
    fi
    touch $S/h/.done
  fi
  if [ ! -f $S/z/.done ]; then
    if [ $i -eq 0 ]; then (cd python && $PY encoder.py --hdir $S/h --out $S/z --d-field $D_FIELD) || exit 1
    else (cd python && $PY encoder.py --hdir $S/h --out $S/z --frozen $RUN/shards/0/z) || exit 1; fi
    touch $S/z/.done
  fi
  metas=$(for k in $(seq 0 $i); do echo $RUN/shards/$k/h/extract_meta.json; done)
  $PY python/check_splits.py $RUN/shards/0/h/extract_meta.json $metas || { jlog "{\"evento\":\"contaminacion\",\"shard\":$i}"; exit 2; }
  # replay: shard 1 (ancla temprana) + REPLAY más recientes
  rp=""; for k in $( (echo 1; seq $((i-REPLAY)) $((i-1))) | sort -un ); do [ $k -ge 0 ] && [ $k -lt $i ] && rp="$rp$RUN/shards/$k/z,"; done
  lr=$(awk -v a=$LR0 -v i=$i -v t=$LR_T 'BEGIN{printf "%.6g", a/sqrt(1+i/t)}')
  init=""; [ -f $RUN/ckpt/last_good/model.bin ] && init="--init-model $RUN/ckpt/last_good"
  ok=0
  for try in $(seq 0 $MAX_RETRY); do
    O=$RUN/runs/$i.try$try; rm -rf $O
    $BIN train --data $S/z --val-data $RUN/shards/0/z --out $O --device $DEVICE --epochs $EPOCHS --batch $BATCH \
         --lr $lr --mom $MOM --threads $THREADS --ridge-max ${RIDGE_MAX:-50000} --seed $((12345+i)) $init ${rp:+--replay ${rp%,}} > $O.stdout 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then ok=1; break; fi
    jlog "{\"evento\":\"divergencia_o_error\",\"shard\":$i,\"try\":$try,\"rc\":$rc,\"lr\":$lr}"
    lr=$(awk -v a=$lr 'BEGIN{printf "%.6g", a/2}')   # rollback implícito: se reintenta desde last_good con lr/2
  done
  [ $ok -eq 1 ] || { jlog "{\"evento\":\"abortado_sin_recuperar\",\"shard\":$i}"; exit 4; }
  mkdir -p $RUN/ckpt/shard_$i && cp $O/model.bin $O/model.json $RUN/ckpt/shard_$i/
  ln -sfn $RUN/ckpt/shard_$i $RUN/ckpt/last_good
  # olvido: z-MSE en train del shard 1 (temprano) y val held-out con el modelo actual
  fg=""; if [ $i -ge 1 ]; then $BIN predict --data $RUN/shards/1/z --model $RUN/ckpt/shard_$i --split train --kmax 1 --threads $THREADS > $O/forget.txt 2>&1 && fg=$(grep -o "campo=[0-9.]*" $O/forget.txt | cut -d= -f2); fi
  val=$(grep -o "best val dz-mse SGD=[0-9.]*" $O/train_log.txt | cut -d= -f2)
  vfz=$(grep -o "val z-mse=[0-9.]*" $O/train_log.txt | tail -1 | cut -d= -f2)
  ntr=$(grep -o "train [0-9]* conv/[0-9]* tok" $O/train_log.txt | head -1)
  jlog "{\"evento\":\"shard_ok\",\"shard\":$i,\"try\":$try,\"lr\":$lr,\"val_dz_mse_lenta\":${val:-null},\"val_z_mse_campo\":${vfz:-null},\"olvido_shard1_z_mse\":${fg:-null},\"datos\":\"$ntr\",\"t\":\"$(date -Iseconds)\"}"
  # rotación de disco: borra h de shards viejos (nunca el 0: contiene val/test)
  old=$((i-KEEP_H)); [ $old -ge 1 ] && rm -f $RUN/shards/$old/h/*_h.npy
  i=$((i+1)); echo $i > $RUN/next_shard
done
jlog "{\"evento\":\"fin\",\"next_shard\":$i}"
