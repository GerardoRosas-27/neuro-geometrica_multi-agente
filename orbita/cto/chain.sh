#!/usr/bin/env bash
# Cadena reanudable (setsid nohup): calib τ -> DEV (train 4 brazos x 16 seeds, flujo+eval) -> CONFIRM -> análisis. Marcadores .done / salidas atómicas.
set -u
cd "$(dirname "$0")"; PY=/workspace/orbita-venv/bin/python; O=${CTO_OUT:-/workspace/orbita-data/cto}; J=${NPAR:-4}
ART=../../artifacts/orbita/cognitive_transformer; mkdir -p $O $ART
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
log(){ echo "$(date '+%F %T %Z') $*"; }
[ -f $O/calib.done ] || { $PY cto.py calib > $O/calib.log 2>&1 && touch $O/calib.done && log "calib done"; }
for ph in dev confirm; do
  if [ $ph = dev ]; then S=$(for i in $(seq 0 15); do printf "0xA4%02X " $i; done); else S=$(for i in $(seq 0 15); do printf "0xB4%02X " $i; done); fi
  if [ ! -f $O/train_$ph.done ]; then
    log "train $ph start"
    for s in $S; do for a in CT SELF MLP NOORB; do echo "$s $a"; done; done | xargs -P $J -L1 bash -c 'nice -n 10 '$PY' cto.py train '$ph' $0 $1 >> '$O'/train_'$ph'.log 2>&1'
    n=$(ls $O/$ph/net_*.pt 2>/dev/null | wc -l); [ $n -eq 64 ] && touch $O/train_$ph.done && log "train $ph done"
  fi
  if [ ! -f $O/stream_$ph.done ]; then
    log "stream $ph start"
    echo $S | tr ' ' '\n' | grep . | xargs -P $J -I{} bash -c 'nice -n 10 '$PY' cto.py stream '$ph' {} >> '$O'/stream_'$ph'.log 2>&1'
    n=$(ls $O/$ph/eval_*.csv 2>/dev/null | wc -l); [ $n -eq 16 ] && touch $O/stream_$ph.done && log "stream $ph done"
  fi
  [ -f $O/stream_$ph.done ] && $PY analyze_cto.py $ph > $ART/summary_$ph.md 2> $O/analyze_$ph.err && log "analyze $ph done"
done
cp $O/tau.json $O/calib.log $ART/ 2>/dev/null; log "chain end"
