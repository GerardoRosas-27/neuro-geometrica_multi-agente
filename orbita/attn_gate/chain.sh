#!/usr/bin/env bash
# Cadena única reanudable (.done) y con flock: DEV (+TUNE para τ) → verificación de convergencia → CONFIRM. docs/prereg_orbita_attn_gate.md
# Lanzar: setsid nohup orbita/attn_gate/chain.sh >> /workspace/orbita-data/attn-gate/chain.log 2>&1 &
set -u
R=/workspace/ngm-attn-gate; AG=$R/orbita/attn_gate; B=$R/orbita/target/release/orbita; P=/workspace/attn-venv/bin/python
W=/workspace/orbita-data/attn-gate; A=$R/artifacts/orbita/attn_gate; J=${J:-6}; STEPS=${STEPS:-40000}
mkdir -p $W $A; exec 9>$W/chain.lock; flock -n 9 || { echo "otra cadena activa; salgo"; exit 1; }
export R AG B P W A STEPS ORBITA_BIN=$B
job() {  # $1 fase $2 seed $3 tarea
  ph=$1; s=$2; t=$3; o=$W/$ph; mkdir -p $o/AG $o/LAT $o/train
  [ -f $o/$t.$s.done ] && return 0
  wt=$R/artifacts/orbita/$ph/weights_$s.json
  case $t in
    AG)
      [ -f $o/train/$s/traj.npy ] || $P $AG/gen_eps.py train $s $o/train/$s > /dev/null || return 1
      [ -f $o/AG/c_$s.pt ] || { $P -W ignore $AG/train_corr.py --orbita $wt --data $o/train/$s --out $o/AG/c_$s.pt --seed $s --steps $STEPS >> $o/AG/train_$s.log 2>&1 || return 1; }
      $P -W ignore $AG/eval_ag.py $wt $o/AG/c_$s.pt $W/$ph/eps $o/AG/eval_$s.csv > $o/AG/eval_$s.log 2>&1 || return 1
      if [ $ph = dev ]; then $P -W ignore $AG/eval_ag.py $wt $o/AG/c_$s.pt $W/tune/eps $o/AG/tune_$s.csv > $o/AG/tune_$s.log 2>&1 || return 1; fi ;;
    LAT)
      for k in 2 4 8; do tau=$(case $k in 2) echo 0.014656;; 4) echo 0.022020;; 8) echo 0.037142;; esac)
        [ -s $o/LAT/lat_${s}_k$k.csv ] && continue
        $B lat-eval exp1 --weights $wt --set $W/$ph/eps/set.csv --ctx $W/$ph/eps/ctx_none.csv --k $k --tau $tau --lam0 1e-4 --lamm 1e-3 > $o/LAT/lat_${s}_k$k.csv.tmp 2> $o/LAT/lat_${s}_k$k.log || return 1
        mv $o/LAT/lat_${s}_k$k.csv.tmp $o/LAT/lat_${s}_k$k.csv; done ;;
  esac
  touch $o/$t.$s.done; echo "job ok $ph $s $t $(date '+%F %T %Z')"
}
export -f job
echo "chain start $(date '+%F %T %Z')"
for ph in tune dev confirm; do [ -f $W/$ph/eps.done ] || { GEN_THREADS=4 $P $AG/gen_eps.py $ph $W/$ph/eps && touch $W/$ph/eps.done; }; done
for ph in dev confirm; do
  if [ $ph = dev ]; then pre=0xA4; else pre=0xB4; fi
  if [ $ph = confirm ]; then
    [ -f $W/tau.json ] || { echo "falta tau.json"; exit 1; }
    [ -f $W/dev/converged ] || { echo "DEV no convergió: corregir la optimización antes de CONFIRM (prerregistro §5)"; exit 2; }
  fi
  for t in AG LAT; do for i in $(seq 0 15); do printf "$ph ${pre}%02X $t\n" $i; done; done | xargs -P $J -L 1 bash -c 'job $0 $1 $2'
  n=$(ls $W/$ph/AG.*.done $W/$ph/LAT.*.done 2>/dev/null | wc -l)
  [ $n -eq 32 ] || { echo "fase $ph incompleta ($n/32); relanzar la cadena"; exit 1; }
  [ $ph = dev ] && { [ -f $W/tau.json ] || $P $AG/calibrate.py $W || exit 1; cp $W/tau.json $A/tau.json; }
  [ -f $W/$ph/analyze.done ] || { rm -f $W/$ph/converged $W/$ph/not_converged; $P $AG/analyze_ag.py $ph $A > $A/analyze_$ph.md && touch $W/$ph/analyze.done && echo "analyze $ph done $(date '+%F %T %Z')"; } || exit 1
  for s in $(seq 0 15); do sd=$(printf "${pre}%02X" $s); cp $W/$ph/AG/c_$sd.pt.curve.csv $A/curve_${ph}_$sd.csv; done
done
echo "chain done $(date '+%F %T %Z')"
