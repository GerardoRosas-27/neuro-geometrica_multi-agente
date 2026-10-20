#!/usr/bin/env bash
# Cadena independiente y reanudable (.done): DEV y luego CONFIRM; lanzar con setsid nohup. docs/prereg_orbita_transformer_fe.md
set -u
R=/workspace/ngm-orbita; TF=$R/orbita/tf; B=$R/orbita/target/release/orbita; P=/workspace/orbita-venv/bin/python
W=/workspace/orbita-data/tf; A=$R/artifacts/orbita/transformer_fe; J=${J:-7}
mkdir -p $W $A; export R TF B P W A
job() {  # $1 phase $2 seed $3 arm
  ph=$1; s=$2; arm=$3; o=$W/$ph/$arm; mkdir -p $o
  [ -f $o/$s.done ] && return 0
  if [ $arm = ORBITA ]; then
    wt=$R/artifacts/orbita/$ph/weights_$s.json
    $B fam-eval --weights $wt --set $W/$ph/eps/set.csv --ctx $W/$ph/eps/ctx_true.csv > $o/fam_$s.csv 2> $o/fam_$s.log || return 1
    for k in 2 4 8; do tau=$(case $k in 2) echo 0.014656;; 4) echo 0.022020;; 8) echo 0.037142;; esac)
      t0=$(date +%s.%N); $B lat-eval exp1 --weights $wt --set $W/$ph/eps/set.csv --ctx $W/$ph/eps/ctx_none.csv --k $k --tau $tau --lam0 1e-4 --lamm 1e-3 > $o/lat_${s}_k$k.csv 2> $o/lat_${s}_k$k.log || return 1; awk "BEGIN{print $(date +%s.%N)-$t0}" > $o/lat_${s}_k$k.time; done
  else
    d=/workspace/orbita-data/$ph/$s
    [ -f $o/w_$s.pt ] || { $P -W ignore $TF/tfe.py --arm $arm --data $d --out $o/w_$s.pt --seed $s > $o/train_$s.log 2>&1 || return 1; }
    t0=$(date +%s.%N)
    $P -W ignore $TF/eval_tf.py $arm $o/w_$s.pt $W/$ph/eps $o/eval_$s.csv > $o/eval_$s.log 2>&1 || return 1; awk "BEGIN{print $(date +%s.%N)-$t0}" > $o/eval_$s.time
  fi
  touch $o/$s.done; echo "job ok $ph $s $arm $(date +%T)"
}
export -f job
for ph in dev confirm; do
  [ -f $W/$ph/eps.done ] || { $P $TF/gen_eps.py $ph $W/$ph/eps && touch $W/$ph/eps.done; }
  if [ $ph = dev ]; then pre=0xA4; else pre=0xB4; fi
  for arm in TFE TF MLP ORBITA; do for i in $(seq 0 15); do printf "$ph ${pre}%02X $arm\n" $i; done; done | xargs -P $J -L 1 bash -c 'job $0 $1 $2'
  n=$(ls $W/$ph/*/*.done 2>/dev/null | wc -l)
  if [ $n -eq 64 ]; then
    [ -f $W/$ph/analyze.done ] || { $P $TF/analyze_tf.py $ph > $A/analyze_$ph.md && touch $W/$ph/analyze.done; echo "analyze $ph done $(date +%T)"; }
  else echo "phase $ph incomplete ($n/64); rerun chain"; exit 1; fi
done
echo "chain done $(date +%T)"
