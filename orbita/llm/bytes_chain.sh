#!/bin/bash
# ÓRBITA-Bytes vs ÓRBITA (docs/prereg_orbita_bytes.md). Reanudable: cada paso deja un .done; relanzar es seguro.
# Lanzar:  cd /workspace/orbita-data/bytes && setsid nohup bash chain.sh >> chain.log 2>&1 &
set -e
O=/workspace/orbita-data/bytes; W=/workspace/ngm-bytes
exec 9> $O/chain.lock; flock -n 9 || { echo "$(date '+%F %T') otra instancia tiene el lock; salgo"; exit 0; }
cd $W/orbita/llm
export PYTHONPATH=$W/orbita ORBITA_BIN=$W/orbita/target/release/orbita NPAR=4 BYTES_THREADS=2
PY=/workspace/orbita-venv/bin/python; B=$ORBITA_BIN; M=/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf
st() { echo "$(date '+%F %T') $*" | tee $O/status.txt; }
echo $$ > $O/chain.pid
st "chain start pid=$$ commit=$(git -C $W rev-parse --short HEAD)"
[ -f $O/sets/done ] || { $PY bytes_exp.py data > $O/logs_data.log 2>&1 && touch $O/sets/done; }
for ph in dev confirm; do for s in U N; do      # una sola instancia de Gemma a la vez (secuencial)
  f=$O/sets/$ph/$s
  if [ ! -f $f.tok.done ]; then
    st "gemma-embed $ph/$s start free=$(free -m | awk 'NR==2{print $7}')MB"
    $B gemma-embed --model $M --in $f.txt --out $f.tok --exit-after 12 --per-token > $f.tok.json 2> $f.tok.log
    [ $(wc -l < $f.tok.idx) -eq 1536 ] && touch $f.tok.done && st "gemma-embed $ph/$s done"
  fi
done; done
for ph in dev confirm; do
  [ -f $O/$ph.done ] && continue
  if [ $ph = dev ]; then VS="MAIN NOVAE NOHASH NONCE FIXPATCH"; else VS="MAIN"; fi
  for v in $VS; do
    [ -f $O/train_${ph}_$v.done ] || { st "train $ph $v"; $PY -W ignore bytes_exp.py train $ph $v && touch $O/train_${ph}_$v.done; }
  done
  [ -f $O/eval_${ph}_GEMMA.done ] || { st "eval $ph GEMMA"; $PY -W ignore bytes_exp.py eval $ph GEMMA && touch $O/eval_${ph}_GEMMA.done; }
  for v in $VS; do [ -f $O/eval_${ph}_$v.done ] || { st "eval $ph $v"; $PY -W ignore bytes_exp.py eval $ph $v && touch $O/eval_${ph}_$v.done; }; done
  for a in GEMMA MAIN; do [ -f $O/learn_${ph}_$a.done ] || { st "learn $ph $a"; $PY -W ignore bytes_exp.py learn $ph $a && touch $O/learn_${ph}_$a.done; }; done
  [ -f $O/recon_$ph.done ] || { st "recon $ph"; $PY -W ignore bytes_exp.py recon $ph && touch $O/recon_$ph.done; }
  [ -f $O/latency_$ph.done ] || { st "latency $ph"; $PY -W ignore bytes_exp.py latency $ph && touch $O/latency_$ph.done; }
  st "analyze $ph"; $PY -W ignore bytes_exp.py analyze $ph > $O/analyze_$ph.log 2>&1 && touch $O/$ph.done && st "$ph done"
done
st "chain done"
