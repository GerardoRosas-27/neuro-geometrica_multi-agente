set -e
cd /workspace/ngm-orbita/orbita/llm
B=/workspace/ngm-orbita/orbita/target/release/orbita; M=/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf
PY=/workspace/orbita-venv/bin/python; export PYTHONPATH=/workspace/ngm-orbita/orbita; O=/workspace/orbita-data/llm_cdt
[ -f $O/pool3.tsv ] || $PY cdt.py pool
if [ ! -f $O/pool3.tok.done ]; then   # reanudable desde .idx
  echo "extract pool3 start $(date +%T) free=$(free -m | awk 'NR==2{print $7}')MB"
  $B gemma-embed --model $M --in $O/pool3.txt --out $O/pool3.tok --exit-after 12 --per-token > $O/pool3.tok.json 2> $O/pool3.tok.log
  [ $(wc -l < $O/pool3.tok.idx) -eq 1536 ] && touch $O/pool3.tok.done && echo "extract done $(date +%T)"
fi
for ph in dev confirm; do
  [ -f $O/$ph.done ] && continue
  $PY -W ignore cdt.py feat $ph && echo "feat $ph $(date +%T)"
  S=$(python3 -c "import sys;sys.path.insert(0,'.');import pipeline;print(' '.join(pipeline.SEEDS['$ph']))")
  AV=$(free -m | awk 'NR==2{print $7}'); J=$(( (AV - 800) / 600 )); [ $J -gt 8 ] && J=8; [ $J -lt 1 ] && J=1
  echo "train $ph J=$J $(date +%T)"
  echo $S | tr ' ' '\n' | xargs -P $J -I{} $PY -W ignore cdt.py train $ph {}
  $PY cdt.py ctx $ph && $PY cdt.py eval $ph && $PY cdt.py analyze $ph > $O/analyze_$ph.log && touch $O/$ph.done && echo "$ph done $(date +%T)"
done
