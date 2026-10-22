set -e
cd /workspace/ngm-orbita/orbita/llm
B=/workspace/ngm-orbita/orbita/target/release/orbita; M=/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf
PY=/workspace/orbita-venv/bin/python; export PYTHONPATH=/workspace/ngm-orbita/orbita; O=/workspace/orbita-data/learn
[ -x $B ] || (cd /workspace/ngm-orbita/orbita && cargo build --release)
for ph in dev confirm; do
  [ -f $O/$ph/texts.tsv ] || $PY learn_eval.py texts $ph
  if [ ! -f $O/$ph/texts.tok.done ]; then
    while pgrep -f "gemma-embed" > /dev/null; do echo "otro Gemma activo; espero $(date +%T)"; sleep 60; done
    echo "extract $ph start $(date +%T) free=$(free -m | awk 'NR==2{print $7}')MB"
    $B gemma-embed --model $M --in $O/$ph/texts.txt --out $O/$ph/texts.tok --exit-after 12 --per-token > $O/$ph/texts.tok.json 2> $O/$ph/texts.tok.log
    [ $(wc -l < $O/$ph/texts.tok.idx) -eq $(wc -l < $O/$ph/texts.txt) ] && touch $O/$ph/texts.tok.done && echo "extract $ph done $(date +%T)"
  fi
  [ -f $O/$ph/ctx.done ] || { $PY -W ignore learn_eval.py ctx $ph && touch $O/$ph/ctx.done; }
done
[ -f $O/calib.done ] || { $PY learn_eval.py calib > $O/calib.log && touch $O/calib.done && echo "calib done $(date +%T)"; }
[ -f $O/grid.done ] || { $PY learn_eval.py grid > $O/grid.log && touch $O/grid.done && echo "grid done $(date +%T)"; }
for ph in dev confirm; do
  [ -f $O/run_$ph.done ] || { $PY learn_eval.py run $ph && $PY learn_eval.py analyze $ph > $O/analyze_$ph.log && touch $O/run_$ph.done && echo "$ph done $(date +%T)"; }
done
echo "chain done $(date +%T)"
