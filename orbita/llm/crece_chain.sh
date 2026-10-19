set -e
cd /workspace/ngm-orbita/orbita/llm
B=/workspace/ngm-orbita/orbita/target/release/orbita; M=/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf
PY=/workspace/orbita-venv/bin/python; export PYTHONPATH=/workspace/ngm-orbita/orbita; O=/workspace/orbita-data/crece
[ -x $B ] || (cd /workspace/ngm-orbita/orbita && cargo build --release)
mkdir -p $O
for ph in dev confirm; do
  [ -f $O/$ph/S.tsv ] || $PY crece.py texts $ph
  for n in H S; do
    if [ ! -f $O/$ph/$n.tok.done ]; then
      while pgrep -f "gemma-embed" > /dev/null; do echo "otro Gemma activo; espero $(date +%T)"; sleep 60; done
      echo "extract $ph $n start $(date +%T) free=$(free -m | awk 'NR==2{print $7}')MB"
      $B gemma-embed --model $M --in $O/$ph/$n.txt --out $O/$ph/$n.tok --exit-after 12 --per-token > $O/$ph/$n.tok.json 2> $O/$ph/$n.tok.log
      [ $(wc -l < $O/$ph/$n.tok.idx) -eq $(wc -l < $O/$ph/$n.txt) ] && touch $O/$ph/$n.tok.done && echo "extract $ph $n done $(date +%T)"
    fi
  done
  [ -f $O/$ph/ctx.done ] || { $PY -W ignore crece.py ctx $ph && touch $O/$ph/ctx.done; }
done
[ -f $O/calib.done ] || { $PY crece.py calib > $O/calib.log && touch $O/calib.done && echo "calib done $(date +%T)"; }
[ -f $O/grid.done ] || { $PY crece.py grid > $O/grid.log && touch $O/grid.done && echo "grid done $(date +%T)"; }
[ -f $O/confirm.done ] || { $PY crece.py confirm > $O/confirm.log && touch $O/confirm.done && echo "confirm done $(date +%T)"; }
for ph in dev confirm; do
  [ -f $O/analyze_$ph.done ] || { $PY crece.py analyze $ph > $O/analyze_$ph.log && touch $O/analyze_$ph.done && echo "analyze $ph done $(date +%T)"; }
done
echo "chain done $(date +%T)"
