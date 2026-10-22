set -e
cd /workspace/ngm-orbita/orbita/llm
B=/workspace/ngm-orbita/orbita/target/release/orbita; M=/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf
PY=/workspace/orbita-venv/bin/python; export PYTHONPATH=/workspace/ngm-orbita/orbita; O=/workspace/orbita-data/llm_e2e
sh test_e2e.sh > $O/smoke.log 2>&1 && touch $O/smoke.done
for ph in dev confirm; do
  [ -f $O/$ph.done ] && continue
  [ -f $O/$ph/robust.tsv ] || $PY e2e.py texts $ph
  if [ ! -f $O/$ph/robust.tok.done ]; then   # reanudable: gemma-embed --per-token retoma desde .idx
    echo "extract $ph start $(date +%T) free=$(free -m | awk 'NR==2{print $7}')MB"
    $B gemma-embed --model $M --in $O/$ph/robust.txt --out $O/$ph/robust.tok --exit-after 12 --per-token > $O/$ph/robust.tok.json 2> $O/$ph/robust.tok.log
    [ $(wc -l < $O/$ph/robust.tok.idx) -eq 1536 ] && touch $O/$ph/robust.tok.done && echo "extract $ph done $(date +%T)"
  fi
  $PY -W ignore e2e.py ctx $ph && $PY e2e.py eval $ph && $PY e2e.py analyze $ph > $O/analyze_$ph.log && touch $O/$ph.done && echo "$ph done $(date +%T)"
done
