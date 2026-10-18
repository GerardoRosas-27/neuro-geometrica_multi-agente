set -e
cd /workspace/ngm-orbita/orbita/llm; PY=/workspace/orbita-venv/bin/python; export PYTHONPATH=/workspace/ngm-orbita/orbita; O=/workspace/orbita-data/mem
for ph in dev confirm; do
  [ -f $O/$ph.done ] && continue
  $PY mem.py run $ph && $PY mem.py analyze $ph > $O/analyze_$ph.log && touch $O/$ph.done && echo "$ph done $(date +%T)"
done
