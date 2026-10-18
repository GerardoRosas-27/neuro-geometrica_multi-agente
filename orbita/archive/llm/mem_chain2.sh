set -e
cd /workspace/ngm-orbita/orbita/llm; PY=/workspace/orbita-venv/bin/python; export PYTHONPATH=/workspace/ngm-orbita/orbita; O=/workspace/orbita-data/mem
for ph in dev confirm; do
  [ -f $O/$ph.s2.done ] && continue
  $PY mem.py run2 $ph && $PY mem.py analyze2 $ph > $O/analyze2_$ph.log && touch $O/$ph.s2.done && echo "$ph s2 done $(date +%T)"
done
