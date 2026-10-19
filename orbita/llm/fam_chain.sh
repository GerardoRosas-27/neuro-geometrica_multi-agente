set -e
cd /workspace/ngm-orbita/orbita/llm
B=/workspace/ngm-orbita/orbita/target/release/orbita; M=/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf
PY=/workspace/orbita-venv/bin/python; export PYTHONPATH=/workspace/ngm-orbita/orbita; O=/workspace/orbita-data/demo_fam
[ -f $O/texts.tsv ] || $PY fam_eval.py texts
if [ ! -f $O/texts.tok.done ]; then
  echo "extract start $(date +%T) free=$(free -m | awk 'NR==2{print $7}')MB"
  $B gemma-embed --model $M --in $O/texts.txt --out $O/texts.tok --exit-after 12 --per-token > $O/texts.tok.json 2> $O/texts.tok.log
  [ $(wc -l < $O/texts.tok.idx) -eq $(wc -l < $O/texts.txt) ] && touch $O/texts.tok.done && echo "extract done $(date +%T)"
fi
for ph in dev confirm; do
  [ -f $O/$ph.done ] && continue
  $PY -W ignore fam_eval.py run $ph && $PY fam_eval.py analyze $ph > $O/analyze_$ph.log && touch $O/$ph.done && echo "$ph done $(date +%T)"
done
