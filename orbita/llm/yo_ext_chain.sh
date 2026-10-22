# espera a que termine la evaluación del yo y luego entrena (docs/preregistro_yo_ext.md)
cd /workspace/ngm-orbita/orbita/llm
B=/workspace/ngm-orbita/orbita/target/release/orbita; M=/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf
PY=/workspace/orbita-venv/bin/python; export PYTHONPATH=/workspace/ngm-orbita/orbita; O=/workspace/orbita-data/yo_ext
while [ ! -f /workspace/orbita-data/yo/confirm.done ]; do sleep 120; done
set -e
echo "start $(date +%T) free=$(free -m | awk 'NR==2{print $7}')MB"
[ -f $O/pool2.tsv ] || $PY yo_ext.py texts
if [ ! -f $O/pool2.tok.done ]; then
  $B gemma-embed --model $M --in $O/pool2.txt --out $O/pool2.tok --exit-after 12 --per-token > $O/pool2.tok.json 2> $O/pool2.tok.log
  [ $(wc -l < $O/pool2.tok.idx) -eq $(wc -l < $O/pool2.txt) ] && touch $O/pool2.tok.done && echo "extract done $(date +%T)"
fi
for ph in dev confirm; do
  [ -f $O/$ph.done ] && continue
  for s in $($PY -c "import pipeline as P; print(' '.join(P.SEEDS['$ph']))"); do
    [ -f $O/$ph/$s.train.done ] || echo $s
  done | xargs -r -P 8 -I{} $PY yo_ext.py train $ph {} > $O/train_$ph.log 2>&1
  echo "train $ph done $(date +%T)"
  $PY -W ignore yo_ext.py feats $ph && $PY yo_ext.py analyze $ph > $O/analyze_$ph.log && touch $O/$ph.done && echo "$ph done $(date +%T)"
done
