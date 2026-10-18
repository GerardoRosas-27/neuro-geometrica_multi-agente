set -e
cd /workspace/ngm-orbita/orbita/llm
B=/workspace/ngm-orbita/orbita/target/release/orbita; M=/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf
PY=/workspace/orbita-venv/bin/python; export PYTHONPATH=/workspace/ngm-orbita/orbita; O=/workspace/orbita-data/yo
[ -f $O/smoke.done ] || { $PY -c "
import yo
assert yo.readout('Familia: rotación con escalado\nConfianza: 73') == (6, 73)
assert yo.readout('Familia: rotación\nConfianza: 180') == (1, 100) and yo.readout('nada') == (None, None)
p = yo.derange(96); assert (p != range(96)).all()
print(yo.prompt(yo.block({'fam_hat':'2','p_d2':'0.5','parse_ok':'0','surprise':'-2','sigma':'0','normr':'0'}, 0.42, 'YO')))
" > $O/smoke.log 2>&1 && touch $O/smoke.done; }
for ph in dev confirm; do
  [ -f $O/$ph.done ] && continue
  mkdir -p $O/$ph
  [ -f $O/$ph/cal.tsv ] || $PY yo.py caltexts $ph
  if [ ! -f $O/$ph/cal.tok.done ]; then
    echo "extract $ph start $(date +%T) free=$(free -m | awk 'NR==2{print $7}')MB"
    $B gemma-embed --model $M --in $O/$ph/cal.txt --out $O/$ph/cal.tok --exit-after 12 --per-token > $O/$ph/cal.tok.json 2> $O/$ph/cal.tok.log
    [ $(wc -l < $O/$ph/cal.tok.idx) -eq $(wc -l < $O/$ph/cal.txt) ] && touch $O/$ph/cal.tok.done && echo "extract $ph done $(date +%T)"
  fi
  [ -f $O/$ph/feats.done ] || { $PY -W ignore yo.py feats $ph && touch $O/$ph/feats.done && echo "feats $ph done $(date +%T)"; }
  $PY yo.py analyze $ph > $O/analyze_h12_$ph.log
  $PY yo.py gen $ph && $PY yo.py analyze $ph > $O/analyze_$ph.log && touch $O/$ph.done && echo "$ph done $(date +%T)"
done
