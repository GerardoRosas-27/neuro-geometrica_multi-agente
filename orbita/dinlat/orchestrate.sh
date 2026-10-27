#!/usr/bin/env bash
# orchestrate.sh — espera DEV, aplica gate, corre CONFIRM si procede, escribe RESULTADOS.md y hace push (sin force).
set -u
H=$(cd "$(dirname "$0")" && pwd); R=$(cd $H/../.. && pwd); D=/workspace/orbita-data/dinlat; P=/workspace/orbita-venv/bin/python
while [ ! -f $D/dev/done ]; do sleep 120; done
G=$($P -c "import json;print(json.load(open('$D/dev/analysis.json'))['gate_dev_a_confirm'])")
echo "gate=$G $(date -Is)"
if [ "$G" = "True" ]; then bash $H/chain.sh confirm > $D/../dinlat_confirm.log 2>&1; fi
mkdir -p $R/docs/dinamica_latente/datos
cp $D/dev/analysis.json $R/docs/dinamica_latente/datos/analysis_dev.json; cp $D/dev/analysis.txt $R/docs/dinamica_latente/datos/analysis_dev.txt
[ -f $D/confirm/analysis.json ] && cp $D/confirm/analysis.json $R/docs/dinamica_latente/datos/analysis_confirm.json && cp $D/confirm/analysis.txt $R/docs/dinamica_latente/datos/analysis_confirm.txt
grep -h FALLO $D/*.log > $R/docs/dinamica_latente/datos/fallos.txt 2>/dev/null || true
$P $H/results_md.py $R/docs/dinamica_latente/RESULTADOS.md
cd $R && git add docs/dinamica_latente orbita/dinlat && \
GIT_AUTHOR_NAME=GerardoRosas-27 GIT_AUTHOR_EMAIL=GerardoRosas-27@users.noreply.github.com GIT_COMMITTER_NAME=GerardoRosas-27 GIT_COMMITTER_EMAIL=GerardoRosas-27@users.noreply.github.com \
git commit -qm "dinlat: RESULTADOS (DEV$( [ -f $D/confirm/done ] && echo ' + CONFIRM'))" && git push origin exp/orbita-dinamica-latente
echo "fin $(date -Is)"; touch $D/orchestrate.done
