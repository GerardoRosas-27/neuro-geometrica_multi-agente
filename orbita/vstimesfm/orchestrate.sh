#!/usr/bin/env bash
# orchestrate.sh — DEV → gate → CONFIRM (si procede; incluye Gemma+ÓRBITA de confirm) → RESULTADOS.md → commit+push (sin force) → done-file.
set -u
H=$(cd "$(dirname "$0")" && pwd); R=$(cd $H/../.. && pwd); D=/workspace/orbita-data/timesfm
bash $H/chain.sh dev > $D/../timesfm_dev.log 2>&1
G=$(python3 -c "import json;print(json.load(open('$D/dev/analysis.json'))['gate_dev_a_confirm'])")
echo "gate=$G $(date -Is)"
cd $R; mkdir -p docs/orbita_vs_timesfm/datos; cp $D/dev/analysis.json docs/orbita_vs_timesfm/datos/analysis_dev.json; cp $D/dev/analysis.txt docs/orbita_vs_timesfm/datos/analysis_dev.txt
python3 $H/results_md.py docs/orbita_vs_timesfm/RESULTADOS.md
export GIT_AUTHOR_NAME=GerardoRosas-27 GIT_AUTHOR_EMAIL=GerardoRosas-27@users.noreply.github.com GIT_COMMITTER_NAME=GerardoRosas-27 GIT_COMMITTER_EMAIL=GerardoRosas-27@users.noreply.github.com
git add docs/orbita_vs_timesfm orbita/vstimesfm && git commit -qm "vstimesfm: RESULTADOS (DEV)" && git push origin exp/orbita-vs-timesfm
if [ "$G" = "True" ]; then
  bash $H/chain.sh confirm > $D/../timesfm_confirm.log 2>&1
  cp $D/confirm/analysis.json docs/orbita_vs_timesfm/datos/analysis_confirm.json; cp $D/confirm/analysis.txt docs/orbita_vs_timesfm/datos/analysis_confirm.txt
  python3 $H/results_md.py docs/orbita_vs_timesfm/RESULTADOS.md
  git add docs/orbita_vs_timesfm && git commit -qm "vstimesfm: RESULTADOS (DEV + CONFIRM)" && git push origin exp/orbita-vs-timesfm
fi
grep -h FALLO $D/*/log_* $D/../timesfm_*.log 2>/dev/null > /dev/null && echo "hubo FALLOS"
echo "fin $(date -Is)"; touch $D/orchestrate.done
