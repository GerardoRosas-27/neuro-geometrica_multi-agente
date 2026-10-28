#!/usr/bin/env bash
# orchestrate_confirm.sh — relanzamiento de CONFIRM (el 1er intento falló: faltaba el binario gemma-embed en el worktree,
# texts.emb vacío → 0 seeds). Gate DEV ya pasó (commit 1890274). Verifica 16/16 antes de escribir RESULTADOS.
set -u
H=$(cd "$(dirname "$0")" && pwd); R=$(cd $H/../.. && pwd); D=/workspace/orbita-data/timesfm
bash $H/chain.sh confirm > $D/../timesfm_confirm.log 2>&1
cd $R; export GIT_AUTHOR_NAME=GerardoRosas-27 GIT_AUTHOR_EMAIL=GerardoRosas-27@users.noreply.github.com GIT_COMMITTER_NAME=GerardoRosas-27 GIT_COMMITTER_EMAIL=GerardoRosas-27@users.noreply.github.com
N=$(ls $D/confirm/tfm_*.json 2>/dev/null | wc -l)
if [ "$N" = 16 ]; then
  cp $D/confirm/analysis.json docs/orbita_vs_timesfm/datos/analysis_confirm.json; cp $D/confirm/analysis.txt docs/orbita_vs_timesfm/datos/analysis_confirm.txt
  python3 $H/results_md.py docs/orbita_vs_timesfm/RESULTADOS.md
  git add docs/orbita_vs_timesfm && git commit -qm "vstimesfm: RESULTADOS (DEV + CONFIRM 16/16)" && git push origin exp/orbita-vs-timesfm
else echo "CONFIRM incompleto: $N/16 seeds; no se escribe resultado" ; fi
echo "fin $(date -Is) N=$N"; touch $D/orchestrate.done
