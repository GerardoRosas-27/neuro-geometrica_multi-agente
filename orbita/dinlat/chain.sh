#!/usr/bin/env bash
# chain.sh PHASE — gen → Gemma (capa 13) → run por seed (8 en paralelo) → analyze. Sin relanzar.
set -eu
PH=$1; H=$(cd "$(dirname "$0")" && pwd); O=${DINLAT_DATA:-/workspace/orbita-data/dinlat}/$PH; P=/workspace/orbita-venv/bin/python
export GEMMA_GGUF_PATH=${GEMMA_GGUF_PATH:-/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf}
mkdir -p $O; date -Is > $O/start
python3 $H/gen.py $PH $O
[ -f $O/texts.emb ] || bash $H/../llm/embed_shards.sh $O/texts.txt $O/texts.emb 2
i=0; : > $O/jobs
for s in $(python3 -c "import sys;sys.path.insert(0,'$H');import gen;print(' '.join(gen.SEEDS['$PH']))"); do
  echo "$s $O/eps_$s.json $O/texts.emb $((i*756)) $O/res_$s.json" >> $O/jobs; i=$((i+1)); done
xargs -P 8 -L 1 sh -c 'OMP_NUM_THREADS=1 '$P' '$H'/run.py "$0" "$@" > '$O'/log_$0.txt 2>&1 || echo FALLO $0' < $O/jobs
$P $H/analyze.py $PH $O > $O/analysis.txt
date -Is > $O/done
