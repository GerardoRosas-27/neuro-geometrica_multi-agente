#!/usr/bin/env bash
# chain.sh PHASE — reutiliza episodios+embeddings Gemma de dinlat (NO re-embebe) → run_hib por seed (8 en paralelo) → analyze.
# Verifica identidad con res_<seed>.json de dinlat (DINLAT_RES). Sin relanzar seeds fallidos.
set -eu
PH=$1; H=$(cd "$(dirname "$0")" && pwd); SRC=${DINLAT_DATA:-/workspace/orbita-data/dinlat}/$PH
O=${HIB_DATA:-/workspace/orbita-data/hibrido}/$PH; P=/workspace/orbita-venv/bin/python
[ -f $SRC/texts.emb ] || { echo "falta $SRC/texts.emb"; exit 1; }
mkdir -p $O; date -Is > $O/start; i=0; : > $O/jobs
for s in $(python3 -c "import sys;sys.path.insert(0,'$H/../dinlat');import gen;print(' '.join(gen.SEEDS['$PH']))"); do
  echo "$s $SRC/eps_$s.json $SRC/texts.emb $((i*756)) $O/res_$s.json $SRC/res_$s.json" >> $O/jobs; i=$((i+1)); done
xargs -P 8 -L 1 sh -c 'DINLAT_RES="$5" OMP_NUM_THREADS=1 '$P' '$H'/run_hib.py "$0" "$1" "$2" "$3" "$4" > '$O'/log_$0.txt 2>&1 || echo FALLO $0' < $O/jobs
$P $H/analyze.py $PH $O > $O/analysis.txt
date -Is > $O/done
