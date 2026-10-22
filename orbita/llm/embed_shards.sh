#!/usr/bin/env bash
# embed_shards.sh IN.txt OUT.emb [N=3] — N procesos Gemma en paralelo (reanudable), luego concatena en orden.
set -eu
IN=$1; OUT=$2; N=${3:-3}
B=$(cd "$(dirname "$0")/.." && pwd)/target/release/orbita
M=${GEMMA_GGUF_PATH:-/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf}
L=$(wc -l < $IN); S=$(( (L + N - 1) / N ))
split -d -l $S $IN $IN.part.
for p in $IN.part.*; do $B gemma-embed --model $M --in $p --out $p.emb --exit-after 12 > $p.json 2> $p.log & done
wait
cat $(ls $IN.part.*.emb | sort) > $OUT
echo "embedded $(( $(stat -c %s $OUT) / 9216 )) / $L -> $OUT"
