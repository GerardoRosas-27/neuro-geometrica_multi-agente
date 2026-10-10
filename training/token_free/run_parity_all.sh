#!/bin/bash
# Paridad + latencia Rust (1 hilo) para las 4 dinámicas x 6 semillas.
set -e
cd "$(dirname "$0")"
C=/workspace/ngm-tokenfree-cache
for s in 0 1 2 3 4 5; do for m in mlp liquid ssm linear; do
  [ -f $C/corpus_${m}_s$s.json ] || /workspace/venv-tokenfree/bin/python parity.py $m 64 $s $C/corpus_${m}_s$s.json
  echo "$m s$s $(RAYON_NUM_THREADS=1 ../../target/release/run_token_free_field $C/e46_${m}_N64_s$s.json $C/corpus_${m}_s$s.json 20)"
done; done
