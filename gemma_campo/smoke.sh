#!/usr/bin/env bash
# SMOKE de extremo a extremo con Gemma2 DIMINUTO ALEATORIO (sin pesos de Gemma). NO produce resultados.
# uso: PY=python3 bash gemma_campo/smoke.sh /tmp/gc_smoke conv.jsonl [gpu|cpu]
set -euo pipefail
S=${1:-/tmp/gc_smoke}; J=${2:?"falta jsonl (un {messages:[...]} por linea)"}; DEV=${3:-gpu}; PY=${PY:-python3}
cd "$(dirname "$0")"
cargo build --release
./target/release/gemma_campo validate-wgsl
./target/release/gemma_campo check-gpu || [ "$DEV" = cpu ]
cd python
$PY extract.py --tiny-random --jsonl "$J" --n-conv 60 --layer 2 --max-tokens 256 --out $S/h
$PY encoder.py --hdir $S/h --out $S/z --d-field 16
../target/release/gemma_campo train --data $S/z --out $S/m --device $DEV --epochs 3 --batch 256 --lr 0.01
../target/release/gemma_campo predict --data $S/z --model $S/m --split test --kmax 8 --stride 4
$PY parity.py --zdir $S/z --model $S/m
$PY eval.py --tiny-random --hdir $S/h --zdir $S/z --model-dir $S/m --out $S/eval
$PY chat_demo.py --tiny-random --hdir $S/h --zdir $S/z --model-dir $S/m --max-new 10
