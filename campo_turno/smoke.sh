#!/usr/bin/env bash
# SMOKE de extremo a extremo con Gemma2 DIMINUTO ALEATORIO (sin pesos reales). NO produce resultados.
# uso: PY=python3 bash campo_turno/smoke.sh /tmp/ct_smoke conv.jsonl [gpu|cpu]
set -euo pipefail
S=${1:-/tmp/ct_smoke}; J=${2:?"falta jsonl"}; DEV=${3:-cpu}; PY=${PY:-python3}
ROOT=$(cd "$(dirname "$0")/.." && pwd); BIN=$ROOT/gemma_campo/target/release/gemma_campo
(cd "$ROOT/gemma_campo" && cargo build --release -q)
$BIN validate-wgsl
cd "$ROOT/campo_turno"
$PY -m unittest discover -s tests -v
$PY turno_extract.py --tiny-random --jsonl "$J" --n-conv 80 --layer 2 --max-turn-tokens 48 --out $S/h
$PY turno_encoder.py --hdir $S/h --out $S/z --d-field 8
$BIN train --data $S/z --out $S/m --device $DEV --ctx 4 --epochs 3 --batch 64 --lr 0.01 --threads 2
$BIN predict --data $S/z --model $S/m --split val --kmax 1 --stride 1
$BIN predict --data $S/z --model $S/m --split test --kmax 1 --stride 1
$PY turno_baselines.py --zdir $S/z --out $S/b --ctx 4 --mlp-epochs 30
$PY turno_adapter.py --tiny-random --hdir $S/h --zdir $S/z --out $S/a --epochs 1 --max-train 60 --max-val 20 --ctx-tokens 128 --width 64
$PY turno_eval.py --tiny-random --hdir $S/h --zdir $S/z --field-dir $S/m --base-dir $S/b --adapter-dir $S/a --out $S/eval --ctx-tokens 128 --max-new 12 --max-eval 40
