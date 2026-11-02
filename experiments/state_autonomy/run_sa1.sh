#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP="$ROOT/experiments/state_autonomy"
OUT="${OUT:-$EXP/out}"
STEPS="${STEPS:-256000}"
SEED="${SEED:-0xA900}"
mkdir -p "$OUT"

python3 "$EXP/generate_state_autonomy.py" --out "$OUT/train" --seed "$SEED" --mechanisms 256 --episodes 32 --split train

python3 "$ROOT/orbita/train_orbita.py" --data "$OUT/train" --out "$OUT/orbita_full.json" --seed "$SEED" --steps "$STEPS"
python3 "$ROOT/orbita/train_orbita.py" --data "$OUT/train" --out "$OUT/orbita_A.json" --seed "$SEED" --steps "$STEPS" --variant A
python3 "$ROOT/orbita/train_orbita.py" --data "$OUT/train" --out "$OUT/orbita_B.json" --seed "$SEED" --steps "$STEPS" --variant B
python3 "$ROOT/orbita/train_orbita.py" --data "$OUT/train" --out "$OUT/orbita_C.json" --seed "$SEED" --steps "$STEPS" --variant C
python3 "$EXP/train_mlp.py" --data "$OUT/train" --out "$OUT/mlp.pt" --seed "$SEED" --steps "$STEPS"

python3 "$EXP/evaluate_state_autonomy.py" \
  --weights "$OUT/orbita_full.json" \
  --weights "$OUT/orbita_A.json" \
  --weights "$OUT/orbita_B.json" \
  --weights "$OUT/orbita_C.json" \
  --seed 0xB900 --mechanisms 64 --episodes 8 \
  --out "$OUT/eval_confirm.json"

echo "SA-1 evaluation written to $OUT/eval_confirm.json"
