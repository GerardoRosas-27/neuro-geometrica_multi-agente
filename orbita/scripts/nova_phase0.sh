#!/usr/bin/env bash
# Fase 0 factibilidad: Nova (C3+C4+C5) vs ÓRBITA en distribución propia (fam 0–7).
# Sin entrenar. Criterio: C4 gana en ≥14/16 seeds (mediana log rel_err h16).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/orbita/target/release/orbita}
WDIR=${WDIR:-$ROOT/artifacts/orbita/dev}
OUT=${OUT:-$ROOT/artifacts/orbita/nova_fase0}
N=${N:-64}
K=${K:-4}
RANK=${RANK:-2}
LAM=${LAM:-1e-6}
NOISE=${NOISE:-0.01}
mkdir -p "$OUT"
echo "bin=$BIN out=$OUT n=$N k=$K rank=$RANK noise=$NOISE"
# seeds DEV 0xA400–0xA40F
SEEDS=(A400 A401 A402 A403 A404 A405 A406 A407 A408 A409 A40A A40B A40C A40D A40E A40F)
SUMMARY="$OUT/summary.csv"
echo "seed,c4_wins,nova_wins,delta_c4,delta_nova,orbita_med_log,c4_med_log,nova_med_log,orbita_acc16,c4_acc16,nova_acc16,nis_coverage,accept_rate" > "$SUMMARY"
wins=0
nwins=0
for s in "${SEEDS[@]}"; do
  W="$WDIR/weights_0x${s}.json"
  if [[ ! -f "$W" ]]; then echo "missing $W" >&2; exit 1; fi
  echo "=== seed 0x$s ==="
  JSON=$("$BIN" nova-eval --weights "$W" --seed "0x$s" --n "$N" --k "$K" --rank "$RANK" --lam "$LAM" --noise "$NOISE" --out "$OUT")
  echo "$JSON" | tee "$OUT/fase0_0x${s}.json" >/dev/null
  # re-print for logs
  echo "$JSON"
  python3 - "$SUMMARY" "$JSON" <<'PY'
import json,sys
summary, raw = sys.argv[1], sys.argv[2]
d=json.loads(raw)
row=",".join([
  d["seed"], str(d["c4_wins"]).lower(), str(d["nova_wins"]).lower(),
  f'{d["delta_c4"]:.8f}', f'{d["delta_nova"]:.8f}',
  f'{d["orbita_med_log_rel_h16"]:.8f}', f'{d["c4_med_log_rel_h16"]:.8f}', f'{d["nova_med_log_rel_h16"]:.8f}',
  f'{d["orbita_acc16"]:.4f}', f'{d["c4_acc16"]:.4f}', f'{d["nova_acc16"]:.4f}',
  f'{d["nis_coverage"]:.4f}', f'{d["accept_rate"]:.4f}',
])
open(summary,"a").write(row+"\n")
print(("C4_WIN" if d["c4_wins"] else "C4_LOSE"), ("NOVA_WIN" if d["nova_wins"] else "NOVA_LOSE"), "Δc4=", d["delta_c4"])
PY
done
python3 - "$SUMMARY" "$OUT" <<'PY'
import csv,sys,statistics,json
from pathlib import Path
summary, out = Path(sys.argv[1]), Path(sys.argv[2])
rows=list(csv.DictReader(summary.open()))
c4=sum(1 for r in rows if r["c4_wins"]=="true")
nv=sum(1 for r in rows if r["nova_wins"]=="true")
nis=[float(r["nis_coverage"]) for r in rows]
mean_nis=sum(nis)/len(nis)
gate=c4>=14
report={
  "phase": 0,
  "criterion": "C4 mejora a ÓRBITA (mediana log rel_err h16) en ≥14/16 seeds",
  "c4_wins": c4,
  "nova_wins": nv,
  "n_seeds": len(rows),
  "pass": gate,
  "nis_coverage_mean": mean_nis,
  "nis_calibrated_95_pm5": abs(mean_nis-0.95)<=0.05,
  "seeds": rows,
}
out.joinpath("FASE0_SUMMARY.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({k:report[k] for k in ["c4_wins","nova_wins","n_seeds","pass","nis_coverage_mean","nis_calibrated_95_pm5"]}, indent=2))
print("PASS" if gate else "FAIL", f"C4 {c4}/16")
PY
