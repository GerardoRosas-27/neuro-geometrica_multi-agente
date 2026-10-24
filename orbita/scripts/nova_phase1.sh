#!/usr/bin/env bash
# Fase 1 suelo: techo ORACLE_EP / C4_EP / ORACLE_FAMNET por familia nueva 8–11 y k∈{2,4,8}.
# Sin train largo. Criterio k=4: best_acc16≤0.02 y best_delta_log≥−0.1 → excluir familia.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/orbita/target/release/orbita}
WDIR=${WDIR:-$ROOT/artifacts/orbita/dev}
OUT=${OUT:-$ROOT/artifacts/orbita/nova_fase1}
NPF=${NPF:-16}
RANK=${RANK:-2}
LAM=${LAM:-1e-6}
mkdir -p "$OUT"
echo "bin=$BIN out=$OUT n_per_fam=$NPF rank=$RANK"
SEEDS=(A400 A401 A402 A403 A404 A405 A406 A407 A408 A409 A40A A40B A40C A40D A40E A40F)
SUMMARY="$OUT/summary.csv"
echo "seed,fam,k,n,orbita_med_log,oep_med_log,c4_med_log,famnet_med_log,orbita_acc16,oep_acc16,c4_acc16,famnet_acc16,delta_oep,delta_c4,delta_famnet,best_acc16,best_delta_log,floor" > "$SUMMARY"
for s in "${SEEDS[@]}"; do
  W="$WDIR/weights_0x${s}.json"
  [[ -f "$W" ]] || { echo "missing $W" >&2; exit 1; }
  echo "=== seed 0x$s ==="
  JSON=$("$BIN" nova-eval --ceiling --weights "$W" --seed "0x$s" --n-per-fam "$NPF" --rank "$RANK" --lam "$LAM" --out "$OUT")
  echo "$JSON" > "$OUT/fase1_0x${s}.json"
  python3 - "$SUMMARY" "$JSON" <<'PY'
import json,sys
summary, raw = sys.argv[1], sys.argv[2]
d=json.loads(raw)
with open(summary,"a") as f:
  for r in d["rows"]:
    f.write(",".join([
      d["seed"], str(r["fam"]), str(r["k"]), str(r["n"]),
      f'{r["orbita_med_log"]:.8f}', f'{r["oep_med_log"]:.8f}', f'{r["c4_med_log"]:.8f}', f'{r["famnet_med_log"]:.8f}',
      f'{r["orbita_acc16"]:.4f}', f'{r["oep_acc16"]:.4f}', f'{r["c4_acc16"]:.4f}', f'{r["famnet_acc16"]:.4f}',
      f'{r["delta_oep"]:.8f}', f'{r["delta_c4"]:.8f}', f'{r["delta_famnet"]:.8f}',
      f'{r["best_acc16"]:.4f}', f'{r["best_delta_log"]:.8f}', str(r["floor"]).lower(),
    ])+"\n")
print("seed", d["seed"], "rows", len(d["rows"]))
PY
done
python3 - "$SUMMARY" "$OUT" <<'PY'
import csv,json,sys
from pathlib import Path
from collections import defaultdict
summary, out = Path(sys.argv[1]), Path(sys.argv[2])
rows=list(csv.DictReader(summary.open()))
# agregar por (fam,k): media de seeds
agg=defaultdict(list)
for r in rows:
  agg[(int(r["fam"]), int(r["k"]))].append(r)

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

fam_names={8:"giro_ondulado",9:"espiral",10:"pulso_radial",11:"cizalla_saturada"}
table=[]
exclude=[]
include=[]
for fam in (8,9,10,11):
  # primario k=4
  rk=agg[(fam,4)]
  floors=sum(1 for r in rk if r["floor"]=="true")
  rec={
    "fam": fam,
    "name": fam_names[fam],
    "k": 4,
    "n_seeds": len(rk),
    "orbita_acc16_mean": mean([float(r["orbita_acc16"]) for r in rk]),
    "oep_acc16_mean": mean([float(r["oep_acc16"]) for r in rk]),
    "c4_acc16_mean": mean([float(r["c4_acc16"]) for r in rk]),
    "famnet_acc16_mean": mean([float(r["famnet_acc16"]) for r in rk]),
    "orbita_med_log_mean": mean([float(r["orbita_med_log"]) for r in rk]),
    "oep_med_log_mean": mean([float(r["oep_med_log"]) for r in rk]),
    "c4_med_log_mean": mean([float(r["c4_med_log"]) for r in rk]),
    "famnet_med_log_mean": mean([float(r["famnet_med_log"]) for r in rk]),
    "best_acc16_mean": mean([float(r["best_acc16"]) for r in rk]),
    "best_delta_log_mean": mean([float(r["best_delta_log"]) for r in rk]),
    "floor_seeds": floors,
    "floor_majority": floors >= 8,  # ≥8/16 seeds en suelo
  }
  # también k=2 y k=8
  for k in (2,8):
    rk2=agg[(fam,k)]
    rec[f"k{k}_best_acc16_mean"]=mean([float(r["best_acc16"]) for r in rk2])
    rec[f"k{k}_best_delta_log_mean"]=mean([float(r["best_delta_log"]) for r in rk2])
    rec[f"k{k}_oep_acc16_mean"]=mean([float(r["oep_acc16"]) for r in rk2])
    rec[f"k{k}_famnet_acc16_mean"]=mean([float(r["famnet_acc16"]) for r in rk2])
    rec[f"k{k}_c4_acc16_mean"]=mean([float(r["c4_acc16"]) for r in rk2])
  # decisión: excluir si mayoría de seeds en suelo a k=4 Y best_acc medio ≤0.02
  excl = rec["floor_majority"] and rec["best_acc16_mean"] <= 0.02
  rec["exclude"] = excl
  (exclude if excl else include).append(fam)
  table.append(rec)

# k_fit/k_val sugeridos (del plan / investigación)
k_fit_val={"2":"1/1","4":"3/1","8":"6/2"}
report={
  "phase": 1,
  "primary_k": 4,
  "floor_rule_per_seed": "best_acc16≤0.02 y best_delta_log≥−0.1 (k=4)",
  "exclude_rule": "mayoría (≥8/16) seeds en suelo y best_acc16_mean≤0.02",
  "include_fams": include,
  "exclude_fams": exclude,
  "k_fit_k_val": k_fit_val,
  "by_fam_k4": table,
  "n_rows_csv": len(rows),
}
out.joinpath("FASE1_SUMMARY.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({
  "include_fams": include,
  "exclude_fams": exclude,
  "by_fam": [{k:rec[k] for k in ("fam","name","best_acc16_mean","best_delta_log_mean","floor_seeds","exclude",
    "oep_acc16_mean","c4_acc16_mean","famnet_acc16_mean")} for rec in table]
}, indent=2, ensure_ascii=False))
PY
