#!/usr/bin/env bash
# Cadena BQ10-aislado-k: [toolchain/binario] → Paso 0 (hash v4 0xC420 + smoke BQ10 0xC4B0 bit a bit) → DEV → CONFIRM → análisis → RESULTADOS.md → push → CHAIN_DONE.
# Reanudable (.done por semilla, .phase_done por fase). Aborta si falta binario/cargo o fase incompleta. Push sólo a exp/bq10-aislado-k (sin force).
set -uo pipefail
cd /workspace/ngm-bq10k; BR=exp/bq10-aislado-k; A=artifacts/bq10_aislado_k; D=docs/bq10_aislado_k; B=orbita/target/release/orbita; P=orbita/scripts/bq10k_phase.sh
ARMS=v4,ALL,CORE,BQ10,BQ10_nov4,ATLAS_nov4; KS=2,4,8,12
die(){ echo "ABORT $(date -Is): $*"; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BR" ] || die "rama incorrecta"
export PATH=$HOME/.cargo/bin:$PATH; command -v cargo >/dev/null || die "falta toolchain Rust (rustup; cd orbita && nice -n 10 cargo build --release)"
[ -x $B ] || die "falta binario $B (cd orbita && nice -n 10 cargo build --release)"
mkdir -p $A
cnt(){ ls $A/$1/*.done 2>/dev/null | wc -l; }
push(){ git add -A $A $D && git commit -qm "$1" && git push -q origin HEAD:$BR || die "push falló"; }
echo "START $(date -Is)"
if [ ! -f $A/paso0.phase_done ]; then
  [ -f $A/paso0_0xC420.json ] || (nice -n 10 $B cst4-run --phase smoke --seed 0xC420 --arms cst --weights artifacts/orbita/dev/weights_0xA400.json --out $A/paso0_c420 > $A/paso0_0xC420.log 2>&1; cp $A/paso0_c420/smoke/metrics_0xC420.json $A/paso0_0xC420.json) &
  JOBS=1 $P bq_tune $ARMS $KS 0 &
  wait
  python3 - <<'PY' || die "Paso 0 FALLA"
import json;A='artifacts/bq10_aislado_k'
h=json.load(open(f'{A}/paso0_0xC420.json'))['arms']['cst']['hash'];assert h=='1985bdbad5eee41f',h
r=json.load(open(f'{A}/bq_tune/metrics_0xC4B0.json'));v=r['arms']['BQ10']['held']['new_med_log'];assert v==-1.4055906062744339,repr(v)
assert r['arms_k']['4']['BQ10']['held']['new_med_log']==v and r['test_queries']==0
print('PASO0 OK',h,repr(v))
PY
  touch $A/paso0.phase_done; push "bq10-aislado-k: Paso 0 OK (hash v4 0xC420 + smoke BQ10 0xC4B0 bit a bit)"
fi
run(){ PH=$1; N=$2; [ -f $A/$PH.phase_done ] && return; for t in 1 2 3; do [ $(cnt $PH) -lt $N ] && $P $PH $ARMS $KS; done
  [ $(cnt $PH) -eq $N ] || die "$PH incompleto: $(cnt $PH)/$N"
  for f in $A/$PH/metrics_*.json; do python3 -c "import json,sys;r=json.load(open('$f'));sys.exit(0 if r['test_queries']==0 and r['v4_train']['hash'] and len(r['arms_k'])==4 else 1)" || die "métrica inválida $f"; done
  touch $A/$PH.phase_done; push "bq10-aislado-k: $PH $(cnt $PH)/$N"; }
run bq_dev 16
run bq_confirm 16
python3 scripts/analyze_bq10k.py > $A/analysis.json || die "análisis falló"
python3 scripts/analyze_bq10k.py --md > $D/RESULTADOS.md || die "md falló"
push "bq10-aislado-k: análisis DEV+CONFIRM y RESULTADOS.md"; touch $A/CHAIN_DONE; echo "CHAIN_DONE $(date -Is)"
