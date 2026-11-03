#!/usr/bin/env bash
# Cadena Bio-Quantum v10: [toolchain/binario] → TUNE (0xC4B0–3) → frozen.json commit+push → DEV → CONFIRM → análisis → RESULTADOS.md.
# Reanudable (.done por semilla, .phase_done por fase). Aborta si falta binario/cargo o fase incompleta. Push sólo a exp/bio-quantum-v10-rust (sin force).
set -uo pipefail
cd /workspace/ngm-bq10; BR=exp/bio-quantum-v10-rust; A=artifacts/bio_quantum_v10; B=orbita/target/release/orbita; P=orbita/scripts/bq10_phase.sh
die(){ echo "ABORT $(date -Is): $*"; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BR" ] || die "rama incorrecta"
export PATH=$HOME/.cargo/bin:$PATH; command -v cargo >/dev/null || die "falta toolchain Rust (instalar rustup y: cd orbita && nice -n 10 cargo build --release)"
[ -x $B ] || die "falta binario $B (cd orbita && nice -n 10 cargo build --release)"
cnt(){ ls $A/$1/*.done 2>/dev/null | wc -l; }
push(){ git add -A $A docs/bio_quantum_v10 && git commit -qm "$1" && git push -q origin HEAD:$BR || die "push falló"; }
run(){ PH=$1; N=$2; ARMS=$3; [ -f $A/$PH.phase_done ] && return; for t in 1 2 3; do [ $(cnt $PH) -lt $N ] && $P $PH "$ARMS"; done
  [ $(cnt $PH) -eq $N ] || die "$PH incompleto: $(cnt $PH)/$N"
  for f in $A/$PH/metrics_*.json; do python3 -c "import json,sys;r=json.load(open('$f'));sys.exit(0 if r['test_queries']==0 and r['v4_train']['hash'] else 1)" || die "métrica inválida $f"; done
  touch $A/$PH.phase_done; push "bio-quantum-v10: $PH $(cnt $PH)/$N"; }
echo "START $(date -Is)"
run bq_tune 4 v4,CORE,BQ3,BQ10,BQ30,BQR3,BQR10,BQR30
if [ ! -f docs/bio_quantum_v10/frozen.json ]; then python3 scripts/bq10_tune_pick.py || die "pick falló"; push "bio-quantum-v10: frozen.json (selección en 0xC4B0–C4B3, antes de DEV)"; fi
ARMS=$(python3 -c "import json;print(','.join(json.load(open('docs/bio_quantum_v10/frozen.json'))['eval_arms']))"); echo "ARMS=$ARMS"
run bq_dev 16 $ARMS
run bq_confirm 16 $ARMS
python3 scripts/analyze_bq10.py > $A/analysis.json || die "análisis falló"
python3 scripts/analyze_bq10.py --md > docs/bio_quantum_v10/RESULTADOS.md || die "md falló"
push "bio-quantum-v10: análisis DEV+CONFIRM y RESULTADOS.md"; touch $A/CHAIN_DONE; echo "CHAIN_DONE $(date -Is)"
