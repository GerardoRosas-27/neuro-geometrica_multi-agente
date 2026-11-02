#!/usr/bin/env bash
# Cadena BIO-CONSTELACIÓN: DEV → CONFIRM → análisis → RESULTADOS.md. Reanudable; aborta si falta binario o fase incompleta.
set -uo pipefail
cd /workspace/ngm-bio; A=artifacts/bio_constelacion; B=orbita/target/release/orbita; P=orbita/scripts/bio_phase.sh
ARMS="hyb hyb_nobio ref v4"; N=64
die(){ echo "ABORT: $*"; exit 1; }
[ -x $B ] || die "falta binario $B (recompilar: cd orbita && nice -n 10 cargo build --release)"
cnt(){ ls $A/$1/*.done 2>/dev/null | wc -l; }
for PH in dev confirm; do
  for t in 1 2 3; do [ $(cnt $PH) -lt $N ] && $P $PH "$ARMS"; done
  [ $(cnt $PH) -eq $N ] || die "$PH incompleto: $(cnt $PH)/$N"
  for f in $A/$PH/metrics_*.json; do python3 -c "import json,sys;r=json.load(open('$f'));sys.exit(0 if r.get('held') or r.get('orbita') else 1)" || die "métrica vacía $f"; done
  git add $A/$PH && git commit -qm "bio-constelacion: $PH $(cnt $PH)/$N" && git push -q origin HEAD:exp/hibrido-constelacion-neocortex
  touch $A/$PH.phase_done
done
python3 scripts/analyze_bio_constelacion.py dev confirm > $A/analysis.json || die "análisis falló"
python3 scripts/write_bio_resultados.py || die "RESULTADOS falló"
git add $A docs/bio_constelacion && git commit -qm "bio-constelacion: análisis DEV+CONFIRM y RESULTADOS.md" && git push -q origin HEAD:exp/hibrido-constelacion-neocortex
touch $A/CHAIN_DONE; echo CHAIN_DONE
