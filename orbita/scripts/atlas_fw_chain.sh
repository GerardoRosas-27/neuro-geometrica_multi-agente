#!/usr/bin/env bash
# Cadena ATLAS-FW: DEV → gate → CONFIRM (solo brazos que pasan + v4) → RESULTADOS.md → commit+push a exp/atlas-fw.
# Veredicto SOLO si 16/16 semillas válidas en cada fase; si no: CHAIN_STOP_ERROR (sin veredicto).
# Termina siempre con CHAIN_DONE, CHAIN_STOP o CHAIN_STOP_ERROR.
set -uo pipefail
cd "$(dirname "$0")/../.."; R=artifacts/atlas_fw; A=scripts/analyze_atlas_fw.py
echo "CHAIN_START $(date -Is) pid=$$ sha=$(git rev-parse --short HEAD)"
check() { # fase brazos
  python3 - "$1" "$2" <<'PY'
import json,sys,os,math
ph,arms=sys.argv[1],sys.argv[2].split(',');P='0xA48' if ph=='dev' else '0xB48';bad=[]
for i in '0123456789ABCDEF':
    s=P+i;d=f'artifacts/atlas_fw/{ph}'
    try:
        assert os.path.exists(f'{d}/{s}.done'),'sin .done'
        assert open(f'{d}/rc_{s}.txt').read().strip()=='0','rc!=0'
        assert 'panicked' not in open(f'{d}/log_{s}.txt').read(),'panic'
        m=json.load(open(f'{d}/metrics_{s}.json'));assert m['test_queries']==0 and m['seed']==s and m['phase']==ph,'meta'
        for a in arms:
            x=m['arms'][a]['held'];assert all(math.isfinite(v) for v in x.values()),f'NaN {a}'
    except Exception as e: bad.append(f'{s}:{e}')
print('OK' if not bad else 'BAD '+' '.join(bad))
PY
}
publish() {
  python3 $A "$@" --md > $R/RESULTADOS.md && python3 $A "$@" > $R/resultados.json
  git add -f $R/RESULTADOS.md $R/resultados.json $R/*/metrics_0x*.json && git commit -qm "ATLAS-FW: RESULTADOS ($1)" && git push -q origin exp/atlas-fw && echo "PUSH_OK $(git rev-parse --short HEAD)" || echo "PUSH_FAIL"
}
ALLARMS=v4,M1,M2,M3,M4,M5,M6,ALL
bash orbita/scripts/atlas_fw.sh dev $ALLARMS; c=$(check dev $ALLARMS); echo "DEV_CHECK $c"
if [ "$c" != OK ]; then echo "CHAIN_STOP_ERROR dev $(date -Is)"; exit 1; fi
G=$(python3 $A dev --gate); echo "DEV_GATE [$G]"
if [ -z "$G" ]; then publish dev; echo "CHAIN_STOP ningún brazo pasa gate DEV $(date -Is)"; exit 0; fi
CA=v4,${G// /,}; bash orbita/scripts/atlas_fw.sh confirm $CA; c=$(check confirm $CA); echo "CONFIRM_CHECK $c"
if [ "$c" != OK ]; then echo "CHAIN_STOP_ERROR confirm $(date -Is)"; exit 1; fi
publish dev confirm; echo "CHAIN_DONE $(date -Is)"
