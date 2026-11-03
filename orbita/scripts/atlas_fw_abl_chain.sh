#!/usr/bin/env bash
# Cadena ablaciones ATLAS-FW: ABL_DEV (todos los brazos) → gate (ablaciones significativas) → ABL_CONFIRM (significativas + ALL + v4)
# → RESULTADOS_ABLACIONES.md → commit+push a exp/atlas-fw. Veredicto SOLO con 16/16 semillas válidas; si no: CHAIN_STOP_ERROR.
# Termina siempre con CHAIN_DONE, CHAIN_STOP o CHAIN_STOP_ERROR.
set -uo pipefail
cd "$(dirname "$0")/../.."; R=artifacts/atlas_fw/ablaciones; A=scripts/analyze_atlas_fw_abl.py
[ "$(git rev-parse --abbrev-ref HEAD)" = exp/atlas-fw ] || { echo "CHAIN_STOP_ERROR rama != exp/atlas-fw $(date -Is)"; exit 1; }
echo "CHAIN_START $(date -Is) pid=$$ sha=$(git rev-parse --short HEAD)"
check() { # fase brazos
  python3 - "$1" "$2" <<'PY'
import json,sys,os,math
ph,arms=sys.argv[1],sys.argv[2].split(',');P='0xA49' if ph=='abl_dev' else '0xB49';bad=[]
for i in '0123456789ABCDEF':
    s=P+i;d=f'artifacts/atlas_fw/ablaciones/{ph}'
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
  python3 $A "$@" --md > $R/RESULTADOS_ABLACIONES.md && python3 $A "$@" > $R/resultados_ablaciones.json
  git add -f $R/RESULTADOS_ABLACIONES.md $R/resultados_ablaciones.json $R/*/metrics_0x*.json $R/*/log_0x*.txt $R/*/rc_0x*.txt $R/*/*.done $R/chain.log && git commit -qm "ATLAS-FW: RESULTADOS ABLACIONES ($*)" && git push -q origin exp/atlas-fw && echo "PUSH_OK $(git rev-parse --short HEAD)" || echo "PUSH_FAIL"
}
DA=v4,ALL,ALL-M1,ALL-M2,ALL-M3,ALL-M4,ALL-M5,ALL-M6
bash orbita/scripts/atlas_fw_abl.sh abl_dev $DA; c=$(check abl_dev $DA); echo "DEV_CHECK $c"
if [ "$c" != OK ]; then echo "CHAIN_STOP_ERROR abl_dev $(date -Is)"; exit 1; fi
G=$(python3 $A abl_dev --gate); echo "DEV_GATE [$G]"
if [ -z "$G" ]; then publish abl_dev; echo "CHAIN_STOP ninguna ablación significativa en DEV $(date -Is)"; exit 0; fi
CA=v4,ALL,${G// /,}; bash orbita/scripts/atlas_fw_abl.sh abl_confirm $CA; c=$(check abl_confirm $CA); echo "CONFIRM_CHECK $c"
if [ "$c" != OK ]; then echo "CHAIN_STOP_ERROR abl_confirm $(date -Is)"; exit 1; fi
publish abl_dev abl_confirm; echo "CHAIN_DONE $(date -Is)"
