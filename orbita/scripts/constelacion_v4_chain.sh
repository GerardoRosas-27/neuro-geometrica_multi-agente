#!/bin/bash
cd /workspace/ngm-constelacion-v4
while pgrep -f "constelacion_v4.sh dev" >/dev/null; do sleep 60; done
for t in 1 2; do [ $(ls artifacts/constelacion_v4/dev/*.done 2>/dev/null | wc -l) -lt 16 ] && orbita/scripts/constelacion_v4.sh dev; done
git add artifacts/constelacion_v4/dev && git commit -qm "constelacion-v4: DEV $(ls artifacts/constelacion_v4/dev/*.done | wc -l)/16" && git push -q
for t in 1 2 3; do [ $(ls artifacts/constelacion_v4/confirm/*.done 2>/dev/null | wc -l) -lt 16 ] && orbita/scripts/constelacion_v4.sh confirm; done
python3 scripts/analyze_constelacion_v4.py > artifacts/constelacion_v4/analysis_dev_confirm.json
git add artifacts/constelacion_v4 && git commit -qm "constelacion-v4: CONFIRM $(ls artifacts/constelacion_v4/confirm/*.done | wc -l)/16 + análisis" && git push -q
echo CHAIN_DONE
