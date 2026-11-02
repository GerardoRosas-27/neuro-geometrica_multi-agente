#!/usr/bin/env python3
# Grilla fija de ajuste (sólo 0xC420–C423, N_A completo, 0 consultas TEST). Parada: grilla agotada; elección: min mean_new del brazo hyb
# (desempate: mean_known). El brazo hyb_nobio usa la misma config (ablación limpia).
import json, itertools, subprocess
from concurrent.futures import ThreadPoolExecutor
W='artifacts/orbita/dev/weights_0xA400.json'; O='docs/bio_constelacion/tuning/grid.jsonl'
grid=[dict(eta_p=ep,eta_f=ef,t_f=128,cons=co,stp_u=0.2,stp_tau=4,rho=rho,omega=0.3) for ef,ep,co,rho in itertools.product([0.1,0.5,1.0],[0.001,0.01],[0.0,0.05,0.2],[0.5,0.9])]
def go(c): return subprocess.run(['nice','-n','10','orbita/target/release/orbita','bio-tune','--seeds','0xC420,0xC421,0xC422,0xC423','--weights',W,'--cfg',json.dumps(c)],capture_output=True,text=True,check=True).stdout
with ThreadPoolExecutor(3) as ex: outs=list(ex.map(go,grid))
rows=[json.loads(l) for o in outs for l in o.splitlines()]
open(O,'w').write(''.join(json.dumps(r)+'\n' for r in rows))
h=[r for r in rows if r['bio']]; best=min(h,key=lambda r:(r['mean_new'],r['mean_known']))
print(json.dumps(best))
