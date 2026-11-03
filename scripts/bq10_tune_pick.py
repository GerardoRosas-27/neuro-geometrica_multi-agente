#!/usr/bin/env python3
"""Regla de selección congelada (PREREG): sobre bq_tune (0xC4B0–C4B3), entre BQ{3,10,30} y BQR{3,10,30}, elegir el brazo con menor media de
new_med_log sujeto a: media known_acc16 ≥ media v4 − 0.05, E33 pass en las 4 semillas, 0 NaN. Empate → menor da, sin M8. Ninguno válido → CORE.
Escribe docs/bio_quantum_v10/frozen.json. 0 consultas TEST."""
import json,glob,numpy as np
M=[json.load(open(f)) for f in sorted(glob.glob('artifacts/bio_quantum_v10/bq_tune/metrics_0x*.json'))];assert len(M)==4 and all(m['test_queries']==0 for m in M)
g=lambda a,k:float(np.mean([m['arms'][a]['held'][k] for m in M]))
C=['BQ3','BQ10','BQ30','BQR3','BQR10','BQR30'];tab={a:{'new_med_log':g(a,'new_med_log'),'known_acc16':g(a,'known_acc16'),'e33_pass':sum(bool(m['arms'][a]['e33']['pass']) for m in M),'nan':sum(bool(m['arms'][a]['nan']) for m in M)} for a in C+['v4','CORE']}
ok=[a for a in C if tab[a]['known_acc16']>=tab['v4']['known_acc16']-0.05 and tab[a]['e33_pass']==4 and tab[a]['nan']==0]
P=min(ok,key=lambda a:(round(tab[a]['new_med_log'],12),C.index(a))) if ok else 'CORE'
sis=({'BQ':'BQR','BQR':'BQ'}[P.rstrip('0123456789')]+P.lstrip('BQR')) if P!='CORE' else None
arms=['v4','ALL','CORE',P]+([sis] if sis else []);arms=list(dict.fromkeys(arms))
json.dump({'tune_seeds':['0xC4B0','0xC4B1','0xC4B2','0xC4B3'],'tune_table':tab,'valid':ok,'primary_arm':P,'sister_arm':sis,'atlas_ref_arm':'ALL','eval_arms':arms,
 'constants':{'RB_K':2.5,'LIB_DEG':[1,1,2,2,2,3,3,3,3,1,1,2,2,1,1,2,2],'N_LIB':17,'LIB_S':0.3,'LAMS':[1e-3,1e-2,1e-1],'THRS':[0.0,0.05]},
 'dev_seeds':'0xA400-0xA40F','confirm_seeds':'0xB400-0xB40F','bootstrap':{'n':10000,'seed':12345,'delta':0.05}},open('docs/bio_quantum_v10/frozen.json','w'),indent=1)
print(','.join(arms))
