"""ESTELA v0 (docs/estela/PREREG.md): bootstrap pareado por seed, 10000 réplicas, semilla 12345, IC95 percentil. δ=0,05.
Uso: python3 scripts/analyze_estela.py   (lee artifacts/estela/{dev,confirm}/metrics_0x*.json)"""
import json,glob,numpy as np
B=10000;DELTA=0.05;R="artifacts/estela"
def load(ph): return [json.load(open(f)) for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json'))]
def boot(x):
    x=np.asarray(x,float);rng=np.random.default_rng(12345);m=x[rng.integers(0,len(x),(B,len(x)))].mean(1)
    return [float(x.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))]
def diff(D,a,b,met,k='k4'): return [d['arms'][k][a][met]-d['arms'][k][b][met] for d in D]
CTRL=['estela_nosig','estela_rec','estela_mlp','estela_r0']
out={}
for ph in ['dev','confirm']:
    D=load(ph)
    if not D: continue
    assert len(D)==16, f'{ph}: {len(D)} seeds'
    assert all(d['test_queries']==0 and d['paso0']['init_vs_orbita_mismatch']==0 and d['paso0']['tape_vs_f64_mismatch']==0 for d in D)
    r={'n':len(D)}
    r['S3_primary_ESTELA-ORBLAT_newlog_k4']=boot(diff(D,'estela','orbita_lat','new_med_log'))
    r['S2_noharm_ESTELA-ORB_knownacc_k4']=boot(diff(D,'estela','orbita','known_acc16'))
    for c in CTRL: r[f'C_ESTELA-{c}_newlog_k4']=boot(diff(D,'estela',c,'new_med_log'))
    for k in ['k2','k8']:
        r[f'X_ESTELA-ORBLAT_newlog_{k}']=boot(diff(D,'estela','orbita_lat','new_med_log',k))
        r[f'X_ESTELA-ORB_knownacc_{k}']=boot(diff(D,'estela','orbita','known_acc16',k))
    r['X_ESTELAbase-ORB_newlog_k4']=boot(diff(D,'estela_base','orbita','new_med_log'))
    er={v:[d['variants'][v]['erank'] for d in D] for v in D[0]['variants']}
    r['S1_erank_mean']={v:float(np.mean(x)) for v,x in er.items()}
    r['S1_erank_ESTELA-NOSIG']=boot(np.array(er['estela'])-np.array(er['estela_nosig']))
    r['S4_auroc_ESTELA-ORB_k4']=boot([d['arms']['k4']['auroc_estela']-d['arms']['k4']['auroc_orbita'] for d in D])
    r['abs_k4']={a:{m:float(np.mean([d['arms']['k4'][a][m] for d in D])) for m in ['new_med_log','known_acc16','new_acc16','known_med_log']} for a in D[0]['arms']['k4'] if not a.startswith('auroc')}
    r['trigger_k4']={m:float(np.mean([d['arms']['k4']['estela'][m] for d in D])) for m in ['trigger_new','trigger_known','accept_new','accept_known']}
    out[ph]=r
V={}
if 'dev' in out and 'confirm' in out:
    dv,cf=out['dev'],out['confirm'];p='S3_primary_ESTELA-ORBLAT_newlog_k4';s='S2_noharm_ESTELA-ORB_knownacc_k4'
    V['S3']='aceptada' if dv[p][2]<0 and cf[p][2]<0 else ('refutada' if cf[p][1]>0 else 'inconclusa')
    V['S2']='aceptada' if dv[s][1]>-DELTA and cf[s][1]>-DELTA else 'falla'
    V['S1']='cumple' if all(o['S1_erank_ESTELA-NOSIG'][1]>0 and o['S1_erank_mean']['estela']>=0.8*8 for o in (dv,cf)) else 'no cumple'
    V['S4']='cumple' if all(o['S4_auroc_ESTELA-ORB_k4'][1]>=-DELTA for o in (dv,cf)) else 'no cumple'
    V['exito']=V['S3']=='aceptada' and V['S2']=='aceptada'
out['veredicto']=V
json.dump(out,open(f'{R}/ANALYSIS.json','w'),indent=1)
for ph,r in out.items():
    if ph=='veredicto': continue
    print('==',ph,'n',r['n'])
    for k,v in r.items():
        if isinstance(v,list) and len(v)==3: print(f'{k:40s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]')
        elif isinstance(v,dict): print(k,json.dumps(v)[:600])
print('veredicto',V)
