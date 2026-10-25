"""Nova-2 H1 NOVA_RET (docs/orbita_nova2/PREREG_H1.md): bootstrap pareado por seed, 10000 réplicas, semilla 12345, IC95. δ=0.05."""
import json,glob,numpy as np
B=10000;DELTA=0.05;R="artifacts/orbita/nova2_h1"
def load(ph): return [json.load(open(f)) for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json'))]
def ck(d,n): return [c for c in d['ckpts'] if c['n']==n][0]
def boot(x):
    x=np.asarray(x,float);rng=np.random.default_rng(12345);m=x[rng.integers(0,len(x),(B,len(x)))].mean(1)
    return [float(x.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))]
def diff(D,a,b,met,k,n=2400): return [ck(d,n)[k][a][met]-ck(d,n)[k][b][met] for d in D]
out={}
for ph in ['dev','confirm']:
    D=load(ph)
    if not D: continue
    r={'n':len(D)}
    for k in ['k2','k4','k8']:
        r[f'H1_RET-NOVA_newlog_{k}']=boot(diff(D,'nova_ret','nova','new_med_log',k))
        r[f'CTRL_RET-SHUF_newlog_{k}']=boot(diff(D,'nova_ret','nova_ret_shuf','new_med_log',k))
        r[f'CTRL_RET-ORBC3_newlog_{k}']=boot(diff(D,'nova_ret','orbita_c3','new_med_log',k))
        r[f'NOHARM_RET-NOVA_knownacc_{k}']=boot(diff(D,'nova_ret','nova','known_acc16',k))
        r[f'NOVA-ORBC3_newlog_{k}']=boot(diff(D,'nova','orbita_c3','new_med_log',k))
        r[f'abs_{k}']={a:[float(np.mean([ck(d,2400)[k][a][m] for d in D])) for m in ['new_med_log','known_med_log','known_acc16','new_acc16']] for a in D[0]['arms']}
        r[f'diag_{k}']={m:float(np.mean([ck(d,2400)[k]['diag'][m] for d in D])) for m in ck(D[0],2400)[k]['diag']}
    r['N2_RET2400-RET150_newlog_k2']=boot([ck(d,2400)['k2']['nova_ret']['new_med_log']-ck(d,150)['k2']['nova_ret']['new_med_log'] for d in D])
    r['lib_size']=float(np.mean([ck(d,2400)['lib_size'] for d in D]))
    out[ph]=r
F={}
if 'dev' in out and 'confirm' in out:
    dv,cf=out['dev'],out['confirm'];k='H1_RET-NOVA_newlog_k2'
    F['H1']='aceptada' if dv[k][2]<0 and cf[k][2]<0 else ('refutada' if cf[k][1]>0 else 'inconclusa')
    F['no_dano']='aceptada' if all(o[x][1]>-DELTA for o in (dv,cf) for x in o if x.startswith('NOHARM_')) else 'falla'
    F['atribuible_biblioteca']=bool(dv['CTRL_RET-SHUF_newlog_k2'][2]<0 and cf['CTRL_RET-SHUF_newlog_k2'][2]<0)
    F['exito']=F['H1']=='aceptada' and F['no_dano']=='aceptada'
out['veredicto']=F
json.dump(out,open(f'{R}/ANALYSIS.json','w'),indent=1)
for ph,r in out.items():
    if ph=='veredicto': continue
    print('==',ph,'n',r['n'],'lib',r['lib_size'])
    for k,v in r.items():
        if isinstance(v,list) and len(v)==3: print(f'{k:34s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]')
    for k in ['k2','k4','k8']:
        print(k,{m:round(x,3) for m,x in r[f'diag_{k}'].items()});[print(f'  {a:14s}',' '.join(f'{x:+.3f}' for x in v)) for a,v in r[f'abs_{k}'].items()]
print('veredicto',F)
