"""Nova-2 H2 NOVA_MIX (docs/orbita_nova2/PREREG_H2.md): bootstrap pareado por seed, 10000 réplicas, semilla 12345, IC95. δ=0.05."""
import json,glob,numpy as np
B=10000;DELTA=0.05;R="artifacts/orbita/nova2_h2"
def load(ph): return [json.load(open(f)) for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json'))]
def ck(d,n=2400): return [c for c in d['ckpts'] if c['n']==n][0]
def boot(x):
    x=np.asarray(x,float);rng=np.random.default_rng(12345);m=x[rng.integers(0,len(x),(B,len(x)))].mean(1)
    return [float(x.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))]
def diff(D,a,b,met,k): return [ck(d)[k][a][met]-ck(d)[k][b][met] for d in D]
out={}
for ph in ['dev','confirm']:
    D=load(ph)
    if not D: continue
    r={'n':len(D),'lib_size':float(np.mean([ck(d)['lib_size'] for d in D]))}
    for k in ['k2','k4','k8']:
        r[f'H2_MIX-NOVA_newlog_{k}']=boot(diff(D,'nova_mix','nova','new_med_log',k))
        r[f'CTRL_MIX-SHUF_newlog_{k}']=boot(diff(D,'nova_mix','nova_mix_shuf','new_med_log',k))
        r[f'CTRL_MIX-ORBC3_newlog_{k}']=boot(diff(D,'nova_mix','orbita_c3','new_med_log',k))
        r[f'MIX-RET_newlog_{k}']=boot(diff(D,'nova_mix','nova_ret','new_med_log',k))
        r[f'NOHARM_MIX-NOVA_knownacc_{k}']=boot(diff(D,'nova_mix','nova','known_acc16',k))
        r[f'abs_{k}']={a:[float(np.mean([ck(d)[k][a][m] for d in D])) for m in ['new_med_log','known_med_log','known_acc16','new_acc16']] for a in D[0]['arms']}
        dg=ck(D[0])[k]['diag']
        r[f'diag_{k}']={m:float(np.mean([ck(d)[k]['diag'][m] for d in D])) for m in dg if not isinstance(dg[m],dict)}
        for wn in ['w_mix','w_mixshuf']:
            r[f'{wn}_{k}']={m:(np.mean([ck(d)[k]['diag'][wn][m] for d in D],axis=0).round(4).tolist()) for m in dg[wn]}
    out[ph]=r
F={}
if 'dev' in out and 'confirm' in out:
    dv,cf=out['dev'],out['confirm'];k='H2_MIX-NOVA_newlog_k2'
    F['H2']='aceptada' if dv[k][2]<0 and cf[k][2]<0 else ('refutada' if cf[k][1]>0 else 'inconclusa')
    F['no_dano']='aceptada' if all(o[x][1]>-DELTA for o in (dv,cf) for x in o if x.startswith('NOHARM_')) else 'falla'
    F['atribuible_biblioteca']=bool(all(o['CTRL_MIX-SHUF_newlog_k2'][2]<0 for o in (dv,cf)))
    F['mejora_sobre_orbc3']=bool(all(o['CTRL_MIX-ORBC3_newlog_k2'][2]<0 for o in (dv,cf)))
    F['exito']=F['H2']=='aceptada' and F['no_dano']=='aceptada'
out['veredicto']=F
json.dump(out,open(f'{R}/ANALYSIS.json','w'),indent=1)
for ph,r in out.items():
    if ph=='veredicto': continue
    print('==',ph,'n',r['n'],'lib',r['lib_size'])
    for k,v in r.items():
        if isinstance(v,list) and len(v)==3: print(f'{k:34s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]')
    for k in ['k2','k4','k8']:
        print(k,{m:round(x,3) for m,x in r[f'diag_{k}'].items()});print('  w_mix',r[f'w_mix_{k}']);print('  w_shuf',r[f'w_mixshuf_{k}'])
        [print(f'  {a:14s}',' '.join(f'{x:+.3f}' for x in v)) for a,v in r[f'abs_{k}'].items()]
print('veredicto',F)
