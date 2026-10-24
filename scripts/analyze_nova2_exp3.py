"""Nova-2 exp3 (docs/orbita_nova2/PREREG_EXP3.md): bootstrap pareado por seed, 10000 réplicas, semilla 12345, IC95. δ=0.05."""
import json,glob,numpy as np
B=10000;DELTA=0.05;R="artifacts/orbita/nova2_exp3"
def load(ph): return [json.load(open(f)) for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json'))]
def ck(d,n): return [c for c in d['ckpts'] if c['n']==n][0]
def boot(x):
    x=np.asarray(x,float);rng=np.random.default_rng(12345);m=x[rng.integers(0,len(x),(B,len(x)))].mean(1)
    return [float(x.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))]
def diff(D,a,b,met,k): return [ck(d,2400)[k][a][met]-ck(d,2400)[k][b][met] for d in D]
def det(D,a,b,k): return [ck(d,2400)[k]['detect'][a]-ck(d,2400)[k]['detect'][b] for d in D]
out={}
for ph in ['dev','confirm']:
    D=load(ph)
    if not D: continue
    r={'n':len(D)}
    for k in ['k2','k4','k8']:
        r[f'F1F2_LATBOCD-LATEP_newlog_{k}']=boot(diff(D,'lat_bocd','lat_ep','new_med_log',k))
        r[f'F2_NOVABOCD-NOVA_newlog_{k}']=boot(diff(D,'nova_bocd','nova','new_med_log',k))
        r[f'F3_LATBOCD-LATEP_knownacc_{k}']=boot(diff(D,'lat_bocd','lat_ep','known_acc16',k))
        r[f'F3_NOVABOCD-NOVA_knownacc_{k}']=boot(diff(D,'nova_bocd','nova','known_acc16',k))
        r[f'F4_bacc_bocd-tau_{k}']=boot(det(D,'bacc_bocd','bacc_tau',k))
        r[f'F5_LATBOCD-SHUF_newlog_{k}']=boot(diff(D,'lat_bocd','lat_bocd_shuf','new_med_log',k))
        r[f'NOVA-LATEP_{k}']=boot(diff(D,'nova','lat_ep','new_med_log',k))
        r[f'NOVA-ORB_{k}']=boot(diff(D,'nova','orbita','new_med_log',k))
        r[f'NOVA-NOVAmC3_{k}']=boot(diff(D,'nova','nova_m_c3','new_med_log',k))
        r[f'detect_{k}']={m:float(np.mean([ck(d,2400)[k]['detect'][m] for d in D])) for m in ck(D[0],2400)[k]['detect']}
        r[f'abs_{k}']={a:[float(np.mean([ck(d,2400)[k][a][m] for d in D])) for m in ['new_med_log','known_acc16','new_acc16']] for a in D[0]['arms']}
    r['stream']={x:float(np.mean([ck(d,2400)[x] for d in D])) for x in ['lib_size','stream_novel','stream_novel_on_known','stream_novel_on_new','stream_known','stream_new','stream_accept_nova']}
    r['ref_kappa']=float(np.mean([ck(d,2400)['ref_ng']['kappa'] for d in D]))
    out[ph]=r
F={}
if 'dev' in out and 'confirm' in out:
    dv,cf=out['dev'],out['confirm']
    k='F1F2_LATBOCD-LATEP_newlog_k2'
    F['F1']='aceptada' if dv[k][2]<0 and cf[k][2]<0 else ('refutada' if cf[k][1]>0 else 'inconclusa')
    F['F3']='aceptada' if all(o[x][1]>-DELTA for o in (dv,cf) for x in o if x.startswith('F3_')) else 'falla'
    F['exito']=F['F1']=='aceptada' and F['F3']=='aceptada'
out['veredicto']=F
json.dump(out,open(f'{R}/ANALYSIS.json','w'),indent=1)
for ph,r in out.items():
    if ph=='veredicto': continue
    print('==',ph,'n',r['n'],'stream',r['stream'],'ref_kappa',r['ref_kappa'])
    for k,v in r.items():
        if isinstance(v,list) and len(v)==3: print(f'{k:34s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]')
    for k in ['k2','k4','k8']:
        print(k,{m:round(x,3) for m,x in r[f'detect_{k}'].items()});[print(f'  {a:14s}',' '.join(f'{x:+.3f}' for x in v)) for a,v in r[f'abs_{k}'].items()]
print('veredicto',F)
