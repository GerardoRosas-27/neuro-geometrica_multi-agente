"""Nova-2 exp2 (docs/orbita_nova2/PREREG_EXP2.md): bootstrap pareado por seed, 10000 réplicas, semilla 12345, IC95."""
import json,glob,numpy as np
B=10000;DELTA=0.05;R='artifacts/orbita/nova2_exp2'
def load(ph): return [json.load(open(f)) for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json'))]
def ck(d,n): return [c for c in d['ckpts'] if c['n']==n][0]
def boot(x):
    x=np.asarray(x);rng=np.random.default_rng(12345);m=x[rng.integers(0,len(x),(B,len(x)))].mean(1)
    return [float(x.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))]
def diff(D,a,b,met,k,na=2400,nb=2400): return [ck(d,na)[k][a][met]-ck(d,nb)[k][b][met] for d in D]
out={}
for ph in ['dev','confirm']:
    D=load(ph)
    if not D: continue
    r={'n':len(D)}
    for k in ['k2','k4','k8']:
        r[f'E1_{k}']=boot(diff(D,'nova_base','nova','new_med_log',k))
        r[f'E3_{k}']=boot(diff(D,'nova_base','nova','known_acc16',k))
        r[f'E4_{k}']=boot(diff(D,'nova_base','nova_base','new_med_log',k,nb=0))
        r[f'E5log_{k}']=boot(diff(D,'nova_base','nova_base_m_sparse','new_med_log',k))
        r[f'E5acc_{k}']=boot(diff(D,'nova_base','nova_base_m_sparse','known_acc16',k))
        r[f'NOVA-NOVAmC3_{k}']=boot(diff(D,'nova','nova_m_c3','new_med_log',k))
        r[f'NOVA-LATEP_{k}']=boot(diff(D,'nova','lat_ep','new_med_log',k))
        r[f'NOVA-ORB_{k}']=boot(diff(D,'nova','orbita','new_med_log',k))
        r[f'abs_{k}']={a:[float(np.mean([ck(d,2400)[k][a][m] for d in D])) for m in ['new_med_log','known_acc16','new_acc16']] for a in D[0]['arms']}
        r[f'ident_{k}']={p:int(sum(ck(d,2400)[k][p.split('=')[0]]==ck(d,2400)[k][p.split('=')[1]] for d in D)) for p in ['nova=nova_m_c3','lat_ep=orbita','nova_base=nova','nova_base=nova_base_m_sparse']}
    r['lib']=[float(np.mean([ck(d,2400)[x] for d in D])) for x in ['lib_size','pool_size','pool_known','base_sparse_n','base_full_n']]
    r['accept_rate']=[float(np.mean([ck(d,2400)['stream_accept_nova']/2400 for d in D]))]
    out[ph]=r
json.dump(out,open(f'{R}/ANALYSIS.json','w'),indent=1)
for ph,r in out.items():
    print('==',ph,'n',r['n'],'lib',r['lib'],'acc',r['accept_rate'])
    for k,v in r.items():
        if isinstance(v,list) and len(v)==3 and k[0] in 'EN': print(f'{k:20s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]')
    for k in ['k2','k4','k8']:
        print(k,r[f'ident_{k}']);[print(f'  {a:16s}',' '.join(f'{x:+.3f}' for x in v)) for a,v in r[f'abs_{k}'].items()]
