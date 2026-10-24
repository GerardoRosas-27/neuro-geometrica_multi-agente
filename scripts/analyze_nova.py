"""Análisis ÓRBITA-Nova según PREREG.md (f64b289). Bootstrap pareado por seed, 10000 réplicas, semilla 12345, IC95.
Métrica agregada por seed = mediana por seed de log(rel_err h16) (new_med_log / known_med_log) y acc16 por seed."""
import json,glob,sys,numpy as np
B=10000;DELTA=0.05
def load(ph):
    return [json.load(open(f)) for f in sorted(glob.glob(f'artifacts/orbita/nova_full/{ph}/metrics_0x*.json'))]
def ck(d,n): return [c for c in d['ckpts'] if c['n']==n][0]
def boot(x):
    x=np.asarray(x);rng=np.random.default_rng(12345)
    idx=rng.integers(0,len(x),(B,len(x)));m=x[idx].mean(1)
    return x.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
def diff(D,a,b,met,k='k4',na=2400,nb=2400):
    return [ck(d,na)[k][a][met]-ck(d,nb)[k][b][met] for d in D]
out={}
for ph in ['dev','confirm']:
    D=load(ph);r={}
    r['N1']=boot(diff(D,'nova','lat_ep','new_med_log'))
    r['N2']=boot(diff(D,'nova','nova','new_med_log',nb=0))
    r['N3a']=boot(diff(D,'nova','orbita','known_acc16'))
    r['N3b']=boot(diff(D,'nova','nova','known_acc16',nb=150))
    r['N4_shuf']=boot(diff(D,'nova','nova_shuf','new_med_log'))
    r['N4_mlp']=boot(diff(D,'nova','mlp','new_med_log'))
    r['N5_c5']=boot(diff(D,'nova','nova_m_c5','new_med_log'))
    r['N5_c6']=boot(diff(D,'nova','nova_m_c6','new_med_log'))
    r['NvsORB_new']=boot(diff(D,'nova','orbita','new_med_log'))
    r['NvsLAT_new']=boot(diff(D,'nova','lat','new_med_log'))
    r['NvsORBc3_known']=boot(diff(D,'nova','orbita_c3','known_acc16'))
    for k in ['k2','k8']:
        r['N1_'+k]=boot(diff(D,'nova','lat_ep','new_med_log',k=k)); r['N4mlp_'+k]=boot(diff(D,'nova','mlp','new_med_log',k=k))
    arms=D[0]['arms']
    r['abs']={a:(float(np.mean([ck(d,2400)['k4'][a]['new_med_log'] for d in D])),float(np.mean([ck(d,2400)['k4'][a]['known_acc16'] for d in D])),float(np.mean([ck(d,2400)['k4'][a]['new_acc16'] for d in D]))) for a in arms}
    r['nova_curve']={n:float(np.mean([ck(d,n)['k4']['nova']['new_med_log'] for d in D])) for n in [0,150,300,600,1200,2400]}
    r['lib']=float(np.mean([d['lib_size_final'] for d in D])); r['tau']=sorted({d['tau'] for d in D}); r['n']=len(D)
    r['ident']={'='.join(p):sum(ck(d,2400)['k4'][p[0]]==ck(d,2400)['k4'][p[1]] for d in D) for p in [('lat','lat_ep'),('lat_ep','orbita'),('nova','nova_m_c3')]}
    out[ph]=r
json.dump(out,open('artifacts/orbita/nova_full/ANALYSIS.json','w'),indent=1,default=str)
for ph,r in out.items():
    print('==',ph,r['n'],'tau',r['tau'],'lib',r['lib'],'ident',r['ident'])
    for k,v in r.items():
        if isinstance(v,tuple): print(f'{k:16s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]')
    print('curve',r['nova_curve'])
    for a,v in r['abs'].items(): print(f'  {a:10s} new_med_log {v[0]:+.3f} known_acc16 {v[1]:.3f} new_acc16 {v[2]:.3f}')
