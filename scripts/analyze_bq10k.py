#!/usr/bin/env python3
"""Análisis preregistrado BQ10 aislado de v4 + barrido k (docs/bq10_aislado_k/PREREG.md). Bootstrap pareado por semilla 10000, seed 12345, IC95 percentil, δ=0.05.
A (primaria, k=4): BQ10_nov4 − BQ10 (no-inferioridad, margen δ) y BQ10_nov4 − ATLAS_nov4 (M7 sin v4). B: pendiente por semilla de Δ_k(BQ10−CORE) en log2 k (k∈KS), IC bootstrap. Uso: analyze_bq10k.py [--md]"""
import json,glob,sys,numpy as np
D=0.05;NB=10000;R='artifacts/bq10_aislado_k';KS=['2','4','8','12'];ARMS=['v4','ALL','CORE','BQ10','BQ10_nov4','ATLAS_nov4','orbita','persistence']
def boot(x,stat=np.mean):
    x=np.asarray(x,float);assert np.all(np.isfinite(x)),x;r=np.random.default_rng(12345);idx=r.integers(0,len(x),(NB,len(x)))
    b=np.array([stat(x[i]) for i in idx]) if stat is not np.mean else x[idx].mean(1);return [float(stat(x)),float(np.quantile(b,.025)),float(np.quantile(b,.975))]
def load(ph):
    M={};[M.__setitem__(m['seed'],m) for m in (json.load(open(f)) for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json')))];S=sorted(M)
    assert len(S)==16 and all(M[s]['test_queries']==0 for s in S);return M,S
def H(M,s,k,a,key='new_med_log'): return M[s]['arms_k'][k][a]['held'][key]
def phase(ph):
    M,S=load(ph);o={'seeds':S,'by_k':{}}
    for k in KS:
        t={}
        for a in ARMS:
            if a not in M[S[0]]['arms_k'][k]: continue
            x={'means':{q:float(np.mean([H(M,s,k,a,q) for s in S])) for q in ('new_med_log','new_acc16','known_med_log','known_acc16')}}
            if a not in('orbita','persistence'): x['e33_pass']=sum(bool(M[s]['arms_k'][k][a]['e33']['pass']) for s in S);x['nan']=sum(bool(M[s]['arms_k'][k][a]['nan']) for s in S);x['wall_s']=float(np.mean([M[s]['arms_k'][k][a]['wall_s'] for s in S]))
            t[a]=x
        pairs=[('BQ10_nov4','BQ10'),('BQ10_nov4','ATLAS_nov4'),('BQ10','CORE'),('ATLAS_nov4','CORE'),('BQ10_nov4','CORE'),('BQ10_nov4','v4'),('BQ10','v4'),('BQ10_nov4','ALL'),('BQ10','ALL'),('BQ10_nov4','orbita'),('BQ10','orbita'),('BQ10_nov4','persistence'),('BQ10','persistence')]
        t['delta']={f'{a}-{b}':{q:boot([H(M,s,k,a,q)-H(M,s,k,b,q) for s in S]) for q in ('new_med_log','known_acc16')} for a,b in pairs if a in t and b in t}
        o['by_k'][k]=t
    lk=np.log2([int(k) for k in KS])
    for name,(a,b) in {'BQ10-CORE':('BQ10','CORE'),'BQ10_nov4-ATLAS_nov4':('BQ10_nov4','ATLAS_nov4')}.items():
        dk=np.array([[H(M,s,k,a)-H(M,s,k,b) for k in KS] for s in S]);sl=np.array([np.polyfit(lk,d,1)[0] for d in dk])
        o.setdefault('trend',{})[name]={'mean_delta_by_k':dict(zip(KS,map(float,dk.mean(0)))),'slope_per_log2k':boot(sl),'monotone_means':bool(np.all(np.diff(dk.mean(0))>0)),'frac_seeds_monotone':float(np.mean([np.all(np.diff(d)>0) for d in dk]))}
    o['time']={'v4_train_s':float(np.mean([M[s]['v4_train']['train_s'] for s in S])),'BQ10_k4_s':o['by_k']['4']['BQ10']['wall_s'],'BQ10_nov4_k4_s':o['by_k']['4']['BQ10_nov4']['wall_s']}
    return o
def ok_nh(x,ref_d): return ref_d['known_acc16'][1]>-D and x['e33_pass']==16 and x['nan']==0
out={'dev':phase('bq_dev'),'confirm':phase('bq_confirm')};V={}
def both(f): return all(f(out[p]) for p in('dev','confirm'))
d=lambda o,p:o['by_k']['4']['delta'][p]
V['A1_v4_innecesario (BQ10_nov4 no inferior a BQ10: IC sup <+δ, no-harm)']=both(lambda o:d(o,'BQ10_nov4-BQ10')['new_med_log'][2]<D and ok_nh(o['by_k']['4']['BQ10_nov4'],d(o,'BQ10_nov4-BQ10')))
V['A1b_nov4_supera_BQ10 (IC sup<0 y |Δ|≥δ)']=both(lambda o:d(o,'BQ10_nov4-BQ10')['new_med_log'][2]<0 and abs(d(o,'BQ10_nov4-BQ10')['new_med_log'][0])>=D)
V['A2_M7_ayuda_sin_v4 (BQ10_nov4−ATLAS_nov4: IC sup<0, |Δ|≥δ)']=both(lambda o:d(o,'BQ10_nov4-ATLAS_nov4')['new_med_log'][2]<0 and abs(d(o,'BQ10_nov4-ATLAS_nov4')['new_med_log'][0])>=D)
V['A3_BQ10_nov4_supera_CORE (IC sup<0, |Δ|≥δ)']=both(lambda o:d(o,'BQ10_nov4-CORE')['new_med_log'][2]<0 and abs(d(o,'BQ10_nov4-CORE')['new_med_log'][0])>=D)
for n in ('BQ10-CORE','BQ10_nov4-ATLAS_nov4'):
    V[f'B_{n}_se_atenua (pendiente IC inf>0 y medias monótonas)']=both(lambda o:o['trend'][n]['slope_per_log2k'][1]>0 and o['trend'][n]['monotone_means'])
    V[f'B_{n}_refutada (pendiente IC sup<0 en CONFIRM)']=out['confirm']['trend'][n]['slope_per_log2k'][2]<0
out['verdict']=V
if '--md' in sys.argv:
    f3=lambda v:f'{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]';L=['# BQ10 aislado de v4 + barrido k — RESULTADOS','','Preregistro: `docs/bq10_aislado_k/PREREG.md`. Bootstrap pareado por semilla (16), 10000, seed 12345, IC95. δ=0.05. 0 consultas TEST. Generado por `scripts/analyze_bq10k.py`.','']
    for ph in('dev','confirm'):
        o=out[ph];L+=[f'## {ph.upper()}','']
        for k in KS:
            t=o['by_k'][k];L+=[f'### k={k}','','| brazo | new_med_log | new_acc16 | known_acc16 | E33 | NaN | s/semilla |','|---|---|---|---|---|---|---|']
            for a in ARMS:
                if a in t: x=t[a];L.append(f"| {a} | {x['means']['new_med_log']:+.4f} | {x['means']['new_acc16']:.4f} | {x['means']['known_acc16']:.4f} | {str(x.get('e33_pass','—'))+('/16' if 'e33_pass' in x else '')} | {x.get('nan','—')} | {x['wall_s']:.1f} |" if 'wall_s' in x else f"| {a} | {x['means']['new_med_log']:+.4f} | {x['means']['new_acc16']:.4f} | {x['means']['known_acc16']:.4f} | — | — | — |")
            L+=['','| Δ | new_med_log | known_acc16 |','|---|---|---|']+[f'| {p} | {f3(v["new_med_log"])} | {f3(v["known_acc16"])} |' for p,v in t['delta'].items()]+['']
        L+=['### Tendencia en k (Δ por semilla vs log2 k)','','| par | Δ medio k=2/4/8/12 | pendiente / log2k | medias monótonas | frac semillas monótonas |','|---|---|---|---|---|']
        for n,x in o['trend'].items(): L.append(f"| {n} | {' / '.join(f'{v:+.3f}' for v in x['mean_delta_by_k'].values())} | {f3(x['slope_per_log2k'])} | {x['monotone_means']} | {x['frac_seeds_monotone']:.2f} |")
        tm=o['time'];L+=['',f"Tiempo medio por semilla: entrenamiento v4 {tm['v4_train_s']:.0f} s; BQ10 k=4 {tm['BQ10_k4_s']:.1f} s (+entrenamiento); BQ10_nov4 k=4 {tm['BQ10_nov4_k4_s']:.2f} s (sin red).",'']
    L+=['## Veredicto preregistrado','']+[f'- {k}: **{"SÍ" if v else "NO"}**' for k,v in V.items()];print('\n'.join(L))
else: print(json.dumps(out,indent=1))
