#!/usr/bin/env python3
"""Análisis preregistrado Bio-Quantum v10 (docs/bio_quantum_v10/PREREG.md). Bootstrap pareado por semilla 10000, seed 12345, IC95 percentil, δ=0.05.
Primaria: frozen.primary_arm − v4 en new_med_log. Co-primaria declarada: primary − ALL (ATLAS-FW de main). Uso: analyze_bq10.py [--md]"""
import json,glob,sys,numpy as np
D=0.05;NB=10000;R='artifacts/bio_quantum_v10';F=json.load(open('docs/bio_quantum_v10/frozen.json'));P=F['primary_arm']
def ci(x):
    x=np.asarray(x,float);assert np.all(np.isfinite(x));r=np.random.default_rng(12345);b=x[r.integers(0,len(x),(NB,len(x)))].mean(1)
    return [float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))]
def phase(ph):
    M={};[M.__setitem__(m['seed'],m) for m in (json.load(open(f)) for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json')))]
    S=sorted(M);assert len(S)==16 and all(M[s]['test_queries']==0 for s in S);g=lambda a,s:M[s]['arms'][a];o={'seeds':S}
    for a in F['eval_arms']:
        x={'means':{k:float(np.mean([g(a,s)['held'][k] for s in S])) for k in ('new_med_log','new_acc16','known_med_log','known_acc16')},'e33_pass':sum(bool(g(a,s)['e33']['pass']) for s in S),'nan':sum(bool(g(a,s)['nan']) for s in S),
           'v4_hash':sorted({M[s]['v4_train']['hash'] for s in S})[:3]}
        for ref in ('v4','ALL'):
            if a!=ref: x['vs_'+ref]={k:ci([g(a,s)['held'][k]-g(ref,s)['held'][k] for s in S]) for k in ('new_med_log','known_acc16','new_acc16')}
        o[a]=x
    return o
def win(x,ref): v=x['vs_'+ref];return v['new_med_log'][2]<0 and abs(v['new_med_log'][0])>=D and v['known_acc16'][1]>-D and x['e33_pass']==16 and x['nan']==0
out={'frozen':F,'dev':phase('bq_dev'),'confirm':phase('bq_confirm')}
out['verdict']={f'{a}_vs_{r}':('PASA (DEV y CONFIRM)' if win(out['dev'][a],r) and win(out['confirm'][a],r) else ('REFUTADO (IC inf>0 en CONFIRM)' if out['confirm'][a]['vs_'+r]['new_med_log'][1]>0 else 'NO PASA')) for a in F['eval_arms'] if a not in('v4',) for r in ('v4','ALL') if a!=r}
if '--md' in sys.argv:
    f3=lambda v:f'{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]';L=['# Bio-Quantum v10 (port al arnés v4) — RESULTADOS','',f'Primario congelado: `{P}`. Referencias: v4 (CONSTELACIÓN v4 fd80320) y ALL (ATLAS-FW, main 1c845d6). 0 TEST.','']
    for ph in ('dev','confirm'):
        L+=[f'## {ph.upper()}','','| brazo | new_med_log | known_acc16 | Δ vs v4 new_med_log | Δ vs v4 known_acc16 | Δ vs ALL new_med_log | E33 | NaN |','|---|---|---|---|---|---|---|---|']
        for a in F['eval_arms']:
            x=out[ph][a];L.append(f"| {a} | {x['means']['new_med_log']:+.4f} | {x['means']['known_acc16']:.4f} | {f3(x['vs_v4']['new_med_log']) if 'vs_v4' in x else '—'} | {f3(x['vs_v4']['known_acc16']) if 'vs_v4' in x else '—'} | {f3(x['vs_ALL']['new_med_log']) if 'vs_ALL' in x else '—'} | {x['e33_pass']}/16 | {x['nan']} |")
        L.append('')
    L+=['## Veredicto','']+[f'- {k}: {v}' for k,v in out['verdict'].items()];print('\n'.join(L))
else: print(json.dumps(out,indent=1))
