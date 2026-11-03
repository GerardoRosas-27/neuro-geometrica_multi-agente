#!/usr/bin/env python3
"""Análisis preregistrado ATLAS-FW (docs/atlas_fw/PREREG.md).
Bootstrap pareado por semilla, 10000 réplicas, semilla 12345, IC percentil, δ=0.05.
Primaria: V − v4 en new_med_log (held-out h16). Singles M1..M6: IC Bonferroni 1−0.05/6. ALL: IC95.
No-daño: known_acc16 V−v4 IC95 inf > −δ; known_acc16 final−pre IC95 inf > −δ; E33 pass ≥ 15/16.
Uso: analyze_atlas_fw.py [dev] [confirm] [--gate] [--md]"""
import json,glob,sys,os,numpy as np
D=0.05;NB=10000;R=os.environ.get('ATLAS_A','artifacts/atlas_fw');VAR=['M1','M2','M3','M4','M5','M6','ALL']
def ci(x,lev=0.95):
    x=np.asarray(x,float);assert np.all(np.isfinite(x)),'NaN = fallo';r=np.random.default_rng(12345)
    b=x[r.integers(0,len(x),(NB,len(x)))].mean(1);a=(1-lev)/2;return [float(x.mean()),float(np.quantile(b,a)),float(np.quantile(b,1-a))]
def lev(v): return 0.95 if v=='ALL' else 1-0.05/6
def load(ph):
    out={}
    for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json')):
        m=json.load(open(f));assert m['test_queries']==0,f;assert m['phase']==ph,f;out[m['seed']]=m
    return out
def phase(ph):
    M=load(ph);seeds=sorted(M);arms=[a for a in ['v4']+VAR if seeds and all(a in M[s]['arms'] for s in seeds)]
    r={'n_seeds':len(seeds),'seeds':seeds,'arms':arms};g=lambda a,s:M[s]['arms'][a]
    for a in arms:
        x={'nan':sum(bool(g(a,s)['nan']) for s in seeds),'collapse':sum(bool(g(a,s)['collapse']['collapsed']) for s in seeds),
           'e33_pass':sum(bool(g(a,s)['e33']['pass']) for s in seeds),'e33_acc_mean':np.mean([g(a,s)['e33']['acc'] for s in seeds],0).round(4).tolist(),
           'means':{k:float(np.mean([g(a,s)['held'][k] for s in seeds])) for k in ('new_med_log','new_acc16','known_med_log','known_acc16')},
           'wall_s_mean':float(np.mean([g(a,s)['wall_s'] for s in seeds])),
           'noharm_known_acc16_final_minus_pre':ci([g(a,s)['held']['known_acc16']-M[s]['arms']['v4']['pre_known']['known_acc16'] for s in seeds])}
        if a!='v4':
            x['meta_alpha']=[g(a,s)['meta'].get('alpha') for s in seeds];x['gate_cls_mean']={k:float(np.mean([g(a,s)['gate_cls'].get(k,0) for s in seeds])) for k in ('known_frac_v4','new_frac_v4')}
            d=lambda f:[f(a,s)-f('v4',s) for s in seeds];L=lev(a)
            x['vs_v4']={'level':L,'n':len(seeds),'new_medlog':ci(d(lambda b,s:g(b,s)['held']['new_med_log']),L),
              'known_acc16':ci(d(lambda b,s:g(b,s)['held']['known_acc16'])),'known_medlog':ci(d(lambda b,s:g(b,s)['held']['known_med_log'])),
              'new_acc16':ci(d(lambda b,s:g(b,s)['held']['new_acc16']))}
        r[a]=x
    return r
def ok(x): return bool(x and x.get('vs_v4',{}).get('n')==16 and x['nan']==0 and x['vs_v4']['new_medlog'][2]<0 and x['vs_v4']['known_acc16'][1]>-D and x['noharm_known_acc16_final_minus_pre'][1]>-D and x['e33_pass']>=15)
def gate(d): return [v for v in VAR if ok(d.get(v))]
def verdict(d,c):
    out={}
    for v in VAR:
        if v not in c: out[v]='NO EVALUADO EN CONFIRM (no pasó gate DEV)';continue
        out[v]='PASA (DEV y CONFIRM)' if ok(d.get(v)) and ok(c[v]) else ('REFUTADO en CONFIRM (IC inf > 0)' if c[v]['vs_v4']['new_medlog'][1]>0 else 'NO PASA en CONFIRM')
    return out
def f3(x): return f'{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]'
def md(o):
    L=['# ATLAS-FW — RESULTADOS','','Generado por scripts/analyze_atlas_fw.py según docs/atlas_fw/PREREG.md. 0 TEST queries.','']
    for ph in ('dev','confirm'):
        p=o.get(ph);
        if not p: continue
        L+=[f'## {ph.upper()} ({p["n_seeds"]} semillas: {p["seeds"][0] if p["seeds"] else "-"}…{p["seeds"][-1] if p["seeds"] else "-"})','',
            '| brazo | new_med_log | V−v4 new_med_log (IC) | V−v4 known_acc16 IC95 | final−pre known_acc16 IC95 | E33 | NaN |','|---|---|---|---|---|---|---|']
        for a in p['arms']:
            x=p[a];v=x.get('vs_v4')
            L.append(f'| {a} | {x["means"]["new_med_log"]:+.4f} | {f3(v["new_medlog"])+f" ({v["level"]:.4f})" if v else "—"} | {f3(v["known_acc16"]) if v else "—"} | {f3(x["noharm_known_acc16_final_minus_pre"])} | {x["e33_pass"]}/16 | {x["nan"]} |')
        L.append('')
    if 'dev_gate' in o: L+=[f'Gate DEV: {", ".join(o["dev_gate"]) or "ninguno"}','']
    if 'verdict' in o: L+=['## Veredicto','']+[f'- {k}: {v}' for k,v in o['verdict'].items()]+['']
    return '\n'.join(L)
if __name__=='__main__':
    args=[x for x in sys.argv[1:] if not x.startswith('--')] or ['dev','confirm']
    out={p:phase(p) for p in args}
    if '--gate' in sys.argv: print(' '.join(gate(out['dev'])));sys.exit()
    if 'dev' in out: out['dev_gate']=gate(out['dev'])
    if 'dev' in out and 'confirm' in out: out['verdict']=verdict(out['dev'],out['confirm'])
    print(md(out) if '--md' in sys.argv else json.dumps(out,indent=1,ensure_ascii=False))
