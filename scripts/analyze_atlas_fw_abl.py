#!/usr/bin/env python3
"""Análisis preregistrado de ablaciones leave-one-out ATLAS-FW (docs/atlas_fw/PREREG_ABLACIONES.md).
Bootstrap pareado por semilla, 10000 réplicas, semilla 12345, IC percentil, δ=0.05.
Primaria: Δ_i = (ALL−Mi) − ALL en new_med_log; IC Bonferroni 1−0.05/6. Δ_i>0 (IC inf > 0) ⇒ Mi contribuye; IC sup < 0 ⇒ Mi perjudica.
No-daño de cada brazo vs v4: known_acc16 V−v4 IC95 inf > −δ; known_acc16 final−pre IC95 inf > −δ; E33 pass ≥ 15/16; 0 NaN.
Uso: analyze_atlas_fw_abl.py [abl_dev] [abl_confirm] [--gate] [--md]"""
import json,glob,sys,os,numpy as np
D=0.05;NB=10000;BONF=1-0.05/6;R=os.environ.get('ATLAS_A','artifacts/atlas_fw/ablaciones')
ABL=[f'ALL-M{i}' for i in range(1,7)];ARMS=['v4','ALL']+ABL
def ci(x,lev=0.95):
    x=np.asarray(x,float);assert np.all(np.isfinite(x)),'NaN = fallo';r=np.random.default_rng(12345)
    b=x[r.integers(0,len(x),(NB,len(x)))].mean(1);a=(1-lev)/2;return [float(x.mean()),float(np.quantile(b,a)),float(np.quantile(b,1-a))]
def load(ph):
    out={}
    for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json')):
        m=json.load(open(f));assert m['test_queries']==0,f;assert m['phase']==ph,f;out[m['seed']]=m
    return out
def phase(ph):
    M=load(ph);seeds=sorted(M);arms=[a for a in ARMS if seeds and all(a in M[s]['arms'] for s in seeds)]
    r={'n_seeds':len(seeds),'seeds':seeds,'arms':arms};g=lambda a,s:M[s]['arms'][a]['held']
    for a in arms:
        A=lambda s:M[s]['arms'][a]
        x={'nan':sum(bool(A(s)['nan']) for s in seeds),'e33_pass':sum(bool(A(s)['e33']['pass']) for s in seeds),
           'means':{k:float(np.mean([g(a,s)[k] for s in seeds])) for k in ('new_med_log','new_acc16','known_med_log','known_acc16')},
           'noharm_known_acc16_final_minus_pre':ci([g(a,s)['known_acc16']-M[s]['arms']['v4']['pre_known']['known_acc16'] for s in seeds])}
        if a!='v4':
            x['vs_v4']={'n':len(seeds),'new_medlog':ci([g(a,s)['new_med_log']-g('v4',s)['new_med_log'] for s in seeds]),
                        'known_acc16':ci([g(a,s)['known_acc16']-g('v4',s)['known_acc16'] for s in seeds])}
        if a in ABL and 'ALL' in arms:
            x['vs_ALL']={'level':BONF,'n':len(seeds),'new_medlog':ci([g(a,s)['new_med_log']-g('ALL',s)['new_med_log'] for s in seeds],BONF),
                         'known_acc16':ci([g(a,s)['known_acc16']-g('ALL',s)['known_acc16'] for s in seeds])}
        r[a]=x
    return r
def noharm(x): return bool(x and x['nan']==0 and x['vs_v4']['known_acc16'][1]>-D and x['noharm_known_acc16_final_minus_pre'][1]>-D and x['e33_pass']>=15)
def sign(x):
    if not x or x.get('vs_ALL',{}).get('n')!=16: return 0
    c=x['vs_ALL']['new_medlog'];return 1 if c[1]>0 else (-1 if c[2]<0 else 0)
def gate(d): return [a for a in ABL if sign(d.get(a))!=0]
def verdict(d,c):
    out={}
    for a in ABL:
        m=a.split('-')[1]
        if a not in c: out[m]='NO SIGNIFICATIVO EN DEV (no evaluado en CONFIRM)';continue
        sd,sc=sign(d.get(a)),sign(c.get(a))
        out[m]=('CONTRIBUYE (DEV y CONFIRM)' if sd==sc==1 else 'PERJUDICA (DEV y CONFIRM)' if sd==sc==-1 else 'NO CONFIRMADO')+f'; no-daño ALL−{m}: {"OK" if noharm(c.get(a)) else "FALLA"}'
    return out
def f3(x): return f'{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]'
def md(o):
    L=['# ATLAS-FW — RESULTADOS ABLACIONES (leave-one-out)','','Generado por scripts/analyze_atlas_fw_abl.py según docs/atlas_fw/PREREG_ABLACIONES.md. 0 TEST queries.','',
       'Δ = (ALL−Mi) − ALL en new_med_log (IC Bonferroni 0.9917). Δ>0 ⇒ Mi contribuye.','']
    for ph in ('abl_dev','abl_confirm'):
        p=o.get(ph)
        if not p: continue
        L+=[f'## {ph.upper()} ({p["n_seeds"]} semillas: {p["seeds"][0] if p["seeds"] else "-"}…{p["seeds"][-1] if p["seeds"] else "-"})','',
            '| brazo | new_med_log | Δ vs ALL new_med_log (IC 0.9917) | Δ vs ALL known_acc16 IC95 | V−v4 new_med_log IC95 | V−v4 known_acc16 IC95 | final−pre known_acc16 IC95 | E33 | NaN | no-daño |','|---|---|---|---|---|---|---|---|---|---|']
        for a in p['arms']:
            x=p[a];v=x.get('vs_v4');w=x.get('vs_ALL')
            L.append(f'| {a} | {x["means"]["new_med_log"]:+.4f} | {f3(w["new_medlog"]) if w else "—"} | {f3(w["known_acc16"]) if w else "—"} | {f3(v["new_medlog"]) if v else "—"} | {f3(v["known_acc16"]) if v else "—"} | {f3(x["noharm_known_acc16_final_minus_pre"])} | {x["e33_pass"]}/16 | {x["nan"]} | {("OK" if noharm(x) else "FALLA") if v else "—"} |')
        L.append('')
    if 'dev_gate' in o: L+=[f'Ablaciones significativas en DEV: {", ".join(o["dev_gate"]) or "ninguna"}','']
    if 'verdict' in o: L+=['## Veredicto','']+[f'- {k}: {v}' for k,v in o['verdict'].items()]+['']
    return '\n'.join(L)
if __name__=='__main__':
    args=[x for x in sys.argv[1:] if not x.startswith('--')] or ['abl_dev','abl_confirm']
    out={p:phase(p) for p in args}
    if '--gate' in sys.argv: print(' '.join(gate(out['abl_dev'])));sys.exit()
    if 'abl_dev' in out: out['dev_gate']=gate(out['abl_dev'])
    if 'abl_dev' in out and 'abl_confirm' in out: out['verdict']=verdict(out['abl_dev'],out['abl_confirm'])
    print(md(out) if '--md' in sys.argv else json.dumps(out,indent=1,ensure_ascii=False))
