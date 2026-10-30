#!/usr/bin/env python3
"""Análisis preregistrado CONSTELACIÓN v4-ESCALADA (docs/constelacion_v4_escalada/PREREG.md).
Bootstrap pareado por semilla, 10000 réplicas, semilla 12345, IC percentil, δ=0.05 (idéntico a v4).
IC95 para PAT y para todo lo que no sea 'V − v4'; IC 98.75% (Bonferroni 4) para 'V − v4' de P, A, T.
Uso: python3 scripts/analyze_constelacion_v4_escalada.py [dev] [confirm]   (gate: --gate imprime brazos para CONFIRM)"""
import json,glob,sys,os,numpy as np
D=0.05;NB=10000;R=os.environ.get('CST4E_A','artifacts/constelacion_v4_escalada');VAR=['P','A','T','PAT']
def ci(x,lev=0.95):
    x=np.asarray(x,float);assert np.all(np.isfinite(x)),'NaN = fallo';r=np.random.default_rng(12345)
    b=x[r.integers(0,len(x),(NB,len(x)))].mean(1);a=(1-lev)/2;return [float(x.mean()),float(np.quantile(b,a)),float(np.quantile(b,1-a))]
def lev(v): return 0.95 if v=='PAT' else 1-0.05/4
def load(ph,arm):
    out={}
    for f in sorted(glob.glob(f'{R}/{arm}/{ph}/metrics_0x*.json')):
        m=json.load(open(f));assert m['test_queries']==0,f;out[m['seed']]=m
    return out
def phase(ph):
    arms={a:load(ph,a) for a in ['v4']+VAR};arms={a:v for a,v in arms.items() if v}
    seeds=sorted(arms['v4']);r={'n':{a:len(v) for a,v in arms.items()},'seeds':seeds}
    g=lambda a,s:arms[a][s]['arms'][a]
    O=lambda s,k:arms['v4'][s]['orbita'][k]
    for a in arms:
        ss=[s for s in seeds if s in arms[a]];x={}
        x['primary_vs_orbita_new_medlog']=ci([g(a,s)['held']['new_med_log']-O(s,'new_med_log') for s in ss])
        x['noharm_known_acc16_post_minus_pre']=ci([g(a,s)['held']['known_acc16']-g(a,s)['pre_known']['known_acc16'] for s in ss])
        x['e33_pass']=sum(g(a,s)['e33']['pass'] for s in ss);x['e33_acc_mean']=np.mean([g(a,s)['e33']['acc'] for s in ss],0).round(4).tolist()
        x['surprise_auroc_mean']=float(np.mean([g(a,s)['surprise_auroc_h1'] for s in ss]))
        x['nan']=sum(g(a,s)['nan'] for s in ss);x['collapse']=sum(g(a,s)['collapse']['collapsed'] for s in ss)
        x['n_params']=g(a,ss[0])['n_params'];x['wall_s_mean']=float(np.mean([g(a,s)['wall_s'] for s in ss]))
        x['means']={k:float(np.mean([g(a,s)['held'][k] for s in ss])) for k in ('new_med_log','new_acc16','known_med_log','known_acc16')}
        if a!='v4':
            L=lev(a);d=lambda f:[f(a,s)-f('v4',s) for s in ss]
            x['vs_v4']={'level':L,'n':len(ss),
              'new_medlog':ci(d(lambda b,s:g(b,s)['held']['new_med_log']),L),
              'known_acc16':ci(d(lambda b,s:g(b,s)['held']['known_acc16'])),
              'known_medlog':ci(d(lambda b,s:g(b,s)['held']['known_med_log'])),
              'e33_acc_h1':ci(d(lambda b,s:g(b,s)['e33']['acc'][0])),
              'e33_acc_h16':ci(d(lambda b,s:g(b,s)['e33']['acc'][4])),
              'surprise_auroc':ci(d(lambda b,s:g(b,s)['surprise_auroc_h1']))}
        r[a]=x
    r['orbita_means']={k:float(np.mean([O(s,k) for s in seeds])) for k in ('new_med_log','new_acc16','known_med_log','known_acc16')}
    r['e33_orbita_pass']=sum(arms['v4'][s]['e33_orbita']['pass'] for s in seeds)
    return r
def gate(d):
    """Gate DEV→CONFIRM (PREREG §Gate): v4 siempre; V pasa si 16/16 sin NaN/colapso/TEST, media(V−v4) new_medlog < 0, no-daño DEV OK y E33 ≥ 15/16."""
    ok=['v4']
    for v in VAR:
        x=d.get(v)
        if not x or x.get('vs_v4',{}).get('n')!=16 or x['nan'] or x['collapse']: continue
        if x['vs_v4']['new_medlog'][0]<0 and x['noharm_known_acc16_post_minus_pre'][1]>-D and x['vs_v4']['known_acc16'][1]>-D and x['e33_pass']>=15: ok.append(v)
    return ok
def verdict(d,c):
    out={}
    for v in VAR:
        if v not in c or v not in d: out[v]={'veredicto':'NO EVALUADA EN CONFIRM (gate DEV)'};continue
        X=[d[v],c[v]];w={}
        w['mejora_vs_v4']=all(x['vs_v4']['new_medlog'][2]<0 for x in X)
        w['mejora_material_vs_v4']=all(x['vs_v4']['new_medlog'][2]<-D for x in X)
        w['refutada_vs_v4']=c[v]['vs_v4']['new_medlog'][1]>0
        w['vs_orbita']=all(x['primary_vs_orbita_new_medlog'][2]<0 for x in X)
        w['no_dano']=all(x['noharm_known_acc16_post_minus_pre'][1]>-D and x['vs_v4']['known_acc16'][1]>-D for x in X)
        w['e33']=all(x['e33_pass']>=15 for x in X)
        w['h1_no_inferior']=all(x['vs_v4']['e33_acc_h1'][1]>-D for x in X)
        w['sorpresa_no_inferior']=all(x['vs_v4']['surprise_auroc'][1]>-D for x in X)
        w['limpia']=all(x['nan']==0 and x['collapse']==0 and x['vs_v4']['n']==16 for x in X)
        w['candidata_main']=all(w[k] for k in ('mejora_vs_v4','vs_orbita','no_dano','e33','h1_no_inferior','sorpresa_no_inferior','limpia'))
        out[v]=w
    cands=[v for v in VAR if out[v].get('candidata_main')]
    # desempate: menor n_params entre las candidatas cuya media CONFIRM (V−v4) esté a ≤ δ de la mejor candidata
    if cands:
        best=min(c[v]['vs_v4']['new_medlog'][0] for v in cands)
        near=[v for v in cands if c[v]['vs_v4']['new_medlog'][0]<=best+D]
        out['propuesta_main']=min(near,key=lambda v:(c[v]['n_params'],c[v]['wall_s_mean']))
    else: out['propuesta_main']=None
    return out
if __name__=='__main__':
    args=[x for x in sys.argv[1:] if not x.startswith('--')] or ['dev','confirm']
    out={p:phase(p) for p in args}
    if '--gate' in sys.argv: print(' '.join(gate(out['dev'])));sys.exit()
    if 'dev' in out and 'confirm' in out: out['dev_gate']=gate(out['dev']);out['verdict']=verdict(out['dev'],out['confirm'])
    print(json.dumps(out,indent=1,ensure_ascii=False))
