#!/usr/bin/env python3
"""Análisis preregistrado CONSTELACIÓN v4-NEURO (docs/constelacion_v4_neuro/PREREG.md).
Bootstrap pareado por semilla, 10000 réplicas, semilla 12345, IC percentil, δ=0.05.
p1: ALL = hipótesis principal (IC95); M1..M6 en 'V − v4' primaria con IC Bonferroni ×7 (1−0.05/7). p2: NEU vs v4, IC95.
Uso: analyze_constelacion_v4_neuro.py p1|p2 [dev] [confirm] [--gate] [--mech]"""
import json,glob,sys,os,numpy as np
D=0.05;NB=10000;R0=os.environ.get('CST4N_A','artifacts/constelacion_v4_neuro')
ST=sys.argv[1];R=f'{R0}/{ST}';VAR=['M1','M2','M3','M4','M5','M6','ALL'] if ST=='p1' else ['NEU'];PRIN='ALL' if ST=='p1' else 'NEU'
def ci(x,lev=0.95):
    x=np.asarray(x,float);assert np.all(np.isfinite(x)),'NaN = fallo';r=np.random.default_rng(12345)
    b=x[r.integers(0,len(x),(NB,len(x)))].mean(1);a=(1-lev)/2;return [float(x.mean()),float(np.quantile(b,a)),float(np.quantile(b,1-a))]
def lev(v): return 0.95 if v==PRIN else 1-0.05/len(VAR)
def load(ph):
    out={}
    for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json')):
        m=json.load(open(f));assert m['test_queries']==0,f;out[m['seed']]=m
    return out
def phase(ph):
    M=load(ph);seeds=sorted(M);arms=[a for a in ['v4']+VAR if all(a in M[s]['arms'] for s in seeds)] if seeds else []
    r={'n_seeds':len(seeds),'seeds':seeds,'arms':arms}
    g=lambda a,s:M[s]['arms'][a];O=lambda s,k:M[s]['orbita'][k]
    for a in arms:
        x={}
        x['primary_vs_orbita_new_medlog']=ci([g(a,s)['held']['new_med_log']-O(s,'new_med_log') for s in seeds])
        x['noharm_known_acc16_post_minus_pre']=ci([g(a,s)['held']['known_acc16']-g(a,s)['pre_known']['known_acc16'] for s in seeds])
        x['e33_pass']=sum(g(a,s)['e33']['pass'] for s in seeds);x['e33_acc_mean']=np.mean([g(a,s)['e33']['acc'] for s in seeds],0).round(4).tolist()
        x['surprise_auroc_mean']=float(np.mean([g(a,s)['surprise_auroc_h1'] for s in seeds]))
        x['nan']=sum(g(a,s)['nan'] for s in seeds);x['collapse']=sum(g(a,s)['collapse']['collapsed'] for s in seeds)
        x['n_params']=g(a,seeds[0])['n_params'];x['n_active_conn_mean']=float(np.mean([g(a,s)['n_active_conn'] for s in seeds]))
        x['fire_frac_streamB_mean']=float(np.mean([g(a,s)['fire_frac_streamB'] for s in seeds]));x['fire_frac_eval_mean']=float(np.mean([g(a,s)['fire_frac_eval'] for s in seeds]))
        x['m4_mean']={k:float(np.mean([g(a,s)['m4'][k] for s in seeds])) for k in ('weakened','pruned','grown')}
        x['m5_consolidated_frac_mean']=float(np.mean([g(a,s)['m5']['frac_active_consolidated'] for s in seeds]))
        x['wall_s_mean']=float(np.mean([g(a,s)['wall_s'] for s in seeds]))
        x['means']={k:float(np.mean([g(a,s)['held'][k] for s in seeds])) for k in ('new_med_log','new_acc16','known_med_log','known_acc16')}
        if a!='v4':
            L=lev(a);d=lambda f:[f(a,s)-f('v4',s) for s in seeds]
            x['vs_v4']={'level':L,'n':len(seeds),
              'new_medlog':ci(d(lambda b,s:g(b,s)['held']['new_med_log']),L),
              'known_acc16':ci(d(lambda b,s:g(b,s)['held']['known_acc16'])),
              'known_medlog':ci(d(lambda b,s:g(b,s)['held']['known_med_log'])),
              'e33_acc_h1':ci(d(lambda b,s:g(b,s)['e33']['acc'][0])),
              'e33_acc_h16':ci(d(lambda b,s:g(b,s)['e33']['acc'][4])),
              'surprise_auroc':ci(d(lambda b,s:g(b,s)['surprise_auroc_h1'])),
              'fire_frac_eval':ci(d(lambda b,s:g(b,s)['fire_frac_eval']))}
        r[a]=x
    if seeds:
        r['orbita_v1_means']={k:float(np.mean([O(s,k) for s in seeds])) for k in ('new_med_log','new_acc16','known_med_log','known_acc16')}
        r['e33_orbita_pass']=sum(M[s]['e33_orbita']['pass'] for s in seeds)
    return r
def gate(d):
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
        w['pasa_fase']=all(w[k] for k in ('mejora_vs_v4','no_dano','e33','limpia'))
        w['candidata_main']=all(w[k] for k in ('mejora_vs_v4','vs_orbita','no_dano','e33','h1_no_inferior','sorpresa_no_inferior','limpia'))
        out[v]=w
    return out
def mech(v):
    """Fase 2 (PREREG §Fase 2): si ALL pasa la fase → M1..M6; si no, unión de los Mi que pasan; vacío → sin fase 2."""
    if v.get('ALL',{}).get('pasa_fase'): return 'M1,M2,M3,M4,M5,M6'
    return ','.join(m for m in VAR if m!='ALL' and v.get(m,{}).get('pasa_fase'))
if __name__=='__main__':
    args=[x for x in sys.argv[2:] if not x.startswith('--')] or ['dev','confirm']
    out={p:phase(p) for p in args}
    if '--gate' in sys.argv: print(' '.join(gate(out['dev'])));sys.exit()
    if 'dev' in out and 'confirm' in out and out['confirm']['n_seeds']:
        out['dev_gate']=gate(out['dev']);out['verdict']=verdict(out['dev'],out['confirm'])
        if ST=='p1': out['fase2_mech']=mech(out['verdict'])
    if '--mech' in sys.argv: print(out.get('fase2_mech',''));sys.exit()
    print(json.dumps(out,indent=1,ensure_ascii=False))
