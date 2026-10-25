#!/usr/bin/env python3
"""Análisis preregistrado CONSTELACIÓN v2 (docs/constelacion_v2/PREREG.md). Bootstrap pareado por semilla 10000, semilla 12345, IC95 percentil, δ=0.05."""
import json,glob,sys,os,numpy as np
D=0.05;NB=10000;A='artifacts/constelacion_v2'
CTRL=['no_err','no_prec','t1','no_prune','no_sleep','fixed_w','one_layer','backprop']
def ci(x):
    x=np.asarray(x,float);r=np.random.default_rng(12345)
    b=x[r.integers(0,len(x),(NB,len(x)))].mean(1);return [float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))]
def load(ph):
    ms=[json.load(open(f)) for f in sorted(glob.glob(f'{A}/{ph}/metrics_0x*.json'))]
    for m in ms: assert m['test_queries']==0, m['seed']
    return ms
def estela_ref(ph,seed):
    f=f'/workspace/ngm-estela/artifacts/estela/{ph}/metrics_{seed}.json'
    if not os.path.exists(f): return None
    return json.load(open(f))['arms']['k4']['orbita']
def phase(ph):
    ms=load(ph);H=lambda a,k:np.array([m['arms'][a]['held'][k] for m in ms]);O=lambda k:np.array([m['orbita'][k] for m in ms])
    r={'n':len(ms),'seeds':[m['seed'] for m in ms]}
    r['primary_new_medlog_cst_minus_orbita']=ci(H('cst','new_med_log')-O('new_med_log'))
    r['noharm_known_acc16_post_minus_pre']=ci(H('cst','known_acc16')-np.array([m['arms']['cst']['pre_known']['known_acc16'] for m in ms]))
    r['e33_pass']={'cst':sum(m['arms']['cst']['e33']['pass'] for m in ms),'orbita':sum(m['e33_orbita']['pass'] for m in ms)}
    r['e33_h16_cst_minus_orbita']=ci([m['arms']['cst']['e33']['acc'][4]-m['e33_orbita']['acc'][4] for m in ms])
    r['controls_new_medlog_cst_minus']={c:ci(H('cst','new_med_log')-H(c,'new_med_log')) for c in CTRL if all(c in m['arms'] for m in ms)}
    r['means']={a:{k:float(H(a,k).mean()) for k in ('new_med_log','new_acc16','known_med_log','known_acc16')} for a in ['cst']+CTRL if all(a in m['arms'] for m in ms)}
    r['means']['orbita']={k:float(O(k).mean()) for k in ('new_med_log','new_acc16','known_med_log','known_acc16')}
    r['collapse_cst']=sum(m['arms']['cst']['collapse']['collapsed'] for m in ms); r['nan_any']={a:sum(m['arms'][a]['nan'] for m in ms) for a in ms[0]['arms']}
    eq=[]
    for m in ms:
        e=estela_ref(ph,m['seed'])
        if e: eq.append(all(abs(e[k]-m['orbita'][k])==0 for k in ('known_med_log','new_med_log','known_acc16','new_acc16')))
    r['orbita_equals_estela_k4']=f"{sum(eq)}/{len(eq)}"
    return r
out={p:phase(p) for p in (sys.argv[1:] or ['dev','confirm'])}
if 'dev' in out and 'confirm' in out:
    d,c=out['dev'],out['confirm'];pd,pc=d['primary_new_medlog_cst_minus_orbita'],c['primary_new_medlog_cst_minus_orbita']
    v={'primary':'ACEPTADA' if pd[2]<0 and pc[2]<0 else ('REFUTADA (CST peor)' if pc[1]>0 else 'NO ACEPTADA (no concluyente)')}
    v['no_harm']='OK' if all(x['noharm_known_acc16_post_minus_pre'][1]>-D for x in (d,c)) else 'VETO'
    att=all(x['controls_new_medlog_cst_minus'][k][2]<0 for x in (d,c) for k in ('fixed_w','no_err','t1'))
    v['attribution_PC']='SÍ' if att else 'NO'
    out['verdict']=v
print(json.dumps(out,indent=1,ensure_ascii=False))
