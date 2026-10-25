#!/usr/bin/env python3
"""Análisis preregistrado MAREA (docs/marea/PREREG.md). Bootstrap pareado por semilla 10000, semilla 12345, δ=0.05."""
import json,glob,sys,numpy as np
D=0.05;NB=10000
CTRL=['marea_mrand','marea_norollback','marea_nohopfield','marea_shufkeys','marea_randjepa','marea_k1']
def load(ph):
    ms=[json.load(open(f)) for f in sorted(glob.glob(f'artifacts/marea/{ph}/metrics_0x*.json'))]
    for m in ms: assert m['test_queries']==0 and m['parity_eta0_vs_frozen']['mismatch']==0, m['seed']
    return ms
def ci(x):
    x=np.asarray(x,float);r=np.random.default_rng(12345);b=x[r.integers(0,len(x),(NB,len(x)))].mean(1)
    return [float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))]
def phase(ph):
    ms=load(ph);A=lambda a,k:np.array([m['arms'][a][k] for m in ms])
    r={'n':len(ms),
       'primary_slope_new_marea_minus_eta0':ci(A('marea','slope_new')-A('marea_eta0','slope_new')),
       'sec_mlog_new_marea_minus_eta0':ci(A('marea','mlog_new')-A('marea_eta0','mlog_new')),
       'noharm_mlog_known_marea_minus_eta0':ci(A('marea','mlog_known')-A('marea_eta0','mlog_known')),
       'ext_mlog_new_post_marea_minus_orbita_lat':ci(A('marea','mlog_new_post')-A('orbita_lat','mlog_new_post')),
       'ext_mlog_known_post_marea_minus_orbita_lat':ci(A('marea','mlog_known_post')-A('orbita_lat','mlog_known_post')),
       'controls_slope_new_marea_minus':{c:ci(A('marea','slope_new')-A(c,'slope_new')) for c in CTRL},
       'controls_mlog_new_marea_minus':{c:ci(A('marea','mlog_new')-A(c,'mlog_new')) for c in CTRL},
       'means':{a:{k:float(A(a,k).mean()) for k in ['slope_new','mlog_new','mlog_known','mlog_new_post']} for a in ['marea','marea_eta0']+CTRL},
       'means_ext':{a:{k:float(A(a,k).mean()) for k in ['mlog_new_post','mlog_known_post']} for a in ['orbita_lat','orbita']},
       'counters':{k:float(A('marea',k).mean()) for k in ['n_rollback','n_retrieve','n_plus','n_pats']}}
    return r
out={p:phase(p) for p in (sys.argv[1:] or ['dev','confirm'])}
if 'dev' in out and 'confirm' in out:
    d,c=out['dev'],out['confirm'];P='primary_slope_new_marea_minus_eta0'
    v={'primary':'ACEPTADA' if d[P][2]<0 and c[P][2]<0 else ('REFUTADA' if c[P][1]>0 else 'NO CONCLUYENTE (H1 no aceptada)'),
       'secondary_mlog_new':'ACEPTADA' if all(x['sec_mlog_new_marea_minus_eta0'][2]<0 for x in (d,c)) else 'NO ACEPTADA',
       'no_harm':'OK' if all(x['noharm_mlog_known_marea_minus_eta0'][2]<D for x in (d,c)) else 'VETO',
       'external_vs_orbita_lat':'MAREA MEJOR' if all(x['ext_mlog_new_post_marea_minus_orbita_lat'][2]<0 for x in (d,c)) else ('MAREA PEOR' if all(x['ext_mlog_new_post_marea_minus_orbita_lat'][1]>0 for x in (d,c)) else 'NO CONCLUYENTE'),
       'controls_marea_better_slope_both_phases':{k:all(x['controls_slope_new_marea_minus'][k][2]<0 for x in (d,c)) for k in CTRL}}
    out['verdict']=v
print(json.dumps(out,indent=1))
