#!/usr/bin/env python3
"""Análisis preregistrado ECO (docs/eco/PREREG.md). Bootstrap pareado 10000, semilla 12345, δ=0.05."""
import json,glob,sys,numpy as np
D=0.05;NB=10000
def load(ph):
    ms=[json.load(open(f)) for f in sorted(glob.glob(f'artifacts/eco/{ph}/metrics_0x*.json'))]
    for m in ms: assert m['test_queries']==0 and m['parity_zero_vs_orbita_lat']['mismatch']==0, m['seed']
    return ms
def ci(x):
    x=np.asarray(x,float);r=np.random.default_rng(12345)
    b=x[r.integers(0,len(x),(NB,len(x)))].mean(1);return float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))
CTRL=['orbita','eco_nosleep','eco_norollback','eco_shuf','eco_pos','eco_e2','eco_m3']
def phase(ph):
    ms=load(ph);A=lambda a,k:[m['arms'][a][k] for m in ms];H=lambda a,k:[m['arms'][a]['held'][k] for m in ms]
    e,l=np.array(A('eco','auc_new')),np.array(A('orbita_lat','auc_new'))
    r={'n':len(ms),'primary_auc_new_eco_minus_orbita_lat':ci(e-l)}
    r['noharm_known_acc16_eco_minus_orbita']=ci(np.array(H('eco','known_acc16'))-H('orbita','known_acc16'))
    r['noharm_known_acc16_eco_minus_orbita_lat']=ci(np.array(H('eco','known_acc16'))-H('orbita_lat','known_acc16'))
    r['e33_h16_eco_minus_orbita']=ci([m['e33']['eco']['acc'][4]-m['e33']['orbita']['acc'][4] for m in ms])
    r['e33_pass']={a:sum(m['e33'][a]['pass'] for m in ms) for a in ('eco','orbita')}
    r['controls_auc_new_eco_minus']={c:ci(e-np.array(A(c,'auc_new'))) for c in CTRL}
    r['sleeps_accepted_mean']=float(np.mean(A('eco','n_accepted')));r['delta_used_new_mean']=float(np.mean(A('eco','delta_used_new')))
    r['means_auc_new']={a:float(np.mean(A(a,'auc_new'))) for a in ['eco','orbita_lat']+CTRL}
    return r
out={p:phase(p) for p in sys.argv[1:] or ['dev','confirm']}
v={}
if 'dev' in out and 'confirm' in out:
    d,c=out['dev'],out['confirm'];pd,pc=d['primary_auc_new_eco_minus_orbita_lat'],c['primary_auc_new_eco_minus_orbita_lat']
    v['primary']='ACEPTADA' if pd[2]<0 and pc[2]<0 else ('REFUTADA' if pc[1]>0 else 'NO CONCLUYENTE (H1 no aceptada)')
    v['no_harm']='OK' if all(x[k][1]>-D for x in (d,c) for k in ('noharm_known_acc16_eco_minus_orbita','noharm_known_acc16_eco_minus_orbita_lat')) else 'VETO'
    v['e33']='OK' if all(x['e33_h16_eco_minus_orbita'][1]>-D for x in (d,c)) else 'FALLA'
out['verdict']=v;print(json.dumps(out,indent=1))
