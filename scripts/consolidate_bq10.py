#!/usr/bin/env python3
"""Tabla consolidada v4 / ATLAS-FW ALL / BQ10 / ÓRBITA-v1 (referencia) desde artifacts/bio_quantum_v10/{bq_dev,bq_confirm}.
Bootstrap pareado por semilla 10000, seed 12345, IC95 percentil (mismo método que analyze_bq10.py). Uso: consolidate_bq10.py [--md]"""
import json,glob,sys,numpy as np
R='artifacts/bio_quantum_v10';NB=10000
def ci(x):
    x=np.asarray(x,float);assert np.all(np.isfinite(x));r=np.random.default_rng(12345);b=x[r.integers(0,len(x),(NB,len(x)))].mean(1)
    return [float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))]
K=('new_med_log','known_acc16','new_acc16')
def load(ph):
    M=[json.load(open(f)) for f in sorted(glob.glob(f'{R}/{ph}/metrics_0x*.json'))];assert len(M)==16 and all(m['test_queries']==0 for m in M);return M
def held(m,a): return m['orbita'] if a=='ORBITA' else m['arms'][a]['held']
def e33(m,a): return (m['e33_orbita'] if a=='ORBITA' else m['arms'][a]['e33'])['pass']
def wall(m,a): return m['v4_train']['train_s'] if a=='v4_train' else m['arms'][a]['wall_s']
A=['ORBITA','v4','ALL','BQ10'];P=[('BQ10','v4'),('BQ10','ALL'),('ALL','v4'),('v4','ORBITA'),('BQ10','ORBITA')]
out={}
for ph in ('bq_dev','bq_confirm'):
    M=load(ph);o={'arms':{},'pairs':{}}
    for a in A: o['arms'][a]={**{k:ci([held(m,a)[k] for m in M]) for k in K},'e33_pass':sum(bool(e33(m,a)) for m in M),'wall_s':(ci([wall(m,a) for m in M]) if a!='ORBITA' else None)}
    o['v4_train_s']=ci([wall(m,'v4_train') for m in M]);o['seed_wall_s']=ci([m['wall_s'] for m in M])
    for x,y in P: o['pairs'][f'{x}-{y}']={k:ci([held(m,x)[k]-held(m,y)[k] for m in M]) for k in K}
    out[ph]=o
if '--md' not in sys.argv: print(json.dumps(out,indent=1));sys.exit()
f=lambda v:f'{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]';g=lambda v:f'{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]'
N={'ORBITA':'ÓRBITA-v1 (ref., pesos congelados)','v4':'CONSTELACIÓN v4','ALL':'ATLAS-FW (ALL)','BQ10':'**BQ10 (bio-quantum v10)**'}
for ph,t in (('bq_dev','DEV 0xA400–0xA40F'),('bq_confirm','CONFIRM 0xB400–0xB40F')):
    o=out[ph];print(f'#### {t} — medias [IC95]\n\n| arquitectura | new_med_log h16 k=4 | known_acc16 | new_acc16 | E33 | tiempo/semilla del brazo (s) |\n|---|---|---|---|---|---|')
    for a in A: x=o['arms'][a];w='no medido (se entrena aparte)' if a=='ORBITA' else (f"{x['wall_s'][0]:.1f}" + (f" (+{o['v4_train_s'][0]:.0f} s entrenar v4, compartido)" if a=='v4' else ' (sobre v4 entrenada)'));print(f"| {N[a]} | {f(x['new_med_log'])} | {g(x['known_acc16'])} | {g(x['new_acc16'])} | {x['e33_pass']}/16 | {w} |")
    print(f"\nPareados (por semilla) — Δ [IC95]:\n\n| contraste | Δ new_med_log | Δ known_acc16 | Δ new_acc16 |\n|---|---|---|---|")
    for p,x in o['pairs'].items(): print(f"| {p.replace('ORBITA','ÓRBITA')} | {f(x['new_med_log'])} | {f(x['known_acc16'])} | {f(x['new_acc16'])} |")
    print(f"\nWall total por semilla (v4 train + v4, ALL, CORE, BQ10, BQR10 + ÓRBITA): {o['seed_wall_s'][0]:.0f} s.\n")
