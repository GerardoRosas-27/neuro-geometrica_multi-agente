#!/usr/bin/env python3
import json,sys,os,numpy as np
A='artifacts/bio_constelacion'; D=0.05
def L(ph,s,a): return json.load(open(f'{A}/{ph}/metrics_{s}_{a}.json'))
def boot(d):
    d=np.array(d); r=np.random.default_rng(12345); m=np.array([np.median(d[r.integers(0,len(d),len(d))]) for _ in range(10000)])
    return dict(delta=float(np.median(d)),lo=float(np.percentile(m,2.5)),hi=float(np.percentile(m,97.5)),n=len(d))
def phase(ph):
    P='A' if ph=='dev' else 'B'; S=[f'0x{P}40{i:X}' for i in range(16)]
    g=lambda a,k: [ (L(ph,s,a)['held'][k] if a!='ref' else None) for s in S]
    ref=[L(ph,s,'ref') for s in S]; e33=lambda a:[L(ph,s,a)['held']['e33']['acc'][4] for s in S]
    hy=g('hyb','new_med_log'); v4=g('v4','new_med_log'); nb=g('hyb_nobio','new_med_log')
    out={'arms_median':{a:{k:float(np.median(g(a,k))) for k in ['new_med_log','known_med_log','new_acc16','known_acc16']} for a in ['v4','hyb','hyb_nobio']},
     'orbita_new_med_log':float(np.median([r['orbita']['new_med_log'] for r in ref])),'persist_new_med_log':float(np.median([r['persist']['new_med_log'] for r in ref])),
     'primary_hyb_minus_v4':boot(np.subtract(hy,v4)),
     'sec_hyb_minus_orbita':boot(np.subtract(hy,[r['orbita']['new_med_log'] for r in ref])),
     'sec_hyb_minus_persist':boot(np.subtract(hy,[r['persist']['new_med_log'] for r in ref])),
     'sec_hyb_minus_nobio':boot(np.subtract(hy,nb)),
     'noharm_known_acc16':boot(np.subtract(g('hyb','known_acc16'),g('v4','known_acc16'))),
     'noharm_e33_h16':boot(np.subtract(e33('hyb'),e33('v4'))),
     'e33_h1_median':{a:float(np.median([L(ph,s,a)['held']['e33']['acc'][0] for s in S])) for a in ['v4','hyb','hyb_nobio']},
     'v4_hash_match_mejoras':all(L(ph,s,'v4')['hash']==json.load(open(f'/workspace/ngm-v4-mejoras/artifacts/constelacion_v4_mejoras/{ph}/metrics_{s}_v4.json'))['hash'] for s in S) if os.path.isdir('/workspace/ngm-v4-mejoras') else None}
    p=out['primary_hyb_minus_v4']; out['primary_pass']=p['hi']<0 and abs(p['delta'])>=D
    out['noharm_pass']=out['noharm_known_acc16']['lo']>=-D and out['noharm_e33_h16']['lo']>=-D
    return out
r={ph:phase(ph) for ph in sys.argv[1:]}; print(json.dumps(r,indent=1))
