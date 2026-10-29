#!/usr/bin/env python3
"""Análisis preregistrado CONSTELACIÓN v4 mejoras (docs/constelacion_v4_mejoras/PREREG.md).
Bootstrap pareado por semilla 10000, semilla 12345. Primaria IC98.33 (Bonferroni×3), Holm como sensibilidad; no-daño IC95, δ=0.05.
Uso: analyze_constelacion_v4_mejoras.py dev | final   (final escribe RESULTADOS.md)"""
import json,glob,sys,os,numpy as np
D=0.05;NB=10000;A='artifacts/constelacion_v4_mejoras';V4='artifacts/constelacion_v4';DOC='docs/constelacion_v4_mejoras'
IDEAS=['m1','m2','m3'];EXPL=['m1_ctrl','m3_rf','m3_orb','all']
def boot(x):
    x=np.asarray(x,float);r=np.random.default_rng(12345);return x[r.integers(0,len(x),(NB,len(x)))].mean(1)
def ci(x,a=0.05):
    b=boot(x);return [float(np.mean(x)),float(np.quantile(b,a/2)),float(np.quantile(b,1-a/2))]
def pval(x):
    b=boot(x);return float(min(1.0,2*min((b>=0).mean(),(b<=0).mean())))
def load(ph):
    out={}
    for f in sorted(glob.glob(f'{A}/{ph}/metrics_0x*_*.json')):
        m=json.load(open(f));assert m['test_queries']==0,f;out.setdefault(m['arm'],{})[m['seed']]=m
    return out
def phase(ph):
    M=load(ph);r={'arms':{a:len(v) for a,v in M.items()}}
    if 'v4' not in M: return r
    seeds=sorted(M['v4']);base=M['v4']
    ref={s:json.load(open(f'{V4}/{ph}/metrics_{s}.json')) for s in seeds if os.path.exists(f'{V4}/{ph}/metrics_{s}.json')}
    r['v4_hash_repro']={'n':len(ref),'match':sum(ref[s]['arms']['cst']['hash']==base[s]['hash'] for s in ref)}
    H=lambda a,k,ss:np.array([M[a][s]['held'][k] for s in ss]);E=lambda a,ss:np.array([M[a][s]['held']['e33']['acc'][4] for s in ss])
    cmp={}
    for a in IDEAS+EXPL:
        if a not in M: continue
        ss=[s for s in seeds if s in M[a]];c={'n':len(ss),'nan_seeds':sum(bool(M[a][s]['nan']) for s in ss)}
        d=H(a,'new_med_log',ss)-H('v4','new_med_log',ss)
        c['primary_new_medlog_minus_v4_ci9833']=ci(d,0.05/3);c['p_boot']=pval(d)
        c['noharm_known_acc16_ci95']=ci(H(a,'known_acc16',ss)-H('v4','known_acc16',ss))
        c['noharm_e33_h16acc_ci95']=ci(E(a,ss)-E('v4',ss))
        c['known_medlog_minus_v4_ci95']=ci(H(a,'known_med_log',ss)-H('v4','known_med_log',ss))
        c['new_acc16_minus_v4_ci95']=ci(H(a,'new_acc16',ss)-H('v4','new_acc16',ss))
        rs=[s for s in ss if s in ref]
        if rs: c['sec_new_medlog_minus_orbita_ci95']=ci(H(a,'new_med_log',rs)-np.array([ref[s]['orbita']['new_med_log'] for s in rs]))
        c['sec_new_h1_minus_persist_ci95']=ci(H(a,'new_h1_med_log',ss)-H(a,'new_h1_persist_med_log',ss))
        c['m_mean_train']=float(np.mean([M[a][s].get('train_m_mean',1.0) for s in ss]))
        p=c['primary_new_medlog_minus_v4_ci9833'];nh=[c['noharm_known_acc16_ci95'],c['noharm_e33_h16acc_ci95']]
        c['noharm_pass']=all(x[1]>-D for x in nh)
        c['pass']=bool(p[2]<0 and c['noharm_pass'] and c['nan_seeds']<=1)
        c['stop']=bool(p[1]>0 or any(x[2]<-D for x in nh) or c['nan_seeds']>1)
        cmp[a]=c
    if 'm1' in M and 'm1_ctrl' in M:
        ss=[s for s in seeds if s in M['m1'] and s in M['m1_ctrl']]
        cmp['m1_vs_m1_ctrl']={'new_medlog_ci95':ci(H('m1','new_med_log',ss)-H('m1_ctrl','new_med_log',ss))}
    ps=sorted([(cmp[a]['p_boot'],a) for a in IDEAS if a in cmp]);mx=0
    for i,(p,a) in enumerate(ps): mx=max(mx,min(1,p*(len(ps)-i)));cmp[a]['p_holm']=mx;cmp[a]['holm_sig']=bool(mx<0.05 and cmp[a]['primary_new_medlog_minus_v4_ci9833'][0]<0)
    r['v4']={'new_med_log_mean':float(H('v4','new_med_log',seeds).mean()),'known_acc16_mean':float(H('v4','known_acc16',seeds).mean())}
    if ref: r['orbita_new_med_log_mean']=float(np.mean([ref[s]['orbita']['new_med_log'] for s in ref]))
    r['cmp']=cmp;return r
def ph_(x): return f"{x['p_holm']:.4f}" if 'p_holm' in x else '—'
def fmt(x): return f'{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]'
if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'final'
    if mode=='dev': print(json.dumps(phase('dev'),indent=1));sys.exit()
    d=phase('dev');c=phase('confirm');fz=json.load(open(f'{DOC}/frozen.json'))
    ver={}
    for a in IDEAS+['all']:
        if a not in d.get('cmp',{}): ver[a]='NO EVALUADA (M2 no aceptada en meta-val)' if a=='m2' else 'NO EVALUADA'; continue
        if d['cmp'][a]['stop']: ver[a]='STOP en DEV'; continue
        if a not in c.get('cmp',{}): ver[a]='DEV sin CONFIRM'; continue
        ver[a]='CONFIRMADA' if (d['cmp'][a]['pass'] and c['cmp'][a]['pass']) else 'NO CONFIRMADA'
    out={'dev':d,'confirm':c,'verdicts':ver};json.dump(out,open(f'{A}/analysis_dev_confirm.json','w'),indent=1)
    L=['# CONSTELACIÓN v4 — mejoras M1/M2/M3: RESULTADOS','',f'Preregistro: `{DOC}/PREREG.md`. Análisis: `scripts/analyze_constelacion_v4_mejoras.py` (bootstrap pareado 10000, semilla 12345). 0 consultas TEST.','',
       'Primaria: Δ mediana log rel_err h16, familias nuevas, k=4 (brazo − v4; negativo = mejor). IC98.33% (Bonferroni×3). No-daño: IC95 Δacc16 conocidas y Δacc h16 E33 > −0.05.','',
       '## Veredictos','','| idea | veredicto |','|---|---|']+[f'| {a} | {v} |' for a,v in ver.items()]
    for ph,r in (('DEV',d),('CONFIRM',c)):
        L+=['',f'## {ph}','',f"Brazos (n semillas): {r.get('arms')}. Reproducción hash v4 vs artifacts/constelacion_v4: {r.get('v4_hash_repro')}. v4 new med log medio {r.get('v4',{}).get('new_med_log_mean','—')}; ÓRBITA-v1 {r.get('orbita_new_med_log_mean','—')}.",'',
            '| brazo | Δ primaria IC98.33 | p Holm | Δacc16 conocidas IC95 | Δacc E33 h16 IC95 | vs ÓRBITA IC95 | h1 − persistencia IC95 | NaN | pass | stop |','|---|---|---|---|---|---|---|---|---|---|']
        for a,x in r.get('cmp',{}).items():
            if 'primary_new_medlog_minus_v4_ci9833' not in x: L.append(f"| {a} | Δ new med log IC95 {fmt(x['new_medlog_ci95'])} | | | | | | | | |");continue
            L.append(f"| {a} | {fmt(x['primary_new_medlog_minus_v4_ci9833'])} | {ph_(x)} | {fmt(x['noharm_known_acc16_ci95'])} | {fmt(x['noharm_e33_h16acc_ci95'])} | {fmt(x['sec_new_medlog_minus_orbita_ci95']) if 'sec_new_medlog_minus_orbita_ci95' in x else '—'} | {fmt(x['sec_new_h1_minus_persist_ci95'])} | {x['nan_seeds']} | {x['pass']} | {x['stop']} |")
    L+=['','## Valores congelados','','```json',json.dumps(fz,indent=1),'```','','JSON completo: `artifacts/constelacion_v4_mejoras/analysis_dev_confirm.json`.']
    open(f'{DOC}/RESULTADOS.md','w').write('\n'.join(L)+'\n');print(json.dumps(ver))
