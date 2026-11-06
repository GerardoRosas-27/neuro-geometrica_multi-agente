#!/usr/bin/env python3
"""Paired seed-bootstrap and report only; model training lives entirely in Rust."""
import csv, json, pathlib, random, statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/soft_bellman_v1'
rows=[]
for path in sorted((DOC/'raw').glob('*/*/metrics.csv')):
    for r in csv.DictReader(path.open()):
        for k in ['nll','accuracy','success','regret','eval_ms']: r[k]=float(r[k])
        for k in ['seed','maps','states']:r[k]=int(r[k])
        rows.append(r)
assert len(rows)==32*16, f'Expected 512 rows, got {len(rows)}'
for phase in ['DEV','CONFIRM']:
    assert len({r['seed'] for r in rows if r['phase']==phase})==16
with (DOC/'metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def quantile(xs,p):
    xs=sorted(xs); i=(len(xs)-1)*p;lo=int(i);hi=min(lo+1,len(xs)-1);return xs[lo]+(i-lo)*(xs[hi]-xs[lo])
def bootstrap(d):
    rng=random.Random(12345); n=len(d)
    means=[sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(10000)]
    return {'difference':statistics.mean(d),'ci95':[quantile(means,.025),quantile(means,.975)],'seeds':n}
summary={'bootstrap':{'resamples':10000,'seed':12345,'delta':.05,'unit':'independent trained model seed; paired maps within seed','method':'percentile with linearly interpolated quantiles; CPython random.Random'},'phases':{}}
for phase in ['DEV','CONFIRM']:
    ps={};rs=[r for r in rows if r['phase']==phase]
    for split in ['train','val','test','ood3d']:
        ss={};lookup={}
        for model in ['field','untrained','constant','mlp']:
            a=sorted([r for r in rs if r['split']==split and r['model']==model],key=lambda r:r['seed']);lookup[model]=a
            ss[model]={k:statistics.mean(r[k] for r in a) for k in ['nll','accuracy','success','regret','eval_ms']}
            ss[model]['maps_total']=sum(r['maps'] for r in a);ss[model]['states_total']=sum(r['states'] for r in a)
        ss['paired']={}
        for control in ['untrained','constant','mlp']:
            ss['paired'][control]={}
            for metric in ['regret','nll','success']:
                assert [r['seed'] for r in lookup['field']]==[r['seed'] for r in lookup[control]]
                d=[a[metric]-b[metric] for a,b in zip(lookup['field'],lookup[control])]
                ss['paired'][control][metric]=bootstrap(d)
        ps[split]=ss
    ps['runtime']=json.loads((DOC/'raw'/phase/'run.json').read_text())
    summary['phases'][phase]=ps
s=summary['phases']['CONFIRM']['test'];summary['primary_pass']=s['paired']['constant']['regret']['ci95'][1]<0 and s['field']['success']>=s['constant']['success']-1e-10
(DOC/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'primary_pass':summary['primary_pass'],'DEV':summary['phases']['DEV']['test'],'CONFIRM':s,'OOD3D':summary['phases']['CONFIRM']['ood3d']},indent=2))
