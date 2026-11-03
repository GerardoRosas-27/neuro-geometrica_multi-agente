# Revisión rápida (exploratoria): CST v4 vs ÓRBITA-v1 con artefactos ya entrenados, sin reentrenar.
import json,glob,random,sys
H=['h1','h2','h4','h8','h16']
def boot(d,B=10000,s=12345):
    r=random.Random(s);n=len(d);m=[sum(d[r.randrange(n)] for _ in range(n))/n for _ in range(B)];m.sort()
    return sum(d)/n,m[int(.025*B)],m[int(.975*B)-1]
out={}
for ph in ['dev','confirm']:
    rows=[json.load(open(f)) for f in sorted(glob.glob(f'artifacts/constelacion_v4/{ph}/metrics_0x*.json'))]
    o={'n':len(rows)}
    for i,h in enumerate(H):
        c=[r['arms']['cst']['e33']['acc'][i] for r in rows];b=[r['e33_orbita']['acc'][i] for r in rows]
        o[f'e33_acc_{h}']={'cst':sum(c)/len(c),'orbita':sum(b)/len(b),'diff':boot([x-y for x,y in zip(c,b)])}
    o['e33_region']={'cst':[r['arms']['cst']['e33']['region'] for r in rows],'orbita':[r['e33_orbita']['region'] for r in rows]}
    o['e33_pass']={'cst':sum(r['arms']['cst']['e33']['pass'] for r in rows),'orbita':sum(r['e33_orbita']['pass'] for r in rows)}
    for k in ['new_med_log','new_acc16','known_med_log','known_acc16']:
        c=[r['arms']['cst']['held'][k] for r in rows];b=[r['orbita'][k] for r in rows]
        o[k]={'cst':sum(c)/len(c),'orbita':sum(b)/len(b),'diff':boot([x-y for x,y in zip(c,b)])}
    out[ph]=o
json.dump(out,open('artifacts/cst_vs_orbita_quick.json','w'),indent=1);print(json.dumps(out,indent=1))
