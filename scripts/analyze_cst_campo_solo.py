#!/usr/bin/env python3
"""Análisis exploratorio CST v4 campo solo (docs/cst_campo_solo/PREREG.md). Uso: analyze_cst_campo_solo.py dev"""
import json, glob, sys
import numpy as np
ph = sys.argv[1] if len(sys.argv) > 1 else 'dev'
R = [json.load(open(f)) for f in sorted(glob.glob(f'artifacts/cst_campo_solo/{ph}/metrics_*.json'))]
def boot(d, B=10000, seed=12345):
    d = np.asarray(d, float); rng = np.random.default_rng(seed); i = rng.integers(0, len(d), (B, len(d)))
    m = d[i].mean(1); return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def met(r, a):
    if a == 'persist': return {'new': r['trivial']['new_med_log'], 'known_acc16': None, 'e33_region': r['e33_persist']['region'], 'e33_acc': float(np.mean(r['e33_persist']['acc']))}
    v = r['arms'][a]; return {'new': v['held']['new_med_log'], 'known_acc16': v['held']['known_acc16'], 'e33_region': v['e33']['region'], 'e33_acc': float(np.mean(v['e33']['acc'])), 'nan': v['nan'], 'hash': v['hash']}
arms = ['cst', 'field_only', 'no_orb', 'persist']
M = {a: [met(r, a) for r in R] for a in arms}
out = {'phase': ph, 'n': len(R), 'seeds': [r['seed'] for r in R], 'test_queries': sum(r['test_queries'] for r in R), 'wall_s': [r['wall_s'] for r in R], 'per_seed': M, 'summary': {}, 'cmp': {}}
for a in arms:
    s = {k: boot([m[k] for m in M[a]]) for k in ['new', 'e33_region', 'e33_acc'] + ([] if a == 'persist' else ['known_acc16'])}
    s['new_median'] = float(np.median([m['new'] for m in M[a]])); s['e33_pass'] = sum(m['e33_region'] >= 16 for m in M[a])
    if a != 'persist': s['nan'] = sum(m['nan'] for m in M[a])
    out['summary'][a] = s
for x, y in [('field_only', 'cst'), ('field_only', 'persist'), ('cst', 'persist'), ('no_orb', 'cst'), ('field_only', 'no_orb')]:
    c = {}
    for k in ['new', 'known_acc16', 'e33_region', 'e33_acc']:
        if M[x][0].get(k) is None or M[y][0].get(k) is None: continue
        c[k] = boot([a[k] - b[k] for a, b in zip(M[x], M[y])])
    out['cmp'][f'{x}-{y}'] = c
print(json.dumps(out, indent=1))
