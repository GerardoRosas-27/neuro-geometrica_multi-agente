#!/usr/bin/env python3
import json
r=json.load(open('artifacts/bio_constelacion/analysis.json'))
L=["# RESULTADOS — BIO-CONSTELACIÓN bajo protocolo CONSTELACIÓN v4","","Generado automáticamente por la cadena a partir de `artifacts/bio_constelacion/analysis.json` (PREREG: `PREREG.md`). Veredicto = CONFIRM.",""]
f=lambda b:f"{b['delta']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]"
for ph in ['dev','confirm']:
    x=r[ph]; L+= [f"## {ph.upper()}","","| brazo | new_med_log | known_med_log | new_acc16 | known_acc16 | E33 acc h1 |","|---|---|---|---|---|---|"]
    for a,m in x['arms_median'].items(): L.append(f"| {a} | {m['new_med_log']:.3f} | {m['known_med_log']:.3f} | {m['new_acc16']:.3f} | {m['known_acc16']:.3f} | {x['e33_h1_median'][a]:.3f} |")
    L+=[f"| ÓRBITA-v1 | {x['orbita_new_med_log']:.3f} | | | | |",f"| persistencia | {x['persist_new_med_log']:.3f} | | | | |","",
     f"- **Primaria** hyb − v4 (new_med_log, mediana [IC95]): {f(x['primary_hyb_minus_v4'])} → {'PASA' if x['primary_pass'] else 'NO PASA'}",
     f"- hyb − ÓRBITA-v1: {f(x['sec_hyb_minus_orbita'])}", f"- hyb − persistencia: {f(x['sec_hyb_minus_persist'])}", f"- hyb − hyb_nobio (ablación bio): {f(x['sec_hyb_minus_nobio'])}",
     f"- No-daño known_acc16: {f(x['noharm_known_acc16'])}; E33 h16: {f(x['noharm_e33_h16'])} → {'OK' if x['noharm_pass'] else 'DAÑO'}",
     f"- v4 reproduce hashes de exp/constelacion-v4-mejoras: {x['v4_hash_match_mejoras']}",""]
open('docs/bio_constelacion/RESULTADOS.md','w').write('\n'.join(L))
