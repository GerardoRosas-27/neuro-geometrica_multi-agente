"""results_md.py → docs/dinamica_latente/RESULTADOS.md a partir de analysis.json (DEV y, si existe, CONFIRM)."""
import json, os, sys
D = "/workspace/orbita-data/dinlat"; OUT = sys.argv[1]
def load(p):
    f = f"{D}/{p}/analysis.json"; return json.load(open(f)) if os.path.exists(f) else None
dev, con = load("dev"), load("confirm")
L = ["# RESULTADOS — exp/orbita-dinamica-latente", "", "Generado automáticamente desde analysis.json. Protocolo: docs/dinamica_latente/PREREG.md. 0 consultas a TEST.", ""]
for nm, A in (("DEV", dev), ("CONFIRM", con)):
    L.append(f"## {nm}")
    if A is None: L += ["falta (no ejecutado; ver criterio de parada)", ""]; continue
    L.append(f"Seeds con resultado: {A['n_seeds']}/16 · varianza explicada PCA media: {A['pca_var_explicada_media']:.4f}"); L.append("")
    L += ["| Contraste G=log(MSE_B/MSE_A) | media | IC95 |", "|---|---|---|"]
    for k, v in A["filas"].items(): L.append(f"| {k} | {v['media']:+.4f} | [{v['ic_lo']:+.4f}, {v['ic_hi']:+.4f}] |")
    L += ["", "Criterios: " + ", ".join(f"{k}={'sí' if v else 'no'}" for k, v in A["criterios"].items())]
    if "gate_dev_a_confirm" in A: L.append(f"Gate DEV→CONFIRM: {'pasa' if A['gate_dev_a_confirm'] else 'NO pasa → parada, CONFIRM no se ejecuta'}")
    L.append("")
h1 = lambda A: A and A["criterios"]["H1"]
v = "ÉXITO (H1 en DEV y CONFIRM)" if h1(dev) and h1(con) and con["criterios"]["V1_sin_ganancia_espuria_C1"] and con["criterios"]["V2_sin_ganancia_espuria_C2"] else \
    ("NO CONFIRMADO (parada en DEV)" if dev and not dev.get("gate_dev_a_confirm") else ("FRACASO de H1" if (dev and con and not (h1(dev) and h1(con))) else "no concluyente"))
L += ["## Veredicto", v, ""]
open(OUT, "w").write("\n".join(L)); print(v)
