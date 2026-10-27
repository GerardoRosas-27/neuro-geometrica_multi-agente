"""vstimesfm/results_md.py OUT.md — RESULTADOS.md desde analysis.json (sin números a mano)."""
import sys, json, os, subprocess
B = "/workspace/orbita-data/timesfm"
def sec(ph):
    p = f"{B}/{ph}/analysis.json"
    if not os.path.exists(p): return f"## {ph.upper()}\nfalta (no ejecutado; ver criterio de parada)\n"
    A = json.load(open(p)); L = [f"## {ph.upper()}", f"Seeds con resultado: {A['n_seeds']}/16", "",
        "| Contraste G=log(MSE_B/MSE_A) | media | IC95 |", "|---|---|---|"]
    L += [f"| {k} | {v['media']:+.4f} | [{v['ic_lo']:+.4f}, {v['ic_hi']:+.4f}] |" for k, v in A["filas"].items()]
    L += ["", "Criterios: " + ", ".join(f"{k}={'sí' if v else 'no'}" for k, v in A["criterios"].items())]
    if "gate_dev_a_confirm" in A: L.append("Gate DEV→CONFIRM: " + ("pasa → CONFIRM" if A["gate_dev_a_confirm"] else "NO pasa → parada, CONFIRM no se ejecuta"))
    return "\n".join(L) + "\n"
v = subprocess.run(["/workspace/timesfm-venv/bin/pip", "show", "timesfm"], capture_output=True, text=True).stdout.split("\n")[1]
txt = ["# RESULTADOS — exp/orbita-vs-timesfm", "",
       f"Generado automáticamente desde analysis.json. Protocolo: docs/orbita_vs_timesfm/PREREG.md. 0 consultas a TEST. timesfm {v}, checkpoint google/timesfm-2.5-200m-pytorch, CPU.",
       "Latentes: reconstruidos de los embeddings Gemma cacheados de dinlat; verificación automática PERS/AR2 idénticos a res_*.json de dinlat.", "",
       sec("dev"), sec("confirm")]
open(sys.argv[1], "w").write("\n".join(txt))
