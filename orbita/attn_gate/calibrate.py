"""Calibración del umbral de sorpresa τ_k (sólo DEV): percentil q de s_known de ÓRBITA en episodios TUNE de familias CONOCIDAS 0–7,
agrupando las 16 seeds DEV. Primario q = 0.95 (do-no-harm: el corrector se activa en ≤ 5 % de episodios conocidos). Escribe tau.json."""
import csv, json, sys, numpy as np
W = sys.argv[1]; SEEDS = [f"0xA4{i:02X}" for i in range(16)]; out = {}
for k in (2, 4, 8):
    s = np.concatenate([[float(r[f"s_k{k}"]) for r in csv.DictReader(open(f"{W}/dev/AG/tune_{sd}.csv")) if int(r["fam"]) < 8] for sd in SEEDS])
    for q in (0.90, 0.95, 0.99): out[f"k{k}_q{q:.2f}"] = float(np.quantile(s, q))
json.dump(out, open(f"{W}/tau.json", "w"), indent=1); print(json.dumps(out))
