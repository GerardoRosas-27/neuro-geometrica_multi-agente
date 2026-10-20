"""Análisis ÓRBITA-TF (prerregistro docs/prereg_orbita_transformer_fe.md). Uso: analyze_tf.py dev|confirm → markdown en stdout.
IC95 bootstrap pareado por seed (10 000 remuestreos, numpy default_rng(12345)), n = 16 seeds."""
import csv, json, sys, os, re, numpy as np
ph = sys.argv[1]; W = f"/workspace/orbita-data/tf/{ph}"; pre = "0xA4" if ph == "dev" else "0xB4"
SEEDS = [f"{pre}{i:02X}" for i in range(16)]; DELTA = 0.05
L = lambda p: list(csv.DictReader(open(p)))
def auroc(score, lab):
    score, lab = np.asarray(score, float), np.asarray(lab, int)
    if lab.min() == lab.max(): return np.nan
    r = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort").astype(float)
    # rangos medios para empates
    u, inv = np.unique(score, return_inverse=True)
    for j in range(len(u)):
        m = inv == j
        if m.sum() > 1: r[m] = r[m].mean()
    n1 = lab.sum(); n0 = len(lab) - n1
    return (r[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
V = {}  # nombre -> lista por seed
def put(k, v): V.setdefault(k, []).append(v)
cost = {}
for s in SEEDS:
    ev = {a: L(f"{W}/{a}/eval_{s}.csv") for a in ("TFE", "TF", "MLP")}
    fam = L(f"{W}/ORBITA/fam_{s}.csv"); lat = {k: L(f"{W}/ORBITA/lat_{s}_k{k}.csv") for k in (2, 4, 8)}
    kn = lambda rs: [r for r in rs if int(r["fam"]) < 8]; nw = lambda rs: [r for r in rs if int(r["fam"]) >= 8]
    mean = lambda rs, c: float(np.mean([float(r[c]) for r in rs]))
    for a in ("TFE", "TF", "MLP"):
        put(f"{a} conocidas acc16 (c verdadero)", mean(kn(ev[a]), "true_ok16")); put(f"{a} conocidas acc8 (c verdadero)", mean(kn(ev[a]), "true_ok8"))
    put("ORBITA conocidas acc16 (c verdadero)", mean(kn(fam), "true_ok16")); put("ORBITA conocidas acc8 (c verdadero)", mean(kn(fam), "true_ok8"))
    for k in (2, 4, 8):
        for grp, g in (("nuevas", nw), ("conocidas", kn)):
            put(f"TFE-FE {grp} k={k}", mean(g(ev["TFE"]), f"fe_k{k}")); put(f"TFE-PE {grp} k={k}", mean(g(ev["TFE"]), f"pe_k{k}"))
            put(f"TFE-OBS {grp} k={k}", mean(g(ev["TFE"]), f"obs_k{k}")); put(f"TF-PE {grp} k={k}", mean(g(ev["TF"]), f"pe_k{k}"))
            put(f"TF-OBS {grp} k={k}", mean(g(ev["TF"]), f"obs_k{k}")); put(f"MLP-OBS {grp} k={k}", mean(g(ev["MLP"]), f"obs_k{k}"))
            for c, n in (("obs16", "OBS"), ("latep16", "LAT_EP"), ("latmem16", "LAT"), ("ls16", "ORACLE_LS")):
                put(f"ORBITA-{n} {grp} k={k}", mean(g(lat[k]), c))
        e = ev["TFE"]; put(f"AUROC TFE F k={k}", auroc([float(r[f"Ffe_k{k}"]) for r in e], [1 - int(float(r[f"fe_k{k}"])) for r in e]))
        e = ev["TF"]; put(f"AUROC TF F_PE k={k}", auroc([float(r[f"Fpe_k{k}"]) for r in e], [1 - int(float(r[f"pe_k{k}"])) for r in e]))
        o = lat[k]; put(f"AUROC ORBITA s_known→OBS k={k}", auroc([float(r["s_known"]) for r in o], [1 - int(r["obs16"]) for r in o]))
        put(f"AUROC ORBITA s_known→LAT_EP k={k}", auroc([float(r["s_known"]) for r in o], [1 - int(r["latep16"]) for r in o]))
    for a in ("TFE", "TF", "MLP"):
        lg = open(f"{W}/{a}/train_{s}.log").read(); secs = [int(x) for x in re.findall(r" (\d+)s\n", lg)]
        cost.setdefault(f"{a} entrenamiento: segundos de pared del último proceso (256 000 pasos si no hubo reanudación)", []).append(None if not secs else secs[-1])
        t = json.load(open(f"{W}/{a}/eval_{s}.csv.time.json"))
        for k in (2, 4, 8):
            for m_ in ("obs", "fe", "pe"):
                if f"{m_}_k{k}" in t: cost.setdefault(f"{a} {m_.upper()} k={k} s/768 episodios", []).append(t[f"{m_}_k{k}"])
        cost.setdefault(f"{a} params", []).append(t["n_params"])
    lo_ = re.findall(r"step 256000 .* (\d+)s", open(f"/workspace/orbita-data/{ph}/log_{s}.txt").read()) if os.path.exists(f"/workspace/orbita-data/{ph}/log_{s}.txt") else []
    cost.setdefault("ORBITA entrenamiento publicado: segundos de pared (log original)", []).append(float(lo_[-1]) if lo_ else None)
    for k in (2, 4, 8): cost.setdefault(f"ORBITA lat-eval exp1 k={k} s/768 episodios (incluye OBS, LAT, LAT_EP, mallas oráculo)", []).append(float(open(f"{W}/ORBITA/lat_{s}_k{k}.time").read()))
rng = np.random.default_rng(12345)
IDX = rng.integers(0, 16, (10000, 16))
def ci(x):
    x = np.asarray(x, float); ok = ~np.isnan(x); xs = x[ok]; n = len(xs)
    if n == 0: return np.nan, np.nan, np.nan, 0
    bi = np.random.default_rng(12345).integers(0, n, (10000, n)); b = xs[bi].mean(1)
    return xs.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), n
f4 = lambda v: f"{v:+.4f}"
print(f"# ÓRBITA-TF — análisis {ph.upper()} (n = 16 seeds; IC95 bootstrap pareado 10 000, semilla 12345)\n")
print("## Medias por brazo\n\n| cantidad | media | IC95 |\n|---|---|---|")
for k_, v in V.items():
    m, lo, hi, n = ci(v); print(f"| {k_} | {m:.4f} | [{lo:.4f}, {hi:.4f}]" + (f" (n={n})" if n < 16 else "") + " |")
def diff(a, b): return np.asarray(V[a], float) - np.asarray(V[b], float)
H = [("H1", "TFE conocidas acc16 (c verdadero)", "ORBITA conocidas acc16 (c verdadero)")]
for k in (2, 4, 8):
    tag = "" if k == 2 else f" (secundario k={k})"
    H += [(f"H2a{tag}", f"TFE-FE nuevas k={k}", f"ORBITA-LAT_EP nuevas k={k}"), (f"H2b{tag}", f"TFE-FE nuevas k={k}", f"ORBITA-OBS nuevas k={k}"),
          (f"H2c sec.{tag}", f"TFE-FE nuevas k={k}", f"ORBITA-LAT nuevas k={k}"),
          (f"H3b{tag}", f"AUROC TFE F k={k}", f"AUROC ORBITA s_known→OBS k={k}"),
          (f"H4a{tag}", f"TFE-FE nuevas k={k}", f"TF-PE nuevas k={k}"), (f"H4b{tag}", f"TFE-FE nuevas k={k}", f"TFE-OBS nuevas k={k}"),
          (f"H4c sec.{tag}", f"TFE-FE nuevas k={k}", f"TFE-PE nuevas k={k}"), (f"C-MLP{tag}", f"TFE-FE nuevas k={k}", f"MLP-OBS nuevas k={k}"),
          (f"C-TF-OBS{tag}", f"TFE-FE nuevas k={k}", f"TF-OBS nuevas k={k}"),
          (f"sec. conocidas{tag}", f"TFE-FE conocidas k={k}", f"ORBITA-LAT_EP conocidas k={k}")]
H += [("H4d", "TFE conocidas acc16 (c verdadero)", "TF conocidas acc16 (c verdadero)"), ("C-MLP conocidas", "TFE conocidas acc16 (c verdadero)", "MLP conocidas acc16 (c verdadero)"),
      ("sec. TF−ÓRBITA conocidas", "TF conocidas acc16 (c verdadero)", "ORBITA conocidas acc16 (c verdadero)")]
print("\n## Diferencias pareadas\n\n| id | A − B | media | IC95 | seeds A>B |\n|---|---|---|---|---|")
for h, a, b in H:
    d = diff(a, b); m, lo, hi, n = ci(d); print(f"| {h} | {a} − {b} | {f4(m)} | [{f4(lo)}, {f4(hi)}]" + (f" (n={n})" if n < 16 else "") + f" | {int(np.nansum(d > 0))}/{n} |")
m, lo, hi, n = ci(np.asarray(V["AUROC TFE F k=2"], float) - 0.5); print(f"| H3a | AUROC TFE F k=2 − 0.5 (umbral +δ = 0.05) | {f4(m)} | [{f4(lo)}, {f4(hi)}] (n={n}) | |")
print("\n## Coste (media por seed)\n\n| cantidad | media | min | max |\n|---|---|---|---|")
for k_, v in cost.items():
    v = [x for x in v if x is not None]
    if not v: print(f"| {k_} | sin dato | | |"); continue
    print(f"| {k_} | {np.mean(v):.2f} | {np.min(v):.2f} | {np.max(v):.2f} |")
json.dump({"per_seed": V, "cost": cost, "seeds": SEEDS}, open(f"/workspace/ngm-orbita/artifacts/orbita/transformer_fe/per_seed_{ph}.json", "w"), indent=1)
