"""Análisis prerregistrado (docs/prereg_cognitive_transformer_orbita.md). Uso: analyze_cto.py dev|confirm [OUT]"""
import sys, os, csv, json, collections
import numpy as np
OUT = os.environ.get("CTO_OUT", "/workspace/orbita-data/cto"); DELTA = 0.05
SEEDS = {"dev": [f"0xA4{i:02X}" for i in range(16)], "confirm": [f"0xB4{i:02X}" for i in range(16)]}
G = {"new": range(8, 12), "known": range(8), "new_8_9": (8, 9), "new_10_11": (10, 11), "known_1_2": (1, 2)}
ARMS = ["ORB", "CT", "CT_NOCDT", "CT_RAND", "CT_SHUF", "SELF", "MLP", "NOORB", "ORACLE_EP", "ORACLE_KNN"]
def boot(x, B=10000, seed=12345):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; r = np.random.default_rng(seed); m = x[r.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(x)
def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); pos, neg = s[y], s[~y]
    return (np.sum(pos[:, None] > neg[None]) + 0.5 * np.sum(pos[:, None] == neg[None])) / (len(pos) * len(neg))
def main(ph):
    per = collections.defaultdict(dict); meta = {}
    for s in SEEDS[ph]:
        p = f"{OUT}/{ph}/eval_{s}.csv"
        if not os.path.exists(p): continue
        R = list(csv.DictReader(open(p))); meta[s] = json.load(open(f"{OUT}/{ph}/meta_{s}.json"))
        for r in R:
            for kk in ("tag", "fam", "k"): r[kk] = int(r[kk])
        for k in (2, 4, 8):
            for n in (0, 150, 600, 2400):
                for g, fams in G.items():
                    X = [r for r in R if r["tag"] == n and r["k"] == k and r["fam"] in fams]
                    for a in ARMS:
                        for h in (8, 16, 32, 64, 128, 256): per[(k, n, g, a, h)][s] = np.mean([int(r[f"{a}_{h}"]) for r in X])
                        per[(k, n, g, a, "re16")][s] = np.median([float(r[f"{a}_re16"]) for r in X])
                X = [r for r in R if r["tag"] == n and r["k"] == k]
                y = [r["fam"] >= 8 for r in X]
                per[(k, n, "auroc_orb")][s] = auroc([r["orb_surp"] for r in X], y); per[(k, n, "auroc_ct")][s] = auroc([float(r["ct_surp"]) for r in X], y)
    S = [s for s in SEEDS[ph] if s in meta]
    v = lambda *key: np.array([per[key][s] for s in S])
    L = [f"# {ph.upper()} — n_seeds = {len(S)}", ""]
    def row(name, x, crit=None):
        b = boot(x); fav = int((np.asarray(x) > 0).sum()); L.append(f"| {name} | {b[0]:+.4f} | [{b[1]:+.4f}, {b[2]:+.4f}] | {fav}/{b[3]} | {crit or ''} |"); return b
    res = {}
    for k in (2, 4, 8):
        L += [f"## k = {k}{' (primario)' if k == 2 else ' (secundario)'} — acc16, n = 2400 salvo indicación", "", "| cantidad | media | IC95 | seeds > 0 | criterio |", "|---|---|---|---|---|"]
        a = lambda arm, g, n=2400, h=16: v(k, n, g, arm, h)
        res[(k, "H1")] = row("**H1** CT − ORB, conocidas 0–7", a("CT", "known") - a("ORB", "known"), "IC95 > 0")
        res[(k, "H2")] = row("**H2** CT − ORB, nuevas 8–11", a("CT", "new") - a("ORB", "new"), "IC95 > 0")
        res[(k, "H3")] = row("**H3** CT n=2400 − CT n=0, nuevas 8–11 (consolidación CDT)", a("CT", "new") - a("CT", "new", 0), "IC95 > 0")
        for g, gn in (("new", "nuevas"), ("known", "conocidas")):
            res[(k, "H4s", g)] = row(f"**H4** CT − CT_SHUF, {gn}", a("CT", g) - a("CT_SHUF", g), "IC95 > 0")
            res[(k, "H4r", g)] = row(f"**H4** CT − CT_RAND, {gn}", a("CT", g) - a("CT_RAND", g), "IC95 > 0")
            res[(k, "H5", g)] = row(f"**H5** CT − SELF, {gn}", a("CT", g) - a("SELF", g), "IC95 > 0")
            res[(k, "H6", g)] = row(f"**H6** CT − MLP (param. emparejado), {gn}", a("CT", g) - a("MLP", g), "IC95 > 0")
            res[(k, "H7", g)] = row(f"**H7** CT − NOORB (atención sin ÓRBITA), {gn}", a("CT", g) - a("NOORB", g), "IC95 > 0")
        res[(k, "H8e")] = row("**H8** CT − ORACLE_EP, nuevas", a("CT", "new") - a("ORACLE_EP", "new"), "IC95 > 0")
        res[(k, "H8k")] = row("**H8** CT − ORACLE_KNN, nuevas", a("CT", "new") - a("ORACLE_KNN", "new"), "IC95 > 0")
        res[(k, "H9")] = row("**H9** olvido: CT n=2400 − CT n=0, conocidas 0–7", a("CT", "known") - a("CT", "known", 0), "IC95 inf > −δ")
        res[(k, "H10")] = row("**H10** AUROC sorpresa CT n=2400 − n=0 (nuevas vs conocidas)", v(k, 2400, "auroc_ct") - v(k, 0, "auroc_ct"), "IC95 < 0")
        row("AUROC sorpresa ÓRBITA (nuevas vs conocidas)", v(k, 0, "auroc_orb")); row("AUROC sorpresa CT n=0", v(k, 0, "auroc_ct")); row("AUROC sorpresa CT n=2400", v(k, 2400, "auroc_ct"))
        row("doc §24: NOORB − ORB, conocidas", a("NOORB", "known") - a("ORB", "known")); row("doc §24: NOORB − ORB, nuevas", a("NOORB", "new") - a("ORB", "new"))
        L += ["", f"### acc por brazo (k = {k}, media de seeds; n = 2400 / n = 0)", "", "| brazo | " + " | ".join(f"{g} h16" for g in G) + " | conocidas h8 | conocidas h32 | conocidas h64 | conocidas h128 | conocidas h256 | nuevas h8 | nuevas h32 | rel_err16 mediana conocidas | nuevas |", "|---" * (len(G) + 10) + "|---|"]
        for arm in ARMS:
            L.append(f"| {arm} | " + " | ".join(f"{a(arm, g).mean():.4f} / {a(arm, g, 0).mean():.4f}" for g in G) + " | " + " | ".join(f"{a(arm, 'known', h=h).mean():.4f}" for h in (8, 32, 64, 128, 256))
                     + f" | {a(arm, 'new', h=8).mean():.4f} | {a(arm, 'new', h=32).mean():.4f} | {v(k, 2400, 'known', arm, 're16').mean():.4f} | {v(k, 2400, 'new', arm, 're16').mean():.4f} |")
        L.append("")
        L += [f"### CT por punto de control (k = {k})", "", "| n | CT nuevas | CT conocidas | ORACLE_KNN nuevas | CT_SHUF nuevas |", "|---|---|---|---|---|"]
        for n in (0, 150, 600, 2400): L.append(f"| {n} | {v(k, n, 'new', 'CT', 16).mean():.4f} | {v(k, n, 'known', 'CT', 16).mean():.4f} | {v(k, n, 'new', 'ORACLE_KNN', 16).mean():.4f} | {v(k, n, 'new', 'CT_SHUF', 16).mean():.4f} |")
        L.append("")
    m0 = meta[S[0]]
    L += ["## Coste y auditoría", "", f"- parámetros entrenables: {m0['n_params']} (ÓRBITA congelada: {m0['orb_params']})"]
    for a in m0["train_s"]: L.append(f"- tiempo de entrenamiento {a}: media {np.mean([meta[s]['train_s'][a] for s in S]):.0f} s/seed (1 hilo, nice 10)")
    L.append(f"- evaluación H (4 puntos de control x 3 k x 10 brazos, rollouts a 256 pasos) + flujo: media {np.mean([meta[s]['stream_eval_s'] for s in S]):.0f} s/seed")
    L.append(f"- sha256 ÓRBITA idéntico antes/después: {sum(meta[s]['orb_sha_before'] == meta[s]['orb_sha_after'] for s in S)}/{len(S)}; queries a TEST: {sum(meta[s]['queries_test'] for s in S)}")
    L.append(f"- banco CDT final (episodios): media {np.mean([meta[s]['events'][-1]['bank'] for s in S]):.1f}; fracción de familias nuevas en el banco: {np.mean([meta[s]['events'][-1]['new_frac'] for s in S]):.4f}")
    txt = "\n".join(L) + "\n"; print(txt)
    json.dump({"|".join(map(str, k)): list(map(float, b[:3])) for k, b in res.items()}, open(f"{OUT}/res_{ph}.json", "w"), indent=0)
if __name__ == "__main__": main(sys.argv[1])
