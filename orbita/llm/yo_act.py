"""«El yo actúa» (docs/preregistro_yo_act.md).  yo_act.py run PH | analyze PH"""
import sys, os, csv, collections, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, yo as Y
O = f"{P0.DATA}/yo_act"; ART = f"{P0.ROOT}/artifacts/orbita/yo_act"
def run(ph):
    os.makedirs(f"{O}/{ph}", exist_ok=True)
    for s in P0.SEEDS[ph]:
        dn = f"{O}/{ph}/{s}.done"
        if os.path.exists(dn): continue
        for part in ("dev", "test"):
            p = subprocess.run([P0.B, "yo-act", "--seed", s, "--weights", f"{P0.DATA}/{ph}/w_{s}.json", "--part", part, "--ctx", f"{Y.O}/{ph}/ctx_{part}_{s}.csv"], capture_output=True, check=True)
            open(f"{O}/{ph}/act_{part}_{s}.csv", "wb").write(p.stdout); open(f"{O}/{ph}/q_{part}_{s}.txt", "wb").write(p.stderr)
        open(dn, "w").write("ok\n"); print(ph, s, "ok", flush=True)
def acc(A, mask, k, h):
    return np.mean([int(a[f"obs{k}_ok{h}"]) if m else int(a[f"base_ok{h}"]) for a, m in zip(A, mask)])
def tau(score, A, k, higher):   # re-observa si score > τ (higher) o score < τ; maximiza acc16 − 0,05·tasa en calibración
    vals = np.r_[np.unique(score), score.min() - 1, score.max() + 1]; best = None
    for t in vals:
        m = score > t if higher else score < t; v = acc(A, m, k, 16) - 0.05 * m.mean()
        if best is None or v > best[0] + 1e-12 or (abs(v - best[0]) <= 1e-12 and m.mean() < best[2]): best = (v, t, m.mean())
    return best[1]
def analyze(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []; curves = collections.defaultdict(list); rng = np.random.default_rng(12345)
    for s in S:
        cal, ev = Y.load(ph, "dev", s), Y.load(ph, "test", s)
        Ac = list(csv.DictReader(open(f"{O}/{ph}/act_dev_{s}.csv"))); Ae = list(csv.DictReader(open(f"{O}/{ph}/act_test_{s}.csv")))
        assert [(a["idx"], a["text_id"]) for a in Ae] == [(r["idx"], r["text_id"]) for r in ev] and len(Ac) == len(cal)
        f = Y.fit(cal, Y.F_YO); pc, pe = f(cal), f(ev); sc = np.array([float(r["surprise"]) for r in cal]); se = np.array([float(r["surprise"]) for r in ev])
        n = len(ev); y0 = np.array([int(a["base_ok16"]) for a in Ae])
        for k in (1, 2):
            M = {"BASE": np.zeros(n, bool), "ALWAYS_OBS": np.ones(n, bool), "YO_OBS": pe < tau(pc, Ac, k, False), "SURPRISE_OBS": se > tau(sc, Ac, k, True), "ORACLE": y0 == 0}
            for c, m in M.items():
                for h in (8, 16): per[(c, k, h)][s] = acc(Ae, m, k, h)
                per[(c, k, "rate")][s] = m.mean()
            r = M["YO_OBS"].sum()
            for h in (8, 16): per[("RANDOM_OBS", k, h)][s] = np.mean([acc(Ae, np.isin(np.arange(n), rng.choice(n, r, replace=False)), k, h) for _ in range(200)])
            per[("RANDOM_OBS", k, "rate")][s] = r / n
            for name, sco in (("yo", pe), ("sorpresa", -se)):
                o = np.argsort(sco, kind="stable")
                curves[(name, k)].append([acc(Ae, np.isin(np.arange(n), o[:int(round(q * n))]), k, 16) for q in np.linspace(0, 1, 11)])
    v = lambda *key: np.array([per[key][s] for s in S]); Pr = lambda nm, x: out.append(f"| {nm} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    out += [f"## {ph.upper()} (16 seeds × 96 filas E2E_R)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for k in (1, 2):
        for c in ("BASE", "ALWAYS_OBS", "RANDOM_OBS", "YO_OBS", "SURPRISE_OBS", "ORACLE"):
            Pr(f"k={k} acc16 {c}", P0.boot(v(c, k, 16))); Pr(f"k={k} acc8 {c}", P0.boot(v(c, k, 8))); Pr(f"k={k} tasa re-observación {c}", P0.boot(v(c, k, "rate")))
            Pr(f"k={k} pasos reales para re-identificar {c}", P0.boot(k * v(c, k, "rate")))
        tag = "**" if k == 1 else ""
        Pr(f"{tag}HA1{tag} k={k} acc16 YO_OBS − RANDOM_OBS", P0.boot(v("YO_OBS", k, 16) - v("RANDOM_OBS", k, 16)))
        Pr(f"HA1 (sec.) k={k} acc8 YO_OBS − RANDOM_OBS", P0.boot(v("YO_OBS", k, 8) - v("RANDOM_OBS", k, 8)))
        Pr(f"{tag}HA2{tag} k={k} acc16 YO_OBS − ALWAYS_OBS (criterio: IC inf > −0,05)", P0.boot(v("YO_OBS", k, 16) - v("ALWAYS_OBS", k, 16)))
        Pr(f"{tag}HA2{tag} k={k} tasa YO_OBS (criterio: IC sup < 1)", P0.boot(v("YO_OBS", k, "rate")))
        Pr(f"HA3 k={k} acc16 YO_OBS − SURPRISE_OBS", P0.boot(v("YO_OBS", k, 16) - v("SURPRISE_OBS", k, 16)))
    out.append("\nCurva acc16 vs fracción re-observada (0, 0,1, …, 1), media de seeds:")
    for (name, k), L in sorted(curves.items()): out.append(f"- {name} k={k}: " + ", ".join(f"{x:.4f}" for x in np.mean(L, 0)))
    q = sum(int(open(f"{O}/{ph}/q_{pt}_{s}.txt").read().split("queries=")[1].split()[0]) for s in S for pt in ("dev", "test")); out.append(f"\nQueries v4 (yo-act): {q}")
    os.makedirs(ART, exist_ok=True); open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
