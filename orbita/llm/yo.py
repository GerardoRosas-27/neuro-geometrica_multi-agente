"""El yo de ÓRBITA + observador Gemma (docs/preregistro_yo.md).
  yo.py caltexts PH | feats PH | gen PH [--subset] | analyze PH | costprobe"""
import sys, os, csv, json, collections, subprocess, time, re, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, numnorm as NN, e2e as E
from make_texts import FAM, ART as ARTI
O = f"{P0.DATA}/yo"; ART = f"{P0.ROOT}/artifacts/orbita/yo"; EO = f"{P0.DATA}/llm_e2e"
LLAMA = P0.LLAMACPP; PORT = 18093
T89 = {"es": ["Hay que aplicar {art} {f}; el valor uno es {a} y el valor dos es {b}.", "Pedido: {art} {f}, con {a} como primer valor y {b} como segundo."],
       "en": ["Please apply {art} {f}; value one is {a} and value two is {b}.", "Order: {art} {f}, with {a} as the first value and {b} as the second."],
       "fr": ["Il faut appliquer {art} {f} ; la valeur un est {a} et la valeur deux est {b}.", "Commande : {art} {f}, avec {a} comme première valeur et {b} comme seconde."]}
def caltexts(ph):
    os.makedirs(f"{O}/{ph}", exist_ok=True); out = []
    for s in P0.SEEDS[ph]:
        d = subprocess.check_output([P0.B, "yo-feat", "--seed", s, "--weights", f"{P0.DATA}/{ph}/w_{s}.json", "--part", "dev", "--dump"]).decode()
        for r in csv.DictReader(d.splitlines()):
            fam, p1, p2 = int(r["fam"]), float(r["p1"]), float(r["p2"])
            for li, lang in enumerate(E.LANGS):
                for k in range(2):
                    fa, fb = E.fmts(r["idx"], k, li)
                    t = T89[lang][k].format(art=ARTI[lang][fam], f=FAM[lang][fam], a=NN.render(p1, lang, fa), b=NN.render(p2, lang, fb))
                    out.append([s, r["idx"], str(fam), r["p1"], r["p2"], lang, str(8 + k), f"{NN.FMT_NAMES[fa]}+{NN.FMT_NAMES[fb]}", t])
    open(f"{O}/{ph}/cal.tsv", "w").write("".join("\t".join(x) + "\n" for x in out)); open(f"{O}/{ph}/cal.txt", "w").write("".join(x[8] + "\n" for x in out))
    print(len(out), "textos de calibración", ph)
def d2_logits(tokpath, ph, seed, idxs):
    import torch, d2 as D2
    st = np.load(f"{E.D2D}/stats.npz"); tok = D2.Tok(tokpath); enc = D2.D2(); enc.load_state_dict(torch.load(f"{E.D2D}/{ph}/enc_D2_{seed}.pt"))
    with torch.no_grad(): c, lg, _ = enc(*tok.batch(idxs, st["mu"], st["sd"]))
    return c.numpy(), lg.numpy()
def feats(ph):
    for part, tsvp, tokp in (("dev", f"{O}/{ph}/cal.tsv", f"{O}/{ph}/cal.tok"), ("test", f"{EO}/{ph}/robust.tsv", f"{EO}/{ph}/robust.tok")):
        rows = P0.tsv(tsvp)
        for s in P0.SEEDS[ph]:
            I = [i for i, r in enumerate(rows) if r[0] == s]; C, LG = d2_logits(tokp, ph, s, I); lines = ["idx,cond,text_id," + ",".join(f"c{i}" for i in range(10))]; meta = []
            for j, i in enumerate(I):
                r = rows[i]; c, ok = E.build_c(C[j], r[8]); pr = np.exp(LG[j] - LG[j].max()); pr /= pr.sum(); so = np.sort(pr)[::-1]
                lines.append(f"{r[1]},E2E_R,{r[5]}-{r[6]}," + ",".join(repr(float(x)) for x in c)); meta.append((r[1], f"{r[5]}-{r[6]}", int(np.argmax(c[:8])), so[0], so[0] - so[1], int(ok)))
            cp = f"{O}/{ph}/ctx_{part}_{s}.csv"; open(cp, "w").write("\n".join(lines) + "\n")
            p = subprocess.run([P0.B, "yo-feat", "--seed", s, "--weights", f"{P0.DATA}/{ph}/w_{s}.json", "--part", part, "--ctx", cp], capture_output=True, check=True)
            R = list(csv.DictReader(p.stdout.decode().splitlines())); assert len(R) == len(meta)
            with open(f"{O}/{ph}/yo_{part}_{s}.csv", "w") as f:
                f.write("idx,text_id,fam,fam_hat,p_d2,margin_d2,parse_ok,surprise,sigma,normr,ok16\n")
                for r, mt in zip(R, meta):
                    assert r["idx"] == mt[0] and r["text_id"] == mt[1]
                    f.write(f"{r['idx']},{r['text_id']},{r['fam']},{mt[2]},{mt[3]:.6f},{mt[4]:.6f},{mt[5]},{np.log(float(r['surprise']) + 1e-9):.6f},{np.log(float(r['sigma'])):.6f},{np.log(float(r['normr'])):.6f},{r['ok16']}\n")
            open(f"{O}/{ph}/q_{part}_{s}.txt", "w").write(p.stderr.decode())
    print("feats ok", ph)
F_YO = ["p_d2", "margin_d2", "parse_ok", "surprise", "sigma", "normr"]
def load(ph, part, s): return list(csv.DictReader(open(f"{O}/{ph}/yo_{part}_{s}.csv")))
def fit(R, cols, lam=1e-2):
    X = np.array([[float(r[c]) for c in cols] for r in R]); y = np.array([int(r["ok16"]) for r in R])
    if y.min() == y.max(): pc = (y.sum() + 1) / (len(y) + 2); return lambda Q: np.full(len(Q), pc)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = np.c_[(X - mu) / sd, np.ones(len(X))]; w = np.zeros(Z.shape[1])
    for _ in range(200):   # Newton
        p = 1 / (1 + np.exp(-Z @ w)); g = Z.T @ (p - y) / len(y) + lam * np.r_[w[:-1], 0]
        H = (Z * (p * (1 - p))[:, None]).T @ Z / len(y) + lam * np.diag(np.r_[np.ones(len(w) - 1), 0]); w -= np.linalg.solve(H + 1e-12 * np.eye(len(w)), g)
    return lambda Q: 1 / (1 + np.exp(-(np.c_[(np.array([[float(r[c]) for c in cols] for r in Q]) - mu) / sd, np.ones(len(Q))] @ w)))
def phat(ph, s):
    cal, ev = load(ph, "dev", s), load(ph, "test", s)
    return ev, {"YO": fit(cal, F_YO)(ev), "MARGIN": fit(cal, ["margin_d2"])(ev), "SURPRISE": fit(cal, ["surprise"])(ev)}
NAMES_ES = FAM["es"]
def block(r, p, kind):
    if kind == "NOYO": return "No tienes acceso a su estado interno."
    u = kind == "EMPTY"; v = lambda x: "desconocido" if u else x
    return ("Este es su estado interno:\n" + f"- familia elegida por el encoder: {v(NAMES_ES[int(r['fam_hat'])])}\n" + f"- probabilidad del encoder para esa familia: {v(f'{float(r['p_d2']):.2f}')}\n"
            + f"- números leídos del texto: {v('sí' if r['parse_ok'] == '1' else 'no, se usó la estimación del encoder')}\n" + f"- sorpresa frente a la primera observación: {v(f'{np.exp(float(r['surprise'])):.3f}')}\n"
            + f"- estabilidad del rollout (sigma máx. del jacobiano): {v(f'{np.exp(float(r['sigma'])):.2f}')}\n" + f"- confianza estimada por el yo: {v(f'{round(100 * p)} %')}")
def prompt(b): return ("<start_of_turn>user\nEres el monitor de un modelo de predicción llamado ÓRBITA. " + b + "\nResponde exactamente con dos líneas:\nFamilia: <nombre de la transformación que ÓRBITA cree que es>\n"
                       "Confianza: <número entero de 0 a 100 de que ÓRBITA acierte>\n<end_of_turn>\n<start_of_turn>model\n")
def derange(n, seed=12345):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or (p != np.arange(n)).all(): return p
def server():
    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); return None
    except Exception: pass
    pr = subprocess.Popen([f"{LLAMA}/llama-server", "-m", E.GEMMA, "--port", str(PORT), "-t", "8", "-c", "2048", "-np", "1", "--no-webui"], stdout=open(f"{O}/server.log", "a"), stderr=subprocess.STDOUT,
                          env={**os.environ, "LD_LIBRARY_PATH": LLAMA})
    for _ in range(120):
        time.sleep(1)
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); return pr
        except Exception: pass
    raise RuntimeError("llama-server no arrancó")
def complete(text):
    q = json.dumps({"prompt": text, "n_predict": 40, "temperature": 0, "top_k": 1, "seed": 0, "cache_prompt": False}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", q, {"Content-Type": "application/json"}), timeout=300))
    return r["content"], r.get("timings", {})
def gen(ph, *opt):
    sub = "--subset" in opt; pr = server()
    for s in P0.SEEDS[ph]:
        dn = f"{O}/{ph}/gen_{s}.done"
        if os.path.exists(dn): continue
        ev, P = phat(ph, s); idx = [i for i, r in enumerate(ev) if not sub or r["text_id"] == f"{['es', 'en', 'fr'][int(r['idx']) % 3]}-6"]
        perm = derange(len(ev)); out = []
        for i in idx:
            r = ev[i]
            for kind in ("YO", "SHUF", "EMPTY", "NOYO"):
                src = ev[perm[i]] if kind == "SHUF" else r; ps = P["YO"][perm[i]] if kind == "SHUF" else P["YO"][i]
                t0 = time.time(); txt, tm = complete(prompt(block(src, ps, "YO" if kind == "SHUF" else kind)))
                out.append({"i": i, "idx": r["idx"], "text_id": r["text_id"], "kind": kind, "out": txt, "sec": time.time() - t0})
        json.dump(out, open(f"{O}/{ph}/gen_{s}.json", "w"), ensure_ascii=False); open(dn, "w").write("ok\n"); print(ph, s, "gen", len(out), flush=True)
    if pr: pr.terminate()
def costprobe():
    pr = server(); os.makedirs(O, exist_ok=True); fake = {"fam_hat": "1", "p_d2": "0.9", "parse_ok": "1", "surprise": "-3", "sigma": "0.1", "normr": "0"}; ts = []
    for k in ("YO", "EMPTY", "NOYO", "YO", "EMPTY", "NOYO"):
        t0 = time.time(); txt, tm = complete(prompt(block(fake, 0.8, k))); ts.append(time.time() - t0); print(k, repr(txt), f"{ts[-1]:.2f}s", flush=True)
    full = 12288 * np.mean(ts[1:]) / 3600; json.dump({"sec_per_gen": ts, "est_full_h": full, "est_subset_h": 2048 * np.mean(ts[1:]) / 3600}, open(f"{O}/cost_probe.json", "w"), indent=1)
    print("estimado completo h", round(full, 2)); pr and pr.terminate()
def readout(txt):
    t = txt.lower(); fam = None; conf = None
    m = re.search(r"familia\s*:\s*(.*)", t)
    if m:
        seg = m.group(1); best = -1
        for i, n in enumerate(NAMES_ES):
            if n in seg and len(n) > best: fam, best = i, len(n)
    m = re.search(r"confianza\s*:\s*(\d+)", t)
    if m: conf = min(100, max(0, int(m.group(1))))
    return fam, conf
def ece(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1); return sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any())
def aurc(p, y):
    o = np.argsort(-p, kind="stable"); e = 1 - y[o]; return np.mean(np.cumsum(e) / np.arange(1, len(e) + 1))
def analyze(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []
    for s in S:
        ev, P = phat(ph, s); y = np.array([int(r["ok16"]) for r in ev])
        for k, p in P.items(): per[("brier", k)][s] = np.mean((p - y) ** 2); per[("ece", k)][s] = ece(p, y); per[("aurc", k)][s] = aurc(p, y)
        per[("aurc", "RANDOM")][s] = 1 - y.mean(); per[("acc16",)][s] = y.mean()
        per[("auroc_yo_all",)][s] = P0.auroc(list(-P["YO"]), list(y == 0))
        gp = f"{O}/{ph}/gen_{s}.json"
        if os.path.exists(gp):
            G = json.load(open(gp)); perm = derange(len(ev)); byk = collections.defaultdict(list)
            for g in G:
                i = g["i"]; fam, conf = readout(g["out"]); byk[g["kind"]].append((i, fam, conf))
            for k, L in byk.items():
                per[("famrec", k)][s] = np.mean([f is not None and f == int(ev[i]["fam_hat"]) for i, f, c in L])
                per[("conferr", k)][s] = np.mean([1.0 if c is None else abs(c / 100 - P["YO"][i]) for i, f, c in L])
                per[("readable", k)][s] = np.mean([f is not None and c is not None for i, f, c in L])
            L = byk["YO"]; ii = [i for i, f, c in L]
            per[("auroc_g",)][s] = P0.auroc([-(c if c is not None else 50) for i, f, c in L], [y[i] == 0 for i in ii])
            per[("auroc_p",)][s] = P0.auroc(list(-P["YO"][ii]), [y[i] == 0 for i in ii]); per[("ngen",)][s] = len(G); per[("secgen",)][s] = np.mean([g["sec"] for g in G])
    v = lambda *k: np.array([per[k].get(s, np.nan) for s in S]); out += [f"## {ph.upper()} (16 seeds)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    def Pr(n, x):
        x = np.asarray(x, float); ok = ~np.isnan(x)
        if ok.sum() == 0: out.append(f"| {n} | nan | | 0 |"); return
        b = P0.boot(x[ok]); out.append(f"| {n} | {b[0]:+.4f} | [{b[1]:+.4f}, {b[2]:+.4f}] | {b[3]} |")
    Pr("acc16 E2E_R (tasa de acierto real)", v("acc16"))
    for m_ in ("brier", "ece", "aurc"):
        for k in ("YO", "MARGIN", "SURPRISE") + (("RANDOM",) if m_ == "aurc" else ()): Pr(f"{m_} {k}", v(m_, k))
    Pr("**H1** Brier: min(MARGIN, SURPRISE) − YO", np.minimum(v("brier", "MARGIN"), v("brier", "SURPRISE")) - v("brier", "YO"))
    Pr("H1 (sec.) ECE: min(MARGIN, SURPRISE) − YO", np.minimum(v("ece", "MARGIN"), v("ece", "SURPRISE")) - v("ece", "YO"))
    Pr("**H2** AURC: min(MARGIN, SURPRISE, RANDOM) − YO", np.minimum.reduce([v("aurc", "MARGIN"), v("aurc", "SURPRISE"), v("aurc", "RANDOM")]) - v("aurc", "YO"))
    Pr("AUROC(−p̂ YO → fallo), 96 filas", v("auroc_yo_all"))
    if per[("ngen",)]:
        for k in ("YO", "SHUF", "EMPTY", "NOYO"):
            Pr(f"familia recuperada {k}", v("famrec", k)); Pr(f"|conf/100 − p̂| {k}", v("conferr", k)); Pr(f"frase legible {k}", v("readable", k))
        Pr("**H3a** familia: YO − max(SHUF, EMPTY, NOYO)", v("famrec", "YO") - np.maximum.reduce([v("famrec", k) for k in ("SHUF", "EMPTY", "NOYO")]))
        Pr("**H3b** error conf: min(SHUF, EMPTY, NOYO) − YO", np.minimum.reduce([v("conferr", k) for k in ("SHUF", "EMPTY", "NOYO")]) - v("conferr", "YO"))
        Pr("AUROC(−conf Gemma → fallo)", v("auroc_g")); Pr("AUROC(−p̂ → fallo), mismas filas", v("auroc_p"))
        Pr("**H4** AUROC Gemma − AUROC p̂", v("auroc_g") - v("auroc_p")); Pr("generaciones por seed", v("ngen")); Pr("s por generación", v("secgen"))
    q = sum(int(open(f"{O}/{ph}/q_{pt}_{s}.txt").read().split("queries=")[1].split()[0]) for s in S for pt in ("dev", "test"))
    out.append(f"\nQueries v4 (yo-feat, dev+test): {q}")
    os.makedirs(ART, exist_ok=True); open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__":
    c = sys.argv[1]
    {"caltexts": caltexts, "feats": feats, "gen": gen, "analyze": analyze, "costprobe": costprobe}[c](*sys.argv[2:])
