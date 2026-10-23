"""ÓRBITA-Bytes vs ÓRBITA (Gemma + D2/D3-num). Prerregistro docs/prereg_orbita_bytes.md. Mismo núcleo ÓRBITA-v1 congelado (w_<seed>.json) en ambos brazos.
  bytes_exp.py data              conjuntos de evaluación U (pt/it/ru, no vistos) y N (ruido BLT) a partir de robust.tsv; manifiesto SHA-256
  bytes_exp.py train PH VARIANT  16 seeds (paralelo NPAR) → models/PH/VARIANT/SEED
  bytes_exp.py eval PH VARIANT   ctx (R/U/N) + llm-eval por seed (VARIANT=GEMMA: brazo base, D2 sobre tokens Gemma cacheados)
  bytes_exp.py learn PH ARM      H6: learn-eval k=2 (τ y config seleccionados del experimento de aprendizaje online, sin re-ajustar)
  bytes_exp.py recon PH          H7: fidelidad de reconstrucción (MAIN)
  bytes_exp.py latency PH        H5: latencia y memoria pico por consulta (32 textos, subprocess por consulta, ambos brazos)
  bytes_exp.py analyze PH        tablas + IC95 bootstrap pareado (10000, semilla 12345)"""
import sys, os, csv, json, collections, subprocess, hashlib, random, zlib, time, resource
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, numnorm as NN, e2e as E
BD = os.environ.get("BYTES_DATA", f"{P0.DATA}/bytes"); LD = f"{P0.DATA}/learn"; ART = f"{P0.ROOT}/artifacts/orbita/bytes"
NPAR = int(os.environ.get("NPAR", "4")); PY = sys.executable; HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["MAIN", "NOVAE", "NOHASH", "NONCE", "FIXPATCH"]; SETS = ("R", "U", "N")
UL = {"pt": (["translação", "rotação", "reflexão", "escala", "cisalhamento", "transformação afim", "rotação com escala", "deformação radial"],
             ["uma", "uma", "uma", "uma", "um", "uma", "uma", "uma"],
             ["Transforme a figura com {art} {f}: o primeiro valor é {a} e o segundo é {b}.", "Pede-se {art} {f} com valores {a} e {b}."]),
      "it": (["traslazione", "rotazione", "riflessione", "scalatura", "taglio", "trasformazione affine", "rotazione con scalatura", "deformazione radiale"],
             ["una", "una", "una", "una", "un", "una", "una", "una"],
             ["Trasforma la figura con {art} {f}: il primo valore è {a} e il secondo è {b}.", "Si richiede {art} {f} con valori {a} e {b}."]),
      "ru": (["перенос", "поворот", "отражение", "масштабирование", "сдвиг", "аффинное преобразование", "поворот с масштабированием", "радиальная деформация"],
             [""] * 8, ["Преобразуйте фигуру: {f}, первое значение {a}, второе {b}.", "Требуется {f} со значениями {a} и {b}."])}
NOISE = ["randcase", "delete10", "repeat10", "allcaps"]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def noisy(t, kind, rng):
    if kind == "randcase": return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in t)
    if kind == "delete10": return "".join(c for c in t if rng.random() >= 0.10)
    if kind == "repeat10": return "".join(c * 2 if rng.random() < 0.10 else c for c in t)
    return t.upper()
def rpath(ph, st): return f"{P0.DATA}/llm_e2e/{ph}/robust" if st == "R" else f"{BD}/sets/{ph}/{st}"
def rows_of(ph, st): return P0.tsv(rpath(ph, st) + ".tsv")
def data():
    man = {}
    for ph in ("dev", "confirm"):
        os.makedirs(f"{BD}/sets/{ph}", exist_ok=True); R = rows_of(ph, "R"); seen = set(); U, N = [], []
        for j, r in enumerate(R):
            rng = random.Random(zlib.crc32(f"{r[0]}|{r[1]}|{r[5]}|{r[6]}".encode())); kind = NOISE[j % 4]
            N.append(r[:7] + [kind, noisy(r[8], kind, rng)])
            if (r[0], r[1]) in seen: continue
            seen.add((r[0], r[1])); fam, p1, p2 = int(r[2]), float(r[3]), float(r[4])
            for lang, (names, arts, tm) in UL.items():
                for k in range(2):
                    a, b = f"{p1:.2f}".replace(".", ","), f"{p2:.2f}".replace(".", ",")
                    U.append(r[:5] + [lang, str(8 + k), "digitos_coma", " ".join(tm[k].format(art=arts[fam], f=names[fam], a=a, b=b).split())])
        for st, X in (("U", U), ("N", N)):
            p = f"{BD}/sets/{ph}/{st}"; open(p + ".tsv", "w").write("".join("\t".join(x) + "\n" for x in X)); open(p + ".txt", "w").write("".join(x[8] + "\n" for x in X))
            man[f"{ph}/{st}.tsv"] = sha(p + ".tsv")
        man[f"{ph}/R(robust.tsv)"] = sha(rpath(ph, "R") + ".tsv"); man[f"{ph}/learn texts.tsv"] = sha(f"{LD}/{ph}/texts.tsv")
        print(ph, "U", len(U), "N", len(N))
    man["pool2.tsv"] = sha(f"{P0.DATA}/llm_d2/pool2.tsv"); json.dump(man, open(f"{BD}/sets/manifest.json", "w"), indent=1); print(json.dumps(man, indent=1))

def train(ph, v):
    def one(s):
        if os.path.exists(f"{BD}/models/{ph}/{v}/{s}/done.json"): return
        os.makedirs(f"{BD}/logs", exist_ok=True)
        with open(f"{BD}/logs/train_{ph}_{v}_{s}.log", "w") as f: subprocess.run([PY, "-W", "ignore", f"{HERE}/bytes_enc.py", "train", ph, s, v], stdout=f, stderr=subprocess.STDOUT, check=True)
        print("train", ph, v, s, open(f"{BD}/logs/train_{ph}_{v}_{s}.log").read().strip().splitlines()[-1], flush=True)
    with ThreadPoolExecutor(NPAR) as ex: [f.result() for f in [ex.submit(one, s) for s in P0.SEEDS[ph]]]

def predict(ph, s, v, st, I, rows):
    """→ c̃ (10) por texto: Gemma = D2 sobre tokens de capa 13 cacheados; bytes = encoder entrenado"""
    if v == "GEMMA": return E.d2_predict(rpath(ph, st) + ".tok", ph, s, I)
    import bytes_enc as BE
    return BE.Encoder(f"{BD}/models/{ph}/{v}/{s}/model.pt")([rows[i][8] for i in I])[0]
def ev(ph, v):
    od = f"{BD}/eval/{ph}/{v}"; os.makedirs(od, exist_ok=True); allrows = {st: rows_of(ph, st) for st in SETS}
    def one(s):
        dn = f"{od}/{s}.done"
        if os.path.exists(dn): return
        L = ["idx,cond,text_id," + ",".join(f"c{i}" for i in range(10))]; H = ["set,idx,text_id,fam,p1,p2,fam_hat,p1_head,p2_head,p1_c,p2_c,parse_ok"]
        for st in SETS:
            rows = allrows[st]; I = [i for i, r in enumerate(rows) if r[0] == s]; C = predict(ph, s, v, st, I, rows)
            for j, i in enumerate(I):
                r = rows[i]; c, ok = E.build_c(C[j], r[8]); tid = f"{r[5]}-{r[6]}"
                L.append(f"{r[1]},{st},{tid}," + ",".join(repr(float(x)) for x in c))
                H.append(f"{st},{r[1]},{tid},{r[2]},{r[3]},{r[4]},{int(np.argmax(c[:8]))},{float(C[j][8])!r},{float(C[j][9])!r},{float(c[8])!r},{float(c[9])!r},{int(ok)}")
        open(f"{od}/ctx_{s}.csv", "w").write("\n".join(L) + "\n"); open(f"{od}/read_{s}.csv", "w").write("\n".join(H) + "\n")
        o, e = E.llm_eval(s, ph, f"{od}/ctx_{s}.csv"); open(f"{od}/eval_{s}.csv", "w").write(o); open(f"{od}/queries_{s}.txt", "w").write(e)
        open(dn, "w").write("ok\n"); print("eval", ph, v, s, flush=True)
    with ThreadPoolExecutor(2 if v == "GEMMA" else NPAR) as ex: [f.result() for f in [ex.submit(one, s) for s in P0.SEEDS[ph]]]

def learn(ph, arm):
    """H6. Réplica de learn_eval.ctx/one con el encoder del brazo; τ (tau.json) y (D, λ) de 'learn' (selected.json) congelados del experimento previo."""
    tau = {int(k): v for k, v in json.load(open(f"{LD}/tau.json"))["tau"].items()}; d, lam = json.load(open(f"{LD}/selected.json"))["selected"]["learn"]
    rows = P0.tsv(f"{LD}/{ph}/texts.tsv"); od = f"{BD}/learn/{ph}/{arm}"; os.makedirs(od, exist_ok=True)
    def one(s):
        dn = f"{od}/{s}.done"
        if os.path.exists(dn): return
        si = int(s, 16); I = list(range(len(rows)))
        if arm == "GEMMA": C = E.d2_predict(f"{LD}/{ph}/texts.tok", ph, s, I)
        else:
            import bytes_enc as BE; C = BE.Encoder(f"{BD}/models/{ph}/{arm}/{s}/model.pt")([r[7] for r in rows])[0]
        ex = sorted({int(r[0]) for r in rows}); order = np.random.default_rng(si).permutation(len(ex)); L = ["idx,cond,text_id," + ",".join(f"c{i}" for i in range(10))]
        for oi in order:
            idx = ex[oi]; lang = E.LANGS[(idx + si) % 3]; j = next(j for j, r in enumerate(rows) if int(r[0]) == idx and r[4] == lang); r = rows[j]
            c, ok = E.build_c(C[j], r[7]); L.append(f"{r[0]},E2E,{r[4]}-{r[5]}," + ",".join(repr(float(x)) for x in c))
        open(f"{od}/ctx_{s}.csv", "w").write("\n".join(L) + "\n"); w = f"{P0.DATA}/{ph}/w_{s}.json"; h0 = sha(w)
        p = subprocess.run([P0.B, "learn-eval", "--weights", w, "--set", f"{LD}/{ph}/set.csv", "--ctx", f"{od}/ctx_{s}.csv", "--k", "2", "--deg", str(d), "--lam", lam, "--tau", repr(tau[2])], capture_output=True, check=True)
        assert sha(w) == h0
        open(f"{od}/k2_{s}.csv", "wb").write(p.stdout); open(f"{od}/k2_{s}.err", "w").write(p.stderr.decode()); open(dn, "w").write("ok\n"); print("learn", ph, arm, s, flush=True)
    with ThreadPoolExecutor(NPAR) as ex: [f.result() for f in [ex.submit(one, s) for s in P0.SEEDS[ph]]]

def recon(ph):
    import bytes_enc as BE; R = rows_of(ph, "R"); out = {}
    for s in P0.SEEDS[ph]:
        enc = BE.Encoder(f"{BD}/models/{ph}/MAIN/{s}/model.pt"); T = [r[8] for r in R if r[0] == s]
        out[s] = dict(zip(("byte_acc", "exact"), enc.recon(T))); out[s]["mean_patches"] = float(np.mean(enc(T)[3])); out[s]["mean_bytes_per_patch"] = float(np.mean([len(t.encode()) for t in T]) / out[s]["mean_patches"])
    json.dump(out, open(f"{BD}/eval/{ph}/recon.json", "w"), indent=1); print("recon ok", ph)

def latency(ph):
    """32 consultas (primeros textos R de la primera seed); cada consulta = proceso nuevo, como en demo.py. Memoria = ru_maxrss de los hijos."""
    s = P0.SEEDS[ph][0]; R = [r for r in rows_of(ph, "R") if r[0] == s][:32]; out = {"seed": s, "n": len(R), "GEMMA": [], "BYTES": []}
    import tempfile
    for r in R:
        with tempfile.TemporaryDirectory() as td:
            open(f"{td}/in.txt", "w").write(r[8] + "\n"); t0 = time.perf_counter()
            subprocess.run([P0.B, "gemma-embed", "--model", E.GEMMA, "--in", f"{td}/in.txt", "--out", f"{td}/t.tok", "--exit-after", "12", "--per-token"], check=True, capture_output=True)
            m1 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            subprocess.run([PY, "-c", f"import sys; sys.path.insert(0,{HERE!r}); import e2e as E; E.d2_predict({td + '/t.tok'!r}, {ph!r}, {s!r}, [0])"], check=True, capture_output=True)
            out["GEMMA"].append(time.perf_counter() - t0)
    gm = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    for r in R:
        t0 = time.perf_counter()
        p = subprocess.run([PY, "-c", f"import sys,resource; sys.path.insert(0,{HERE!r}); import bytes_enc as BE; BE.Encoder({BD + '/models/' + ph + '/MAIN/' + s + '/model.pt'!r})([sys.argv[1]]); print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)", r[8]], check=True, capture_output=True)
        out["BYTES"].append(time.perf_counter() - t0); out.setdefault("BYTES_maxrss_kb", []).append(int(p.stdout.split()[-1]))
    out["GEMMA_children_maxrss_kb"] = gm; out["BYTES_maxrss_kb"] = max(out["BYTES_maxrss_kb"])
    json.dump(out, open(f"{BD}/eval/{ph}/latency.json", "w"), indent=1); print("latency ok", ph, np.median(out["GEMMA"]), np.median(out["BYTES"]))

def analyze(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []; arms = ["GEMMA"] + [v for v in VARIANTS if os.path.exists(f"{BD}/eval/{ph}/{v}/{S[-1]}.done")]; qs = 0
    for a in arms:
        for s in S:
            R = list(csv.DictReader(open(f"{BD}/eval/{ph}/{a}/eval_{s}.csv"))); Hd = list(csv.DictReader(open(f"{BD}/eval/{ph}/{a}/read_{s}.csv")))
            qs += int(open(f"{BD}/eval/{ph}/{a}/queries_{s}.txt").read().split("queries=")[1].split()[0])
            for st in SETS:
                G = [r for r in R if r["cond"] == st]; Hs = [h for h in Hd if h["set"] == st]
                per[(a, st, "fam")][s] = np.mean([r["fam_hat_ctx"] == r["fam"] for r in G]); per[(a, st, "acc16")][s] = np.mean([int(r["ok16"]) for r in G])
                per[(a, st, "acc8")][s] = np.mean([int(r["ok8"]) for r in G])
                per[(a, st, "perr")][s] = np.median([max(abs(float(h["p1_c"]) - float(h["p1"])), abs(float(h["p2_c"]) - float(h["p2"]))) for h in Hs])
                per[(a, st, "perr_head")][s] = np.median([max(abs(float(h["p1_head"]) - float(h["p1"])), abs(float(h["p2_head"]) - float(h["p2"]))) for h in Hs])
                per[(a, st, "auroc_res")][s] = P0.auroc([float(r["teacher_div16"]) for r in G], [r["ok16"] == "0" for r in G])
                per[(a, st, "parse_fail")][s] = np.mean([h["parse_ok"] == "0" for h in Hs])
                for lg in ("pt", "it", "ru") if st == "U" else ():
                    per[(a, "U_" + lg, "fam")][s] = np.mean([r["fam_hat_ctx"] == r["fam"] for r in G if r["text_id"].startswith(lg)])
            per[(a, "drop", "fam")][s] = per[(a, "R", "fam")][s] - per[(a, "N", "fam")][s]
            per[(a, "drop", "acc16")][s] = per[(a, "R", "acc16")][s] - per[(a, "N", "acc16")][s]
    for a in ("GEMMA", "MAIN"):
        for s in S:
            f = f"{BD}/learn/{ph}/{a}/k2_{s}.csv"
            if not os.path.exists(f): continue
            L = list(csv.DictReader(open(f))); sk = np.array([float(r["s_known"]) for r in L]); fam = np.array([int(r["fam"]) for r in L])
            per[(a, "learn", "auroc")][s] = P0.auroc(sk, fam >= 8); per[(a, "learn", "acc16_known")][s] = np.mean([int(r["learn_ok16"]) for r in L if int(r["fam"]) < 8])
            qs += int(open(f[:-4] + ".err").read().split()[0].split("=")[-1]) if "queries=" in open(f[:-4] + ".err").read().split()[0] else 0
    v = lambda *k: np.array([per[k].get(s, np.nan) for s in S])
    def Pr(n, x):
        x = np.asarray(x, float)
        if np.isnan(x).all(): return
        b = P0.boot(x); out.append(f"| {n} | {b[0]:+.4f} | [{b[1]:+.4f}, {b[2]:+.4f}] | {b[3]} |")
    def NI(n, x, d=0.05, higher=True):
        x = np.asarray(x, float)
        if np.isnan(x).all(): return
        b = P0.boot(x); ok = (b[1] > -d) if higher else (b[2] < d); out.append(f"| {n} | {b[0]:+.4f} | [{b[1]:+.4f}, {b[2]:+.4f}] | {b[3]} | {'NO INFERIOR' if ok else 'NO DEMOSTRADA'} |")
    out += [f"## {ph.upper()} (16 seeds; R = robust.tsv es/en/fr plantillas 6–7; U = pt/it/ru no vistos; N = ruido estilo BLT)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for a in arms:
        for st in SETS:
            for m in ("fam", "perr", "perr_head", "acc8", "acc16", "auroc_res", "parse_fail"): Pr(f"{a} {st} {m}", v(a, st, m))
        for lg in ("pt", "it", "ru"): Pr(f"{a} U_{lg} fam", v(a, "U_" + lg, "fam"))
        Pr(f"{a} caída fam R−N", v(a, "drop", "fam")); Pr(f"{a} caída acc16 R−N", v(a, "drop", "acc16"))
    for a in ("GEMMA", "MAIN"): Pr(f"{a} AUROC sorpresa s_known nuevas 8–9 vs 0–7 (k=2)", v(a, "learn", "auroc")); Pr(f"{a} acc16 LEARN2 conocidas 0–7", v(a, "learn", "acc16_known"))
    out += ["", "### Primarias (no inferioridad, δ = 0,05; MAIN − GEMMA, pareado por seed)", "", "| hipótesis | media | IC95 | n | veredicto |", "|---|---|---|---|---|"]
    if "MAIN" in arms:
        NI("**H1** familia (R) MAIN − GEMMA", v("MAIN", "R", "fam") - v("GEMMA", "R", "fam"))
        NI("**H2** max|p̂−p| mediano (R) MAIN − GEMMA (menor es mejor)", v("MAIN", "R", "perr") - v("GEMMA", "R", "perr"), higher=False)
        NI("**H3** acc h16 (R) MAIN − GEMMA", v("MAIN", "R", "acc16") - v("GEMMA", "R", "acc16"))
        NI("**H6** AUROC sorpresa MAIN − GEMMA", v("MAIN", "learn", "auroc") - v("GEMMA", "learn", "auroc"))
        out += ["", "### Secundarias", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
        Pr("**H4** familia U (pt/it/ru) MAIN − GEMMA (superior si IC > 0)", v("MAIN", "U", "fam") - v("GEMMA", "U", "fam"))
        Pr("**H8** caída fam R−N MAIN − GEMMA (mejor si IC < 0)", v("MAIN", "drop", "fam") - v("GEMMA", "drop", "fam"))
        Pr("H8b caída acc16 R−N MAIN − GEMMA", v("MAIN", "drop", "acc16") - v("GEMMA", "drop", "acc16"))
        Pr("H3b acc h16 (R) MAIN − 0,323 (referencia fija del Page)", v("MAIN", "R", "acc16") - 0.323)
        Pr("H2b max|p̂−p| cabeza sin numnorm (R) MAIN − GEMMA", v("MAIN", "R", "perr_head") - v("GEMMA", "R", "perr_head"))
        for ab in VARIANTS[1:]:
            if ab in arms: Pr(f"ablación acc16 (R) MAIN − {ab}", v("MAIN", "R", "acc16") - v(ab, "R", "acc16")); Pr(f"ablación fam (R) MAIN − {ab}", v("MAIN", "R", "fam") - v(ab, "R", "fam"))
    rf = f"{BD}/eval/{ph}/recon.json"
    if os.path.exists(rf):
        rc = json.load(open(rf)); Pr("**H7** exactitud de bytes reconstruidos (R limpio, MAIN)", [rc[s]["byte_acc"] for s in S]); Pr("H7 textos reconstruidos exactos", [rc[s]["exact"] for s in S])
        Pr("bytes por parche (R)", [rc[s]["mean_bytes_per_patch"] for s in S])
    lf = f"{BD}/eval/{ph}/latency.json"
    if os.path.exists(lf):
        la = json.load(open(lf)); g, b = np.array(la["GEMMA"]), np.array(la["BYTES"]); rng = np.random.default_rng(12345)
        bb = [np.median(g[i]) / np.median(b[i]) for i in rng.integers(0, len(g), (10000, len(g)))]
        out.append(f"\n**H5** latencia mediana por consulta: GEMMA {np.median(g):.3f} s, BYTES {np.median(b):.3f} s; razón {np.median(g)/np.median(b):.2f} [IC95 {np.percentile(bb,2.5):.2f}, {np.percentile(bb,97.5):.2f}] (n={len(g)}); "
                   f"memoria pico GEMMA (max hijos) {la['GEMMA_children_maxrss_kb']/1024:.0f} MB, BYTES {la['BYTES_maxrss_kb']/1024:.0f} MB")
    conv = collections.defaultdict(list)
    for a in arms[1:]:
        for s in S:
            d = json.load(open(f"{BD}/models/{ph}/{a}/{s}/done.json"))
            for k in ("s0", "s1", "s2"):
                if k in d: conv[(a, k)].append(d[k]["stopped_by_patience"])
    out.append("\nConvergencia (seeds paradas por paciencia / 16): " + "; ".join(f"{a}-{k}: {sum(x)}" for (a, k), x in sorted(conv.items())))
    out.append(f"Queries a TEST/memoria (llm-eval + learn-eval): {qs}")
    os.makedirs(ART, exist_ok=True); open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    {"data": data, "train": train, "eval": ev, "learn": learn, "recon": recon, "latency": latency, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
