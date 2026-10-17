"""CDT-RQM-epr → inferencia de ÓRBITA: aporte DIRECTO vs pasando por SUEÑO (+poda). Prerregistro docs/preregistro_cdt_sueno.md.
ÓRBITA-v1 y el cuerpo de D2 congelados; se entrena la cabeza de familia de D2 (fine-tune de la capa f) más 3 pesos de fusión
sobre rasgos CDT por familia candidata φ_g = [log(r_g+1e-3), sup_g, tr_g] (orbita cdt-feat):  logit_g = f(u)_g + w·φ_g + b.
  cdt.py pool | feat PH | train PH SEED | ctx PH | eval PH | analyze PH"""
import sys, os, csv, json, random, collections, subprocess
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, numnorm as NN, d2 as D2
from make_texts import texts as mk
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
O = "/workspace/orbita-data/llm_cdt"; E2E = "/workspace/orbita-data/llm_e2e"; L = "/workspace/orbita-data/llm"; D2D = "/workspace/orbita-data/llm_d2"; D3D = "/workspace/orbita-data/llm_d3"
ART = f"{P0.ROOT}/artifacts/orbita/llm_cdt"; SIGS = ("full", "sleep", "corrupt")
CONDS = {"NOCDT": None, "DIRECTO": "full", "SUENO": "sleep", "CORRUPT": "corrupt", "SHUF": "full"}
STEPS, LR = 600, 1e-2

def pool():
    rng = random.Random(0xC0DF); rows = []
    for fam in range(8):
        for i in range(16):
            while True:              # como pool/pool2: distribución TRAIN, sin el hueco 0,10 < p1 < 0,45
                p1 = rng.uniform(-1, 1)
                if p1 <= 0.10 or p1 >= 0.45: break
            p2 = rng.uniform(-1, 1)
            for lang, k, t in mk(fam, p1, p2, (0, 1, 4, 5)): rows.append((f"pool3_{fam}_{i}_{lang}{k}", fam, p1, p2, lang, k, t))
    open(f"{O}/pool3.tsv", "w").write("".join("\t".join(map(str, r)) + "\n" for r in rows)); open(f"{O}/pool3.txt", "w").write("".join(r[6] + "\n" for r in rows))
    print(len(rows), "textos pool3")

def test_sets(ph):
    """(nombre, filas tsv, tok, función id) — robust (plantillas 6–7, primario) y dígitos (plantillas 2–3, secundario)"""
    R = P0.tsv(f"{E2E}/{ph}/robust.tsv"); Dg = P0.tsv(f"{L}/{ph}/all.tsv")
    return [("R", R, f"{E2E}/{ph}/robust.tok", lambda r: (r[0], r[1], f"{r[5]}-{r[6]}", r[8])),
            ("DIG", Dg, f"{D2D}/{ph}/all.tok", lambda r: (r[0], r[1], f"{r[5]}-{r[6]}", r[7]))]

def d2_out(tokpath, ph, seed, ids):
    st = np.load(f"{D2D}/stats.npz"); tok = D2.Tok(tokpath); enc = D2.D2(); enc.load_state_dict(torch.load(f"{D2D}/{ph}/enc_D2_{seed}.pt"))
    with torch.no_grad(): ch, logit, u = enc(*tok.batch(ids, st["mu"], st["sd"]))
    return ch.numpy(), u, enc

def pparse(text, ch): p = NN.parse2(text); return (p if p is not None else (float(ch[8]), float(ch[9])))

def feat(ph):
    os.makedirs(f"{O}/{ph}", exist_ok=True); pr = P0.tsv(f"{O}/pool3.tsv")
    for s in P0.SEEDS[ph]:
        if os.path.exists(f"{O}/{ph}/feat_corrupt_{s}.csv"): continue
        lines = ["id,kind,key,p1,p2"]
        ch, _, _ = d2_out(f"{O}/pool3.tok", ph, s, list(range(len(pr))))
        for i, r in enumerate(pr): p = pparse(r[6], ch[i]); lines.append(f"{r[0]},train,{r[1]},{p[0]!r},{p[1]!r}")
        for nm, rows, tk, fid in test_sets(ph):
            I = [i for i, r in enumerate(rows) if r[0] == s]; ch, _, _ = d2_out(tk, ph, s, I)
            for j, i in enumerate(I): _, idx, tid, t = fid(rows[i]); p = pparse(t, ch[j]); lines.append(f"{nm}|{idx}|{tid},test,{idx},{p[0]!r},{p[1]!r}")
        open(f"{O}/{ph}/rows_{s}.csv", "w").write("\n".join(lines) + "\n")
        for sg in SIGS:
            q = subprocess.run([P0.B, "cdt-feat", "--seed", s, "--weights", f"/workspace/orbita-data/{ph}/w_{s}.json", "--rows", f"{O}/{ph}/rows_{s}.csv", "--signal", sg], capture_output=True, check=True)
            open(f"{O}/{ph}/feat_{sg}_{s}.log", "wb").write(q.stderr); open(f"{O}/{ph}/feat_{sg}_{s}.csv.tmp", "wb").write(q.stdout); os.replace(f"{O}/{ph}/feat_{sg}_{s}.csv.tmp", f"{O}/{ph}/feat_{sg}_{s}.csv")
    print("feat ok", ph)

def phi(path):
    R = list(csv.DictReader(open(path)))
    return {r["id"]: np.array([[np.log(float(r[f"r{g}"]) + 1e-3), float(r[f"sup{g}"]), float(r[f"tr{g}"])] for g in range(8)]) for r in R}

def train(ph, s):
    out = f"{O}/{ph}/pred_{s}.json"
    if os.path.exists(out): return
    pr = P0.tsv(f"{O}/pool3.tsv"); y = torch.tensor([int(r[1]) for r in pr]); Fs = {sg: phi(f"{O}/{ph}/feat_{sg}_{s}.csv") for sg in SIGS}
    _, U, enc = d2_out(f"{O}/pool3.tok", ph, s, list(range(len(pr)))); res = {}; perm = np.random.RandomState(int(s, 16) ^ 0x5F).permutation(len(pr))
    tests = {}
    for nm, rows, tk, fid in test_sets(ph):
        I = [i for i, r in enumerate(rows) if r[0] == s]; _, Ut, _ = d2_out(tk, ph, s, I); tests[nm] = (Ut, [f"{nm}|{fid(rows[i])[1]}|{fid(rows[i])[2]}" for i in I])
    for cond, sg in CONDS.items():
        torch.manual_seed(int(s, 16) ^ 0xCD7); f = torch.nn.Linear(D2.HID, 8); f.load_state_dict(enc.f.state_dict()); w = torch.zeros(3, requires_grad=True)
        X = torch.from_numpy(np.stack([Fs[sg][r[0]] for r in pr])) if sg else torch.zeros(len(pr), 8, 3)
        if cond == "SHUF": X = X[torch.from_numpy(perm)]
        opt = torch.optim.Adam(list(f.parameters()) + [w], lr=LR); hist = []
        for st in range(STEPS):
            loss = F.cross_entropy(f(U) + X @ w, y); opt.zero_grad(); loss.backward(); opt.step()
            if st % 200 == 0 or st == STEPS - 1: hist.append(float(loss.detach()))
        res[cond] = {"w": w.detach().tolist(), "loss": hist}
        with torch.no_grad():
            for nm, (Ut, ids) in tests.items():
                Xt = torch.from_numpy(np.stack([Fs[sg][i] for i in ids])) if sg else torch.zeros(len(ids), 8, 3)
                if cond == "SHUF": Xt = Xt[torch.from_numpy(np.random.RandomState(int(s, 16) ^ 0x5E).permutation(len(ids)))]
                res[cond][nm] = dict(zip(ids, (f(Ut) + Xt @ w).argmax(-1).tolist()))
    json.dump(res, open(out + ".tmp", "w")); os.replace(out + ".tmp", out); print(s, "train ok", flush=True)

def ctx(ph):
    for s in P0.SEEDS[ph]:
        pred = json.load(open(f"{O}/{ph}/pred_{s}.json")); base = open(f"{E2E}/{ph}/ctx_{s}.csv").read().splitlines(); out = [base[0]]
        for l in base[1:]:
            f = l.split(","); nm = {"E2E_R": "R", "D3NUM_DIG": "DIG"}.get(f[1])
            if f[1] == "TRUE": out.append(l)
            if nm is None: continue
            out.append(l.replace(f",{f[1]},", f",BASE_{nm},", 1))
            for cond in CONDS:
                g = pred[cond][nm][f"{nm}|{f[0]}|{f[2]}"]; oh = ["0.0"] * 8; oh[g] = "1.0"
                out.append(",".join([f[0], f"{cond}_{nm}", f[2]] + oh + f[11:13]))
        open(f"{O}/{ph}/ctx_{s}.csv", "w").write("\n".join(out) + "\n")
    print("ctx ok", ph)

def ev(ph):
    os.makedirs(f"{ART}/{ph}", exist_ok=True)
    for s in P0.SEEDS[ph]:
        p = subprocess.run([P0.B, "llm-eval", "--seed", s, "--weights", f"/workspace/orbita-data/{ph}/w_{s}.json", "--ctx", f"{O}/{ph}/ctx_{s}.csv"], capture_output=True, check=True)
        open(f"{ART}/{ph}/eval_{s}.csv", "wb").write(p.stdout); open(f"{ART}/{ph}/queries_{s}.txt", "wb").write(p.stderr)
    print("eval ok", ph)

def analyze(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []; size = collections.defaultdict(dict)
    for s in S:
        R = list(csv.DictReader(open(f"{ART}/{ph}/eval_{s}.csv"))); Fs = {sg: list(csv.DictReader(open(f"{O}/{ph}/feat_{sg}_{s}.csv"))) for sg in SIGS}
        Fs = {sg: {r["id"]: r for r in v} for sg, v in Fs.items()}
        for nm in ("R", "DIG"):
            for c in ["TRUE", f"BASE_{nm}"] + [f"{k}_{nm}" for k in CONDS]:
                G = [r for r in R if r["cond"] == c]
                for h in (1, 8, 16): per[(c if c != "TRUE" else "TRUE", h)][s] = np.mean([int(r[f"ok{h}"]) for r in G])
                if c == "TRUE": continue
                per[("famerr", c)][s] = np.mean([r["fam_hat_ctx"] != r["fam"] for r in G])
                sg = CONDS.get(c.rsplit("_", 1)[0]) or "full"
                sc = [float(Fs[sg][f"{nm}|{r['idx']}|{r['text_id']}"][f"r{r['fam_hat_ctx']}"]) for r in G]
                per[("auroc", c)][s] = P0.auroc(sc, [r["ok16"] == "0" for r in G])
        for sg in SIGS:
            t = open(f"{O}/{ph}/feat_{sg}_{s}.log").read()
            for k in ("numbers_full", "numbers_used", "rules_used"): size[(sg, k)][s] = int(t.split(k + "=")[1].split()[0])
            size[(sg, "q")][s] = int(t.split("queries=")[1].split()[0])
    v = lambda *k: np.array([per[k][s] for s in S]); Pr = lambda n, x: out.append(f"| {n} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    out += [f"## {ph.upper()} (16 seeds; 16 ejemplos TEST iterables × 6 textos)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for h in (1, 8, 16): Pr(f"acc h{h} TRUE", P0.boot(v("TRUE", h)))
    for nm in ("R", "DIG"):
        for c in [f"BASE_{nm}"] + [f"{k}_{nm}" for k in CONDS]:
            for h in (1, 8, 16): Pr(f"acc h{h} {c}", P0.boot(v(c, h)))
            Pr(f"error de familia {c}", P0.boot(v("famerr", c))); Pr(f"AUROC r(familia elegida)→fallo h16 {c}", P0.boot(v("auroc", c)))
    for k, a, b in (("H-D1", "DIRECTO", "BASE"), ("H-D2", "DIRECTO", "NOCDT"), ("H-D3", "DIRECTO", "CORRUPT"), ("H-D4", "DIRECTO", "SHUF"),
                    ("H-S1", "SUENO", "BASE"), ("H-S2", "SUENO", "NOCDT"), ("H-S3", "SUENO", "CORRUPT"), ("H-S4", "SUENO", "SHUF"), ("H-SD", "SUENO", "DIRECTO")):
        for h in (16, 8, 1): Pr(f"{'**' + k + '**' if h == 16 else k} acc{h} {a}_R − {b}_R", P0.boot(v(f"{a}_R", h) - v(f"{b}_R", h)))
    for c in ("DIRECTO", "SUENO"): Pr(f"acc16 {c}_R − TRUE", P0.boot(v(f"{c}_R", 16) - v("TRUE", 16)))
    for k, a in (("H-F1", "DIRECTO"), ("H-F2", "SUENO")): Pr(f"**{k}** error familia {a}_R − BASE_R", P0.boot(v("famerr", f"{a}_R") - v("famerr", "BASE_R")))
    Pr("**H-SDf** error familia SUENO_R − DIRECTO_R", P0.boot(v("famerr", "SUENO_R") - v("famerr", "DIRECTO_R")))
    for c in ("DIRECTO", "SUENO"): Pr(f"AUROC {c}_R − 0.5", P0.boot(v("auroc", f"{c}_R") - 0.5))
    for sg in SIGS:
        for k in ("numbers_full", "numbers_used", "rules_used"): Pr(f"tamaño señal {sg}: {k}", P0.boot(np.array([size[(sg, k)][s] for s in S], float)))
    q = sum(int(open(f"{ART}/{ph}/queries_{s}.txt").read().split("queries=")[1].split()[0]) for s in S) + sum(size[(sg, "q")][s] for sg in SIGS for s in S)
    W = {c: np.array([json.load(open(f"{O}/{ph}/pred_{s}.json"))[c]["w"] for s in S]).mean(0).round(4).tolist() for c in CONDS}
    out.append(f"\nQueries totales (llm-eval + cdt-feat): {q}\nPesos de fusión medios w=[log r, sup, tr]: {W}")
    open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    os.makedirs(O, exist_ok=True)
    {"pool": pool, "feat": feat, "train": train, "ctx": ctx, "eval": ev, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
