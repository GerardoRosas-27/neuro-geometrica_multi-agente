"""D1 — encoder entrenable agnóstico al lenguaje (prerregistro docs/preregistro_encoder_d1.md).
  d1.py noise PHASE            H-E0: c verdadero + ruido gaussiano en p (sin entrenar)
  d1.py train PHASE SEED       entrena el encoder D1 (GEMMA y NGRAM) con ÓRBITA-v1 de esa seed congelada
  d1.py ctx PHASE              CSV de contextos: TRUE, GEMMA_F0, RAND, D1, D1_SWAP, D1_NGRAM
  d1.py eval PHASE             orbita llm-eval por seed
  d1.py analyze PHASE          tablas + IC95 (bootstrap pareado por seed, 10000, semilla 12345)"""
import sys, os, json, time, csv, collections
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline as P0
from train_orbita import Orbita
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
L = f"{P0.DATA}/llm"; O = f"{P0.DATA}/llm_d1"; ART = f"{P0.ROOT}/artifacts/orbita/llm_d1"
SIGMAS = [0.005, 0.01, 0.02, 0.05]; ND = 6
# hiperparámetros FIJOS (prerregistro)
HID, STEPS0, BATCH, LR, LAM_P, LAM_T, LAM_I, TAU_I, NX, KS = 128, 3000, 32, 1e-3, 10.0, 1.0, 0.1, 0.1, 8, [1, 2, 4, 8]
STEPS = int(os.environ.get("D1_STEPS_SMOKE", STEPS0))  # sólo para medir coste; resultados usan 3000

def load_orbita(path):
    w = json.load(open(path)); m = Orbita(hh=w["hh"], hg=w["hg"], kc=w["kc"], r=w["r"], eta=w["eta"], variant=w.get("variant", "full"))
    with torch.no_grad():
        for mod, a, b in ((m.l1, "w1", "b1"), (m.l2, "w2", "b2"), (m.l3, "w3", "b3"), (m.l4, "w4", "b4")):
            mod.weight.copy_(torch.tensor(w[a])); mod.bias.copy_(torch.tensor(w[b]))
        m.wc.copy_(torch.tensor(w["wc"]))
    for p in m.parameters(): p.requires_grad_(False)
    return m

class Enc(torch.nn.Module):
    def __init__(s, d):
        super().__init__(); s.l1 = torch.nn.Linear(d, HID); s.f = torch.nn.Linear(HID, 8); s.p = torch.nn.Linear(HID, 2)
    def forward(s, x):
        u = torch.tanh(s.l1(x)); pi = F.softmax(s.f(u), -1)
        oh = F.one_hot(pi.argmax(-1), 8).to(pi.dtype)
        return torch.cat([oh + pi - pi.detach(), torch.tanh(s.p(u))], -1), s.f(u), u

def feats(kind, ph=None):
    rows = P0.tsv(f"{L}/pool.tsv" if ph is None else f"{L}/{ph}/all.tsv")
    if kind == "GEMMA": X = P0.emb(f"{L}/pool.emb" if ph is None else f"{L}/{ph}/all.emb", len(rows))
    else: X = P0.ngram([r[6] if ph is None else r[7] for r in rows])
    return rows, X

def rollout(m, x, c, k):
    for _ in range(k): x = m(x, c)[0]
    return x

def train(ph, seed):
    t0 = time.time(); torch.manual_seed(int(seed, 16) ^ 0xD1)
    m = load_orbita(f"{P0.DATA}/{ph}/w_{seed}.json")
    meta = json.load(open(f"{P0.DATA}/{ph}/{seed}/meta.json")); z, cd = meta["zdim"], meta["cdim"]
    tr = np.fromfile(f"{P0.DATA}/{ph}/{seed}/train.bin", dtype="<f8").reshape(-1, z + cd + z)
    XT = torch.from_numpy(tr[:, :z].copy())          # sólo objetos TRAIN de esa seed
    rows = P0.tsv(f"{L}/pool.tsv"); ctxid = [r[0] for r in rows]; ids = sorted(set(ctxid))
    byc = collections.defaultdict(list)
    for i, r in enumerate(rows): byc[r[0]].append(i)
    C = torch.tensor(np.array([P0.ctx_true(int(r[1]), float(r[2]), float(r[3])) for r in rows]))
    FAM = torch.tensor([int(r[1]) for r in rows]); out = {}
    for kind in ("GEMMA", "NGRAM"):
        _, X = feats(kind); mu, sd = X.mean(0), X.std(0) + 1e-8; X = torch.from_numpy((X - mu) / sd)
        enc = Enc(X.shape[1]); opt = torch.optim.Adam(enc.parameters(), lr=LR); g = torch.Generator().manual_seed(int(seed, 16) ^ 0xD1E)
        hist = []
        for step in range(STEPS):
            cb = torch.randint(0, len(ids), (BATCH,), generator=g)
            a, b = [], []
            for j in cb.tolist():
                lst = byc[ids[j]]; pr = torch.randperm(len(lst), generator=g)[:2].tolist(); a.append(lst[pr[0]]); b.append(lst[pr[1]])
            a, b = torch.tensor(a), torch.tensor(b)
            ch, logit, ua = enc(X[a]); _, _, ub = enc(X[b])
            lce = F.cross_entropy(logit, FAM[a]); lp = ((ch[:, 8:] - C[a, 8:]) ** 2).sum(1).mean()
            na, nb = F.normalize(ua, dim=-1), F.normalize(ub, dim=-1)
            li = F.cross_entropy(na @ nb.T / TAU_I, torch.arange(BATCH))
            xi = XT[torch.randint(0, len(XT), (BATCH * NX,), generator=g)]
            chx = ch.repeat_interleave(NX, 0); cx = C[a].repeat_interleave(NX, 0)
            lt, zh, zt, prev = 0.0, xi, xi, 0
            for k in KS:
                zh = rollout(m, zh, chx, k - prev)
                with torch.no_grad(): zt = rollout(m, zt, cx, k - prev)
                prev = k
                lt = lt + 0.25 * (((zh - zt) ** 2).sum(1) / (zt ** 2).sum(1).clamp_min(1e-9)).clamp(max=10.0).mean()
            loss = lce + LAM_P * lp + LAM_T * lt + LAM_I * li
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 500 == 0 or step == STEPS - 1: hist.append([step, float(lce), float(lp), float(lt), float(li)])
        torch.save({"enc": enc.state_dict(), "mu": mu, "sd": sd}, f"{O}/{ph}/enc_{kind}_{seed}.pt"); out[kind] = hist
    out["wall_s"] = time.time() - t0
    json.dump(out, open(f"{O}/{ph}/train_{seed}.json", "w")); print(seed, "wall_s", round(out["wall_s"], 1), flush=True)

def predict(ph, seed, kind):
    st = torch.load(f"{O}/{ph}/enc_{kind}_{seed}.pt", weights_only=False)
    rows, X = feats(kind, ph); I = [i for i, r in enumerate(rows) if r[0] == seed]
    enc = Enc(X.shape[1]); enc.load_state_dict(st["enc"])
    with torch.no_grad(): ch, _, _ = enc(torch.from_numpy((X[I] - st["mu"]) / st["sd"]))
    return rows, I, ch.numpy()

def noise(ph):
    os.makedirs(f"{ART}/{ph}", exist_ok=True); rng = np.random.default_rng(0xE0)
    for s in P0.SEEDS[ph]:
        rr = list(csv.DictReader(open(f"{L}/{ph}/ctx_{s}.csv"))); Lh = ["idx,cond,text_id," + ",".join(f"c{i}" for i in range(10))]
        for r in rr:
            c = P0.ctx_true(int(r["fam"]), float(r["p1"]), float(r["p2"])); Lh.append(f"{r['idx']},TRUE,0," + ",".join(repr(float(v)) for v in c))
            for sg in SIGMAS:
                for d in range(ND):
                    cn = c.copy(); cn[8:] += rng.normal(0, sg, 2); Lh.append(f"{r['idx']},NOISE{sg},{d}," + ",".join(repr(float(v)) for v in cn))
        open(f"{O}/{ph}/noise_{s}.csv", "w").write("\n".join(Lh) + "\n")
        p = P0.subprocess.run([P0.B, "llm-eval", "--seed", s, "--weights", f"{P0.DATA}/{ph}/w_{s}.json", "--ctx", f"{O}/{ph}/noise_{s}.csv"], capture_output=True, check=True)
        open(f"{ART}/{ph}/noise_eval_{s}.csv", "wb").write(p.stdout); open(f"{ART}/{ph}/noise_queries_{s}.txt", "wb").write(p.stderr)
    print("noise ok", ph)

def ctx(ph):
    for s in P0.SEEDS[ph]:
        old = [l for l in open(f"{L}/{ph}/ctxin_{s}.csv").read().splitlines()]
        keep = [old[0]] + [l.replace(",GEMMA,", ",GEMMA_F0,") for l in old[1:] if l.split(",")[1] in ("TRUE", "GEMMA", "RAND")]
        rows, I, CG = predict(ph, s, "GEMMA"); _, _, CN = predict(ph, s, "NGRAM")
        ex = collections.defaultdict(list)
        for j, i in enumerate(I): ex[rows[i][1]].append(j)
        keys = list(ex); fmt = lambda v: ",".join(repr(float(x)) for x in v)
        for k in keys:
            same = [q for q in keys if rows[I[ex[q][0]]][2] == rows[I[ex[k][0]]][2]]; nxt = same[(same.index(k) + 1) % len(same)]
            for t, j in enumerate(ex[k]):
                tid = f"{rows[I[j]][5]}-{rows[I[j]][6]}"
                keep += [f"{k},D1,{tid}," + fmt(CG[j]), f"{k},D1_NGRAM,{tid}," + fmt(CN[j]), f"{k},D1_SWAP,{tid}," + fmt(CG[ex[nxt][t]])]
        open(f"{O}/{ph}/ctxin_{s}.csv", "w").write("\n".join(keep) + "\n")
    print("ctx ok", ph)

def ev(ph):
    os.makedirs(f"{ART}/{ph}", exist_ok=True)
    for s in P0.SEEDS[ph]:
        p = P0.subprocess.run([P0.B, "llm-eval", "--seed", s, "--weights", f"{P0.DATA}/{ph}/w_{s}.json", "--ctx", f"{O}/{ph}/ctxin_{s}.csv"], capture_output=True, check=True)
        open(f"{ART}/{ph}/llm_eval_{s}.csv", "wb").write(p.stdout); open(f"{ART}/{ph}/queries_{s}.txt", "wb").write(p.stderr)
    print("eval ok", ph)

def analyze(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []
    for s in S:
        N = list(csv.DictReader(open(f"{ART}/{ph}/noise_eval_{s}.csv")))
        for c in ["TRUE"] + [f"NOISE{sg}" for sg in SIGMAS]: per[("n_" + c)][s] = np.mean([int(r["ok16"]) for r in N if r["cond"] == c])
        R = list(csv.DictReader(open(f"{ART}/{ph}/llm_eval_{s}.csv"))); g = lambda c: [r for r in R if r["cond"] == c]
        for c in ("TRUE", "GEMMA_F0", "RAND", "D1", "D1_NGRAM", "D1_SWAP"):
            for h in (1, 8, 16, 32): per[(c, h)][s] = np.mean([int(r[f"ok{h}"]) for r in g(c)])
            per[("rel1", c)][s] = np.median([float(r["rel1"]) for r in g(c)])
        for lg in ("es", "en", "fr"): per[("D1_" + lg,)][s] = np.mean([int(r["ok16"]) for r in g("D1") if r["text_id"].startswith(lg)])
        z = {(r["idx"], r["cond"], r["text_id"]): np.array([float(v) for v in r["z16"].split(";")]) for r in R}
        ds, dd = [], []
        for idx in {r["idx"] for r in R}:
            own = [z[k] for k in sorted(z) if k[0] == idx and k[1] == "D1"]; sw = [z[k] for k in sorted(z) if k[0] == idx and k[1] == "D1_SWAP"]
            ref = np.linalg.norm(z[(idx, "TRUE", "0")]) + 1e-12
            ds.append(np.median([np.linalg.norm(a - b) / ref for i, a in enumerate(own) for b in own[i+1:]])); dd.append(np.median([np.linalg.norm(a - b) / ref for a, b in zip(own, sw)]))
        per[("Dsame",)][s], per[("Ddiff",)][s] = np.median(ds), np.median(dd)
        rows, I, CG = predict(ph, s, "GEMMA")
        per[("famacc",)][s] = np.mean([CG[j][:8].argmax() == int(rows[i][2]) for j, i in enumerate(I)])
        per[("perr",)][s] = np.median([np.abs(CG[j][8:] - [float(rows[i][3]), float(rows[i][4])]).max() for j, i in enumerate(I)])
        G = g("D1")
        per[("famT",)][s] = np.mean([r["fam_hat_teacher"] == r["fam"] for r in G])
        per[("auroc",)][s] = P0.auroc([float(r["teacher_div16"]) for r in G], [r["ok16"] == "0" for r in G])
    v = lambda k: np.array([per[k][s] for s in S])
    Pr = lambda name, x: out.append(f"| {name} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    out += [f"## {ph.upper()} (16 seeds; 16 ejemplos TEST iterables × 6 textos)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for c in ["TRUE"] + [f"NOISE{sg}" for sg in SIGMAS]: Pr(f"H-E0 acc16 {c}", P0.boot(v(("n_" + c))))
    for sg in SIGMAS: Pr(f"**H-E0** acc16 NOISE{sg} − TRUE", P0.boot(v(("n_NOISE" + str(sg))) - v(("n_TRUE"))))
    for c in ("TRUE", "GEMMA_F0", "RAND", "D1", "D1_NGRAM"):
        for h in (1, 8, 16, 32): Pr(f"acc h{h} {c}", P0.boot(v((c, h))))
        Pr(f"rel_err mediano h1 {c}", P0.boot(v(("rel1", c))))
    Pr("argmax familia D1", P0.boot(v(("famacc",)))); Pr("max|p̂−p| mediano D1", P0.boot(v(("perr",))))
    Pr("**H-E1** acc16 D1 − GEMMA_F0", P0.boot(v(("D1", 16)) - v(("GEMMA_F0", 16))))
    Pr("**H-E2** acc16 D1 − TRUE", P0.boot(v(("D1", 16)) - v(("TRUE", 16))))
    for a, b in (("es", "en"), ("es", "fr"), ("en", "fr")): Pr(f"**H-E3** acc16 {a} − {b}", P0.boot(v(("D1_" + a,)) - v(("D1_" + b,))))
    Pr("**H-E3** acc16 D1 − RAND", P0.boot(v(("D1", 16)) - v(("RAND", 16)))); Pr("**H-E3** Dsame − Ddiff", P0.boot(v(("Dsame",)) - v(("Ddiff",))))
    Pr("**H-E4** acc16 D1 − D1_NGRAM", P0.boot(v(("D1", 16)) - v(("D1_NGRAM", 16))))
    Pr("**H-C1a** familia teacher − 1/8", P0.boot(v(("famT",)) - 0.125)); Pr("**H-C1b** AUROC − 0.5", P0.boot(v(("auroc",)) - 0.5))
    q = sum(int(open(f"{ART}/{ph}/{f}_{s}.txt").read().split("queries=")[1].split()[0]) for s in S for f in ("queries", "noise_queries"))
    out.append(f"\nQueries totales (llm-eval, D1 + H-E0): {q}")
    tw = [json.load(open(f"{O}/{ph}/train_{s}.json"))["wall_s"] for s in S]; out.append(f"Tiempo de entrenamiento D1 por seed (GEMMA+NGRAM): media {np.mean(tw):.1f} s, rango {min(tw):.1f}–{max(tw):.1f} s")
    open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    os.makedirs(f"{O}/dev", exist_ok=True); os.makedirs(f"{O}/confirm", exist_ok=True)
    {"noise": noise, "train": train, "ctx": ctx, "eval": ev, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
