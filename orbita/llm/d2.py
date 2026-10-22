"""D2 — atención por token sobre la capa 13/26 de Gemma (prerregistro docs/preregistro_encoder_d2.md).
  d2.py stats                  media/desv. por dimensión sobre los tokens del pool2
  d2.py train PHASE SEED       entrena D2 (atención) y D1_POOL2 (media de tokens, arquitectura D1) con ÓRBITA-v1 de esa seed congelada
  d2.py ctx PHASE | eval PHASE | analyze PHASE"""
import sys, os, json, time, csv, collections
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline as P0, d1 as D1
from d1 import load_orbita, rollout
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
O = f"{P0.DATA}/llm_d2"; L = f"{P0.DATA}/llm"; ART = f"{P0.ROOT}/artifacts/orbita/llm_d2"; DIM = 2304
HID, STEPS, BATCH, LR, LAM_P, LAM_T, LAM_I, TAU_I, NX, KS = 128, 3000, 32, 1e-3, 10.0, 1.0, 0.1, 0.1, 8, [1, 2, 4, 8]
DK, DV = 32, 64

class Tok:
    def __init__(s, path):
        n = [int(l.split()[1]) for l in open(path + ".idx")]
        s.off = np.concatenate([[0], np.cumsum(n)]); s.X = np.memmap(path, dtype="<f4", mode="r", shape=(int(s.off[-1]), DIM))
    def __len__(s): return len(s.off) - 1
    def batch(s, ids, mu, sd):
        T = max(int(s.off[i + 1] - s.off[i]) for i in ids); X = np.zeros((len(ids), T, DIM)); M = np.zeros((len(ids), T), bool)
        for b, i in enumerate(ids):
            a, e = s.off[i], s.off[i + 1]; X[b, :e - a] = (np.asarray(s.X[a:e], np.float64) - mu) / sd; M[b, :e - a] = True
        return torch.from_numpy(X), torch.from_numpy(M)
    def mean(s, ids, mu, sd):
        return torch.from_numpy(np.stack([(np.asarray(s.X[s.off[i]:s.off[i + 1]], np.float64).mean(0) - mu) / sd for i in ids]))

class D2(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.k = torch.nn.Linear(DIM, DK); s.v = torch.nn.Linear(DIM, DV); s.q = torch.nn.Parameter(torch.randn(3, DK) / DK ** 0.5)
        s.l1 = torch.nn.Linear(3 * DV, HID); s.f = torch.nn.Linear(HID, 8); s.p = torch.nn.Linear(HID, 2)
    def forward(s, X, M):
        a = torch.einsum("qd,btd->bqt", s.q, s.k(X)) / DK ** 0.5; a = a.masked_fill(~M[:, None, :], -1e30).softmax(-1)
        u = torch.tanh(s.l1(torch.einsum("bqt,btd->bqd", a, s.v(X)).flatten(1))); pi = F.softmax(s.f(u), -1)
        oh = F.one_hot(pi.argmax(-1), 8).to(pi.dtype)
        return torch.cat([oh + pi - pi.detach(), torch.tanh(s.p(u))], -1), s.f(u), u

def stats():
    t = Tok(f"{O}/pool2.tok"); n = t.X.shape[0]; s1 = np.zeros(DIM); s2 = np.zeros(DIM)
    for a in range(0, n, 4096): x = np.asarray(t.X[a:a + 4096], np.float64); s1 += x.sum(0); s2 += (x * x).sum(0)
    mu = s1 / n; np.savez(f"{O}/stats.npz", mu=mu, sd=np.sqrt(np.maximum(s2 / n - mu * mu, 0)) + 1e-8); print("stats", n)

def train(ph, seed):
    t0 = time.time(); torch.manual_seed(int(seed, 16) ^ 0xD2)
    m = load_orbita(f"{P0.DATA}/{ph}/w_{seed}.json")
    meta = json.load(open(f"{P0.DATA}/{ph}/{seed}/meta.json")); z, cd = meta["zdim"], meta["cdim"]
    XT = torch.from_numpy(np.fromfile(f"{P0.DATA}/{ph}/{seed}/train.bin", dtype="<f8").reshape(-1, z + cd + z)[:, :z].copy())
    rows = P0.tsv(f"{O}/pool2.tsv"); ids = sorted({r[0] for r in rows}); byc = collections.defaultdict(list)
    for i, r in enumerate(rows): byc[r[0]].append(i)
    C = torch.tensor(np.array([P0.ctx_true(int(r[1]), float(r[2]), float(r[3])) for r in rows])); FAM = torch.tensor([int(r[1]) for r in rows])
    st = np.load(f"{O}/stats.npz"); mu, sd = st["mu"], st["sd"]; tok = Tok(f"{O}/pool2.tok"); assert len(tok) == len(rows)
    out = {}
    for kind in ("D2", "D1_POOL2"):
        enc = D2() if kind == "D2" else D1.Enc(DIM); opt = torch.optim.Adam(enc.parameters(), lr=LR); g = torch.Generator().manual_seed(int(seed, 16) ^ 0xD2E); hist = []
        for step in range(STEPS):
            cb = torch.randint(0, len(ids), (BATCH,), generator=g); a, b = [], []
            for j in cb.tolist():
                lst = byc[ids[j]]; pr = torch.randperm(len(lst), generator=g)[:2].tolist(); a.append(lst[pr[0]]); b.append(lst[pr[1]])
            if kind == "D2":
                Xa, Ma = tok.batch(a, mu, sd); Xb, Mb = tok.batch(b, mu, sd); ch, logit, ua = enc(Xa, Ma); _, _, ub = enc(Xb, Mb)
            else:
                ch, logit, ua = enc(tok.mean(a, mu, sd)); _, _, ub = enc(tok.mean(b, mu, sd))
            a = torch.tensor(a)
            lce = F.cross_entropy(logit, FAM[a]); lp = ((ch[:, 8:] - C[a, 8:]) ** 2).sum(1).mean()
            li = F.cross_entropy(F.normalize(ua, dim=-1) @ F.normalize(ub, dim=-1).T / TAU_I, torch.arange(BATCH))
            xi = XT[torch.randint(0, len(XT), (BATCH * NX,), generator=g)]; chx = ch.repeat_interleave(NX, 0); cx = C[a].repeat_interleave(NX, 0)
            lt, zh, zt, prev = 0.0, xi, xi, 0
            for k in KS:
                zh = rollout(m, zh, chx, k - prev)
                with torch.no_grad(): zt = rollout(m, zt, cx, k - prev)
                prev = k; lt = lt + 0.25 * (((zh - zt) ** 2).sum(1) / (zt ** 2).sum(1).clamp_min(1e-9)).clamp(max=10.0).mean()
            loss = lce + LAM_P * lp + LAM_T * lt + LAM_I * li
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 500 == 0 or step == STEPS - 1: hist.append([step, float(lce.detach()), float(lp.detach()), float(lt.detach()), float(li.detach())])
        torch.save(enc.state_dict(), f"{O}/{ph}/enc_{kind}_{seed}.pt"); out[kind] = hist
    out["wall_s"] = time.time() - t0; json.dump(out, open(f"{O}/{ph}/train_{seed}.json", "w")); print(seed, "wall_s", round(out["wall_s"], 1), flush=True)

def predict(ph, seed, kind):
    st = np.load(f"{O}/stats.npz"); rows = P0.tsv(f"{L}/{ph}/all.tsv"); tok = Tok(f"{O}/{ph}/all.tok"); assert len(tok) == len(rows)
    I = [i for i, r in enumerate(rows) if r[0] == seed]
    enc = D2() if kind == "D2" else D1.Enc(DIM); enc.load_state_dict(torch.load(f"{O}/{ph}/enc_{kind}_{seed}.pt"))
    with torch.no_grad():
        ch = enc(*tok.batch(I, st["mu"], st["sd"]))[0] if kind == "D2" else enc(tok.mean(I, st["mu"], st["sd"]))[0]
    return rows, I, ch.numpy()

def ctx(ph):
    for s in P0.SEEDS[ph]:
        old = open(f"{P0.DATA}/llm_d1/{ph}/ctxin_{s}.csv").read().splitlines()
        keep = [old[0]] + [l for l in old[1:] if l.split(",")[1] in ("TRUE", "GEMMA_F0", "RAND", "D1", "D1_NGRAM")]
        rows, I, CG = predict(ph, s, "D2"); _, _, CP = predict(ph, s, "D1_POOL2")
        ex = collections.defaultdict(list)
        for j, i in enumerate(I): ex[rows[i][1]].append(j)
        keys = list(ex); fmt = lambda v: ",".join(repr(float(x)) for x in v)
        for k in keys:
            same = [q for q in keys if rows[I[ex[q][0]]][2] == rows[I[ex[k][0]]][2]]; nxt = same[(same.index(k) + 1) % len(same)]
            for t, j in enumerate(ex[k]):
                tid = f"{rows[I[j]][5]}-{rows[I[j]][6]}"
                keep += [f"{k},D2,{tid}," + fmt(CG[j]), f"{k},D1_POOL2,{tid}," + fmt(CP[j]), f"{k},D2_SWAP,{tid}," + fmt(CG[ex[nxt][t]])]
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
    d1rows, _, _ = D1.predict(ph, S[0], "GEMMA")
    for s in S:
        R = list(csv.DictReader(open(f"{ART}/{ph}/llm_eval_{s}.csv"))); g = lambda c: [r for r in R if r["cond"] == c]
        for c in ("TRUE", "GEMMA_F0", "RAND", "D1", "D1_NGRAM", "D1_POOL2", "D2"):
            for h in (1, 8, 16, 32): per[(c, h)][s] = np.mean([int(r[f"ok{h}"]) for r in g(c)])
            per[("rel1", c)][s] = np.median([float(r["rel1"]) for r in g(c)])
        for lg in ("es", "en", "fr"): per[("D2_" + lg,)][s] = np.mean([int(r["ok16"]) for r in g("D2") if r["text_id"].startswith(lg)])
        zz = {(r["idx"], r["cond"], r["text_id"]): np.array([float(v) for v in r["z16"].split(";")]) for r in R}; ds, dd = [], []
        for idx in {r["idx"] for r in R}:
            own = [zz[k] for k in sorted(zz) if k[0] == idx and k[1] == "D2"]; sw = [zz[k] for k in sorted(zz) if k[0] == idx and k[1] == "D2_SWAP"]
            ref = np.linalg.norm(zz[(idx, "TRUE", "0")]) + 1e-12
            ds.append(np.median([np.linalg.norm(a - b) / ref for i, a in enumerate(own) for b in own[i+1:]])); dd.append(np.median([np.linalg.norm(a - b) / ref for a, b in zip(own, sw)]))
        per[("Dsame",)][s], per[("Ddiff",)][s] = np.median(ds), np.median(dd)
        for kind, fn in (("D2", predict), ("D1_POOL2", predict), ("D1", D1.predict)):
            rows, I, CG = fn(ph, s, kind if kind != "D1" else "GEMMA")
            per[("fam", kind)][s] = np.mean([CG[j][:8].argmax() == int(rows[i][2]) for j, i in enumerate(I)])
            per[("perr", kind)][s] = np.median([np.abs(CG[j][8:] - [float(rows[i][3]), float(rows[i][4])]).max() for j, i in enumerate(I)])
        G = g("D2"); per[("famT",)][s] = np.mean([r["fam_hat_teacher"] == r["fam"] for r in G])
        per[("auroc",)][s] = P0.auroc([float(r["teacher_div16"]) for r in G], [r["ok16"] == "0" for r in G])
    v = lambda k: np.array([per[k][s] for s in S]); Pr = lambda n, x: out.append(f"| {n} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    out += [f"## {ph.upper()} (16 seeds; 16 ejemplos TEST iterables × 6 textos)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for c in ("TRUE", "GEMMA_F0", "RAND", "D1", "D1_NGRAM", "D1_POOL2", "D2"):
        for h in (1, 8, 16, 32): Pr(f"acc h{h} {c}", P0.boot(v((c, h))))
        Pr(f"rel_err mediano h1 {c}", P0.boot(v(("rel1", c))))
    for k in ("D1", "D1_POOL2", "D2"): Pr(f"argmax familia {k}", P0.boot(v(("fam", k)))); Pr(f"max|p̂−p| mediano {k}", P0.boot(v(("perr", k))))
    Pr("**H-E5a** max|p̂−p| D2 − D1", P0.boot(v(("perr", "D2")) - v(("perr", "D1")))); Pr("**H-E5a** acc16 D2 − D1", P0.boot(v(("D2", 16)) - v(("D1", 16))))
    Pr("**H-E5b** acc16 D2 − D1_POOL2", P0.boot(v(("D2", 16)) - v(("D1_POOL2", 16)))); Pr("H-E5b max|p̂−p| D2 − D1_POOL2", P0.boot(v(("perr", "D2")) - v(("perr", "D1_POOL2"))))
    Pr("**H-E2'** acc16 D2 − TRUE", P0.boot(v(("D2", 16)) - v(("TRUE", 16))))
    for a, b in (("es", "en"), ("es", "fr"), ("en", "fr")): Pr(f"**H-E3'** acc16 {a} − {b}", P0.boot(v(("D2_" + a,)) - v(("D2_" + b,))))
    Pr("**H-E3'** acc16 D2 − RAND", P0.boot(v(("D2", 16)) - v(("RAND", 16)))); Pr("**H-E3'** Dsame − Ddiff", P0.boot(v(("Dsame",)) - v(("Ddiff",))))
    Pr("**H-E4'** acc16 D2 − D1_NGRAM", P0.boot(v(("D2", 16)) - v(("D1_NGRAM", 16))))
    Pr("**H-C1a'** familia teacher − 1/8", P0.boot(v(("famT",)) - 0.125)); Pr("**H-C1b'** AUROC − 0.5", P0.boot(v(("auroc",)) - 0.5))
    q = sum(int(open(f"{ART}/{ph}/queries_{s}.txt").read().split("queries=")[1].split()[0]) for s in S); out.append(f"\nQueries totales (llm-eval): {q}")
    tw = [json.load(open(f"{O}/{ph}/train_{s}.json"))["wall_s"] for s in S]; out.append(f"Tiempo de entrenamiento por seed (D2 + D1_POOL2): media {np.mean(tw):.1f} s, rango {min(tw):.1f}–{max(tw):.1f} s")
    open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    os.makedirs(f"{O}/dev", exist_ok=True); os.makedirs(f"{O}/confirm", exist_ok=True)
    {"stats": stats, "train": train, "ctx": ctx, "eval": ev, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
