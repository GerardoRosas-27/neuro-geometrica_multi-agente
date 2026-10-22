"""Entrenamiento extenso (docs/preregistro_yo_ext.md): D2_EXT (más contextos/plantillas/idiomas/pasos) + yo recalibrado. ÓRBITA-v1 congelada.
  yo_ext.py texts | train PH SEED | feats PH | analyze PH"""
import sys, os, csv, json, collections, random, inspect, subprocess, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, numnorm as NN, e2e as E, yo as Y, d2 as D2M
from make_texts import FAM, ART as ARTI, T as T05
O = f"{P0.DATA}/yo_ext"; ART = f"{P0.ROOT}/artifacts/orbita/yo_ext"; STEPS_EXT = 9000
FAM_X = {"pt": ["translação", "rotação", "reflexão", "escala", "cisalhamento", "transformação afim", "rotação com escala", "deformação radial"],
         "it": ["traslazione", "rotazione", "riflessione", "scalatura", "taglio", "trasformazione affine", "rotazione con scala", "deformazione radiale"]}
ART_X = {"pt": ["uma"] * 8, "it": ["una"] * 8}
T_X = {"pt": ["Aplique {art} {f} com parâmetros {a} e {b}.", "A figura sofre {art} {f}; os dois valores são {a} e {b}."],
       "it": ["Applica {art} {f} con parametri {a} e {b}.", "La figura subisce {art} {f}; i due valori sono {a} e {b}."]}
T1011 = {"es": ["Queremos {art} {f}; primero {a}, después {b}.", "Instrucción: {f}, parámetros {a} / {b}."],
         "en": ["We want {art} {f}; first {a}, then {b}.", "Instruction: {f}, parameters {a} / {b}."],
         "fr": ["Nous voulons {art} {f} ; d'abord {a}, puis {b}.", "Instruction : {f}, paramètres {a} / {b}."]}
def texts():
    os.makedirs(O, exist_ok=True); rows = [r for r in P0.tsv(f"{E.D2D}/pool2.tsv")]   # pool2 original (plantillas 0,1,4,5)
    rng = random.Random(0xC0DE3)
    for fam in range(8):
        for i in range(32):
            while True:
                p1 = rng.uniform(-1, 1)
                if p1 <= 0.10 or p1 >= 0.45: break
            p2 = rng.uniform(-1, 1); cid = f"ext{fam}_{i}"
            for lang in ("es", "en", "fr"):
                for k in (0, 1, 4, 5): rows.append([cid, str(fam), repr(p1), repr(p2), lang, str(k), T05[lang][k].format(art=ARTI[lang][fam], f=FAM[lang][fam], a=f"{p1:.2f}".replace(".", "," if lang != "en" else "."), b=f"{p2:.2f}".replace(".", "," if lang != "en" else "."))])
                for k in range(2):
                    fa, fb = (i + k) % 5, (i + k + 2) % 5
                    rows.append([cid, str(fam), repr(p1), repr(p2), lang, str(10 + k), T1011[lang][k].format(art=ARTI[lang][fam], f=FAM[lang][fam], a=NN.render(p1, lang, fa), b=NN.render(p2, lang, fb))])
            for lang in ("pt", "it"):
                for k in range(2): rows.append([cid, str(fam), repr(p1), repr(p2), lang, str(20 + k), T_X[lang][k].format(art=ART_X[lang][fam], f=FAM_X[lang][fam], a=f"{p1:.2f}".replace(".", ","), b=f"{p2:.2f}".replace(".", ","))])
    open(f"{O}/pool2.tsv", "w").write("".join("\t".join(r) + "\n" for r in rows)); open(f"{O}/pool2.txt", "w").write("".join(r[6].replace("\n", " ") + "\n" for r in rows))
    shutil.copy(f"{E.D2D}/stats.npz", f"{O}/stats.npz"); print(len(rows), "textos de entrenamiento")
def train(ph, seed):
    os.makedirs(f"{O}/{ph}", exist_ok=True)
    src = inspect.getsource(D2M.train).replace('("D2", "D1_POOL2")', '("D2",)')
    D2M.O = O; D2M.STEPS = STEPS_EXT; ns = dict(D2M.__dict__); exec(src, ns); r = ns["train"](ph, seed)
    open(f"{O}/{ph}/{seed}.train.done", "w").write("ok\n"); return r
def feats(ph):
    E.D2D_SAVE = E.D2D
    for part, tsvp, tokp in (("dev", f"{Y.O}/{ph}/cal.tsv", f"{Y.O}/{ph}/cal.tok"), ("test", f"{Y.EO}/{ph}/robust.tsv", f"{Y.EO}/{ph}/robust.tok")):
        rows = P0.tsv(tsvp)
        for s in P0.SEEDS[ph]:
            import torch
            st = np.load(f"{O}/stats.npz"); tok = D2M.Tok(tokp); enc = D2M.D2(); enc.load_state_dict(torch.load(f"{O}/{ph}/enc_D2_{s}.pt"))
            I = [i for i, r in enumerate(rows) if r[0] == s]
            with torch.no_grad(): C, LG, _ = enc(*tok.batch(I, st["mu"], st["sd"]))
            C, LG = C.numpy(), LG.numpy(); lines = ["idx,cond,text_id," + ",".join(f"c{i}" for i in range(10))]; meta = []
            for j, i in enumerate(I):
                r = rows[i]; c, ok = E.build_c(C[j], r[8]); pr = np.exp(LG[j] - LG[j].max()); pr /= pr.sum(); so = np.sort(pr)[::-1]
                lines.append(f"{r[1]},E2E_R,{r[5]}-{r[6]}," + ",".join(repr(float(x)) for x in c)); meta.append((r[1], f"{r[5]}-{r[6]}", int(np.argmax(c[:8])), so[0], so[0] - so[1], int(ok)))
            cp = f"{O}/{ph}/ctx_{part}_{s}.csv"; open(cp, "w").write("\n".join(lines) + "\n")
            p = subprocess.run([P0.B, "yo-feat", "--seed", s, "--weights", f"{P0.DATA}/{ph}/w_{s}.json", "--part", part, "--ctx", cp], capture_output=True, check=True)
            R = list(csv.DictReader(p.stdout.decode().splitlines())); assert len(R) == len(meta)
            with open(f"{O}/{ph}/yo_{part}_{s}.csv", "w") as f:
                f.write("idx,text_id,fam,fam_hat,p_d2,margin_d2,parse_ok,surprise,sigma,normr,ok16\n")
                for r, mt in zip(R, meta):
                    f.write(f"{r['idx']},{r['text_id']},{r['fam']},{mt[2]},{mt[3]:.6f},{mt[4]:.6f},{mt[5]},{np.log(float(r['surprise']) + 1e-9):.6f},{np.log(float(r['sigma'])):.6f},{np.log(float(r['normr'])):.6f},{r['ok16']}\n")
            open(f"{O}/{ph}/q_{part}_{s}.txt", "w").write(p.stderr.decode())
    print("feats ok", ph)
def metrics(base, ph, s):
    cal = list(csv.DictReader(open(f"{base}/{ph}/yo_dev_{s}.csv"))); ev = list(csv.DictReader(open(f"{base}/{ph}/yo_test_{s}.csv")))
    y = np.array([int(r["ok16"]) for r in ev]); p = Y.fit(cal, Y.F_YO)(ev)
    return {"acc16": y.mean(), "fam": np.mean([r["fam_hat"] == r["fam"] for r in ev]), "brier": np.mean((p - y) ** 2), "ece": Y.ece(p, y), "aurc": Y.aurc(p, y)}
def analyze(ph):
    S = P0.SEEDS[ph]; out = [f"## {ph.upper()} (16 seeds) EXT vs BASE", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    B = {s: metrics(Y.O, ph, s) for s in S}; X = {s: metrics(O, ph, s) for s in S}
    Pr = lambda n, x: out.append(f"| {n} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    for k in ("acc16", "fam", "brier", "ece", "aurc"):
        Pr(f"{k} BASE", P0.boot(np.array([B[s][k] for s in S]))); Pr(f"{k} EXT", P0.boot(np.array([X[s][k] for s in S])))
    d = lambda k, sg: P0.boot(np.array([sg * (X[s][k] - B[s][k]) for s in S]))
    Pr("**HX1** acc16 EXT − BASE", d("acc16", 1)); Pr("**HX2** familia EXT − BASE", d("fam", 1)); Pr("**HX3** Brier BASE − EXT", d("brier", -1)); Pr("HX3 (sec.) AURC BASE − EXT", d("aurc", -1))
    q = sum(int(open(f"{O}/{ph}/q_{pt}_{s}.txt").read().split("queries=")[1].split()[0]) for s in S for pt in ("dev", "test")); out.append(f"\nQueries v4 (yo-feat EXT): {q}")
    os.makedirs(ART, exist_ok=True); open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__":
    {"texts": texts, "train": train, "feats": feats, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
