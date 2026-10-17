"""E0–E2 memoria por capas (docs/preregistro_memoria_capas.md).  mem.py run PH | analyze PH"""
import sys, os, csv, collections, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0
O = "/workspace/orbita-data/mem"; ART = f"{P0.ROOT}/artifacts/orbita/mem"
CTRL = ["NOCDT", "SHUF", "CORRUPT", "ORACLE_ORB", "ORACLE_LS"]
def run(ph):
    os.makedirs(f"{O}/{ph}", exist_ok=True); os.makedirs(f"{ART}/{ph}", exist_ok=True)
    for s in P0.SEEDS[ph]:
        if os.path.exists(f"{O}/{ph}/{s}.done"): continue
        p = subprocess.run([P0.B, "mem", "--seed", s, "--weights", f"/workspace/orbita-data/{ph}/w_{s}.json"], capture_output=True, check=True)
        open(f"{O}/{ph}/out_{s}.csv", "wb").write(p.stdout); open(f"{ART}/{ph}/log_{s}.txt", "wb").write(p.stderr)
        open(f"{O}/{ph}/{s}.done", "w").write("ok\n"); print(ph, s, "ok", flush=True)
def analyze(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []
    for s in S:
        R = list(csv.DictReader(open(f"{O}/{ph}/out_{s}.csv")))
        g = collections.defaultdict(list)
        for r in R: g[(r["cond"], r["set"])].append(r)
        for (c, st), G in g.items():
            for h in ("1", "4"): per[(c, st, h)][s] = np.mean([int(r[f"ok{h}"]) for r in G])
        for c in {k[0] for k in g}: per[(c, "S", "4")][s] = 0.5 * (per[(c, "A", "4")][s] + per[(c, "B", "4")][s])
        no = {(r["set"], i): r for i, r in enumerate(x for x in R if x["cond"] == "NOCDT")}
        cd = [r for r in R if r["cond"] == "CDT"]
        per[("E0", "auroc", "")][s] = P0.auroc([float(r["r"]) for r in cd], [no[(r["set"], i)]["ok4"] == "0" for i, r in enumerate(cd)])
    v = lambda *k: np.array([per[k][s] for s in S]); Pr = lambda n, x: out.append(f"| {n} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    out += [f"## {ph.upper()} (16 seeds)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    conds = sorted({k[0] for k in per if k[0] != "E0"})
    for c in conds:
        for st in ("A", "B"):
            for h in ("1", "4"): Pr(f"acc{h} {st} {c}", P0.boot(v(c, st, h)))
        Pr(f"S {c}", P0.boot(v(c, "S", "4")))
    Pr("**E0a** AUROC(r_CDT → fallo acc4 NOCDT) − 0,5", P0.boot(v("E0", "auroc", "") - 0.5))
    Pr("**E1a** ORACLE_LS acc4(B)", P0.boot(v("ORACLE_LS", "B", "4")))
    Pr("**E1b** NOCDT acc4(B) − acc4(A)", P0.boot(v("NOCDT", "B", "4") - v("NOCDT", "A", "4")))
    for V in ("CDT", "CDT_NP", "CDT_GEN"):
        Pr(f"**E2-kill** S {V} − S RAW_{V}", P0.boot(v(V, "S", "4") - v(f"RAW_{V}", "S", "4")))
        ctl = CTRL + [f"RAW_{V}"]; mx = np.max(np.stack([v(c, "S", "4") for c in ctl]), 0)
        Pr(f"**E2-H** S {V} − max(controles+RAW)", P0.boot(v(V, "S", "4") - mx))
        for c in ctl: Pr(f"  S {V} − {c}", P0.boot(v(V, "S", "4") - v(c, "S", "4")))
    out.append("\nMeta por seed (sueños, novedad, presupuesto):")
    for s in S: out.append("- " + open(f"{ART}/{ph}/log_{s}.txt").read().strip())
    open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__" and sys.argv[1] in ("run", "analyze"):
    {"run": run, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])

# ---------------- etapa 2: E3, E4 (códec de campo, traductor = BANCO DE FRASES), E5 ----------------
import numnorm as NN
NAMES = {"es": ["traslación", "rotación", "reflexión", "escalado", "cizalla", "espiral afín", "rotación con escala", "deformación radial", "giro ondulado", "espiral nueva"],
         "en": ["translation", "rotation", "reflection", "scaling", "shear", "affine spiral", "scaled rotation", "radial warp", "wavy turn", "new spiral"],
         "fr": ["translation", "rotation", "réflexion", "mise à l'échelle", "cisaillement", "spirale affine", "rotation avec échelle", "déformation radiale", "virage ondulé", "nouvelle spirale"]}
TPL = [("es", "familia «{n}», alfa = {a}, beta = {b}", 2), ("es", "la regla es «{n}» con alfa {a} y beta {b}", 0),
       ("en", "family «{n}», alpha = {a}, beta = {b}", 2), ("en", "the rule is «{n}» with alpha {a} and beta {b}", 0),
       ("fr", "famille «{n}», alpha = {a}, bêta = {b}", 2), ("fr", "la règle est «{n}» avec alpha {a} et bêta {b}", 0)]
GRID = [round(-1 + 0.25 * i, 2) for i in range(9)]
def val(v, lang, f): return NN.loc(v, lang, 2) if f == 2 else NN.render(v, lang, 0)
def feat(slot, a, b): x = np.zeros(12); x[slot] = 1; x[10] = a; x[11] = b; return x
class FieldEncoder:  # mínimo: 12 aristas (octaedro), matriz compleja fija, fasores unitarios
    def __init__(s, seed): r = np.random.default_rng(seed); s.w = r.normal(size=(12, 12)) + 1j * r.normal(size=(12, 12))
    def encode(s, x): z = s.w @ x; return z / np.maximum(np.abs(z), 1e-12)
def bank(ph):
    T = [t for i, t in enumerate(TPL) if i % 2 == (0 if ph == "dev" else 1)]  # split por plantilla
    return [(sl, a, b, t.format(n=NAMES[l][sl], a=val(a, l, f), b=val(b, l, f))) for (l, t, f) in T for sl in range(10) for a in GRID for b in GRID]
def interpret(text):
    t = text.lower(); fam = None
    for L in NAMES.values():
        for i, n in enumerate(L):
            if f"«{n}»" in t: fam = i
    p = NN.parse2(t.split("»", 1)[1]) if "»" in t else None
    return fam, p
def run2(ph):
    os.makedirs(f"{O}/{ph}", exist_ok=True); os.makedirs(f"{ART}/{ph}", exist_ok=True)
    for s in P0.SEEDS[ph]:
        if os.path.exists(f"{O}/{ph}/{s}.s2.done"): continue
        p = subprocess.run([P0.B, "mem", "--seed", s, "--weights", f"/workspace/orbita-data/{ph}/w_{s}.json", "--stage2"], capture_output=True, check=True)
        open(f"{O}/{ph}/s2_{s}.csv", "wb").write(p.stdout); open(f"{ART}/{ph}/log2_{s}.txt", "wb").write(p.stderr)
        open(f"{O}/{ph}/{s}.s2.done", "w").write("ok\n"); print(ph, s, "s2 ok", flush=True)
def analyze2(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []
    BK = bank(ph); parsed = [interpret(t) for *_, t in BK]
    bad = sum(1 for (sl, a, b, _), (f, p) in zip(BK, parsed) if f != sl or p is None or abs(p[0] - a) > 1e-6 or abs(p[1] - b) > 1e-6)
    for s in S:
        R = list(csv.DictReader(open(f"{O}/{ph}/s2_{s}.csv")))
        R1 = list(csv.DictReader(open(f"{O}/{ph}/out_{s}.csv")))
        for c in ("CDT", "CDT_FIXED", "CDT_RANDPRUNE"):
            for st in "AB": per[(c, st)][s] = np.mean([int(r["ok4"]) for r in R if r["cond"] == c and r["set"] == st])
            per[(c, "S")][s] = 0.5 * (per[(c, "A")][s] + per[(c, "B")][s])
        for c in ("CDT_NP",):
            for st in "AB": per[(c, st)][s] = np.mean([int(r["ok4"]) for r in R1 if r["cond"] == c and r["set"] == st])
            per[(c, "S")][s] = 0.5 * (per[(c, "A")][s] + per[(c, "B")][s])
        enc = FieldEncoder(int(s, 16) ^ 0xF1E1D); codes = np.stack([enc.encode(feat(sl, a, b)) for sl, a, b, _ in BK])
        rng = np.random.default_rng(int(s, 16) ^ 0xBA4C); perm = rng.permutation(len(BK))
        for cond, src, pb in (("CODEC", "CDT", False), ("CODEC_SHUF", "SHUF", False), ("CODEC_BANKPERM", "CDT", True)):
            G = [r for r in R if r["cond"] == src]; fok, perr = [], []
            for r in G:
                z = enc.encode(feat(int(r["slot"]), float(r["p1"]), float(r["p2"])))
                j = int(np.argmax(np.real(codes.conj() @ z)))
                f, p = parsed[perm[j]] if pb else parsed[j]
                fok.append(f == int(r["truefam"])); perr.append(np.nan if p is None else 0.5 * (abs(p[0] - float(r["p1"])) + abs(p[1] - float(r["p2"]))))
            per[(cond, "fam")][s] = np.mean(fok); per[(cond, "perr")][s] = np.nanmean(perr)
    v = lambda *k: np.array([per[k][s] for s in S]); Pr = lambda n, x: out.append(f"| {n} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    out += [f"## {ph.upper()} etapa 2 (16 seeds)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for c in ("CDT", "CDT_FIXED", "CDT_RANDPRUNE", "CDT_NP"):
        for k in ("A", "B", "S"): Pr(f"{'acc4 ' + k if k != 'S' else 'S'} {c}", P0.boot(v(c, k)))
    Pr("**E3** S CDT (disparo por saturación) − S CDT_FIXED", P0.boot(v("CDT", "S") - v("CDT_FIXED", "S")))
    Pr("**E5a** S CDT (poda trusted) − S CDT_NP (sin poda)", P0.boot(v("CDT", "S") - v("CDT_NP", "S")))
    Pr("**E5b** S CDT (poda trusted) − S CDT_RANDPRUNE", P0.boot(v("CDT", "S") - v("CDT_RANDPRUNE", "S")))
    for c in ("CODEC", "CODEC_SHUF", "CODEC_BANKPERM"):
        Pr(f"familia recuperada {c}", P0.boot(v(c, "fam"))); Pr(f"|Δp| medio {c}", P0.boot(v(c, "perr")))
    Pr("**E4** familia CODEC − max(CODEC_SHUF, CODEC_BANKPERM)", P0.boot(v("CODEC", "fam") - np.maximum(v("CODEC_SHUF", "fam"), v("CODEC_BANKPERM", "fam"))))
    out.append(f"\nBanco de frases {ph}: {len(BK)} frases, plantillas {[i for i in range(len(TPL)) if i % 2 == (0 if ph == 'dev' else 1)]}; frases mal interpretadas por el lector (familia o p): {bad}")
    out.append("Meta etapa 2 por seed:"); out += ["- " + open(f"{ART}/{ph}/log2_{s}.txt").read().strip() for s in S]
    open(f"{ART}/summary2_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__" and sys.argv[1] in ("run2", "analyze2"):
    {"run2": run2, "analyze2": analyze2}[sys.argv[1]](*sys.argv[2:])
