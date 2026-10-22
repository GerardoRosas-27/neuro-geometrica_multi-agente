"""D3-num (opción A) vs ÓRBITA-robusta (opción B). Prerregistro docs/preregistro_d3_vs_robusta.md.
  d3.py ctx PHASE      A: familia de D2 (argmax) + p leído por parser de números del texto; B: ĉ de D2 tal cual
  d3.py eval PHASE     llm-eval: ÓRBITA-v1 con {TRUE,RAND,D2,D1_POOL2,D3NUM}; ÓRBITA-B con {TRUE,D2,D1_POOL2}; orbita eval (§16) de B
  d3.py analyze PHASE  contrastes, IC95 pareado (10000, semilla 12345) y regla de selección"""
import sys, os, re, csv, collections, subprocess, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0
O = "/workspace/orbita-data/llm_d3"; L = "/workspace/orbita-data/llm"; D2D = "/workspace/orbita-data/llm_d2"; ART = f"{P0.ROOT}/artifacts/orbita/llm_d3"
NUM = re.compile(r"[-−]?\d+(?:[.,]\d+)?")
def parse(t):
    v = [float(x.replace("−", "-").replace(",", ".")) for x in NUM.findall(t)]
    return (max(-1.0, min(1.0, v[0])), max(-1.0, min(1.0, v[1]))) if len(v) >= 2 else None
def ctx(ph):
    rows = P0.tsv(f"{L}/{ph}/all.tsv"); txt = {(r[0], r[1], f"{r[5]}-{r[6]}"): r[7] for r in rows}; nfail = 0
    for s in P0.SEEDS[ph]:
        old = open(f"{D2D}/{ph}/ctxin_{s}.csv").read().splitlines(); hdr = old[0]
        base = [l for l in old[1:] if l.split(",")[1] in ("TRUE", "RAND", "D2", "D1_POOL2")]
        A = [hdr] + base; B = [hdr] + [l for l in base if l.split(",")[1] != "RAND"]
        for l in base:
            f = l.split(",")
            if f[1] != "D2": continue
            c = [float(x) for x in f[3:13]]; fam = int(np.argmax(c[:8])); p = parse(txt[(s, f[0], f[2])])
            if p is None: nfail += 1; p = (c[8], c[9])
            oh = [0.0] * 8; oh[fam] = 1.0
            A.append(f"{f[0]},D3NUM,{f[2]}," + ",".join(repr(x) for x in oh + list(p)))
        open(f"{O}/{ph}/ctxA_{s}.csv", "w").write("\n".join(A) + "\n"); open(f"{O}/{ph}/ctxB_{s}.csv", "w").write("\n".join(B) + "\n")
    json.dump({"parse_fail": nfail}, open(f"{O}/{ph}/parse.json", "w")); print("ctx ok", ph, "parse_fail", nfail)
def ev(ph):
    os.makedirs(f"{ART}/{ph}/B_eval", exist_ok=True)
    for s in P0.SEEDS[ph]:
        for tag, w, cf in (("A", f"/workspace/orbita-data/{ph}/w_{s}.json", f"{O}/{ph}/ctxA_{s}.csv"), ("B", f"{O}/B/{ph}/w_{s}.json", f"{O}/{ph}/ctxB_{s}.csv")):
            p = subprocess.run([P0.B, "llm-eval", "--seed", s, "--weights", w, "--ctx", cf], capture_output=True, check=True)
            open(f"{ART}/{ph}/eval{tag}_{s}.csv", "wb").write(p.stdout); open(f"{ART}/{ph}/queries{tag}_{s}.txt", "wb").write(p.stderr)
        subprocess.run([P0.B, "eval", "--seed", s, "--weights", f"{O}/B/{ph}/w_{s}.json", "--meta", f"/workspace/orbita-data/{ph}/{s}/meta.json", "--out", f"{ART}/{ph}/B_eval", "--label", "ORBITA_B"], check=True, capture_output=True)
    print("eval ok", ph)
def analyze(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []
    rows = P0.tsv(f"{L}/{ph}/all.tsv"); truep = {(r[0], r[1]): (float(r[3]), float(r[4])) for r in rows}
    for s in S:
        for tag in ("A", "B"):
            R = list(csv.DictReader(open(f"{ART}/{ph}/eval{tag}_{s}.csv")))
            for c in {r["cond"] for r in R}:
                G = [r for r in R if r["cond"] == c]
                for h in (1, 8, 16, 32): per[(tag, c, h)][s] = np.mean([int(r[f"ok{h}"]) for r in G])
                per[(tag, c, "rel1")][s] = np.median([float(r["rel1"]) for r in G])
            G = [r for r in R if r["cond"] == ("D3NUM" if tag == "A" else "D2")]
            per[(tag, "famT")][s] = np.mean([r["fam_hat_teacher"] == r["fam"] for r in G])
            per[(tag, "auroc")][s] = P0.auroc([float(r["teacher_div16"]) for r in G], [r["ok16"] == "0" for r in G])
        A = [l.split(",") for l in open(f"{O}/{ph}/ctxA_{s}.csv").read().splitlines()[1:] if ",D3NUM," in l]
        per[("A", "perr")][s] = np.median([max(abs(float(f[11]) - truep[(s, f[0])][0]), abs(float(f[12]) - truep[(s, f[0])][1])) for f in A])
        m = list(csv.reader(open(f"{ART}/{ph}/B_eval/metrics_{s}.csv"))); hd = m[0]
        g = [dict(zip(hd, r)) for r in m[1:]]
        per[("B", "gate")][s] = float(any(r["condition"] == "GATE" and r["status"] == "PASS" for r in g))
        per[("B", "acc1_test")][s] = float([r for r in g if r["partition"] == "TEST"][0]["accuracy"])
    v = lambda *k: np.array([per[k][s] for s in S]); Pr = lambda n, x: out.append(f"| {n} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    out += [f"## {ph.upper()} (16 seeds; 16 ejemplos TEST iterables × 6 textos)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for c in ("TRUE", "RAND", "D2", "D1_POOL2", "D3NUM"):
        for h in (1, 8, 16, 32): Pr(f"v1 + {c} acc h{h}", P0.boot(v("A", c, h)))
    for c in ("TRUE", "D2", "D1_POOL2"):
        for h in (1, 8, 16, 32): Pr(f"B + {c} acc h{h}", P0.boot(v("B", c, h)))
    Pr("max|p̂−p| mediano D3NUM", P0.boot(v("A", "perr"))); Pr("B: acc 1 paso TEST completo (§16, c real)", P0.boot(v("B", "acc1_test"))); Pr("B: fracción GATE E33 PASS", P0.boot(v("B", "gate")))
    Pr("**H-A1** acc16 D3NUM − D2 (v1)", P0.boot(v("A", "D3NUM", 16) - v("A", "D2", 16)))
    Pr("**H-A2** acc16 D3NUM − TRUE (v1)", P0.boot(v("A", "D3NUM", 16) - v("A", "TRUE", 16)))
    Pr("**H-B1** acc16 B+D2 − v1+D2", P0.boot(v("B", "D2", 16) - v("A", "D2", 16)))
    Pr("**H-B2** acc16 B+TRUE − v1+TRUE", P0.boot(v("B", "TRUE", 16) - v("A", "TRUE", 16)))
    Pr("**H-B3** acc16 B+D2 − v1+TRUE", P0.boot(v("B", "D2", 16) - v("A", "TRUE", 16)))
    Pr("**SEL** acc16 A(v1+D3NUM) − B(B+D2)", P0.boot(v("A", "D3NUM", 16) - v("B", "D2", 16)))
    Pr("SEL-2 acc8 A − B", P0.boot(v("A", "D3NUM", 8) - v("B", "D2", 8)))
    Pr("H-C1 A: familia teacher − 1/8", P0.boot(v("A", "famT") - 0.125)); Pr("H-C1 A: AUROC − 0.5", P0.boot(v("A", "auroc") - 0.5))
    Pr("H-C1 B: familia teacher − 1/8", P0.boot(v("B", "famT") - 0.125)); Pr("H-C1 B: AUROC − 0.5", P0.boot(v("B", "auroc") - 0.5))
    q = sum(int(open(f"{ART}/{ph}/queries{t}_{s}.txt").read().split("queries=")[1].split()[0]) for s in S for t in "AB")
    out.append(f"\nQueries totales (llm-eval A+B): {q}; parse_fail: {json.load(open(f'{O}/{ph}/parse.json'))['parse_fail']}")
    open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__":
    for ph in ("dev", "confirm"): os.makedirs(f"{O}/{ph}", exist_ok=True); os.makedirs(f"{O}/B/{ph}", exist_ok=True)
    {"ctx": ctx, "eval": ev, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
