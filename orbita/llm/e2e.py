"""Pipeline end-to-end texto(es/en/fr) → Gemma capa 13 por token → D2 (familia, argmax) + parser numérico multilingüe (numnorm)
→ ÓRBITA-v1 congelada → CDT-RQM-epr consolidado (familia inferida por el teacher y residuo teacher_div16). Prerregistro:
docs/preregistro_integracion_e2e.md. No toca v4 (sólo lee el crate y usa el binario `orbita llm-eval`).
  e2e.py run --phase dev --seed 0xA400 --idx N --text "..."   [--json]   un texto → ĉ + predicción + salida CDT
  e2e.py texts PH        textos de robustez (plantillas 6–7, nunca vistas por D2; p en palabras/fracciones/grados/vueltas/radianes)
  e2e.py ctx PH | eval PH | analyze PH                                 experimento de robustez"""
import sys, os, csv, json, collections, subprocess, tempfile, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, numnorm as NN
from make_texts import FAM, ART as ARTI
O = f"{P0.DATA}/llm_e2e"; L = f"{P0.DATA}/llm"; D2D = f"{P0.DATA}/llm_d2"; D3D = f"{P0.DATA}/llm_d3"
ART = f"{P0.ROOT}/artifacts/orbita/llm_e2e"; GEMMA = P0.GEMMA_GGUF
T = {"es": ["Transforma la figura con {art} {f}: el primer valor es {a} y el segundo es {b}.", "Se pide {art} {f} de valores {a} y {b}."],
     "en": ["Transform the shape with {art} {f}: the first value is {a} and the second is {b}.", "Requested: {art} {f} with values {a} and {b}."],
     "fr": ["Transforme la figure avec {art} {f} : la première valeur est {a} et la seconde est {b}.", "Demandé : {art} {f} de valeurs {a} et {b}."]}
LANGS = ("es", "en", "fr")
def fmts(idx, k, li): f = (int(idx) + k + li) % 5; return f, (f + 2) % 5

def build_c(cd2, text):
    """ĉ = one-hot(argmax familia D2) ++ p parseado; si el parser no encuentra 2 cantidades, fallback p̂ de D2"""
    fam = int(np.argmax(cd2[:8])); p = NN.parse2(text); ok = p is not None
    if not ok: p = (float(cd2[8]), float(cd2[9]))
    oh = [0.0] * 8; oh[fam] = 1.0
    return oh + [float(p[0]), float(p[1])], ok

def d2_predict(tokpath, ph, seed, idxs):
    import torch, d2 as D2
    st = np.load(f"{D2D}/stats.npz"); tok = D2.Tok(tokpath); enc = D2.D2(); enc.load_state_dict(torch.load(f"{D2D}/{ph}/enc_D2_{seed}.pt"))
    with torch.no_grad(): return enc(*tok.batch(idxs, st["mu"], st["sd"]))[0].numpy()

def llm_eval(seed, ph, ctxfile):
    p = subprocess.run([P0.B, "llm-eval", "--seed", seed, "--weights", f"{P0.DATA}/{ph}/w_{seed}.json", "--ctx", ctxfile], capture_output=True, check=True)
    return p.stdout.decode(), p.stderr.decode()

def run(argv):
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="dev"); ap.add_argument("--seed", required=True); ap.add_argument("--idx", required=True)
    ap.add_argument("--text", required=True); a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as td:
        open(f"{td}/in.txt", "w").write(a.text.replace("\n", " ") + "\n")
        subprocess.run([P0.B, "gemma-embed", "--model", GEMMA, "--in", f"{td}/in.txt", "--out", f"{td}/t.tok", "--exit-after", "12", "--per-token"], check=True, capture_output=True)
        cd2 = d2_predict(f"{td}/t.tok", a.phase, a.seed, [0])[0]; c, ok = build_c(cd2, a.text)
        open(f"{td}/c.csv", "w").write("idx,cond,text_id," + ",".join(f"c{i}" for i in range(10)) + "\n" + f"{a.idx},E2E,cli," + ",".join(repr(x) for x in c) + "\n")
        out, err = llm_eval(a.seed, a.phase, f"{td}/c.csv")
    r = next(csv.DictReader(out.splitlines()))
    res = {"seed": a.seed, "idx": int(a.idx), "text": a.text, "parse_ok": ok, "fam_hat": int(np.argmax(c[:8])), "p_hat": c[8:], "c_hat": c,
           "orbita_ok": {h: int(r[f"ok{h}"]) for h in (1, 2, 4, 8, 16, 32)}, "orbita_rel": {h: float(r[f"rel{h}"]) for h in (1, 16, 32)},
           "z16": [float(v) for v in r["z16"].split(";")], "cdt_fam_hat": int(r["fam_hat_teacher"]), "cdt_residual16": float(r["teacher_div16"]),
           "fam_true": int(r["fam"]), "queries": err.strip()}
    print(json.dumps(res, ensure_ascii=False))

def texts(ph):
    seen, out = set(), []
    for r in P0.tsv(f"{L}/{ph}/all.tsv"):
        key = (r[0], r[1])
        if key in seen: continue
        seen.add(key); fam, p1, p2 = int(r[2]), float(r[3]), float(r[4])
        for li, lang in enumerate(LANGS):
            for k in range(2):
                fa, fb = fmts(r[1], k, li)
                t = T[lang][k].format(art=ARTI[lang][fam], f=FAM[lang][fam], a=NN.render(p1, lang, fa), b=NN.render(p2, lang, fb))
                out.append([r[0], r[1], r[2], r[3], r[4], lang, str(6 + k), f"{NN.FMT_NAMES[fa]}+{NN.FMT_NAMES[fb]}", t])
    os.makedirs(f"{O}/{ph}", exist_ok=True)
    open(f"{O}/{ph}/robust.tsv", "w").write("".join("\t".join(x) + "\n" for x in out)); open(f"{O}/{ph}/robust.txt", "w").write("".join(x[8] + "\n" for x in out))
    print(len(out), "textos", ph)

def ctx(ph):
    rows = P0.tsv(f"{O}/{ph}/robust.tsv"); stats = collections.Counter(); fails = {}
    for s in P0.SEEDS[ph]:
        old = open(f"{D3D}/{ph}/ctxA_{s}.csv").read().splitlines(); hdr = old[0]
        out = [hdr] + [l.replace(",D2,", ",D2_DIG,").replace(",D3NUM,", ",D3NUM_DIG,") for l in old[1:] if l.split(",")[1] in ("TRUE", "D2", "D3NUM")]
        I = [i for i, r in enumerate(rows) if r[0] == s]; C = d2_predict(f"{O}/{ph}/robust.tok", ph, s, I); nf = 0
        for j, i in enumerate(I):
            r = rows[i]; tid = f"{r[5]}-{r[6]}"; c, ok = build_c(C[j], r[8]); nf += (not ok); stats[(r[7].split("+")[0], ok)] += 1; stats[(r[7].split("+")[1], ok)] += 1
            fmt = lambda v: ",".join(repr(float(x)) for x in v)
            out += [f"{r[1]},D2_R,{tid}," + fmt(C[j]), f"{r[1]},E2E_R,{tid}," + fmt(c)]
        fails[s] = nf; open(f"{O}/{ph}/ctx_{s}.csv", "w").write("\n".join(out) + "\n")
    json.dump({"parse_fail_por_seed": fails, "parse_fail": sum(fails.values()), "n_textos": len(rows),
               "por_formato": {f"{k[0]}|{'ok' if k[1] else 'fail'}": v for k, v in sorted(stats.items())}}, open(f"{O}/{ph}/parse.json", "w"), indent=1)
    print("ctx ok", ph, "parse_fail", sum(fails.values()))

def ev(ph):
    os.makedirs(f"{ART}/{ph}", exist_ok=True)
    for s in P0.SEEDS[ph]:
        o, e = llm_eval(s, ph, f"{O}/{ph}/ctx_{s}.csv"); open(f"{ART}/{ph}/eval_{s}.csv", "w").write(o); open(f"{ART}/{ph}/queries_{s}.txt", "w").write(e)
    print("eval ok", ph)

def analyze(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []; rows = P0.tsv(f"{O}/{ph}/robust.tsv"); pj = json.load(open(f"{O}/{ph}/parse.json"))
    truep = {(r[0], r[1]): (float(r[3]), float(r[4])) for r in rows}; fmt_of = {(r[0], r[1], f"{r[5]}-{r[6]}"): r[7] for r in rows}
    CONDS = ("TRUE", "D2_DIG", "D3NUM_DIG", "D2_R", "E2E_R")
    for s in S:
        R = list(csv.DictReader(open(f"{ART}/{ph}/eval_{s}.csv"))); g = lambda c: [r for r in R if r["cond"] == c]
        for c in CONDS:
            for h in (1, 8, 16, 32): per[(c, h)][s] = np.mean([int(r[f"ok{h}"]) for r in g(c)])
            per[("rel1", c)][s] = np.median([float(r["rel1"]) for r in g(c)])
            if c != "TRUE": per[("fam", c)][s] = np.mean([r["fam_hat_ctx"] == r["fam"] for r in g(c)])
        for lg in LANGS: per[("E2E_" + lg,)][s] = np.mean([int(r["ok16"]) for r in g("E2E_R") if r["text_id"].startswith(lg)])
        for fi in range(5): per[("E2Ef", fi)][s] = np.mean([int(r["ok16"]) for r in g("E2E_R") if fmt_of[(s, r["idx"], r["text_id"])].split("+")[0] == NN.FMT_NAMES[fi]])
        per[("pfail",)][s] = pj["parse_fail_por_seed"][s] / len(g("E2E_R"))
        cf = [l.split(",") for l in open(f"{O}/{ph}/ctx_{s}.csv").read().splitlines()[1:] if ",E2E_R," in l]
        per[("perr",)][s] = np.median([max(abs(float(f[11]) - truep[(s, f[0])][0]), abs(float(f[12]) - truep[(s, f[0])][1])) for f in cf])
        G = g("E2E_R"); per[("famT",)][s] = np.mean([r["fam_hat_teacher"] == r["fam"] for r in G])
        per[("auroc",)][s] = P0.auroc([float(r["teacher_div16"]) for r in G], [r["ok16"] == "0" for r in G])
        per[("res_ok",)][s] = np.median([float(r["teacher_div16"]) for r in G if r["ok16"] == "1"] or [np.nan]); per[("res_fail",)][s] = np.median([float(r["teacher_div16"]) for r in G if r["ok16"] == "0"] or [np.nan])
    v = lambda *k: np.array([per[k][s] for s in S]); Pr = lambda n, x: out.append(f"| {n} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    out += [f"## {ph.upper()} (16 seeds; 16 ejemplos TEST iterables × 6 textos por condición)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for c in CONDS:
        for h in (1, 8, 16, 32): Pr(f"acc h{h} {c}", P0.boot(v(c, h)))
        Pr(f"rel_err mediano h1 {c}", P0.boot(v("rel1", c)))
        if c != "TRUE": Pr(f"familia correcta (argmax D2) {c}", P0.boot(v("fam", c)))
    for lg in LANGS: Pr(f"acc16 E2E_R {lg}", P0.boot(v("E2E_" + lg)))
    for fi in range(5): Pr(f"acc16 E2E_R, 1er parámetro en {NN.FMT_NAMES[fi]}", P0.boot(v("E2Ef", fi)))
    Pr("max|p̂−p| mediano E2E_R", P0.boot(v("perr"))); Pr("**H-R0** tasa parse_fail E2E_R", P0.boot(v("pfail")))
    Pr("**H-R1** acc16 E2E_R − D3NUM_DIG", P0.boot(v("E2E_R", 16) - v("D3NUM_DIG", 16)))
    Pr("**H-R2** acc16 E2E_R − D2_R", P0.boot(v("E2E_R", 16) - v("D2_R", 16)))
    Pr("**H-R3** acc16 E2E_R − TRUE", P0.boot(v("E2E_R", 16) - v("TRUE", 16)))
    Pr("**H-C2a** familia teacher (E2E_R) − 1/8", P0.boot(v("famT") - 0.125)); Pr("**H-C2b** AUROC residuo16 → fallo − 0.5", P0.boot(v("auroc") - 0.5))
    Pr("residuo16 mediano (aciertos h16)", P0.boot(v("res_ok"))); Pr("residuo16 mediano (fallos h16)", P0.boot(v("res_fail")))
    q = sum(int(open(f"{ART}/{ph}/queries_{s}.txt").read().split("queries=")[1].split()[0]) for s in S)
    out.append(f"\nQueries totales (llm-eval): {q}; parse_fail: {pj['parse_fail']} de {pj['n_textos']} textos; por formato: {pj['por_formato']}")
    open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run": run(sys.argv[2:])
    else: {"texts": texts, "ctx": ctx, "eval": ev, "analyze": analyze}[cmd](*sys.argv[2:])
