"""Demo de la pila final (docs/demo_orbita.md):
texto → Gemma capa 13 → D2 (3000 pasos) + numnorm → ÓRBITA-v1 congelada [+ re-observación k] → yo (informador) → explicación Gemma (greedy).
  python orbita/llm/demo.py --text "Aplica una rotación con parámetros 0,5 y -0,2." [--obs 1|2] [--observed "y1|y2" | --simulate fam,p1,p2]
                            [--x0 "x,y,...(12)"] [--seed 0xA400] [--no-explain] [--json]"""
import sys, os, json, argparse, subprocess, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, numnorm as NN, e2e as E, yo as Y
from make_texts import FAM
X0_DEFAULT = [0.8, 0.0, 0.4, 0.69282, -0.4, 0.69282, -0.8, 0.0, -0.4, -0.69282, 0.4, -0.69282]   # hexágono (círculo k=6, r=0,8)
PROMPT = ("<start_of_turn>user\nEres el monitor de un modelo de predicción llamado ÓRBITA. {b}\nExplica en una o dos frases, en español y sin inventar datos, "
          "qué transformación cree ÓRBITA que se aplica y qué tan seguro está.\n<end_of_turn>\n<start_of_turn>model\n")
def simulate(spec, x0, k):
    sys.path.insert(0, P0.ROOT + "/orbita/llm"); f, p1, p2 = spec.split(","); ys = []; y = x0
    for _ in range(k):
        o = subprocess.check_output([P0.B, "demo-rule", "--fam", f, "--p", f"{p1},{p2}", "--x", ",".join(map(repr, y))]).decode().strip(); y = [float(v) for v in o.split(",")]; ys.append(y)
    return ys
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--text", required=True); ap.add_argument("--seed", default="0xA400"); ap.add_argument("--obs", type=int, default=0)
    ap.add_argument("--observed"); ap.add_argument("--simulate"); ap.add_argument("--x0"); ap.add_argument("--no-explain", action="store_true"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv); ph = "dev" if a.seed.upper().startswith("0XA4") else "confirm"
    x0 = [float(v) for v in a.x0.split(",")] if a.x0 else X0_DEFAULT
    ys = [[float(v) for v in t.split(",")] for t in a.observed.split("|")][:a.obs or None] if a.observed else (simulate(a.simulate, x0, a.obs) if a.simulate and a.obs else [])
    if a.obs and len(ys) < a.obs: sys.exit("--obs k requiere k estados en --observed o --simulate")
    with tempfile.TemporaryDirectory() as td:
        open(f"{td}/in.txt", "w").write(a.text.replace("\n", " ") + "\n")
        subprocess.run([P0.B, "gemma-embed", "--model", E.GEMMA, "--in", f"{td}/in.txt", "--out", f"{td}/t.tok", "--exit-after", "12", "--per-token"], check=True, capture_output=True)
        C, LG = Y.d2_logits(f"{td}/t.tok", ph, a.seed, [0])
    c, ok = E.build_c(C[0], a.text); pr = np.exp(LG[0] - LG[0].max()); pr /= pr.sum(); so = np.sort(pr)[::-1]
    cmd = [P0.B, "demo-step", "--weights", f"/workspace/orbita-data/{ph}/w_{a.seed}.json", "--c", ",".join(map(repr, c)), "--x0", ",".join(map(repr, x0))]
    if ys: cmd += ["--obs", "|".join(",".join(map(repr, y)) for y in ys)]
    st = json.loads(subprocess.check_output(cmd).decode())
    feat = {"fam_hat": str(int(np.argmax(c[:8]))), "p_d2": f"{so[0]}", "margin_d2": f"{so[0] - so[1]}", "parse_ok": str(int(ok)), "sigma": f"{np.log(st['sigma'])}", "normr": f"{np.log(st['normr'])}"}
    cal = Y.load(ph, "dev", a.seed); cols = [f for f in Y.F_YO if f != "surprise" or "surprise" in st]
    if "surprise" in st: feat["surprise"] = f"{np.log(st['surprise'] + 1e-9)}"
    phat = float(Y.fit(cal, cols)([feat])[0])
    res = {"texto": a.text, "seed": a.seed, "familia_leida": FAM["es"][int(feat["fam_hat"])], "p_leido": c[8:], "p_fuente": "numnorm" if ok else "fallback D2",
           "prob_encoder": round(float(so[0]), 4), "trayectoria_h1_h16": st["base_traj"], "yo": {"confianza_acierto_h16": round(phat, 4), "rasgos": cols,
           "sorpresa_vs_obs": st.get("surprise"), "sigma_jacobiano": st["sigma"], "ratio_norma_h16": st["normr"]}}
    if "reid" in st:
        res["re_observacion"] = {"k": st["reid"]["k"], "familia_reidentificada": FAM["es"][st["reid"]["fam"]], "p_reidentificado": st["reid"]["p"],
                                 "error_ajuste": st["reid"]["fit_err"], "trayectoria_h1_h16": st["reid_traj"]}
    if not a.no_explain:
        b = Y.block(feat | {"surprise": feat.get("surprise", f"{np.log(1e-9)}")}, phat, "YO")
        if "surprise" not in st: b = b.replace(f"- sorpresa frente a la primera observación: {1e-9:.3f}\n", "- sorpresa frente a la primera observación: sin observación\n")
        if "reid" in st: b += f"\n- tras observar {st['reid']['k']} paso(s) reales, la familia re-identificada es: {res['re_observacion']['familia_reidentificada']}"
        pr_ = Y.server(); txt, _ = Y.complete(PROMPT.format(b=b)); res["explicacion_gemma"] = txt.strip(); res["nota"] = "explicación generada por Gemma 2 2B Q4_K_M (greedy) a partir del yo; no verifica el mundo"
        if pr_: pr_.terminate()
    print(json.dumps(res, ensure_ascii=False, indent=None if a.json else 1)); return res
if __name__ == "__main__": main()
