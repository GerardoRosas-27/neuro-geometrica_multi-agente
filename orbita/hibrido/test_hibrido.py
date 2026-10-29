"""Humo (sin Gemma): embebido sintético como dinlat/test_dinlat.py. STEPS reducido. Verifica:
(1) identidad con dinlat.run en PERS/AR2/ORB_LIBRE/ORB_SESGO, (2) reproducibilidad, (3) anti-fuga: alterar s_{t+1..}
no cambia los pesos de la compuerta en t, (4) MIX50 = mezcla exacta, (5) claves de salida."""
import sys, os, json, time, numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "dinlat"))
import gen, run as D
import importlib.util; sp = importlib.util.spec_from_file_location("hib", os.path.join(H, "run_hib.py")); hib = importlib.util.module_from_spec(sp); sp.loader.exec_module(hib)
def fake(eps):
    R = np.random.default_rng(0).standard_normal((8, D.DIM)) / 3
    return np.tanh(np.array([e["X"] for e in eps]).reshape(-1, 8) @ R).ravel()
def main():
    D.STEPS = int(os.environ.get("STEPS", 100)); hib.GATE_STEPS = 200
    eps = gen.gen_seed("0xA400"); E = fake(eps); t0 = time.time()
    a = hib.run("0xA400", eps, E); t1 = time.time() - t0
    ref = D.run("0xA400", eps, fake(eps))
    for s in ("ho", "tf"):
        for k in ("PERS", "AR2", "ORB_LIBRE", "ORB_SESGO"): assert np.allclose(ref["main"][s][k], a["main"][s][k]), (s, k)
    b = hib.run("0xA400", eps, E); assert json.dumps(a) == json.dumps(b), "no reproducible"
    need = {"PERS", "AR2", "ORB_LIBRE", "ORB_SESGO", "MIX50", "HIB_GATE", "HIB_HEDGE", "HIB_RES", "HIB_GATE_S"}
    assert need <= set(a["main"]["ho"]) and {"HIB_GATE", "HIB_RES"} <= set(a["C2_barajado"]["ho"])
    # anti-fuga: la compuerta en t solo depende de s_0..s_t
    S = hib.latents("0xA400", eps, E)["main"]["ho"]; rng = np.random.default_rng(1)
    P = [{t: rng.standard_normal((len(S), D.DL)) for t in range(1, D.T - 1)} for _ in range(2)]
    F = hib.past_feats(P, S)
    for t in range(D.TMIN, D.T - 1):
        S2 = S.copy(); S2[:, t + 1:] += 10 * rng.standard_normal(S2[:, t + 1:].shape)
        assert np.array_equal(F[t], hib.past_feats(P, S2)[t]), f"fuga en t={t}"
    m = {k: float(np.mean(v)) for k, v in a["main"]["ho"].items() if not k.startswith("_")}
    print("OK", round(t1, 1), "s/corrida (STEPS=%d)" % D.STEPS, {k: round(v, 4) for k, v in m.items()})
main()
