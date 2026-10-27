"""Pruebas Paso 0 (sin Gemma): embebido sintético = tanh(proyección aleatoria del estado)."""
import json, sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import gen, run
def fake(eps):
    R = np.random.default_rng(0).standard_normal((8, run.DIM)) / 3
    return np.tanh(np.array([e["X"] for e in eps]).reshape(-1, 8) @ R).ravel()
def main():
    eps = gen.gen_seed("0xA400"); assert len(eps) == 84 and all(np.array(e["X"]).shape == (9, 8) for e in eps)
    assert gen.gen_seed("0xA400") == eps and gen.gen_seed("0xA401") != eps            # generación determinista
    assert not set(gen.TRAIN_FAMS) & set(gen.TRANSFER_FAMS)
    assert set(json.dumps(gen.gen_seed("0xB400"))) and gen.SEEDS["dev"][0] == "0xA400" and gen.SEEDS["confirm"][-1] == "0xB40F"
    run.STEPS = int(os.environ.get("STEPS", 300)); t0 = time.time()
    a = run.run("0xA400", eps, fake(eps)); b = run.run("0xA400", eps, fake(eps))
    assert json.dumps(a) == json.dumps(b), "no reproducible"                            # reproducibilidad bit a bit
    m = a["main"]["ho"]; mean = {k: float(np.mean(v)) for k, v in m.items()}
    assert mean["AR2"] < mean["PERS"], mean                                              # el AR aprende algo
    print("OK", round(time.time() - t0, 1), "s (2 corridas)", {k: round(v, 4) for k, v in mean.items()})
main()
