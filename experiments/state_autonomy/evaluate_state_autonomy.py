#!/usr/bin/env python3
"""Evaluate ORBITA weights on unseen latent mechanisms and states.

Usage:
  python evaluate_state_autonomy.py --weights full.json --weights a.json
"""
import argparse, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
ORB = os.path.join(ROOT, "orbita")
sys.path.insert(0, ORB)
from train_orbita import Orbita
from generate_state_autonomy import make_mechanism, step, fingerprint, Z


def load_model(path):
    w = json.load(open(path, encoding="utf-8"))
    m = Orbita(cdim=10, variant=w.get("variant", "full"))
    import torch
    with torch.no_grad():
        for mod, wn, bn in ((m.l1,"w1","b1"),(m.l2,"w2","b2"),(m.l3,"w3","b3"),(m.l4,"w4","b4")):
            mod.weight.copy_(torch.tensor(w[wn])); mod.bias.copy_(torch.tensor(w[bn]))
        m.wc.copy_(torch.tensor(w["wc"]))
    m.eval()
    return m


def make_episode(m, rng, support=5, horizon=16, composite=None):
    x = rng.normal(0, 0.65, Z)
    states = [x.copy()]
    for _ in range(support + horizon):
        x = step(m, x) if composite is None else step(composite[1], step(composite[0], x))
        states.append(x.copy())
    states = np.asarray(states)
    c = fingerprint(states[:support + 1])
    return states[support - 1], c, states[support:support + horizon]


def pred(model, x, c, horizon):
    import torch
    xx = torch.tensor(x[None, :], dtype=torch.float64)
    cc = torch.tensor(c[None, :], dtype=torch.float64)
    ys = []
    with torch.no_grad():
        for _ in range(horizon):
            y = model(xx, cc)[0].numpy()[0]
            ys.append(y.copy())
            xx = torch.tensor(y[None, :], dtype=torch.float64)
    return np.asarray(ys)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9))


def evaluate(model, seed, mechanisms, episodes, horizon=16, context_mode="normal", composite=False):
    rng = np.random.default_rng(seed ^ 0xA11CE)
    vals = {h: [] for h in [1,2,4,8,16]}
    perm_contexts = []
    samples = []
    for i in range(mechanisms):
        m1 = make_mechanism(seed + i * 0x9E3779B1)
        m2 = make_mechanism(seed + 0x40000000 + i * 0x517CC1)
        pair = (m1, m2) if composite else None
        for _ in range(episodes):
            x, c, truth = make_episode(m1, rng, horizon=horizon, composite=pair)
            samples.append((x, c, truth))
    if context_mode == "permuted":
        cs = [s[1].copy() for s in samples]
        rng.shuffle(cs)
        samples = [(s[0], cs[i], s[2]) for i,s in enumerate(samples)]
    elif context_mode == "random":
        samples = [(s[0], rng.normal(size=10), s[2]) for s in samples]
    for x, c, truth in samples:
        yp = pred(model, x, c, horizon)
        for h in vals:
            vals[h].append(rel(yp[h-1], truth[h-1]))
    return {str(h): float(np.median(np.log(np.maximum(v, 1e-12)))) for h,v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", action="append", required=True)
    ap.add_argument("--seed", type=lambda s:int(s,0), default=0xB900)
    ap.add_argument("--mechanisms", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = {"protocol":"SA-1", "seed":args.seed, "mechanisms":args.mechanisms, "episodes":args.episodes, "models":{}}
    for path in args.weights:
        model = load_model(path)
        name = os.path.basename(path)
        report["models"][name] = {
            "new_state": evaluate(model, args.seed, args.mechanisms, args.episodes),
            "context_permuted": evaluate(model, args.seed, args.mechanisms, args.episodes, context_mode="permuted"),
            "random_context": evaluate(model, args.seed, args.mechanisms, args.episodes, context_mode="random"),
            "new_composition": evaluate(model, args.seed ^ 0xCC33, args.mechanisms, args.episodes, composite=True),
        }
    text = json.dumps(report, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(text)

if __name__ == "__main__":
    main()
