"""Paso 3b: baselines/controles del campo, mismo layout de filas que <model>/<split>_pred1.f32 de Rust
(fila t = predicción de z_t con z_{<t}). Solo se usan en turnos assistant (t>=1).
  repeat      : z_{t-1} (= turno user actual)
  repeat_asst : z del último assistant anterior (si no hay, z_{t-1})
  mean        : media de z de turnos assistant de train
  ridge       : lineal [ctx turnos] -> z_t, λ en val, entrenado en objetivos assistant
  mlp         : torch, 2x256 GELU, early stopping en val, entrenado en objetivos assistant"""
import argparse, os
import numpy as np
from tc_common import read_z, load_json, save_json, ctx_features, asst_positions


def dataset(z, offs, roles, ctx):
    X, Y = [], []
    for c, r in enumerate(roles):
        Zc = z[offs[c]:offs[c + 1]]
        for t in asst_positions(r):
            X.append(ctx_features(Zc, t, ctx)); Y.append(Zc[t])
    return np.array(X, np.float64), np.array(Y, np.float64)


def predict_rows(z, offs, roles, ctx, fn):
    P = z.copy()
    for c, r in enumerate(roles):
        Zc = z[offs[c]:offs[c + 1]]
        for t in range(1, len(r)):
            P[offs[c] + t] = fn(Zc, t, r)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zdir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=4); ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--mlp-epochs", type=int, default=200); ap.add_argument("--mlp-hidden", type=int, default=256)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    S = {s: (*read_z(a.zdir, s), load_json(os.path.join(a.zdir, f"{s}_roles.json"))) for s in ["train", "val", "test"]}
    Xtr, Ytr = dataset(S["train"][0], S["train"][2], S["train"][3], a.ctx)
    Xva, Yva = dataset(S["val"][0], S["val"][2], S["val"][3], a.ctx)
    mu = Ytr.mean(0); log = {"n_train": len(Xtr), "n_val": len(Xva)}
    # ridge
    Xb = np.c_[Xtr, np.ones(len(Xtr))]; best = (np.inf, None, None)
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        R = lam * len(Xb) * np.eye(Xb.shape[1]); R[-1, -1] = 0
        W = np.linalg.solve(Xb.T @ Xb + R, Xb.T @ Ytr)
        vm = float(((np.c_[Xva, np.ones(len(Xva))] @ W - Yva) ** 2).mean()); log[f"ridge_val_mse_lam{lam:g}"] = vm
        if vm < best[0]: best = (vm, lam, W)
    W = best[2]; log["ridge_lambda"] = best[1]
    # mlp
    import torch
    torch.manual_seed(a.seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], a.mlp_hidden), torch.nn.GELU(), torch.nn.Linear(a.mlp_hidden, a.mlp_hidden),
                              torch.nn.GELU(), torch.nn.Linear(a.mlp_hidden, Ytr.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32)
    xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(Yva, dtype=torch.float32)
    bst, bstate, pat = np.inf, None, 0; g = torch.Generator().manual_seed(a.seed)
    for ep in range(a.mlp_epochs):
        net.train()
        for b in torch.randperm(len(xt), generator=g).split(128):
            opt.zero_grad(); l = ((net(xt[b]) - yt[b]) ** 2).mean(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = float(((net(xv) - yv) ** 2).mean())
        if vl < bst: bst, bstate, pat, log["mlp_best_epoch"] = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            pat += 1
            if pat >= 20: break
    net.load_state_dict(bstate); net.eval(); log["mlp_val_mse"] = bst

    def mlp_fn(Zc, t, r):
        with torch.no_grad(): return net(torch.tensor(ctx_features(Zc, t, a.ctx)[None], dtype=torch.float32))[0].numpy()

    def rep_asst(Zc, t, r):
        prev = [u for u in range(t) if r[u] == "assistant"]
        return Zc[prev[-1]] if prev else Zc[t - 1]
    fns = {"repeat": lambda Zc, t, r: Zc[t - 1], "repeat_asst": rep_asst, "mean": lambda Zc, t, r: mu,
           "ridge": lambda Zc, t, r: np.r_[ctx_features(Zc, t, a.ctx), 1.0] @ W, "mlp": mlp_fn}
    for s in ["val", "test"]:
        z, _, offs, roles = S[s]
        for k, fn in fns.items():
            predict_rows(z, offs, roles, a.ctx, fn).astype("<f4").tofile(os.path.join(a.out, f"{s}_pred_{k}.f32"))
    save_json(os.path.join(a.out, "baselines_log.json"), log); print(log)


if __name__ == "__main__":
    main()
