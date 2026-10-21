import sys, numpy as np, torch
sys.path.insert(0, "/workspace/ngm-cto/orbita/cto")
import cto
from cto import *
ph, s = sys.argv[1], sys.argv[2]; k = 2
orb = Orb(wpath(ph, s)); nets = {a: load_net(ph, s, a)[0] for a in ("CT", "SELF")}
Hs = load_set(ph, "H"); Hc = load_ctx(ph, s, "H")
idxs = [i for i, _, _ in Hc]; fam = np.array([Hs[i][0] for i in idxs]); kn = fam < 8
TR = torch.tensor(np.stack([traj(Hs[i][0], Hs[i][1], Hs[i][2], 16) for i in idxs])); obs = TR[:, :k+1]
Cg = torch.tensor(np.stack([c for _, _, c in Hc])); Ct = torch.tensor(np.stack([context_of(Hs[i][0], Hs[i][1]) if Hs[i][0] < 8 else c for i, _, c in Hc]))
def acc(pr): e = rel_err(pr[:, 16-k-1].numpy(), TR[:, 16].numpy()); return f"known {np.mean(e[kn]<.05):.3f} new {np.mean(e[~kn]<.05):.3f} medre_known {np.median(e[kn]):.3f}"
# stream bank reconstruction up to n
Ss = load_set(ph, "S"); Sc = load_ctx(ph, s, "S"); tau = taus()
def sbank(N):
    K=[];V=[];Mm=[];G=[];F=[]
    for n in range(N):
        i, cond, c = Sc[n]; f, p, x0 = Ss[i]; kk = int(cond[1:])
        o = torch.tensor(traj(f, p, x0, kk))[None]; cpp = orb.ctx(torch.tensor(c)[None])
        with torch.no_grad(): kin, vin, sig, surp = episode_entries(orb, o, cpp)
        if float(surp[0]) > tau[kk]:
            ki = full_keys(kin, sig)[0]; pad = 48 - ki.shape[0]
            K.append(torch.nn.functional.pad(ki, (0,0,0,pad))); V.append(torch.nn.functional.pad(vin[0], (0,0,0,pad)))
            m = torch.zeros(48, dtype=torch.bool); m[:ki.shape[0]] = True; Mm.append(m); G.append(sig[0]); F.append(f)
    return dict(keys=torch.stack(K), vals=torch.stack(V), mask=torch.stack(Mm), sig=torch.stack(G)), F
with torch.no_grad():
    tb = make_bank(orb, gen_episodes(np.random.default_rng([int(s,16), 0xBA]), BANK_TRAIN))
    b150, F150 = sbank(150); b2400, F2400 = sbank(2400)
    for cname, C in (("gemma", Cg), ("true", Ct)):
        cp = orb.ctx(C)
        print(cname, "ORB", acc(rollout(None, orb, obs, cp, None, 16, "orb")))
        hk, hv, hm = hist_mem(orb, obs, cp)
        print(cname, "SELF", acc(rollout(nets["SELF"], orb, obs, cp, dict(hk=hk, hv=hv, hmask=hm), 16, "net")))
        for bn, b in (("none", None), ("trainbank", tb), ("stream150", b150), ("stream2400", b2400)):
            mem = build_mem(None, orb, obs, cp, b)
            print(cname, "CT", bn, acc(rollout(nets["CT"], orb, obs, cp, mem, 16, "net")))
    # delta magnitude
    cp = orb.ctx(Cg)
    for bn, b in (("none", None), ("stream150", b150)):
        mem = build_mem(None, orb, obs, cp, b); x = obs[:, -1]; z = orb.encode(x, cp); u0 = orb.evolve(z, cp); q = x.view(-1, P, 2)
        qi = torch.cat([z, u0[..., :2]-q, mem["sig"][:, None].expand(-1, P, -1)], -1); D, w = nets["CT"].att(qi, mem["ki"], mem["vi"], mem["mask"])
        nep = 6*k; print(bn, "|D| known", float(D[kn].norm(dim=-1).mean()), "|z|", float(z[kn].norm(dim=-1).mean()), "attn mass on CDT entries", float(w[..., nep:].sum(-1).mean()) if w.shape[-1] > nep else 0)
    print("bank150 fams", np.bincount(F150, minlength=12)); print("sig scale H", mem["sig"].abs().mean(0).numpy().round(3), "bank sig", b150["sig"].abs().mean(0).numpy().round(3))
    # training distribution: ORB vs CT
    rng = np.random.default_rng(999); eps = gen_episodes(rng, 256, (2,))
    tr = torch.tensor(np.stack([traj(e[0], e[1], e[2], 16) for e in eps])); cpt = orb.ctx(torch.tensor(np.stack([e[4] for e in eps]))); ob = tr[:, :3]
    def a2(pr): e = rel_err(pr[:, 13].numpy(), tr[:, 16].numpy()); return f"acc16 {np.mean(e<.05):.3f} med {np.median(e):.3f}"
    print("TRAINDIST ORB", a2(rollout(None, orb, ob, cpt, None, 16, "orb")))
    print("TRAINDIST CT trainbank", a2(rollout(nets["CT"], orb, ob, cpt, build_mem(None, orb, ob, cpt, tb), 16, "net")))
    hk, hv, hm = hist_mem(orb, ob, cpt); print("TRAINDIST SELF", a2(rollout(nets["SELF"], orb, ob, cpt, dict(hk=hk, hv=hv, hmask=hm), 16, "net")))
