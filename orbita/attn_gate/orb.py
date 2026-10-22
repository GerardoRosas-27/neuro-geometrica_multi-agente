"""ÓRBITA-v1 congelado en PyTorch (pesos JSON publicados), re-identificación OBS idéntica a fam::reid (Rust) y corrector de atención."""
import json, math, os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_orbita import Orbita
torch.set_default_dtype(torch.float64)
GRID = [(0.0, 0.0)] + [(-1 + 0.25 * i, -1 + 0.25 * j) for i in range(9) for j in range(9)]   # p̂ = (0,0) (ctx_none) ∪ malla 9×9
def load_orbita(path):
    w = json.load(open(path)); m = Orbita(variant=w.get("variant", "full"))
    with torch.no_grad():
        for mod, wn, bn in ((m.l1, "w1", "b1"), (m.l2, "w2", "b2"), (m.l3, "w3", "b3"), (m.l4, "w4", "b4")):
            mod.weight.copy_(torch.tensor(w[wn])); mod.bias.copy_(torch.tensor(w[bn]))
        m.wc.copy_(torch.tensor(w["wc"]))
    for p in m.parameters(): p.requires_grad_(False)
    return m.eval()
def ctx(g, p): c = torch.zeros(10); c[g] = 1; c[8], c[9] = p; return c
CAND = torch.stack([ctx(g, p) for g in range(8) for p in GRID]); CFAM = torch.tensor([g for g in range(8) for _ in GRID])
def rel(a, b):
    r = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-9); return torch.where(torch.isfinite(r), r, torch.full_like(r, 1e9))
def step(m, x, c, cp=None):
    """Un paso de ÓRBITA (= Orbita.forward()[0]); cp = m.ctx(c) precalculado."""
    if cp is None: cp = m.ctx(c)
    q = x.view(x.shape[0], -1, 2); u = m.correct(cp, m.core(cp, m.encode(q, c)))
    return u[..., :2].reshape(x.shape)
def rep(cp, n): return tuple(p.repeat_interleave(n, 0) for p in cp)
@torch.no_grad()
def reid(m, T, k, excl=None, chunk=8):
    """T [N,17,12] → (c* [N,10], s_known [N] = Σ_t min(rel,10)/k en free-run desde x0, g* [N]). excl [N]: familia excluida (sólo entrenamiento)."""
    N, NC = T.shape[0], CAND.shape[0]; CP = m.ctx(CAND); S = torch.zeros(N, NC)
    for i0 in range(0, N, chunk):
        ii = slice(i0, min(N, i0 + chunk)); n = ii.stop - ii.start
        x = T[ii, 0].repeat_interleave(NC, 0); c = CAND.repeat(n, 1); cp = tuple(p.repeat(n, *([1] * (p.dim() - 1))) for p in CP)
        s = torch.zeros(n * NC)
        for t in range(1, k + 1):
            x = step(m, x, c, cp); fin = torch.isfinite(x).all(1)
            s = s + torch.where(fin, rel(x, T[ii, t].repeat_interleave(NC, 0)).clamp(max=10.0), torch.full_like(s, 10.0))
        S[ii] = s.view(n, NC)
    if excl is not None: S = S + 1e18 * (CFAM[None, :] == excl[:, None])
    b = S.argmin(1)
    return CAND[b], S[torch.arange(N), b] / k, CFAM[b]

class Corrector(torch.nn.Module):
    """Corrector residual de atención (estilo TF-PE). Consultas = 6 puntos del estado actual [x_j, f(x)_j]; memoria = transiciones observadas
    del episodio, un token por (t, punto i): [y_{t-1,i}, f(y_{t-1})_i, r_{t,i}], r = y_t − f(y_{t-1}) (error de predicción de ÓRBITA).
    Logits = q·k/√dh − softplus(w_h)·‖x_j − y_{t-1,i}‖² (sesgo de distancia). Salida: x_{t+1} = f(x) + Δ."""
    def __init__(s, d=16, heads=2, ff=48):
        super().__init__(); s.d, s.h = d, heads
        s.qin = torch.nn.Linear(4, d); s.kin = torch.nn.Linear(6, d); s.lnq = torch.nn.LayerNorm(d); s.lnk = torch.nn.LayerNorm(d)
        s.wq = torch.nn.Linear(d, d); s.wk = torch.nn.Linear(d, d); s.wv = torch.nn.Linear(d, d); s.wo = torch.nn.Linear(d, d)
        s.rv = torch.nn.Linear(2, d, bias=False)
        s.dw = torch.nn.Parameter(torch.full((heads,), 1.0))
        s.ln2 = torch.nn.LayerNorm(d); s.f1 = torch.nn.Linear(d, ff); s.f2 = torch.nn.Linear(ff, d); s.out = torch.nn.Linear(d, 2)
        with torch.no_grad(): s.out.weight.mul_(0.1); s.out.bias.zero_()
    def n_params(s): return sum(p.numel() for p in s.parameters())
    def memory(s, Xp, Fp, Y):
        B, k = Xp.shape[:2]; xp = Xp.reshape(B, k * 6, 2); fp = Fp.reshape(B, k * 6, 2); r = (Y - Fp).reshape(B, k * 6, 2)
        return xp, s.lnk(s.kin(torch.cat([xp, fp, r], -1))), r
    def forward(s, x, f, mem):
        xp, km, r = mem; B = x.shape[0]; q = x.view(B, 6, 2); fq = f.view(B, 6, 2); dh = s.d // s.h
        hq = s.qin(torch.cat([q, fq], -1)); a = s.lnq(hq)
        sp = lambda t: t.view(B, -1, s.h, dh).transpose(1, 2)
        Q, K, V = sp(s.wq(a)), sp(s.wk(km)), sp(s.wv(km) + s.rv(r))
        d2 = ((q[:, :, None, :] - xp[:, None, :, :]) ** 2).sum(-1)
        lg = Q @ K.transpose(-1, -2) / math.sqrt(dh) - F.softplus(s.dw)[None, :, None, None] * d2[:, None]
        h = hq + s.wo((torch.softmax(lg, -1) @ V).transpose(1, 2).reshape(B, 6, s.d))
        h = h + s.f2(F.gelu(s.f1(s.ln2(h))))
        return (fq + s.out(h)).reshape(B, -1)
def episode_memory(m, C, T, c, cp, k):
    N = T.shape[0]
    with torch.no_grad():
        Fp = step(m, T[:, :k].reshape(N * k, 12), c.repeat_interleave(k, 0), rep(cp, k)).view(N, k, 12)
    return C.memory(T[:, :k], Fp, T[:, 1:k + 1])
