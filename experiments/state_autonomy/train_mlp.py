#!/usr/bin/env python3
"""Parameter-matched MLP baseline for SA-1.

The hidden width 96 gives ~2.6k parameters, close to ORBITA-v1 full.
"""
import argparse, json, os, math
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

class MLP(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(18, hidden), nn.Tanh(), nn.Linear(hidden, 8))
    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=-1))


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    a = np.fromfile(os.path.join(d, "train.bin"), dtype="<f8").reshape(-1, 26)
    return meta, torch.from_numpy(a[:, :8].copy()), torch.from_numpy(a[:, 8:18].copy()), torch.from_numpy(a[:, 18:].copy())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--out",required=True); ap.add_argument("--seed",required=True); ap.add_argument("--steps",type=int,default=256000); ap.add_argument("--batch",type=int,default=32); ap.add_argument("--lr",type=float,default=3e-3)
    a=ap.parse_args(); seed=int(a.seed,16) if a.seed.startswith("0x") else int(a.seed); torch.manual_seed(seed)
    meta,x,c,y=load(a.data); m=MLP(); opt=torch.optim.Adam(m.parameters(),lr=a.lr); g=torch.Generator().manual_seed(seed^0x7A)
    for step in range(a.steps):
        t=step/max(a.steps,1); lr=3e-5+0.5*(a.lr-3e-5)*(1+math.cos(math.pi*t)); opt.param_groups[0]["lr"]=lr
        idx=torch.randint(0,x.shape[0],(a.batch,),generator=g); yp=m(x[idx],c[idx]); scale=1/(y[idx].square().sum(1).clamp_min(1e-6)); loss=((yp-y[idx]).square().sum(1)*scale).mean(); opt.zero_grad(); loss.backward(); opt.step()
    torch.save(m.state_dict(),a.out)
    json.dump({"arch":"MLP-SA1","hidden":96,"n_params":sum(p.numel() for p in m.parameters()),"train":vars(a)},open(a.out+".json","w"),indent=2)

if __name__=="__main__": main()
