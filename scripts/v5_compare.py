#!/usr/bin/env python3
"""v4 (E3_v4) vs v5 (P2_*) head-to-head per docs/comparacion_v4_v5_preregistro.md.

Usage: v5_compare.py artifacts/v5/dev [artifacts/v5/confirm ...] [--md out.md]
Reads metrics.csv if present, else the per-seed cache (seeds/*.json), so it
also works on a partially finished phase. Bootstrap: 10 000 resamples, seed 12345.
"""
import csv, glob, io, json, os, random, re, statistics as st, sys

ARMS = ["E3_v4", "P2_gated", "P2_gated_eqc", "P2_mlp", "P2_shuffled"]
HS = [1, 2, 4, 8, 16, 32, 64]
OOD = ["reflect", "compose_rot_scale", "radial_warp", "in_dist(n=128)", "interp_hole(n=128)"]

def rows(d):
    f = os.path.join(d, "metrics.csv")
    if os.path.exists(f):
        return list(csv.reader(open(f)))[1:]
    out = []
    for p in sorted(glob.glob(os.path.join(d, "seeds", "*.json"))):
        for line in json.load(open(p))["rows"]:
            out += list(csv.reader(io.StringIO(line)))
    return out

def num(pat, s, default=float("nan")):
    m = re.search(pat, s)
    return float(m.group(1)) if m else default

def load(d):
    per = {}  # seed -> arm -> dict
    for r in rows(d):
        seed, cond, part, acc, rel, status, notes = r[1], r[2], r[3], float(r[7]), float(r[11]), r[-2], r[-1]
        if cond == "GATE":
            arm = part[2:]
            x = per.setdefault(seed, {}).setdefault(arm, {})
            x["region"] = num(r"region=(\d+)", notes)
            x["pass"] = status == "PASS"
            x["clean"] = "clean=true" in notes
            x["fresh"] = "fresh_ok=true" in notes
            for h in HS:
                x[f"h{h}"] = num(rf"h{h}:acc=([0-9.]+)", notes)
        elif part == "TEST" and cond in ARMS:
            x = per.setdefault(seed, {}).setdefault(cond, {})
            x.update(acc1=acc, rel1=rel, train_s=num(r"train_s=([0-9.]+)", notes),
                     lat=num(r"lat_us=([0-9.]+)", notes))
            for o in OOD:
                x[o] = num(re.escape(o) + r":rel=([0-9.]+)", notes)
    return {s: a for s, a in per.items() if all(k in a and "region" in a[k] for k in ARMS)}

def boot(diffs, n=10000, seed=12345):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return st.mean(diffs), ms[int(0.025 * n)], ms[int(0.975 * n) - 1]

def verdict(lo_hi_pairs):
    pos = any(lo > 0 for lo, _ in lo_hi_pairs)
    neg = all(hi < 0 for _, hi in lo_hi_pairs)
    return pos, neg

def phase(d):
    per = load(d)
    seeds = sorted(per)
    out = {"dir": d, "n": len(seeds), "arms": {}, "paired": {}}
    for a in ARMS:
        vals = [per[s][a] for s in seeds]
        if not vals:
            continue
        m = lambda k: st.mean(v[k] for v in vals)
        out["arms"][a] = {"pass": sum(v["pass"] for v in vals), "clean": all(v["clean"] for v in vals),
                          "fresh": all(v["fresh"] for v in vals),
                          **{k: m(k) for k in ["acc1", "rel1", "region", "train_s", "lat"] + [f"h{h}" for h in HS] + OOD}}
    for a in ["P2_gated", "P2_gated_eqc", "P2_mlp"]:
        if not seeds:
            break
        res = {}
        for k in ["region", "h16", "acc1"]:
            res[k] = boot([per[s][a][k] - per[s]["E3_v4"][k] for s in seeds])
        out["paired"][a] = res
    return out

def table(p):
    L = [f"Seeds completas: {p['n']}", "",
         "| brazo | gate (región≥8) | acc 1 paso | err rel 1 paso | región E33 | " + " | ".join(f"h{h}" for h in HS) +
         " | train s | lat µs | limpio | proc. nuevo |",
         "|---" * (9 + len(HS)) + "|"]
    for a, v in p["arms"].items():
        L.append(f"| {a} | {v['pass']}/{p['n']} | {v['acc1']:.3f} | {v['rel1']:.4f} | {v['region']:.2f} | " +
                 " | ".join(f"{v[f'h{h}']:.3f}" for h in HS) +
                 f" | {v['train_s']:.0f} | {v['lat']:.1f} | {'sí' if v['clean'] else 'NO'} | {'sí' if v['fresh'] else 'NO'} |")
    L += ["", "OOD/composición (error relativo de un paso, media):", "",
          "| brazo | " + " | ".join(OOD) + " |", "|---" * (1 + len(OOD)) + "|"]
    for a, v in p["arms"].items():
        L.append(f"| {a} | " + " | ".join(f"{v[o]:.4f}" for o in OOD) + " |")
    L += ["", "Diferencias pareadas vs `E3_v4` (media [IC95 bootstrap]):", "",
          "| brazo | Δ región | Δ acc h16 | Δ acc 1 paso |", "|---|---|---|---|"]
    for a, r in p["paired"].items():
        L.append(f"| {a} | " + " | ".join(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]" for m, lo, hi in (r["region"], r["h16"], r["acc1"])) + " |")
    return "\n".join(L)

def decide(phases):
    lines = []
    for a in ["P2_gated", "P2_gated_eqc"]:
        if not all(a in p["paired"] for p in phases):
            continue
        win = all((p["paired"][a]["region"][1] > 0 or p["paired"][a]["h16"][1] > 0)
                  and p["paired"][a]["acc1"][0] >= -0.02 for p in phases)
        lose = all(p["paired"][a]["region"][2] < 0 or p["paired"][a]["h16"][2] < 0 for p in phases)
        v = "v5 GANA" if win else ("v4 GANA" if lose else "EMPATE / no concluyente")
        lines.append(f"- `{a}` vs `E3_v4` en {len(phases)} fase(s): **{v}**")
    return "\n".join(lines)

if __name__ == "__main__":
    args = sys.argv[1:]
    md = None
    if "--md" in args:
        i = args.index("--md"); md = args[i + 1]; del args[i:i + 2]
    ps = [phase(d) for d in args]
    text = "\n\n".join(f"## {os.path.basename(p['dir'].rstrip('/')).upper()}\n\n" + table(p) for p in ps)
    text += "\n\n## Veredicto (regla pre-registrada)\n\n" + decide(ps) + "\n"
    print(text)
    if md:
        open(md, "w").write(text)
