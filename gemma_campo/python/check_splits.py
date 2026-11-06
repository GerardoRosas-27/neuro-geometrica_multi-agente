"""Contaminación: ningún hash de conversación de un shard de train puede estar en val/test held-out ni repetirse entre shards."""
import json, sys
held = json.load(open(sys.argv[1]))["split_conv_hashes"]
ho = set(held["val"]) | set(held["test"])
assert not set(held["val"]) & set(held["test"]), "val ∩ test != ∅"
seen = set()
for p in sys.argv[2:]:
    tr = json.load(open(p))["split_conv_hashes"]["train"]
    bad = ho & set(tr)
    assert not bad, f"CONTAMINACIÓN {p}: {len(bad)} conv held-out en train"
    rep = seen & set(tr)
    assert not rep, f"{p}: {len(rep)} conv repetidas de shards anteriores"
    seen |= set(tr)
print(f"contaminacion OK: {len(sys.argv) - 2} shards, {len(seen)} conv train, {len(ho)} held-out")
