#!/usr/bin/env python3
import csv, json, sys, statistics as st
d = sys.argv[1]
rows = list(csv.DictReader(open(f"{d}/metrics.csv")))
mans = [json.loads(l) for l in open(f"{d}/encoder_manifest.jsonl")]
keys = sorted({(r["exp"], r["encoder"], r["method"]) for r in rows})
for k in keys:
    accs = [st.mean(float(r["acc"]) for r in rows if (r["exp"], r["encoder"], r["method"]) == k and r["lang"] == l) for l in ["es", "en", "fr", "ja"]]
    print(f"{k[0]} {k[1]:<15} {k[2]:<24} es/en/fr/ja = " + " / ".join(f"{a:.3f}" for a in accs))
for k in ["gemma_E3", "qwen_aligned", "random_aligned", "gemma_aligned"]:
    print(f"pooled {k}: {st.mean(m['pooled'][k] for m in mans):.3f}")
print("E43 PASS", sum(m["e43_pass"] for m in mans), "/", len(mans), " wins dist", [m["wins"] for m in mans])
print("E44 PASS", sum(m["e44_pass"] for m in mans), "/", len(mans))
print("leakage total", sum(m["leakage_novel_words_in_training"] for m in mans), " fresh ok", sum(m["fresh_process"]["ok"] for m in mans), " queries", sum(m["queries"] for m in mans))
for t in ["gemma", "random"]:
    s = [m[f"stats_{t}"] for m in mans]
    print(f"encoder {t}: words={st.mean(x['n_words'] for x in s):.1f} loss {st.mean(x['loss0'] for x in s):.2f}->{st.mean(x['loss_final'] for x in s):.2e} train_concept_acc={st.mean(x['train_concept_acc'] for x in s):.3f} transition_form_acc={st.mean(x['train_transition_form_acc'] for x in s):.3f}")
print("teacher LOO gemma", round(st.mean((m["teacher_loo_g"]["0"] + m["teacher_loo_g"]["1"]) / 2 for m in mans), 3), "trusted any:", sum(bool(m["trusted_g"]) for m in mans))
