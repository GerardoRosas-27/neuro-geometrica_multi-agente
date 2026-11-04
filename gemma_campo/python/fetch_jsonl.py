"""Descarga un dataset de diálogos a JSONL ({"messages":[...]} por línea) vía la API pública datasets-server de HF,
sin depender del paquete `datasets`. Varias fuentes: ejecutar varias veces y concatenar (extract.py deduplica por hash)."""
import argparse, json, os, time, urllib.request, urllib.parse

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HuggingFaceTB/everyday-conversations-llama3.1-2k")
ap.add_argument("--config", default="default"); ap.add_argument("--split", default="train_sft")
ap.add_argument("--field", default="messages"); ap.add_argument("--max-rows", type=int, default=10**9)
ap.add_argument("--out", required=True)
a = ap.parse_args()
hdr = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
n, off = 0, 0
with open(a.out, "w") as f:
    while n < a.max_rows:
        q = urllib.parse.urlencode({"dataset": a.dataset, "config": a.config, "split": a.split, "offset": off, "length": 100})
        for attempt in range(5):
            try:
                r = json.load(urllib.request.urlopen(urllib.request.Request(f"https://datasets-server.huggingface.co/rows?{q}", headers=hdr), timeout=60)); break
            except Exception as e:
                if attempt == 4: raise
                time.sleep(2 ** attempt)
        rows = r.get("rows", [])
        if not rows: break
        for row in rows:
            msgs = row["row"].get(a.field)
            if msgs and n < a.max_rows:
                f.write(json.dumps({"messages": [{"role": m["role"], "content": m["content"]} for m in msgs]}, ensure_ascii=False) + "\n"); n += 1
        off += len(rows)
print(f"{n} conversaciones -> {a.out}")
