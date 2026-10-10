"""Textos para tfl-gen-v2: 1000 estados (a,b,c)∈Z10³ × 3 plantillas (paráfrasis)."""
import json, itertools
W = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
T = [
    "La caja roja tiene {a} canicas, la azul tiene {b} y la verde tiene {c}.",
    "Hay {b} canicas en la caja azul, {c} en la verde y {a} en la roja.",
    "Inventario: roja = {a}, azul = {b}, verde = {c} canicas.",
]
def text(t, a, b, c):
    return T[t].format(a=W[a], b=W[b], c=W[c])
if __name__ == "__main__":
    out = [text(t, a, b, c) for t in range(3) for a, b, c in itertools.product(range(10), repeat=3)]
    json.dump(out, open("/workspace/ngm-tokenfree-cache/texts_v2.json", "w"), ensure_ascii=False)
    print(len(out), out[0], out[1999])
