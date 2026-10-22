import math, itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from numnorm import render, parse_all, parse2, FMT_NAMES
TOL = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005]
bad = 0; worst = [0.0] * 5
for lang in ("es", "en", "fr"):
    for fmt in range(5):
        for n in range(-100, 101):
            v = n / 100; s = render(v, lang, fmt); r = parse_all(s)
            tol = 0.0004 if fmt == 3 else 1e-4
            if len(r) != 1 or abs(r[0] - v) > tol:
                bad += 1
                if bad < 15: print("FALLO", lang, FMT_NAMES[fmt], v, repr(s), r)
            else: worst[fmt] = max(worst[fmt], abs(r[0] - v))
T = {"es": "Transforma la figura con una rotación: el primer valor es {a} y el segundo es {b}.", "en": "Requested: a shear with values {a} and {b}.", "fr": "Demandé : une mise à l'échelle de valeurs {a} et {b}."}
for lang in T:
    for fa, fb in itertools.product(range(5), range(5)):
        for a, b in ((-0.95, 0.83), (0.21, -0.71), (0.0, 1.0), (0.5, -0.25)):
            s = T[lang].format(a=render(a, lang, fa), b=render(b, lang, fb)); r = parse2(s)
            if r is None or abs(r[0] - a) > 4e-4 or abs(r[1] - b) > 4e-4:
                bad += 1
                if bad < 30: print("FALLO frase", lang, repr(s), r)
for s, e in (("un cuarto de vuelta", math.pi / 2), ("a quarter turn", None), ("trois dixièmes", 0.3), ("zero point three", 0.3), ("30 grados", math.pi / 6), ("quatre-vingt-dix-sept centièmes", 0.97), ("soixante et onze centièmes", 0.71)):
    r = parse_all(s); print("ejemplo", repr(s), r, "esperado", e)
# compatibilidad: en las plantillas 0–5 con dígitos el parser nuevo coincide con el de D3 (sólo regex de dígitos)
import random, re
from make_texts import texts
NUM = re.compile(r"[-−]?\d+(?:[.,]\d+)?")
rng = random.Random(7)
for _ in range(300):
    fam, a, b = rng.randrange(8), rng.uniform(-1, 1), rng.uniform(-1, 1)
    for lang in ("es", "en", "fr"):
        for k in range(6):
            t = list(texts(fam, a, b, (k,)))[[ "es", "en", "fr"].index(lang)][2]
            old = [float(x.replace(",", ".")) for x in NUM.findall(t)][:2]
            if parse2(t) != (old[0], old[1]): bad += 1; print("FALLO compat", repr(t), parse2(t), old)
print("fallos", bad, "error máx por formato", [f"{w:.2e}" for w in worst])
sys.exit(1 if bad else 0)
