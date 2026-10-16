"""Textos descriptivos de la transformación (es/en/fr), deterministas.
Plantillas 0-1: entrenamiento de P; 2-3: sólo evaluación (redacción nunca vista).
Uso: make_texts.py pool OUT.tsv   |   make_texts.py test CTX.csv OUT.tsv"""
import sys, random, csv
FAM = {
 "es": ["traslación", "rotación", "reflexión", "escalado", "cizalla", "transformación afín", "rotación con escalado", "deformación radial"],
 "en": ["translation", "rotation", "reflection", "scaling", "shear", "affine map", "rotation with scaling", "radial warp"],
 "fr": ["translation", "rotation", "réflexion", "mise à l'échelle", "cisaillement", "application affine", "rotation avec mise à l'échelle", "déformation radiale"],
}
ART = {
 "es": ["una", "una", "una", "un", "una", "una", "una", "una"],
 "en": ["a", "a", "a", "a", "a", "an", "a", "a"],
 "fr": ["une", "une", "une", "une", "un", "une", "une", "une"],
}
T = {
 "es": ["Aplica {art} {f} con parámetros {a} y {b}.", "Transformación: {f}; primer parámetro {a}, segundo parámetro {b}.",
        "A cada punto de la figura se le aplica {art} {f} cuyos parámetros valen {a} y {b}.", "Los puntos sufren {art} {f} (parámetro uno = {a}, parámetro dos = {b}).",
        "Se transforma la figura mediante {art} {f}; sus dos parámetros son {a} y {b}.", "Usa {art} {f}. Valores: {a} para el primero y {b} para el segundo."],
 "en": ["Apply {art} {f} with parameters {a} and {b}.", "Transformation: {f}; first parameter {a}, second parameter {b}.",
        "Every point of the shape undergoes {art} {f} whose parameters are {a} and {b}.", "The points are moved by {art} {f} (parameter one = {a}, parameter two = {b}).",
        "The shape is transformed by {art} {f}; its two parameters are {a} and {b}.", "Use {art} {f}. Values: {a} for the first one and {b} for the second one."],
 "fr": ["Applique {art} {f} avec les paramètres {a} et {b}.", "Transformation : {f} ; premier paramètre {a}, second paramètre {b}.",
        "Chaque point de la figure subit {art} {f} dont les paramètres valent {a} et {b}.", "Les points subissent {art} {f} (paramètre un = {a}, paramètre deux = {b}).",
        "La figure est transformée par {art} {f} ; ses deux paramètres sont {a} et {b}.", "Utilise {art} {f}. Valeurs : {a} pour le premier et {b} pour le second."],
}
def num(v, lang):
    s = f"{v:.2f}"
    return s.replace(".", ",") if lang in ("es", "fr") else s
def texts(fam, p1, p2, tmpls):
    for lang in ("es", "en", "fr"):
        for k in tmpls:
            yield lang, k, T[lang][k].format(art=ART[lang][fam], f=FAM[lang][fam], a=num(p1, lang), b=num(p2, lang))
if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[-1]
    rows = []
    if mode == "pool":                      # contextos de una semilla fija ajena a DEV/CONFIRM, distribución TRAIN (sin el hueco)
        rng = random.Random(0xC0DE)
        for fam in range(8):
            for i in range(16):
                while True:
                    p1 = rng.uniform(-1, 1)
                    if p1 <= 0.10 or p1 >= 0.45: break
                p2 = rng.uniform(-1, 1)
                for lang, k, t in texts(fam, p1, p2, (0, 1)):
                    rows.append((f"pool{fam}_{i}", fam, p1, p2, lang, k, t))
    elif mode == "pool2":                   # D2: mismos 128 contextos que el pool de Fase 0, plantillas de entrenamiento 0,1,4,5
        rng = random.Random(0xC0DE)
        for fam in range(8):
            for i in range(16):
                while True:
                    p1 = rng.uniform(-1, 1)
                    if p1 <= 0.10 or p1 >= 0.45: break
                p2 = rng.uniform(-1, 1)
                for lang, k, t in texts(fam, p1, p2, (0, 1, 4, 5)):
                    rows.append((f"pool{fam}_{i}", fam, p1, p2, lang, k, t))
    else:
        for r in csv.DictReader(open(sys.argv[2])):
            fam, p1, p2 = int(r["fam"]), float(r["p1"]), float(r["p2"])
            for lang, k, t in texts(fam, p1, p2, (2, 3)):
                rows.append((r["idx"], fam, p1, p2, lang, k, t))
    with open(out, "w") as f:
        for r in rows: f.write("\t".join(map(str, r)) + "\n")
    print(len(rows), "textos ->", out)
