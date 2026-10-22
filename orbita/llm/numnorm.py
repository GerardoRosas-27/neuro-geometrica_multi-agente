"""Normalización numérica multilingüe (es/en/fr) para leer p del texto: dígitos, palabras, fracciones y unidades
(radianes por defecto; grados ×π/180; vueltas ×2π). Incluye el generador de esas formas (para textos de robustez)."""
import re, math
UNITS_ES = "cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciséis diecisiete dieciocho diecinueve".split()
TENS_ES = {20: "veinte", 30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta", 70: "setenta", 80: "ochenta", 90: "noventa"}
VEINTI = "veinte veintiuno veintidós veintitrés veinticuatro veinticinco veintiséis veintisiete veintiocho veintinueve".split()
UNITS_EN = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS_EN = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
UNITS_FR = "zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize dix-sept dix-huit dix-neuf".split()
TENS_FR = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante"}

def int_words(n, lang):
    assert 0 <= n <= 99
    if lang == "es":
        if n < 20: return UNITS_ES[n]
        if n < 30: return VEINTI[n - 20]
        t, u = divmod(n, 10); return TENS_ES[t * 10] + ("" if u == 0 else " y " + UNITS_ES[u])
    if lang == "en":
        if n < 20: return UNITS_EN[n]
        t, u = divmod(n, 10); return TENS_EN[t * 10] + ("" if u == 0 else "-" + UNITS_EN[u])
    if n < 17: return UNITS_FR[n]
    if n < 20: return UNITS_FR[n]
    if n < 70:
        t, u = divmod(n, 10); return TENS_FR[t * 10] + ("" if u == 0 else (" et un" if u == 1 else "-" + UNITS_FR[u]))
    if n < 80: return "soixante" + (" et onze" if n == 71 else "-" + UNITS_FR[n - 60])
    if n == 80: return "quatre-vingts"
    return "quatre-vingt-" + UNITS_FR[n - 80]

NEG = {"es": "menos", "en": "minus", "fr": "moins"}
DEC = {"es": "coma", "en": "point", "fr": "virgule"}
def words_decimal(v, lang):
    n = round(abs(v) * 100); i, d = divmod(n, 100); s = int_words(i, lang) + " " + DEC[lang] + " "
    if lang == "en": s += UNITS_EN[d // 10] + " " + UNITS_EN[d % 10]
    else: s += (int_words(0, lang) + " " + int_words(d, lang)) if 0 < d < 10 else int_words(d, lang)
    return (NEG[lang] + " " if v < 0 and n else "") + s
FR_DEN = {"es": {2: ("medio", "medios"), 4: ("cuarto", "cuartos"), 10: ("décimo", "décimos"), 100: ("centésimo", "centésimos")},
          "en": {2: ("half", "halves"), 4: ("quarter", "quarters"), 10: ("tenth", "tenths"), 100: ("hundredth", "hundredths")},
          "fr": {2: ("demi", "demis"), 4: ("quart", "quarts"), 10: ("dixième", "dixièmes"), 100: ("centième", "centièmes")}}
def fraction(v, lang):
    n = round(abs(v) * 100)
    if n % 100 == 0: return words_decimal(v, lang)
    for den in (2, 4, 10, 100):
        if (n * den) % 100 == 0:
            k = n * den // 100; sg, pl = FR_DEN[lang][den]; s = int_words(k, lang) + " " + (sg if k == 1 else pl); break
    return (NEG[lang] + " " if v < 0 else "") + s
def loc(x, lang, nd): s = f"{x:.{nd}f}"; return s.replace(".", ",") if lang in ("es", "fr") else s
UNIT = {"es": ("grados", "vueltas", "radianes"), "en": ("degrees", "turns", "radians"), "fr": ("degrés", "tours", "radians")}
def render(v, lang, fmt):
    v = round(v, 2)
    if fmt == 0: return words_decimal(v, lang)
    if fmt == 1: return fraction(v, lang)
    if fmt == 2: return loc(v * 180 / math.pi, lang, 2) + " " + UNIT[lang][0]
    if fmt == 3: return loc(v / (2 * math.pi), lang, 4) + " " + UNIT[lang][1]
    return loc(v, lang, 2) + " " + UNIT[lang][2]
FMT_NAMES = ["palabras", "fraccion", "grados", "vueltas", "radianes"]

# ---------------- parser ----------------
W = {}
for i, w in enumerate(UNITS_ES): W[w] = i
for i, w in enumerate(VEINTI): W[w] = 20 + i
for k, w in TENS_ES.items(): W[w] = k
for i, w in enumerate(UNITS_EN): W[w] = i
for k, w in TENS_EN.items(): W[w] = k
for i, w in enumerate(UNITS_FR[:17]): W[w] = i
for k, w in TENS_FR.items(): W[w] = k
W.update({"una": 1, "une": 1, "vingts": 20, "veintiun": 21, "dieciseis": 16, "veintidos": 22, "veintitres": 23, "veintiseis": 26, "zero": 0, "zéro": 0, "a": 1, "an": 1})
ZERO = {"cero", "zero", "zéro"}; ARTICLE = {"un", "una", "une", "one", "a", "an"}
NEGW = {"menos", "minus", "moins", "negativo", "negative", "négatif"}; DECW = {"coma", "punto", "point", "virgule"}; CONN = {"y", "et", "and"}
DEN = {}
for lang in FR_DEN:
    for d, (a, b) in FR_DEN[lang].items(): DEN[a] = d; DEN[b] = d
DEN.update({"media": 2, "medias": 2, "décima": 10, "décimas": 10, "centésima": 100, "centésimas": 100, "tercio": 3, "tercios": 3, "third": 3, "thirds": 3, "tiers": 3})
UNITW = {"grado": math.pi / 180, "grados": math.pi / 180, "degree": math.pi / 180, "degrees": math.pi / 180, "degré": math.pi / 180, "degrés": math.pi / 180, "°": math.pi / 180,
         "vuelta": 2 * math.pi, "vueltas": 2 * math.pi, "turn": 2 * math.pi, "turns": 2 * math.pi, "tour": 2 * math.pi, "tours": 2 * math.pi,
         "radian": 1.0, "radianes": 1.0, "radians": 1.0, "rad": 1.0}
SKIP_BEFORE_UNIT = {"de", "of", "d", "du"}
NUMRE = re.compile(r"[-−]?\d+(?:[.,]\d+)?")
def tokenize(t):
    t = t.lower().replace("−", "-").replace("’", "'")
    out = []
    for m in re.finditer(r"[-]?\d+(?:[.,]\d+)?|°|[a-zàâçéèêëîïôûùüÿñæœáíóú]+|-", t):
        g = m.group(0); out.append(g)
    return [w for w in out if w not in ("-",)] if False else out
def _int(tok, i):
    """entero 0–99 en palabras desde tok[i]; devuelve (valor, j) o None"""
    if i >= len(tok) or tok[i] not in W: return None
    total, j, prev = 0, i, None
    while j < len(tok):
        w = tok[j]
        if w in CONN and prev is not None and total >= 20 and total % 10 == 0 and j + 1 < len(tok) and tok[j + 1] in W and W[tok[j + 1]] < 10 or (w in CONN and total in (20, 30, 40, 50, 60) and j + 1 < len(tok) and tok[j + 1] in ("un", "onze")):
            j += 1; continue
        if w == "-": j += 1; continue
        if w not in W: break
        v = W[w]
        if w in ("vingt", "vingts") and prev == "quatre": total += 76; prev = w; j += 1; continue
        if prev is not None:
            pv = W[prev]
            if not ((pv >= 20 and pv % 10 == 0 and v < 20 and (v < 10 or pv in (60, 80) or prev in ("vingt", "vingts"))) or (prev in ("vingt", "vingts") and v < 20) or (prev == "dix" and v in (7, 8, 9) and total % 10 == 0)): break
        total += v; prev = w; j += 1
        if total % 10 != 0 or total < 20 or (total >= 70 and False): pass
    return total, j
def _value(tok, i):
    """cantidad desde tok[i]: (valor_en_radianes, j) o None"""
    sign = 1.0; j = i
    if j < len(tok) and tok[j] in NEGW: sign = -1.0; j += 1
    if j < len(tok) and NUMRE.fullmatch(tok[j]):
        v = float(tok[j].replace(",", ".")); j += 1
        if v < 0: sign, v = -sign, -v
    else:
        r = _int(tok, j)
        if r is None: return None
        n, k = r; first = tok[j]
        if k < len(tok) and tok[k] in DECW:
            dec = tok[k + 1:]; m = k + 1
            if m < len(tok) and tok[m] in ("zero", "cero", "zéro") and m + 1 < len(tok) and tok[m + 1] in W and W[tok[m + 1]] < 10 and not (m + 2 < len(tok) and tok[m + 2] in W and W[tok[m + 2]] < 10 and tok[m] == "zero" and False):
                if tok[m] == "zero":   # inglés: dígito a dígito
                    ds = ""
                    while m < len(tok) and tok[m] in W and W[tok[m]] < 10 and tok[m] not in ARTICLE - {"one"}: ds += str(W[tok[m]]); m += 1
                else:
                    ds = "0" + str(W[tok[m + 1]]); m += 2
            elif m < len(tok) and tok[m] in UNITS_EN and W[tok[m]] < 10 and m + 1 < len(tok) and tok[m + 1] in UNITS_EN and W[tok[m + 1]] < 10:
                ds = ""
                while m < len(tok) and tok[m] in UNITS_EN and W[tok[m]] < 10: ds += str(W[tok[m]]); m += 1
            else:
                r2 = _int(tok, m)
                if r2 is None: return None
                ds, m = str(r2[0]), r2[1]
            v = n + float("0." + ds); j = m
        elif k < len(tok) and tok[k] in DEN:
            v = n / DEN[tok[k]]; j = k + 1
        else:
            m2 = k + 1 if (k < len(tok) and tok[k] in SKIP_BEFORE_UNIT) else k
            # entero en palabras suelto ("uno", "one", "un", "dos"): sólo cuenta si es cero o lleva unidad (evita artículos y ordinales)
            if first not in ZERO and not (m2 < len(tok) and tok[m2] in UNITW): return None
            v = float(n); j = k
    m = j
    if m < len(tok) and tok[m] in SKIP_BEFORE_UNIT and m + 1 < len(tok) and tok[m + 1] in UNITW: m += 1
    if m < len(tok) and tok[m] in UNITW: v *= UNITW[tok[m]]; j = m + 1
    return sign * v, j
def parse_all(text):
    tok = tokenize(text); out = []; i = 0
    while i < len(tok):
        r = _value(tok, i)
        if r is None: i += 1; continue
        out.append(r[0]); i = max(r[1], i + 1)
    return out
def parse2(text):
    v = parse_all(text)
    return (max(-1.0, min(1.0, v[0])), max(-1.0, min(1.0, v[1]))) if len(v) >= 2 else None
