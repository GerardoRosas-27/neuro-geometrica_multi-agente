#!/usr/bin/env python3
"""Pure-stdlib SVG scientific figure generator for riemann-collapse-v1 (Suites C1-C4)."""
import csv
import json
import math
import pathlib
from xml.sax.saxutils import escape

CRATE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CRATE_DIR.parent
DOCS_DIR = REPO_ROOT / "docs" / "riemann_collapse_v1"
FIG_DIR = DOCS_DIR / "figures"


def svg_header(w: int, h: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'style="background:#ffffff;font-family:Inter,system-ui,-apple-system,sans-serif;">',
        '<defs><style>'
        '.title{font-size:15px;font-weight:700;fill:#0f172a}'
        '.sub{font-size:11.5px;font-weight:600;fill:#334155}'
        '.axis{font-size:10px;fill:#475569}'
        '.legend{font-size:10.5px;fill:#1e293b}'
        '.grid{stroke:#e2e8f0;stroke-width:1}'
        '.frame{stroke:#94a3b8;stroke-width:1.1;fill:none}'
        '</style></defs>',
    ]


def polyline(points: list[tuple[float, float]], stroke: str, sw: float = 1.6, dash: str = "", opacity: float = 1.0) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    op = f' stroke-opacity="{opacity:.2f}"' if opacity < 1.0 else ""
    return f'<polyline fill="none" stroke="{stroke}" stroke-width="{sw}"{extra}{op} points="{pts}"/>'


def map_pt(x, y, x0, y0, pw, ph, xmin, xmax, ymin, ymax):
    px = x0 + (x - xmin) / (xmax - xmin) * pw
    yc = max(ymin, min(ymax, y))
    py = y0 + ph - (yc - ymin) / (ymax - ymin) * ph
    return px, py


def draw_axes(out: list[str], x0: float, y0: float, pw: float, ph: float, xmin: float, xmax: float, ymin: float, ymax: float, xlabel: str, ylabel: str, title: str, nx: int = 5, ny: int = 4):
    out.append(f'<text x="{x0}" y="{y0 - 10}" class="sub">{escape(title)}</text>')
    for i in range(ny + 1):
        frac = i / ny
        yv = ymin + frac * (ymax - ymin)
        py = y0 + ph - frac * ph
        out.append(f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x0 + pw:.1f}" y2="{py:.1f}" class="grid"/>')
        out.append(f'<text x="{x0 - 6:.1f}" y="{py + 3.5:.1f}" text-anchor="end" class="axis">{yv:.2f}</text>')
    for i in range(nx + 1):
        frac = i / nx
        xv = xmin + frac * (xmax - xmin)
        px = x0 + frac * pw
        out.append(f'<line x1="{px:.1f}" y1="{y0:.1f}" x2="{px:.1f}" y2="{y0 + ph:.1f}" class="grid"/>')
        out.append(f'<text x="{px:.1f}" y="{y0 + ph + 14:.1f}" text-anchor="middle" class="axis">{xv:.1f}</text>')
    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{pw:.1f}" height="{ph:.1f}" class="frame"/>')
    out.append(f'<text x="{x0 + pw / 2:.1f}" y="{y0 + ph + 30:.1f}" text-anchor="middle" class="axis">{escape(xlabel)}</text>')
    out.append(f'<text x="{x0 - 44:.1f}" y="{y0 + ph / 2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 {x0 - 44:.1f} {y0 + ph / 2:.1f})">{escape(ylabel)}</text>')


def is_prime(n: int) -> bool:
    return n >= 2 and all(n % d != 0 for d in range(2, int(n ** 0.5) + 1))


def normalize_unit(vals: list[float]) -> list[float]:
    m = max(abs(v) for v in vals) or 1.0
    return [v / m for v in vals]


def fig1_past_future_emergence(trace_dir: pathlib.Path):
    W, H = 940, 630
    out = svg_header(W, H)
    out.append('<text x="24" y="26" class="title">Suite C1 — Emergencia Estricta: Onda Pasada Sola y Onda Futura Sola vs. Colapso en Primos de Riemann</text>')

    rows = list(csv.DictReader((trace_dir / "trace_past_future_collapse.csv").open()))
    xs = [float(r["x"]) for r in rows]
    yp = normalize_unit([float(r["past_alone"]) for r in rows])
    yc = normalize_unit([float(r["collapse_exact"]) for r in rows])
    yoff = normalize_unit([float(r["collapse_off_time"]) for r in rows])
    ynaive = normalize_unit([float(r["naive_bilinear"]) for r in rows])

    # Panel A: Past alone (squares only) vs Collapse exact (all primes!)
    x0, y0, pw, ph = 66, 62, 844, 215
    draw_axes(out, x0, y0, pw, ph, 1.5, 55.0, -0.5, 1.15, "x", "Amplitud norm.", "A. Onda pasada sola (semifrecuencias +γ_n/2: solo cuadrados 4, 9, 16, 25, 49) vs. Colapso Pasado×Futuro en τ = 0 (TODOS los primos)", nx=10, ny=4)
    for p in range(2, 55):
        if is_prime(p):
            px, _ = map_pt(p, 0, x0, y0, pw, ph, 1.5, 55.0, -0.5, 1.15)
            out.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + ph}" stroke="#dc2626" stroke-width="0.8" stroke-dasharray="3,3" stroke-opacity="0.35"/>')
            if p <= 53:
                out.append(f'<text x="{px:.1f}" y="{y0 + 11}" text-anchor="middle" style="font-size:8px;font-weight:700;fill:#dc2626">{p}</text>')
    out.append(polyline([map_pt(x, y, x0, y0, pw, ph, 1.5, 55.0, -0.5, 1.15) for x, y in zip(xs, yp)], "#f59e0b", 1.2, opacity=0.7))
    out.append(polyline([map_pt(x, y, x0, y0, pw, ph, 1.5, 55.0, -0.5, 1.15) for x, y in zip(xs, yc)], "#1d4ed8", 1.65))
    out.append('<rect x="615" y="72" width="286" height="42" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    out.append('<line x1="625" y1="86" x2="645" y2="86" stroke="#1d4ed8" stroke-width="2"/><text x="652" y="90" class="legend">Colapso Pasado×Futuro τ=0 (primos)</text>')
    out.append('<line x1="625" y1="103" x2="645" y2="103" stroke="#f59e0b" stroke-width="1.5"/><text x="652" y="107" class="legend">Onda pasada sola (0 picos en primos)</text>')

    # Panel B: Off-time decoherence & Naive bilinear negative control
    x0b, y0b, pwb, phb = 66, 350, 844, 215
    draw_axes(out, x0b, y0b, pwb, phb, 1.5, 55.0, -1.05, 1.05, "x", "Amplitud norm.", "B. Controles negativos físicos: colapso fuera de tiempo (τ = 0.25) y producto bilineal ingenuo sin emparejamiento modal", nx=10, ny=4)
    out.append(polyline([map_pt(x, y, x0b, y0b, pwb, phb, 1.5, 55.0, -1.05, 1.05) for x, y in zip(xs, ynaive)], "#9333ea", 1.1, opacity=0.55))
    out.append(polyline([map_pt(x, y, x0b, y0b, pwb, phb, 1.5, 55.0, -1.05, 1.05) for x, y in zip(xs, yoff)], "#64748b", 1.35, opacity=0.85))
    out.append('<rect x="615" y="360" width="286" height="42" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    out.append('<line x1="625" y1="374" x2="645" y2="374" stroke="#64748b" stroke-width="1.8"/><text x="652" y="378" class="legend">Colapso fuera de tiempo τ=0.25 (ruido)</text>')
    out.append('<line x1="625" y1="391" x2="645" y2="391" stroke="#9333ea" stroke-width="1.5"/><text x="652" y="395" class="legend">Bilineal ingenuo (términos cruzados)</text>')

    out.append("</svg>")
    (FIG_DIR / "fig1_past_future_emergence.svg").write_text("\n".join(out) + "\n")


def heat_color(val_norm: float) -> str:
    # Diverging dark-navy -> slate -> amber -> bright gold/crimson for positive prime spikes
    v = max(-1.0, min(1.0, val_norm))
    if v < 0:
        t = 1.0 + v
        r = int(15 + 25 * t)
        g = int(23 + 38 * t)
        b = int(62 + 55 * t)
    else:
        t = v ** 0.65
        r = int(40 + 215 * t)
        g = int(61 + 175 * (t * (1.0 - 0.35 * t)))
        b = int(117 * (1.0 - t) + 25 * t)
    return f"rgb({r},{g},{b})"


def fig2_spacetime_and_pulse(trace_dir: pathlib.Path):
    W, H = 940, 640
    out = svg_header(W, H)
    out.append('<text x="24" y="26" class="title">Suites C1 &amp; C3 — Diagrama Espacio-Tiempo del Colapso Pasado–Futuro y Pulso Temporal de Emergencia</text>')

    # Panel A: Spacetime heatmap (tau vs x)
    hrows = list(csv.DictReader((trace_dir / "trace_spacetime_collapse.csv").open()))
    taus = sorted({float(r["tau"]) for r in hrows})
    xs = sorted({float(r["x"]) for r in hrows})
    amps = [float(r["amplitude"]) for r in hrows]
    amax = max(abs(a) for a in amps) or 1.0

    x0, y0, pw, ph = 66, 62, 844, 240
    out.append(f'<text x="{x0}" y="{y0 - 10}" class="sub">A. Mapa espacio-tiempo I(x, τ): la onda pasada (τ &lt; 0) y la futura (τ &gt; 0) colisionan en τ = 0 encendiendo los primos</text>')
    nt, nx = len(taus), len(xs)
    cw = pw / nx
    ch = ph / nt
    for idx, r in enumerate(hrows):
        ti = idx // nx
        xi = idx % nx
        px = x0 + xi * cw
        py = y0 + ph - (ti + 1) * ch
        col = heat_color(float(r["amplitude"]) / amax)
        out.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{cw + 0.4:.2f}" height="{ch + 0.4:.2f}" fill="{col}"/>')

    # Collapse line tau = 0
    py_zero = y0 + ph * 0.5
    out.append(f'<line x1="{x0}" y1="{py_zero:.1f}" x2="{x0 + pw}" y2="{py_zero:.1f}" stroke="#ffffff" stroke-width="1.1" stroke-dasharray="5,4" stroke-opacity="0.8"/>')
    out.append(f'<text x="{x0 + pw - 8}" y="{py_zero - 5:.1f}" text-anchor="end" style="font-size:10px;font-weight:700;fill:#ffffff">τ = 0 (Colapso exacto)</text>')

    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        px = x0 + (p - xs[0]) / (xs[-1] - xs[0]) * pw
        out.append(f'<text x="{px:.1f}" y="{y0 + ph + 14}" text-anchor="middle" style="font-size:9px;font-weight:700;fill:#dc2626">{p}</text>')
    for tv in [-0.12, -0.06, 0.0, 0.06, 0.12]:
        py = y0 + ph - (tv - taus[0]) / (taus[-1] - taus[0]) * ph
        out.append(f'<text x="{x0 - 6}" y="{py + 3.5:.1f}" text-anchor="end" class="axis">{tv:+.2f}</text>')
    out.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" class="frame"/>')
    out.append(f'<text x="{x0 + pw / 2}" y="{y0 + ph + 29}" text-anchor="middle" class="axis">Espacio x (marcadores rojos = números primos)</text>')
    out.append(f'<text x="{x0 - 44}" y="{y0 + ph / 2}" text-anchor="middle" class="axis" transform="rotate(-90 {x0 - 44} {y0 + ph / 2})">Tiempo τ (Pasado &lt; 0 &lt; Futuro)</text>')

    # Panel B: Temporal AUC pulse
    prows = list(csv.DictReader((trace_dir / "trace_tau_pulse.csv").open()))
    ptau = [float(r["tau"]) for r in prows]
    pauc_t = [float(r["transactional_auc"]) for r in prows]
    pauc_l = [float(r["linear_auc"]) for r in prows]
    x0b, y0b, pwb, phb = 66, 376, 844, 210
    draw_axes(out, x0b, y0b, pwb, phb, -0.25, 0.25, 0.35, 1.05, "Desfase temporal τ respecto al punto de colapso", "ROC-AUC primos [2..100]", "B. Vida media temporal del colapso: el patrón nace bruscamente al cruzarse ambas ondas en τ = 0 y se disuelve fuera de fase", nx=10, ny=4)
    out.append(polyline([map_pt(t, a, x0b, y0b, pwb, phb, -0.25, 0.25, 0.35, 1.05) for t, a in zip(ptau, pauc_l)], "#0d9488", 1.5, dash="5,3"))
    out.append(polyline([map_pt(t, a, x0b, y0b, pwb, phb, -0.25, 0.25, 0.35, 1.05) for t, a in zip(ptau, pauc_t)], "#1d4ed8", 2.0))
    out.append('<rect x="645" y="386" width="256" height="42" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    out.append('<line x1="655" y1="400" x2="675" y2="400" stroke="#1d4ed8" stroke-width="2"/><text x="682" y="404" class="legend">Colapso transaccional (semifrec.)</text>')
    out.append('<line x1="655" y1="417" x2="675" y2="417" stroke="#0d9488" stroke-width="1.7" stroke-dasharray="5,3"/><text x="682" y="421" class="legend">Superposición lineal dispersiva</text>')

    out.append("</svg>")
    (FIG_DIR / "fig2_spacetime_collapse_and_pulse.svg").write_text("\n".join(out) + "\n")


def fig3_search_and_rayleigh(trace_dir: pathlib.Path, summary: dict):
    W, H = 940, 440
    out = svg_header(W, H)
    out.append('<text x="24" y="26" class="title">Suites C2 &amp; C3 — Búsqueda Automática de la Configuración Exacta y Transición de Resolución de Rayleigh</text>')

    # Left: Convergence trajectory of optimizer on seed 0xB800
    rows = list(csv.DictReader((trace_dir / "trace_config_search.csv").open()))
    evs = [float(r["eval"]) for r in rows]
    objs = [float(r["full_obj"]) for r in rows]
    ymax = max(objs) * 1.12
    ymin = min(0.0, min(objs) * 1.1)
    x0, y0, pw, ph = 66, 68, 390, 295
    draw_axes(out, x0, y0, pw, ph, 0.0, 3000.0, ymin, ymax, "Evaluaciones de J_train (solo ventana x ∈ [2, 60])", "Objetivo J_train (M=128)", "A. Búsqueda inversa (Autofoco 1D/2D → DE 7D → Polish M=128) (0xB800)", nx=6, ny=4)
    out.append(polyline([map_pt(e, o, x0, y0, pw, ph, 0.0, 3000.0, ymin, ymax) for e, o in zip(evs, objs)], "#1d4ed8", 2.0))
    for mark_e, lbl in [(980, "DE 7D"), (2150, "Polish M=128")]:
        px, _ = map_pt(mark_e, 0, x0, y0, pw, ph, 0.0, 3000.0, ymin, ymax)
        out.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + ph}" stroke="#94a3b8" stroke-dasharray="3,3"/>')
        out.append(f'<text x="{px + 3:.1f}" y="{y0 + 14}" style="font-size:8.5px;fill:#475569">{lbl}</text>')

    # Right: Mode count M transition (Suite C3)
    c3 = summary["phases"]["CONFIRM"]["descriptive"]["c3"]
    ms = [8, 16, 32, 64, 128, 256]
    a100 = [c3[f"modes_M_{m}"]["auc_2_100"]["mean"] for m in ms]
    a150 = [c3[f"modes_M_{m}"]["auc_61_150"]["mean"] for m in ms]
    x0r, y0r, pwr, phr = 524, 68, 386, 295
    draw_axes(out, x0r, y0r, pwr, phr, 0.0, 5.0, 0.45, 1.03, "Número de pares de ondas M (escala log₂)", "ROC-AUC potencias de primos", "B. Transición de fase de Rayleigh por número de pares de ondas M", nx=5, ny=4)
    pts100 = [map_pt(i, v, x0r, y0r, pwr, phr, 0.0, 5.0, 0.45, 1.03) for i, v in enumerate(a100)]
    pts150 = [map_pt(i, v, x0r, y0r, pwr, phr, 0.0, 5.0, 0.45, 1.03) for i, v in enumerate(a150)]
    out.append(polyline(pts100, "#1d4ed8", 2.0))
    out.append(polyline(pts150, "#dc2626", 2.0))
    for i, m in enumerate(ms):
        px, py1 = pts100[i]
        _, py2 = pts150[i]
        out.append(f'<circle cx="{px:.1f}" cy="{py1:.1f}" r="3.5" fill="#1d4ed8"/>')
        out.append(f'<circle cx="{px:.1f}" cy="{py2:.1f}" r="3.5" fill="#dc2626"/>')
        out.append(f'<text x="{px:.1f}" y="{y0r + phr + 14}" text-anchor="middle" style="font-size:9.5px;font-weight:600;fill:#0f172a">M={m}</text>')
    out.append('<rect x="695" y="295" width="206" height="44" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    out.append('<line x1="705" y1="310" x2="725" y2="310" stroke="#1d4ed8" stroke-width="2"/><text x="732" y="314" class="legend">Ventana x ∈ [2, 100]</text>')
    out.append('<line x1="705" y1="327" x2="725" y2="327" stroke="#dc2626" stroke-width="2"/><text x="732" y="331" class="legend">Held-out x ∈ [61, 150]</text>')

    out.append("</svg>")
    (FIG_DIR / "fig3_exact_config_search_and_rayleigh.svg").write_text("\n".join(out) + "\n")


def fig4_blind_zeros_and_chebyshev(trace_dir: pathlib.Path):
    W, H = 940, 620
    out = svg_header(W, H)
    out.append('<text x="24" y="26" class="title">Suite C4 — Descubrimiento Ciego de los Ceros de Riemann (0 llamadas a ζ) y Predicción de ψ(x) − x</text>')

    # Panel A: Guinand-Weil resonance spectrum + discovered zeros
    grows = list(csv.DictReader((trace_dir / "trace_guinand_weil.csv").open()))
    ws = [float(r["omega"]) for r in grows]
    rs = [float(r["resonance"]) for r in grows]
    zrows = list(csv.DictReader((trace_dir / "trace_discovered_zeros.csv").open()))
    exact_z = [float(r["exact_gamma"]) for r in zrows]
    dyson_z = [float(r["dyson_blind"]) for r in zrows]
    norep_z = [float(r["norep_blind"]) for r in zrows]

    ymin_r, ymax_r = min(rs) - 0.5, max(rs) + 0.9
    x0, y0, pw, ph = 66, 62, 844, 215
    draw_axes(out, x0, y0, pw, ph, 10.0, 82.0, ymin_r, ymax_r, "Frecuencia de colapso ω", "Resonancia dual R(ω)", "A. Paisaje de Guinand–Weil R(ω) desde criba [2..600]: ceros descubiertos a ciegas con repulsión de Dyson vs. sin repulsión", nx=9, ny=4)
    for gz in exact_z:
        px, _ = map_pt(gz, 0, x0, y0, pw, ph, 10.0, 82.0, ymin_r, ymax_r)
        out.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + ph}" stroke="#16a34a" stroke-width="0.9" stroke-dasharray="3,3" stroke-opacity="0.5"/>')
    out.append(polyline([map_pt(w, r, x0, y0, pw, ph, 10.0, 82.0, ymin_r, ymax_r) for w, r in zip(ws, rs)], "#334155", 1.3))
    for w in dyson_z:
        px, py = map_pt(w, ymax_r - 0.35, x0, y0, pw, ph, 10.0, 82.0, ymin_r, ymax_r)
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="#1d4ed8" fill-opacity="0.85"/>')
    for w in norep_z:
        px, py = map_pt(w, ymin_r + 0.35, x0, y0, pw, ph, 10.0, 82.0, ymin_r, ymax_r)
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.0" fill="#ef4444" fill-opacity="0.7"/>')
    out.append('<rect x="612" y="70" width="290" height="58" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    out.append('<line x1="622" y1="84" x2="640" y2="84" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="3,3"/><text x="648" y="88" class="legend">Ceros reales γ_1..γ_20 (referencia)</text>')
    out.append('<circle cx="631" cy="100" r="3.5" fill="#1d4ed8"/><text x="648" y="104" class="legend">Descubiertos a ciegas + Dyson (arriba)</text>')
    out.append('<circle cx="631" cy="116" r="3.0" fill="#ef4444"/><text x="648" y="120" class="legend">Sin repulsión — colisiones (abajo)</text>')

    # Panel B: Out-of-sample Chebyshev psi(x) - x on [601, 1200]
    crows = list(csv.DictReader((trace_dir / "trace_chebyshev_extrap.csv").open()))
    cx = [float(r["x"]) for r in crows]
    ct = [float(r["true_delta_psi"]) for r in crows]
    ce = [float(r["exact20_pred"]) for r in crows]
    cd = [float(r["dyson_blind_pred"]) for r in crows]
    cn = [float(r["norep_blind_pred"]) for r in crows]
    ymin_c = min(min(ct), min(cd), min(cn)) * 1.1
    ymax_c = max(max(ct), max(cd), max(cn)) * 1.15
    x0b, y0b, pwb, phb = 66, 348, 844, 215
    draw_axes(out, x0b, y0b, pwb, phb, 600.0, 1200.0, ymin_c, ymax_c, "x nunca visto ∈ [601, 1200]", "Fluctuación ψ(x) − x", "B. Extrapolación fuera de muestra de oscilaciones de conteo de primos ψ(x) − x usando los 20 ceros descubiertos a ciegas en [2, 600]", nx=6, ny=4)
    out.append(polyline([map_pt(x, y, x0b, y0b, pwb, phb, 600.0, 1200.0, ymin_c, ymax_c) for x, y in zip(cx, ct)], "#0f172a", 1.5, opacity=0.75))
    out.append(polyline([map_pt(x, y, x0b, y0b, pwb, phb, 600.0, 1200.0, ymin_c, ymax_c) for x, y in zip(cx, cn)], "#ef4444", 1.2, dash="4,3", opacity=0.7))
    out.append(polyline([map_pt(x, y, x0b, y0b, pwb, phb, 600.0, 1200.0, ymin_c, ymax_c) for x, y in zip(cx, ce)], "#16a34a", 1.3, dash="2,2", opacity=0.85))
    out.append(polyline([map_pt(x, y, x0b, y0b, pwb, phb, 600.0, 1200.0, ymin_c, ymax_c) for x, y in zip(cx, cd)], "#1d4ed8", 1.85))
    out.append('<rect x="646" y="356" width="256" height="72" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    out.append('<line x1="656" y1="370" x2="674" y2="370" stroke="#0f172a" stroke-width="1.6"/><text x="682" y="374" class="legend">Fluctuación real ψ(x) − x</text>')
    out.append('<line x1="656" y1="386" x2="674" y2="386" stroke="#1d4ed8" stroke-width="2"/><text x="682" y="390" class="legend">Ceros ciegos + Dyson (predicción)</text>')
    out.append('<line x1="656" y1="402" x2="674" y2="402" stroke="#16a34a" stroke-width="1.4" stroke-dasharray="2,2"/><text x="682" y="406" class="legend">Techo teórico (20 ceros exactos)</text>')
    out.append('<line x1="656" y1="418" x2="674" y2="418" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="4,3"/><text x="682" y="422" class="legend">Ceros ciegos sin repulsión</text>')

    out.append("</svg>")
    (FIG_DIR / "fig4_blind_zero_discovery_and_chebyshev.svg").write_text("\n".join(out) + "\n")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((DOCS_DIR / "summary.json").read_text())
    trace_dir = DOCS_DIR / "raw" / "CONFIRM" / "0xB800"
    fig1_past_future_emergence(trace_dir)
    fig2_spacetime_and_pulse(trace_dir)
    fig3_search_and_rayleigh(trace_dir, summary)
    fig4_blind_zeros_and_chebyshev(trace_dir)
    print(f"[PLOTS] Generated 4 SVG figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
