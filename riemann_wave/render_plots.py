#!/usr/bin/env python3
"""Pure-stdlib SVG scientific figure generator from CONFIRM traces and summary.json."""
import csv
import json
import math
import pathlib
from xml.sax.saxutils import escape

CRATE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CRATE_DIR.parent
DOCS_DIR = REPO_ROOT / "docs" / "riemann_wave_v1"
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


def draw_axes(out: list[str], x0: float, y0: float, pw: float, ph: float, xmin: float, xmax: float, ymin: float, ymax: float, xlabel: str, ylabel: str, title: str, nx: int = 5, ny: int = 4):
    out.append(f'<text x="{x0}" y="{y0 - 10}" class="sub">{escape(title)}</text>')
    # Grid & ticks
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
    out.append(f'<text x="{x0 - 42:.1f}" y="{y0 + ph / 2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 {x0 - 42:.1f} {y0 + ph / 2:.1f})">{escape(ylabel)}</text>')


def map_pt(x, y, x0, y0, pw, ph, xmin, xmax, ymin, ymax):
    px = x0 + (x - xmin) / (xmax - xmin) * pw
    yc = max(ymin, min(ymax, y))
    py = y0 + ph - (yc - ymin) / (ymax - ymin) * ph
    return px, py


def is_prime(n: int) -> bool:
    return n >= 2 and all(n % d != 0 for d in range(2, int(n ** 0.5) + 1))


def fig1_primes_and_gue(trace_dir: pathlib.Path):
    W, H = 920, 620
    out = svg_header(W, H)
    out.append('<text x="24" y="26" class="title">Suite R1 — Interferencia de Ondas de Riemann (Primos) y Ley de Espaciamiento GUE</text>')

    # Panel A: Prime interference
    rows = list(csv.DictReader((trace_dir / "trace_primes.csv").open()))
    xs = [float(r["x"]) for r in rows]
    yr = [float(r["riemann_wave"]) for r in rows]
    yp = [float(r["poisson_wave"]) for r in rows]
    ymin, ymax = min(min(yr), min(yp)) * 1.05, max(max(yr), max(yp)) * 1.12
    x0, y0, pw, ph = 66, 62, 824, 210
    draw_axes(out, x0, y0, pw, ph, 1.5, 35.0, ymin, ymax, "x", "Amplitud W(x)", "A. Superposición de ondas de Riemann (M=128 ceros exactos) vs. frecuencias aleatorias Poisson", nx=7, ny=4)

    for p in range(2, 35):
        if is_prime(p):
            px, _ = map_pt(p, 0, x0, y0, pw, ph, 1.5, 35.0, ymin, ymax)
            out.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + ph}" stroke="#dc2626" stroke-width="0.9" stroke-dasharray="3,3" stroke-opacity="0.45"/>')
            out.append(f'<text x="{px:.1f}" y="{y0 + 11}" text-anchor="middle" style="font-size:8.5px;font-weight:700;fill:#dc2626">{p}</text>')

    out.append(polyline([map_pt(x, y, x0, y0, pw, ph, 1.5, 35.0, ymin, ymax) for x, y in zip(xs, yp)], "#f59e0b", 1.0, opacity=0.55))
    out.append(polyline([map_pt(x, y, x0, y0, pw, ph, 1.5, 35.0, ymin, ymax) for x, y in zip(xs, yr)], "#1d4ed8", 1.5))
    out.append('<rect x="655" y="72" width="228" height="40" rx="4" fill="#ffffff" fill-opacity="0.9" stroke="#cbd5e1"/>')
    out.append('<line x1="665" y1="86" x2="685" y2="86" stroke="#1d4ed8" stroke-width="2"/><text x="692" y="90" class="legend">Ceros de Riemann (AUC=1.00)</text>')
    out.append('<line x1="665" y1="102" x2="685" y2="102" stroke="#f59e0b" stroke-width="1.5"/><text x="692" y="106" class="legend">Control Poisson (mismo rango)</text>')

    # Panel B: Spacing histograms
    srows = list(csv.DictReader((trace_dir / "trace_spacings.csv").open()))
    by_src: dict[str, list[float]] = {}
    for r in srows:
        by_src.setdefault(r["source"], []).append(float(r["spacing"]))

    x0b, y0b, pwb, phb = 66, 345, 824, 225
    draw_axes(out, x0b, y0b, pwb, phb, 0.0, 3.2, 0.0, 1.15, "Espaciamiento normalizado (s)", "Densidad P(s)", "B. Repulsión de niveles en s = 0: Gas de Dyson (β=2), Ceros de Riemann y GUE Matricial vs. Poisson", nx=8, ny=4)

    nbins = 28
    bw = 3.2 / nbins

    def hist_dens(vals: list[float]) -> list[float]:
        c = [0] * nbins
        for v in vals:
            idx = int(v / bw)
            if 0 <= idx < nbins:
                c[idx] += 1
        tot = len(vals) * bw
        return [x / tot for x in c]

    hd = hist_dens(by_src["dyson_gue"])
    hr = hist_dens(by_src["riemann"])
    for i, d in enumerate(hd):
        px1, py1 = map_pt(i * bw, d, x0b, y0b, pwb, phb, 0.0, 3.2, 0.0, 1.15)
        px2, py0 = map_pt((i + 1) * bw, 0.0, x0b, y0b, pwb, phb, 0.0, 3.2, 0.0, 1.15)
        out.append(f'<rect x="{px1:.1f}" y="{py1:.1f}" width="{max(1.0, px2 - px1 - 1):.1f}" height="{max(0.0, py0 - py1):.1f}" fill="#3b82f6" fill-opacity="0.38" stroke="#1d4ed8" stroke-width="0.6"/>')

    step_pts = []
    for i, d in enumerate(hr):
        step_pts.append(map_pt(i * bw, d, x0b, y0b, pwb, phb, 0.0, 3.2, 0.0, 1.15))
        step_pts.append(map_pt((i + 1) * bw, d, x0b, y0b, pwb, phb, 0.0, 3.2, 0.0, 1.15))
    out.append(polyline(step_pts, "#dc2626", 1.8))

    s_grid = [i * 3.2 / 240 for i in range(241)]
    gue_pts = [map_pt(s, (32.0 / math.pi ** 2) * s * s * math.exp(-4.0 * s * s / math.pi), x0b, y0b, pwb, phb, 0.0, 3.2, 0.0, 1.15) for s in s_grid]
    poi_pts = [map_pt(s, math.exp(-s), x0b, y0b, pwb, phb, 0.0, 3.2, 0.0, 1.15) for s in s_grid]
    out.append(polyline(gue_pts, "#0f172a", 2.1))
    out.append(polyline(poi_pts, "#475569", 1.5, dash="5,4"))

    out.append('<rect x="620" y="356" width="262" height="72" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    out.append('<rect x="630" y="366" width="16" height="9" fill="#3b82f6" fill-opacity="0.45" stroke="#1d4ed8"/><text x="654" y="374" class="legend">Gas de Dyson β=2 (simulado)</text>')
    out.append('<line x1="630" y1="386" x2="646" y2="386" stroke="#dc2626" stroke-width="2"/><text x="654" y="390" class="legend">Ceros reales de Riemann (n=128)</text>')
    out.append('<line x1="630" y1="402" x2="646" y2="402" stroke="#0f172a" stroke-width="2.1"/><text x="654" y="406" class="legend">Teoría Wigner GUE (β=2)</text>')
    out.append('<line x1="630" y1="418" x2="646" y2="418" stroke="#475569" stroke-width="1.5" stroke-dasharray="5,4"/><text x="654" y="422" class="legend">Control Poisson (sin repulsión)</text>')

    out.append("</svg>")
    (FIG_DIR / "fig1_riemann_primes_and_gue.svg").write_text("\n".join(out) + "\n")


def fig2_inference_and_dyson(trace_dir: pathlib.Path):
    W, H = 920, 620
    out = svg_header(W, H)
    out.append('<text x="24" y="26" class="title">Suite R2 — Inferencia de Ondas bajo Perturbación de Fase (σ_φ = 0.015 rad) y Dinámica de Dyson</text>')

    rows = list(csv.DictReader((trace_dir / "trace_wave_inference.csv").open()))
    xs = [float(r["x"]) for r in rows]
    yt = [float(r["y_true"]) for r in rows]
    yd = [float(r["dyson_ols"]) for r in rows]
    yr = [float(r["riemann_ols"]) for r in rows]
    yp = [float(r["poisson_ols"]) for r in rows]

    x0, y0, pw, ph = 66, 62, 824, 215
    draw_axes(out, x0, y0, pw, ph, 1.5, 35.0, -1.0, 1.6, "Espacio de memoria (x, escala logarítmica)", "Respuesta de inferencia", "A. Inferencia OLS con jitter de fase (semilla 0xB700): estabilidad Dyson/Riemann vs. caos Poisson (acotado a [-1, 1.6])", nx=7, ny=5)
    out.append(polyline([map_pt(x, y, x0, y0, pw, ph, 1.5, 35.0, -1.0, 1.6) for x, y in zip(xs, yp)], "#ef4444", 1.1, opacity=0.55))
    out.append(polyline([map_pt(x, y, x0, y0, pw, ph, 1.5, 35.0, -1.0, 1.6) for x, y in zip(xs, yr)], "#0d9488", 1.5, opacity=0.85))
    out.append(polyline([map_pt(x, y, x0, y0, pw, ph, 1.5, 35.0, -1.0, 1.6) for x, y in zip(xs, yd)], "#1d4ed8", 2.0))
    out.append(polyline([map_pt(x, y, x0, y0, pw, ph, 1.5, 35.0, -1.0, 1.6) for x, y in zip(xs, yt)], "#0f172a", 1.3, dash="4,3"))

    out.append('<rect x="640" y="72" width="242" height="72" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    out.append('<line x1="650" y1="86" x2="668" y2="86" stroke="#1d4ed8" stroke-width="2"/><text x="676" y="90" class="legend">Patrón Dyson β=2 (estable)</text>')
    out.append('<line x1="650" y1="102" x2="668" y2="102" stroke="#0d9488" stroke-width="1.8"/><text x="676" y="106" class="legend">Ceros reales de Riemann</text>')
    out.append('<line x1="650" y1="118" x2="668" y2="118" stroke="#ef4444" stroke-width="1.4"/><text x="676" y="122" class="legend">Sin repulsión / Poisson (caos)</text>')
    out.append('<line x1="650" y1="134" x2="668" y2="134" stroke="#0f172a" stroke-width="1.4" stroke-dasharray="4,3"/><text x="676" y="138" class="legend">Conceptos objetivo</text>')

    # Panel B: Dyson trajectories
    trows = list(csv.DictReader((trace_dir / "trace_dyson_trajectories.csv").open()))
    steps = [float(r["step"]) for r in trows]
    n_modes = len(trows[0]) - 1
    ymin_t, ymax_t = -16.5, 16.5
    x0b, y0b, pwb, phb = 66, 345, 824, 225
    draw_axes(out, x0b, y0b, pwb, phb, 0.0, max(steps), ymin_t, ymax_t, "Paso de consolidación Langevin (t)", "Autovalor ω_i", "B. Consolidación espectral de Dyson (60 modos): las frecuencias se repelen sin cruzarse hacia el semicírculo de Wigner", nx=8, ny=4)
    for k in range(n_modes):
        col = f"w{k}"
        pts = [map_pt(s, float(r[col]), x0b, y0b, pwb, phb, 0.0, max(steps), ymin_t, ymax_t) for s, r in zip(steps, trows)]
        out.append(polyline(pts, "#2563eb", 0.85, opacity=0.45))

    out.append("</svg>")
    (FIG_DIR / "fig2_wave_inference_and_dyson.svg").write_text("\n".join(out) + "\n")


def fig3_ghost_and_controls(trace_dir: pathlib.Path, summary: dict):
    W, H = 920, 620
    out = svg_header(W, H)
    out.append('<text x="24" y="26" class="title">Suite R3 — Anti-Aliasing Fuera de Periodo (u ∈ [2π, 4π)) y Comparativa Honesta de 6 Familias (CONFIRM)</text>')

    rows = list(csv.DictReader((trace_dir / "trace_ghost_aliasing.csv").open()))
    us = [float(r["u"]) for r in rows]
    yf = [float(r["fourier_ghost"]) for r in rows]
    yd = [float(r["dyson_ghost"]) for r in rows]
    yr = [float(r["riemann_ghost"]) for r in rows]
    yg = [float(r["golden_ghost"]) for r in rows]

    x0, y0, pw, ph = 66, 62, 824, 210
    draw_axes(out, x0, y0, pw, ph, min(us), max(us), -0.4, 1.25, "Coordenada fuera de periodo u ∈ [2π, 4π) (objetivo verdadero = 0)", "Eco fantasma ŷ(u)", "A. Alucinación periódica de Fourier entero vs. supresión por frecuencias inconmensurables (semilla 0xB700)", nx=6, ny=4)
    out.append(polyline([map_pt(u, y, x0, y0, pw, ph, min(us), max(us), -0.4, 1.25) for u, y in zip(us, yf)], "#ea580c", 1.8, opacity=0.85))
    out.append(polyline([map_pt(u, y, x0, y0, pw, ph, min(us), max(us), -0.4, 1.25) for u, y in zip(us, yg)], "#9333ea", 1.1, opacity=0.65))
    out.append(polyline([map_pt(u, y, x0, y0, pw, ph, min(us), max(us), -0.4, 1.25) for u, y in zip(us, yr)], "#0d9488", 1.4, opacity=0.85))
    out.append(polyline([map_pt(u, y, x0, y0, pw, ph, min(us), max(us), -0.4, 1.25) for u, y in zip(us, yd)], "#1d4ed8", 1.6))

    out.append('<rect x="642" y="70" width="240" height="72" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    out.append('<line x1="652" y1="84" x2="670" y2="84" stroke="#ea580c" stroke-width="2"/><text x="678" y="88" class="legend">Fourier entero (ecos fantasma ≈1.0)</text>')
    out.append('<line x1="652" y1="100" x2="670" y2="100" stroke="#1d4ed8" stroke-width="2"/><text x="678" y="104" class="legend">Dyson β=2 (suprime aliasing)</text>')
    out.append('<line x1="652" y1="116" x2="670" y2="116" stroke="#0d9488" stroke-width="1.8"/><text x="678" y="120" class="legend">Ceros reales de Riemann</text>')
    out.append('<line x1="652" y1="132" x2="670" y2="132" stroke="#9333ea" stroke-width="1.4"/><text x="678" y="136" class="legend">Control Golden Jitter</text>')

    # Panel B: Grouped bar chart across 6 families on CONFIRM (16 seeds)
    conf_r3 = summary["phases"]["CONFIRM"]["descriptive"]["r3"]
    fams = ["dyson_gue", "riemann_zeros", "golden_jitter", "fourier_uniform", "goe_beta1", "poisson"]
    labels = ["Dyson β=2", "Riemann", "Golden", "Fourier", "GOE β=1", "Poisson"]
    x0b, y0b, pwb, phb = 66, 345, 824, 215
    ymax_b = max(conf_r3[f]["ghost_peak"]["mean"] for f in fams) * 1.15
    draw_axes(out, x0b, y0b, pwb, phb, 0.0, 6.0, 0.0, ymax_b, "Familia espectral (media de 16 semillas frescas CONFIRM)", "Error / Amplitud", "B. Compromiso In-Domain (Ridge) vs. Pico Fantasma Fuera de Periodo vs. Muestreo Irregular (menor es mejor)", nx=6, ny=4)

    group_w = pwb / len(fams)
    bar_w = group_w * 0.23
    metrics = [("in_domain_rmse", "#2563eb", "RMSE in-domain"), ("ghost_peak", "#ea580c", "Pico fantasma [2π,4π)"), ("irreg_rmse_ridge", "#16a34a", "RMSE muestreo irregular")]
    for fi, fam in enumerate(fams):
        gx = x0b + fi * group_w + group_w * 0.14
        for mi, (mk, col, _) in enumerate(metrics):
            val = conf_r3[fam][mk]["mean"]
            bx = gx + mi * (bar_w + 3)
            _, py = map_pt(0, val, x0b, y0b, pwb, phb, 0, 1, 0, ymax_b)
            bh = (y0b + phb) - py
            out.append(f'<rect x="{bx:.1f}" y="{py:.1f}" width="{bar_w:.1f}" height="{max(1.0, bh):.1f}" fill="{col}" fill-opacity="0.85"/>')
            out.append(f'<text x="{bx + bar_w / 2:.1f}" y="{py - 3:.1f}" text-anchor="middle" style="font-size:8.5px;fill:#334155">{val:.2f}</text>')
        out.append(f'<text x="{x0b + (fi + 0.5) * group_w:.1f}" y="{y0b + phb + 14:.1f}" text-anchor="middle" style="font-size:10px;font-weight:600;fill:#0f172a">{labels[fi]}</text>')

    out.append('<rect x="650" y="352" width="234" height="58" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    for mi, (_, col, lbl) in enumerate(metrics):
        ly = 366 + mi * 16
        out.append(f'<rect x="660" y="{ly - 8}" width="12" height="9" fill="{col}"/><text x="678" y="{ly}" class="legend">{lbl}</text>')

    out.append("</svg>")
    (FIG_DIR / "fig3_ghost_aliasing_and_controls.svg").write_text("\n".join(out) + "\n")


def fig4_streaming_and_rff(summary: dict):
    W, H = 920, 440
    out = svg_header(W, H)
    out.append('<text x="24" y="26" class="title">Suites R4 &amp; R5 — Memoria Online en Streaming, Ablación de Repulsión e Inferencia de Kernel (CONFIRM)</text>')

    # Left panel: Suite R4 streaming RMSE
    r4 = summary["phases"]["CONFIRM"]["descriptive"]["r4"]
    arms4 = ["fourier_uniform", "dyson_gue", "riemann_zeros", "poisson", "adaptive_dyson", "adaptive_no_repulsion"]
    lbl4 = ["Fourier", "Dyson β=2", "Riemann", "Poisson", "Adapt+Dyson", "Adapt sin rep."]
    col4 = ["#64748b", "#1d4ed8", "#0d9488", "#ef4444", "#2563eb", "#dc2626"]
    ymax4 = max(r4[a]["stream_rmse"]["mean"] + r4[a]["stream_rmse"]["std"] for a in arms4) * 1.15
    x0, y0, pw, ph = 64, 68, 390, 300
    draw_axes(out, x0, y0, pw, ph, 0, 6, 0, ymax4, "Brazo (8 conceptos secuenciales, σ_φ=0.02)", "RMSE global", "A. Suite R4: Streaming RLS + Ablación Causal de Repulsión", nx=6, ny=4)
    bw = pw / len(arms4)
    for i, a in enumerate(arms4):
        mu = r4[a]["stream_rmse"]["mean"]
        sd = r4[a]["stream_rmse"]["std"]
        bx = x0 + i * bw + bw * 0.18
        wbar = bw * 0.64
        _, py = map_pt(0, mu, x0, y0, pw, ph, 0, 1, 0, ymax4)
        _, py_hi = map_pt(0, mu + sd, x0, y0, pw, ph, 0, 1, 0, ymax4)
        _, py_lo = map_pt(0, max(0.0, mu - sd), x0, y0, pw, ph, 0, 1, 0, ymax4)
        out.append(f'<rect x="{bx:.1f}" y="{py:.1f}" width="{wbar:.1f}" height="{(y0 + ph) - py:.1f}" fill="{col4[i]}" fill-opacity="0.82"/>')
        cx = bx + wbar / 2
        out.append(f'<line x1="{cx:.1f}" y1="{py_hi:.1f}" x2="{cx:.1f}" y2="{py_lo:.1f}" stroke="#0f172a" stroke-width="1.3"/>')
        out.append(f'<text x="{cx:.1f}" y="{py_hi - 4:.1f}" text-anchor="middle" style="font-size:8.5px;fill:#0f172a">{mu:.3f}</text>')
        out.append(f'<text x="{cx:.1f}" y="{y0 + ph + 14:.1f}" text-anchor="middle" style="font-size:8.5px;font-weight:600;fill:#0f172a">{lbl4[i]}</text>')

    # Right panel: Suite R5 kernel Frobenius error (1D & 2D)
    r5_1 = summary["phases"]["CONFIRM"]["descriptive"]["r5_1d"]
    r5_2 = summary["phases"]["CONFIRM"]["descriptive"]["r5_2d"]
    items5 = [
        ("1D Poisson", r5_1["poisson_rff"]["kernel_rel_frob"], "#ef4444"),
        ("1D Dyson", r5_1["dyson_rff"]["kernel_rel_frob"], "#1d4ed8"),
        ("1D Riemann", r5_1["riemann_rff"]["kernel_rel_frob"], "#0d9488"),
        ("1D QMC", r5_1["qmc_rff"]["kernel_rel_frob"], "#16a34a"),
        ("2D Poisson", r5_2["poisson_2d_rff"]["kernel_rel_frob"], "#f87171"),
        ("2D Ginibre", r5_2["ginibre_2d_rff"]["kernel_rel_frob"], "#2563eb"),
        ("2D QMC", r5_2["qmc_2d_rff"]["kernel_rel_frob"], "#22c55e"),
    ]
    ymax5 = max(it[1]["mean"] + it[1]["std"] for it in items5) * 1.18
    x0r, y0r, pwr, phr = 515, 68, 380, 300
    draw_axes(out, x0r, y0r, pwr, phr, 0, 7, 0, ymax5, "Muestreador espectral (1D M=48 | 2D M=64)", "Error Frobenius rel.", "B. Suite R5: Fidelidad del Kernel RBF (DPP-GUE / Ginibre vs. Poisson vs. QMC)", nx=7, ny=4)
    bw5 = pwr / len(items5)
    for i, (lbl, st, col) in enumerate(items5):
        mu, sd = st["mean"], st["std"]
        bx = x0r + i * bw5 + bw5 * 0.16
        wbar = bw5 * 0.68
        _, py = map_pt(0, mu, x0r, y0r, pwr, phr, 0, 1, 0, ymax5)
        _, py_hi = map_pt(0, mu + sd, x0r, y0r, pwr, phr, 0, 1, 0, ymax5)
        _, py_lo = map_pt(0, max(0.0, mu - sd), x0r, y0r, pwr, phr, 0, 1, 0, ymax5)
        out.append(f'<rect x="{bx:.1f}" y="{py:.1f}" width="{wbar:.1f}" height="{(y0r + phr) - py:.1f}" fill="{col}" fill-opacity="0.85"/>')
        cx = bx + wbar / 2
        out.append(f'<line x1="{cx:.1f}" y1="{py_hi:.1f}" x2="{cx:.1f}" y2="{py_lo:.1f}" stroke="#0f172a" stroke-width="1.3"/>')
        out.append(f'<text x="{cx:.1f}" y="{py_hi - 4:.1f}" text-anchor="middle" style="font-size:8.5px;fill:#0f172a">{mu:.3f}</text>')
        out.append(f'<text x="{cx:.1f}" y="{y0r + phr + 14:.1f}" text-anchor="middle" style="font-size:8.2px;font-weight:600;fill:#0f172a">{lbl}</text>')

    out.append("</svg>")
    (FIG_DIR / "fig4_streaming_and_kernel_rff.svg").write_text("\n".join(out) + "\n")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((DOCS_DIR / "summary.json").read_text())
    trace_dir = DOCS_DIR / "raw" / "CONFIRM" / "0xB700"
    fig1_primes_and_gue(trace_dir)
    fig2_inference_and_dyson(trace_dir)
    fig3_ghost_and_controls(trace_dir, summary)
    fig4_streaming_and_rff(summary)
    print(f"[PLOTS] Generated 4 SVG figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
