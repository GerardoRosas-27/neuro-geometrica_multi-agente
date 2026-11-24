#!/usr/bin/env python3
"""Pure-stdlib SVG figure generator for `exp/riemann-prime-inference`.

Reads `docs/riemann_prime_inference_v1/summary_confirm.json` and visual trace CSVs
from `raw/confirm/seed_0xB900/` to produce 4 self-contained SVG charts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DOCS_DIR = REPO_ROOT / "docs" / "riemann_prime_inference_v1"
FIG_DIR = DOCS_DIR / "figures"
TRACE_DIR = DOCS_DIR / "raw" / "confirm" / "seed_0xB900"

# Clean scientific palette
COL_RIEMANN = "#2563eb"
COL_RANDOM = "#dc2626"
COL_DYSON = "#d97706"
COL_FOURIER = "#7c3aed"
COL_BAYES = "#059669"
COL_PRIOR = "#64748b"
COL_TEXT = "#0f172a"
COL_MUTED = "#475569"
COL_GRID = "#e2e8f0"
COL_BG = "#ffffff"
COL_PANEL = "#f8fafc"


def svg_header(w: int, h: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'style="background:{COL_BG};font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif">',
        f'<rect width="{w}" height="{h}" fill="{COL_BG}"/>',
        f'<text x="28" y="34" font-size="18" font-weight="700" fill="{COL_TEXT}">{title}</text>',
        f'<text x="28" y="54" font-size="12" fill="{COL_MUTED}">{subtitle}</text>',
    ]


def draw_panel_box(x: float, y: float, w: float, h: float, title: str) -> list[str]:
    return [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="{COL_PANEL}" stroke="{COL_GRID}"/>',
        f'<text x="{x + 14:.1f}" y="{y + 24:.1f}" font-size="13" font-weight="700" fill="{COL_TEXT}">{title}</text>',
    ]


def map_x(val: float, lo: float, hi: float, x0: float, x1: float) -> float:
    return x0 + (val - lo) / max(hi - lo, 1e-12) * (x1 - x0)


def map_y(val: float, lo: float, hi: float, y0: float, y1: float) -> float:
    return y1 - (val - lo) / max(hi - lo, 1e-12) * (y1 - y0)


def polyline_path(pts: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def draw_bar_group(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    items: list[tuple[str, float, float, float, str]],
    y_max: float,
    y_label: str,
    fmt_pct: bool = True,
) -> list[str]:
    out = []
    # Horizontal grid lines
    for i in range(5):
        frac = i / 4.0
        yv = y_max * frac
        py = map_y(yv, 0.0, y_max, y0, y1)
        out.append(f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x1:.1f}" y2="{py:.1f}" stroke="{COL_GRID}" stroke-dasharray="3,3"/>')
        lbl = f"{100*yv:.0f}%" if fmt_pct else f"{yv:.2f}"
        out.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" font-size="10" text-anchor="end" fill="{COL_MUTED}">{lbl}</text>')

    out.append(f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{COL_MUTED}"/>')
    out.append(
        f'<text x="{x0 - 40:.1f}" y="{(y0 + y1)/2:.1f}" font-size="11" fill="{COL_MUTED}" '
        f'text-anchor="middle" transform="rotate(-90 {x0 - 40:.1f} {(y0 + y1)/2:.1f})">{y_label}</text>'
    )

    n = len(items)
    slot = (x1 - x0) / n
    bar_w = min(56.0, slot * 0.56)
    for idx, (label, val, lo, hi, color) in enumerate(items):
        cx = x0 + (idx + 0.5) * slot
        bx = cx - bar_w / 2.0
        top = map_y(val, 0.0, y_max, y0, y1)
        bh = max(1.0, y1 - top)
        out.append(f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="4" fill="{color}" opacity="0.90"/>')

        # 95% CI whisker
        ylo = map_y(max(0.0, lo), 0.0, y_max, y0, y1)
        yhi = map_y(min(y_max, hi), 0.0, y_max, y0, y1)
        out.append(f'<line x1="{cx:.1f}" y1="{ylo:.1f}" x2="{cx:.1f}" y2="{yhi:.1f}" stroke="{COL_TEXT}" stroke-width="1.6"/>')
        out.append(f'<line x1="{cx - 5:.1f}" y1="{ylo:.1f}" x2="{cx + 5:.1f}" y2="{ylo:.1f}" stroke="{COL_TEXT}" stroke-width="1.6"/>')
        out.append(f'<line x1="{cx - 5:.1f}" y1="{yhi:.1f}" x2="{cx + 5:.1f}" y2="{yhi:.1f}" stroke="{COL_TEXT}" stroke-width="1.6"/>')

        vtxt = f"{100*val:.1f}%" if fmt_pct else f"{val:.2f}"
        out.append(f'<text x="{cx:.1f}" y="{min(yhi, top) - 6:.1f}" font-size="10.5" font-weight="700" text-anchor="middle" fill="{COL_TEXT}">{vtxt}</text>')

        # Multiline x-label split on \n
        parts = label.split("\n")
        for li, part in enumerate(parts):
            out.append(f'<text x="{cx:.1f}" y="{y1 + 15 + 12*li:.1f}" font-size="10" text-anchor="middle" fill="{COL_TEXT}">{part}</text>')
    return out


def render_fig1(confirm: dict) -> None:
    w, h = 1040, 440
    lines = svg_header(
        w,
        h,
        "Figura 1 — Rastreo Retrodictivo de Origen (Suite I1, CONFIRM N=16)",
        "Colapso Pasado x Futuro sobre el patrón primo de Riemann frente a colapso aleatorio, Dyson GUE y Fourier lineal",
    )

    # Panel A: continuous retrodictive profile for N = 17 x 19 = 323
    lines += draw_panel_box(24, 72, 488, 344, "A. Perfil continuo S_origen(u | u_323) para N = 17 x 19 = 323")
    rows = []
    with (TRACE_DIR / "trace_retro_example.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["case_id"] == "N323_17x19":
                rows.append(r)

    x0, y0, x1, y1 = 76.0, 118.0, 492.0, 356.0
    by_arm: dict[str, list[tuple[float, float]]] = {}
    u_true1 = 0.0
    for r in rows:
        arm = r["arm"]
        if arm == "fourier_linear_collapse":
            continue
        u_true1 = float(r["u_true1"])
        by_arm.setdefault(arm, []).append((float(r["u_coord"]), float(r["collapse_score"])))

    u_min = min(p[0] for p in by_arm["riemann_prime_collapse"])
    u_max = max(p[0] for p in by_arm["riemann_prime_collapse"])
    max_r = max(p[1] for p in by_arm["riemann_prime_collapse"]) or 1.0

    # Grid & axes
    for i in range(5):
        py = map_y(i * 0.25, 0.0, 1.05, y0, y1)
        lines.append(f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x1:.1f}" y2="{py:.1f}" stroke="{COL_GRID}" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" font-size="10" text-anchor="end" fill="{COL_MUTED}">{i*0.25:.2f}</text>')
    lines.append(f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{COL_MUTED}"/>')

    # True origin marker u = ln(17)
    tx = map_x(u_true1, u_min, u_max, x0, x1)
    lines.append(f'<line x1="{tx:.1f}" y1="{y0:.1f}" x2="{tx:.1f}" y2="{y1:.1f}" stroke="#059669" stroke-width="1.4" stroke-dasharray="4,3"/>')
    lines.append(f'<text x="{tx - 4:.1f}" y="{y0 + 12:.1f}" font-size="10" font-weight="700" text-anchor="end" fill="#059669">u = ln(17)</text>')

    for arm, col, sw in [
        ("dyson_gue_collapse", COL_DYSON, 1.4),
        ("random_collapse", COL_RANDOM, 1.5),
        ("riemann_prime_collapse", COL_RIEMANN, 2.2),
    ]:
        pts = [
            (map_x(u, u_min, u_max, x0, x1), map_y(min(1.05, s / max_r), 0.0, 1.05, y0, y1))
            for u, s in by_arm[arm]
        ]
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="{sw}" points="{polyline_path(pts)}"/>')

    # X ticks across full [ln(1.7), u_obs/2]
    for tick, lbl in [(0.693, "ln2"), (1.099, "ln3"), (1.609, "ln5"), (1.946, "ln7"), (2.398, "ln11"), (2.833, "ln17")]:
        if u_min <= tick <= u_max:
            px = map_x(tick, u_min, u_max, x0, x1)
            lines.append(f'<text x="{px:.1f}" y="{y1 + 16:.1f}" font-size="9.5" text-anchor="middle" fill="{COL_MUTED}">{lbl}</text>')
    lines.append(f'<text x="{(x0 + x1)/2:.1f}" y="{y1 + 34:.1f}" font-size="11" text-anchor="middle" fill="{COL_MUTED}">Coordenada candidata u = ln(p) (origen pasado)</text>')
    lines.append(f'<text x="{x0 - 40:.1f}" y="{(y0 + y1)/2:.1f}" font-size="11" text-anchor="middle" fill="{COL_MUTED}" transform="rotate(-90 {x0 - 40:.1f} {(y0 + y1)/2:.1f})">Colapso normalizado</text>')

    # Legend inside panel A
    for idx, (lbl, col) in enumerate(
        [("Riemann Primo", COL_RIEMANN), ("Aleatorio", COL_RANDOM), ("Dyson GUE", COL_DYSON)]
    ):
        lx = x0 + 8 + idx * 125
        lines.append(f'<rect x="{lx:.1f}" y="{y0 + 4:.1f}" width="12" height="4" fill="{col}"/>')
        lines.append(f'<text x="{lx + 16:.1f}" y="{y0 + 9:.1f}" font-size="10" fill="{COL_TEXT}">{lbl}</text>')

    # Panel B: Bar chart of Suite I1 origin_exact_acc at sigma=0.002 across 4 arms
    lines += draw_panel_box(528, 72, 488, 344, "B. Exactitud de rastreo exacto {c_a, c_b} (sigma = 0.002, IC 95%)")
    i1_002 = confirm["suites"]["I1"]["0.0020"]
    items = [
        (
            "Riemann\nPrimo",
            i1_002["riemann_prime_collapse"]["origin_exact_acc"]["mean"],
            i1_002["riemann_prime_collapse"]["origin_exact_acc"]["ci95_lo"],
            i1_002["riemann_prime_collapse"]["origin_exact_acc"]["ci95_hi"],
            COL_RIEMANN,
        ),
        (
            "Fourier\nLineal",
            i1_002["fourier_linear_collapse"]["origin_exact_acc"]["mean"],
            i1_002["fourier_linear_collapse"]["origin_exact_acc"]["ci95_lo"],
            i1_002["fourier_linear_collapse"]["origin_exact_acc"]["ci95_hi"],
            COL_FOURIER,
        ),
        (
            "Dyson GUE\n(sin primos)",
            i1_002["dyson_gue_collapse"]["origin_exact_acc"]["mean"],
            i1_002["dyson_gue_collapse"]["origin_exact_acc"]["ci95_lo"],
            i1_002["dyson_gue_collapse"]["origin_exact_acc"]["ci95_hi"],
            COL_DYSON,
        ),
        (
            "Colapso\nAleatorio",
            i1_002["random_collapse"]["origin_exact_acc"]["mean"],
            i1_002["random_collapse"]["origin_exact_acc"]["ci95_lo"],
            i1_002["random_collapse"]["origin_exact_acc"]["ci95_hi"],
            COL_RANDOM,
        ),
    ]
    lines += draw_bar_group(590.0, 126.0, 996.0, 356.0, items, 1.05, "Exactitud de par exacto", fmt_pct=True)

    # Theoretical linear additive ceiling line at 21/66 = 31.8%
    ceil_y = map_y(21.0 / 66.0, 0.0, 1.05, 126.0, 356.0)
    lines.append(f'<line x1="590" y1="{ceil_y:.1f}" x2="996" y2="{ceil_y:.1f}" stroke="{COL_FOURIER}" stroke-dasharray="5,4" opacity="0.7"/>')
    lines.append(f'<text x="992" y="{ceil_y - 5:.1f}" font-size="9.5" text-anchor="end" fill="{COL_FOURIER}">Techo aditivo 21/66 (31.8%)</text>')

    lines.append("</svg>")
    (FIG_DIR / "fig1_retrodictive_tracing.svg").write_text("\n".join(lines), encoding="utf-8")


def render_fig2(confirm: dict) -> None:
    w, h = 1040, 440
    lines = svg_header(
        w,
        h,
        "Figura 2 — Inferencia 'Del Otro Lado': Predicción No Lineal y Contrafactuales (Suite I2, CONFIRM N=16)",
        "Exactitud de clase futura 4-way y respuesta a intervenciones contrafactuales do(c_a -> c_new) por desplazamiento de fase",
    )
    i2 = confirm["suites"]["I2"]
    order = [
        ("Riemann\nPrimo", "riemann_prime_collapse", COL_RIEMANN),
        ("Fourier\nLineal", "fourier_linear_collapse", COL_FOURIER),
        ("Dyson GUE\n(sin primos)", "dyson_gue_collapse", COL_DYSON),
        ("Colapso\nAleatorio", "random_collapse", COL_RANDOM),
    ]

    lines += draw_panel_box(24, 72, 488, 344, "A. Predicción futura 4-way 'del otro lado' (class_acc, IC 95%)")
    items_a = [
        (lbl, i2[k]["class_acc"]["mean"], i2[k]["class_acc"]["ci95_lo"], i2[k]["class_acc"]["ci95_hi"], col)
        for lbl, k, col in order
    ]
    lines += draw_bar_group(84.0, 126.0, 492.0, 356.0, items_a, 1.05, "Exactitud clase futura", fmt_pct=True)

    lines += draw_panel_box(528, 72, 488, 344, "B. Intervención contrafactual do(c_a -> c_new) (cf_exact_acc, IC 95%)")
    items_b = [
        (lbl, i2[k]["cf_exact_acc"]["mean"], i2[k]["cf_exact_acc"]["ci95_lo"], i2[k]["cf_exact_acc"]["ci95_hi"], col)
        for lbl, k, col in order
    ]
    lines += draw_bar_group(590.0, 126.0, 996.0, 356.0, items_b, 1.05, "Exactitud contrafactual", fmt_pct=True)

    lines.append("</svg>")
    (FIG_DIR / "fig2_downstream_and_counterfactual.svg").write_text("\n".join(lines), encoding="utf-8")


def render_fig3(confirm: dict) -> None:
    w, h = 1040, 440
    lines = svg_header(
        w,
        h,
        "Figura 3 — Desambiguación Bayesiana Transaccional bajo Alto Ruido (Suite I3, sigma=0.015)",
        "El prior pasado suave acoplado a la onda futura de Riemann desambigua productos casi colisionantes (323 vs 319)",
    )

    lines += draw_panel_box(24, 72, 488, 344, "A. Caso ambiguo: 17x19=323 desplazado -0.008 hacia 11x29=319")
    rows = []
    with (TRACE_DIR / "trace_bayesian_example.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u_val = float(r["u_coord"])
            if u_val >= 2.25:
                rows.append(
                    (u_val, float(r["uniform_score"]), float(r["bayes_score"]))
                )

    x0, y0, x1, y1 = 76.0, 118.0, 492.0, 356.0
    u_min = rows[0][0]
    u_max = rows[-1][0]
    max_u = max(r[1] for r in rows) or 1.0
    max_b = max(r[2] for r in rows) or 1.0

    for i in range(5):
        py = map_y(i * 0.25, 0.0, 1.05, y0, y1)
        lines.append(f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x1:.1f}" y2="{py:.1f}" stroke="{COL_GRID}" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" font-size="10" text-anchor="end" fill="{COL_MUTED}">{i*0.25:.2f}</text>')
    lines.append(f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{COL_MUTED}"/>')

    # Reference lines for ln(11) = 2.3979 and ln(17) = 2.8332
    for val, lbl, col in [(2.3979, "ln(11) [rival 319]", COL_RANDOM), (2.8332, "ln(17) [real 323]", COL_BAYES)]:
        px = map_x(val, u_min, u_max, x0, x1)
        lines.append(f'<line x1="{px:.1f}" y1="{y0:.1f}" x2="{px:.1f}" y2="{y1:.1f}" stroke="{col}" stroke-dasharray="4,3"/>')
        anc = "start" if val < 2.6 else "end"
        dx = 4 if val < 2.6 else -4
        lines.append(f'<text x="{px + dx:.1f}" y="{y0 + 14:.1f}" font-size="9.5" font-weight="700" text-anchor="{anc}" fill="{col}">{lbl}</text>')

    pts_unif = [(map_x(u, u_min, u_max, x0, x1), map_y(s / max_u, 0.0, 1.05, y0, y1)) for u, s, _ in rows]
    pts_bay = [(map_x(u, u_min, u_max, x0, x1), map_y(b / max_b, 0.0, 1.05, y0, y1)) for u, _, b in rows]
    lines.append(f'<polyline fill="none" stroke="{COL_RIEMANN}" stroke-width="1.8" stroke-dasharray="5,3" points="{polyline_path(pts_unif)}"/>')
    lines.append(f'<polyline fill="none" stroke="{COL_BAYES}" stroke-width="2.4" points="{polyline_path(pts_bay)}"/>')

    for idx, (lbl, col) in enumerate([("Riemann Bayesiano", COL_BAYES), ("Riemann Uniforme", COL_RIEMANN)]):
        lx = x0 + 10 + idx * 165
        lines.append(f'<rect x="{lx:.1f}" y="{y1 - 22:.1f}" width="12" height="4" fill="{col}"/>')
        lines.append(f'<text x="{lx + 16:.1f}" y="{y1 - 17:.1f}" font-size="10" fill="{COL_TEXT}">{lbl}</text>')
    lines.append(f'<text x="{(x0 + x1)/2:.1f}" y="{y1 + 28:.1f}" font-size="11" text-anchor="middle" fill="{COL_MUTED}">Coordenada candidata u = ln(p)</text>')

    # Panel B: Suite I3 bars
    lines += draw_panel_box(528, 72, 488, 344, "B. Exactitud bajo ruido severo sigma = 0.015 (200 eventos/semilla)")
    i3 = confirm["suites"]["I3"]
    items = [
        ("Riemann\nBayesiano", i3["riemann_bayes_collapse"]["origin_exact_acc"]["mean"], i3["riemann_bayes_collapse"]["origin_exact_acc"]["ci95_lo"], i3["riemann_bayes_collapse"]["origin_exact_acc"]["ci95_hi"], COL_BAYES),
        ("Riemann\nUniforme", i3["riemann_uniform_collapse"]["origin_exact_acc"]["mean"], i3["riemann_uniform_collapse"]["origin_exact_acc"]["ci95_lo"], i3["riemann_uniform_collapse"]["origin_exact_acc"]["ci95_hi"], COL_RIEMANN),
        ("Solo\nPrior", i3["prior_only_ctrl"]["origin_exact_acc"]["mean"], i3["prior_only_ctrl"]["origin_exact_acc"]["ci95_lo"], i3["prior_only_ctrl"]["origin_exact_acc"]["ci95_hi"], COL_PRIOR),
        ("Aleatorio\n+ Prior", i3["random_bayes_collapse"]["origin_exact_acc"]["mean"], i3["random_bayes_collapse"]["origin_exact_acc"]["ci95_lo"], i3["random_bayes_collapse"]["origin_exact_acc"]["ci95_hi"], COL_RANDOM),
    ]
    lines += draw_bar_group(590.0, 126.0, 996.0, 356.0, items, 1.05, "Exactitud de par exacto", fmt_pct=True)

    lines.append("</svg>")
    (FIG_DIR / "fig3_bayesian_disambiguation.svg").write_text("\n".join(lines), encoding="utf-8")


def render_fig4(confirm: dict) -> None:
    w, h = 1120, 440
    lines = svg_header(
        w,
        h,
        "Figura 4 — Transferencia Analógica (A:B::C:?) y Límites Físicos Honestos del Canal 1D (Suite I4)",
        "A. Analogía composicional · B. Degradación real bajo ruido severo · C. Techo de capacidad al crecer K de 8 a 18",
    )

    # Panel A: Analogy bars
    lines += draw_panel_box(20, 72, 350, 344, "A. Analogía A:B::C:? (80/semilla)")
    an = confirm["suites"]["I4"]["analogy"]
    items_a = [
        ("Riemann", an["riemann_prime_collapse"]["exact_acc"]["mean"], an["riemann_prime_collapse"]["exact_acc"]["ci95_lo"], an["riemann_prime_collapse"]["exact_acc"]["ci95_hi"], COL_RIEMANN),
        ("Fourier", an["fourier_linear_collapse"]["exact_acc"]["mean"], an["fourier_linear_collapse"]["exact_acc"]["ci95_lo"], an["fourier_linear_collapse"]["exact_acc"]["ci95_hi"], COL_FOURIER),
        ("Dyson", an["dyson_gue_collapse"]["exact_acc"]["mean"], an["dyson_gue_collapse"]["exact_acc"]["ci95_lo"], an["dyson_gue_collapse"]["exact_acc"]["ci95_hi"], COL_DYSON),
        ("Aleat.", an["random_collapse"]["exact_acc"]["mean"], an["random_collapse"]["exact_acc"]["ci95_lo"], an["random_collapse"]["exact_acc"]["ci95_hi"], COL_RANDOM),
    ]
    lines += draw_bar_group(72.0, 126.0, 354.0, 356.0, items_a, 1.05, "Exactitud analógica", fmt_pct=True)

    # Panel B: Noise curve
    lines += draw_panel_box(385, 72, 350, 344, "B. Límite de ruido (Riemann, K=12)")
    ns = confirm["suites"]["I4"]["noise_sweep"]
    sigmas = ["0.0000", "0.0010", "0.0030", "0.0080", "0.0150", "0.0300"]
    x0, y0, x1, y1 = 440.0, 126.0, 718.0, 356.0
    for i in range(5):
        py = map_y(i * 0.25, 0.0, 1.05, y0, y1)
        lines.append(f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x1:.1f}" y2="{py:.1f}" stroke="{COL_GRID}" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" font-size="10" text-anchor="end" fill="{COL_MUTED}">{i*25}%</text>')
    lines.append(f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{COL_MUTED}"/>')

    pts_n = []
    for idx, s in enumerate(sigmas):
        px = x0 + (idx / (len(sigmas) - 1)) * (x1 - x0)
        st = ns[s]["exact_acc"]
        py = map_y(st["mean"], 0.0, 1.05, y0, y1)
        ylo = map_y(st["ci95_lo"], 0.0, 1.05, y0, y1)
        yhi = map_y(st["ci95_hi"], 0.0, 1.05, y0, y1)
        pts_n.append((px, py))
        lines.append(f'<line x1="{px:.1f}" y1="{ylo:.1f}" x2="{px:.1f}" y2="{yhi:.1f}" stroke="{COL_TEXT}" stroke-width="1.5"/>')
        lines.append(f'<text x="{px:.1f}" y="{y1 + 16:.1f}" font-size="9.5" text-anchor="middle" fill="{COL_MUTED}">{float(s):.3f}</text>')
    lines.append(f'<polyline fill="none" stroke="{COL_RIEMANN}" stroke-width="2.4" points="{polyline_path(pts_n)}"/>')
    for idx, (px, py) in enumerate(pts_n):
        val = ns[sigmas[idx]]["exact_acc"]["mean"]
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{COL_RIEMANN}"/>')
        lines.append(f'<text x="{px:.1f}" y="{py - 9:.1f}" font-size="9.5" font-weight="700" text-anchor="middle" fill="{COL_TEXT}">{100*val:.0f}%</text>')
    lines.append(f'<text x="{(x0 + x1)/2:.1f}" y="{y1 + 36:.1f}" font-size="10.5" text-anchor="middle" fill="{COL_MUTED}">Ruido de fase sigma_eta</text>')

    # Panel C: Capacity curve K=8, 12, 18
    lines += draw_panel_box(750, 72, 350, 344, "C. Límite de capacidad 1D (sigma=0.003)")
    cap = confirm["suites"]["I4"]["capacity_sweep"]
    items_c = [
        (
            f"K=8\n(du={cap['8']['min_log_gap']:.3f})",
            cap["8"]["exact_acc"]["mean"],
            cap["8"]["exact_acc"]["ci95_lo"],
            cap["8"]["exact_acc"]["ci95_hi"],
            "#0ea5e9",
        ),
        (
            f"K=12\n(du={cap['12']['min_log_gap']:.3f})",
            cap["12"]["exact_acc"]["mean"],
            cap["12"]["exact_acc"]["ci95_lo"],
            cap["12"]["exact_acc"]["ci95_hi"],
            COL_RIEMANN,
        ),
        (
            f"K=18\n(du={cap['18']['min_log_gap']:.3f})",
            cap["18"]["exact_acc"]["mean"],
            cap["18"]["exact_acc"]["ci95_lo"],
            cap["18"]["exact_acc"]["ci95_hi"],
            COL_DYSON,
        ),
    ]
    lines += draw_bar_group(804.0, 126.0, 1084.0, 356.0, items_c, 1.05, "Exactitud de par exacto", fmt_pct=True)

    lines.append("</svg>")
    (FIG_DIR / "fig4_analogy_and_limits.svg").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    summary_path = DOCS_DIR / "summary_confirm.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing {summary_path}; run summarize.py first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    confirm = summary["CONFIRM"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    render_fig1(confirm)
    render_fig2(confirm)
    render_fig3(confirm)
    render_fig4(confirm)
    print(f"Rendered 4 SVG figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
