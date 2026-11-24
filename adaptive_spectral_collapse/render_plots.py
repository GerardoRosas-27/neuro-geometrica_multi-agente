#!/usr/bin/env python3
"""Pure-stdlib SVG figure generator for `exp/adaptive-spectral-collapse`.

Generates 4 self-contained SVG figures in `docs/adaptive_spectral_collapse_v1/figures/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DOCS_DIR = REPO_ROOT / "docs" / "adaptive_spectral_collapse_v1"
FIG_DIR = DOCS_DIR / "figures"
TRACE_DIR = DOCS_DIR / "raw" / "confirm" / "seed_0xD100"

PALETTE = {
    "adaptive_spectral": "#059669",
    "riemann_fixed": "#2563eb",
    "dyson_gue_fixed": "#0284c7",
    "qmc_halton_fixed": "#0891b2",
    "fourier_fixed": "#7c3aed",
    "poisson_fixed": "#d97706",
    "mlp_conventional": "#dc2626",
}

SHORT_LABELS = {
    "adaptive_spectral": "Adapt.\nEspectral",
    "riemann_fixed": "Riemann\nFijo",
    "dyson_gue_fixed": "Dyson\nGUE",
    "qmc_halton_fixed": "QMC\nHalton",
    "fourier_fixed": "Fourier\nFijo",
    "poisson_fixed": "Poisson\nAleat.",
    "mlp_conventional": "MLP\nConvenc.",
}

ARMS_ORDER = [
    "adaptive_spectral",
    "riemann_fixed",
    "dyson_gue_fixed",
    "qmc_halton_fixed",
    "fourier_fixed",
    "poisson_fixed",
    "mlp_conventional",
]

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
        f'<text x="24" y="32" font-size="17" font-weight="700" fill="{COL_TEXT}">{title}</text>',
        f'<text x="24" y="52" font-size="11.5" fill="{COL_MUTED}">{subtitle}</text>',
    ]


def panel_box(x: float, y: float, w: float, h: float, title: str) -> list[str]:
    return [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="{COL_PANEL}" stroke="{COL_GRID}"/>',
        f'<text x="{x + 14:.1f}" y="{y + 23:.1f}" font-size="12.5" font-weight="700" fill="{COL_TEXT}">{title}</text>',
    ]


def map_x(val: float, lo: float, hi: float, x0: float, x1: float) -> float:
    return x0 + (val - lo) / max(hi - lo, 1e-12) * (x1 - x0)


def map_y(val: float, lo: float, hi: float, y0: float, y1: float) -> float:
    return y1 - (val - lo) / max(hi - lo, 1e-12) * (y1 - y0)


def draw_bars(
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
    for i in range(5):
        yv = y_max * i / 4.0
        py = map_y(yv, 0.0, y_max, y0, y1)
        out.append(f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x1:.1f}" y2="{py:.1f}" stroke="{COL_GRID}" stroke-dasharray="3,3"/>')
        lbl = f"{100*yv:.0f}%" if fmt_pct else f"{yv:.2f}"
        out.append(f'<text x="{x0 - 7:.1f}" y="{py + 3.5:.1f}" font-size="9.5" text-anchor="end" fill="{COL_MUTED}">{lbl}</text>')
    out.append(f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{COL_MUTED}"/>')
    out.append(
        f'<text x="{x0 - 38:.1f}" y="{(y0 + y1)/2:.1f}" font-size="10.5" fill="{COL_MUTED}" '
        f'text-anchor="middle" transform="rotate(-90 {x0 - 38:.1f} {(y0 + y1)/2:.1f})">{y_label}</text>'
    )

    n = len(items)
    slot = (x1 - x0) / n
    bw = min(40.0, slot * 0.62)
    for idx, (label, val, lo, hi, col) in enumerate(items):
        cx = x0 + (idx + 0.5) * slot
        bx = cx - bw / 2.0
        top = map_y(min(y_max, val), 0.0, y_max, y0, y1)
        bh = max(1.0, y1 - top)
        out.append(f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3.5" fill="{col}" opacity="0.90"/>')
        ylo = map_y(max(0.0, lo), 0.0, y_max, y0, y1)
        yhi = map_y(min(y_max, hi), 0.0, y_max, y0, y1)
        out.append(f'<line x1="{cx:.1f}" y1="{ylo:.1f}" x2="{cx:.1f}" y2="{yhi:.1f}" stroke="{COL_TEXT}" stroke-width="1.4"/>')
        out.append(f'<line x1="{cx - 4:.1f}" y1="{ylo:.1f}" x2="{cx + 4:.1f}" y2="{ylo:.1f}" stroke="{COL_TEXT}" stroke-width="1.4"/>')
        out.append(f'<line x1="{cx - 4:.1f}" y1="{yhi:.1f}" x2="{cx + 4:.1f}" y2="{yhi:.1f}" stroke="{COL_TEXT}" stroke-width="1.4"/>')
        vtxt = f"{100*val:.1f}%" if fmt_pct else f"{val:.2f}"
        out.append(f'<text x="{cx:.1f}" y="{min(yhi, top) - 5:.1f}" font-size="9.2" font-weight="700" text-anchor="middle" fill="{COL_TEXT}">{vtxt}</text>')
        for li, part in enumerate(label.split("\n")):
            out.append(f'<text x="{cx:.1f}" y="{y1 + 14 + 11*li:.1f}" font-size="9" text-anchor="middle" fill="{COL_TEXT}">{part}</text>')
    return out


def render_fig1(confirm: dict) -> None:
    w, h = 1080, 440
    lines = svg_header(
        w,
        h,
        "Figura 1 — Entrada Desconocida, Autoorganización del Campo y Abducción OOD (Suite A1, CONFIRM N=16)",
        "K=24 conceptos opacos en R^16 (0 primos, 0 ceros de Riemann en Adaptive Spectral) · Evaluación en 166 pares nunca vistos",
    )

    # Panel A: Learned 1D wave field W_0(u) vs fixed baselines
    lines += panel_box(20, 70, 460, 348, "A. Campo continuo aprendido W_0(u) sobre estados autoorganizados")
    by_arm: dict[str, list[tuple[float, float]]] = {}
    with (TRACE_DIR / "trace_1d_wave.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_arm.setdefault(r["arm"], []).append((float(r["u_coord"]), float(r["w0_val"])))
    states_c0 = []
    with (TRACE_DIR / "trace_states_freqs.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "state" and r["channel"] == "0":
                states_c0.append(float(r["value"]))

    x0, y0, x1, y1 = 68.0, 118.0, 462.0, 356.0
    u_lo, u_hi = -2.15, 2.15
    y_lo, y_hi = -2.0, 2.8
    for yv in [-1.5, -0.5, 0.0, 1.0, 2.0]:
        py = map_y(yv, y_lo, y_hi, y0, y1)
        st = COL_MUTED if yv == 0.0 else COL_GRID
        da = "" if yv == 0.0 else ' stroke-dasharray="3,3"'
        lines.append(f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x1:.1f}" y2="{py:.1f}" stroke="{st}"{da}/>')
        lines.append(f'<text x="{x0 - 7:.1f}" y="{py + 3.5:.1f}" font-size="9.5" text-anchor="end" fill="{COL_MUTED}">{yv:.1f}</text>')

    # Learned state markers along top
    for zk in states_c0:
        px = map_x(zk, u_lo, u_hi, x0, x1)
        lines.append(f'<line x1="{px:.1f}" y1="{y0:.1f}" x2="{px:.1f}" y2="{y1:.1f}" stroke="#94a3b8" opacity="0.22"/>')

    for arm, sw in [("poisson_fixed", 1.2), ("fourier_fixed", 1.3), ("adaptive_spectral", 2.2)]:
        pts = " ".join(
            f"{map_x(u, u_lo, u_hi, x0, x1):.2f},{map_y(max(y_lo, min(y_hi, v)), y_lo, y_hi, y0, y1):.2f}"
            for u, v in by_arm[arm]
        )
        lines.append(f'<polyline fill="none" stroke="{PALETTE[arm]}" stroke-width="{sw}" points="{pts}"/>')

    for idx, (arm, lbl) in enumerate([("adaptive_spectral", "Adapt. Espectral"), ("fourier_fixed", "Fourier Fijo"), ("poisson_fixed", "Poisson Fijo")]):
        lx = x0 + 6 + idx * 128
        lines.append(f'<rect x="{lx:.1f}" y="{y0 + 3:.1f}" width="11" height="4" fill="{PALETTE[arm]}"/>')
        lines.append(f'<text x="{lx + 15:.1f}" y="{y0 + 8:.1f}" font-size="9.5" fill="{COL_TEXT}">{lbl}</text>')
    lines.append(f'<text x="{(x0 + x1)/2:.1f}" y="{y1 + 28:.1f}" font-size="10.5" text-anchor="middle" fill="{COL_MUTED}">Coordenada aprendida u_0 (líneas grises = 24 estados descubiertos)</text>')

    # Panel B: 7-arm unseen abduction bar chart
    lines += panel_box(496, 70, 564, 348, "B. Abducción exacta sobre 166 pares nunca vistos (sigma_x = 0.025, IC 95%)")
    a1 = confirm["suites"]["A1"]
    items = [
        (SHORT_LABELS[a], a1[a]["unseen_abduction_acc"]["mean"], a1[a]["unseen_abduction_acc"]["ci95_lo"], a1[a]["unseen_abduction_acc"]["ci95_hi"], PALETTE[a])
        for a in ARMS_ORDER
    ]
    lines += draw_bars(552.0, 122.0, 1044.0, 356.0, items, 1.05, "Exactitud abducción OOD", fmt_pct=True)

    lines.append("</svg>")
    (FIG_DIR / "fig1_adaptive_field_and_abduction.svg").write_text("\n".join(lines), encoding="utf-8")


def render_fig2(confirm: dict) -> None:
    w, h = 1120, 440
    lines = svg_header(
        w,
        h,
        "Figura 2 — Generalización 'Del Otro Lado' sobre 166 Pares Nunca Vistos (Suite A2, CONFIRM N=16)",
        "Izquierda: error predictivo no lineal (RMSE menor es mejor) · Derecha: exactitud contrafactual do(c_a -> c_new) hacia pares inéditos",
    )
    a2 = confirm["suites"]["A2"]

    lines += panel_box(20, 70, 532, 348, "A. Predicción no lineal en pares nunca vistos (unseen_pred_rmse ↓)")
    max_r = max(a2[a]["unseen_pred_rmse"]["ci95_hi"] for a in ARMS_ORDER) * 1.15
    items_a = [
        (SHORT_LABELS[a], a2[a]["unseen_pred_rmse"]["mean"], a2[a]["unseen_pred_rmse"]["ci95_lo"], a2[a]["unseen_pred_rmse"]["ci95_hi"], PALETTE[a])
        for a in ARMS_ORDER
    ]
    lines += draw_bars(76.0, 122.0, 536.0, 356.0, items_a, max_r, "RMSE predictivo OOD (↓)", fmt_pct=False)

    lines += panel_box(568, 70, 532, 348, "B. Contrafactual do(c_a -> c_new) hacia pares inéditos (cf_exact_acc ↑)")
    items_b = [
        (SHORT_LABELS[a], a2[a]["cf_exact_acc"]["mean"], a2[a]["cf_exact_acc"]["ci95_lo"], a2[a]["cf_exact_acc"]["ci95_hi"], PALETTE[a])
        for a in ARMS_ORDER
    ]
    lines += draw_bars(624.0, 122.0, 1084.0, 356.0, items_b, 1.05, "Exactitud contrafactual OOD (↑)", fmt_pct=True)

    lines.append("</svg>")
    (FIG_DIR / "fig2_unseen_prediction_and_counterfactuals.svg").write_text("\n".join(lines), encoding="utf-8")


def render_fig3(confirm: dict) -> None:
    w, h = 1120, 440
    lines = svg_header(
        w,
        h,
        "Figura 3 — Analogía Composicional OOD y Curva Honesta de Estrés de Ruido (Suite A3, CONFIRM N=16)",
        "Izquierda: 100 analogías A:B::C:? hacia destinos en UNSEEN_PAIRS · Derecha: degradación real con ruido sensorial sigma_x hasta 0.100",
    )
    a3 = confirm["suites"]["A3"]

    lines += panel_box(20, 70, 548, 348, "A. Transferencia analógica A:B::C:? hacia pares nunca vistos (IC 95%)")
    items_a = [
        (SHORT_LABELS[a], a3["analogy"][a]["exact_acc"]["mean"], a3["analogy"][a]["exact_acc"]["ci95_lo"], a3["analogy"][a]["exact_acc"]["ci95_hi"], PALETTE[a])
        for a in ARMS_ORDER
    ]
    lines += draw_bars(76.0, 122.0, 552.0, 356.0, items_a, 1.05, "Exactitud analógica OOD", fmt_pct=True)

    lines += panel_box(584, 70, 516, 348, "B. Degradación honesta bajo ruido sensorial sigma_x (166 pares OOD)")
    x0, y0, x1, y1 = 640.0, 122.0, 1080.0, 356.0
    sigmas = ["0.0000", "0.0250", "0.0500", "0.1000"]
    for i in range(5):
        py = map_y(i * 0.25, 0.0, 1.05, y0, y1)
        lines.append(f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x1:.1f}" y2="{py:.1f}" stroke="{COL_GRID}" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{x0 - 8:.1f}" y="{py + 3.5:.1f}" font-size="9.5" text-anchor="end" fill="{COL_MUTED}">{i*25}%</text>')
    lines.append(f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{COL_MUTED}"/>')

    for idx, s in enumerate(sigmas):
        px = x0 + (idx / (len(sigmas) - 1)) * (x1 - x0)
        lines.append(f'<text x="{px:.1f}" y="{y1 + 16:.1f}" font-size="10" text-anchor="middle" fill="{COL_MUTED}">{float(s):.3f}</text>')
    lines.append(f'<text x="{(x0 + x1)/2:.1f}" y="{y1 + 34:.1f}" font-size="10.5" text-anchor="middle" fill="{COL_MUTED}">Ruido sensorial sigma_x</text>')

    sweep_arms = ["adaptive_spectral", "riemann_fixed", "fourier_fixed", "mlp_conventional"]
    for idx_a, arm in enumerate(sweep_arms):
        pts = []
        for idx, s in enumerate(sigmas):
            px = x0 + (idx / (len(sigmas) - 1)) * (x1 - x0)
            st = a3["noise_sweep"][arm][s]["exact_acc"]
            py = map_y(st["mean"], 0.0, 1.05, y0, y1)
            pts.append((px, py))
            ylo = map_y(st["ci95_lo"], 0.0, 1.05, y0, y1)
            yhi = map_y(st["ci95_hi"], 0.0, 1.05, y0, y1)
            lines.append(f'<line x1="{px:.1f}" y1="{ylo:.1f}" x2="{px:.1f}" y2="{yhi:.1f}" stroke="{PALETTE[arm]}" stroke-width="1.3" opacity="0.7"/>')
        pstr = " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
        sw = 2.4 if arm == "adaptive_spectral" else 1.8
        lines.append(f'<polyline fill="none" stroke="{PALETTE[arm]}" stroke-width="{sw}" points="{pstr}"/>')
        for px, py in pts:
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{PALETTE[arm]}"/>')
        lx = x0 + 8 + idx_a * 108
        lines.append(f'<rect x="{lx:.1f}" y="{y0 + 3:.1f}" width="10" height="4" fill="{PALETTE[arm]}"/>')
        lines.append(f'<text x="{lx + 14:.1f}" y="{y0 + 8:.1f}" font-size="9.2" fill="{COL_TEXT}">{SHORT_LABELS[arm].replace(chr(10), " ")}</text>')

    lines.append("</svg>")
    (FIG_DIR / "fig3_analogy_and_noise_stress.svg").write_text("\n".join(lines), encoding="utf-8")


def render_fig4(confirm: dict) -> None:
    w, h = 1120, 440
    lines = svg_header(
        w,
        h,
        "Figura 4 — Aprendizaje Continuo Autónomo de 4 Conceptos Nuevos (Suite A4, K: 24 -> 28, Few-Shot)",
        "4 conceptos inéditos vistos en solo 4 pares c/u · Izquierda: generalización zero-shot a 86 pares inéditos · Derecha: retención sin olvido",
    )
    a4 = confirm["suites"]["A4"]

    lines += panel_box(20, 70, 532, 348, "A. Generalización zero-shot en 86 pares nuevos nunca vistos (novel_all_acc)")
    items_a = [
        (SHORT_LABELS[a], a4[a]["novel_all_acc"]["mean"], a4[a]["novel_all_acc"]["ci95_lo"], a4[a]["novel_all_acc"]["ci95_hi"], PALETTE[a])
        for a in ARMS_ORDER
    ]
    lines += draw_bars(76.0, 122.0, 536.0, 356.0, items_a, 1.05, "Exactitud pares nuevos OOD", fmt_pct=True)

    lines += panel_box(568, 70, 532, 348, "B. Retención sobre 166 pares antiguos nunca vistos (old_unseen_acc)")
    items_b = [
        (SHORT_LABELS[a], a4[a]["old_unseen_acc"]["mean"], a4[a]["old_unseen_acc"]["ci95_lo"], a4[a]["old_unseen_acc"]["ci95_hi"], PALETTE[a])
        for a in ARMS_ORDER
    ]
    lines += draw_bars(624.0, 122.0, 1084.0, 356.0, items_b, 1.05, "Retención pares antiguos", fmt_pct=True)

    lines.append("</svg>")
    (FIG_DIR / "fig4_continual_novel_concepts.svg").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    summary_path = DOCS_DIR / "summary_confirm.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing {summary_path}; run summarize.py first.")
    confirm = json.loads(summary_path.read_text(encoding="utf-8"))["CONFIRM"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    render_fig1(confirm)
    render_fig2(confirm)
    render_fig3(confirm)
    render_fig4(confirm)
    print(f"Rendered 4 SVG figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
