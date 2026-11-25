#!/usr/bin/env python3
"""Generate 4 crisp publication-ready SVG figures for docs/wave_collision_v1/figures/."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "wave_collision_v1"
FIGS = DOCS / "figures"

BG = "#0f172a"
PANEL = "#1e293b"
GRID = "#334155"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
TEAL = "#14b8a6"
AMBER = "#f59e0b"
VIOLET = "#a78bfa"
ROSE = "#fb7185"
SKY = "#38bdf8"
SLATE = "#64748b"
EMERALD = "#34d399"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_header(w: int, h: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{BG}" rx="12"/>',
        f'<text x="28" y="34" fill="{TEXT}" font-family="system-ui,sans-serif" font-size="18" font-weight="700">{esc(title)}</text>',
        f'<text x="28" y="54" fill="{MUTED}" font-family="system-ui,sans-serif" font-size="12">{esc(subtitle)}</text>',
    ]


def draw_hbar_panel(
    out: list[str], x: float, y: float, w: float, h: float, title: str,
    rows: list[tuple[str, float, float, float, str]], max_val: float, unit: str = "%", scale: float = 100.0,
) -> None:
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{PANEL}" rx="8" stroke="{GRID}"/>')
    out.append(f'<text x="{x+14}" y="{y+22}" fill="{TEXT}" font-family="system-ui,sans-serif" font-size="13" font-weight="600">{esc(title)}</text>')
    label_w = 158
    bar_x = x + label_w
    bar_w = w - label_w - 64
    top_y = y + 38
    row_h = (h - 52) / max(1, len(rows))

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = bar_x + frac * bar_w
        out.append(f'<line x1="{gx:.1f}" y1="{top_y-4}" x2="{gx:.1f}" y2="{y+h-16}" stroke="{GRID}" stroke-dasharray="3,3"/>')
        lbl = f"{frac*max_val*scale:.0f}{unit}" if unit == "%" else f"{frac*max_val*scale:.2f}"
        out.append(f'<text x="{gx:.1f}" y="{y+h-4}" fill="{MUTED}" font-family="system-ui,sans-serif" font-size="9" text-anchor="middle">{lbl}</text>')

    for idx, (label, mean, lo, hi, color) in enumerate(rows):
        cy = top_y + idx * row_h + row_h * 0.5
        bh = min(17.0, row_h * 0.62)
        bw = max(1.0, min(1.0, mean / max_val) * bar_w)
        out.append(f'<text x="{bar_x-8}" y="{cy+4:.1f}" fill="{TEXT}" font-family="system-ui,sans-serif" font-size="10.5" text-anchor="end">{esc(label)}</text>')
        out.append(f'<rect x="{bar_x:.1f}" y="{cy-bh/2:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" rx="3" opacity="0.92"/>')
        x_lo = bar_x + max(0.0, min(1.0, lo / max_val)) * bar_w
        x_hi = bar_x + max(0.0, min(1.0, hi / max_val)) * bar_w
        out.append(f'<line x1="{x_lo:.1f}" y1="{cy:.1f}" x2="{x_hi:.1f}" y2="{cy:.1f}" stroke="{TEXT}" stroke-width="1.5"/>')
        out.append(f'<line x1="{x_lo:.1f}" y1="{cy-4:.1f}" x2="{x_lo:.1f}" y2="{cy+4:.1f}" stroke="{TEXT}" stroke-width="1.3"/>')
        out.append(f'<line x1="{x_hi:.1f}" y1="{cy-4:.1f}" x2="{x_hi:.1f}" y2="{cy+4:.1f}" stroke="{TEXT}" stroke-width="1.3"/>')
        val_txt = f"{mean*scale:.1f}{unit}" if unit == "%" else f"{mean*scale:.3f}"
        out.append(f'<text x="{bar_x+bar_w+6:.1f}" y="{cy+4:.1f}" fill="{TEXT}" font-family="system-ui,sans-serif" font-size="10.5" font-weight="600">{val_txt}</text>')


def draw_curve_panel(
    out: list[str], x: float, y: float, w: float, h: float, title: str,
    budgets: list[int], series: list[tuple[str, list[float], str]],
) -> None:
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{PANEL}" rx="8" stroke="{GRID}"/>')
    out.append(f'<text x="{x+14}" y="{y+22}" fill="{TEXT}" font-family="system-ui,sans-serif" font-size="13" font-weight="600">{esc(title)}</text>')
    px, py = x + 46, y + 40
    pw, ph = w - 66, h - 102
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        gy = py + ph * (1.0 - frac)
        out.append(f'<line x1="{px}" y1="{gy:.1f}" x2="{px+pw}" y2="{gy:.1f}" stroke="{GRID}" stroke-dasharray="3,3"/>')
        out.append(f'<text x="{px-6}" y="{gy+4:.1f}" fill="{MUTED}" font-family="system-ui,sans-serif" font-size="9.5" text-anchor="end">{frac*100:.0f}%</text>')
    xs = [px + pw * (i / max(1, len(budgets) - 1)) for i in range(len(budgets))]
    for bx, b in zip(xs, budgets):
        out.append(f'<text x="{bx:.1f}" y="{py+ph+16:.1f}" fill="{MUTED}" font-family="system-ui,sans-serif" font-size="10" text-anchor="middle">N={b}</text>')
    for label, vals, color in series:
        pts = " ".join(f"{xs[i]:.1f},{py + ph*(1.0 - vals[i]):.1f}" for i in range(len(vals)))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for i, v in enumerate(vals):
            cy = py + ph * (1.0 - v)
            out.append(f'<circle cx="{xs[i]:.1f}" cy="{cy:.1f}" r="4" fill="{color}"/>')
    ly = py + ph + 34
    for idx, (label, _, color) in enumerate(series):
        lx = x + 16 + (idx % 3) * 140
        row_y = ly + (idx // 3) * 16
        out.append(f'<rect x="{lx}" y="{row_y-7}" width="10" height="4" fill="{color}" rx="1"/>')
        out.append(f'<text x="{lx+14}" y="{row_y-2}" fill="{TEXT}" font-family="system-ui,sans-serif" font-size="10">{esc(label)}</text>')


def fig1(s: dict) -> None:
    out = svg_header(980, 410,
        "Suite T1 — Red de 768 Ondas Colisionantes Auto-Calibrables en Dominio A (CONFIRM, N=16)",
        "0 matrices densas W_pair · 100% frecuencias y fases (3,328 params, 0 Riemann) vs espectros fijos y MLP (3,832 params)")
    t1 = s["suite_t1"]
    order = [
        ("adaptive_wave", "768 Ondas Auto-Cal", TEAL),
        ("riemann_fixed", "Riemann-256 Fija", VIOLET),
        ("dyson_fixed", "Dyson GUE Fija", SKY),
        ("fourier_fixed", "Fourier 4D Fija", AMBER),
        ("poisson_fixed", "Poisson Fija", SLATE),
        ("mlp_conventional", "MLP (3,832 params)", ROSE),
    ]
    rows_acc = [(lbl, t1[k]["cls_acc"]["mean"], t1[k]["cls_acc"]["ci_lo"], t1[k]["cls_acc"]["ci_hi"], col) for k, lbl, col in order]
    rows_abd = [(lbl, t1[k]["abd_top1"]["mean"], t1[k]["abd_top1"]["ci_lo"], t1[k]["abd_top1"]["ci_hi"], col) for k, lbl, col in order]
    draw_hbar_panel(out, 24, 72, 458, 315, "Inferencia Relacional Inédita (332 pares, 4 clases)", rows_acc, 1.0)
    draw_hbar_panel(out, 498, 72, 458, 315, "Abducción Pasado × Futuro (Top-1 entre 23 candidatos)", rows_abd, 1.0)
    out.append("</svg>")
    (FIGS / "fig1_wave768_domain_a.svg").write_text("\n".join(out))


def fig2(s: dict) -> None:
    out = svg_header(980, 460,
        "Suite T2 — Disección Causal: Frecuencias, Fases, Colisión Primaria y Colisión Secundaria",
        "Efecto sobre exactitud inédita y abducción al desactivar cada mecanismo interferométrico (CONFIRM, N=16)")
    t2 = s["suite_t2"]
    order = [
        ("full_adaptive", "768 Ondas (ref)", TEAL),
        ("ablate_no_stage2", "Sin Colisión Etapa 2", EMERALD),
        ("train_zero_phase", "Fase: φ = 0 (train)", SKY),
        ("ablate_freq_riemann", "Freq: Riemann fija", VIOLET),
        ("ablate_freq_fourier", "Freq: Fourier fija", AMBER),
        ("ablate_freq_poisson", "Freq: Poisson fija", SLATE),
        ("ablate_pairing_additive", "Sin Colisión (aditivo)", "#f97316"),
        ("ablate_pairing_permuted", "Emparej. Permutado", "#ef4444"),
        ("ablate_phase_scramble", "Fase Aleatorizada", ROSE),
    ]
    rows_acc = [(lbl, t2[k]["cls_acc"]["mean"], t2[k]["cls_acc"]["ci_lo"], t2[k]["cls_acc"]["ci_hi"], col) for k, lbl, col in order]
    rows_abd = [(lbl, t2[k]["abd_top1"]["mean"], t2[k]["abd_top1"]["ci_lo"], t2[k]["abd_top1"]["ci_hi"], col) for k, lbl, col in order]
    draw_hbar_panel(out, 24, 72, 458, 366, "Exactitud Inédita en Dominio A (4 clases)", rows_acc, 1.0)
    draw_hbar_panel(out, 498, 72, 458, 366, "Abducción Pasado × Futuro Top-1", rows_abd, 1.0)
    out.append("</svg>")
    (FIGS / "fig2_wave768_causal_dissection.svg").write_text("\n".join(out))


def fig3(s: dict) -> None:
    out = svg_header(980, 490,
        "Suite T3 — Transferencia Auto-Calibrable A → B (Osciladores Físicos R^12, 3,328 Ondas Congeladas)",
        "219 params adaptables (171 interfaz + 48 auto-cal resonante) vs MLP (+48 FiLM), espectros fijos y scratch (CONFIRM, N=16)")
    t3 = s["suite_t3"]
    order = [
        ("transfer_autocal_A", "Wave Auto-Cal A→B", TEAL),
        ("transfer_frozen_A", "Wave Congelado A→B", EMERALD),
        ("riemann_A", "Riemann-256 A→B", VIOLET),
        ("dyson_A", "Dyson A→B", SKY),
        ("fourier_A", "Fourier A→B", AMBER),
        ("poisson_A", "Poisson A→B", SLATE),
        ("transfer_mlp_A", "MLP +48 FiLM A→B", ROSE),
        ("neg_ctrl_noise_A", "Ctrl: Ruido en A", "#f97316"),
        ("neg_ctrl_phase_scramble", "Ctrl: Fase rota", "#ec4899"),
        ("untrained_poisson", "Ctrl: Sin entrenar", "#64748b"),
        ("scratch_wave", "Scratch 768 Ondas B", "#22d3ee"),
        ("scratch_mlp", "Scratch MLP B", "#fb923c"),
    ]
    rows = [(lbl, t3[f"48|{k}"]["cls_acc"]["mean"], t3[f"48|{k}"]["cls_acc"]["ci_lo"], t3[f"48|{k}"]["cls_acc"]["ci_hi"], col) for k, lbl, col in order]
    draw_hbar_panel(out, 24, 72, 492, 396, "Exactitud en 258 Pares Físicos Inéditos (N_few = 48)", rows, 1.0)
    budgets = [24, 48, 96]
    curves = [
        ("Wave Auto-Cal A→B", [t3[f"{b}|transfer_autocal_A"]["cls_acc"]["mean"] for b in budgets], TEAL),
        ("Riemann-256 A→B", [t3[f"{b}|riemann_A"]["cls_acc"]["mean"] for b in budgets], VIOLET),
        ("MLP +48 FiLM A→B", [t3[f"{b}|transfer_mlp_A"]["cls_acc"]["mean"] for b in budgets], ROSE),
        ("Scratch Wave B", [t3[f"{b}|scratch_wave"]["cls_acc"]["mean"] for b in budgets], "#22d3ee"),
        ("Scratch MLP B", [t3[f"{b}|scratch_mlp"]["cls_acc"]["mean"] for b in budgets], AMBER),
        ("Ctrl Ruido A", [t3[f"{b}|neg_ctrl_noise_A"]["cls_acc"]["mean"] for b in budgets], SLATE),
    ]
    draw_curve_panel(out, 530, 72, 426, 396, "Eficiencia Muestral Few-Shot (N_few ∈ {24, 48, 96})", budgets, curves)
    out.append("</svg>")
    (FIGS / "fig3_wave768_transfer_b.svg").write_text("\n".join(out))


def fig4(s: dict) -> None:
    out = svg_header(980, 490,
        "Suite T4 — Transferencia Auto-Calibrable A → C (Grafo Relacional Z5×{0..3} R^16)",
        "184 params adaptables (136 interfaz + 48 auto-cal resonante) vs MLP (+48 FiLM), espectros fijos y scratch (CONFIRM, N=16)")
    t4 = s["suite_t4"]
    order = [
        ("transfer_autocal_A", "Wave Auto-Cal A→C", TEAL),
        ("transfer_frozen_A", "Wave Congelado A→C", EMERALD),
        ("riemann_A", "Riemann-256 A→C", VIOLET),
        ("dyson_A", "Dyson A→C", SKY),
        ("fourier_A", "Fourier A→C", AMBER),
        ("poisson_A", "Poisson A→C", SLATE),
        ("transfer_mlp_A", "MLP +48 FiLM A→C", ROSE),
        ("neg_ctrl_noise_A", "Ctrl: Ruido en A", "#f97316"),
        ("neg_ctrl_phase_scramble", "Ctrl: Fase rota", "#ec4899"),
        ("untrained_poisson", "Ctrl: Sin entrenar", "#64748b"),
        ("scratch_wave", "Scratch 768 Ondas C", "#22d3ee"),
        ("scratch_mlp", "Scratch MLP C", "#fb923c"),
    ]
    rows = [(lbl, t4[f"60|{k}"]["cls_acc"]["mean"], t4[f"60|{k}"]["cls_acc"]["ci_lo"], t4[f"60|{k}"]["cls_acc"]["ci_hi"], col) for k, lbl, col in order]
    draw_hbar_panel(out, 24, 72, 492, 396, "Exactitud en 320 Arcos Inéditos (N_few = 60)", rows, 1.0)
    budgets = [30, 60, 120]
    curves = [
        ("Wave Auto-Cal A→C", [t4[f"{b}|transfer_autocal_A"]["cls_acc"]["mean"] for b in budgets], TEAL),
        ("Riemann-256 A→C", [t4[f"{b}|riemann_A"]["cls_acc"]["mean"] for b in budgets], VIOLET),
        ("MLP +48 FiLM A→C", [t4[f"{b}|transfer_mlp_A"]["cls_acc"]["mean"] for b in budgets], ROSE),
        ("Scratch Wave C", [t4[f"{b}|scratch_wave"]["cls_acc"]["mean"] for b in budgets], "#22d3ee"),
        ("Scratch MLP C", [t4[f"{b}|scratch_mlp"]["cls_acc"]["mean"] for b in budgets], AMBER),
        ("Ctrl Fase Rota", [t4[f"{b}|neg_ctrl_phase_scramble"]["cls_acc"]["mean"] for b in budgets], "#ec4899"),
    ]
    draw_curve_panel(out, 530, 72, 426, 396, "Eficiencia Muestral Few-Shot (N_few ∈ {30, 60, 120})", budgets, curves)
    out.append("</svg>")
    (FIGS / "fig4_wave768_transfer_c.svg").write_text("\n".join(out))


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    p = DOCS / "summary_confirm.json"
    if not p.exists():
        p = DOCS / "summary_dev.json"
    s = json.loads(p.read_text())
    fig1(s)
    fig2(s)
    fig3(s)
    fig4(s)
    print(f"Generated 4 SVG figures in {FIGS} from {p.name}.")


if __name__ == "__main__":
    main()
