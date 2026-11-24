#!/usr/bin/env python3
"""Render 4 publication-quality SVG figures for exp/orbita-adaptive-spectral (stdlib Python only)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs" / "orbita_adaptive_spectral_v1"
FIG_DIR = DOCS_DIR / "figures"

COLORS = {
    "ORBITA_ASC": "#10b981",
    "ORBITA_BASE": "#64748b",
    "ORBITA_OBS": "#475569",
    "ORBITA_LAT": "#0ea5e9",
    "ORBITA_MLP_RES": "#f59e0b",
    "ORBITA_MLP": "#f59e0b",
    "ORBITA_RIEMANN_FIXED": "#8b5cf6",
    "ORBITA_FOURIER_FIXED": "#3b82f6",
    "ORBITA_POISSON_FIXED": "#ef4444",
    "full_adaptive": "#10b981",
    "prime_harmonic_exact": "#f59e0b",
    "riemann_prime_dual": "#8b5cf6",
    "log_prime_exact": "#06b6d4",
    "composite_only": "#ef4444",
    "even_only": "#ec4899",
    "random_matched": "#64748b",
}

LABELS = {
    "ORBITA_ASC": "ÓRBITA + ASC (0 Riemann)",
    "ORBITA_BASE": "ÓRBITA Base",
    "ORBITA_OBS": "ÓRBITA + OBS (reid)",
    "ORBITA_LAT": "ÓRBITA + LAT",
    "ORBITA_MLP_RES": "ÓRBITA + MLP (1:1)",
    "ORBITA_MLP": "ÓRBITA + MLP (1:1)",
    "ORBITA_RIEMANN_FIXED": "ÓRBITA + Riemann Fijo",
    "ORBITA_FOURIER_FIXED": "ÓRBITA + Fourier Fijo",
    "ORBITA_POISSON_FIXED": "ÓRBITA + Poisson Fijo",
    "full_adaptive": "Campo Completo (100%)",
    "prime_harmonic_exact": "Solo Primos Exactos",
    "riemann_prime_dual": "Solo Ceros Riemann (Dual)",
    "log_prime_exact": "Solo log(p) Exactos",
    "composite_only": "Solo Compuestos (ctrl)",
    "even_only": "Solo Pares (ctrl)",
    "random_matched": "Aleatorio Emparejado",
}


def svg_wrap(w: int, h: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        '<style>'
        'text{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;fill:#0f172a}'
        '.t{font-size:14px;font-weight:700}.s{font-size:11px;fill:#475569}'
        '.l{font-size:10.5px;fill:#1e293b}.v{font-size:10px;font-weight:600;font-family:ui-monospace,monospace}'
        '.g{stroke:#e2e8f0;stroke-width:1}.ax{stroke:#94a3b8;stroke-width:1.2}'
        '</style>\n'
        f'<rect width="{w}" height="{h}" fill="#ffffff" rx="8"/>\n'
        + body
        + "\n</svg>\n"
    )


def hbars(
    items: list[tuple[str, float, float, float, str]],
    x0: float,
    y0: float,
    w: float,
    h: float,
    max_v: float,
    title: str,
    fmt: str = "{:.1f}%",
    scale: float = 100.0,
) -> str:
    parts = [f'<text x="{x0}" y="{y0 - 12}" class="t">{title}</text>']
    n = len(items)
    row_h = h / max(1, n)
    bar_h = min(20.0, row_h * 0.58)
    for k in range(5):
        gx = x0 + (w * k) / 4.0
        parts.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0 + h}" class="g"/>')
        parts.append(f'<text x="{gx:.1f}" y="{y0 + h + 14}" text-anchor="middle" class="s">{fmt.format(max_v * k / 4.0 * scale)}</text>')
    for i, (lbl, m, lo, hi, col) in enumerate(items):
        cy = y0 + i * row_h + row_h * 0.5
        bw = max(1.0, min(w, w * (m / max(1e-9, max_v))))
        xlo = x0 + max(0.0, min(w, w * (lo / max(1e-9, max_v))))
        xhi = x0 + max(0.0, min(w, w * (hi / max(1e-9, max_v))))
        parts.append(f'<text x="{x0 - 8}" y="{cy + 4:.1f}" text-anchor="end" class="l">{lbl}</text>')
        parts.append(f'<rect x="{x0}" y="{cy - bar_h / 2:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="{col}" rx="3" opacity="0.9"/>')
        parts.append(f'<line x1="{xlo:.1f}" y1="{cy:.1f}" x2="{xhi:.1f}" y2="{cy:.1f}" stroke="#0f172a" stroke-width="1.6"/>')
        parts.append(f'<text x="{min(x0 + w + 6, xhi + 6):.1f}" y="{cy + 4:.1f}" class="v">{fmt.format(m * scale)}</text>')
    return "\n".join(parts)


def render_fig1(data: dict) -> None:
    o1 = data["suites"]["O1"]
    arms = ["ORBITA_ASC", "ORBITA_RIEMANN_FIXED", "ORBITA_MLP_RES", "ORBITA_FOURIER_FIXED", "ORBITA_POISSON_FIXED", "ORBITA_BASE"]
    items_h16 = [(LABELS[a], o1[a]["h16"]["mean"], o1[a]["h16"]["ci95_lo"], o1[a]["h16"]["ci95_hi"], COLORS[a]) for a in arms]
    items_rel = [(LABELS[a], o1[a]["rel1"]["mean"], o1[a]["rel1"]["ci95_lo"], o1[a]["rel1"]["ci95_hi"], COLORS[a]) for a in arms]
    max_h16 = max(0.6, max(x[3] for x in items_h16) * 1.15)
    max_rel = max(0.012, max(x[3] for x in items_rel) * 1.15)

    # Panel A: E33 rollout curves h1, h4, h8, h16, h32, h64
    hs = ["h1", "h4", "h8", "h16", "h32", "h64"]
    x0, y0, pw, ph = 65.0, 55.0, 380.0, 220.0
    body = [f'<text x="24" y="26" class="t">Suite O1 — Protocolo Oficial ÓRBITA §16 ({data["phase"]}, N={data["n_seeds"]} semillas)</text>']
    body.append(f'<text x="{x0}" y="{y0 - 12}" class="s">Curva E33 Free-Run (rotate + reflect, acc h1 → h64)</text>')
    for k in range(5):
        gy = y0 + ph * (1.0 - k / 4.0)
        body.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0 + pw}" y2="{gy:.1f}" class="g"/>')
        body.append(f'<text x="{x0 - 8}" y="{gy + 4:.1f}" text-anchor="end" class="s">{25 * k}%</text>')
    for hi, hlbl in enumerate(hs):
        gx = x0 + pw * hi / (len(hs) - 1)
        body.append(f'<text x="{gx:.1f}" y="{y0 + ph + 16}" text-anchor="middle" class="s">{hlbl}</text>')
    for a in ["ORBITA_BASE", "ORBITA_POISSON_FIXED", "ORBITA_FOURIER_FIXED", "ORBITA_MLP_RES", "ORBITA_RIEMANN_FIXED", "ORBITA_ASC"]:
        pts = []
        for hi, hlbl in enumerate(hs):
            gx = x0 + pw * hi / (len(hs) - 1)
            gy = y0 + ph * (1.0 - o1[a][hlbl]["mean"])
            pts.append(f"{gx:.1f},{gy:.1f}")
        sw = "2.8" if a == "ORBITA_ASC" else ("2.2" if a == "ORBITA_BASE" else "1.6")
        dash = ' stroke-dasharray="4,3"' if a == "ORBITA_BASE" else ""
        body.append(f'<polyline fill="none" stroke="{COLORS[a]}" stroke-width="{sw}"{dash} points="{" ".join(pts)}"/>')

    body.append(hbars(items_h16, 615.0, 55.0, 250.0, 210.0, max_h16, "Exactitud E33 h16 (IC 95%)"))
    body.append(hbars(items_rel, 210.0, 330.0, 650.0, 175.0, max_rel, "Error relativo 1 paso TEST 256 (menor es mejor)", fmt="{:.4f}", scale=1.0))
    (FIG_DIR / "fig1_orbita_s16_e33.svg").write_text(svg_wrap(940, 535, "\n".join(body)), encoding="utf-8")


def render_fig2(data: dict) -> None:
    o2 = data["suites"]["O2"]
    arms = ["ORBITA_ASC", "ORBITA_RIEMANN_FIXED", "ORBITA_FOURIER_FIXED", "ORBITA_POISSON_FIXED", "ORBITA_MLP"]
    items = [(LABELS[a], o2[a]["unseen_pair_acc"]["mean"], o2[a]["unseen_pair_acc"]["ci95_lo"], o2[a]["unseen_pair_acc"]["ci95_hi"], COLORS[a]) for a in arms]
    items_t = [(LABELS[a], o2[a]["train_pair_acc"]["mean"], o2[a]["train_pair_acc"]["ci95_lo"], o2[a]["train_pair_acc"]["ci95_hi"], COLORS[a]) for a in arms]
    body = [f'<text x="24" y="26" class="t">Suite O2 — Abducción de Reglas ÓRBITA Desconocidas y 40 Pares Compuestos Inéditos ({data["phase"]})</text>']
    body.append(hbars(items, 195.0, 62.0, 260.0, 200.0, 1.0, "Pares compuestos inéditos (60.6% OOD)"))
    body.append(hbars(items_t, 635.0, 62.0, 230.0, 200.0, 1.0, "Pares de entrenamiento (brecha de sobreajuste)"))
    (FIG_DIR / "fig2_orbita_o2_compositional.svg").write_text(svg_wrap(940, 300, "\n".join(body)), encoding="utf-8")


def render_fig3(data: dict) -> None:
    o3 = data["suites"]["O3"]
    arms = ["ORBITA_ASC", "ORBITA_MLP", "ORBITA_RIEMANN_FIXED", "ORBITA_FOURIER_FIXED", "ORBITA_POISSON_FIXED", "ORBITA_LAT", "ORBITA_OBS", "ORBITA_BASE"]
    items_new = [(LABELS[a], o3[a]["new_h16"]["mean"], o3[a]["new_h16"]["ci95_lo"], o3[a]["new_h16"]["ci95_hi"], COLORS[a]) for a in arms]
    items_kn = [(LABELS[a], o3[a]["known_h16"]["mean"], o3[a]["known_h16"]["ci95_lo"], o3[a]["known_h16"]["ci95_hi"], COLORS[a]) for a in arms]
    max_n = max(0.5, max(x[3] for x in items_new) * 1.15)
    max_k = max(0.6, max(x[3] for x in items_kn) * 1.15)
    body = [f'<text x="24" y="26" class="t">Suite O3 — Aprendizaje Continuo Few-Shot de 4 Familias Nuevas No Lineales (8..11) vs Retención (0..7)</text>']
    body.append(hbars(items_new, 195.0, 62.0, 250.0, 265.0, max_n, "Nuevas 8..11: acc h16 held-out (sin etiquetas)"))
    body.append(hbars(items_kn, 635.0, 62.0, 230.0, 265.0, max_k, "Conocidas 0..7: retención acc h16 (0 olvido)"))
    (FIG_DIR / "fig3_orbita_o3_continual_new.svg").write_text(svg_wrap(940, 365, "\n".join(body)), encoding="utf-8")


def render_fig4(data: dict) -> None:
    o4a = data["suites"]["O4"]["O4a_dynamics"]
    o4b = data["suites"]["O4"]["O4b_collapse"]
    slices = ["full_adaptive", "prime_harmonic_exact", "riemann_prime_dual", "log_prime_exact", "composite_only", "even_only", "random_matched"]
    it_pair = [(LABELS[s], o4b[s]["acc16_or_pair"]["mean"], o4b[s]["acc16_or_pair"]["ci95_lo"], o4b[s]["acc16_or_pair"]["ci95_hi"], COLORS[s]) for s in slices]
    it_psr = [(LABELS[s], o4b[s]["disp_cos_or_psr"]["mean"], o4b[s]["disp_cos_or_psr"]["ci95_lo"], o4b[s]["disp_cos_or_psr"]["ci95_hi"], COLORS[s]) for s in slices]
    it_odd = [(LABELS[s], o4a[s]["odd_parity_frac"]["mean"], o4a[s]["odd_parity_frac"]["ci95_lo"], o4a[s]["odd_parity_frac"]["ci95_hi"], COLORS[s]) for s in slices]
    it_cos = [(LABELS[s], o4a[s]["disp_cos_or_psr"]["mean"], o4a[s]["disp_cos_or_psr"]["ci95_lo"], o4a[s]["disp_cos_or_psr"]["ci95_hi"], COLORS[s]) for s in slices]
    max_psr = max(3.0, max(x[3] for x in it_psr) * 1.15)
    body = [f'<text x="24" y="26" class="t">Suite O4 — Tamiz Espectral Primo: ¿Qué Inferencia Se Esconde en Frecuencias Exactas Primas? ({data["phase"]})</text>']
    body.append(hbars(it_pair, 200.0, 62.0, 240.0, 210.0, 1.0, "O4b: Abducción Pasado×Futuro (pares inéditos)"))
    body.append(hbars(it_psr, 645.0, 62.0, 225.0, 210.0, max_psr, "O4b: Supresión de ecos fantasma (PSR)", fmt="{:.2f}×", scale=1.0))
    body.append(hbars(it_odd, 200.0, 325.0, 240.0, 210.0, 1.0, "O4a: Paridad impar aislada (p>2 impar)"))
    body.append(hbars(it_cos, 645.0, 325.0, 225.0, 210.0, 1.0, "O4a: Fidelidad direccional del flujo (disp_cos)", fmt="{:.3f}", scale=1.0))
    (FIG_DIR / "fig4_prime_spectral_sieve.svg").write_text(svg_wrap(940, 565, "\n".join(body)), encoding="utf-8")


def main() -> int:
    summary = json.loads((DOCS_DIR / "summary_confirm.json").read_text(encoding="utf-8"))
    data = summary.get("CONFIRM") or summary.get("DEV")
    if not data:
        raise RuntimeError("No DEV/CONFIRM summary found")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    render_fig1(data)
    render_fig2(data)
    render_fig3(data)
    render_fig4(data)
    print(f"Rendered 4 SVG figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
