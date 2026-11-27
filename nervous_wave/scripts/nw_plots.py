#!/usr/bin/env python3
"""Generate 4 publication-ready SVG figures for docs/nervous_wave_v1/figures/."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "nervous_wave_v1"
FIGS = DOCS / "figures"

BG, PANEL, GRID, TEXT, MUTED = "#0f172a", "#1e293b", "#334155", "#f8fafc", "#94a3b8"
TEAL, AMBER, VIOLET, ROSE, SKY, SLATE = "#14b8a6", "#f59e0b", "#a78bfa", "#fb7185", "#38bdf8", "#64748b"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_header(w: int, h: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{BG}" rx="12"/>',
        f'<text x="28" y="34" fill="{TEXT}" font-family="system-ui,sans-serif" font-size="18" font-weight="700">{esc(title)}</text>',
        f'<text x="28" y="54" fill="{MUTED}" font-family="system-ui,sans-serif" font-size="12">{esc(subtitle)}</text>',
    ]


def draw_hbar(out: list[str], x: float, y: float, w: float, h: float, title: str,
              rows: list[tuple[str, float, float, float, str]], max_val: float, unit: str = "%", scale: float = 100.0) -> None:
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{PANEL}" rx="8" stroke="{GRID}"/>')
    out.append(f'<text x="{x+14}" y="{y+22}" fill="{TEXT}" font-family="system-ui,sans-serif" font-size="13" font-weight="600">{esc(title)}</text>')
    lw = 166
    bx, bw = x + lw, w - lw - 66
    ty = y + 38
    rh = (h - 54) / max(1, len(rows))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = bx + frac * bw
        out.append(f'<line x1="{gx:.1f}" y1="{ty-4}" x2="{gx:.1f}" y2="{y+h-16}" stroke="{GRID}" stroke-dasharray="3,3"/>')
        lbl = f"{frac*max_val*scale:.0f}{unit}" if unit == "%" else f"{frac*max_val*scale:.2f}"
        out.append(f'<text x="{gx:.1f}" y="{y+h-4}" fill="{MUTED}" font-family="system-ui,sans-serif" font-size="9" text-anchor="middle">{lbl}</text>')
    for idx, (label, mean, lo, hi, col) in enumerate(rows):
        cy = ty + idx * rh + rh * 0.5
        bh = min(19.0, rh * 0.62)
        barw = max(1.0, min(1.0, mean / max_val) * bw)
        out.append(f'<text x="{bx-8}" y="{cy+4:.1f}" fill="{TEXT}" font-family="system-ui,sans-serif" font-size="10.5" text-anchor="end">{esc(label)}</text>')
        out.append(f'<rect x="{bx:.1f}" y="{cy-bh/2:.1f}" width="{barw:.1f}" height="{bh:.1f}" fill="{col}" rx="3" opacity="0.92"/>')
        xlo = bx + max(0.0, min(1.0, lo / max_val)) * bw
        xhi = bx + max(0.0, min(1.0, hi / max_val)) * bw
        out.append(f'<line x1="{xlo:.1f}" y1="{cy:.1f}" x2="{xhi:.1f}" y2="{cy:.1f}" stroke="{TEXT}" stroke-width="1.5"/>')
        out.append(f'<line x1="{xlo:.1f}" y1="{cy-4:.1f}" x2="{xlo:.1f}" y2="{cy+4:.1f}" stroke="{TEXT}" stroke-width="1.3"/>')
        out.append(f'<line x1="{xhi:.1f}" y1="{cy-4:.1f}" x2="{xhi:.1f}" y2="{cy+4:.1f}" stroke="{TEXT}" stroke-width="1.3"/>')
        vt = f"{mean*scale:.1f}{unit}" if unit == "%" else f"{mean*scale:.3f}"
        out.append(f'<text x="{bx+bw+6:.1f}" y="{cy+4:.1f}" fill="{TEXT}" font-family="system-ui,sans-serif" font-size="10.5" font-weight="600">{vt}</text>')


ORDER = [
    ("nervous_system", "Sistema Nervioso (256)", TEAL),
    ("all_to_all_flat", "Global Plano (4,096)", VIOLET),
    ("nervous_no_inhibition", "Nervioso s/ Inhib. (256)", SKY),
    ("grid_local_lattice", "Retícula Local 8×8 (256)", AMBER),
    ("random_erdos_renyi", "Aleatorio ER (256)", SLATE),
    ("nervous_no_delay", "Nervioso s/ Fase (256)", ROSE),
]


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    p = DOCS / "summary_confirm.json"
    if not p.exists():
        p = DOCS / "summary_dev.json"
    s = json.loads(p.read_text())

    # Fig 1: P1 Physics
    p1 = s["suite_p1"]
    out = svg_header(980, 400, "Prueba A (Suite P1) — Propagación Física de Ondas sobre Topologías (CONFIRM, N=16)",
                     "Localización de ruta sensorial en t=1 e índice de colisión interferométrica Pasado × Futuro en hubs")
    draw_hbar(out, 24, 72, 458, 305, "Localización Sensorial de Ruta (SNR t=1)",
              [(lbl, p1[k]["localization_snr"]["mean"], p1[k]["localization_snr"]["ci_lo"], p1[k]["localization_snr"]["ci_hi"], col) for k, lbl, col in ORDER], 1.0)
    draw_hbar(out, 498, 72, 458, 305, "Coincidencia Interferométrica Pasado × Futuro (Hubs)",
              [(lbl, p1[k]["collision_coinc"]["mean"], p1[k]["collision_coinc"]["ci_lo"], p1[k]["collision_coinc"]["ci_hi"], col) for k, lbl, col in ORDER], 1.5, "", 1.0)
    out.append("</svg>")
    (FIGS / "fig1_nervous_physics.svg").write_text("\n".join(out))

    # Fig 2: P2 Inference
    p2 = s["suite_p2"]
    out = svg_header(980, 400, "Prueba A (Suite P2) — Inferencia Relacional por Propagación de Ondas (CONFIRM, N=16)",
                     "Exactitud inédita en Dominio A (continuo) y Dominio C (grafo Z5×4): 256 aristas nerviosas vs 4,096 planas")
    draw_hbar(out, 24, 72, 458, 305, "Dominio A — Exactitud Inédita (332 pares)",
              [(lbl, p2[k]["acc_a"]["mean"], p2[k]["acc_a"]["ci_lo"], p2[k]["acc_a"]["ci_hi"], col) for k, lbl, col in ORDER], 1.0)
    draw_hbar(out, 498, 72, 458, 305, "Dominio C — Grafo Discreto Z5×{0..3} (260 arcos)",
              [(lbl, p2[k]["acc_c"]["mean"], p2[k]["acc_c"]["ci_lo"], p2[k]["acc_c"]["ci_hi"], col) for k, lbl, col in ORDER], 1.0)
    out.append("</svg>")
    (FIGS / "fig2_nervous_inference.svg").write_text("\n".join(out))

    # Fig 3: P3 Resilience
    p3 = s["suite_p3"]
    out = svg_header(980, 400, "Prueba A (Suite P3) — Resiliencia ante Lesiones Axonales y Ruido Sensorial (CONFIRM, N=16)",
                     "Evaluación sin reentrenamiento: 15% axones seccionados y ruido sensorial aditivo σ=0.50")
    draw_hbar(out, 24, 72, 458, 305, "Exactitud tras Lesión Axonal 15% (8 máscaras/semilla)",
              [(lbl, p3[k]["acc_lesion15"]["mean"], p3[k]["acc_lesion15"]["ci_lo"], p3[k]["acc_lesion15"]["ci_hi"], col) for k, lbl, col in ORDER], 1.0)
    draw_hbar(out, 498, 72, 458, 305, "Exactitud bajo Ruido Sensorial (σ = 0.50)",
              [(lbl, p3[k]["acc_noise50"]["mean"], p3[k]["acc_noise50"]["ci_lo"], p3[k]["acc_noise50"]["ci_hi"], col) for k, lbl, col in ORDER], 1.0)
    out.append("</svg>")
    (FIGS / "fig3_nervous_resilience.svg").write_text("\n".join(out))

    # Fig 4: P4 Myelin Transfer
    p4 = s["suite_p4"]
    t_ord = [
        ("nervous_transfer", "Nervioso + Mielina (256)", TEAL),
        ("flat_transfer", "Global Plano + Mielina (4,096)", VIOLET),
        ("grid_transfer", "Retícula Local + Mielina (256)", AMBER),
        ("random_transfer", "Aleatorio ER + Mielina (256)", SLATE),
        ("ctrl_permuted_synapses", "Ctrl: Sinapsis Permutadas", ROSE),
    ]
    out = svg_header(980, 380, "Prueba A (Suite P4) — Transferencia Cruzada A → C con Sinapsis Congeladas (CONFIRM, N=16)",
                     "Sinapsis 100% congeladas bit a bit (IEEE-754) + 64 perillas nodales de mielinización (N_few = 60)")
    draw_hbar(out, 24, 72, 458, 285, "Exactitud Few-Shot en Dominio C (320 arcos inéditos)",
              [(lbl, p4[k]["cls_acc"]["mean"], p4[k]["cls_acc"]["ci_lo"], p4[k]["cls_acc"]["ci_hi"], col) for k, lbl, col in t_ord], 1.0)
    draw_hbar(out, 498, 72, 458, 285, "Consistencia Algebraica en Dominio C",
              [(lbl, p4[k]["alg_consistency"]["mean"], p4[k]["alg_consistency"]["ci_lo"], p4[k]["alg_consistency"]["ci_hi"], col) for k, lbl, col in t_ord], 1.0)
    out.append("</svg>")
    (FIGS / "fig4_nervous_transfer.svg").write_text("\n".join(out))
    print(f"Generated 4 SVG figures in {FIGS} from {p.name}.")


if __name__ == "__main__":
    main()
