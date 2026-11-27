#!/usr/bin/env python3
"""Generate 4 publication-grade SVG figures for Prueba B blind CONFIRM results."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "nervous_wave_b_v1"
FIGS = DOCS / "figures"

ARM_LABELS = {
    "nervous_coherence_b": "Nervous B (1024)",
    "nervous_no_coherence_gate": "No Coh Gate (1024)",
    "nervous_no_delay": "No Phase (1024)",
    "grid_local_128": "Grid Local (1024)",
    "random_er_128": "Random ER (1024)",
    "flat_dense_4096": "Flat Dense (4096)",
}
ARM_COLORS = {
    "nervous_coherence_b": "#0f766e",
    "nervous_no_coherence_gate": "#0284c7",
    "nervous_no_delay": "#64748b",
    "grid_local_128": "#d97706",
    "random_er_128": "#9333ea",
    "flat_dense_4096": "#dc2626",
}
TRANSFER_LABELS = {
    "nervous_transfer": "Nervous B",
    "grid_transfer": "Grid 128",
    "random_transfer": "Random ER",
    "flat_transfer": "Flat 4096",
    "ctrl_permuted_synapses": "Permuted Ctrl",
}
TRANSFER_COLORS = {
    "nervous_transfer": "#0f766e",
    "grid_transfer": "#d97706",
    "random_transfer": "#9333ea",
    "flat_transfer": "#dc2626",
    "ctrl_permuted_synapses": "#64748b",
}


def svg_wrap(w: int, h: int, title: str, subtitle: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="system-ui, -apple-system, sans-serif">
  <rect width="{w}" height="{h}" fill="#ffffff"/>
  <text x="24" y="30" font-size="16" font-weight="700" fill="#0f172a">{title}</text>
  <text x="24" y="48" font-size="11.5" fill="#475569">{subtitle}</text>
  {body}
</svg>
"""


def grouped_bars(
    items: list[tuple[str, float, float, str]],
    x0: float,
    y0: float,
    w: float,
    h: float,
    ymin: float,
    ymax: float,
    panel_title: str,
    unit: str = "%",
    ref_line: tuple[float, str] | None = None,
) -> str:
    parts = [
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#f8fafc" stroke="#e2e8f0" rx="6"/>',
        f'<text x="{x0+10}" y="{y0+18}" font-size="12" font-weight="700" fill="#1e293b">{panel_title}</text>',
    ]
    px, py, pw, ph = x0 + 44, y0 + 28, w - 58, h - 72
    for i in range(5):
        frac = i / 4.0
        yv = ymin + (ymax - ymin) * frac
        yy = py + ph - frac * ph
        parts.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px+pw}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lbl = f"{yv:.0f}{unit}" if unit == "%" else f"{yv:.2f}"
        parts.append(f'<text x="{px-6}" y="{yy+3.5:.1f}" font-size="9.5" fill="#64748b" text-anchor="end">{lbl}</text>')

    if ref_line is not None:
        rv, rlbl = ref_line
        if ymin <= rv <= ymax:
            ry = py + ph - ((rv - ymin) / (ymax - ymin)) * ph
            parts.append(f'<line x1="{px}" y1="{ry:.1f}" x2="{px+pw}" y2="{ry:.1f}" stroke="#0f172a" stroke-dasharray="4 3" stroke-width="1.2"/>')
            parts.append(f'<text x="{px+pw-2}" y="{ry-4:.1f}" font-size="9" fill="#0f172a" text-anchor="end">{rlbl}</text>')

    n = len(items)
    step = pw / max(n, 1)
    bw = step * 0.62
    for idx, (lbl, val, sd, col) in enumerate(items):
        bx = px + idx * step + (step - bw) / 2.0
        clamped = max(ymin, min(ymax, val))
        bh = ((clamped - ymin) / max(ymax - ymin, 1e-9)) * ph
        by = py + ph - bh
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}" rx="3" opacity="0.9"/>')
        # SD whisker
        top_v = min(ymax, val + sd)
        bot_v = max(ymin, val - sd)
        ty = py + ph - ((top_v - ymin) / max(ymax - ymin, 1e-9)) * ph
        boty = py + ph - ((bot_v - ymin) / max(ymax - ymin, 1e-9)) * ph
        cx = bx + bw / 2.0
        parts.append(f'<line x1="{cx:.1f}" y1="{ty:.1f}" x2="{cx:.1f}" y2="{boty:.1f}" stroke="#0f172a" stroke-width="1.2"/>')
        vtxt = f"{val:.1f}" if unit == "%" else f"{val:.2f}"
        parts.append(f'<text x="{cx:.1f}" y="{max(py+9, ty-3):.1f}" font-size="9" font-weight="600" fill="#0f172a" text-anchor="middle">{vtxt}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{py+ph+13:.1f}" font-size="8.5" fill="#334155" text-anchor="middle" transform="rotate(-18 {cx:.1f} {py+ph+13:.1f})">{lbl}</text>')
    return "\n  ".join(parts)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    s = json.loads((DOCS / "summary_confirm.json").read_text())
    arms = list(ARM_LABELS.keys())

    # Fig 1: Suite B1 Physics & Kuramoto Coherence
    p1 = grouped_bars(
        [(ARM_LABELS[a], s["b1"][a]["arrival_time"]["mean"], s["b1"][a]["arrival_time"]["sd"], ARM_COLORS[a]) for a in arms],
        20, 64, 420, 230, 0.0, 4.5, "Wavefront Arrival Time (t_arr, lower = faster)", unit="",
    )
    p2 = grouped_bars(
        [(ARM_LABELS[a], s["b1"][a]["localization_snr"]["mean"] * 100, s["b1"][a]["localization_snr"]["sd"] * 100, ARM_COLORS[a]) for a in arms],
        460, 64, 420, 230, 0.0, 105.0, "Spatial Isolation SNR at t=1 (%)", unit="%",
    )
    p3 = grouped_bars(
        [(ARM_LABELS[a], s["b1"][a]["coincidence"]["mean"], s["b1"][a]["coincidence"]["sd"], ARM_COLORS[a]) for a in arms],
        20, 310, 420, 230, 0.0, 1.05, "Past-Future Collision Coincidence at Hubs", unit="",
    )
    p4 = grouped_bars(
        [(ARM_LABELS[a], s["b1"][a]["hub_r_coh"]["mean"], s["b1"][a]["hub_r_coh"]["sd"], ARM_COLORS[a]) for a in arms],
        460, 310, 420, 230, 0.0, 1.05, "Kuramoto Phase-Coherence Order R_coh (Hubs)", unit="",
    )
    (FIGS / "fig1_physics_kuramoto.svg").write_text(
        svg_wrap(900, 560, "Prueba B — Suite B1: Physical Wave Propagation & Kuramoto Coherence",
                 f"Blind CONFIRM (N={s['n_seeds']} seeds, 0xB200..0xB20F) | Mean ± SD across 6 topologies", p1 + p2 + p3 + p4)
    )

    # Fig 2: Suite B2 Inference Domain A & Domain C
    p1 = grouped_bars(
        [(ARM_LABELS[a], s["b2"][a]["domA_acc"]["mean"] * 100, s["b2"][a]["domA_acc"]["sd"] * 100, ARM_COLORS[a]) for a in arms],
        20, 64, 420, 240, 40.0, 80.0, "Domain A Relational Accuracy (%)", unit="%",
    )
    p2 = grouped_bars(
        [(ARM_LABELS[a], s["b2"][a]["domA_abduction"]["mean"] * 100, s["b2"][a]["domA_abduction"]["sd"] * 100, ARM_COLORS[a]) for a in arms],
        460, 64, 420, 240, 0.0, 55.0, "Domain A Abduction Top-1 (%)", unit="%",
    )
    p3 = grouped_bars(
        [(ARM_LABELS[a], s["b2"][a]["domC_acc"]["mean"] * 100, s["b2"][a]["domC_acc"]["sd"] * 100, ARM_COLORS[a]) for a in arms],
        20, 320, 420, 240, 50.0, 90.0, "Domain C Discrete Graph Accuracy (%)", unit="%",
    )
    p4 = grouped_bars(
        [(ARM_LABELS[a], s["b2"][a]["domC_consistency"]["mean"] * 100, s["b2"][a]["domC_consistency"]["sd"] * 100, ARM_COLORS[a]) for a in arms],
        460, 320, 420, 240, 30.0, 85.0, "Domain C Algebraic Consistency (%)", unit="%",
    )
    (FIGS / "fig2_inference_domAC.svg").write_text(
        svg_wrap(900, 580, "Prueba B — Suite B2: Relational Inference & Sparse 4x Parity",
                 f"Blind CONFIRM (N={s['n_seeds']} seeds) | N=128, E=1,024 sparse arms vs E=4,096 dense control", p1 + p2 + p3 + p4)
    )

    # Fig 3: Suite B3 Resilience (Lesions + Phase Noise + Sensory Noise)
    p1 = grouped_bars(
        [(ARM_LABELS[a], s["b3"][a]["lesion15_acc"]["mean"] * 100, s["b3"][a]["lesion15_acc"]["sd"] * 100, ARM_COLORS[a]) for a in arms],
        20, 64, 420, 240, 30.0, 75.0, "15% Axonal Lesions — Domain A Acc (%)", unit="%",
    )
    p2 = grouped_bars(
        [(ARM_LABELS[a], s["b3"][a]["lesion30_acc"]["mean"] * 100, s["b3"][a]["lesion30_acc"]["sd"] * 100, ARM_COLORS[a]) for a in arms],
        460, 64, 420, 240, 25.0, 70.0, "30% Axonal Lesions — Domain A Acc (%)", unit="%",
    )
    p3 = grouped_bars(
        [(ARM_LABELS[a], s["b3"][a]["noise50_acc"]["mean"] * 100, s["b3"][a]["noise50_acc"]["sd"] * 100, ARM_COLORS[a]) for a in arms],
        20, 320, 420, 240, 30.0, 75.0, "Synaptic Phase Noise (sigma=0.50) Acc (%)", unit="%",
    )
    p4 = grouped_bars(
        [(ARM_LABELS[a], s["b3"][a]["sens_noise50_acc"]["mean"] * 100, s["b3"][a]["sens_noise50_acc"]["sd"] * 100, ARM_COLORS[a]) for a in arms],
        460, 320, 420, 240, 30.0, 70.0, "Sensory Input Noise (sigma=0.50) Acc (%)", unit="%",
    )
    (FIGS / "fig3_resilience_coherence.svg").write_text(
        svg_wrap(900, 580, "Prueba B — Suite B3: Zero-Retraining Structural & Phase-Noise Resilience",
                 f"Blind CONFIRM (N={s['n_seeds']} seeds, 6 masks/realizations per seed) | Kuramoto Coherence Gate vs Ablations", p1 + p2 + p3 + p4)
    )

    # Fig 4: Suite B4 Frozen-Synapse Resonant Spring Transfer (A -> B and A -> C)
    tarms = list(TRANSFER_LABELS.keys())
    p1 = grouped_bars(
        [(TRANSFER_LABELS[a], s["b4"]["domain_c_graph"][a]["acc"]["mean"] * 100, s["b4"]["domain_c_graph"][a]["acc"]["sd"] * 100, TRANSFER_COLORS[a]) for a in tarms],
        20, 64, 420, 260, 35.0, 75.0, "A -> C Discrete Graph (N=60, Frozen Synapses)", unit="%",
        ref_line=(52.54, "Prueba A baseline (52.54%)"),
    )
    p2 = grouped_bars(
        [(TRANSFER_LABELS[a], s["b4"]["domain_b_oscillators"][a]["acc"]["mean"] * 100, s["b4"]["domain_b_oscillators"][a]["acc"]["sd"] * 100, TRANSFER_COLORS[a]) for a in tarms],
        460, 64, 420, 260, 35.0, 75.0, "A -> B Continuous Oscillators (N=48, Frozen)", unit="%",
    )
    (FIGS / "fig4_transfer_ab_ac.svg").write_text(
        svg_wrap(900, 350, "Prueba B — Suite B4: Frozen-Synapse Resonant Spring Auto-Calibration Transfer",
                 f"Blind CONFIRM (N={s['n_seeds']} seeds) | 100% bit-exact frozen tissue (IEEE-754 verified) + myelin resonant spring", p1 + p2)
    )
    print(f"Generated 4 SVG figures in {FIGS}")


if __name__ == "__main__":
    main()
