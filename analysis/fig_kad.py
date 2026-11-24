"""Thesis figure: realism (PANNs KAD x 100 against the same 85 real windows, 95% bootstrap interval) of the main
versions, synthesizer alone vs with a vocoder. Reads analysis/results/e12_kad.json (analysis/e12_kad.py) and writes
the PDF into the thesis document repo (figuras/kad-versiones.pdf); --png also writes a PNG for a visual check.

    uv run python analysis/fig_kad.py [--out ~/Documentos/tesis-doc-2/figuras]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
L = "e6-structured_v12_final_learnedB"
# (group, label, version key in e12_kad.json, kind) in display order; kind: ddsp | voc | ref
ROWS = [
    ("Sintetizador solo", "Final + adv (20 grabaciones)", f"model-{L}_noise_strongD_rise", "ddsp"),
    ("Sintetizador solo", "M1 + timbre 32 + adv (favorito anterior)", "model-e6-structured_v9_base_t32_noise_strongD", "ddsp"),
    ("Sintetizador solo", "Final · 100 grabaciones + adv", f"model-{L}_data100_noise_strongD_rise", "ddsp"),
    ("Sintetizador solo", "Final · 100 grabaciones · 5× + adv", f"model-{L}_data100x5_noise_strongD_rise", "ddsp"),
    ("Vocoder sobre Final + adv", "BigVGAN preentrenado", "voc-p0_pretrained", "voc"),
    ("Vocoder sobre Final + adv", "BigVGAN ajustado en audio real", "voc-p2_real", "voc"),
    ("Vocoder sobre Final + adv", "Mel reparado (difusión)", "diff-v3_x0res", "voc"),
    ("Vocoder sobre Final + adv", "Mel reparado (flujo)", "diff-v3b_flow", "voc"),
    ("Vocoder sobre Final + adv", "V1", "voc-v1_mix20", "voc"),
    ("Vocoder sobre Final + adv", "S", "voc-s_stems20b", "voc"),
    ("Vocoder sobre Final + adv", "V2", "voc-v2_inject20", "voc"),
    ("Vocoder sobre Final + adv", "V2 + vector del sintetizador", "voc-v2_recddsp20", "voc"),
    ("Vocoder con pares de 100 grabaciones", "V1 (primer modelo de 100)", "voc-v1_mix100", "voc"),
    ("Vocoder con pares de 100 grabaciones", "V2 (modelo 5×)", "voc-v2_x5", "voc"),
    ("Vocoder con pares de 100 grabaciones", "V2 + vector (modelo 5×)", "voc-v2_recddspx5", "voc"),
    ("Vocoder con pares de 100 grabaciones", "V2 + vector (primer modelo de 100)", "voc-v2_recddsp100", "voc"),
    ("Referencia", "Grabación real → BigVGAN ajustado", "voc-p2_real-real", "ref"),
]
COLOR = {"ddsp": "#2a78d6", "voc": "#eb6834", "ref": "#8a8984"}  # reference palette slots 1-2, neutral gray
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "Documentos" / "tesis-doc-2" / "figuras"))
    ap.add_argument("--png", default=None, help="also write a PNG here, for a visual check")
    a = ap.parse_args()
    v = json.loads((REPO / "analysis/results/e12_kad.json").read_text())["panns-wavegram-logmel vs real"]["versions"]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": TEXT})
    # one row per version, an empty row before each group header
    ys, labels, y, prev = [], [], 0, None
    heads = []
    for g, lab, key, kind in ROWS:
        if g != prev:
            if prev is not None:
                y += 0.6
            heads.append((y, g))
            y += 1
            prev = g
        ys.append(y)
        labels.append(lab)
        y += 1
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    for yy, (g, lab, key, kind) in zip(ys, ROWS):
        k, (lo, hi) = v[key]["kad"], v[key]["ci"]
        ax.plot([lo, hi], [yy, yy], color=COLOR[kind], lw=1.5, solid_capstyle="round", zorder=2)
        ax.plot([k], [yy], "o", ms=6.5, color=COLOR[kind], mec="white", mew=1.2, zorder=3)
        ax.text(hi + 0.15, yy, f"{k:.2f}", va="center", ha="left", fontsize=8, color=MUTED)
    for yy, g in heads:
        ax.text(-0.02, yy, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9,
                fontweight="bold", color=TEXT)
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 9.4)
    ax.set_xlabel("KAD × 100, PANNs (menor es más real)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "kad-versiones.pdf")
    if a.png:
        fig.savefig(a.png, dpi=150)
    print(out / "kad-versiones.pdf")


if __name__ == "__main__":
    main()
