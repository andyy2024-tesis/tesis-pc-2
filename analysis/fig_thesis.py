"""Thesis figures (all but the KAD one, analysis/fig_kad.py), from the saved results. Writes PDFs into the thesis
document repo (figuras/); --png DIR also writes PNGs there for a visual check.

    uv run python analysis/fig_thesis.py [--only arquitectura,perdida,...] [--png DIR]

arquitectura   the synthesizer and the vocoder stage (diagram)
perdida        loss of the main structured models stage by stage, DDSP for reference (catalog_metrics.json)
viento         where the residual noise sits around each pluck, M1 (wind_map.json)
inarmonicidad  log B against fret per string, tab-matched clean notes (inharmonicity_strings.csv)
rigidez        learned vs measured stiffness per guitar and string (model weights, inharmonicity_guitar.csv)
espectrogramas one excerpt: original, Final + adv, Final + adv -> V2 (listening excerpts)
vocoder        held-out loss over training steps of the vocoder routes (e12/<run>/log.txt)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "analysis" / "results"
sys.path.insert(0, str(REPO / "analysis"))

# reference palette (light mode): categorical slots in fixed order, text and surface tokens
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TEXT, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cfe0f6", "#6fa3e3", "#2a78d6", "#123f7a", "#06172e"])
W = 6.3  # full-width figures (figure*), drawn at 6.3 in and scaled to the 6.9 in text width
C = 3.35  # one column

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9, "legend.frameon": False})


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------------------------------------------
def fig_arquitectura():
    fig, ax = plt.subplots(figsize=(W, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 100)
    ax.axis("off")
    col = {"in": ("#f1f0ec", GRAY), "net": ("#dce8f8", S1), "part": ("#fbe3d8", S2), "out": ("#ffffff", TEXT)}

    def box(x, y, w, h, txt, kind, fs=6.3):
        fc, ec = col[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=TEXT, linespacing=1.15)

    def arrow(p, q, rad=0.0, color=MUTED):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))

    ax.text(0, 99, "a) Sintetizador estructurado (modelo final)", fontsize=9, fontweight="bold", color=TEXT, va="top")
    ins = [("Notas de la partitura\n(altura, inicio, fin,\ncuerda y traste)", 83.5, 9.5),
           ("Pistas del audio\n(volumen y afinación\ncada 10 ms)", 71.5, 9.5), ("Timbre del ataque\n(32 parciales)", 61, 7.5),
           ("Vector de la\ngrabación (32)", 50.5, 7.5)]
    for t, y, h in ins:
        box(0.5, y, 20.5, h, t, "in", fs=6.1)
    box(26, 76, 21, 8.5, "MLP de entrada\n(por cuerda)", "net", fs=6.4)
    box(26, 61.5, 21, 10.5, "Atención que anticipa\n(64 cuadros atrás,\n16 adelante)", "net", fs=6.3)
    box(26, 49.5, 21, 8.5, "MLP de nota\n(una vez por nota)", "net", fs=6.4)
    for y in (88.25, 76.25, 64.75):
        arrow((21.3, y), (26, 80.25))
    for y in (64.75, 54.25):
        arrow((21.3, y), (26, 53.75))
    arrow((21.3, 86), (26, 55.5), rad=0.3)
    arrow((36.5, 76), (36.5, 72))
    parts = ["Cuerdas: parciales de cuerda rígida,\ndecaimiento físico y corrección δ",
             "Respuesta del cuerpo a cada pulsación", "Ataque propio de cada nota (33 bandas)",
             "Contacto (16 bandas por grabación)", "Piso de ruido de la sala", "Ruido residual (se apaga al generar)"]
    ys = [86, 78.6, 71.2, 63.8, 56.4, 49]
    for t, y in zip(parts, ys):
        box(52.5, y, 34, 6, t, "part", fs=6.0)
    arrow((47.3, 69.5), (52.5, 89), rad=-0.15)          # correction
    arrow((47.3, 66.8), (52.5, 66.8))                   # contact level
    arrow((47.3, 64), (52.5, 52), rad=0.15)             # noise filter
    for y in (88, 81.6, 74.2):                      # template, body level, attack per note
        arrow((47.3, 55.5), (52.5, y), rad=-0.12)
    box(90.5, 65, 5, 6, "Σ", "out", fs=11)
    for y in ys:
        arrow((86.8, y + 3), (90.5, 68))
    ax.text(93, 63.5, "audio del\nsintetizador", ha="center", va="top", fontsize=6.6, color=TEXT)
    ax.text(52.5 + 17, 46.5, "El vector de la grabación llega también al decaimiento, al cuerpo, al ataque,\n"
            "al fundido final y a la rigidez de cada cuerda (grabación en todo).", ha="center", va="top", fontsize=6.2,
            color=MUTED, style="italic")

    ax.text(0, 36, "b) Vocoder V2 (BigVGAN-v2 ajustado)", fontsize=9, fontweight="bold", color=TEXT, va="top")
    box(0.5, 14.5, 15, 8, "Audio del\nsintetizador", "in", fs=6.6)
    box(0.5, -1.5, 15, 7, "Vector de la\ngrabación", "in", fs=6.6)
    box(20.5, 20, 14.5, 7, "Espectrograma\nmel", "in", fs=6.1)
    box(20.5, 7, 14.5, 7, "Forma de onda", "in", fs=6.1)
    arrow((15.8, 20.5), (20.5, 23.5))
    arrow((15.8, 16.5), (20.5, 10.5))
    st = [(40, 10.5, "Capa de\nentrada"), (52, 10.5, "Sobre-\nmuestreo 1"), (64, 10.5, "Sobre-\nmuestreo 2"),
          (76.5, 4, "…"), (82.5, 9, "Salida")]
    for x, w, t in st:
        box(x, 19, w, 9, t, "net", fs=6.3)
    arrow((35.3, 23.5), (40, 23.5))
    for (x1, w1, _), (x2, _, _) in zip(st[:-1], st[1:]):
        arrow((x1 + w1, 23.5), (x2, 23.5))
    ax.text(97, 23.5, "audio", ha="center", va="center", fontsize=7, color=TEXT)
    arrow((91.8, 23.5), (94.5, 23.5))
    for x, w, _ in st[1:3]:  # the waveform enters every upsampling stage
        arrow((35.3, 10.5), (x + w / 2, 19), rad=0.12, color=S2)
    ax.text(77, 12.5, "la forma de onda entra en cada\netapa de sobremuestreo\n(convoluciones con paso,\nempiezan en cero)",
            fontsize=6.1, color=S2, ha="left", va="center")
    for x, w, _ in st[:3]:
        arrow((15.8, 2), (x + w / 2, 19), rad=0.18, color=S1)
    ax.text(40, -2.5, "escala y desplazamiento por canal (empiezan en cero)", fontsize=6.1, color=S1, ha="left")
    return fig


# ---------------------------------------------------------------------------------------------------------------
def fig_perdida():
    import build_listening_catalog as b
    m = json.loads((RES / "catalog_metrics.json").read_text())
    run = {e["n"]: e["run"] for e in b.C if e.get("run")}

    def loss(n):
        return m[run[n]]["mean"]

    stages = [("Ronda 1", "S1 Estructurado", "D1 DDSP"), ("Cuerpo", "S1 + cuerpo", None), ("Tablas por\ngrabación", "S2", None),
              ("Información\ndel audio", "M1 Base", "D2"), ("Fenómenos", "M2 Combinado", None),
              ("Timbre del\nataque", "M2 + timbre 32", None),
              ("Ataque por nota\ny fondo medido", "M1 + timbre 32 + ataque por nota + fondo medido", None),
              ("Cuerdas", "M1 + timbre 32 + ataque por nota + fondo medido + cuerdas", None),
              ("Final", "Final · rigidez aprendida", None),
              ("100 grabaciones\n(5×)", "Final · 100 grabaciones · 5× entrenamiento", None)]
    x = np.arange(len(stages))
    ys = np.array([loss(s[1]) for s in stages])
    fig, ax = plt.subplots(figsize=(C, 3.1))
    ax.fill_between(x, ys - 0.025, ys + 0.025, color=S1, alpha=0.15, lw=0, label="variación entre semillas (±0.025)")
    ax.plot(x, ys, "-o", color=S1, lw=1.6, ms=4.5, mec="white", mew=0.8, label="Estructurado")
    dx = [i for i, s in enumerate(stages) if s[2]]
    ax.plot(dx, [loss(stages[i][2]) for i in dx], "s", color=S2, ms=5, mec="white", mew=0.8, label="DDSP (misma red)")
    for i in dx:
        ax.annotate(stages[i][2].split()[0], (i, loss(stages[i][2])), xytext=(5, 2), textcoords="offset points",
                    fontsize=6.5, color=MUTED)
    for i, yv in enumerate(ys):
        ax.annotate(f"{yv:.2f}", (i, yv), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=5.8,
                    color=MUTED)
    short = ["Ronda 1", "Cuerpo", "Tablas", "Info. audio", "Fenómenos", "Timbre", "Ataque y fondo", "Cuerdas",
             "Final", "100 grab. (5×)"]  # stage names, short for one column (the caption gives them in full)
    ax.set_xticks(x, short, fontsize=6.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Pérdida espectral\n(media, 85 ventanas)", fontsize=7.5)
    ax.set_ylim(5.5, 6.5)
    clean(ax)
    ax.legend(loc="upper right", fontsize=6.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------
def fig_viento():
    d = json.loads((RES / "wind_map.json").read_text())["structured_v5_body_audio"]["time"]
    cls = [("pre_40_12ms", "−40 a\n−12"), ("attack_0_10ms", "0 a\n10"), ("attack_10_30ms", "10 a\n30"),
           ("attack_30_60ms", "30 a\n60"), ("attack_60_150ms", "60 a\n150"), ("held", "Nota\nsosten."),
           ("ringing", "Tras el\nfinal")]
    x = np.arange(len(cls))
    noise = [100 * d[k]["share_of_noise"] for k, _ in cls]
    orig = [100 * d[k]["share_of_orig"] for k, _ in cls]
    fig, ax = plt.subplots(figsize=(C, 2.5))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, orig, w, color=GRAY, label="Energía del original")
    ax.bar(x + w / 2 + 0.01, noise, w, color=S2, label="Energía del ruido residual")
    for i, k in enumerate(cls):
        c = d[k[0]]["share_of_noise"] / d[k[0]]["share_of_orig"]
        ax.annotate(f"×{c:.1f}", (x[i] + w / 2, noise[i]), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=6, color=TEXT if c > 1.5 else MUTED, fontweight="bold" if c > 1.5 else "normal")
    ax.set_xticks(x, [c[1] for c in cls], fontsize=6.3)
    ax.set_xlabel("Respecto de cada pulsación (ms)", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Parte de la energía (%)", fontsize=7.5)
    clean(ax)
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------
def fig_inarmonicidad():
    import pandas as pd
    from tesis.model.guitar_net import STRING_LOG_B
    d = pd.read_csv(RES / "inharmonicity_strings.csv")
    d = d[d.matched == 1]
    open_pitch = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
    names = {1: "1: mi 4, nailon", 2: "2: si 3, nailon", 3: "3: sol 3, nailon", 4: "4: re 3, entorchada",
             5: "5: la 2, entorchada", 6: "6: mi 2, entorchada"}
    fig, axs = plt.subplots(2, 3, figsize=(W, 3.9), sharex=True, sharey=True)
    fr = np.linspace(0, 14, 50)
    for s, ax in zip(range(1, 7), axs.flat):
        q = d[d.string == s]
        ax.plot(q.fret, np.log(q.B), "o", ms=3, color=S1, alpha=0.35, mew=0)
        ax.plot(fr, STRING_LOG_B[s] + np.log(2) / 6 * fr, color=S1, lw=2, label="ley por cuerda (modelo)")
        ax.plot(fr, -11.35 + 0.0304 * (open_pitch[s] + fr), color=S2, lw=2, ls="--", label="ley única anterior")
        ax.set_title(f"{names[s]} ({len(q)})", fontsize=7.5, color=TEXT)
        clean(ax, grid="both")
    for ax in axs[1]:
        ax.set_xlabel("Traste")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\log B$")
    axs[0, 0].legend(loc="lower right", fontsize=6.5)
    axs[0, 0].set_ylim(-13.2, -7.6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------
def fig_rigidez():
    import pandas as pd
    import torch
    from tesis.model.guitar_net import STRING_LOG_B
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()  # row order of the recording embedding
    sd = torch.load(RES / "e4" / "structured_v12_final_learnedB" / "model.pt", map_location="cpu")
    sh = (0.5 * torch.tanh(sd["rec_emb.weight"] @ sd["rec_b.weight"].T + sd["rec_b.bias"])).numpy()
    g = pd.read_csv(RES / "inharmonicity_guitar.csv")
    g = g[g.id.isin(ids)]
    meas = g.log_b_open.values - np.array([STRING_LOG_B[int(s)] for s in g.string])
    lear = np.array([sh[ids.index(i), int(s) - 1] for i, s in zip(g.id, g.string)])
    r = np.corrcoef(meas, lear)[0, 1]
    few = g.n.values < 3
    fig, ax = plt.subplots(figsize=(C, 3.0))
    lim = 1.0
    ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=1.2, zorder=1)
    ax.axhspan(0.5, lim, color="#f1f0ec", zorder=0)
    ax.axhspan(-lim, -0.5, color="#f1f0ec", zorder=0)
    ax.plot(meas[~few], lear[~few], "o", ms=4, color=S1, mec="white", mew=0.8, zorder=3, label="3 notas o más")
    ax.plot(meas[few], lear[few], "o", ms=4, mfc="none", mec=S1, mew=1, zorder=3, label="menos de 3 notas")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel(r"Rigidez medida: $\log B$ al aire $-$ tabla de la cuerda", fontsize=7.5)
    ax.set_ylabel(r"Desplazamiento aprendido de $\log B$", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.text(0.03, 0.97, f"r = {r:.2f} ({len(meas)} pares)", transform=ax.transAxes, va="top",
            fontsize=6.5, color=TEXT)
    ax.text(0.97, 0.97, "gris: fuera del límite (±0.5)", transform=ax.transAxes, ha="right",
            va="top", fontsize=5.8, color=MUTED)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.02), fontsize=5.8, ncol=2, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0.2)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    clean(ax, grid="both")
    fig.tight_layout()
    print(f"rigidez: r = {r:.3f}, sd learned {lear.std():.3f}, measured {meas.std():.3f}")
    return fig


# ---------------------------------------------------------------------------------------------------------------
def fig_espectrogramas(rec="276_jw1wc"):
    import librosa
    import soundfile as sf
    L = "e6-structured_v12_final_learnedB_noise_strongD_rise"
    files = [("Grabación original", "original.wav"), ("Final + adv (sintetizador)", f"{L}.wav"),
             ("Final + adv → V2", f"voc-v2_inject20-{L}.wav")]
    specs = []
    for _, f in files:
        y, sr = sf.read(RES / "slides" / "audio" / rec / f)
        y = y.mean(1) if y.ndim > 1 else y
        if sr != 24000:
            y = librosa.resample(y, orig_sr=sr, target_sr=24000)
            sr = 24000
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=256, n_mels=128, fmin=40, fmax=12000)
        specs.append(M)
    ref = specs[0].max()
    fig, axs = plt.subplots(3, 1, figsize=(W, 5.4), sharex=True)
    for ax, (title, _), M in zip(axs, files, specs):
        db = librosa.power_to_db(M, ref=ref, top_db=None)
        t = np.arange(M.shape[1]) * 256 / 24000
        im = ax.imshow(np.clip(db, -80, 0), origin="lower", aspect="auto", cmap=SEQ,
                       extent=[0, t[-1], 0, M.shape[0]], vmin=-80, vmax=0)
        fr = librosa.mel_frequencies(n_mels=128, fmin=40, fmax=12000)
        ticks = [100, 500, 1000, 2000, 5000, 10000]
        ax.set_yticks([np.interp(f, fr, np.arange(128)) for f in ticks], ["100", "500", "1k", "2k", "5k", "10k"])
        ax.set_ylabel("Hz")
        ax.set_title(title, loc="left", fontsize=8.5, color=TEXT)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[-1].set_xlabel("Tiempo (s)")
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes((0.92, 0.12, 0.018, 0.76))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("dB (relativo al máximo del original)", fontsize=7.5)
    cb.outline.set_visible(False)
    return fig


# ---------------------------------------------------------------------------------------------------------------
def evals(run):
    txt = (RES / "e12" / run / "log.txt").read_text()
    pts = [(0, float(re.search(r"step 0 eval ([0-9.]+)", txt).group(1)))]
    pts += [(int(s), float(v)) for s, v in re.findall(r"EVAL step (\d+): ([0-9.]+)", txt)]
    return np.array(pts).T


def fig_vocoder():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
    raw = 5.815  # Final + adv alone on the same windows (catalog_metrics.json)
    a.axhline(raw, color=GRAY, lw=1, ls="--", label="sintetizador solo")
    b.axhline(raw, color=GRAY, lw=1, ls="--")
    for run, lab, c in [("v1_mix20", "V1", S1), ("v2_inject20", "V2", S2), ("s_stems20b", "S", S3),
                        ("v2_recddsp20", "V2 + vector", S4)]:
        x, y = evals(run)
        a.plot(x, y, "-o", color=c, lw=2, ms=4.5, mec="white", mew=0.8, label=lab)
    a.set_title("a) Pares de 20 grabaciones", loc="left", fontsize=8.5, color=TEXT)
    a.set_xlabel("Pasos")
    a.set_ylabel("Pérdida en las ventanas reservadas")
    a.set_ylim(5.68, 5.92)
    a.legend(fontsize=7, loc="lower left")
    for run, lab, c in [("v2_x5", "V2, modelo 5×", S1), ("v2_recddspx5", "V2 + vector, modelo 5×", S2),
                        ("v2_recddsp100", "V2 + vector, primer modelo", S3)]:
        x, y = evals(run)
        yc = np.minimum(y, 5.935)
        b.plot(x, yc, "-o", color=c, lw=2, ms=4.5, mec="white", mew=0.8, label=lab)
        for xi, yi in zip(x, y):
            if yi > 5.935:
                b.annotate(f"{yi:.3f} (fuera de escala)", (xi, 5.935), xytext=(4, -10), textcoords="offset points",
                           fontsize=6.8, color=c)
    b.set_title("b) Pares de 100 grabaciones", loc="left", fontsize=8.5, color=TEXT)
    b.set_xlabel("Pasos")
    b.set_ylim(5.6, 5.94)
    b.legend(fontsize=7, loc="lower left")
    for ax in (a, b):
        clean(ax)
    fig.tight_layout()
    return fig


FIGS = {"arquitectura": fig_arquitectura, "perdida": fig_perdida, "viento": fig_viento,
        "inarmonicidad": fig_inarmonicidad, "rigidez": fig_rigidez, "espectrogramas": fig_espectrogramas,
        "vocoder": fig_vocoder}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "Documentos" / "tesis-doc-2" / "figuras"))
    ap.add_argument("--only", default=",".join(FIGS))
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for name in a.only.split(","):
        fig = FIGS[name]()
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        if a.png:
            fig.savefig(Path(a.png) / f"{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(out / f"{name}.pdf")


if __name__ == "__main__":
    main()
