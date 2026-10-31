"""Diagnostics for a fitted PieceModel: body-mode stealing and onset (transient) behaviour.

stealing: Spearman correlation across body modes between the mode's amplitude and how much partial energy of the
          piece's notes falls within +/-2.75% of the mode frequency (half the grid spacing). High = modes sit on partials.
onsets:   high-frequency (2-12 kHz) energy rise at each onset, [0, 10 ms] vs [-20, -5 ms], target vs model parts.

Usage: uv run python analysis/e2_diagnostics.py <id> <model.pt> [config flags as JSON]
"""
import json
import sys

import numpy as np
import pandas as pd
import scipy.signal as sps
import scipy.stats as st
import torch

sys.path.insert(0, "scripts")
import e2_one_piece as e  # noqa: E402
from tesis.synth.piece import PieceConfig, PieceModel, note_table  # noqa: E402

SR = 24000


def load(rid, pt, cfg_kw):
    cfg = PieceConfig(sr=SR, **cfg_kw)
    man = pd.read_csv(e.D / "manifest.csv").set_index("id")
    notes = pd.read_csv(e.D / "notes" / f"{rid}.csv")
    y = np.load(e.D / "audio" / f"{rid}.npy").astype(np.float32) / 32768
    L = int(cfg.win * SR)
    y = y[: (len(y) // L) * L]
    notes = notes[notes.onset * SR < len(y)].reset_index(drop=True)
    tab = note_table(notes, cfg, man.loc[rid, "tuning_cents"])
    m = PieceModel(tab, len(y), cfg, "cuda")
    m.load_state_dict(torch.load(pt, map_location="cuda"))
    return m, y, notes, tab


def stealing(m, tab):
    f_m = m.body_f.cpu().numpy()
    amp = m.body_logamp.detach().cpu().numpy()
    with torch.no_grad():
        lv = m.note_levels_db().cpu().numpy()
    w = 10 ** ((lv - lv.max()) / 10)
    occ = np.zeros(len(f_m))
    for fn, wn in zip(tab["freqs"], w):
        for j, fm in enumerate(f_m):
            occ[j] += wn * np.sum(np.abs(fn / fm - 1) < 0.0275)
    rho = st.spearmanr(amp, np.log(occ + 1e-6)).statistic
    top = np.argsort(-amp)[:5]
    return dict(stealing_rho=float(rho), top_modes_hz=[round(float(f_m[i]), 1) for i in top],
                top_modes_occupancy_rank=[int((occ > occ[i]).sum()) + 1 for i in top])


def onsets(parts, y, notes):
    sos = sps.butter(4, [2000, 11500], "bandpass", fs=SR, output="sos")
    hp = {k: sps.sosfiltfilt(sos, v) ** 2 for k, v in parts.items()}
    hp["target"] = sps.sosfiltfilt(sos, y) ** 2
    hp["model"] = sps.sosfiltfilt(sos, sum(parts.values())) ** 2
    rows = []
    for _, r in notes.iterrows():
        s = int(r.onset * SR)
        if s < int(0.03 * SR) or s + int(0.01 * SR) > len(y):
            continue
        pre = slice(s - int(0.02 * SR), s - int(0.005 * SR))
        post = slice(s, s + int(0.01 * SR))
        row = dict(refined=r.onset_conf >= 5)
        for k in ("target", "model", "strings", "burst"):
            row[f"{k}_rise_db"] = 10 * np.log10(hp[k][post].mean() + 1e-12) - 10 * np.log10(hp["target"][pre].mean() + 1e-12)
        rows.append(row)
    df = pd.DataFrame(rows)
    return {f"{'refined' if g else 'unrefined'}": {c: float(df[df.refined == g][c].median()) for c in df.columns if c.endswith("db")}
            | {"n": int((df.refined == g).sum())} for g in (True, False)}


def main():
    rid, pt = sys.argv[1], sys.argv[2]
    cfg_kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    m, y, notes, tab = load(rid, pt, cfg_kw)
    parts = e.render(m, len(y))
    out = dict(stealing=stealing(m, tab), onsets=onsets(parts, y, notes))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
