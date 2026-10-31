"""Is there a squeak signature before notes on the wound strings?

Notes below G3 (MIDI 55) can only be played on the three wound strings (4-6), where left-hand shifts squeak. For every
onset in the held-out windows, measures 1.5-6 kHz energy in the 12-40 ms before it (21 ms window): original vs the
model without its catch-all noise ("missing" = original - model, dB) and the noise vs the original. Compares
wound-string onsets that follow another wound-string note at a different pitch (a likely shift) with treble onsets.
Usage: uv run python analysis/squeak_check.py <run>
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import load  # noqa: E402
from tesis.data.gaps import REPO  # noqa: E402

SR, L, NFFT, HOP = 24000, 72000, 512, 120
DEV = os.environ.get("WIND_DEV", "cuda")


def power(x):
    x = torch.as_tensor(x, dtype=torch.float32, device=DEV)
    return (torch.stft(x, NFFT, HOP, window=torch.hann_window(NFFT, device=DEV), return_complex=True)
            .abs() ** 2).cpu().numpy()


def main(run):
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    m, recs, splits = load(run, ids)
    m = m.to(DEV)
    fb = np.fft.rfftfreq(NFFT, 1 / SR)
    band = (fb >= 1500) & (fb < 6000)
    rows = []
    for rec, (_, test) in zip(recs, splits):
        order = np.argsort(rec.onset)
        for j, w0 in enumerate(test):
            w0 = int(w0)
            with torch.no_grad():
                _, parts = m(rec, w0, noise_seed=j)
            n = parts["noise"].cpu().numpy() if "noise" in parts else np.zeros(L, np.float32)
            d = sum(v for k, v in parts.items() if k != "noise").cpu().numpy()
            Y, D, N = (power(x)[band].sum(0) for x in (rec.y[w0:w0 + L], d, n))
            for oi, i in enumerate(order):
                o = rec.onset[i] - w0
                if o < int(0.05 * SR) or o > L - int(0.01 * SR):
                    continue
                fr = np.arange((o - int(0.04 * SR)) // HOP, (o - int(0.012 * SR)) // HOP)
                prev = rec.pitch[order[oi - 1]] if oi > 0 else -1
                gap = (rec.onset[i] - rec.onset[order[oi - 1]]) / SR if oi > 0 else 9
                rows.append(dict(wound=rec.pitch[i] < 55, shift=(rec.pitch[i] < 55) and (prev < 55) and (prev != rec.pitch[i]),
                                 gap=gap, y=Y[fr].sum(), d=D[fr].sum(), n=N[fr].sum()))
    df = pd.DataFrame(rows)
    db = lambda a, b: 10 * np.log10(a.sum() / b.sum())  # noqa: E731
    for name, s in [("wound, after another wound note (likely shift)", df["shift"]),
                    ("wound, other", df.wound & ~df["shift"]), ("treble (G3 and up)", ~df.wound)]:
        g = df[s]
        print(f"{name:48s} n={len(g):4d}  model/orig {db(g.d, g.y):6.1f} dB  noise/orig {db(g.n, g.y):6.1f} dB")


if __name__ == "__main__":
    main(sys.argv[1])
