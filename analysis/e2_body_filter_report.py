"""Compare learned body frequency responses (config body="filter") across runs.

A straight line in log-frequency is removed from each response first: an overall tilt of the body can't be told apart
from string brightness. Only 80 Hz - 5 kHz is compared (where the pieces have partials).

Usage: uv run python analysis/e2_body_filter_report.py <run> [<run> ...]   (run = folder in analysis/results/e2)
"""
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as sps
import torch

from tesis.data.gaps import REPO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = REPO / "analysis" / "results" / "e2"
F_LO, F_HI = 80.0, 5000.0


def response(run):
    sd = torch.load(R / run / "model_real.pt", map_location="cpu")
    f = sd["body_centers"].numpy()
    db = sd["body_resp"].numpy() * 20 / np.log(10)
    m = (f >= F_LO) & (f <= F_HI)
    lf = np.log(f[m])
    resid = db[m] - np.polyval(np.polyfit(lf, db[m], 1), lf)
    return f[m], resid


def main(runs):
    resp = {r: response(r) for r in runs}
    f = resp[runs[0]][0]
    print("Detrended body response, correlation between runs (Pearson, 80 Hz - 5 kHz):")
    print(pd.DataFrame([[np.corrcoef(resp[a][1], resp[b][1])[0, 1] for b in runs] for a in runs], index=runs, columns=runs
                       ).to_string(float_format=lambda v: f"{v:+.2f}"))
    print("\nPeaks (>= 2 dB prominence) per run:")
    for r in runs:
        pk, pr = sps.find_peaks(resp[r][1], prominence=2.0)
        print(f"  {r:22s} " + ", ".join(f"{f[i]:.0f} Hz ({pr['prominences'][j]:.1f} dB)" for j, i in enumerate(pk)))
    fig, ax = plt.subplots(figsize=(11, 5))
    for r in runs:
        ax.semilogx(f, resp[r][1], label=r, lw=1.2)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB (detrended)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "body_filter_profiles.png", dpi=90)
    print(f"\nplot: {R / 'body_filter_profiles.png'}")


if __name__ == "__main__":
    main(sys.argv[1:])
