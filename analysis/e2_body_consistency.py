"""Are learned body modes properties of the guitar, or do they follow the notes of each piece?

Compares the body-mode amplitude profiles (48 modes, 60-800 Hz) of several fitted pieces:
- same performer (likely same guitar/room) should agree if the modes are real;
- each profile's correlation with its own piece's partial occupancy shows stealing.

Usage: uv run python analysis/e2_body_consistency.py <id> [<id> ...]   (reads analysis/results/e2/<id>/model_real.pt)
"""
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
import torch

from tesis.data.gaps import PROCESSED, REPO
from tesis.synth.piece import LOG_B

SR = 24000
R = REPO / "analysis" / "results" / "e2"


def occupancy(rid, f_m):
    """Partial count of the piece's notes within +/-2.75% of each mode (all notes weighted equally)."""
    man = pd.read_csv(PROCESSED / f"sr{SR}" / "manifest.csv").set_index("id")
    notes = pd.read_csv(PROCESSED / f"sr{SR}" / "notes" / f"{rid}.csv")
    p = notes.pitch.values
    f0 = 440 * 2 ** ((p - 69 + man.loc[rid, "tuning_cents"] / 100) / 12)
    k = np.arange(1, 61)
    fk = (f0[:, None] * k * np.sqrt(1 + np.exp(LOG_B[0] + LOG_B[1] * p)[:, None] * k ** 2)).ravel()
    return np.array([np.sum(np.abs(fk / f - 1) < 0.0275) for f in f_m])


def main(ids):
    prof, occ = {}, {}
    for rid in ids:
        sd = torch.load(R / rid / "model_real.pt", map_location="cpu")
        f_m = sd["body_f"].numpy()
        a = sd["body_logamp"].numpy()
        prof[rid] = a - a.max()
        occ[rid] = occupancy(rid, f_m)
        top = np.argsort(-a)[:4]
        print(f"{rid}: strongest modes {', '.join(f'{f_m[i]:.0f}' for i in top)} Hz | "
              f"corr(amplitude, own partial occupancy) = {st.spearmanr(a, occ[rid]).statistic:+.2f}")
    print("\nBody profile correlation between pieces (Spearman):")
    print(pd.DataFrame([[st.spearmanr(prof[a], prof[b]).statistic for b in ids] for a in ids], index=ids, columns=ids
                       ).to_string(float_format=lambda v: f"{v:+.2f}"))
    print("\nBody profile of A vs partial occupancy of B (if modes follow notes, the diagonal stands out):")
    print(pd.DataFrame([[st.spearmanr(prof[a], occ[b]).statistic for b in ids] for a in ids], index=ids, columns=ids
                       ).to_string(float_format=lambda v: f"{v:+.2f}"))


if __name__ == "__main__":
    main(sys.argv[1:])
