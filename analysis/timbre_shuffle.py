"""Does a timbre-conditioned model use each note's own attack timbre?

Held-out loss of a run with NetConfig note_timbre, (a) as trained, (b) with every note's timbre swapped for that of a
random other note of the same pitch in the same recording, (c) with all timbres set to 0 (the pitch's average-ish).
If (b) is close to (a), the gain comes from the timbre of the pitch/string in general, not from the individual pluck.
Usage: uv run python analysis/timbre_shuffle.py <run>
"""
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
import e4_train as e  # noqa: E402
from compare_runs import load  # noqa: E402
from tesis.data.gaps import REPO  # noqa: E402
from tesis.synth.dsp import MultiScaleSpectralLoss  # noqa: E402


def main(run):
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    m, recs, splits = load(run, ids)
    mss = MultiScaleSpectralLoss()
    orig = [r.timbre.copy() for r in recs]
    rng = np.random.default_rng(0)
    for name in ["as trained", "same-pitch shuffle", "zero"]:
        for r, t in zip(recs, orig):
            if name == "as trained":
                r.timbre = t
            elif name == "zero":
                r.timbre = np.zeros_like(t)
            else:
                s = t.copy()
                for p in np.unique(r.pitch):
                    ii = np.where(r.pitch == p)[0]
                    s[ii] = t[rng.permutation(ii)]
                r.timbre = s
        with torch.no_grad():
            res = e.evaluate(m, recs, splits, mss)
        print(f"{name:20s} loss {res['test_loss']:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
