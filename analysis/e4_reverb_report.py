"""Learned room reverb per recording (models with reverb=True): RT60 and wet level, mid bands (~250 Hz - 2.6 kHz).

Usage: uv run python analysis/e4_reverb_report.py <run>   (run = folder in analysis/results/e4)
"""
import sys

import numpy as np
import pandas as pd
import torch

from tesis.data.gaps import REPO

R = REPO / "analysis" / "results"


def main(run):
    sd = torch.load(R / "e4" / run / "model.pt", map_location="cpu")
    w = sd["rec_reverb.weight"]
    nb = w.shape[1] // 2
    rt60 = 0.2 + 2.3 * torch.sigmoid(w[:, nb:])
    ids = pd.read_csv(R / "e4_recordings.csv").id.tolist()
    df = pd.DataFrame({"id": ids, "rt60_mid_s": rt60[:, 2:6].mean(1).numpy(),
                       "rt60_low_s": rt60[:, :2].mean(1).numpy(), "rt60_high_s": rt60[:, 6:].mean(1).numpy(),
                       "wet_level_mid_db": (20 / np.log(10) * w[:, 2:6]).mean(1).numpy()}).set_index("id")
    dh = pd.read_csv(R / "dynamics_hypothesis.csv").set_index("id")
    df = df.join(dh[["title"]])
    print(df.sort_values("rt60_mid_s", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
