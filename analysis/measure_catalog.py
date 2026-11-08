"""Measure every model of the listening page's catalog the same way: the 85 held-out windows of the 20 recordings.

Loss is the mean multi-scale spectral loss (`mean`; `median` also kept: the E4 tables report medians), wind is the catch-all
noise relative to the original in the quietest 20 % of frames. Merges into analysis/results/catalog_metrics.json.
Usage: uv run python analysis/measure_catalog.py <run> [<run> ...]   (runs as in compare_runs.py)
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import RES, load  # noqa: E402
from e6_adversarial import quiet_noise_db  # noqa: E402
from tesis.synth.dsp import MultiScaleSpectralLoss  # noqa: E402

OUT = RES / "catalog_metrics.json"


def main(runs):
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    mss = MultiScaleSpectralLoss()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for run in runs:
        try:
            m, recs, splits = load(run, ids)
            losses = []
            with torch.no_grad():
                for rec, (_, test) in zip(recs, splits):
                    L = int(m.cfg.win * m.cfg.sr)
                    for j, w0 in enumerate(test):
                        out, _ = m(rec, int(w0), noise_seed=j)
                        y = torch.tensor(rec.y[w0:w0 + L], device=out.device)
                        losses.append(mss(out[None], y[None]).item())
                wind = quiet_noise_db(m, recs, splits) if m.cfg.noise else None
            res[run] = dict(mean=round(float(np.mean(losses)), 3), median=round(float(np.median(losses)), 3),
                            n=len(losses), wind=None if wind is None else round(wind, 1))
            del m
            torch.cuda.empty_cache()
        except Exception as ex:  # an old checkpoint that no longer fits the code
            res[run] = dict(error=f"{type(ex).__name__}: {str(ex)[:300]}")
        print(run, json.dumps(res[run]), flush=True)
        OUT.write_text(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
