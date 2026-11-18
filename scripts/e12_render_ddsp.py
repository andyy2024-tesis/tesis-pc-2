"""Experiment 12: render whole recordings with a trained DDSP model, mix and every part, as vocoder training pairs.

Covers each recording's music span (first onset - 0.5 s to last onset + 2 s, the span the DDSP trains on) with
consecutive 3 s windows overlapping 0.5 s and crossfaded (like e6_adversarial.render_long). The real audio is
data/processed/sr24000/audio/<id>.npy; renders line up with it sample by sample.
Writes analysis/results/e12/renders/<run name>/<id>.npz: lo (first sample), parts (names), x (n_parts + 1, n) float16
with the mix first.
Usage: uv run python scripts/e12_render_ddsp.py e6/structured_v12_final_noise_strongD_rise [ids...]
(default: every recording the run trained on, extras included)
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

from tesis.data.gaps import REPO

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "analysis"))
from compare_runs import load  # noqa: E402
from e6_adversarial import L, SR, XFADE_S  # noqa: E402

OUT = REPO / "analysis" / "results" / "e12" / "renders"


@torch.no_grad()
def render_span(model, rec, lo, hi, seed=0):
    """Mix and parts over [lo, hi) from crossfaded 3 s windows."""
    xf = int(XFADE_S * SR)
    hop = L - xf
    ramp = np.ones(L, np.float32)
    ramp[:xf] = np.linspace(0, 1, xf)
    ramp[-xf:] = np.linspace(1, 0, xf)
    n = hi - lo
    acc, wsum = None, np.zeros(n + L, np.float32)
    names = None
    for i, a in enumerate(range(0, n, hop)):
        w0 = lo + a
        if w0 + L > len(rec.y):
            w0 = len(rec.y) - L
            a = w0 - lo
        y, parts = model(rec, int(w0), noise_seed=seed + i)
        if names is None:
            names = sorted(parts)
            acc = np.zeros((len(names) + 1, n + L), np.float32)
        acc[0, a:a + L] += y.cpu().numpy() * ramp
        for k, p in enumerate(names):
            acc[k + 1, a:a + L] += parts[p].detach().cpu().numpy().reshape(-1)[:L] * ramp
        wsum[a:a + L] += ramp
        if w0 + L >= hi:
            break
    m = wsum > 1e-3
    acc[:, m] /= wsum[m]
    return acc[:, :n], names


def main():
    run = sys.argv[1]
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    model, recs, _ = load(run, ids)
    res = REPO / "analysis" / "results"
    meta = res / run / "meta.json"
    base = json.loads(meta.read_text())["base"] if meta.exists() else run
    allids = ids + json.loads((res / "e4" / base / "config.json").read_text()).get("extra_ids", [])
    want = sys.argv[2:] or allids  # default: every recording the DDSP trained on
    out = OUT / run.replace("e6/", "").replace("/", "-")
    out.mkdir(parents=True, exist_ok=True)
    for rid in want:
        rec = recs[allids.index(rid)]
        lo = max(int(rec.onset.min()) - SR // 2, 0)  # onsets are in samples
        hi = min(int(rec.onset.max()) + 2 * SR, len(rec.y))
        x, names = render_span(model, rec, lo, hi)
        np.savez(out / f"{rid}.npz", lo=lo, parts=np.array(names), x=x.astype(np.float16))
        print(rid, f"{(hi - lo) / SR:.0f} s", names, flush=True)


if __name__ == "__main__":
    main()
