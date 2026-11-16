"""Does the catch-all noise play the attacks? (the "snare" heard in the noise stem)

On the 85 held-out windows, frames (20 ms, 5 ms hop) are split into
- attack: from 10 ms before to 60 ms after an onset;
- far: more than 200 ms after the last onset and more than 50 ms before the next one.
For each part (and the original), by band:
- lock (dB): mean power in attack frames over mean power in far frames. A steady bed gives ~0 dB; a part that hits at
  every pluck gives a large positive value. The original's lock is how much attack the real audio has.
- share of the attack: in attack frames, the noise's power over the power of all non-harmonic parts (noise, burst,
  click, body, squeak, floor). 1.0 = the noise plays the whole attack; the attack parts do nothing.
- attack level (dB): each part's power in attack frames relative to the original's.
Writes analysis/results/noise_attack.json (merged, one entry per run).
Usage: uv run python analysis/noise_attack.py <run> ...
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import RES, load  # noqa: E402

SR, HOP, NFFT, DEV = 24000, 120, 480, "cuda"
BANDS = {"0-1k": (0, 1000), "1-4k": (1000, 4000), "4-12k": (4000, 12000)}
PRE, ATTACK, FAR_AFTER, FAR_BEFORE = 0.010, 0.060, 0.200, 0.050
NON_HARMONIC = ["noise", "burst", "click", "body", "squeak", "floor"]
OUT = RES / "noise_attack.json"
WIN = torch.hann_window(NFFT, device=DEV)
FREQ = np.fft.rfftfreq(NFFT, 1 / SR)


def band_power(x):
    """(bands, frames) power."""
    z = torch.stft(torch.as_tensor(x, device=DEV), NFFT, HOP, window=WIN, return_complex=True).abs() ** 2
    z = z.cpu().numpy()
    return np.stack([z[(FREQ >= lo) & (FREQ < hi)].sum(0) for lo, hi in BANDS.values()])


def frame_masks(rec, w0, n_frames):
    t = (w0 + HOP * np.arange(n_frames)) / SR
    on = rec.onset / SR
    j = np.searchsorted(on, t, side="right")  # next onset index
    prev = np.where(j > 0, t - on[np.clip(j - 1, 0, None)], np.inf)
    nxt = np.where(j < len(on), on[np.clip(j, None, len(on) - 1)] - t, np.inf)
    attack = (prev < ATTACK) | (nxt < PRE)
    far = (prev > FAR_AFTER) & (nxt > FAR_BEFORE)
    return attack, far


@torch.no_grad()
def measure(run, ids):
    m, recs, splits = load(run, ids)
    acc = {}  # part -> {"attack": (bands,), "far": (bands,)} summed power
    n_att = n_far = 0
    for rec, (_, test) in zip(recs, splits):
        for j, w0 in enumerate(test):
            _, parts = m(rec, int(w0), noise_seed=j)
            L = len(next(iter(parts.values())))
            parts = {k: v.float() for k, v in parts.items()}
            parts["original"] = torch.as_tensor(rec.y[int(w0):int(w0) + L], device=DEV).float()
            P = {k: band_power(v) for k, v in parts.items()}
            att, far = frame_masks(rec, int(w0), next(iter(P.values())).shape[1])
            n_att += att.sum()
            n_far += far.sum()
            for k, p in P.items():
                a = acc.setdefault(k, {"attack": 0.0, "far": 0.0})
                a["attack"] = a["attack"] + p[:, att].sum(1)
                a["far"] = a["far"] + p[:, far].sum(1)
    del m
    torch.cuda.empty_cache()
    mean = {k: {"attack": v["attack"] / n_att, "far": v["far"] / n_far} for k, v in acc.items()}
    db = lambda x: 10 * np.log10(np.maximum(x, 1e-20))  # noqa: E731
    res = {"frames": {"attack": int(n_att), "far": int(n_far)}, "lock_db": {}, "attack_level_db": {}}
    for k, v in mean.items():
        res["lock_db"][k] = dict(zip(BANDS, np.round(db(v["attack"]) - db(v["far"]), 1).tolist()))
        res["attack_level_db"][k] = dict(zip(BANDS, np.round(db(v["attack"]) - db(mean["original"]["attack"]), 1).tolist()))
    nh = sum(mean[k]["attack"] for k in NON_HARMONIC if k in mean)
    if "noise" in mean:
        res["noise_share_of_attack"] = dict(zip(BANDS, np.round(mean["noise"]["attack"] / nh, 2).tolist()))
    return res


def main():
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for run in sys.argv[1:]:
        out[run] = measure(run, ids)
        OUT.write_text(json.dumps(out, indent=1))
        print(run, json.dumps(out[run]), flush=True)


if __name__ == "__main__":
    main()
