"""Is the catch-all noise only the room? Its level relative to each recording's own measured background.

Background: per FFT bin, the 10th percentile of the recording's power away from all sounding partials while the piece
plays (tesis.model.guitar_net.room_background_bins; minimum statistics). On the 85 held-out windows, the noise part's
power per bin and frame (same STFT) over that background, in dB: median over frames and bins of each band, and the share
of (frame, bin) cells more than 6 dB above it. ~0 dB and a small share: the noise is room tone ("we won"); well above:
it plays something else (gusts, attacks). Writes analysis/results/noise_vs_room.json (merged).
Usage: uv run python analysis/noise_vs_room.py <run> ...
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import RES, load  # noqa: E402
from tesis.data.gaps import PROCESSED  # noqa: E402
from tesis.model.guitar_net import room_background_bins  # noqa: E402

SR, NFFT, HOP = 24000, 2048, 512
BANDS = {"0-700": (40, 700), "700-4k": (700, 4000), "4k-11k": (4000, 11000)}
OUT = RES / "noise_vs_room.json"
CACHE = PROCESSED / f"sr{SR}" / "room_background"


def background(rid, rec):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{rid}.npy"
    if f.exists():
        return np.load(f)
    fb, bg = room_background_bins(rec.y, SR, rec.onset, rec.end, rec.freqs)
    np.save(f, bg)
    return bg


@torch.no_grad()
def measure(run, ids):
    m, recs, splits = load(run, ids)
    if not m.cfg.noise:
        return None
    fb = np.fft.rfftfreq(NFFT, 1 / SR)
    rel = {b: [] for b in BANDS}
    for rid, rec, (_, test) in zip(ids, recs, splits):
        if not len(test):
            continue
        bg = background(rid, rec)
        for j, w0 in enumerate(test):
            _, parts = m(rec, int(w0), noise_seed=j)
            x = parts["noise"].float().cpu()
            P = (torch.stft(x, NFFT, HOP, window=torch.hann_window(NFFT), return_complex=True).abs() ** 2).numpy()
            d = 10 * np.log10(np.maximum(P, 1e-20) / np.maximum(bg, 1e-20)[:, None])
            for b, (lo, hi) in BANDS.items():
                rel[b].append(d[(fb >= lo) & (fb < hi)].ravel())
    del m
    torch.cuda.empty_cache()
    out = {}
    for b, v in rel.items():
        v = np.concatenate(v)
        out[b] = dict(median_db=round(float(np.median(v)), 1), share_above_6db=round(float((v > 6).mean()), 2))
    return out


def main():
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for run in sys.argv[1:]:
        res[run] = measure(run, ids)
        OUT.write_text(json.dumps(res, indent=1))
        print(run, json.dumps(res[run]), flush=True)


if __name__ == "__main__":
    main()
