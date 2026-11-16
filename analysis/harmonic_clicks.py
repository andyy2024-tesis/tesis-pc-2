"""Where do the short high clicks in the harmonics (strings part) come from?

The user hears light high-pitched clicks in the strings stem of M2 + timbre 32 + adv (306, 277), also with the noise off.
Detector: STFT (256 samples, 2 ms hop) of the strings part; power above 4 kHz per frame; a click is a frame more than
CLICK_DB above the median of the surrounding +-40 ms and above an absolute floor (relative to the excerpt's HF RMS).
Each click is located relative to the nearest onset, note end (voice cut) and MIDI end.
Ablations at render time (same weights): no detune between the two polarizations (two_decay phase), no GRU correction
(delta = 0), no measured pitch deviation (track_dev = 0).
Usage: uv run python analysis/harmonic_clicks.py <run> [<run> ...]
"""
import copy
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import RES, load  # noqa: E402
from e6_adversarial import LISTEN_IDS, SR  # noqa: E402
from render_stems import render_parts  # noqa: E402

NFFT, HOP = 256, 48
CLICK_DB, CTX = 10.0, 0.040
FREQ = np.fft.rfftfreq(NFFT, 1 / SR)


def hf_power(x):
    z = torch.stft(torch.as_tensor(x), NFFT, HOP, window=torch.hann_window(NFFT), return_complex=True).abs() ** 2
    return z[torch.as_tensor(FREQ >= 4000)].sum(0).numpy()


def clicks(x):
    p = hf_power(x) + 1e-20
    n = int(CTX * SR / HOP)
    db = 10 * np.log10(p)
    med = np.array([np.median(db[max(0, i - n):i + n + 1]) for i in range(len(db))])
    ref = 10 * np.log10(np.mean(p))
    hit = (db - med > CLICK_DB) & (db > ref - 20)
    # one event per run of frames
    idx = np.where(hit & ~np.r_[False, hit[:-1]])[0]
    return idx * HOP / SR, (db - med)[idx]


def nearest(t, ev):
    if len(ev) == 0:
        return np.full(len(t), np.inf)
    d = t[:, None] - ev[None]
    return d[np.arange(len(t)), np.abs(d).argmin(1)]


def ablate(m, kind):
    m2 = copy.deepcopy(m)
    with torch.no_grad():
        if kind == "no_detune" and hasattr(m2, "pol_mlp"):
            m2.pol_mlp[-1].weight[2] = 0
            m2.pol_mlp[-1].bias[2] = 0
        if kind == "no_delta":
            m2.voice_out.weight.zero_()
            m2.voice_out.bias.zero_()
    return m2


def main():
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    out = {}
    for run in sys.argv[1:]:
        m, recs, splits = load(run, ids)
        res = {}
        for kind in ["base", "no_detune", "no_delta", "no_dev"]:
            mm = ablate(m, kind) if kind in ("no_detune", "no_delta") else m
            tot, rows = 0, []
            for rid in LISTEN_IDS:
                r = ids.index(rid)
                rec, w0 = recs[r], int(splits[r][1][0])
                saved = rec.track_dev
                if kind == "no_dev" and rec.track_dev is not None:
                    rec.track_dev = np.zeros_like(saved)
                x = render_parts(mm, rec, w0)["strings"]
                rec.track_dev = saved
                t, h = clicks(x)
                ta = w0 / SR + t
                d_on = nearest(ta, rec.onset / SR)
                d_end = nearest(ta, rec.end / SR)
                d_mid = nearest(ta, rec.midi_end / SR)
                away = np.abs(d_on) > 0.03
                tot += away.sum()
                for i in np.where(away)[0]:
                    rows.append(dict(rec=rid, t=round(float(t[i]), 3), height_db=round(float(h[i]), 1),
                                     to_onset=round(float(d_on[i]), 3), to_end=round(float(d_end[i]), 3),
                                     to_midi_end=round(float(d_mid[i]), 3)))
            res[kind] = dict(clicks_away_from_onsets=int(tot), events=rows)
            print(run, kind, tot, flush=True)
        out[run] = res
        del m
        torch.cuda.empty_cache()
    path = RES / "harmonic_clicks.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(out)
    path.write_text(json.dumps(cur, indent=1))


if __name__ == "__main__":
    main()
