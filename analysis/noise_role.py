"""What the catch-all noise does, and what separates the models heard as buzzy from the clean ones.

On the 85 held-out windows, for each run, in sustained frames (more than 100 ms after any onset):
- wobble: frame-to-frame change (dB, 10 ms frames) of the GRU correction of the sounding voices, by partial range;
- level (mean dB) of original / strings / noise, on the partials of the sounding notes and between them, by band;
- loss with the noise on and off, and how the noise level follows the original's level, within each recording.
Writes analysis/results/noise_role.json (merged, one entry per run).
Usage: uv run python analysis/noise_role.py <run> ...
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import RES, load  # noqa: E402
from tesis.synth.dsp import MultiScaleSpectralLoss  # noqa: E402

SR, HOP, NFFT, DEV = 24000, 240, 2048, "cuda"
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000)]
SUSTAIN = int(0.1 * SR)  # frames this soon after any onset are the attack, not the sustain
MARGIN = 4  # bins around a partial that count as "on the partial"
OUT = RES / "noise_role.json"
WIN = torch.hann_window(NFFT, device=DEV)
FREQ = np.fft.rfftfreq(NFFT, 1 / SR)


def db(x):
    z = torch.stft(x, NFFT, HOP, window=WIN, return_complex=True).abs()
    return 20 * torch.log10(z + 1e-7).cpu().numpy()[:, :-1]  # (bins, frames)


def masks(rec, w0, n_frames):
    """(bins, frames) on-partial mask, frames that are sustained with a note sounding, and which voice has a note."""
    t = w0 + HOP * np.arange(n_frames)
    on = np.zeros((len(FREQ), n_frames), bool)
    voiced = np.zeros((rec.voice.max() + 1, n_frames), bool)
    last_onset = np.full(n_frames, -10 ** 9)
    for i in np.where((rec.onset < t[-1]) & (rec.end > t[0]))[0]:
        act = (t >= rec.onset[i]) & (t < rec.end[i])
        bins = np.round(rec.freqs[i] / SR * NFFT).astype(int)
        bins = bins[bins < len(FREQ)]
        for b in bins:
            on[max(0, b - MARGIN):b + MARGIN + 1, act] = True
        voiced[rec.voice[i], act] = True
        after = t >= rec.onset[i]
        last_onset[after] = np.maximum(last_onset[after], rec.onset[i])
    return on, voiced, (t - last_onset > SUSTAIN) & voiced.any(0)


def within(a, b, who):
    """Correlation across windows after removing each recording's mean: does it follow the playing, not the recording?"""
    a, b, who = np.array(a), np.array(b), np.array(who)
    for w in np.unique(who):
        a[who == w] -= a[who == w].mean()
        b[who == w] -= b[who == w].mean()
    return round(float(np.corrcoef(a, b)[0, 1]), 2)


def main():
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    mss = MultiScaleSpectralLoss()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for run in sys.argv[1:]:
        m, recs, splits = load(run, ids)
        L = int(m.cfg.win * SR)
        cap = {}
        m.voice_out.register_forward_hook(lambda _, __, o: cap.__setitem__("v", o.detach()))
        acc = {k: {b: [[], []] for b in BANDS} for k in ("orig", "strings", "noise")}
        loss_on, loss_off, nl, ob, ol, who = [], [], [], [], [], []
        wob = {"k1-8": [], "k9-24": [], "k25-60": []}
        with torch.no_grad():
            for rec, (_, test) in zip(recs, splits):
                for j, w0 in enumerate(test):
                    w0 = int(w0)
                    out, parts = m(rec, w0, noise_seed=j)
                    y = torch.tensor(rec.y[w0:w0 + L], device=DEV)
                    noise = parts.get("noise")
                    dry = out - noise if noise is not None else out
                    loss_on.append(mss(out[None], y[None]).item())
                    loss_off.append(mss(dry[None], y[None]).item())
                    So, Ss = db(y), db(parts["strings"])
                    Sn = db(noise) if noise is not None else None
                    on, voiced, sus = masks(rec, w0, So.shape[1])
                    if sus.sum() == 0:
                        continue
                    for b in BANDS:
                        band = (FREQ >= b[0]) & (FREQ < b[1])
                        for k, S in (("orig", So), ("strings", Ss), ("noise", Sn)):
                            if S is None:
                                continue
                            sub, mask = S[band][:, sus], on[band][:, sus]
                            if mask.any():
                                acc[k][b][0].append(float(sub[mask].mean()))
                            if (~mask).any():
                                acc[k][b][1].append(float(sub[~mask].mean()))
                    between = np.where(on[:, sus], np.nan, So[:, sus])
                    ob.append(float(np.nanmean(between)))
                    ol.append(float(So[:, sus].mean()))
                    who.append(rec.idx)
                    if Sn is not None:
                        nl.append(float(Sn[:, sus].mean()))
                    # wobble of the correction on the sounding voices, sustained frames only
                    v = cap["v"].cpu().numpy() * 8.686  # log amplitude -> dB; (V, F, K)
                    F = min(v.shape[1], len(sus))
                    d = np.abs(np.diff(v[:, :F], axis=1))
                    keep = voiced[:v.shape[0], :F][:, 1:] & sus[1:F][None] & sus[:F - 1][None]
                    if keep.any():
                        for name, (a, c) in {"k1-8": (0, 8), "k9-24": (8, 24), "k25-60": (24, 60)}.items():
                            wob[name].append(float(np.sqrt((d[:, :, a:c] ** 2).mean(-1)[keep].mean())))
        res[run] = dict(
            loss_on=round(float(np.mean(loss_on)), 3), loss_off=round(float(np.mean(loss_off)), 3),
            wobble={k: round(float(np.mean(v)), 3) for k, v in wob.items()},
            levels={k: {f"{b[0]}-{b[1]}": dict(on=round(float(np.mean(v[0])), 1) if v[0] else None, between=round(float(np.mean(v[1])), 1) if v[1] else None)
                        for b, v in acc[k].items()} for k in acc},
            corr_noise_between=within(nl, ob, who) if nl else None,
            corr_noise_overall=within(nl, ol, who) if nl else None)
        OUT.write_text(json.dumps(res, indent=1))
        print(run, json.dumps({k: res[run][k] for k in ("loss_on", "loss_off", "wobble")}), flush=True)
        del m, recs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
