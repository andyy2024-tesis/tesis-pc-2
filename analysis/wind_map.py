"""Where does the catch-all noise ("wind") go, and what does the original have there that the synthesizer lacks?

On all held-out windows of a run, compares three magnitude spectrograms (2048-point FFT, 10 ms hop):
    Y = original, D = model without its catch-all noise (strings, transient, body, floor, reverb), N = catch-all noise.
1. Time (21 ms window): each frame is classed by the notes (12-40 ms before an onset, 0-10, 10-30, 30-60, 60-150 ms after one, note held,
   ringing after the written note ends, nothing written). Per class: noise level vs the original, and the share
   of all noise energy that lands there.
2. Frequency: per octave band, noise vs original and deterministic part vs original.
3. Between partials: while notes sound, the share of energy away from every sounding partial, in Y, D and N.
4. Sympathetic resonance check: in bins away from all sounding partials, is the original louder at the partials of
   the six open strings (E2 A2 D3 G3 B3 E4) than elsewhere? The model D is the control (it has no such mechanism).
Usage: uv run python analysis/wind_map.py <run> [<run> ...]   (runs as in compare_runs.py)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import load  # noqa: E402
from tesis.data.gaps import REPO  # noqa: E402

SR, L, NFFT, HOP = 24000, 72000, 2048, 240
RES = REPO / "analysis" / "results"
DEV = os.environ.get("WIND_DEV", "cuda")  # WIND_DEV=cpu when the GPU is busy training
OPEN = np.array([40, 45, 50, 55, 59, 64])
CLASSES = ["pre_40_12ms", "attack_0_10ms", "attack_10_30ms", "attack_30_60ms", "attack_60_150ms", "held", "ringing",
           "nothing"]
NFFT_T = 512  # time classes need a short window (21 ms); the 85 ms one smears the attack into "before the onset"
OCT = [50 * 2 ** i for i in range(9)]  # 50 Hz ... 12.8 kHz


def mag(x, nfft=NFFT):
    x = torch.as_tensor(x, dtype=torch.float32, device=DEV)
    return torch.stft(x, nfft, HOP, window=torch.hann_window(nfft, device=DEV), return_complex=True,
                      center=True).abs().cpu().numpy() ** 2  # (bins, frames) power


def frame_classes(rec, w0, n_frames):
    t = w0 + HOP * np.arange(n_frames)
    cls = np.full(n_frames, 7)
    ring = ((rec.onset[None] <= t[:, None]) & (rec.end[None] > t[:, None])).any(1)
    held = ((rec.onset[None] <= t[:, None]) & (rec.midi_end[None] > t[:, None])).any(1)
    d = (t[:, None] - rec.onset[None]) / SR
    since = np.where(d >= 0, d, np.inf).min(1)  # time since the latest onset
    cls[ring] = 6
    cls[held] = 5
    for ci, (lo, hi) in enumerate([(0.06, 0.15), (0.03, 0.06), (0.01, 0.03), (0.0, 0.01)]):
        cls[(since >= lo) & (since < hi)] = 4 - ci
    pre = ((d < -0.012) & (d >= -0.04)).any(1) & (since >= 0.06)
    cls[pre] = 0
    return cls


def partial_mask(freqs_active, fb, tol):
    """Bins within tol (relative) or 2 bins of any frequency in freqs_active."""
    m = np.zeros(len(fb), bool)
    bw = SR / NFFT
    for f in freqs_active:
        m |= np.abs(fb - f) <= max(tol * f, 2 * bw)
    return m


def main(runs):
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    man = pd.read_csv(REPO / "data" / "processed" / f"sr{SR}" / "manifest.csv").set_index("id")
    fb = np.fft.rfftfreq(NFFT, 1 / SR)
    report = {}
    for run in runs:
        m, recs, splits = load(run, ids)
        m = m.to(DEV)
        E = {c: np.zeros(3) for c in CLASSES}  # Y, D, N energy per time class
        B = np.zeros((len(OCT) - 1, 3))
        off = np.zeros(3)
        tot = np.zeros(3)
        symp, other, symp_d, other_d = [], [], [], []
        for r, (rec, (_, test)) in enumerate(zip(recs, splits)):
            tc = man.loc[ids[r], "tuning_cents"]
            f_open = (440 * 2 ** ((OPEN - 69 + tc / 100) / 12))[:, None] * np.arange(1, 13)
            f_open = f_open[f_open < 3000]
            for j, w0 in enumerate(test):
                w0 = int(w0)
                with torch.no_grad():
                    _, parts = m(rec, w0, noise_seed=j)
                n = parts.get("noise")
                n = n.cpu().numpy() if n is not None else np.zeros(L, np.float32)
                d = sum(v for k, v in parts.items() if k != "noise").cpu().numpy()
                Y, Dm, N = mag(rec.y[w0:w0 + L]), mag(d), mag(n)
                Yt, Dt, Nt = mag(rec.y[w0:w0 + L], NFFT_T), mag(d, NFFT_T), mag(n, NFFT_T)
                cls_t = frame_classes(rec, w0, Yt.shape[1])
                for ci, c in enumerate(CLASSES):
                    s = cls_t == ci
                    E[c] += [Yt[:, s].sum(), Dt[:, s].sum(), Nt[:, s].sum()]
                cls = frame_classes(rec, w0, Y.shape[1])
                for bi in range(len(OCT) - 1):
                    s = (fb >= OCT[bi]) & (fb < OCT[bi + 1])
                    B[bi] += [Y[s].sum(), Dm[s].sum(), N[s].sum()]
                # between partials, frames where something written is held (every 5th frame, for speed)
                for fi in np.where(cls == 5)[0][::5]:
                    t = w0 + HOP * fi
                    act = np.where((rec.onset <= t) & (rec.end > t))[0]
                    pm = partial_mask(rec.freqs[act].ravel()[rec.freqs[act].ravel() < SR / 2], fb, 0.015)
                    band = (fb > 60) & (fb < 3000)
                    sel = band & ~pm
                    off += [Y[sel, fi].sum(), Dm[sel, fi].sum(), N[sel, fi].sum()]
                    tot += [Y[band, fi].sum(), Dm[band, fi].sum(), N[band, fi].sum()]
                    om = partial_mask(f_open, fb, 0.004) & sel
                    rest = sel & ~partial_mask(f_open, fb, 0.012)
                    if om.any() and rest.any():
                        symp.append(10 * np.log10(Y[om, fi].mean() + 1e-12))
                        other.append(10 * np.log10(Y[rest, fi].mean() + 1e-12))
                        symp_d.append(10 * np.log10(Dm[om, fi].mean() + 1e-12))
                        other_d.append(10 * np.log10(Dm[rest, fi].mean() + 1e-12))
        db = lambda a, b: float(10 * np.log10(a / (b + 1e-20) + 1e-20))  # noqa: E731
        allN = sum(v[2] for v in E.values())
        rep = dict(
            time={c: dict(noise_vs_orig_db=db(v[2], v[0]), det_vs_orig_db=db(v[1], v[0]),
                          share_of_noise=float(v[2] / allN), share_of_orig=float(v[0] / sum(u[0] for u in E.values())))
                  for c, v in E.items()},
            octave={f"{OCT[i]}-{OCT[i + 1]}": dict(noise_vs_orig_db=db(B[i, 2], B[i, 0]),
                                                    det_vs_orig_db=db(B[i, 1], B[i, 0]),
                                                    share_of_noise=float(B[i, 2] / B[:, 2].sum()))
                    for i in range(len(OCT) - 1)},
            between_partials=dict(orig=float(off[0] / tot[0]), det=float(off[1] / tot[1]), noise=float(off[2] / tot[2]),
                                  noise_vs_orig_db=db(off[2], off[0]), det_vs_orig_db=db(off[1], off[0])),
            open_string_excess_db=dict(orig=float(np.mean(np.array(symp) - np.array(other))),
                                       model=float(np.mean(np.array(symp_d) - np.array(other_d))), n_frames=len(symp)),
        )
        report[run] = rep
        print(run, json.dumps(rep, indent=1), flush=True)
        del m
        torch.cuda.empty_cache()
    out = RES / "wind_map.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**old, **report}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
