"""Does the catch-all noise ("wind") behave like a room's reverb tail?

On all held-out windows, per octave band (200 Hz - 6.4 kHz), 10 ms frames:
    D = model without its catch-all noise, N = catch-all noise, Y = original,
    R = max(Y - D, 0) per bin: the energy the deterministic part lacks.
Lag test: correlate log N (and log R) with log of D smeared by a one-sided exponential of time constant tau (0 = the
current frame only). A reverb tail follows the recent past (best tau ~ RT60 / 6.9, e.g. 0.5 s RT60 -> ~70 ms); noise
tied to the attacks follows the present (best tau ~ 0). Per recording, so different rooms show different taus.
Between partials (sharper): only bins away from every sounding partial (Rb = original there, minus the deterministic
part), 200 Hz - 6.4 kHz summed. Predictors: the playing level (D smeared by tau) and the attacks (positive frame-to-frame
rise of D, smeared by tau). Also the steadiness of Rb and N per recording: 10th percentile over median, in dB (0 dB = a
constant background, very negative = comes and goes with the music).
Also prints the RT60s learned by a run with the room reverb on (structured_v4_body_reverb), per recording.
Usage: uv run python analysis/room_check.py <run> [<run> ...]   (runs as in compare_runs.py)
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
from wind_map import partial_mask  # noqa: E402
from tesis.data.gaps import REPO  # noqa: E402

SR, L, NFFT, HOP = 24000, 72000, 1024, 240
DEV = os.environ.get("WIND_DEV", "cuda")
RES = REPO / "analysis" / "results"
TAUS = [0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]  # s
OCT = [200 * 2 ** i for i in range(6)]  # 200 ... 6400 Hz


def bands(x):
    x = torch.as_tensor(x, dtype=torch.float32, device=DEV)
    P = (torch.stft(x, NFFT, HOP, window=torch.hann_window(NFFT, device=DEV), return_complex=True).abs() ** 2)
    return P.cpu().numpy()


def smear(e, tau):
    if tau == 0:
        return e
    a = np.exp(-HOP / SR / tau)
    out = np.empty_like(e)
    acc = np.zeros(e.shape[0])
    for t in range(e.shape[1]):
        acc = a * acc + (1 - a) * e[:, t]
        out[:, t] = acc
    return out


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def learned_rt60(ids):
    folder = RES / "e4" / "structured_v4_body_reverb"
    if not folder.exists():
        return None
    w = torch.load(folder / "model.pt", map_location="cpu")["rec_reverb.weight"]
    nb = w.shape[1] // 2
    print(f"\nlearned room reverb (structured_v4_body_reverb), {nb} bands 60 Hz-10 kHz: raw RT60 parameters per recording")
    for rid, row in zip(ids, w.numpy()):
        print(f"  {rid}  level {np.round(row[:nb], 1)}  rt60-param {np.round(row[nb:], 2)}")


def main(runs):
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    fb = np.fft.rfftfreq(NFFT, 1 / SR)
    report = {}
    for run in runs:
        m, recs, splits = load(run, ids)
        m = m.to(DEV)
        per_rec = {}
        for r, (rec, (_, test)) in enumerate(zip(recs, splits)):
            cN, cR = np.zeros(len(TAUS)), np.zeros(len(TAUS))
            cB = {k: np.zeros(len(TAUS)) for k in ["level", "attack", "noise_level", "noise_attack"]}
            n_used, nb_used, steady_r, steady_n = 0, 0, [], []
            for j, w0 in enumerate(test):
                w0 = int(w0)
                with torch.no_grad():
                    _, parts = m(rec, w0, noise_seed=j)
                if "noise" not in parts:
                    continue
                n = parts["noise"].cpu().numpy()
                d = sum(v for k, v in parts.items() if k != "noise").cpu().numpy()
                Y, D, N = bands(rec.y[w0:w0 + L]), bands(d), bands(n)
                Rm = np.clip(Y - D, 0, None)
                # between partials, 200-6400 Hz
                band = (fb >= OCT[0]) & (fb < OCT[-1])
                t_abs = w0 + HOP * np.arange(Y.shape[1])
                rb = np.zeros(Y.shape[1])
                for fi, t in enumerate(t_abs):
                    act = np.where((rec.onset <= t) & (rec.end > t))[0]
                    f = rec.freqs[act].ravel()
                    sel = band & ~partial_mask(f[f < SR / 2], fb, 0.015) if len(act) else band
                    rb[fi] = Y[sel, fi].sum() - D[sel, fi].sum()
                rb = np.clip(rb, 1e-10, None)
                dT = D[band].sum(0)
                nT = N[band].sum(0)
                flux = np.clip(np.diff(dT, prepend=dT[0]), 0, None)
                for ti, tau in enumerate(TAUS):
                    lv = np.log(smear(dT[None], tau)[0] + 1e-10)
                    at = np.log(smear(flux[None], max(tau, 0.01))[0] + 1e-10)
                    cB["level"][ti] += corr(np.log(rb), lv)
                    cB["attack"][ti] += corr(np.log(rb), at)
                    cB["noise_level"][ti] += corr(np.log(nT + 1e-10), lv)
                    cB["noise_attack"][ti] += corr(np.log(nT + 1e-10), at)
                nb_used += 1
                steady_r.append(10 * np.log10(np.percentile(rb, 10) / np.median(rb)))
                steady_n.append(10 * np.log10(np.percentile(nT, 10) / (np.median(nT) + 1e-12) + 1e-12))
                for bi in range(len(OCT) - 1):
                    s = (fb >= OCT[bi]) & (fb < OCT[bi + 1])
                    dE, nE, rE = D[s].sum(0)[None], N[s].sum(0), Rm[s].sum(0)
                    ln, lr = np.log(nE + 1e-10), np.log(rE + 1e-10)
                    for ti, tau in enumerate(TAUS):
                        ls = np.log(smear(dE, tau)[0] + 1e-10)
                        cN[ti] += corr(ln, ls)
                        cR[ti] += corr(lr, ls)
                    n_used += 1
            if n_used:
                per_rec[ids[r]] = dict(noise=list(np.round(cN / n_used, 3)), missing=list(np.round(cR / n_used, 3)),
                                       **{f"between_{k}": list(np.round(v / nb_used, 3)) for k, v in cB.items()},
                                       steady_missing_db=float(np.mean(steady_r)), steady_noise_db=float(np.mean(steady_n)))
        report[run] = per_rec
        print(f"\n{run}: correlation with the deterministic part smeared by tau = {TAUS} s")
        allN = np.mean([v["noise"] for v in per_rec.values()], 0)
        allR = np.mean([v["missing"] for v in per_rec.values()], 0)
        print(f"  all recordings  noise {np.round(allN, 3)} (best {TAUS[int(np.argmax(allN))]})   "
              f"missing {np.round(allR, 3)} (best {TAUS[int(np.argmax(allR))]})")
        for k in ["level", "attack", "noise_level", "noise_attack"]:
            c = np.mean([v[f"between_{k}"] for v in per_rec.values()], 0)
            print(f"  between partials, {k:13s} {np.round(c, 3)} (best tau {TAUS[int(np.argmax(c))]})")
        print(f"  steadiness (10th pct / median): missing between partials "
              f"{np.mean([v['steady_missing_db'] for v in per_rec.values()]):.1f} dB, noise "
              f"{np.mean([v['steady_noise_db'] for v in per_rec.values()]):.1f} dB")
        for rid, v in per_rec.items():
            print(f"  {rid}  steady missing {v['steady_missing_db']:6.1f} dB  noise {v['steady_noise_db']:6.1f} dB  "
                  f"between: level best tau {TAUS[int(np.argmax(v['between_level']))]:5.3f}  "
                  f"attack best tau {TAUS[int(np.argmax(v['between_attack']))]:5.3f}")
        for rid, v in per_rec.items():
            print(f"  {rid}  noise best tau {TAUS[int(np.argmax(v['noise']))]:5.3f}  "
                  f"missing best tau {TAUS[int(np.argmax(v['missing']))]:5.3f}")
        del m
        torch.cuda.empty_cache()
    out = RES / "room_check.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**old, **report}, indent=1))
    learned_rt60(ids)


if __name__ == "__main__":
    main(sys.argv[1:])
