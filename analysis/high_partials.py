"""Is the energy the noise supplies really partials the strings miss? Isolated held-out notes, original vs the model's strings.

For notes with no other onset from 0.4 s before to 0.8 s after (up to 30 per recording, anywhere in it: held-out alone has
only 11, so this includes training audio): before and after the onset, each partial's level, its prominence (level at the
partial minus level halfway to the next partials), the level halfway between partials, and, when a partial clearly stands
out, its frequency in the original against the model's. The prominence of the model's noise part is the reference for what
a band without partials looks like.
Writes analysis/results/high_partials.json. Usage: uv run python analysis/high_partials.py <run> ...
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import RES, load  # noqa: E402

SR, NFFT = 24000, 4096
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000)]
TIMES = [-0.1, 0.05, 0.1, 0.2, 0.4, 0.7]
LEAD = int(0.2 * SR)
OUT = RES / "high_partials.json"
WIN = np.hanning(NFFT)
HZ = SR / NFFT


def spectrum(x, c):
    a = c - NFFT // 2
    if a < 0 or a + NFFT > len(x):
        return None
    return 20 * np.log10(np.abs(np.fft.rfft(x[a:a + NFFT] * WIN)) + 1e-7)


def peak(P, f, half):
    """(peak frequency in Hz, peak level) near f, searching half Hz each way, with parabolic interpolation."""
    lo, hi = max(1, int((f - half) / HZ)), min(len(P) - 2, int((f + half) / HZ) + 1)
    if hi <= lo:
        return None, None
    i = lo + int(np.argmax(P[lo:hi + 1]))
    a, b, c = P[i - 1], P[i], P[i + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c - 1e-9)
    return (i + d) * HZ, b


def main():
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for run in sys.argv[1:]:
        m, recs, _ = load(run, ids)
        L = int(m.cfg.win * SR)
        rows = []
        n_notes = 0
        for rec in recs:
            on = rec.onset
            n_used = 0
            for i in np.where((on > LEAD) & (on + L < len(rec.y)))[0]:
                others = np.delete(on, i)
                if n_used >= 30 or ((others > on[i] - 0.4 * SR) & (others < on[i] + 0.8 * SR)).any():
                    continue
                n_used += 1
                n_notes += 1
                w0 = int(on[i] - LEAD)
                with torch.no_grad():
                    _, parts = m(rec, w0, noise_seed=0)
                s = parts["strings"].cpu().numpy()
                noise = parts["noise"].cpu().numpy()
                y = rec.y[w0:w0 + L]
                f0 = rec.freqs[i][0]
                half = max(f0 / 3, 3 * HZ)
                for t in TIMES:
                    c = LEAD + int(t * SR)
                    Po, Ps, Pn = spectrum(y, c), spectrum(s, c), spectrum(noise, c)
                    if Po is None or Ps is None or on[i] + t * SR >= rec.end[i]:
                        continue
                    for k, f in enumerate(rec.freqs[i]):
                        if f + f0 >= SR / 2:
                            break
                        fo, lo = peak(Po, f, half)
                        fs, ls = peak(Ps, f, half)
                        if fo is None:
                            continue
                        mids = [int(round((f + d * f0 / 2) / HZ)) for d in (-1, 1)]
                        po, ps = lo - np.mean(Po[mids]), ls - np.mean(Ps[mids])
                        _, ln = peak(Pn, f, half)
                        rows.append((run, t, f, k, lo, ls, po, ps, fo - f, fs - f, np.mean(Po[mids]), np.mean(Pn[mids]), ln - np.mean(Pn[mids])))
        df = pd.DataFrame(rows, columns="run t f k lo ls po ps do ds mo mn pn".split())
        out = dict(n_notes=n_notes, n_rows=len(df), tables={})
        for b in BANDS:
            for t in TIMES:
                d = df[(df.f >= b[0]) & (df.f < b[1]) & (df.t == t)]
                if not len(d):
                    continue
                clear = d[(d.po > 10) & (d.ps > 10)]
                out["tables"][f"{b[0]}-{b[1]}@{t}"] = dict(
                    n=len(d), orig_level=round(float(d.lo.median()), 1), model_level=round(float(d.ls.median()), 1),
                    orig_prom=round(float(d.po.median()), 1), model_prom=round(float(d.ps.median()), 1), noise_prom=round(float(d.pn.median()), 1),
                    orig_between=round(float(d.mo.median()), 1), noise_between=round(float(d.mn.median()), 1),
                    share_orig_clear=round(float((d.po > 10).mean()), 2), share_model_clear=round(float((d.ps > 10).mean()), 2),
                    n_clear=len(clear),
                    freq_err_hz=round(float((clear.do - clear.ds).abs().median()), 1) if len(clear) else None,
                    freq_err_cents=round(float((1200 * np.log2((clear.f + clear.do) / (clear.f + clear.ds))).abs().median()), 1) if len(clear) else None)
        res[run] = out
        OUT.write_text(json.dumps(res, indent=1))
        print(run, "notes", n_notes, "rows", len(df), flush=True)
        del m, recs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
