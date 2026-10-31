"""Where does the "buzz" in chords come from?

On all held-out windows, per octave band (400 Hz - 6.4 kHz): the Hilbert envelope in dB at 1 kHz, cut into 250 ms
segments; buzz = power of the envelope's fluctuation at 30-150 Hz (fast roughness, faster than the flapping range), in
dB. Segments are grouped by how many notes sound (by the score): single (1 note) or chord (3 or more). Reported as the
model's buzz minus the original's (0 = as rough as the recording, positive = buzzier), for the full output, the output
without the catch-all noise, and each part on its own (strings, body, click, ...: the part's own envelope).
Usage: uv run python analysis/buzz_check.py <run> [<run> ...]   (runs as in compare_runs.py)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.signal as sps
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import load  # noqa: E402
from tesis.data.gaps import REPO  # noqa: E402

SR, L = 24000, 72000
ENV_SR, SEG = 1000, 250  # envelope rate, segment length (envelope samples)
DEV = os.environ.get("WIND_DEV", "cuda")
RES = REPO / "analysis" / "results"
EDGES = [400 * 2 ** i for i in range(5)]  # 400 ... 6400 Hz
SOS = [sps.butter(4, [lo, hi], "bandpass", fs=SR, output="sos") for lo, hi in zip(EDGES[:-1], EDGES[1:])]
MOD = sps.butter(4, [30, 150], "bandpass", fs=ENV_SR, output="sos")


def env_db(x):
    """(bands, T) envelope in dB at ENV_SR."""
    out = []
    for sos in SOS:
        e = np.abs(sps.hilbert(sps.sosfiltfilt(sos, x)))
        out.append(20 * np.log10(np.clip(sps.resample_poly(e, 1, SR // ENV_SR), 1e-6, None)))
    return np.stack(out)


def buzz(x, segs):
    """Mean over bands of the 30-150 Hz envelope power (dB^2) per segment, in dB; segs: list of start indices."""
    m = sps.sosfiltfilt(MOD, env_db(x), axis=-1)
    return np.array([10 * np.log10((m[:, s:s + SEG] ** 2).mean() + 1e-9) for s in segs])


def main(runs):
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    report = {}
    for run in runs:
        m, recs, splits = load(run, ids)
        m = m.to(DEV)
        acc = {}  # (kind, signal) -> list of model - original buzz
        for rec, (_, test) in zip(recs, splits):
            for j, w0 in enumerate(test):
                w0 = int(w0)
                segs, kinds = [], []
                for s in range(0, L // (SR // ENV_SR) - SEG + 1, SEG):
                    t = w0 + (s + SEG // 2) * (SR // ENV_SR)
                    n = int(((rec.onset <= t) & (rec.midi_end > t)).sum())
                    if n == 1 or n >= 3:
                        segs.append(s)
                        kinds.append("single" if n == 1 else "chord")
                if not segs:
                    continue
                with torch.no_grad():
                    y, parts = m(rec, w0, noise_seed=j)
                parts = {k: v.cpu().numpy() for k, v in parts.items()}
                sig = {"full": y.cpu().numpy(), **parts}
                if "noise" in parts:
                    sig["no noise"] = sig["full"] - parts["noise"]
                ref = buzz(rec.y[w0:w0 + L], segs)
                for name, x in sig.items():
                    if name in ("floor", "noise", "burst") or np.abs(x).max() < 1e-6:
                        continue
                    d = buzz(x, segs) - ref
                    for k, v in zip(kinds, d):
                        acc.setdefault((k, name), []).append(v)
        rep = {f"{k}/{n}": (round(float(np.mean(v)), 2), len(v)) for (k, n), v in sorted(acc.items())}
        report[run] = rep
        print(f"\n{run}: buzz (30-150 Hz envelope fluctuation), model minus original, dB")
        for name in sorted({n for _, n in acc}):
            s, c = acc.get(("single", name), [np.nan]), acc.get(("chord", name), [np.nan])
            print(f"  {name:10s} single {np.mean(s):+6.2f} ({len(s)})   chord {np.mean(c):+6.2f} ({len(c)})")
        del m
        torch.cuda.empty_cache()
    out = RES / "buzz_check.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**old, **report}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
