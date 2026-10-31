"""Fast loudness fluctuation ("flapping") in rendered excerpts.

Per log-spaced band (100 Hz - 8 kHz), the envelope in dB at 500 Hz; ratio of its fluctuation energy at 4-50 Hz
(fast wobble) to 0-4 Hz (note-level changes), averaged over bands, in dB. Higher = more flapping.
Usage: uv run python analysis/flapping_check.py <audio dir>
"""
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf


def flapping_db(x, sr=24000):
    out = []
    edges = np.geomspace(100, 8000, 9)
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = sps.butter(4, [lo, hi], "bandpass", fs=sr, output="sos")
        env = np.clip(sps.resample_poly(np.abs(sps.hilbert(sps.sosfilt(b, x))), 1, sr // 500), 1e-6, None)
        le = 20 * np.log10(env)
        f, P = sps.welch(le - le.mean(), fs=500, nperseg=256)
        out.append(P[(f >= 4) & (f <= 50)].sum() / max(P[(f > 0) & (f < 4)].sum(), 1e-12))
    return 10 * np.log10(np.mean(out))


def main(d):
    files = sorted(Path(d).glob("*.wav"))
    ids = sorted({f.name.split("_")[0] + "_" + f.name.split("_")[1] for f in files})
    for rid in ids:
        row = {f.stem[len(rid) + 1:]: round(flapping_db(sf.read(f)[0]), 1) for f in files if f.name.startswith(rid)}
        print(rid, row)


if __name__ == "__main__":
    main(sys.argv[1])
