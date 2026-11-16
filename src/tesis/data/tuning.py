"""Per-recording tuning offset relative to the MIDI pitches."""
import numpy as np


def tuning_vs_midi(y: np.ndarray, sr: int, onset: np.ndarray, offset: np.ndarray, pitch: np.ndarray,
                   max_notes: int = 400):
    """Median deviation (cents) of measured partials 1-3 from the MIDI-implied frequencies, and its spread (MAD).

    Partials are picked 60 ms after each onset, within +/-50 cents of the expected frequency, with parabolic
    interpolation of the log-magnitude peak. Low partials keep inharmonicity out of the estimate.
    """
    n_fft = 16384 if sr > 30000 else 8192
    win = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    cands = np.where(offset - onset > 0.25)[0]
    if len(cands) > max_notes:
        cands = cands[np.linspace(0, len(cands) - 1, max_notes).astype(int)]
    devs = []
    for i in cands:
        a = int((onset[i] + 0.06) * sr)
        seg = y[a:a + n_fft]
        if len(seg) < n_fft:
            continue
        mag = np.log(np.abs(np.fft.rfft(seg * win)) + 1e-9)
        f0 = 440 * 2 ** ((pitch[i] - 69) / 12)
        for k in (1, 2, 3):
            fk = k * f0
            lo, hi = np.searchsorted(freqs, [fk * 2 ** (-0.5 / 12), fk * 2 ** (0.5 / 12)])
            if hi - lo < 3 or hi >= len(freqs) - 1:
                continue
            j = lo + int(np.argmax(mag[lo:hi]))
            if j in (lo, hi - 1):
                continue
            al, b, c = mag[j - 1], mag[j], mag[j + 1]
            p = 0.5 * (al - c) / (al - 2 * b + c)
            devs.append(1200 * np.log2((freqs[j] + p * freqs[1]) / fk))
    if not devs:
        return 0.0, np.nan
    devs = np.array(devs)
    med = float(np.median(devs))
    return med, float(np.median(np.abs(devs - med)))
