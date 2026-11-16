"""Per-note loudness read directly from the (polyphonic) mix.

Energy in the note's partial bins, 20-105 ms after its onset. On GuitarSet this was within ~0.3 dB of per-string
truth even with 4+ notes sounding (experiment 3), better than inferring it by analysis-by-synthesis.
"""
import numpy as np

N_FFT = 2048


def partial_freqs(pitch: np.ndarray, tuning_cents: float, log_b, k_max: int = 60) -> np.ndarray:
    f0 = 440 * 2 ** ((pitch - 69 + tuning_cents / 100) / 12)
    B = np.exp(log_b[0] + log_b[1] * pitch)
    k = np.arange(1, k_max + 1)
    return f0[:, None] * k * np.sqrt(1 + B[:, None] * k ** 2)


def note_loudness_db(y: np.ndarray, sr: int, onset_s: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """dB energy in each note's partial bins (NaN if the window runs past the end)."""
    win = np.hanning(N_FFT)
    out = np.full(len(onset_s), np.nan)
    for i, (t, f) in enumerate(zip(onset_s, freqs)):
        a = int(round((t + 0.02) * sr))
        seg = y[a:a + N_FFT]
        if a < 0 or len(seg) < N_FFT:
            continue
        X = np.abs(np.fft.rfft(seg * win)) ** 2
        b = np.round(f[f < sr / 2] / sr * N_FFT).astype(int)
        out[i] = 10 * np.log10(X[b].sum() + 1e-12)
    return out


def note_timbre_db(y: np.ndarray, sr: int, onset_s: np.ndarray, freqs: np.ndarray, n: int = 16,
                   start: float = 0.02) -> np.ndarray:
    """(notes, n) level of each note's first n partials at its attack, in dB relative to their mean (0 if unknown).

    Power 20-105 ms after the onset minus power in the window ending 5 ms before it (what was already sounding), at the
    strongest bin within +-1 of each partial. Pluck position (comb-shaped dips) and nail vs flesh show up here.
    start: window start after the onset, s (a later start reads how the note's spectrum has changed by then).
    """
    win = np.hanning(N_FFT)
    out = np.zeros((len(onset_s), n))
    for i, (t, f) in enumerate(zip(onset_s, freqs)):
        a, p = int(round((t + start) * sr)), int(round((t - 0.005) * sr)) - N_FFT
        if p < 0 or a + N_FFT > len(y):
            continue
        X = np.abs(np.fft.rfft(y[a:a + N_FFT] * win)) ** 2
        P = np.abs(np.fft.rfft(y[p:p + N_FFT] * win)) ** 2
        b = np.round(f[:n] / sr * N_FFT).astype(int)
        ok = b + 1 < len(X)
        bb = np.clip(b[:, None] + np.arange(-1, 2), 0, len(X) - 1)
        e = np.clip(X[bb].max(1) - P[bb].max(1), 1e-3 * X[bb].max(), None)
        db = 10 * np.log10(e + 1e-12)
        if ok.sum() < 2:
            continue
        out[i, ok] = db[ok] - db[ok].mean()
    return out
