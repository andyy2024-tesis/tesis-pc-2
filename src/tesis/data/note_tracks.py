"""Frame-by-frame per-note features read from the (polyphonic) mix, one track per voice.

For each note, while it rings (onset to voice reuse / max ring), every hop:
- loudness: energy (dB) in the note's partial bins;
- pitch deviation (cents): weighted average of how far partials 1-8 (below 4 kHz) sit from their nominal frequency,
  found by parabolic peak interpolation within +/-60 cents; outliers dropped, median-smoothed.
Cleanup, using what a plucked string can do: after its onset a note's loudness may ripple (beating, up to RIPPLE_DB)
but not rise again, so jumps from other notes bleeding into its bins are clipped; pitch is trusted only while the
note is within TRUST_DB of its peak and not bleeding, otherwise the last trusted value is held.
This gives a resynthesis setting like the original DDSP (which reads f0 and loudness from the audio) for a polyphonic
instrument, with the notes' identities coming from the score.
"""
import warnings

import numpy as np
import scipy.ndimage as ndi

N_FFT = 2048
MAX_K_PITCH = 8
F_MAX_PITCH = 4000.0
SEARCH_CENTS = 60.0
RIPPLE_DB = 6.0
TRUST_DB = 20.0


def stft_mag(y, hop):
    win = np.hanning(N_FFT).astype(np.float32)
    pad = np.pad(y, (N_FFT // 2, N_FFT // 2))
    n = 1 + (len(pad) - N_FFT) // hop
    out = np.empty((n, N_FFT // 2 + 1), np.float32)
    for a in range(0, n, 2048):  # chunks to bound memory
        b = min(n, a + 2048)
        idx = np.arange(N_FFT)[None] + hop * np.arange(a, b)[:, None]
        out[a:b] = np.abs(np.fft.rfft(pad[idx] * win, axis=1))
    return out  # frame i is centered at sample i * hop


def voice_tracks(y, sr, hop, onset, end, voice, freqs, n_voices):
    """Returns (loud_db, dev_cents), each (n_voices, n_frames); NaN where a voice is silent."""
    X = stft_mag(y, hop)
    logX = np.log(X + 1e-9)
    n_frames = X.shape[0]
    bin_hz = sr / N_FFT
    loud = np.full((n_voices, n_frames), np.nan, np.float32)
    dev = np.full((n_voices, n_frames), np.nan, np.float32)
    for n in np.argsort(onset, kind="stable"):
        fa = int(np.ceil(onset[n] / hop))
        fe = min(int(np.ceil(end[n] / hop)), n_frames)
        if fe <= fa:
            continue
        fr = np.arange(fa, fe)
        f = freqs[n]
        b = np.round(f[f < sr / 2] / bin_hz).astype(int)
        raw = 10 * np.log10((X[fa:fe][:, b] ** 2).sum(1) + 1e-12)
        floor_so_far = np.minimum.accumulate(raw)
        clean = np.minimum(raw, floor_so_far + RIPPLE_DB)
        loud[voice[n], fr] = clean
        trusted = (clean >= clean.max() - TRUST_DB) & (raw - clean < 3.0)
        devs, ws = [], []
        for k in range(min(MAX_K_PITCH, len(f))):
            fk = f[k]
            if fk > F_MAX_PITCH:
                break
            lo = int(np.floor(fk * 2 ** (-SEARCH_CENTS / 1200) / bin_hz))
            hi = int(np.ceil(fk * 2 ** (SEARCH_CENTS / 1200) / bin_hz)) + 1
            if lo < 1 or hi >= X.shape[1] - 1 or hi - lo < 2:
                continue
            seg = logX[fa:fe, lo:hi]
            j = seg.argmax(1)
            jj = np.clip(j + lo, 1, X.shape[1] - 2)
            a, c, d = logX[fr, jj - 1], logX[fr, jj], logX[fr, jj + 1]
            den = a - 2 * c + d
            p = np.where(np.abs(den) > 1e-9, 0.5 * (a - d) / den, 0.0).clip(-0.5, 0.5)
            fm = (jj + p) * bin_hz
            edge = (j == 0) | (j == hi - lo - 1)  # peak on the window edge: no real partial there
            dk = np.where(edge, np.nan, 1200 * np.log2(fm / fk))
            devs.append(dk)
            ws.append(np.where(edge, 0.0, X[fr, jj]))
        if not devs:
            continue
        D, W = np.stack(devs), np.stack(ws)
        with warnings.catch_warnings():  # all-NaN frames are expected (no partial found)
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(W > 0, D, np.nan), axis=0)
        keep = (np.abs(D - med) < 25) & (W > 0)
        Wk = np.where(keep, W, 0.0)
        tot = Wk.sum(0)
        dv = np.where((tot > 0) & trusted, (np.nan_to_num(D) * Wk).sum(0) / np.maximum(tot, 1e-12), np.nan)
        good = np.where(np.isfinite(dv))[0]
        if len(good) >= 3:
            # hold the last trusted value forward (and the first one backward)
            last = np.maximum.accumulate(np.where(np.isfinite(dv), np.arange(len(dv)), -1))
            filled = np.where(last >= 0, dv[np.maximum(last, 0)], dv[good[0]])
            dev[voice[n], fr] = ndi.median_filter(filled, size=5, mode="nearest")
    return loud, dev
