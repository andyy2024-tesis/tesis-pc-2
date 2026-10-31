"""Estimate f0 and the inharmonicity coefficient B of a plucked note from its spectrum.

Stiff string: f_k = k * f0 * sqrt(1 + B k^2)  =>  (f_k / k)^2 = f0^2 + f0^2 B k^2,
which is linear in k^2, so (f0, B) come from a weighted least-squares fit on picked partial peaks.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PartialFit:
    f0: float
    B: float
    k: np.ndarray  # partial numbers that were found
    freq: np.ndarray  # measured frequency of each found partial (Hz)
    amp: np.ndarray  # linear peak amplitude (sinusoid amplitude units)
    resid_cents: float  # weighted RMS misfit of the fit


def _pick_peak(mag, freqs, f, tol_hz):
    lo, hi = np.searchsorted(freqs, [f - tol_hz, f + tol_hz])
    if hi - lo < 3 or hi >= len(freqs) - 1:
        return None
    j = lo + int(np.argmax(mag[lo:hi]))
    if j in (lo, hi - 1):
        return None
    a, b, c = np.log(mag[j - 1:j + 2] + 1e-12)
    p = 0.5 * (a - c) / (a - 2 * b + c)
    return freqs[j] + p * freqs[1], float(np.exp(b - 0.25 * (a - c) * p))


def fit_partials(seg: np.ndarray, sr: int, f0_guess: float, f_max: float = None, snr_db: float = 20.0) -> PartialFit:
    """Fit (f0, B) to the partials of `seg` (a mono segment starting just after the attack)."""
    f_max = f_max or 0.45 * sr
    n = len(seg)
    win = np.hanning(n)
    n_fft = int(2 ** np.ceil(np.log2(n)) * 2)
    mag = np.abs(np.fft.rfft(seg * win, n_fft)) * 2 / win.sum()
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    floor = np.median(mag)
    f0, B = f0_guess, 0.0
    best = (np.zeros(0, int), np.zeros(0), np.zeros(0))
    k_all = np.arange(1, int(f_max / f0_guess) + 1)
    for k_max in (6, 15, 30, len(k_all)):  # widen the fit as the model gets better
        ks, fs, amps = [], [], []
        for k in k_all[:k_max]:
            fk = k * f0 * np.sqrt(1 + B * k * k)
            if fk > f_max:
                break
            pk = _pick_peak(mag, freqs, fk, tol_hz=max(0.25 * f0, 3 * freqs[1]) if k_max <= 6 else 0.12 * f0)
            if pk is None or 20 * np.log10(pk[1] / floor + 1e-12) < snr_db:
                continue
            ks.append(k), fs.append(pk[0]), amps.append(pk[1])
        if len(ks) < 3:
            break
        ks, fs, amps = np.array(ks), np.array(fs), np.array(amps)
        X = np.stack([np.ones_like(ks, dtype=float), ks.astype(float) ** 2], 1)
        w = np.log1p(amps / floor)
        coef, *_ = np.linalg.lstsq(X * w[:, None], (fs / ks) ** 2 * w, rcond=None)
        if coef[0] <= 0:
            break
        f0 = float(np.sqrt(coef[0]))
        B = float(max(coef[1] / coef[0], 0.0))
        best = (ks, fs, amps)
    ks, fs, amps = best
    if len(ks) == 0:
        return PartialFit(f0=f0, B=B, k=ks, freq=fs, amp=amps, resid_cents=np.nan)
    pred = ks * f0 * np.sqrt(1 + B * ks ** 2)
    resid = 1200 * np.log2(fs / pred)
    w = np.log1p(amps / floor)
    return PartialFit(f0=f0, B=B, k=ks, freq=fs, amp=amps, resid_cents=float(np.sqrt(np.sum(w * resid ** 2) / w.sum())))
