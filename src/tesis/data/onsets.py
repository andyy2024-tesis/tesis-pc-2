"""Refine GAPS MIDI onsets against the audio.

GAPS MIDI onsets usually sit within a few ms of the attack already (checked by eye on 001_mvswc), so this is a
small correction, not a search. Each chord (onset group) moves to the steepest rise of the high-passed amplitude
envelope (the pluck click) within +/-SEARCH of its MIDI time. The envelope and its derivative are centered, so
the estimate is not pulled early the way STFT log-flux is.
"""
import numpy as np
import scipy.signal as sps

GROUP_GAP = 0.03  # MIDI onsets closer than this form one chord
SEARCH = 0.02  # +/- search window (s)
HP_HZ = 1500
SMOOTH_S = 0.0015
DERIV_S = 0.001


def group_onsets(onset: np.ndarray) -> np.ndarray:
    """Group id per note (notes must be sorted by onset)."""
    gid = np.zeros(len(onset), dtype=np.int64)
    for i in range(1, len(onset)):
        gid[i] = gid[i - 1] + (onset[i] - onset[i - 1] > GROUP_GAP)
    return gid


BACKTRACK = 0.2  # attack start: where the envelope is back to 20% of its rise


def attack_envelope(y: np.ndarray, sr: int):
    """High-passed RMS envelope and its centered derivative, per sample."""
    sos = sps.butter(4, HP_HZ, "hp", fs=sr, output="sos")
    h = sps.sosfiltfilt(sos, y)
    k = max(int(SMOOTH_S * sr), 1)
    env = np.sqrt(np.convolve(h * h, np.ones(k) / k, mode="same"))
    d = int(DERIV_S * sr)
    rise = np.zeros_like(env)
    rise[d:-d] = env[2 * d:] - env[:-2 * d]
    return env, rise


def _backtrack(env, k, lo, sr):
    """From the steepest-rise sample k, walk back to where the burst starts."""
    base = np.percentile(env[lo:k + 1], 20)
    top = env[k:k + int(0.003 * sr)].max()
    thr = base + BACKTRACK * (top - base)
    i = k
    while i > lo and env[i] > thr:
        i -= 1
    return i


def refine_onsets(y: np.ndarray, sr: int, onset: np.ndarray):
    """Return (refined onset per note, confidence per note).

    Confidence is the peak rise divided by the median absolute rise in the window (higher = clearer attack).
    """
    env, rise = attack_envelope(y, sr)
    gid = group_onsets(onset)
    n_groups = gid.max() + 1
    g_time = np.array([onset[gid == g].min() for g in range(n_groups)])
    refined = onset.copy()
    conf = np.zeros(len(onset))
    for g in range(n_groups):
        members = gid == g
        t0 = g_time[g]
        # never search past the midpoint to a neighbouring chord
        lo_t = t0 - SEARCH if g == 0 else max(t0 - SEARCH, (g_time[g - 1] + t0) / 2)
        hi_t = t0 + SEARCH if g == n_groups - 1 else min(t0 + SEARCH, (g_time[g + 1] + t0) / 2)
        a, b = max(int(lo_t * sr), 0), min(int(hi_t * sr), len(rise))
        if b - a < 3:
            continue
        w = rise[a:b]
        k = int(np.argmax(w))
        refined[members] = _backtrack(env, a + k, a, sr) / sr
        conf[members] = w[k] / (np.median(np.abs(w)) + 1e-9)
    return refined, conf
