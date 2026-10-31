"""Per-recording audit of GAPS: audio format/quality, tuning, MIDI-audio onset alignment, polyphony.

Usage: python analysis/gaps_audit.py <gaps_root> <out_csv> [--n N]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pretty_midi
import scipy.signal as sps
import soundfile as sf

SR = 48000
HOP = 240  # 5 ms


def load_mono(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, sr
    stereo_corr = np.nan
    if x.shape[1] == 2:
        a, b = x[:, 0], x[:, 1]
        stereo_corr = float(np.corrcoef(a[::10], b[::10])[0, 1])
    return x.mean(axis=1), x, stereo_corr


def bandwidth(y):
    f, p = sps.welch(y, fs=SR, nperseg=8192)
    db = 10 * np.log10(p + 1e-20)
    db = np.convolve(db, np.ones(9) / 9, mode="same")
    above = np.where(db > db.max() - 80)[0]
    return float(f[above.max()]) if len(above) else np.nan


def level_stats(y):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=1024)[0]
    db = 20 * np.log10(rms + 1e-9)
    return float(np.percentile(db, 95)), float(np.percentile(db, 5))


def midi_notes(path):
    m = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in m.instruments for n in inst.notes]
    notes.sort(key=lambda n: n.start)
    return notes


def chroma_shift(y, notes, dur):
    """Best semitone shift between audio chroma and MIDI chroma (catches whole-semitone retunings)."""
    y22 = librosa.resample(y, orig_sr=SR, target_sr=22050)
    hop = 2048
    ca = librosa.feature.chroma_cqt(y=y22, sr=22050, hop_length=hop, tuning=0.0)
    T = ca.shape[1]
    cm = np.zeros((12, T))
    for n in notes:
        a, b = int(n.start * 22050 / hop), int(n.end * 22050 / hop) + 1
        cm[n.pitch % 12, a:min(b, T)] += 1
    scores = {}
    for s in range(-3, 4):
        r = np.roll(cm, s, axis=0)
        num = (ca * r).sum()
        scores[s] = num / (np.linalg.norm(ca) * np.linalg.norm(r) + 1e-9)
    best = max(scores, key=scores.get)
    return best, scores[best], scores[0]


def tuning_vs_midi(y, notes, max_notes=400):
    """Median deviation (cents) of measured partials 1-3 from the MIDI-implied frequencies.

    (librosa.estimate_tuning segfaults in numba here, and this also catches tuning relative to the score.)
    """
    n_fft = 16384
    win = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    cands = [n for n in notes if n.end - n.start > 0.25]
    if len(cands) > max_notes:
        cands = [cands[i] for i in np.linspace(0, len(cands) - 1, max_notes).astype(int)]
    devs = []
    for n in cands:
        a = int((n.start + 0.06) * SR)
        seg = y[a:a + n_fft]
        if len(seg) < n_fft:
            continue
        mag = np.log(np.abs(np.fft.rfft(seg * win)) + 1e-9)
        f0 = 440 * 2 ** ((n.pitch - 69) / 12)
        for k in (1, 2, 3):
            fk = k * f0
            lo, hi = np.searchsorted(freqs, [fk * 2 ** (-0.5 / 12), fk * 2 ** (0.5 / 12)])
            if hi - lo < 3 or hi >= len(freqs) - 1:
                continue
            i = lo + int(np.argmax(mag[lo:hi]))
            if i in (lo, hi - 1):
                continue
            al, b, c = mag[i - 1], mag[i], mag[i + 1]
            p = 0.5 * (al - c) / (al - 2 * b + c)
            fpk = freqs[i] + p * (freqs[1] - freqs[0])
            devs.append(1200 * np.log2(fpk / fk))
    if not devs:
        return np.nan, np.nan
    devs = np.array(devs)
    return float(np.median(devs)), float(np.median(np.abs(devs - np.median(devs))))


def onset_alignment(y, notes):
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, n_fft=2048)
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="time", backtrack=False)
    # group MIDI onsets closer than 30 ms (chords)
    starts = np.array(sorted(n.start for n in notes))
    groups = [starts[0]]
    for s in starts[1:]:
        if s - groups[-1] > 0.03:
            groups.append(s)
    groups = np.array(groups)
    idx = np.searchsorted(peaks, groups)
    lo = peaks[np.clip(idx - 1, 0, len(peaks) - 1)]
    hi = peaks[np.clip(idx, 0, len(peaks) - 1)]
    nearest = np.where(np.abs(lo - groups) < np.abs(hi - groups), lo, hi)
    resid = nearest - groups
    ok = np.abs(resid) < 0.15
    r = resid[ok]
    return dict(
        n_onset_groups=len(groups),
        onset_matched_150ms=float(ok.mean()),
        onset_resid_median_ms=float(np.median(r) * 1000) if len(r) else np.nan,
        onset_absresid_median_ms=float(np.median(np.abs(r)) * 1000) if len(r) else np.nan,
        onset_resid_iqr_ms=float((np.percentile(r, 75) - np.percentile(r, 25)) * 1000) if len(r) else np.nan,
        onset_within_10ms=float((np.abs(resid) < 0.010).mean()),
        onset_within_25ms=float((np.abs(resid) < 0.025).mean()),
        onset_within_50ms=float((np.abs(resid) < 0.050).mean()),
    ), groups


def polyphony(notes, groups, dur):
    fs = 100
    T = int(dur * fs) + 1
    count = np.zeros(T, dtype=int)
    for n in notes:
        count[int(n.start * fs):int(n.end * fs)] += 1
    active = count[count > 0]
    # "isolated" onset groups: single note, no other onset in [-0.5 s, +1.0 s], nothing else sounding 50 ms after onset
    starts = np.array([n.start for n in notes])
    iso = 0
    for g in groups:
        same = np.abs(starts - g) < 0.03
        if same.sum() != 1:
            continue
        others = starts[~same]
        if np.any((others > g - 0.5) & (others < g + 1.0)):
            continue
        k = int((g + 0.05) * fs)
        if k < T and count[k] == 1:
            iso += 1
    return dict(
        poly_mean_when_active=float(active.mean()) if len(active) else np.nan,
        frac_time_mono=float((count == 1).sum() / max((count > 0).sum(), 1)),
        frac_time_silent=float((count == 0).mean()),
        n_isolated_notes=iso,
        mean_note_dur_s=float(np.mean([n.end - n.start for n in notes])),
    )


def audit(args):
    rid, root = args
    try:
        y, _, stereo_corr = load_mono(root / "audio" / f"{rid}.wav")
        dur = len(y) / SR
        notes = midi_notes(root / "midi" / f"{rid}.mid")
        hi, lo = level_stats(y)
        out = dict(
            id=rid,
            dur_s=dur,
            stereo_corr=stereo_corr,
            peak=float(np.abs(y).max()),
            clip_frac=float((np.abs(y) > 0.99).mean()),
            bandwidth_hz=bandwidth(y),
            loud_db_p95=hi,
            quiet_db_p5=lo,
            n_notes=len(notes),
            midi_end_s=max(n.end for n in notes),
            pitch_min=min(n.pitch for n in notes),
            pitch_max=max(n.pitch for n in notes),
        )
        out["tuning_cents"], out["tuning_spread_cents"] = tuning_vs_midi(y, notes)
        s, sc, sc0 = chroma_shift(y, notes, dur)
        out.update(chroma_best_shift=s, chroma_corr_best=float(sc), chroma_corr_0=float(sc0))
        al, groups = onset_alignment(y, notes)
        out.update(al)
        out.update(polyphony(notes, groups, dur))
        return out
    except Exception as e:  # keep going on bad files
        return dict(id=rid, error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ids = sorted(p.stem for p in (a.root / "audio").glob("*.wav") if (a.root / "midi" / f"{p.stem}.mid").exists())
    ids = ids[: a.n] if a.n else ids
    with Pool(a.workers) as pool:
        rows = pool.map(audit, [(i, a.root) for i in ids])
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.describe().T.to_string())


if __name__ == "__main__":
    main()
