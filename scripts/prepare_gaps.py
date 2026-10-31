"""Step 0: build the processed GAPS dataset.

Writes under data/processed/sr<SR>/:
  manifest.csv          usable recordings with split, tuning, bandwidth and note stats
  audio/<id>.npy        mono int16 audio at SR
  notes/<id>.csv        onset (refined), midi_onset, offset, pitch, voice, onset_conf
  isolated_notes.csv    single notes with nothing else starting nearby (for the one-note experiment)

Usage: uv run python scripts/prepare_gaps.py [--sr 24000] [--workers 12]
"""
import argparse
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd
import scipy.signal as sps

from tesis.data.gaps import GAPS_ROOT, PROCESSED, REPO, load_audio, load_metadata, load_notes
from tesis.data.onsets import group_onsets, refine_onsets
from tesis.data.tuning import tuning_vs_midi
from tesis.data.voices import allocate_voices

AUDIT = REPO / "analysis" / "results" / "gaps_audit.csv"
MIN_CONF = 5.0  # below this, keep the MIDI onset
ISO_BEFORE, ISO_AFTER = 1.0, 1.0  # s with no other onset around an isolated note


def bandwidth_hz(y, sr):
    """Highest frequency within 60 dB of the 1-4 kHz level (codec low-pass)."""
    f, p = sps.welch(y, fs=sr, nperseg=4096)
    db = 10 * np.log10(p + 1e-20)
    ref = np.median(db[(f > 1000) & (f < 4000)])
    return float(f[np.where(db > ref - 60)[0].max()])


def rms_db(x):
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)


def process(job):
    rid, sr, out = job
    y = load_audio(rid, sr)
    n = load_notes(rid)
    refined, conf = refine_onsets(y, sr, n.onset)
    onset = np.where(conf >= MIN_CONF, refined, n.onset)
    order = np.lexsort((n.pitch, onset))
    onset, midi_onset, offset, pitch, conf = onset[order], n.onset[order], n.offset[order], n.pitch[order], conf[order]
    offset = np.maximum(offset, onset + 0.01)
    voice, steals = allocate_voices(onset, offset, pitch)
    tuning, tuning_spread = tuning_vs_midi(y, sr, onset, offset, pitch)
    notes = pd.DataFrame(dict(onset=onset, midi_onset=midi_onset, offset=offset, pitch=pitch, voice=voice,
                              onset_conf=conf.round(2)))
    notes.to_csv(out / "notes" / f"{rid}.csv", index=False, float_format="%.5f")
    np.save(out / "audio" / f"{rid}.npy", np.clip(y * 32767, -32768, 32767).astype(np.int16))

    gid = group_onsets(onset)
    iso = []
    for i in range(len(onset)):
        if (gid == gid[i]).sum() != 1:
            continue
        others = np.delete(onset, i)
        if np.any((others > onset[i] - ISO_BEFORE) & (others < onset[i] + ISO_AFTER)):
            continue
        a = int(onset[i] * sr)
        pre, post = y[max(a - int(0.1 * sr), 0):a - int(0.005 * sr)], y[a:a + int(0.3 * sr)]
        if len(pre) < 10 or len(post) < 10:
            continue
        iso.append(dict(id=rid, onset=onset[i], pitch=int(pitch[i]), onset_conf=conf[i],
                        pre_minus_post_db=rms_db(pre) - rms_db(post)))
    stats = dict(id=rid, tuning_cents=tuning, tuning_spread_cents=tuning_spread, dur_s=len(y) / sr, n_notes=len(notes), voice_steals=steals,
                 frac_onset_refined=float((conf >= MIN_CONF).mean()),
                 refine_shift_median_ms=float(np.median(refined - n.onset) * 1000),
                 bandwidth_hz=bandwidth_hz(y, sr) if sr >= 44100 else np.nan)
    return stats, iso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = PROCESSED / f"sr{a.sr}"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "notes").mkdir(exist_ok=True)

    meta = load_metadata()
    audit = pd.read_csv(AUDIT)
    m = meta[["id", "split", "f-measure", "performer_name", "title"]].merge(
        audit[["id", "chroma_best_shift"]],
        on="id", how="left")
    keep = m.split.isin(["train", "test"]) & (m.chroma_best_shift.fillna(0) == 0)
    dropped = m[~keep & m.split.isin(["train", "test"])]
    print(f"recordings with a split: {m.split.isin(['train', 'test']).sum()}, dropped for transposed MIDI: {len(dropped)}")
    m = m[keep].drop(columns=["chroma_best_shift"])

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(process, [(rid, a.sr, out) for rid in m.id])
    stats = pd.DataFrame([r[0] for r in res])
    iso = pd.DataFrame([x for r in res for x in r[1]])
    m = m.merge(stats, on="id")
    if a.sr < 44100:
        m = m.drop(columns=["bandwidth_hz"])
    m.to_csv(out / "manifest.csv", index=False)
    iso.to_csv(out / "isolated_notes.csv", index=False)

    print(f"kept {len(m)} recordings, {m.dur_s.sum() / 3600:.2f} h  (train {m[m.split == 'train'].dur_s.sum() / 3600:.2f} h)")
    print(f"onsets refined: {m.frac_onset_refined.mean():.1%} of notes; median shift {m.refine_shift_median_ms.median():.1f} ms")
    print(f"voice steals per recording: median {m.voice_steals.median():.0f}, total {m.voice_steals.sum()} "
          f"of {m.n_notes.sum()} notes")
    print(f"tuning offset (cents): {m.tuning_cents.min():.0f} .. {m.tuning_cents.max():.0f}, "
          f"within-file spread median {m.tuning_spread_cents.median():.1f}")
    for t in (-20, -15, -10):
        print(f"isolated notes with the sound before the onset {-t} dB below the note: {(iso.pre_minus_post_db < t).sum()}"
              f" of {len(iso)}")


if __name__ == "__main__":
    main()
