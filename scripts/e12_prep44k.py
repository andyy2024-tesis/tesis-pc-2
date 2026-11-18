"""Experiment 12, phase 5: the real audio at 44.1 kHz for the full-band vocoder (BigVGAN-v2 44 kHz).

Same recipe as scripts/prepare_gaps.py (tesis.data.gaps.load_audio: mono mean of the raw 48 kHz file, soxr_hq
resampling, int16 x 32767), so a time in the 24 kHz files is the same time here (sample n at 24 kHz = n * 147 / 80).
Notes, onsets and DDSP renders stay at 24 kHz. Recordings: the GAPS training split, the 20 evaluation recordings and
the 80 extra ones of the 100-recording DDSP (analysis/results/e12_more_ids.txt).
Writes data/processed/sr44100/audio/<id>.npy.
Usage: uv run python scripts/e12_prep44k.py [--workers 8]
"""
import argparse
from multiprocessing import get_context

import numpy as np
import pandas as pd

from tesis.data.gaps import PROCESSED, REPO, load_audio

OUT = PROCESSED / "sr44100" / "audio"
RES = REPO / "analysis" / "results"


def one(rid):
    f = OUT / f"{rid}.npy"
    if f.exists():
        return rid, "kept"
    y = load_audio(rid, 44100)
    n24 = len(np.load(PROCESSED / "sr24000" / "audio" / f"{rid}.npy", mmap_mode="r"))
    assert abs(len(y) - n24 * 147 / 80) < 4, (rid, len(y), n24)
    np.save(f, np.clip(y * 32767, -32768, 32767).astype(np.int16))
    return rid, f"{len(y) / 44100:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(PROCESSED / "sr24000" / "manifest.csv")
    ids = set(man[man.split == "train"].id) | set(pd.read_csv(RES / "e4_recordings.csv").id)
    ids |= set((RES / "e12_more_ids.txt").read_text().split())
    with get_context("spawn").Pool(a.workers) as pool:
        for k, (rid, msg) in enumerate(pool.imap_unordered(one, sorted(ids)), 1):
            print(f"{k}/{len(ids)} {rid} {msg}", flush=True)


if __name__ == "__main__":
    main()
