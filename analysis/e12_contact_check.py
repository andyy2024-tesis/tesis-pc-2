"""Experiment 12: is the final DDSP's "contact" part contact, or a catch-all noise? (the user, 2026-11-19: on the page
it sounds like noise, and it may be why the final DDSP is less real by KAD than the earlier favourite).

contact = one noise per recording (16-band colour) times a per-voice level the network sets every frame; nothing ties
it to the notes, so it can become a second catch-all. On the pair renders of the 20 evaluation recordings
(analysis/results/e12/renders/<run>/, the whole music span, mix = sum of parts), per part, 10 ms frames:
- level: energy relative to the mix, overall and in the quietest 20% of frames of the mix (where "wind" lives);
- timing: share of the part's energy within 50 ms of a note onset or of a note offset (from the note files), divided
  by the share of frames there: ~1 = unrelated to the notes (a catch-all), well above 1 = tied to touching/plucking;
- correlation of its frame level (dB) with the noise part's and the strings part's;
- spectral centroid.
The burst part (the pluck transient) is the reference for "tied to onsets". KAD of the mix without a part:
analysis/e12_kad_clips.py ddsp-no<part>.
Usage: uv run python analysis/e12_contact_check.py [run]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from e12_vocoder import D, EVAL_IDS, RES, SR  # noqa: E402

HOP = SR // 100
NEAR = 0.05


def frames(x):
    n = len(x) // HOP
    return (x[:n * HOP].astype(np.float64) ** 2).reshape(n, HOP).mean(1)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "structured_v12_final_learnedB_noise_strongD_rise"
    rows, corr = [], []
    for rid in EVAL_IDS:
        f = RES / "e12" / "renders" / run / f"{rid}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        lo, names = int(z["lo"]), list(z["parts"])
        e = {"mix": frames(z["x"][0])}
        for i, p in enumerate(names):
            e[p] = frames(z["x"][1 + i])
        n = len(e["mix"])
        t = (lo + HOP * np.arange(n) + HOP / 2) / SR
        notes = pd.read_csv(D / "notes" / f"{rid}.csv")
        def near(times):
            m = np.zeros(n, bool)
            idx = np.searchsorted(t, np.sort(times))
            for k, tt in zip(idx, np.sort(times)):
                a, b = np.searchsorted(t, tt - NEAR), np.searchsorted(t, tt + NEAR)
                m[a:b] = True
            return m
        on, off = near(notes.onset.values), near(notes.offset.values)
        quiet = e["mix"] <= np.quantile(e["mix"], 0.2)
        for p in names:
            ep = e[p]
            tot = ep.sum() + 1e-30
            rows.append(dict(id=rid, part=p, level_db=10 * np.log10(tot / e["mix"].sum()),
                             quiet_db=10 * np.log10(ep[quiet].sum() / (e["mix"][quiet].sum() + 1e-30) + 1e-30),
                             onset_ratio=(ep[on].sum() / tot) / on.mean(), offset_ratio=(ep[off].sum() / tot) / off.mean()))
        if "contact" in e:
            lc = 10 * np.log10(e["contact"] + 1e-20)
            corr.append(dict(id=rid, noise=np.corrcoef(lc, 10 * np.log10(e["noise"] + 1e-20))[0, 1],
                             strings=np.corrcoef(lc, 10 * np.log10(e["strings"] + 1e-20))[0, 1]))
        # spectral centroid of each part over the whole render
        for i, p in enumerate(names):
            X = np.abs(np.fft.rfft(z["x"][1 + i][:SR * 60].astype(np.float64))) ** 2
            fr = np.fft.rfftfreq(min(len(z["x"][0]), SR * 60), 1 / SR)
            rows[-len(names) + i]["centroid_hz"] = float((X * fr).sum() / (X.sum() + 1e-30))
    df = pd.DataFrame(rows)
    print(f"{run}: {df.id.nunique()} recordings; medians over recordings")
    print(df.groupby("part")[["level_db", "quiet_db", "onset_ratio", "offset_ratio", "centroid_hz"]].median()
          .round(2).to_string())
    if corr:
        c = pd.DataFrame(corr)
        print(f"contact level vs noise level: r = {c.noise.median():.2f}; vs strings level: r = {c.strings.median():.2f}")


if __name__ == "__main__":
    main()
