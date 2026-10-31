"""User's hypothesis: recordings with wide dynamics / hard plucks are the ones the model gets wrong, because a
harder pluck changes the timbre (brighter), not just the level.

Per recording (the 20 of experiment 4):
- dynamic range: p90 - p10 of per-note loudness (dB, read from the mix);
- brightness-loudness slope: per note, brightness = energy of partials 6-20 minus partials 1-5 (dB), 20-105 ms
  after onset; regressed on loudness with pitch held fixed (pitch as categorical), notes with <=2 sounding only;
- held-out spectral loss of the round-2 structured model (with body response) per recording.

Usage: uv run python analysis/dynamics_hypothesis.py
"""
import numpy as np
import pandas as pd
import scipy.stats as st

from tesis.data.gaps import PROCESSED, REPO
from tesis.data.loudness import N_FFT, note_loudness_db, partial_freqs
from tesis.synth.piece import LOG_B

SR = 24000
D = PROCESSED / f"sr{SR}"
R = REPO / "analysis" / "results"


def band_db(y, onset_s, freqs, k_lo, k_hi):
    win = np.hanning(N_FFT)
    out = np.full(len(onset_s), np.nan)
    for i, (t, f) in enumerate(zip(onset_s, freqs)):
        a = int(round((t + 0.02) * SR))
        seg = y[a:a + N_FFT]
        if len(seg) < N_FFT:
            continue
        X = np.abs(np.fft.rfft(seg * win)) ** 2
        ff = f[k_lo - 1:k_hi]
        b = np.round(ff[ff < SR / 2] / SR * N_FFT).astype(int)
        out[i] = 10 * np.log10(X[b].sum() + 1e-12) if len(b) else np.nan
    return out


def main():
    ids = pd.read_csv(R / "e4_recordings.csv").id.tolist()
    man = pd.read_csv(D / "manifest.csv").set_index("id")
    tm = pd.read_csv(R / "e4" / "targeted_structured_v2-structured_v2_bodyir-structured_v2_bodyir_nonoise-ddsp_v2-ddsp_v2_nonoise.csv")
    loss = tm.groupby("id").structured_v2_bodyir_loss.median()
    rows = []
    for rid in ids:
        y = np.load(D / "audio" / f"{rid}.npy").astype(np.float32) / 32768
        n = pd.read_csv(D / "notes" / f"{rid}.csv")
        f = partial_freqs(n.pitch.values, man.loc[rid, "tuning_cents"], LOG_B)
        loud = note_loudness_db(y, SR, n.onset.values, f)
        bright = band_db(y, n.onset.values, f, 6, 20) - band_db(y, n.onset.values, f, 1, 5)
        on, off = n.onset.values, n.offset.values
        sounding = np.array([((on <= t) & (off > t)).sum() for t in on])
        d = pd.DataFrame(dict(pitch=n.pitch, loud=loud, bright=bright, ok=sounding <= 2)).dropna()
        d = d[d.ok]
        # within-pitch slope: demean loudness and brightness per pitch, then regress
        dm = d.groupby("pitch")[["loud", "bright"]].transform(lambda x: x - x.mean())
        keep = d.groupby("pitch").pitch.transform("size") >= 3
        slope = np.polyfit(dm.loud[keep], dm.bright[keep], 1)[0] if keep.sum() > 20 else np.nan
        rows.append(dict(id=rid, title=man.loc[rid, "title"], dyn_range_db=np.nanpercentile(loud, 90) - np.nanpercentile(loud, 10),
                         bright_per_10db=10 * slope, n_notes_used=int(keep.sum()), heldout_loss=loss.get(rid, np.nan),
                         onsets_refined=man.loc[rid, "frac_onset_refined"]))
    df = pd.DataFrame(rows).set_index("id")
    pd.set_option("display.width", 200)
    print(df.round(2).to_string())
    m = df.heldout_loss.notna()
    for c in ["dyn_range_db", "bright_per_10db", "onsets_refined"]:
        r = st.spearmanr(df[c][m], df.heldout_loss[m])
        print(f"Spearman({c}, held-out loss) = {r.statistic:+.2f} (p={r.pvalue:.2f}, n={m.sum()})")
    print(f"\nMedian brightness change per +10 dB of loudness (same pitch): {df.bright_per_10db.median():+.2f} dB")
    df.to_csv(R / "dynamics_hypothesis.csv")


if __name__ == "__main__":
    main()
