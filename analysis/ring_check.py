"""Is the original's extra energy at open-string frequencies from open strings that are still ringing?

Same measure as wind_map.py (original only, no model): in held-note frames, bins away from every sounding partial,
level at open-string partials minus elsewhere. Frames are split by whether that open string was played (as a note at
exactly its pitch) in the last 4 s but its note has already been ended by the synthesizer (voice stealing / ring
limit), i.e. whether the synthesizer cuts a string the player let ring. Also: how often notes are cut by the next
note on their voice before ringing 1 s, per recording.
Usage: uv run python analysis/ring_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
import e4_train as e  # noqa: E402
from tesis.data.gaps import REPO  # noqa: E402
from tesis.model.guitar_net import NetConfig  # noqa: E402
from wind_map import OPEN, partial_mask  # noqa: E402

SR, L, NFFT, HOP = 24000, 72000, 2048, 240


def main():
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    man = pd.read_csv(REPO / "data" / "processed" / f"sr{SR}" / "manifest.csv").set_index("id")
    cfg = NetConfig(sr=SR, body_ir=True, audio_feats=True)
    recs, splits = e.load_recordings(ids, cfg)
    fb = np.fft.rfftfreq(NFFT, 1 / SR)
    win = np.hanning(NFFT)
    rows = []
    cut = []
    for r, (rec, (_, test)) in enumerate(zip(recs, splits)):
        ring = (rec.end - rec.onset) / SR
        nxt_same_voice = ring < cfg.max_ring - 1e-3
        cut.append(np.mean(nxt_same_voice & (ring < 1.0)))
        tc = man.loc[ids[r], "tuning_cents"]
        f_open = (440 * 2 ** ((OPEN - 69 + tc / 100) / 12))[:, None] * np.arange(1, 13)
        for w0 in test:
            w0 = int(w0)
            y = rec.y[w0:w0 + L]
            for fi in range(0, (L - NFFT) // HOP, 5):
                t = w0 + fi * HOP + NFFT // 2
                act = np.where((rec.onset <= t) & (rec.end > t))[0]
                if not len(act) or not ((rec.onset <= t) & (rec.midi_end > t)).any():
                    continue
                Y = np.abs(np.fft.rfft(y[fi * HOP:fi * HOP + NFFT] * win)) ** 2
                pf = rec.freqs[act].ravel()
                sel = (fb > 60) & (fb < 3000) & ~partial_mask(pf[pf < SR / 2], fb, 0.015)
                rest = sel & ~partial_mask(f_open[f_open < 3000], fb, 0.012)
                for s, p in enumerate(OPEN):
                    om = partial_mask(f_open[s][f_open[s] < 3000], fb, 0.004) & sel
                    if not om.any() or not rest.any():
                        continue
                    played = np.where((rec.pitch == p) & (rec.onset <= t) & (rec.onset > t - 4 * SR))[0]
                    rows.append(dict(cut_ringing=bool(len(played)) and bool((rec.end[played] <= t).all()),
                                     played_recently=bool(len(played)),
                                     excess=10 * np.log10(Y[om].mean() + 1e-12) - 10 * np.log10(Y[rest].mean() + 1e-12)))
    df = pd.DataFrame(rows)
    print(f"notes cut by the next note on their voice before ringing 1 s: {100 * np.mean(cut):.0f} % (mean over recordings)")
    for name, s in [("open string played in the last 4 s, its note already ended", df.cut_ringing),
                    ("open string not played in the last 4 s", ~df.played_recently)]:
        print(f"{name:60s} n={s.sum():6d}  excess {df.excess[s].mean():5.1f} dB")


if __name__ == "__main__":
    main()
