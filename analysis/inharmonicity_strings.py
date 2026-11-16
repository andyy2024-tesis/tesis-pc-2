"""Inharmonicity per string, now that every note has a string and fret (analysis/tab_strings.py).

Physics: on one string B = pi^3 E d^4 / (64 T L^2), so stopping the string at fret n (L -> L * 2^(-n/12)) multiplies B by
2^(n/6): log B rises 0.1155 per fret. The model today uses one line over all strings, log B = -11.35 + 0.0304 * pitch.
For notes with no other onset from 50 ms before to 350 ms after (cleaner spectra), fit (f0, B) on 30-330 ms after the
onset (tesis.synth.inharmonicity.fit_partials); keep fits with >= 6 partials and residual < 3 cents. Per string:
slope of log B vs fret, and log B at the open string with the slope fixed to the physical 0.1155.
Writes analysis/results/inharmonicity_strings.csv (per note) and prints the per-string summary.
Usage: uv run python analysis/inharmonicity_strings.py
"""
import numpy as np
import pandas as pd

from tesis.data.gaps import PROCESSED, REPO
from tesis.synth.inharmonicity import fit_partials

SR = 24000
D = PROCESSED / f"sr{SR}"
FRET_SLOPE = np.log(2) / 6
LOG_B = (-11.3465, 0.0304)


def main():
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    man = pd.read_csv(D / "manifest.csv").set_index("id")
    rows = []
    for rid in ids:
        y = np.load(D / "audio" / f"{rid}.npy").astype(np.float32) / 32768
        n = pd.read_csv(D / "notes" / f"{rid}.csv")
        st = pd.read_csv(D / "strings" / f"{rid}.csv")
        on = n.onset.values
        for i in range(len(n)):
            others = np.delete(on, i)
            if np.any((others > on[i] - 0.05) & (others < on[i] + 0.35)):
                continue
            a = int((on[i] + 0.03) * SR)
            seg = y[a:a + int(0.3 * SR)]
            if len(seg) < int(0.3 * SR):
                continue
            f0g = 440 * 2 ** ((n.pitch[i] - 69 + man.loc[rid, "tuning_cents"] / 100) / 12)
            fit = fit_partials(seg, SR, f0g)
            if len(fit.k) < 6 or not np.isfinite(fit.B) or fit.B <= 0 or fit.resid_cents > 3:
                continue
            rows.append(dict(id=rid, pitch=int(n.pitch[i]), string=int(st.string[i]), fret=int(st.fret[i]),
                             matched=int(st.matched[i]), B=fit.B, n_partials=len(fit.k), resid=fit.resid_cents))
        print(rid, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "analysis" / "results" / "inharmonicity_strings.csv", index=False)
    df = df[df.matched == 1]
    print(f"\n{len(df)} clean notes on tab-matched strings")
    print("string  notes  slope/fret (physics 0.116)  log B open (slope fixed)  B open      global-law B at open pitch")
    open_p = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
    for s in range(1, 7):
        g = df[df.string == s]
        if len(g) < 5:
            print(s, len(g))
            continue
        lb = np.log(g.B.values)
        slope = np.polyfit(g.fret, lb, 1)[0] if g.fret.nunique() > 2 else np.nan
        lo = np.median(lb - FRET_SLOPE * g.fret)
        print(f"{s:6d} {len(g):6d}  {slope:24.3f}  {lo:24.2f}  {np.exp(lo):.2e}   {np.exp(LOG_B[0] + LOG_B[1] * open_p[s]):.2e}")
    # how well each law predicts the measured B (median absolute error of log B)
    lb = np.log(df.B.values)
    glob = LOG_B[0] + LOG_B[1] * df.pitch.values
    lo_s = df.groupby("string").apply(lambda g: np.median(np.log(g.B) - FRET_SLOPE * g.fret))
    per = lo_s[df.string].values + FRET_SLOPE * df.fret.values
    print(f"\nmedian |log B error|: global pitch law {np.median(np.abs(lb - glob)):.3f}, "
          f"per string + fret law {np.median(np.abs(lb - per)):.3f}")


if __name__ == "__main__":
    main()
