"""Measure f0/B on GAPS isolated notes and compare with a purely harmonic fit.

Usage: uv run python analysis/inharmonicity_isolated.py
"""
import numpy as np
import pandas as pd

from tesis.data.gaps import PROCESSED
from tesis.synth.inharmonicity import fit_partials

SR = 24000
D = PROCESSED / f"sr{SR}"
iso = pd.read_csv(D / "isolated_notes.csv")
man = pd.read_csv(D / "manifest.csv").set_index("id")
rows = []
for rid, g in iso.groupby("id"):
    y = np.load(D / "audio" / f"{rid}.npy").astype(np.float32) / 32768
    for _, r in g.iterrows():
        a = int((r.onset + 0.03) * SR)
        seg = y[a:a + int(0.6 * SR)]
        f0g = 440 * 2 ** ((r.pitch - 69 + man.loc[rid, "tuning_cents"] / 100) / 12)
        fit = fit_partials(seg, SR, f0g)
        # harmonic-only misfit using the same peaks
        k, f = fit.k, fit.freq
        f0h = np.sum(f * k) / np.sum(k * k)
        harm_resid = np.sqrt(np.mean((1200 * np.log2(f / (k * f0h))) ** 2)) if len(k) else np.nan
        rows.append(dict(id=rid, onset=r.onset, pitch=r.pitch, pre_minus_post_db=r.pre_minus_post_db, f0=fit.f0,
                         B=fit.B, n_partials=len(fit.k), k_max=int(fit.k.max()) if len(fit.k) else 0,
                         resid_cents=fit.resid_cents, harmonic_resid_cents=harm_resid,
                         top_partial_shift_cents=1200 * np.log2(np.sqrt(1 + fit.B * k.max() ** 2)) if len(k) else np.nan))
df = pd.DataFrame(rows)
df.to_csv("analysis/results/inharmonicity_isolated.csv", index=False)
ok = df[df.n_partials >= 8]
print(f"notes fitted with >=8 partials: {len(ok)} of {len(df)}")
print(f"partials found: median {ok.n_partials.median():.0f}, highest partial number median {ok.k_max.median():.0f}")
print(f"B: median {ok.B.median():.2e}, IQR {ok.B.quantile(.25):.2e} .. {ok.B.quantile(.75):.2e}")
print(f"misfit (RMS cents): stiff-string model {ok.resid_cents.median():.1f}, harmonic model {ok.harmonic_resid_cents.median():.1f}")
print(f"shift of the highest found partial from harmonic position: median {ok.top_partial_shift_cents.median():.0f} cents")
ok = ok.assign(reg=pd.cut(ok.pitch, [0, 52, 59, 64, 71, 100], labels=["E2-E3", "F3-B3", "C4-E4", "F4-B4", "C5+"]))
print(ok.groupby("reg", observed=True).agg(n=("B", "size"), B_median=("B", "median"), harm=("harmonic_resid_cents", "median"),
                                           stiff=("resid_cents", "median")).to_string(float_format=lambda v: f"{v:.3g}"))
