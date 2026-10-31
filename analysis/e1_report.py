"""Summarize experiment 1 (analysis/results/e1/e1_results.csv): paired comparisons between variants.

Usage: uv run python analysis/e1_report.py
"""
import numpy as np
import pandas as pd

from tesis.data.gaps import REPO

R = REPO / "analysis" / "results" / "e1"
df = pd.concat([pd.read_csv(f) for f in sorted(R.glob("e1_results*.csv"))])
key = ["id", "onset"]
order = [v for v in ["ddsp", "ddsp_inh", "phys_nob", "phys", "phys_body", "hybrid", "ddsp_full"] if v in df.variant.unique()]
wide = {m: df.pivot_table(index=key, columns="variant", values=m)[order] for m in
        ["spectral_loss", "noise_frac", "noise_frac_first50ms", "noise_frac_below_f0"]}
n = len(wide["spectral_loss"])


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-9))


print(f"{n} notes\n")
summ = pd.DataFrame({
    "spectral loss (median)": wide["spectral_loss"].median(),
    "noise share, whole note (dB)": db(wide["noise_frac"]).median(),
    "noise share, first 50 ms (dB)": db(wide["noise_frac_first50ms"]).median(),
    "noise share below f0 (%)": 100 * wide["noise_frac_below_f0"].median(),
    "params for the note": df.groupby("variant").note_params.first()[order],
})
print(summ.to_string(float_format=lambda v: f"{v:.3g}"))

print("\nPaired comparisons (share of notes where B beats A, median change):")
pairs = [("ddsp", "ddsp_inh", "inharmonicity, DDSP-style amplitudes"),
         ("ddsp_inh", "phys_nob", "physical decay instead of free amplitudes"),
         ("phys_nob", "phys", "adding the pluck transient"),
         ("phys", "phys_body", "adding body resonances"),
         ("phys_body", "hybrid", "slow correction on the physical decay"),
         ("ddsp_inh", "hybrid", "hybrid vs flexible amplitudes (inh)"),
         ("ddsp_inh", "ddsp_full", "transient + body on flexible amplitudes"),
         ("ddsp", "ddsp_full", "all structured parts vs plain DDSP")]
for a, b, what in pairs:
    if a not in order or b not in order:
        continue
    sl = wide["spectral_loss"]
    d_loss = sl[b] - sl[a]
    nf = wide["noise_frac"]
    d_noise = db(nf[b]) - db(nf[a])
    e50 = wide["noise_frac_first50ms"]
    d50 = db(e50[b]) - db(e50[a])
    print(f"  {what:45s} loss: B better on {np.mean(d_loss < 0):.0%}, median change {d_loss.median():+.3f}  |  "
          f"noise share: lower on {np.mean(d_noise < 0):.0%}, median {d_noise.median():+.1f} dB  |  "
          f"first 50 ms: {d50.median():+.1f} dB")

# where inharmonicity matters: by register
pitch = df.drop_duplicates(key).set_index(key).pitch.reindex(wide["spectral_loss"].index)
reg = pd.cut(pitch, [0, 52, 59, 64, 71, 100], labels=["E2-E3", "F3-B3", "C4-E4", "F4-B4", "C5+"])
if {"ddsp", "ddsp_inh"} <= set(order):
    d = (wide["spectral_loss"]["ddsp_inh"] - wide["spectral_loss"]["ddsp"]).groupby(reg, observed=True)
    print("\nInharmonicity gain by register (median spectral-loss change, ddsp_inh - ddsp):")
    print(d.agg(["size", "median"]).to_string(float_format=lambda v: f"{v:+.3f}"))
summ.to_csv(R / "e1_summary_table.csv")
