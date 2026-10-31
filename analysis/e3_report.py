"""Summarize experiment 3 (analysis/results/e3/note_levels.csv): note levels from the mix vs per-string truth.

Errors are computed per track after removing that track's median offset (overall level is arbitrary).
Usage: uv run python analysis/e3_report.py
"""
import numpy as np
import pandas as pd

from tesis.data.gaps import REPO

df = pd.read_csv(REPO / "analysis" / "results" / "e3" / "note_levels.csv").dropna(subset=["truth_db"])
for c in ("model", "naive"):
    df[f"{c}_err"] = df[f"{c}_db"] - df.truth_db
    df[f"{c}_err"] -= df.groupby(["id", "mix"])[f"{c}_err"].transform("median")
    df[f"{c}_abs"] = df[f"{c}_err"].abs()
# second view: also remove a fixed offset per (track, mix, string): the hex pickup's sensitivity differs per string,
# and the mic hears body and room, so a constant per-string level difference is not a note-level error
for c in ("model", "naive"):
    e = df[f"{c}_db"] - df.truth_db
    df[f"{c}_abs_str"] = (e - e.groupby([df.id, df.mix, df.string]).transform("median")).abs()
df["sounding_cat"] = pd.cut(df.sounding, [0, 1, 3, 99], labels=["alone", "2-3 sounding", "4+ sounding"])


def table(g):
    return pd.Series({
        "notes": len(g),
        "model corr": np.corrcoef(g.model_db - g.groupby("id").model_db.transform("median"),
                                  g.truth_db - g.groupby("id").truth_db.transform("median"))[0, 1],
        "naive corr": np.corrcoef(g.naive_db - g.groupby("id").naive_db.transform("median"),
                                  g.truth_db - g.groupby("id").truth_db.transform("median"))[0, 1],
        "model median err dB": g.model_abs.median(), "naive median err dB": g.naive_abs.median(),
        "model p90 err dB": g.model_abs.quantile(0.9), "naive p90 err dB": g.naive_abs.quantile(0.9),
        "model better on": (g.model_abs < g.naive_abs).mean(),
        "model err, per-string offset removed": g.model_abs_str.median(),
        "naive err, per-string offset removed": g.naive_abs_str.median(),
    })


fmt = dict(float_format=lambda v: f"{v:.2f}")
print("All notes, by mix:")
print(df.groupby("mix").apply(table, include_groups=False).to_string(**fmt))
print("\nBy how many notes are sounding at the onset:")
print(df.groupby(["mix", "sounding_cat"], observed=True).apply(table, include_groups=False).to_string(**fmt))
