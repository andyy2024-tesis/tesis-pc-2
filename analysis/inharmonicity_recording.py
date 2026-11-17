"""How much stiffer or softer each recording's guitar is than the average, for models without strings (rec_all).

From the per-note fits of analysis/inharmonicity_strings.py (clean, tab-matched notes): each note's log B minus the
pooled per-string law (STRING_LOG_B + fret law), so a piece played mostly on the basses doesn't look softer just
because wound strings are. Per recording: the sum of those residuals over (n + 3), i.e. shrunk toward 0 with the weight
of 3 notes, then centered over the recordings. The model multiplies its stiffness law by exp(offset): the average
stiffness stays where it was (the color the user hears in the basses), only the differences between guitars are added.
Writes analysis/results/inharmonicity_recording.csv (id, n, log_b_offset).
Usage: uv run python analysis/inharmonicity_recording.py
"""
import numpy as np
import pandas as pd

from tesis.data.gaps import REPO
from tesis.model.guitar_net import STRING_LOG_B

PRIOR = 3.0


def main():
    d = pd.read_csv(REPO / "analysis" / "results" / "inharmonicity_strings.csv")
    d = d[d.matched == 1].copy()
    d["r"] = np.log(d.B) - (d.string.map(STRING_LOG_B) + np.log(2) / 6 * d.fret)
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    rows = []
    for rid in ids:
        r = d[d.id == rid].r
        rows.append(dict(id=rid, n=len(r), log_b_offset=float(r.sum() / (len(r) + PRIOR))))
    out = pd.DataFrame(rows)
    out["log_b_offset"] = (out.log_b_offset - out.log_b_offset.mean()).round(4)
    out.to_csv(REPO / "analysis" / "results" / "inharmonicity_recording.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nsd {out.log_b_offset.std():.2f} (x{np.exp(out.log_b_offset.std()):.2f}), "
          f"range {out.log_b_offset.min():.2f} .. {out.log_b_offset.max():.2f}")


if __name__ == "__main__":
    main()
