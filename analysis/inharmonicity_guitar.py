"""Stiffness of each recording's strings (each GAPS recording is a different guitar and string set).

From the per-note fits of analysis/inharmonicity_strings.py (clean, tab-matched notes), log B at the open string is
log B - (ln 2 / 6) * fret. Within a recording a string is very consistent (median |deviation| 0.06-0.11), between
recordings it varies by a factor of 2-8 (sd of the recording medians 0.3-0.66), so one table for all guitars is a poor
average. Per (recording, string): the median, shrunk toward the pooled table with the weight of 3 notes,
(n * median + 3 * pooled) / (n + 3); strings with no clean note keep the pooled value.
Writes analysis/results/inharmonicity_guitar.csv (id, string, n, log_b_open).
Usage: uv run python analysis/inharmonicity_guitar.py
"""
import numpy as np
import pandas as pd

from tesis.data.gaps import REPO
from tesis.model.guitar_net import STRING_LOG_B

PRIOR = 3.0


def main():
    d = pd.read_csv(REPO / "analysis" / "results" / "inharmonicity_strings.csv")
    d = d[d.matched == 1].copy()
    d["lo"] = np.log(d.B) - np.log(2) / 6 * d.fret
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    rows = []
    for rid in ids:
        for s in range(1, 7):
            g = d[(d.id == rid) & (d.string == s)].lo
            n = len(g)
            lo = (n * g.median() + PRIOR * STRING_LOG_B[s]) / (n + PRIOR) if n else STRING_LOG_B[s]
            rows.append(dict(id=rid, string=s, n=n, log_b_open=round(float(lo), 4)))
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "analysis" / "results" / "inharmonicity_guitar.csv", index=False)
    w = out.pivot(index="id", columns="string", values="log_b_open")
    print(w.round(2).to_string())
    print("\nshift from the pooled table (log B), per string: sd over recordings",
          (w - pd.Series(STRING_LOG_B)).std().round(2).to_dict())


if __name__ == "__main__":
    main()
