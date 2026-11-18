"""Assign a string to every note of the training recordings from the GAPS tablature (tesis.data.tab).

For each recording: bars vs syncpoints, share of MIDI notes matched to a tab note, and consistency (fret = pitch - open
string pitch, within 0-19). Writes data/processed/sr24000/strings/<id>.csv (string, fret per row of notes/<id>.csv;
string 0 = unmatched) and analysis/results/tab_strings.csv.
Usage: uv run python analysis/tab_strings.py [ids...]   (default: the 20 of e4_recordings.csv; with ids the summary
goes to analysis/results/tab_strings_more.csv)
"""
import sys

import numpy as np
import pandas as pd

from tesis.data.gaps import GAPS_ROOT, PROCESSED, REPO
from tesis.data.tab import OPEN, assign_strings, complete_strings, tab_times

D = PROCESSED / "sr24000"


def main(ids, out_name="tab_strings.csv"):
    rows = []
    (D / "strings").mkdir(exist_ok=True)
    for rid in ids:
        notes = pd.read_csv(D / "notes" / f"{rid}.csv")
        tab, cost = tab_times(GAPS_ROOT / "musicxml" / f"{rid}.xml", GAPS_ROOT / "syncpoints" / f"{rid}.json",
                              notes.midi_onset.values, notes.pitch.values)
        s0, f0 = assign_strings(notes.midi_onset.values, notes.pitch.values, tab)
        ok = s0 > 0
        open_p = np.array([OPEN.get(x, 0) for x in s0])
        consistent = ok & (notes.pitch.values - open_p == f0)
        s, f, opens, matched = complete_strings(notes.midi_onset.values, notes.pitch.values, s0, f0)
        pd.DataFrame(dict(string=s, fret=f, open_pitch=[opens[x] for x in s], matched=matched.astype(int))).to_csv(
            D / "strings" / f"{rid}.csv", index=False)
        # timing residual of matched notes: how well the bar-linear estimate places tab notes
        r = dict(id=rid, notes=len(notes), tab_notes=len(tab), bar_cost=round(cost, 3), matched=round(ok.mean(), 3),
                 fret_consistent=round(consistent.sum() / max(ok.sum(), 1), 3),
                 tuning="-".join(str(opens[k]) for k in range(6, 0, -1)),
                 **{f"s{k}": int((s == k).sum()) for k in range(1, 7)})
        rows.append(r)
        print(r, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "analysis" / "results" / out_name, index=False)
    if "matched" in out:
        print("median matched", out.matched.median(), "min", out.matched.min())


if __name__ == "__main__":
    if sys.argv[1:]:  # other recordings (experiment 12: more DDSP training pairs) keep the summary of the 20 intact
        main(sys.argv[1:], "tab_strings_more.csv")
    else:
        main(pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist())
