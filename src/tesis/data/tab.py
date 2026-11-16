"""Which string plays each note: GAPS tablature (MusicXML part with <string>/<fret>) aligned to the MIDI notes.

The syncpoints give the audio time of every bar (in performance order). Each tab note gets an estimated time from its
position inside its bar (linear between the bar's start and the next bar's start); each MIDI note is then matched to
the tab note with the same pitch closest in estimated time (one-to-one, within a tolerance). Strings: 1 = high E ...
6 = low E, as in MusicXML.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
OPEN = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}  # standard tuning (MIDI); the score's own tuning is read if present


def _pitch(note):
    p = note.find("pitch")
    if p is None:
        return None
    alt = p.find("alter")
    return 12 * (int(p.find("octave").text) + 1) + STEP[p.find("step").text.strip()] + \
        (int(round(float(alt.text))) if alt is not None else 0)


def tab_bars(xml_path):
    """Bars of the tab part in score order: list of (length in quarters, [(onset in quarters, pitch, string, fret)]),
    plus the repeat structure: per bar (forward repeat, backward repeat count, ending numbers)."""
    root = ET.parse(xml_path).getroot()
    part = None
    for p in root.iter("part"):  # the part whose notes carry strings
        if any(n.find("notations/technical/string") is not None for n in p.iter("note")):
            part = p
            break
    if part is None:
        raise ValueError("no tablature part")
    bars, rep = [], []
    div = 1
    active = None  # numbers of the ending (volta) the current bars belong to
    for meas in part.findall("measure"):
        cur = mx = 0
        last_start = 0
        notes = []
        fwd, bwd, close = False, 0, False
        for el in meas:
            if el.tag == "attributes" and el.find("divisions") is not None:
                div = int(el.find("divisions").text)
            elif el.tag == "barline":
                r = el.find("repeat")
                if r is not None and r.get("direction") == "forward":
                    fwd = True
                if r is not None and r.get("direction") == "backward":
                    bwd = int(r.get("times", 2)) - 1
                e = el.find("ending")
                if e is not None and e.get("type") == "start":
                    active = [int(x) for x in (e.get("number") or "").replace(" ", "").split(",") if x.isdigit()] or None
                if e is not None and e.get("type") in ("stop", "discontinue"):
                    close = True
            elif el.tag == "backup":
                cur -= int(el.find("duration").text)
            elif el.tag == "forward":
                cur += int(el.find("duration").text)
                mx = max(mx, cur)
            elif el.tag == "note":
                d = el.find("duration")
                d = int(d.text) if d is not None else 0  # grace notes have no duration
                if el.find("chord") is not None:
                    start = last_start
                else:
                    start = cur
                    cur += d
                    last_start = start
                mx = max(mx, cur)
                tech = el.find("notations/technical")
                if el.find("rest") is None and tech is not None and tech.find("string") is not None and \
                        el.find("tie[@type='stop']") is None:
                    pitch = _pitch(el)
                    fret = tech.find("fret")
                    notes.append((start / div, pitch, int(tech.find("string").text),
                                  int(fret.text) if fret is not None else None))
        bars.append((mx / div, notes))
        rep.append((fwd, bwd, active))
        if close:
            active = None
    return bars, rep


def unroll(rep):
    """Performance order of bar indices with repeats (and first/second endings) expanded."""
    order, i, start, n = [], 0, 0, len(rep)
    taken = {}  # backward repeat bar -> times jumped back
    pass_no = 1
    while i < n:
        fwd, bwd, ending = rep[i]
        if fwd and i != start:
            start, pass_no = i, 1
        if ending and pass_no not in ending:  # bar of another ending: skip on this pass
            i += 1
            continue
        order.append(i)
        if bwd:
            if taken.get(i, 0) < bwd:
                taken[i] = taken.get(i, 0) + 1
                pass_no += 1
                i = start
                continue
            start, pass_no = i + 1, 1  # repeat done: a later backward repeat without a forward one returns here
        i += 1
        if len(order) > 20 * n:
            raise ValueError("repeat structure did not terminate")
    return order


def bar_starts(sync_path):
    """Audio time of the start of each performed bar. Syncpoints are [bar, time] or [bar, time, offset inside the bar];
    the first point of each bar is its start."""
    sync = json.loads(Path(sync_path).read_text())
    first = {}
    for p in sync:
        first.setdefault(int(p[0]), float(p[1]))
    idx = sorted(first)
    return np.array([first[i] for i in idx])


def align_bars(bars, t_bar, midi_onset, midi_pitch, jump_cost=0.5):
    """Which written bar each performed bar plays (Viterbi). Emission: 1 - Dice overlap between the MIDI pitches
    sounding in the performed bar and the written bar's pitches; moving on to the next written bar is free, any other
    move (repeat, jump, performer's own repeat) costs jump_cost. Performers repeat sections the score doesn't mark."""
    J, N = len(t_bar), len(bars)
    t_end = np.r_[t_bar[1:], t_bar[-1] + np.median(np.diff(t_bar))]
    score = [np.bincount([p for _, p, _, _ in b[1] if p is not None], minlength=128) for b in bars]
    E = np.ones((J, N))
    for j in range(J):
        a = np.bincount(midi_pitch[(midi_onset >= t_bar[j] - 0.03) & (midi_onset < t_end[j] - 0.03)], minlength=128)
        for b in range(N):
            tot = a.sum() + score[b].sum()
            E[j, b] = 1 - 2 * np.minimum(a, score[b]).sum() / tot if tot else 0.0
    C = E[0].copy() + jump_cost * (np.arange(N) > 0)  # starting anywhere but bar 0 costs a jump
    back = np.zeros((J, N), int)
    for j in range(1, J):
        stay = np.r_[np.inf, C[:-1]]  # from b - 1 (next bar): free
        best = C.argmin()
        jump = C[best] + jump_cost
        back[j] = np.where(stay <= jump, np.arange(N) - 1, best)
        C = np.minimum(stay, jump) + E[j]
    order = [int(C.argmin())]
    for j in range(J - 1, 0, -1):
        order.append(int(back[j, order[-1]]))
    return order[::-1], float(C.min() / J)


def tab_times(xml_path, sync_path, midi_onset, midi_pitch):
    """(estimated onset s, pitch, string, fret) of every tab note in performance order, and the mean alignment cost."""
    bars, _ = tab_bars(xml_path)
    t_bar = bar_starts(sync_path)
    order, cost = align_bars(bars, t_bar, midi_onset, midi_pitch)
    out = []
    for j, b in enumerate(order):
        length, notes = bars[b]
        t0 = t_bar[j]
        t1 = t_bar[j + 1] if j + 1 < len(t_bar) else t0 + (t0 - t_bar[j - 1])
        for q, pitch, s, f in notes:
            if pitch is not None:
                out.append((t0 + (t1 - t0) * q / max(length, 1e-9), pitch, s, np.nan if f is None else f))
    return np.array(out, dtype=float).reshape(-1, 4), cost


def assign_strings(onset, pitch, tab, tols=(0.25, 0.6)):
    """String (1-6, 0 = unmatched) and fret of each MIDI note: same-pitch tab note closest in time, one-to-one,
    greedy by distance; a first pass within tols[0] seconds, then the rest within tols[1]."""
    string = np.zeros(len(onset), int)
    fret = np.full(len(onset), -1)
    used_m, used_t = set(), set()
    for tol in tols:
        pairs = []
        for p in np.unique(pitch):
            mi = np.array([i for i in np.where(pitch == p)[0] if i not in used_m], int)
            ti = np.array([i for i in np.where(tab[:, 1] == p)[0] if i not in used_t], int)
            if len(ti) == 0 or len(mi) == 0:
                continue
            d = np.abs(onset[mi][:, None] - tab[ti, 0][None])
            for a, b in zip(*np.where(d < tol)):
                pairs.append((d[a, b], mi[a], ti[b]))
        for d, m, t in sorted(pairs):
            if m in used_m or t in used_t:
                continue
            used_m.add(m)
            used_t.add(t)
            string[m] = int(tab[t, 2])
            fret[m] = int(tab[t, 3]) if not np.isnan(tab[t, 3]) else -1
    return string, fret


def open_pitches(pitch, string, fret):
    """Open-string pitch of each string (1-6) in this recording: median of pitch - fret over matched notes
    (alternate tunings such as a lowered 6th string), standard tuning where there is no evidence."""
    out = dict(OPEN)
    for s in range(1, 7):
        m = (string == s) & (fret >= 0)
        if m.sum() >= 3:
            out[s] = int(np.median(pitch[m] - fret[m]))
    return out


def complete_strings(onset, pitch, string, fret):
    """Every note gets a string: unmatched notes take the string of the closest-in-time matched note of the same pitch,
    else the string where the pitch has the lowest fret. Frets are then pitch - open pitch (consistent by construction).
    Returns string (1-6), fret (>= 0), open pitches {string: midi}, and which notes were matched in the tab."""
    opens = open_pitches(pitch, string, fret)
    matched = string > 0
    s = string.copy()
    for i in np.where(~matched)[0]:
        same = np.where(matched & (pitch == pitch[i]))[0]
        if len(same):
            s[i] = string[same[np.abs(onset[same] - onset[i]).argmin()]]
        else:
            ok = [(pitch[i] - opens[k], k) for k in range(1, 7) if pitch[i] - opens[k] >= 0]
            s[i] = min(ok)[1] if ok else 6
    f = np.array([max(pitch[i] - opens[s[i]], 0) for i in range(len(s))])
    return s, f, opens, matched
