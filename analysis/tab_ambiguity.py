"""How often does a MIDI pitch map to a single string in a piece's tablature?

If a pitch is always played on the same string within a piece, the MIDI note's string is known
without aligning score positions to the audio. Usage: python analysis/tab_ambiguity.py <gaps_root>
"""
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pretty_midi

STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def tab_pitch_strings(xml_path):
    root = ET.parse(xml_path).getroot()
    tuning = {}
    for sl in root.iter("staff-tuning"):
        line = int(sl.get("line"))
        step = sl.find("tuning-step").text.strip()
        octv = int(sl.find("tuning-octave").text)
        alt = sl.find("tuning-alter")
        tuning[line] = 12 * (octv + 1) + STEP[step] + (int(float(alt.text)) if alt is not None else 0)
    use = defaultdict(Counter)
    for note in root.iter("note"):
        tech = note.find("notations/technical")
        if tech is None or tech.find("string") is None or note.find("pitch") is None:
            continue
        p = note.find("pitch")
        alt = p.find("alter")
        try:
            midi = 12 * (int(p.find("octave").text) + 1) + STEP[p.find("step").text.strip()] + (int(float(alt.text)) if alt is not None else 0)
        except (KeyError, ValueError, AttributeError):
            continue
        use[midi][int(tech.find("string").text)] += 1
    return use, tuning


def main(root):
    root = Path(root)
    fracs, n_total, n_unamb, n_nopitch = [], 0, 0, 0
    for mid in sorted((root / "midi").glob("*.mid")):
        xml = root / "musicxml" / f"{mid.stem}.xml"
        if not xml.exists():
            continue
        try:
            use, _ = tab_pitch_strings(xml)
        except ET.ParseError:
            continue
        notes = [n for i in pretty_midi.PrettyMIDI(str(mid)).instruments for n in i.notes]
        unamb = sum(1 for n in notes if len(use.get(n.pitch, {})) == 1)
        missing = sum(1 for n in notes if n.pitch not in use)
        n_total += len(notes)
        n_unamb += unamb
        n_nopitch += missing
        fracs.append(unamb / len(notes))
    print(f"files: {len(fracs)}  notes: {n_total}")
    print(f"MIDI notes whose pitch uses exactly one string in the piece's tab: {n_unamb / n_total:.1%}")
    print(f"MIDI notes whose pitch never appears in the tab: {n_nopitch / n_total:.1%}")
    print(f"per-file fraction unambiguous: median {np.median(fracs):.1%}, p10 {np.percentile(fracs, 10):.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
