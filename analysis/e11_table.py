"""One row of measures per run, from the saved results: loss and wind (catalog_metrics.json), snare and the noise's
share of the attack (noise_attack.json), noise over the room background (noise_vs_room.json), harmonic ticks
(harmonic_clicks.json; only clicks inside the excerpt: the detector also fires at the cut edges, t = 0 and 9 s, on
every model, and those are not in the sound). Missing measures print as blanks.
Usage: uv run python analysis/e11_table.py <run> [<run> ...]
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "analysis" / "results"
EXCERPT_S, EDGE_S = 9.0, 0.05


def inner_ticks(t):
    return sum(EDGE_S < e["t"] < EXCERPT_S - EDGE_S for e in t["base"]["events"])


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def row(run, cat, att, room, ticks):
    c, a, r, t = cat.get(run, {}), att.get(run, {}), room.get(run, {}), ticks.get(run, {})
    lock = a.get("lock_db", {}).get("noise", {})
    share = a.get("noise_share_of_attack", {})
    cells = [
        f"{c['mean']:.2f}" if "mean" in c else "",
        f"{c['wind']:.1f}" if "wind" in c else "",
        f"{lock['1-4k']:+.1f} / {lock['4-12k']:+.1f}" if lock else "",
        f"{100 * share['1-4k']:.0f}-{100 * share['4-12k']:.0f} %" if share else "",
        (f"{r['700-4k']['median_db']:+.1f} / {r['4k-11k']['median_db']:+.1f} "
         f"({100 * r['4k-11k']['share_above_6db']:.0f} %)") if r else "",
        str(inner_ticks(t)) if t else "",
    ]
    return f"| {run} | " + " | ".join(cells) + " |"


def main():
    cat, att, room, ticks = (load(n) for n in ("catalog_metrics", "noise_attack", "noise_vs_room", "harmonic_clicks"))
    print("| run | loss | wind | snare | share | room | ticks |\n|---|---|---|---|---|---|---|")
    for run in sys.argv[1:]:
        print(row(run, cat, att, room, ticks))


if __name__ == "__main__":
    main()
