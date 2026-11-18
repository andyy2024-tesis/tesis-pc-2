"""Experiment 12: pass the 5 listening excerpts through a trained vocoder run (scripts/e12_vocoder.py) for the page.

Inputs as in training (Inputs): the real excerpt (original.wav) for --input real runs, otherwise the DDSP excerpt
(<label>.wav) and, when the run takes stems or the strings part, the parts rendered by analysis/render_stems.py
(slides/audio/<rid>/stems/<label>/<part>.wav, raw level; a part with no file is silent in that excerpt).
Writes slides/audio/<rid>/voc-<run>-<label>.wav (label "original" for real-input runs).
Usage: uv run python analysis/e12_refine_excerpts.py <run> [<ddsp label>]
  e.g. p2_real   |   v1_mix20 e6-structured_v12_final_learnedB_noise_strongD_rise
"""
import json
import sys
from argparse import Namespace

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "scripts")
from e12_vocoder import CKPT, RES, Inputs, Refiner, vocode_long  # noqa: E402
from env import AttrDict  # noqa: E402

from render_stems import AUDIO, LISTEN_IDS  # noqa: E402

PARTS = ["body", "burst", "contact", "floor", "noise", "strings"]  # the final DDSP's parts (render order: sorted)


def main():
    run = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "original"
    cfg = json.loads((RES / "e12" / run / "config.json").read_text())
    a = Namespace(**cfg)
    h = AttrDict(json.loads((CKPT / "config.json").read_text()))
    inputs = Inputs(a, h, PARTS)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = Refiner(h, n_stems=len(inputs.groups), inject=a.inject != "none").to(dev)
    G.load_state_dict(torch.load(RES / "e12" / run / "generator.pt", map_location="cpu"))
    G.g.remove_weight_norm()
    G.eval()
    for rid in LISTEN_IDS:
        y = sf.read(AUDIO / rid / "original.wav", dtype="float32")[0]
        x = None
        if a.input != "real":
            mix = sf.read(AUDIO / rid / f"{label}.wav", dtype="float32")[0]
            rows = [mix]
            for p in inputs.parts:
                f = AUDIO / rid / "stems" / label / f"{p}.wav"
                rows.append(sf.read(f, dtype="float32")[0][:len(mix)] if f.exists() else np.zeros_like(mix))
            n = min(len(r) for r in rows)
            x = torch.tensor(np.stack([r[:n] for r in rows]))[None]
            y = y[:n]
        out = vocode_long(G, inputs, torch.tensor(y)[None, None], x, dev)
        sf.write(AUDIO / rid / f"voc-{run}-{label}.wav", out, h.sampling_rate, subtype="FLOAT")
        print("done", rid, flush=True)


if __name__ == "__main__":
    main()
