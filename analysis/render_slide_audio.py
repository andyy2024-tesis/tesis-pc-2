"""Render the same ~9 s held-out excerpt of the five listening recordings for every model shown in the slides.

Writes raw (not normalized) float wavs to analysis/results/slides/audio/<recording>/<label>.wav so clips of one
recording keep their relative loudness. "<run>:noise" renders only the catch-all noise of that run (the "wind");
"<run>:nonoise" renders the run with its catch-all noise switched off.
Usage: uv run python analysis/render_slide_audio.py [run spec ...]   (default: all of RUNS)
"""
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import load  # noqa: E402
from e6_adversarial import LISTEN_IDS, render_long  # noqa: E402
from tesis.data.gaps import REPO  # noqa: E402

SR = 24000
OUT = REPO / "analysis" / "results" / "slides" / "audio"
RUNS = ["ddsp_v2", "ddsp_v2_nonoise", "structured_v2", "structured_v2_bodyir", "structured_v2_bodyir_nonoise",
        "structured_v4_body", "structured_v4_body_reverb", "ddsp_v5_audio", "structured_v5_body_audio",
        "structured_v5_body_audio:noise", "structured_v6_mel_noise", "structured_v6_mel_nonoise",
        "e6/structured_v5_body_audio_nonoise_strongD", "e6/structured_v5_body_audio_nonoise_strongD_9k",
        "e6/structured_v5_body_audio_noise_strongD", "e6/structured_v5_body_audio_noise_strongD:noise"]


class Part(torch.nn.Module):
    """Wraps a model so that calling it returns one of its parts instead of the mix."""

    def __init__(self, model, part):
        super().__init__()
        self.model, self.part = model, part

    def forward(self, rec, w0, noise_seed=None):
        y, parts = self.model(rec, w0, noise_seed=noise_seed)
        return parts[self.part], parts


def main():
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    for spec in sys.argv[1:] or RUNS:
        run, _, part = spec.partition(":")
        m, recs, splits = load(run, ids)
        if part == "nonoise":
            m.cfg.noise = False
        elif part:
            m = Part(m, part)
        label = spec.replace("/", "-").replace(":", "-")
        for rid in LISTEN_IDS:
            r = ids.index(rid)
            start = int(splits[r][1][0])
            y_hat, y = render_long(m, recs[r], start)
            (OUT / rid).mkdir(parents=True, exist_ok=True)
            sf.write(OUT / rid / f"{label}.wav", y_hat.astype(np.float32), SR, subtype="FLOAT")
            if not (OUT / rid / "original.wav").exists():
                sf.write(OUT / rid / "original.wav", np.asarray(y, np.float32), SR, subtype="FLOAT")
        print("done", spec, flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
