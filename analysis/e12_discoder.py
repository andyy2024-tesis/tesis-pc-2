"""Experiment 12, phase 1: pass listening excerpts through the pretrained DisCoder music vocoder (44.1 kHz).

Runs in DisCoder's own environment (external/discoder/.venv; its pins clash with the project's):
  external/discoder/.venv/bin/python analysis/e12_discoder.py original44k e6-structured_v9_base_t32_noise_strongD ...
Reads analysis/results/slides/audio/<rid>/<label>.wav (any rate, resampled to 44.1 kHz) and writes
<rid>/voc-discoder-<label>.wav.
  ... analysis/e12_discoder.py --npz <in.npz> <out.npz>: vocode every row of in.npz's "x" (44.1 kHz) into out.npz "y"
  (used by analysis/e12_voc_metrics.py for the 85 held-out windows).
"""
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "discoder"))
from discoder import utils  # noqa: E402
from discoder.models import DisCoder  # noqa: E402

AUDIO = REPO / "analysis" / "results" / "slides" / "audio"
LISTEN_IDS = ["023_Ypswc", "176_mM1wc", "276_jw1wc", "277_Pw1wc", "306_7c1wc"]
SR = 44100


def main():
    dev = os.environ.get("DISCODER_DEV", "cuda")  # "cpu" when the GPU is busy training
    model = DisCoder.from_pretrained("disco-eth/discoder").eval().to(dev)
    if sys.argv[1] == "--npz":
        import numpy as np
        xs = np.load(sys.argv[2])["x"]
        outs = []
        for row in xs:
            with torch.inference_mode():
                mel = utils.get_mel_spectrogram_from_config(torch.as_tensor(row, device=dev)[None], model.config)
                o = model(mel)[0, 0].float().cpu().numpy()
            outs.append(np.pad(o, (0, max(0, len(row) - len(o))))[:len(row)])
        np.savez(sys.argv[3], y=np.stack(outs))
        return
    for rid in LISTEN_IDS:
        for label in sys.argv[1:]:
            y, sr = sf.read(AUDIO / rid / f"{label}.wav", dtype="float32")
            x = torch.as_tensor(y, device=dev)[None]
            if sr != SR:
                x = torchaudio.functional.resample(x, sr, SR)
            with torch.inference_mode():
                mel = utils.get_mel_spectrogram_from_config(x, model.config)
                out = model(mel)[0, 0].float().cpu().numpy()
            sf.write(AUDIO / rid / f"voc-discoder-{label}.wav", out, SR, subtype="FLOAT")
        print("done", rid, flush=True)


if __name__ == "__main__":
    main()
