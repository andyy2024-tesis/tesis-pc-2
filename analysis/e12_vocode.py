"""Experiment 12, phase 1 (ceilings, no training): pass listening excerpts through a pretrained vocoder.

For each listening recording, takes analysis/results/slides/audio/<rid>/<label>.wav (24 kHz excerpts of the real
recording or of a DDSP model), computes the vocoder's mel spectrogram and resynthesises it. Writes
<rid>/voc-<vocoder>-<label>.wav at the vocoder's rate. "original48" is the same excerpt cut from the raw 48 kHz GAPS file
(full band), for the 44 kHz vocoders.

Usage: uv run python analysis/e12_vocode.py bigvgan24 original e6-structured_v9_base_t32_noise_strongD ...
"""
import json
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

from tesis.data.gaps import REPO

sys.path.insert(0, str(REPO / "external" / "BigVGAN"))
import bigvgan  # noqa: E402
from env import AttrDict  # noqa: E402
from meldataset import get_mel_spectrogram  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from e6_adversarial import LISTEN_IDS  # noqa: E402

AUDIO = REPO / "analysis" / "results" / "slides" / "audio"
CKPT = {"bigvgan24": "bigvgan_v2_24khz_100band_256x", "bigvgan44": "bigvgan_v2_44khz_128band_512x"}
SR_IN = 24000


def excerpt48(rid, n24):
    """The listening excerpt cut from the raw 48 kHz GAPS file (mono), located by cross-correlating with the 24 kHz one."""
    y24, _ = sf.read(AUDIO / rid / "original.wav", dtype="float32")
    full = np.load(REPO / "data" / "processed" / f"sr{SR_IN}" / "audio" / f"{rid}.npy").astype(np.float32) / 32768
    # the excerpt is a slice of the processed file: find its start exactly (FFT cross-correlation of its first second)
    key = y24[:SR_IN]
    n = len(full) + len(key)
    nf = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(full, nf) * np.conj(np.fft.rfft(key, nf)), nf)[:len(full) - len(key)]
    s24 = int(np.argmax(c))
    raw, sr = sf.read(REPO / "data" / "GAPS" / "audio" / f"{rid}.wav", dtype="float32")
    raw = raw.mean(1) if raw.ndim == 2 else raw
    s = int(round(s24 * sr / SR_IN))
    return raw[s:s + int(round(n24 * sr / SR_IN))], sr


def load_bigvgan(name, weight_norm=False):
    """Pretrained BigVGAN generator from external/ckpt/<name> (config.json + bigvgan_generator.pt)."""
    d = REPO / "external" / "ckpt" / name
    h = AttrDict(json.loads((d / "config.json").read_text()))
    model = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    model.load_state_dict(torch.load(d / "bigvgan_generator.pt", map_location="cpu", weights_only=False)["generator"])
    if not weight_norm:
        model.remove_weight_norm()
    return model


def main():
    voc, labels = sys.argv[1], sys.argv[2:]
    dev = "cuda"
    model = load_bigvgan(CKPT[voc])
    model = model.eval().to(dev)
    h = model.h
    for rid in LISTEN_IDS:
        n24 = sf.info(AUDIO / rid / "original.wav").frames
        for label in labels:
            if label == "original48":
                y, sr = excerpt48(rid, n24)
            else:
                y, sr = sf.read(AUDIO / rid / f"{label}.wav", dtype="float32")
            x = torch.as_tensor(y, device=dev)[None]
            if sr != h.sampling_rate:
                x = torchaudio.functional.resample(x, sr, h.sampling_rate)
            with torch.inference_mode():
                mel = get_mel_spectrogram(x, h)
                out = model(mel)[0, 0].float().cpu().numpy()
            sf.write(AUDIO / rid / f"voc-{voc}-{label}.wav", out, h.sampling_rate, subtype="FLOAT")
            if label == "original48":  # the full-band reference itself, at the vocoder's rate
                ref = x[0].float().cpu().numpy()
                sf.write(AUDIO / rid / f"original{h.sampling_rate // 1000}k.wav", ref, h.sampling_rate, subtype="FLOAT")
        print("done", rid, flush=True)


if __name__ == "__main__":
    main()
