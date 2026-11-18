"""Experiment 12: loss of each pretrained vocoder version on the 85 held-out windows (same windows and MSS loss as every
DDSP model), for the listening page's catalog. Writes analysis/results/e12_voc_metrics.json {file key: {"mean": loss}}.

Inputs per window: the real audio (24 kHz processed file; for the 44.1 kHz vocoders the raw 48 kHz GAPS file at the same
place, resampled to 44.1 kHz) and the DDSP mix from analysis/results/e12/renders/<run> (upsampled for the 44.1 kHz
vocoders). Every output is compared at 24 kHz with the real 24 kHz window (44.1 kHz outputs resampled down).
Usage: uv run python analysis/e12_voc_metrics.py e6/structured_v12_final_learnedB_noise_strongD_rise
"""
import json
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from tesis.data.gaps import REPO
from tesis.synth.dsp import MultiScaleSpectralLoss

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "analysis"))
from e12_vocode import CKPT, load_bigvgan  # noqa: E402
from e12_vocoder import D, EVAL_IDS, SR, TEST_FROM, eval_windows  # noqa: E402
import e2_one_piece as e2  # noqa: E402
import pandas as pd  # noqa: E402
from meldataset import get_mel_spectrogram  # noqa: E402

RES = REPO / "analysis" / "results"
OUT = RES / "e12_voc_metrics.json"


def main():
    run = sys.argv[1]
    label = run.replace("/", "-")
    wins = eval_windows(RES / "e12" / "renders" / run.replace("e6/", ""))
    y24 = np.stack([w[1] for w in wins])
    x24 = np.stack([w[2][1][0] for w in wins])  # the DDSP mix
    real44 = []  # the same windows (starts as in eval_windows), cut from the raw 48 kHz file
    renders = RES / "e12" / "renders" / run.replace("e6/", "")
    for rid in EVAL_IDS:
        if not (renders / f"{rid}.npz").exists():
            continue
        y = np.load(D / "audio" / f"{rid}.npy").astype(np.float32) / 32768
        on = pd.read_csv(D / "notes" / f"{rid}.csv").onset.values
        st = e2.valid_starts(y, 3 * SR, 240)
        st = st[(st >= int((on.min() - 0.5) * SR)) & (st + 3 * SR <= int((on.max() + 2.0) * SR))]
        test = st[st >= int(TEST_FROM * len(y))]
        test = test[np.linspace(0, len(test) - 1, min(5, len(test))).astype(int)] if len(test) else test
        raw, sr = sf.read(REPO / "data" / "GAPS" / "audio" / f"{rid}.wav", dtype="float32")
        raw = raw.mean(1) if raw.ndim == 2 else raw
        assert sr == 48000, sr
        for w0 in test:
            seg = raw[2 * w0: 2 * (w0 + 3 * SR)]
            real44.append(resample_poly(seg, 147, 160).astype(np.float32))
    real44 = np.stack(real44)
    assert len(real44) == len(y24)
    x44 = resample_poly(x24, 147, 80, axis=1).astype(np.float32)
    mss = MultiScaleSpectralLoss()
    to24 = lambda o: resample_poly(o, 80, 147, axis=-1).astype(np.float32)[..., : y24.shape[1]]

    def score(outs):
        return round(float(np.mean([mss(torch.tensor(o)[None], torch.tensor(y)[None]).item()
                                    for o, y in zip(outs, y24)])), 3)

    # sanity: the raw 44.1 kHz windows line up with the 24 kHz ones
    print("44.1 kHz real vs 24 kHz real:", score(to24(real44)), flush=True)
    dev = "cuda"
    res = json.loads(OUT.read_text()) if OUT.exists() else {}

    def bigvgan(name, xs):
        m = load_bigvgan(CKPT[name]).eval().to(dev)
        outs = []
        for row in xs:
            with torch.inference_mode():
                o = m(get_mel_spectrogram(torch.as_tensor(row, device=dev)[None], m.h))[0, 0].float().cpu().numpy()
            outs.append(np.pad(o, (0, max(0, len(row) - len(o))))[: len(row)])
        del m
        torch.cuda.empty_cache()
        return np.stack(outs)

    res["voc-bigvgan24-original"] = {"mean": score(bigvgan("bigvgan24", y24))}
    res[f"voc-bigvgan24-{label}"] = {"mean": score(bigvgan("bigvgan24", x24))}
    res["voc-bigvgan44-original48"] = {"mean": score(to24(bigvgan("bigvgan44", real44)))}
    res[f"voc-bigvgan44-{label}"] = {"mean": score(to24(bigvgan("bigvgan44", x44)))}
    py = str(REPO / "external" / "discoder" / ".venv" / "bin" / "python")
    with tempfile.TemporaryDirectory() as td:
        for key, xs in [("voc-discoder-original44k", real44), (f"voc-discoder-{label}", x44)]:
            np.savez(f"{td}/in.npz", x=xs)
            subprocess.run([py, str(REPO / "analysis" / "e12_discoder.py"), "--npz", f"{td}/in.npz", f"{td}/out.npz"],
                           check=True)
            res[key] = {"mean": score(to24(np.load(f"{td}/out.npz")["y"]))}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
