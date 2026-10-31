"""Targeted held-out metrics for experiment 4 (structured vs plain DDSP), plus listening excerpts.

attack:  log-spectral distance (FFT 128, hop 16) over the 50 ms after each onset in the window
high:    log-spectral distance 2-8 kHz with FFT 4096 (partials resolved: inharmonicity shows here)
low:     log-spectral distance below 200 Hz with FFT 8192 (body resonance region)
Lower is better. Also reports the spectral loss and the catch-all noise share per run.
Usage: uv run python analysis/e4_targeted.py <run> [<run> ...]   (run = folder in analysis/results/e4)
"""
import json
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, "scripts")
import e4_train as e  # noqa: E402
from tesis.data.gaps import REPO  # noqa: E402
from tesis.model.guitar_net import GuitarNet, NetConfig  # noqa: E402
from tesis.synth.dsp import MultiScaleSpectralLoss  # noqa: E402

SR, L = 24000, 72000
OUT = REPO / "analysis" / "results" / "e4"


def lsd(x, y, n_fft, hop, f_lo=0, f_hi=SR / 2, frames=None):
    w = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(x, n_fft, hop, window=w, return_complex=True).abs()
    Y = torch.stft(y, n_fft, hop, window=w, return_complex=True).abs()
    f = torch.fft.rfftfreq(n_fft, 1 / SR).to(x.device)
    m = (f >= f_lo) & (f < f_hi)
    d = (20 * torch.log10(X[m] + 1e-5) - 20 * torch.log10(Y[m] + 1e-5)).abs()  # (bins, frames)
    if frames is not None:
        d = d[:, frames]
    return d.mean().item() if d.numel() else np.nan


def upgrade_state_dict(sd):
    """Older checkpoints computed per-recording settings as Linear(rec_emb); turn them into per-recording tables."""
    emb = sd["rec_emb.weight"]
    for old, new in [("body_rec", "rec_body_filter"), ("floor", "rec_floor"), ("burst", "rec_burst"),
                     ("body_eq", "rec_body_eq"), ("reverb", "rec_reverb")]:
        if f"{old}.weight" in sd and f"{new}.weight" not in sd:
            sd[f"{new}.weight"] = emb @ sd.pop(f"{old}.weight").T + sd.pop(f"{old}.bias")
    return sd


def load_run(run, ids, overrides=None):
    c = json.loads((OUT / run / "config.json").read_text()) if (OUT / run / "config.json").exists() else \
        dict(variant=run, overrides={})
    cfg = NetConfig(sr=SR, structured=c["variant"] == "structured", **{**c["overrides"], **(overrides or {})})
    recs, splits = e.load_recordings(ids + c.get("extra_ids", []), cfg, n_eval=len(ids))  # extras: training only
    m = GuitarNet(cfg, len(recs)).cuda()
    m.load_state_dict(upgrade_state_dict(torch.load(OUT / run / "model.pt")))
    m.eval()
    return m, (recs, splits)


def main():
    runs = sys.argv[1:] or ["structured", "ddsp"]
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    rows, models, data = [], {}, {}
    loss_fn = MultiScaleSpectralLoss()
    for v in runs:
        models[v], data[v] = load_run(v, ids)
    recs, splits = data[runs[0]]
    listen = OUT / ("listen_" + "-".join(runs))
    listen.mkdir(exist_ok=True)
    with torch.no_grad():
        for r, (rid, (_, test)) in enumerate(zip(ids, splits)):
            best = None
            for j, w0 in enumerate(test):
                w0 = int(w0)
                y = torch.tensor(recs[r].y[w0:w0 + L]).cuda()
                ons = recs[r].onset[(recs[r].onset >= w0) & (recs[r].onset < w0 + L - int(0.05 * SR))] - w0
                hop_a = 16
                att = np.unique(np.concatenate([np.arange(o // hop_a, (o + int(0.05 * SR)) // hop_a) for o in ons])) if len(ons) else None
                row = dict(id=rid, w0=w0, onsets=len(ons))
                outs = {}
                for v, m in models.items():
                    out, parts = m(data[v][0][r], w0, noise_seed=j)
                    outs[v] = out
                    row[f"{v}_loss"] = loss_fn(out[None], y[None]).item()
                    tot = sum((q ** 2).sum().item() for q in parts.values())
                    row[f"{v}_noise"] = (parts["noise"] ** 2).sum().item() / tot if "noise" in parts else 0.0
                    row[f"{v}_body"] = (parts["body"] ** 2).sum().item() / tot if "body" in parts else 0.0
                    row[f"{v}_attack"] = lsd(out, y, 128, hop_a, frames=torch.as_tensor(att).cuda()) if att is not None else np.nan
                    row[f"{v}_high"] = lsd(out, y, 4096, 1024, 2000, 8000)
                    row[f"{v}_low"] = lsd(out, y, 8192, 2048, 0, 200)
                rows.append(row)
                if best is None or len(ons) > best[0]:
                    best = (len(ons), w0, y.cpu().numpy(), {k: o.cpu().numpy() for k, o in outs.items()})
            if best is not None and r < 6:
                _, w0, y, outs = best
                pk = max(np.abs(y).max(), *(np.abs(o).max() for o in outs.values())) * 1.05
                sf.write(listen / f"{rid}_1_original.wav", y / pk, SR)
                for i, (k, o) in enumerate(outs.items()):
                    sf.write(listen / f"{rid}_{i + 2}_{k}.wav", o / pk, SR)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / ("targeted_" + "-".join(runs) + ".csv"), index=False)
    print(f"{len(d)} held-out music windows, {d.id.nunique()} recordings. Medians; LSD = log-spectral distance (dB).\n")
    tab = pd.DataFrame({v: {
        "spectral loss": d[f"{v}_loss"].median(),
        "noise share (dB)": 10 * np.log10(d[f"{v}_noise"].median() + 1e-12),
        "body share (dB)": 10 * np.log10(d[f"{v}_body"].median() + 1e-12),
        "attack LSD": d[f"{v}_attack"].median(), "2-8 kHz LSD": d[f"{v}_high"].median(),
        "<200 Hz LSD": d[f"{v}_low"].median()} for v in runs}).T
    print(tab.to_string(float_format=lambda x: f"{x:.2f}"))
    base = runs[0]
    print(f"\nShare of windows where each run beats {base}:")
    for v in runs[1:]:
        print(f"  {v:28s} " + "  ".join(f"{m} {np.mean(d[f'{v}_{m}'] < d[f'{base}_{m}']):.0%}"
                                        for m in ["loss", "attack", "high", "low"]))
    print(f"\nlistening excerpts: {listen}")


if __name__ == "__main__":
    main()
