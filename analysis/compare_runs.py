"""Compare trained runs: held-out losses, wind (catch-all noise in quiet passages), flapping, ~9 s listening excerpts.

Runs are folders in analysis/results/e4 (or e6/<name> for adversarial fine-tunes, written as "e6/<name>").
Usage: uv run python analysis/compare_runs.py <out_name> <run> [<run> ...]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
import e4_train as e  # noqa: E402
from e4_targeted import upgrade_state_dict  # noqa: E402
from e6_adversarial import LISTEN_IDS, quiet_noise_db, render_long  # noqa: E402
from flapping_check import flapping_db  # noqa: E402
from tesis.data.gaps import REPO  # noqa: E402
from tesis.model.guitar_net import GuitarNet, NetConfig  # noqa: E402
from tesis.synth.dsp import MultiScaleMelLoss, MultiScaleSpectralLoss  # noqa: E402

SR = 24000
RES = REPO / "analysis" / "results"
DEV = os.environ.get("COMPARE_DEV", "cuda")  # COMPARE_DEV=cpu while the GPU is busy training


def load(run, ids):
    folder = RES / ("e4" if not run.startswith("e6/") else "") / run
    cfg_path = folder / "config.json"
    if not cfg_path.exists():  # e6 fine-tunes: config of the base run + noted overrides
        meta = json.loads((folder / "meta.json").read_text())
        c = json.loads((RES / "e4" / meta["base"] / "config.json").read_text())
        c["overrides"] = {**c["overrides"], **meta.get("overrides", {})}
    else:
        c = json.loads(cfg_path.read_text())
    cfg = NetConfig(sr=SR, structured=c["variant"] == "structured", **c["overrides"])
    recs, splits = e.load_recordings(ids + c.get("extra_ids", []), cfg, n_eval=len(ids))  # extras: training only
    m = GuitarNet(cfg, len(recs)).to(DEV)
    m.load_state_dict(upgrade_state_dict(torch.load(folder / "model.pt", map_location=DEV)))
    m.eval()
    return m, recs, splits


def main(out_name, runs):
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    out = RES / "compare" / out_name
    out.mkdir(parents=True, exist_ok=True)
    mss, mel = MultiScaleSpectralLoss(), MultiScaleMelLoss(SR).to(DEV)
    rows = {}
    wrote_orig = set()
    for i, run in enumerate(runs):
        m, recs, splits = load(run, ids)
        ev = e.evaluate(m, recs, splits, mss)
        with torch.no_grad():
            mel_l = np.mean([mel(m(rec, int(w0), noise_seed=j)[0][None],
                                 torch.tensor(rec.y[w0:w0 + 72000], device=DEV)[None]).item()
                             for rec, (_, test) in zip(recs, splits) for j, w0 in enumerate(test)])
        flap, flap_orig = [], []
        for rid in LISTEN_IDS:
            r = ids.index(rid)
            if not len(splits[r][1]):
                continue
            y_hat, y = render_long(m, recs[r], int(splits[r][1][0]))
            pk = max(np.abs(y).max(), np.abs(y_hat).max()) * 1.05
            if rid not in wrote_orig:
                sf.write(out / f"{rid}_0_original.wav", y / max(np.abs(y).max() * 1.05, 1e-9), SR)
                wrote_orig.add(rid)
            sf.write(out / f"{rid}_{i + 1}_{run.replace('/', '-')}.wav", y_hat / pk, SR)
            flap.append(flapping_db(y_hat))
            flap_orig.append(flapping_db(y))
        noise_on = m.cfg.noise
        m.cfg.noise = False  # what the deterministic parts cover on their own
        ev_det = e.evaluate(m, recs, splits, mss)
        m.cfg.noise = noise_on
        rows[run] = {"held-out loss (linear)": ev["test_loss"], "held-out loss (mel)": mel_l,
                     "held-out loss without catch-all noise": ev_det["test_loss"],
                     "wind: noise in quiet passages (dB)": quiet_noise_db(m, recs, splits),
                     "flapping vs original (dB)": float(np.mean(np.array(flap) - np.array(flap_orig))),
                     **{f"share {k[6:-3]} (dB)": v for k, v in ev.items() if k.startswith("share_")}}
        print(run, {k: round(v, 2) for k, v in rows[run].items()}, flush=True)
    df = pd.DataFrame(rows).T
    print("\n" + df.to_string(float_format=lambda v: f"{v:.2f}"))
    df.to_csv(out / "summary.csv")
    print(f"\nlistening excerpts: {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
