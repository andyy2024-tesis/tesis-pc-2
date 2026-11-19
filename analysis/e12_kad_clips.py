"""Experiment 12: audio clips for the Kernel Audio Distance (KAD, analysis/e12_kad.py), one folder per version.

Every version is the 85 held-out 3 s windows of the 20 evaluation recordings (as every loss in the thesis), 24 kHz,
each clip brought to the same loudness (RMS -23 dBFS, peak at most -1 dBFS) so KAD compares sound, not level.
Folders under analysis/results/e12/kad/clips/ (gitignored):
  real            the real windows (reference A: same passages)
  realtest        real guitar no model has heard: the 30 GAPS test-split recordings, 20 clips each over the music span
                  (reference B)
  ddsp            the final DDSP (its pair renders)
  voc-<run>       a vocoder run (scripts/e12_vocoder.py) on its own DDSP renders (the run's --renders; real-input
                  runs, phase 2 and the pretrained p0_pretrained, take the final DDSP's mix), as in its evaluation.
                  The windows are the same for every renders folder (they depend only on the real audio and notes).
                  A 44.1 kHz run (phase 5) takes the real window at 44.1 kHz or the DDSP upsampled, and its output is
                  brought back to 24 kHz (as its evaluation)
  voc-<run>-real  a real-input run on the real windows (the ceiling)
  voc-pretrained44[-real]  BigVGAN-v2 44 kHz as released, through the phase-5 path (the reference for p5_real44)
  diff-<run>      a mel refiner (scripts/e12_diffusion.py) vocoded by p2_real (shallow: depth 20)
  model-<key>     a DDSP model rendered window by window as analysis/measure_catalog.py does for its loss (key = its
                  listening-page key, e.g. e6-structured_v9_base_t32_noise_strongD; COMPARE_DEV=cpu while training)
Usage: uv run python analysis/e12_kad_clips.py real realtest ddsp voc-p2_real voc-v1_mix20 diff-v3b_flow ...
"""
import json
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from e12_vocoder import CKPTS, D, RES, SR, UP, Inputs, Refiner, eval_windows, vocode_long  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from env import AttrDict  # noqa: E402

OUT = RES / "e12" / "kad" / "clips"
REN = RES / "e12" / "renders" / "structured_v12_final_learnedB_noise_strongD_rise"
L = 3 * SR


def level(x):
    x = x.astype(np.float64)
    x = x * (10 ** (-23 / 20) / max(np.sqrt(np.mean(x ** 2)), 1e-9))
    return (x * min(1.0, 10 ** (-1 / 20) / max(np.abs(x).max(), 1e-9))).astype(np.float32)


def write(name, clips):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for k, x in clips:
        sf.write(d / f"{k}.wav", level(x), SR, subtype="FLOAT")
    print(name, len(clips), "clips", flush=True)


def realtest():
    man = pd.read_csv(D / "manifest.csv")
    out = []
    for rid in man[man.split == "test"].id:
        y = np.load(D / "audio" / f"{rid}.npy").astype(np.float32) / 32768
        on = pd.read_csv(D / "notes" / f"{rid}.csv").onset.values
        lo, hi = max(int((on.min() - 0.5) * SR), 0), min(int((on.max() + 2.0) * SR), len(y)) - L
        for j, s in enumerate(np.linspace(lo, hi, 20).astype(int)):
            out.append((f"{rid}_{j:02d}", y[s:s + L]))
    return out


def vocoder(run, wins, real_in, dev):
    pre44 = run == "pretrained44"  # BigVGAN-v2 44 kHz as released (phase 1), through the phase-5 path
    cfg = json.loads((RES / "e12" / ("p5_real44" if pre44 else run) / "config.json").read_text())
    sr = cfg.get("sr", SR)
    ren = RES / "e12" / "renders" / cfg["renders"] if cfg.get("renders") else REN
    if ren != REN or sr != SR:  # 44.1 kHz runs: real window at 44.1 kHz, DDSP upsampled, as in their evaluation
        w2 = eval_windows(ren, sr)
        assert len(w2) == 85 and all(np.array_equal(p[1], q[1]) for p, q in zip(w2, wins)), "windows differ"
        wins = w2
    a = Namespace(**cfg)
    if real_in:
        assert a.input == "real", run
    elif a.input == "real":
        a.input = "mix"
    h = AttrDict(json.loads((CKPTS[sr] / "config.json").read_text()))
    names = wins[0][2][0]
    inputs = Inputs(a, h, names)
    G = Refiner(h, n_stems=len(inputs.groups), inject=a.inject != "none").to(dev)
    if pre44:
        G.g.load_state_dict(torch.load(CKPTS[sr] / "bigvgan_generator.pt", map_location="cpu")["generator"])
    else:
        G.load_state_dict(torch.load(RES / "e12" / run / "generator.pt", map_location="cpu"))
    G.g.remove_weight_norm()
    G.eval()
    out = []
    for k, w in enumerate(wins):
        rid, y, (nm, arr) = w[:3]
        yt = torch.tensor(w[3] if len(w) > 3 else y)[None, None]
        xt = None if real_in else torch.tensor(arr[[0] + [1 + nm.index(p) for p in inputs.parts]])[None]
        o = vocode_long(G, inputs, yt, xt, dev, win=sr * 6, xf=sr // 2)
        if sr != SR:  # scored at 24 kHz, as every loss
            o = resample_poly(o, UP[1], UP[0]).astype(np.float32)[:len(y)]
        out.append((f"{rid}_{k:02d}", o))
    return out


def refiner(run, wins, dev):
    from e12_diffusion import Denoiser, Schedule, condition, denorm, refine
    cfg = json.loads((RES / "e12" / run / "config.json").read_text())
    h = AttrDict(json.loads((CKPTS[SR] / "config.json").read_text()))
    assert not cfg.get("stems")
    net = Denoiser(cond_ch=100 + 1).to(dev)
    net.load_state_dict(torch.load(RES / "e12" / run / "model.pt", map_location=dev))
    net.eval()
    sch = Schedule(dev)
    voc = Refiner(h).to(dev)
    voc.load_state_dict(torch.load(RES / "e12" / "p2_real" / "generator.pt", map_location="cpu"))
    voc.g.remove_weight_norm()
    voc.eval()
    d = 20 if cfg["kind"] == "shallow" else None
    gen = torch.Generator(device=dev).manual_seed(0)
    out = []
    for k, (rid, y, (nm, arr)) in enumerate(wins):
        xx = torch.tensor(arr[[0]])[None].to(dev)
        n = xx.shape[-1]
        xx = F.pad(xx, (0, (-n) % 256))
        g = 0.95 / max(float(xx[:, 0].abs().max()), 1e-4)
        with torch.no_grad():
            cond = condition(h, xx * g, [], True)
            m = refine(net, sch, cfg["kind"], cond, cond[:, :100], d, gen=gen, param=cfg.get("param", "eps"),
                       src_noise=cfg.get("src_noise", 0.0))
            o = voc.g(denorm(m))[0, 0, :n].float().cpu().numpy() / g
        out.append((f"{rid}_{k:02d}", o))
    return out


def ddsp_model(key, wins):
    sys.path.insert(0, "analysis")
    from compare_runs import load
    run = key.replace("e6-", "e6/", 1) if key.startswith("e6-") else key
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()  # all 20 (3 have no test window)
    m, recs, splits = load(run, ids)
    out, k = [], 0
    with torch.no_grad():
        for rid, rec, (_, test) in zip(ids, recs, splits):
            for j, w0 in enumerate(test):
                assert wins[k][0] == rid and np.allclose(rec.y[int(w0):int(w0) + L], wins[k][1], atol=1e-4), (rid, k)
                o, _ = m(rec, int(w0), noise_seed=j)
                out.append((f"{rid}_{k:02d}", o.float().cpu().numpy()))
                k += 1
    assert k == len(wins), k
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wins = eval_windows(REN)
    assert len(wins) == 85, len(wins)
    for name in sys.argv[1:]:
        if name == "real":
            clips = [(f"{rid}_{k:02d}", y) for k, (rid, y, _) in enumerate(wins)]
        elif name == "realtest":
            clips = realtest()
        elif name == "ddsp":
            clips = [(f"{rid}_{k:02d}", x[1][0]) for k, (rid, _, x) in enumerate(wins)]
        elif name.startswith("voc-"):
            run = name[4:].removesuffix("-real")
            clips = vocoder(run, wins, name.endswith("-real"), dev)
        elif name.startswith("diff-"):
            clips = refiner(name[5:], wins, dev)
        elif name.startswith("model-"):
            clips = ddsp_model(name[6:], wins)
        else:
            raise SystemExit(f"unknown version {name}")
        write(name, clips)


if __name__ == "__main__":
    main()
