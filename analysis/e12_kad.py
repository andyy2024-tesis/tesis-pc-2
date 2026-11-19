"""Experiment 12: Kernel Audio Distance (KAD; Chung et al. 2025) of every version to real guitar, with confidence.

Runs in the kadtk environment (external/kadtk/.venv; kadtk 1.x with two local fixes: a stray `from turtle import st`
removed, checkpoints loaded on the CPU), on the clip folders of analysis/e12_kad_clips.py.
- Embeddings (kadtk models, cached per clip as kadtk does, clips/<version>/embeddings/<model>/<clip>.npy):
  panns-wavegram-logmel (2048 per clip; the KAD authors' choice) and encodec-emb (128 per second of audio; codec
  features, the most sensitive to distortion and low-pass in Gui et al. 2024). CLAP is left out: it pads every clip
  to 10 s, and ours are 3 s.
- KAD = 100 x unbiased MMD^2 with a Gaussian kernel, as kadtk, except that the bandwidth is the median pairwise
  distance of the REFERENCE embeddings for every version (kadtk takes it from the evaluated set, so each version would
  be measured with a different kernel).
- References: "real" (the same 85 held-out windows; versions are paired with it window by window) and "realtest"
  (600 clips of the 30 GAPS test-split recordings that no model heard).
- Estimator: the U-statistic over clips. The kernel between two clips is the mean over their frames, and only
  distinct clips are compared: within a set, no clip with itself (frames of one clip are not independent); against
  the paired reference "real", no window with its own real window (a version's clip shares notes and timing with
  it, which made kadtk's cross term too large and the score negative). Against "realtest" (independent) every
  cross pair counts. For PANNs (one vector per clip) against "realtest" this is kadtk's formula.
- Confidence: 1000 bootstrap resamples of the clips (the same window resample for every version and for "real", an
  independent one for "realtest"); copies of one clip count as the same clip. 95% intervals for each KAD and for
  each difference to the final DDSP and to V1.
Writes analysis/results/e12_kad.json.
Usage: external/kadtk/.venv/bin/python analysis/e12_kad.py ddsp voc-p2_real voc-v1_mix20 ...
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": "cpu"})  # PANNs checkpoint was saved on a GPU
import soundfile as sf  # noqa: E402
import torchaudio  # noqa: E402
from kadtk.model_loader import ModelLoader, get_all_models  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CLIPS = REPO / "analysis" / "results" / "e12" / "kad" / "clips"
OUT = REPO / "analysis" / "results" / "e12_kad.json"
MODELS = ["panns-wavegram-logmel", "encodec-emb"]
NB = 1000
# paired differences (same resamples) with 95% intervals; KAD_BASES=a,b,... adds more
BASES = ["ddsp", "voc-v1_mix20"] + [b for b in os.environ.get("KAD_BASES", "").split(",") if b]


def load(model, f):
    """The clip at the model's rate. Models with their own reader (EnCodec) resample there; the base reader (PANNs)
    does not, so resample here with kadtk's own Kaiser settings (its EmbeddingLoader does it through a file cache)."""
    if type(model).load_wav is not ModelLoader.load_wav:
        return model.load_wav(f)
    x, sr = sf.read(f, dtype="float32")
    if sr != model.sr:
        x = torchaudio.functional.resample(torch.tensor(x)[None], sr, model.sr, lowpass_filter_width=64,
                                           rolloff=0.9475937167399596, resampling_method="sinc_interp_kaiser",
                                           beta=14.769656459379492)[0].numpy()
    return x


def embed(model, name):
    """{clip: (frames, d) float32} for a version folder, computing and caching what is missing."""
    d = CLIPS / name
    cache = d / "embeddings" / model.name
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in sorted(d.glob("*.wav")):
        c = cache / f"{f.stem}.npy"
        if not c.exists():
            if model.model is None:
                model.load_model()
            np.save(c, model.get_embedding(load(model, f)))
        out[f.stem] = np.load(c).astype(np.float32).reshape(-1, model.num_features)
    return out


def blocks(a, b, gamma):
    """Mean kernel between the frames of each clip of a and each clip of b: (clips of a, clips of b)."""
    xa, xb = (torch.tensor(np.concatenate(list(e.values()))) for e in (a, b))
    ia = torch.repeat_interleave(torch.arange(len(a)), torch.tensor([len(v) for v in a.values()]))
    ib = torch.repeat_interleave(torch.arange(len(b)), torch.tensor([len(v) for v in b.values()]))
    pa = torch.zeros(len(a), len(xa)).index_put_((ia, torch.arange(len(xa))), torch.tensor(1.0))
    pb = torch.zeros(len(b), len(xb)).index_put_((ib, torch.arange(len(xb))), torch.tensor(1.0))
    k = torch.exp(-gamma * torch.cdist(xa.double(), xb.double()).pow(2)).float()
    return ((pa @ k @ pb.T) / (pa.sum(1)[:, None] * pb.sum(1)[None, :])).double().numpy()


def distinct(B, w):
    """Mean of B[c, d] over pairs of distinct clips c != d, clip weights w (rows: resamples)."""
    num = np.einsum("rc,cd,rd->r", w, B, w) - (w ** 2) @ np.diag(B)
    return num / (w.sum(1) ** 2 - (w ** 2).sum(1))


def main():
    names = sys.argv[1:]
    ms = {m.name: m for m in get_all_models()}
    rng = np.random.default_rng(0)
    res = {}
    wins = None
    for mname in MODELS:
        model = ms[mname]
        emb = {v: embed(model, v) for v in ["real", "realtest"] + names}
        keys = sorted(emb["real"])
        wins = wins or keys
        assert keys == wins and all(sorted(emb[v]) == keys for v in names), "versions must share the 85 windows"
        # bootstrap weights: windows (shared by "real" and every version) and the realtest clips
        ww = np.vstack([np.ones(len(keys)), rng.multinomial(len(keys), np.ones(len(keys)) / len(keys), NB)])
        nt = len(emb["realtest"])
        wt = np.vstack([np.ones(nt), rng.multinomial(nt, np.ones(nt) / nt, NB)])
        for ref, wr in (("real", ww), ("realtest", wt)):
            xr = np.concatenate(list(emb[ref].values()))
            sigma = float(torch.pdist(torch.tensor(xr).double()).median())
            gamma = 1 / (2 * sigma ** 2)
            krr = distinct(blocks(emb[ref], emb[ref], gamma), wr)
            kad = {}
            for v in names + (["real"] if ref == "realtest" else []):
                kss = distinct(blocks(emb[v], emb[v], gamma), ww)
                Brs = blocks(emb[ref], emb[v], gamma)
                if ref == "real":  # paired: a window is never compared with its own real window
                    krs = distinct(Brs, ww)
                else:
                    krs = np.einsum("rc,cd,rd->r", wr, Brs, ww) / (wr.sum(1) * ww.sum(1))
                kad[v] = 100 * (krr + kss - 2 * krs)
            r = {}
            for v, s in kad.items():
                e = dict(kad=round(float(s[0]), 4), ci=[round(float(q), 4) for q in np.percentile(s[1:], [2.5, 97.5])])
                for base in BASES:
                    if base in kad and v != base:
                        dlt = s - kad[base]
                        e[f"minus_{base}"] = round(float(dlt[0]), 4)
                        e[f"minus_{base}_ci"] = [round(float(q), 4) for q in np.percentile(dlt[1:], [2.5, 97.5])]
                r[v] = e
            res[f"{mname} vs {ref}"] = dict(bandwidth=round(sigma, 4), versions=r)
            print(f"\n{mname} vs {ref} (bandwidth {sigma:.3f})", flush=True)
            for v, e in r.items():
                extra = "".join(f"   - {b}: {e[f'minus_{b}']:+.3f} [{e[f'minus_{b}_ci'][0]:+.3f}, {e[f'minus_{b}_ci'][1]:+.3f}]"
                                for b in BASES if f"minus_{b}" in e)
                print(f"  {v:24s} {e['kad']:7.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]{extra}", flush=True)
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    for k, v in res.items():
        old.setdefault(k, {"bandwidth": v["bandwidth"], "versions": {}})
        old[k]["bandwidth"] = v["bandwidth"]
        old[k]["versions"].update(v["versions"])
    OUT.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
