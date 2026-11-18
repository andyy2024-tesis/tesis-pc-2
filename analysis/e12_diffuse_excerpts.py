"""Experiment 12: refine the 5 listening excerpts of the final DDSP with a V3 / V3b run (scripts/e12_diffusion.py) and
make audio with a vocoder run, for the listening page.

The DDSP excerpt (<label>.wav, plus its parts from analysis/render_stems.py when the run takes stems) -> mel -> refined
mel (shallow diffusion from depth k, or the flow) -> vocoder. Fixed noise seed per excerpt.
Writes slides/audio/<rid>/voc-<vocoder>-<run>[-d<k>]-<label>.wav.
Usage: uv run python analysis/e12_diffuse_excerpts.py <run> <vocoder run> <label> [depth ...]
  e.g. v3_x0res p2_real e6-structured_v12_final_learnedB_noise_strongD_rise 20 40
"""
import json
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from e12_diffusion import Denoiser, Schedule, condition, denorm, refine  # noqa: E402
from e12_vocoder import CKPT, RES, Refiner, stems_of  # noqa: E402
from env import AttrDict  # noqa: E402

from render_stems import AUDIO, LISTEN_IDS  # noqa: E402

PARTS = ["body", "burst", "contact", "floor", "noise", "strings"]
WIN = 24000 * 6  # long excerpts in 6 s windows, crossfaded 0.5 s (as e12_vocoder.vocode_long)
XF = 12000


def main():
    run, vrun, label = sys.argv[1:4]
    depths = [int(d) for d in sys.argv[4:]] or [None]
    cfg = json.loads((RES / "e12" / run / "config.json").read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h = AttrDict(json.loads((CKPT / "config.json").read_text()))
    groups = stems_of(PARTS) if cfg.get("stems") else []
    parts = sorted({p for g in groups for p in g})
    gidx = [[parts.index(p) for p in g] for g in groups]
    net = Denoiser(cond_ch=100 * (1 + len(groups)) + 1).to(dev)
    net.load_state_dict(torch.load(RES / "e12" / run / "model.pt", map_location=dev))
    net.eval()
    sch = Schedule(dev)
    voc = Refiner(h).to(dev)
    voc.load_state_dict(torch.load(RES / "e12" / vrun / "generator.pt", map_location="cpu"))
    voc.g.remove_weight_norm()
    voc.eval()
    ramp = np.ones(WIN, np.float32)
    ramp[:XF], ramp[-XF:] = np.linspace(0, 1, XF), np.linspace(1, 0, XF)
    for d in depths:
        for rid in LISTEN_IDS:
            mix = sf.read(AUDIO / rid / f"{label}.wav", dtype="float32")[0]
            rows = [mix] + [sf.read(f, dtype="float32")[0][:len(mix)] if (f := AUDIO / rid / "stems" / label / f"{p}.wav").exists()
                            else np.zeros_like(mix) for p in parts]
            x = torch.tensor(np.stack(rows))[None]
            n = x.shape[-1]
            out, wsum = np.zeros(n + WIN, np.float32), np.zeros(n + WIN, np.float32)
            gen = torch.Generator(device=dev).manual_seed(0)
            for a in range(0, n, WIN - XF):
                xx = x[..., a:a + WIN].to(dev)
                m = xx.shape[-1]
                if m < 2048:
                    break
                xx = F.pad(xx, (0, (-m) % 256))
                g = 0.95 / max(float(xx[:, 0].abs().max()), 1e-4)
                xx = xx * g
                with torch.no_grad():
                    cond = condition(h, xx, gidx, True)
                    mm = refine(net, sch, cfg["kind"], cond, cond[:, :100], d, gen=gen, param=cfg.get("param", "eps"),
                                src_noise=cfg.get("src_noise", 0.0))
                    o = voc.g(denorm(mm))[0, 0, :m].float().cpu().numpy() / g
                out[a:a + m] += o * ramp[:m]
                wsum[a:a + m] += ramp[:m]
            k = wsum > 1e-3
            out[k] /= wsum[k]
            tag = f"{run}-d{d}" if d is not None else run
            sf.write(AUDIO / rid / f"voc-{vrun}-{tag}-{label}.wav", out[:n], h.sampling_rate, subtype="FLOAT")
        print("done depth", d, flush=True)


if __name__ == "__main__":
    main()
