"""Experiment 12, V3 / V3b: repair the DDSP mel spectrogram with shallow diffusion or with a flow from DDSP to real, then
make audio with the phase-2 BigVGAN (real mel -> real audio).

Both use the same denoiser (DiffSinger's WaveNet-style network, ~20M parameters) on BigVGAN's log mel (100 bands,
10.7 ms frames), scaled to [-1, 1].
  --kind shallow  DDPM, 100 steps (DiffSinger's linear schedule). Conditioned on the DDSP mel; trained on the DDSP
                  pairs and, with the condition switched off (zeros + flag 0), on real mels of every GAPS training
                  recording, so the network also knows real guitar mels without DDSP. At inference the reverse process
                  starts from the DDSP mel noised to depth k (instead of from pure noise): k is the dial between
                  trusting the DDSP (small) and freedom (large), swept without retraining.
                  --param x0res: the network predicts the clean mel as the DDSP mel plus a correction (zero at the
                  start, so an untrained model returns the DDSP mel); the reverse step uses the DDPM posterior mean.
                  (First run with --param eps, DiffSinger's noise prediction: after 10k steps its noise error stayed
                  ~0.15 even at heavy noise, so its clean-mel guess at depth 20 was already worse than the DDSP mel.)
  --kind flow     rectified flow from the DDSP mel (t = 0) to the real mel (t = 1) with a little bridge noise, conditioned
                  on the DDSP mel; pairs only. Inference integrates from the DDSP mel in --flow-steps Euler steps.
  --stems         the condition also has the mel of every DDSP part (as scripts/e12_vocoder.py --stems).
Evaluation: the 85 held-out windows, multi-scale spectral loss of the audio vs the real recording (the DDSP measure).
Writes analysis/results/e12/<tag>/ (model.pt, log.txt).
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tesis.data.gaps import REPO
from tesis.synth.dsp import MultiScaleSpectralLoss

sys.path.insert(0, str(REPO / "scripts"))
from e12_vocoder import CKPT, D, RES, SR, Data, Refiner, eval_windows, mel, stems_of  # noqa: E402
from env import AttrDict  # noqa: E402  (BigVGAN's, on the path through e12_vocoder)

M_MIN, M_MAX = -11.5, 2.5  # BigVGAN log mel range (log of clamp(mel, 1e-5) ... loud peaks)
T_STEPS = 100


def norm(m):
    return (m - M_MIN) / (M_MAX - M_MIN) * 2 - 1


def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2 * (M_MAX - M_MIN) + M_MIN


class StepEmb(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.Mish(), nn.Linear(4 * c, c))

    def forward(self, t):  # t: (B,) float, 0..1000
        half = self.c // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        e = t[:, None] * f[None]
        return self.mlp(torch.cat([e.sin(), e.cos()], -1))


class ResBlock(nn.Module):
    def __init__(self, c, cond_ch, dil):
        super().__init__()
        self.step = nn.Linear(c, c)
        self.conv = nn.Conv1d(c, 2 * c, 3, padding=dil, dilation=dil)
        self.cond = nn.Conv1d(cond_ch, 2 * c, 1)
        self.out = nn.Conv1d(c, 2 * c, 1)

    def forward(self, x, cond, s):
        y = self.conv(x + self.step(s)[..., None]) + self.cond(cond)
        g, f = y.chunk(2, 1)
        r, skip = self.out(torch.sigmoid(g) * torch.tanh(f)).chunk(2, 1)
        return (x + r) / math.sqrt(2), skip


class Denoiser(nn.Module):
    """DiffSinger's denoiser: dilated residual layers (cycle of 10), step embedding, 1x1 condition at every layer."""

    def __init__(self, n_mel=100, cond_ch=101, c=256, layers=20):
        super().__init__()
        self.inp = nn.Conv1d(n_mel, c, 1)
        self.emb = StepEmb(c)
        self.blocks = nn.ModuleList([ResBlock(c, cond_ch, 2 ** (i % 10)) for i in range(layers)])
        self.skip = nn.Conv1d(c, c, 1)
        self.outp = nn.Conv1d(c, n_mel, 1)
        nn.init.zeros_(self.outp.weight)
        nn.init.zeros_(self.outp.bias)

    def forward(self, x, t, cond):
        h = F.relu(self.inp(x))
        s = self.emb(t)
        acc = 0
        for b in self.blocks:
            h, sk = b(h, cond, s)
            acc = acc + sk
        return self.outp(F.relu(self.skip(acc / math.sqrt(len(self.blocks)))))


class Schedule:
    def __init__(self, dev):
        self.beta = torch.linspace(1e-4, 0.06, T_STEPS, device=dev)
        self.alpha = 1 - self.beta
        self.abar = torch.cumprod(self.alpha, 0)


def condition(h, x, groups, paired):
    """(B, 100 [+ 100 per stem] + 1, frames): DDSP mel (scaled), stems, flag; zeros and flag 0 for unpaired rows."""
    c = [norm(mel(h, x[:, 0]))]
    for g in groups:
        c.append(norm(mel(h, sum(x[:, 1 + j] for j in g))))
    c = torch.cat(c, 1)
    flag = torch.ones_like(c[:, :1])
    if not paired:
        c, flag = torch.zeros_like(c), torch.zeros_like(flag)
    return torch.cat([c, flag], 1)


@torch.no_grad()
def refine(net, sch, kind, cond, start, depth, flow_steps=32, gen=None, param="eps"):
    """start: scaled DDSP mel (B, 100, F). Shallow: noise it to step `depth` and run the reverse process to 0.
    Flow: integrate the velocity from t = 0 (DDSP) to 1."""
    if kind == "flow":
        x = start.clone()
        for i in range(flow_steps):
            t = torch.full((x.shape[0],), i / flow_steps, device=x.device)
            x = x + net(x, t * 1000, cond) / flow_steps
        return x
    if depth == 0:
        return start
    k = depth - 1
    noise = torch.randn(start.shape, device=start.device, generator=gen)
    x = sch.abar[k].sqrt() * start + (1 - sch.abar[k]).sqrt() * noise
    for i in range(k, -1, -1):
        t = torch.full((x.shape[0],), float(i), device=x.device)
        out = net(x, t * 10, cond)  # steps 0..99 -> 0..990 for the embedding
        a, ab, b = sch.alpha[i], sch.abar[i], sch.beta[i]
        if param == "x0res":  # posterior mean from the predicted clean mel (DDSP mel + correction)
            x0 = (cond[:, :100] + out).clamp(-1.2, 1.2)
            if i == 0:
                return x0
            abp = sch.abar[i - 1]
            x = (abp.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - abp) / (1 - ab)) * x
        else:
            x = (x - b / (1 - ab).sqrt() * out) / a.sqrt()
        if i > 0:
            var = b * (1 - sch.abar[i - 1]) / (1 - ab)
            x = x + var.sqrt() * torch.randn(x.shape, device=x.device, generator=gen)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kind", choices=["shallow", "flow"], default="shallow")
    ap.add_argument("--param", choices=["eps", "x0res"], default="eps", help="shallow: what the network predicts")
    ap.add_argument("--renders", required=True, help="analysis/results/e12/renders/<name>")
    ap.add_argument("--vocoder", default="p2_real", help="e12 run whose generator turns mels into audio")
    ap.add_argument("--stems", action="store_true")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--frames", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--unpaired", type=float, default=0.5, help="shallow: share of real-only batches (condition off)")
    ap.add_argument("--bridge", type=float, default=0.1, help="flow: bridge noise")
    ap.add_argument("--depths", default="0,20,40,60,80", help="shallow: depths evaluated (of 100)")
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--eval-only", action="store_true", help="load <tag>/model.pt and only evaluate (e.g. with p2_real)")
    a = ap.parse_args()
    dev = "cuda"
    out = RES / "e12" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.eval_only:
        (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    h = AttrDict(json.loads((CKPT / "config.json").read_text()))
    renders = RES / "e12" / "renders" / a.renders
    rids = sorted(p.stem for p in renders.glob("*.npz"))
    names = list(np.load(renders / f"{rids[0]}.npz")["parts"])
    groups = stems_of(names) if a.stems else []
    parts = sorted({p for g in groups for p in g})
    gidx = [[parts.index(p) for p in g] for g in groups]
    paired = Data(rids, renders)
    real = None
    if a.kind == "shallow" and a.unpaired > 0:
        man = pd.read_csv(D / "manifest.csv")
        real = Data(man[man.split == "train"].id.tolist())
    net = Denoiser(cond_ch=100 * (1 + len(groups)) + 1).to(dev)
    print(f"denoiser: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000) * max(0.05, 1 - s / a.steps))
    sch = Schedule(dev)
    voc = Refiner(h).to(dev)
    voc.load_state_dict(torch.load(RES / "e12" / a.vocoder / "generator.pt", map_location="cpu"))
    voc.g.remove_weight_norm()
    voc.eval()
    mss = MultiScaleSpectralLoss()
    wins = eval_windows(renders)
    depths = [int(d) for d in a.depths.split(",")] if a.kind == "shallow" else [None]
    log = open(out / "log.txt", "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    def evaluate():
        net.eval()
        res = {}
        gen = torch.Generator(device=dev).manual_seed(0)
        for d in depths:
            vals = []
            for rid, y, x in wins:
                nm, arr = x
                rows = [0] + [1 + nm.index(p) for p in parts]
                xx = torch.tensor(arr[rows])[None].to(dev)
                n = xx.shape[-1]
                xx = F.pad(xx, (0, (-n) % 256))
                g = 0.95 / max(float(xx[:, 0].abs().max()), 1e-4)
                xx = xx * g
                cond = condition(h, xx, gidx, True)
                m = refine(net, sch, a.kind, cond, cond[:, :100], d, gen=gen, param=a.param)
                with torch.no_grad():
                    o = voc.g(denorm(m))[0, 0, :n].float().cpu().numpy() / g
                vals.append(mss(torch.tensor(o)[None], torch.tensor(y)[None]).item())
            res[str(d)] = round(float(np.mean(vals)), 3)
        net.train()
        return res

    if a.eval_only:
        net.load_state_dict(torch.load(out / "model.pt", map_location=dev))
        say(f"{a.tag}: eval with vocoder {a.vocoder}: {evaluate()}")
        return
    say(f"{a.tag}: step 0 eval {evaluate()}")
    rng = np.random.default_rng(0)
    t0, hist = time.time(), []
    seg = a.frames * 256
    for step in range(1, a.steps + 1):
        use_real = real is not None and rng.random() < a.unpaired
        if use_real:
            y, _ = real.batch(a.batch, seg, rng)
            y = y.to(dev)
            x1 = norm(mel(h, y[:, 0]))
            cond = condition(h, torch.zeros(a.batch, 1 + len(parts), seg, device=dev), gidx, False)
        else:
            y, x = paired.batch(a.batch, seg, rng, parts)
            y, x = y.to(dev), x.to(dev)
            x1 = norm(mel(h, y[:, 0]))
            cond = condition(h, x, gidx, True)
        if a.kind == "shallow":
            k = torch.randint(0, T_STEPS, (a.batch,), device=dev)
            eps = torch.randn_like(x1)
            ab = sch.abar[k][:, None, None]
            out = net(ab.sqrt() * x1 + (1 - ab).sqrt() * eps, k.float() * 10, cond)
            loss = F.mse_loss(cond[:, :100] + out, x1) if a.param == "x0res" else F.mse_loss(out, eps)
        else:
            x0 = cond[:, :100]
            t = torch.rand(a.batch, device=dev)
            tt = t[:, None, None]
            xt = (1 - tt) * x0 + tt * x1 + a.bridge * (tt * (1 - tt)).sqrt() * torch.randn_like(x1)
            loss = F.mse_loss(net(xt, t * 1000, cond), x1 - x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(loss.item())
        if step % 500 == 0:
            say(f"  step {step:6d}  loss {np.mean(hist):.4f}  ({time.time() - t0:.0f} s)")
            hist = []
        if step % a.eval_every == 0 or step == a.steps:
            say(f"  EVAL step {step}: {evaluate()}")
            torch.save(net.state_dict(), out / "model.pt")
    say("done")


if __name__ == "__main__":
    main()
