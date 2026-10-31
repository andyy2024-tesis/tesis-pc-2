"""Experiment 6: adversarial fine-tuning of a trained experiment-4 model.

Starts from analysis/results/e4/<run>/model.pt and trains with
    multi-scale spectral loss (log term weighted by --spec-log-weight) + ADV * hinge adversarial loss + FM * feature matching
against a multi-resolution STFT discriminator (tesis.model.discriminator). Then evaluates on the held-out windows,
measures the catch-all noise level in the quietest parts ("wind"), and renders ~9 s held-out excerpts
(original, before, after) for listening.

Every --ckpt-every steps the generator, discriminator, both optimizers and the RNG states are saved to checkpoint.pt,
so a run can be continued exactly with --resume. Every --clip-every steps the listening excerpts are rendered to
audio/step<N>/ so one can hear how the sound evolves.

Usage: uv run python scripts/e6_adversarial.py --run structured_v5_body_audio [--steps 3000] [--resume]
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

import e4_train as e  # noqa: E402
from tesis.data.gaps import REPO
from tesis.model.discriminator import MultiResDiscriminator, d_hinge_loss, feature_matching, g_adv_loss
from tesis.synth.dsp import MultiScaleSpectralLoss

sys.path.insert(0, str(REPO / "analysis"))
from e4_targeted import load_run  # noqa: E402

SR, L = 24000, 72000
OUT = REPO / "analysis" / "results" / "e6"
LISTEN_IDS = ["023_Ypswc", "176_mM1wc", "276_jw1wc", "277_Pw1wc", "306_7c1wc"]
EXCERPT_S, XFADE_S = 9.0, 0.5


@torch.no_grad()
def render_long(model, rec, start, dur=EXCERPT_S, seed=0):
    """Render consecutive 3 s windows overlapping by XFADE_S with linear crossfades; returns (model, original)."""
    hop = L - int(XFADE_S * SR)
    n = int(dur * SR)
    out = np.zeros(n + L, np.float32)
    wsum = np.zeros(n + L, np.float32)
    ramp = np.ones(L, np.float32)
    xf = int(XFADE_S * SR)
    ramp[:xf] = np.linspace(0, 1, xf)
    ramp[-xf:] = np.linspace(1, 0, xf)
    for i, a in enumerate(range(0, n, hop)):
        w0 = start + a
        if w0 + L > len(rec.y):
            break
        y, _ = model(rec, int(w0), noise_seed=seed + i)
        out[a:a + L] += y.cpu().numpy() * ramp
        wsum[a:a + L] += ramp
    m = wsum > 1e-3
    out[m] /= wsum[m]
    k = int(min(n, len(rec.y) - start))
    return out[:k], rec.y[start:start + k]


@torch.no_grad()
def quiet_noise_db(model, recs, splits):
    """Catch-all noise level relative to the original in the quietest 20% of 50 ms frames of each test window."""
    vals = []
    for rec, (_, test) in zip(recs, splits):
        for j, w0 in enumerate(test):
            _, parts = model(rec, int(w0), noise_seed=j)
            if "noise" not in parts:
                continue
            y = rec.y[w0:w0 + L].reshape(-1, 1200)
            nz = parts["noise"].cpu().numpy().reshape(-1, 1200)
            q = np.argsort((y ** 2).mean(1))[: len(y) // 5]
            vals.append(10 * np.log10((nz[q] ** 2).mean() / ((y[q] ** 2).mean() + 1e-12) + 1e-12))
    return float(np.median(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--adv", type=float, default=1.0)
    ap.add_argument("--fm", type=float, default=2.0)
    ap.add_argument("--warmup-d", type=int, default=200, help="discriminator-only steps first")
    ap.add_argument("--d-ch", type=int, default=32, help="discriminator channels")
    ap.add_argument("--d-lr-mult", type=float, default=1.0)
    ap.add_argument("--d-kind", choices=["complex", "logmag", "both"], default="complex",
                    help="discriminator view of the STFT: complex (linear), log magnitude, or both")
    ap.add_argument("--config", default="{}", help='NetConfig overrides, e.g. {"noise": false}')
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--clip-every", type=int, default=0, help="render listening excerpts every N steps (0: only at end)")
    ap.add_argument("--resume", action="store_true", help="continue from <out_dir>/checkpoint.pt")
    ap.add_argument("--spec-log-weight", type=float, default=1.0,
                    help="weight of the log-magnitude term of the training spectral loss (0: linear term only)")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    out_dir = OUT / f"{a.run}{a.tag}"
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    model, (recs, splits) = load_run(a.run, ids, json.loads(a.config))
    base, _ = load_run(a.run, ids)  # frozen copy for "before" renders
    model.train()
    disc = MultiResDiscriminator(ch=a.d_ch, kind=a.d_kind).to(dev)
    opt_g = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.lr * a.d_lr_mult, betas=(0.5, 0.9))
    mss = MultiScaleSpectralLoss()  # evaluation metric, comparable across runs
    train_spec = MultiScaleSpectralLoss(log_weight=a.spec_log_weight)
    rng = np.random.default_rng(0)
    (out_dir / "meta.json").write_text(json.dumps(dict(base=a.run, overrides=json.loads(a.config), args=vars(a))))
    first = 1
    if a.resume:
        ck = torch.load(out_dir / "checkpoint.pt", weights_only=False)
        model.load_state_dict(ck["model"])
        disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        before, hist, first = ck["before"], ck["hist"], ck["step"] + 1
        print(f"resumed at step {ck['step']}", flush=True)
    else:
        before = e.evaluate(model, recs, splits, mss)
        before["quiet_noise_db"] = quiet_noise_db(model, recs, splits)
        hist = []
        print("before:", json.dumps(before), flush=True)
    t0 = time.time()
    for step in range(first, a.steps + 1):
        opt_g.zero_grad()
        opt_d.zero_grad()
        logs = np.zeros(4)
        for _ in range(a.batch):
            r = int(rng.integers(len(recs)))
            w0 = int(rng.choice(splits[r][0]))
            fake, _ = model(recs[r], w0)
            real = torch.tensor(recs[r].y[w0:w0 + L], device=dev)[None]
            fake = fake[None]
            # discriminator
            d_loss = d_hinge_loss(disc(real), disc(fake.detach()))
            (d_loss / a.batch).backward()
            # generator (discriminator weights not updated by this backward)
            if step > a.warmup_d:
                for p in disc.parameters():
                    p.requires_grad_(False)
                real_out, fake_out = disc(real), disc(fake)
                spec = train_spec(fake, real)
                adv, fm = g_adv_loss(fake_out), feature_matching(real_out, fake_out)
                g_loss = spec + a.adv * adv + a.fm * fm + model.regularizer()
                (g_loss / a.batch).backward()
                for p in disc.parameters():
                    p.requires_grad_(True)
                logs += np.array([d_loss.item(), spec.item(), adv.item(), fm.item()]) / a.batch
            else:
                logs[0] += d_loss.item() / a.batch
        opt_d.step()
        if step > a.warmup_d:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()
        if step % 100 == 0:
            print(f"  step {step:5d}  D {logs[0]:.3f}  spec {logs[1]:.3f}  adv {logs[2]:.3f}  fm {logs[3]:.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        if step % 1000 == 0 or step == a.steps:
            ev = e.evaluate(model, recs, splits, mss)
            ev["quiet_noise_db"] = quiet_noise_db(model, recs, splits)
            hist.append(dict(step=step, **ev))
            print(f"  EVAL step {step}: {json.dumps(ev)}", flush=True)
            torch.save(model.state_dict(), out_dir / "model.pt")
            (out_dir / "history.json").write_text(json.dumps(dict(before=before, after=hist), indent=1))
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save(dict(step=step, model=model.state_dict(), disc=disc.state_dict(), opt_g=opt_g.state_dict(),
                            opt_d=opt_d.state_dict(), np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(), before=before, hist=hist), out_dir / "checkpoint.pt")
        if a.clip_every and step % a.clip_every == 0 and step != a.steps:
            model.eval()
            write_clips(model, base, recs, splits, ids, out_dir / "audio" / f"step{step}")
            model.train()
    model.eval()
    write_clips(model, base, recs, splits, ids, out_dir / "audio")


def write_clips(model, base, recs, splits, ids, folder):
    """~9 s held-out excerpts: original, before fine-tuning, current model (same peak normalization for all three)."""
    folder.mkdir(parents=True, exist_ok=True)
    for rid in LISTEN_IDS:
        r = ids.index(rid)
        test = splits[r][1]
        if not len(test):
            continue
        start = int(test[0])
        after_y, orig = render_long(model, recs[r], start)
        before_y, _ = render_long(base, recs[r], start)
        pk = max(np.abs(orig).max(), np.abs(after_y).max(), np.abs(before_y).max()) * 1.05
        sf.write(folder / f"{rid}_1_original.wav", orig / pk, SR)
        sf.write(folder / f"{rid}_2_before.wav", before_y / pk, SR)
        sf.write(folder / f"{rid}_3_after_adversarial.wav", after_y / pk, SR)


if __name__ == "__main__":
    main()
