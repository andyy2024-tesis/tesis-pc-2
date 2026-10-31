"""Experiment 4: train the note-conditioned guitar synthesizer (structured vs plain DDSP) on ~20 GAPS recordings.

Train on the first 85% of each recording, test on windows that start after the 85% point (never seen).
Reports test spectral loss and energy shares, and renders held-out excerpts for listening.

Usage: uv run python scripts/e4_train.py --variant structured|ddsp [--steps 6000] [--batch 8]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from tesis.data.gaps import PROCESSED, REPO
from tesis.data.loudness import note_loudness_db, partial_freqs
from tesis.data.note_tracks import voice_tracks
from tesis.model.guitar_net import GuitarNet, NetConfig, Recording
from tesis.synth.dsp import MultiScaleMelLoss, MultiScaleSpectralLoss
from tesis.synth.piece import LOG_B

import e2_one_piece as e2  # noqa: E402  (valid_starts)

SR = 24000
D = PROCESSED / f"sr{SR}"
OUT = REPO / "analysis" / "results" / "e4"
TEST_FROM = 0.85
EVAL_PER_REC = 5


def extra_ids(n):
    """n more GAPS training-split recordings with good transcriptions (f-measure >= 0.93, like the 20 of
    e4_recordings.csv), in a fixed random order; used for training only."""
    man = pd.read_csv(D / "manifest.csv")
    base = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id
    c = man[(~man.id.isin(base)) & (man["f-measure"] >= 0.93) & (man.split == "train")].id.sort_values()
    return c.sample(frac=1, random_state=0).tolist()[:n]


def load_recordings(ids, cfg, n_eval=None):
    """Recordings and their (train starts, test windows); only the first n_eval (default all) get test windows."""
    man = pd.read_csv(D / "manifest.csv").set_index("id")
    recs, splits = [], []
    L = int(cfg.win * SR)
    for i, rid in enumerate(ids):
        y = np.load(D / "audio" / f"{rid}.npy").astype(np.float32) / 32768
        notes = pd.read_csv(D / "notes" / f"{rid}.csv")
        tc = man.loc[rid, "tuning_cents"]
        loud = note_loudness_db(y, SR, notes.onset.values, partial_freqs(notes.pitch.values, tc, LOG_B))
        notes["loud_rel_db"] = loud - np.nanmedian(loud)
        rec = Recording(i, y, notes, tc, cfg, LOG_B)
        if cfg.audio_feats:
            cache = D / "tracks" / f"{rid}.npz"
            if cache.exists():
                z = np.load(cache)
                loud_t, dev_t = z["loud"], z["dev"]
            else:  # measured at stiff-string partial positions for both variants
                loud_t, dev_t = voice_tracks(y, SR, cfg.hop, rec.onset, rec.end, rec.voice,
                                             partial_freqs(rec.pitch, tc, LOG_B), cfg.n_voices)
                cache.parent.mkdir(exist_ok=True)
                np.savez(cache, loud=loud_t, dev=dev_t)
            rec.set_tracks(loud_t, dev_t)
        recs.append(rec)
        starts = e2.valid_starts(y, L, cfg.hop)
        # only where the piece is playing: YouTube recordings have talk, applause or room noise before/after
        music_lo = int((notes.onset.min() - 0.5) * SR)
        music_hi = int((notes.onset.max() + 2.0) * SR)
        starts = starts[(starts >= music_lo) & (starts + L <= music_hi)]
        split = int(TEST_FROM * len(y))
        train = starts[starts + L <= split]
        test = starts[starts >= split]
        test = test[np.linspace(0, len(test) - 1, min(EVAL_PER_REC, len(test))).astype(int)] if len(test) else test
        if n_eval is not None and i >= n_eval:
            test = test[:0]
        splits.append((train, test))
    return recs, splits


@torch.no_grad()
def evaluate(model, recs, splits, loss_fn):
    model.eval()
    losses, shares = [], {}
    for rec, (_, test) in zip(recs, splits):
        L = int(model.cfg.win * SR)
        for j, w0 in enumerate(test):
            out, parts = model(rec, int(w0), noise_seed=j)
            y = torch.tensor(rec.y[w0:w0 + L], device=out.device)
            losses.append(loss_fn(out[None], y[None]).item())
            tot = sum((p ** 2).sum().item() for p in parts.values())
            for k, p in parts.items():
                shares.setdefault(k, []).append((p ** 2).sum().item() / tot)
    model.train()
    return dict(test_loss=float(np.mean(losses)), n_windows=len(losses),
                **{f"share_{k}_db": float(10 * np.log10(np.mean(v) + 1e-12)) for k, v in shares.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["structured", "ddsp"], default="structured")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--config", default="{}", help='NetConfig overrides as JSON, e.g. {"noise": false}')
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    ap.add_argument("--loss", choices=["mss", "mel"], default="mss", help="training loss (evaluation always reports mss)")
    ap.add_argument("--seed", type=int, default=0, help="initialization and window-sampling seed")
    ap.add_argument("--extra-recordings", type=int, default=0,
                    help="train also on this many more recordings (evaluation stays on the 20 of e4_recordings.csv)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda"
    cfg = NetConfig(sr=SR, structured=a.variant == "structured", **json.loads(a.config))
    out_dir = OUT / f"{a.variant}{a.tag}"
    (out_dir).mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    extra = extra_ids(a.extra_recordings)
    (out_dir / "config.json").write_text(json.dumps(dict(variant=a.variant, overrides=json.loads(a.config), loss=a.loss,
                                                         seed=a.seed, **(dict(extra_ids=extra) if extra else {}))))
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    recs, splits = load_recordings(ids + extra, cfg, n_eval=len(ids))
    print(f"{a.variant}: {len(recs)} recordings, {sum(len(s[0]) for s in splits)} train starts, "
          f"{sum(len(s[1]) for s in splits)} test windows", flush=True)

    model = GuitarNet(cfg, len(recs)).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    # per-recording settings (reverb, body EQ/filter, noise floor, transient spectrum) are each seen only when their
    # recording is sampled, so they get a 10x learning rate (otherwise they barely move from their initial values)
    rec_p = [p for n, p in model.named_parameters() if n.startswith("rec_") and n != "rec_emb.weight"]
    rest = [p for n, p in model.named_parameters() if not (n.startswith("rec_") and n != "rec_emb.weight")]
    opt = torch.optim.Adam([dict(params=rest, lr=a.lr), dict(params=rec_p, lr=10 * a.lr)])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, 10 * a.lr], total_steps=a.steps, pct_start=0.05)
    loss_fn = MultiScaleSpectralLoss()  # evaluation metric, comparable across runs
    train_loss = MultiScaleMelLoss(SR).to(dev) if a.loss == "mel" else loss_fn
    rng = np.random.default_rng(a.seed)
    L = int(cfg.win * SR)
    history, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        tot = 0.0
        for _ in range(a.batch):
            r = int(rng.integers(len(recs)))
            w0 = int(rng.choice(splits[r][0]))
            out, _ = model(recs[r], w0)
            y = torch.tensor(recs[r].y[w0:w0 + L], device=dev)
            loss = train_loss(out[None], y[None]) + model.regularizer()
            (loss / a.batch).backward()
            tot += loss.item() / a.batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"  step {step:5d}  train loss {tot:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(model, recs, splits, loss_fn)
            history.append(dict(step=step, train_loss=tot, **ev))
            print(f"  EVAL step {step}: {json.dumps(ev)}", flush=True)
            torch.save(model.state_dict(), out_dir / "model.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=1))

    # held-out excerpts for listening: first test window of four recordings, original vs model
    model.eval()
    with torch.no_grad():
        for rec, (_, test), rid in list(zip(recs, splits, ids))[:4]:
            if not len(test):
                continue
            w0 = int(test[0])
            out, parts = model(rec, w0, noise_seed=0)
            y = rec.y[w0:w0 + L]
            pk = max(np.abs(y).max(), out.abs().max().item()) * 1.05
            sf.write(out_dir / "audio" / f"{rid}_original.wav", y / pk, SR)
            sf.write(out_dir / "audio" / f"{rid}_{a.variant}.wav", out.cpu().numpy() / pk, SR)


if __name__ == "__main__":
    main()
