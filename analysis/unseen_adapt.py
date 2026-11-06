"""Unseen recordings: does a trained model generalize to guitars/rooms/players it never heard?

Takes a trained run, adds recordings it never saw (the first N of e4_train.extra_ids), freezes every shared weight and
fits only the per-recording parameters (rec_*: embedding, body filter/EQ, floor, transient, ...) on the first 85 % of
each new recording; then scores the held-out windows of the new recordings (same rule as E4: windows after 85 %).
Reference: a run that was trained with those recordings included (--extra-recordings N --eval-extra), scored on the
same windows, i.e. how well the new recordings can be fitted when they are seen.

Usage: uv run python analysis/unseen_adapt.py <run> [--n 10] [--steps 1500] [--ref <run trained with them>]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
import e4_train as e  # noqa: E402
from compare_runs import RES  # noqa: E402
from e4_targeted import upgrade_state_dict  # noqa: E402
from e6_adversarial import quiet_noise_db  # noqa: E402
from tesis.model.guitar_net import GuitarNet, NetConfig  # noqa: E402
from tesis.synth.dsp import MultiScaleSpectralLoss  # noqa: E402

SR = 24000
DEV = "cuda"


def score(model, recs, splits, mss):
    ev = e.evaluate(model, recs, splits, mss)
    with torch.no_grad():
        wind = quiet_noise_db(model, recs, splits)
    return dict(loss=round(ev["test_loss"], 3), n_windows=ev["n_windows"], wind=round(wind, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--train-seconds", type=float, default=None,
                    help="fit on only this many seconds of each new recording (from the first playable window)")
    a = ap.parse_args()
    torch.manual_seed(0)
    ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
    new = e.extra_ids(a.n)
    c = json.loads((RES / "e4" / a.run / "config.json").read_text())
    assert not c.get("extra_ids"), "the base run must not have trained on extra recordings"
    cfg = NetConfig(sr=SR, structured=c["variant"] == "structured", **c["overrides"])
    recs, splits = e.load_recordings(new, cfg)
    if a.train_seconds:  # windows must lie inside the first train_seconds of music
        L = int(cfg.win * SR)
        splits = [(tr[tr + L <= tr[0] + int(a.train_seconds * SR)] if len(tr) else tr, te) for tr, te in splits]  # >= 1 if >= win
    n0 = len(ids)

    # trained weights; the new recordings' rows of every per-recording table start at the mean of the trained rows
    sd = upgrade_state_dict(torch.load(RES / "e4" / a.run / "model.pt", map_location=DEV))
    model = GuitarNet(cfg, n0 + len(new)).to(DEV)
    own = model.state_dict()
    for k, v in sd.items():
        if k.startswith("rec_") and v.shape[0] == n0:
            own[k] = torch.cat([v, v.mean(0, keepdim=True).expand(len(new), *v.shape[1:])])
        else:
            own[k] = v
    model.load_state_dict(own)
    for rec in recs:  # recording index -> its row in the enlarged tables
        rec.idx += n0
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("rec_"))
    rec_emb = [p for n, p in model.named_parameters() if n == "rec_emb.weight"]
    rec_p = [p for n, p in model.named_parameters() if n.startswith("rec_") and n != "rec_emb.weight"]
    opt = torch.optim.Adam([dict(params=rec_emb, lr=a.lr), dict(params=rec_p, lr=10 * a.lr)])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, 10 * a.lr], total_steps=a.steps, pct_start=0.05)
    mss = MultiScaleSpectralLoss()
    out = RES / "unseen"
    out.mkdir(exist_ok=True)
    res = dict(run=a.run, new_ids=new, adapt=[dict(step=0, **score(model, recs, splits, mss))])
    print(json.dumps(res["adapt"][-1]), flush=True)
    rng = np.random.default_rng(0)
    L = int(cfg.win * SR)
    model.train()
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        for _ in range(8):
            r = int(rng.integers(len(recs)))
            w0 = int(rng.choice(splits[r][0]))
            y_hat, _ = model(recs[r], w0)
            y = torch.tensor(recs[r].y[w0:w0 + L], device=DEV)
            ((mss(y_hat[None], y[None]) + model.regularizer()) / 8).backward()
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            res["adapt"].append(dict(step=step, **score(model, recs, splits, mss)))
            print(json.dumps(res["adapt"][-1]), flush=True)
            (out / f"{a.run}_n{a.n}{f'_{a.train_seconds:g}s' if a.train_seconds else ''}.json").write_text(json.dumps(res, indent=1))

    if a.ref:  # the same windows, scored by a run that trained on these recordings
        rc = json.loads((RES / "e4" / a.ref / "config.json").read_text())
        assert rc.get("extra_ids", [])[:len(new)] == new
        m = GuitarNet(NetConfig(sr=SR, structured=rc["variant"] == "structured", **rc["overrides"]),
                      n0 + len(rc["extra_ids"])).to(DEV)
        m.load_state_dict(upgrade_state_dict(torch.load(RES / "e4" / a.ref / "model.pt", map_location=DEV)))
        for rec in recs:  # same row layout: the 20, then the extras in extra_ids order
            rec.idx = n0 + rc["extra_ids"].index(new[rec.idx - n0])
        res["ref"] = dict(run=a.ref, **score(m, recs, splits, mss))
        print("ref", json.dumps(res["ref"]), flush=True)
        (out / f"{a.run}_n{a.n}{f'_{a.train_seconds:g}s' if a.train_seconds else ''}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
