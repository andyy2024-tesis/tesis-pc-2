"""Experiment 12: fine-tune BigVGAN-v2 (24 kHz, 100 mel bands) as a guitar vocoder and as a DDSP refiner.

Phase 2 (--input real): real mel -> real audio on GAPS training-split recordings, to make the pretrained vocoder a
guitar specialist. Phase 3 (V1, V2, S): DDSP render -> real audio on the recordings rendered by scripts/e12_render_ddsp.py:
  --input mix | strings   mel of the DDSP mix, or of its harmonic part only (the "simple DDSP")
  --stems                 also the mel of every DDSP part, through a zero-initialised input layer (starts as --input mix)
  --inject mix | strings  NSF style: the DDSP waveform enters every upsampling stage through zero-initialised strided
                          convolutions (as the source in NSF-HiFiGAN); at step 0 the model is the plain mel vocoder
Never trains on the last 15% of the 20 evaluation recordings (their held-out windows) nor on the GAPS test split.
Losses as BigVGAN-v2: multi-scale mel L1 x15 + multi-period and CQT discriminators with feature matching. Segments are
peak-normalised to 0.95 like BigVGAN (the same gain on input and target).
--rec-emb ddsp | table   which recording a segment comes from (the vocoder alone cannot know the guitar, microphone and
                          room): "ddsp" = the DDSP's own learned recording embedding (frozen, standardised), "table" = a
                          fresh embedding learned here (10x learning rate); a small MLP turns it into a per-channel scale
                          and shift after the input layer and after every upsampling stage, zero-initialised (step 0 = the
                          vocoder without it). The rows' recording ids are saved in config.json (rec_ids).
Phase 5 (--sr 44100): BigVGAN-v2 44 kHz (128 bands) instead, real audio from data/processed/sr44100 (scripts/
e12_prep44k.py); positions, notes and DDSP renders stay at 24 kHz and the DDSP input is upsampled per segment; the
output is brought back to 24 kHz for the evaluation, so every number compares.
Evaluation: the 85 held-out 3 s windows of the 20 recordings, multi-scale spectral loss of the output vs the real audio
(the measure of every DDSP model). Writes analysis/results/e12/<tag>/ (generator.pt, last.pt, log.txt, eval.json).
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from scipy.signal import resample_poly

from tesis.data.gaps import PROCESSED, REPO
from tesis.synth.dsp import MultiScaleSpectralLoss

sys.path.insert(0, str(REPO / "external" / "BigVGAN"))
from bigvgan import BigVGAN  # noqa: E402
from discriminators import MultiPeriodDiscriminator, MultiScaleSubbandCQTDiscriminator  # noqa: E402
from env import AttrDict  # noqa: E402
from loss import MultiScaleMelSpectrogramLoss, discriminator_loss, feature_loss, generator_loss  # noqa: E402
from meldataset import get_mel_spectrogram  # noqa: E402

SR = 24000  # positions, notes, DDSP renders and the evaluation measure
D = PROCESSED / f"sr{SR}"
RES = REPO / "analysis" / "results"
CKPTS = {24000: REPO / "external" / "ckpt" / "bigvgan_v2_24khz_100band_256x",
         44100: REPO / "external" / "ckpt" / "bigvgan_v2_44khz_128band_512x"}
CKPT = CKPTS[SR]
UP = (147, 80)  # 24 kHz -> 44.1 kHz
M = 80  # base-rate margin around an upsampled segment (= 147 samples at 44.1 kHz), so the filter edges are cut off
TEST_FROM = 0.85  # as scripts/e4_train.py
EVAL_IDS = pd.read_csv(RES / "e4_recordings.csv").id.tolist()


class Refiner(nn.Module):
    """BigVGAN generator with optional extra inputs, all starting at zero so step 0 is the loaded vocoder."""

    def __init__(self, h, n_stems=0, inject=False, ckpt=False, rec="none", rec_ids=None, rec_dim=32, rec_table=None):
        super().__init__()
        self.g = BigVGAN(h, use_cuda_kernel=False)
        self.rec_mode, self.rec_ids = rec, list(rec_ids or [])
        if rec != "none":
            n = len(self.rec_ids)
            if rec == "ddsp":  # frozen: a buffer, saved with the generator
                self.register_buffer("rec_table", torch.zeros(n, rec_dim) if rec_table is None else rec_table.float())
            else:
                assert rec == "table", rec
                self.rec_table = nn.Parameter(torch.randn(n, rec_dim) * 0.1)
            self.rec_mlp = nn.Sequential(nn.Linear(rec_dim, 128), nn.LeakyReLU(0.1))
            chans = [h.upsample_initial_channel] + [h.upsample_initial_channel // 2 ** (i + 1)
                                                    for i in range(len(h.upsample_rates))]
            self.film = nn.ModuleList(nn.Linear(128, 2 * c) for c in chans)
            for f in self.film:
                nn.init.zeros_(f.weight)
                nn.init.zeros_(f.bias)
        self.h = h
        self.ckpt = ckpt  # recompute each AMP block in the backward pass: BigVGAN's anti-aliased activations are huge
        self.stem_in = None
        if n_stems:
            self.stem_in = nn.Conv1d(h.num_mels * n_stems, h.upsample_initial_channel, 7, 1, padding=3)
            nn.init.zeros_(self.stem_in.weight)
            nn.init.zeros_(self.stem_in.bias)
        self.src = None
        if inject:  # NSF-HiFiGAN's source path: from the waveform rate down to each stage's rate
            rates = h.upsample_rates
            self.src = nn.ModuleList()
            for i in range(len(rates)):
                ch = h.upsample_initial_channel // (2 ** (i + 1))
                f = int(np.prod(rates[i + 1:]))
                conv = nn.Conv1d(1, ch, 2 * f, f, padding=f // 2) if f > 1 else nn.Conv1d(1, ch, 1)
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
                self.src.append(conv)

    def rec_of(self, rids, dev):
        """Embedding rows of these recordings (None without --rec-emb)."""
        if self.rec_mode == "none":
            return None
        return torch.tensor([self.rec_ids.index(r) for r in rids], device=dev)

    def _film(self, k, x, c):
        gam, bet = self.film[k](c)[..., None].chunk(2, 1)
        return x * (1 + gam) + bet

    def forward(self, mel, stems=None, src=None, rec=None):
        g = self.g
        c = self.rec_mlp(self.rec_table[rec]) if rec is not None else None
        x = g.conv_pre(mel)
        if c is not None:
            x = self._film(0, x, c)
        if self.stem_in is not None:
            # log-mels floor at log(1e-5) = -11.5: a silent part is a large constant over all its channels, and
            # fed raw it blew up the first S run once the zero-initialised layer moved (mel loss 1.6 -> 12.9 by step
            # 1600). Shifted and scaled, silence is 0 and loud parts about 1.
            x = x + self.stem_in((stems + 11.513) * 0.1)
        for i in range(g.num_upsamples):
            for up in g.ups[i]:
                x = up(x)
            if self.src is not None:
                s = self.src[i](src)
                x = x + s[..., :x.shape[-1]]
            if c is not None:
                x = self._film(i + 1, x, c)
            xs = None
            for j in range(g.num_kernels):
                blk = g.resblocks[i * g.num_kernels + j]
                r = checkpoint(blk, x, use_reentrant=False) if self.ckpt and self.training else blk(x)
                xs = r if xs is None else xs + r
            x = xs / g.num_kernels
        x = g.conv_post(g.activation_post(x))
        return torch.tanh(x) if g.use_tanh_at_final else torch.clamp(x, -1.0, 1.0)


class Data:
    """Random aligned segments: real audio and, for phase 3, the DDSP render (mix + parts) of the same samples."""

    def __init__(self, ids, renders=None, out_sr=SR):
        self.items = []
        self.out_sr = out_sr
        self.y_out = {}  # the real audio at the output rate when it is not 24 kHz
        for rid in ids:
            y = np.load(D / "audio" / f"{rid}.npy", mmap_mode="r")
            if out_sr != SR:
                self.y_out[rid] = np.load(PROCESSED / f"sr{out_sr}" / "audio" / f"{rid}.npy", mmap_mode="r")
            on = pd.read_csv(D / "notes" / f"{rid}.csv").onset.values
            lo, hi = max(int((on.min() - 0.5) * SR), 0), min(int((on.max() + 2.0) * SR), len(y))
            if rid in EVAL_IDS:
                hi = min(hi, int(TEST_FROM * len(y)))
            x = None
            if renders is not None:
                z = np.load(renders / f"{rid}.npz")
                rlo = int(z["lo"])
                x = (rlo, list(z["parts"]), z["x"])
                lo, hi = max(lo, rlo), min(hi, rlo + z["x"].shape[1])
            if hi - lo > SR * 4:
                self.items.append((rid, y, lo, hi, x))
        w = np.array([hi - lo for _, _, lo, hi, _ in self.items], np.float64)
        self.p = w / w.sum()
        self.last_rids = []  # recordings of the last batch
        print(f"data: {len(self.items)} recordings, {w.sum() / SR / 3600:.2f} h", flush=True)

    def batch(self, B, seg, rng, parts=()):
        """seg: samples at the output rate. At 44.1 kHz the segment starts at a 24 kHz sample s (real audio from
        round(s * 147 / 80), within half a 44.1 kHz sample) and the DDSP input is upsampled with a margin."""
        ys, xs = [], []
        self.last_rids = []
        up = self.out_sr != SR
        nb = int(np.ceil(seg * SR / self.out_sr)) + 1 if up else seg  # base-rate samples covering the segment
        for k in rng.choice(len(self.items), B, p=self.p):
            rid, y, lo, hi, x = self.items[k]
            self.last_rids.append(rid)
            if up:
                s = int(rng.integers(lo + M, hi - nb - M))
                s2 = int(round(s * UP[0] / UP[1]))
                real = np.asarray(self.y_out[rid][s2:s2 + seg], np.float32) / 32768
            else:
                s = int(rng.integers(lo, hi - seg))
                real = np.asarray(y[s:s + seg], np.float32) / 32768
            peak = np.abs(real).max()
            if x is not None:
                rlo, names, arr = x
                rows = [0] + [1 + names.index(p) for p in parts]
                if up:
                    xx = np.asarray(arr[rows, s - M - rlo:s - rlo + nb + M], np.float32)
                    xx = resample_poly(xx, *UP, axis=1)[:, M * UP[0] // UP[1]:][:, :seg].astype(np.float32)
                else:
                    xx = np.asarray(arr[rows, s - rlo:s - rlo + seg], np.float32)
                peak = max(peak, np.abs(xx[0]).max())
            g = 0.95 / max(peak, 1e-4)  # one gain for input and target, both peaks <= 0.95
            ys.append(real * g)
            if x is not None:
                xs.append(xx * g)
        y = torch.tensor(np.stack(ys))[:, None]
        return y, (torch.tensor(np.stack(xs)) if xs else None)


def mel(h, x):  # (B, T) -> (B, 100, T / 256)
    return get_mel_spectrogram(x, h)


def stems_of(names):
    """Groups of DDSP parts given to the vocoder as separate mels (--stems)."""
    groups = [["strings"], ["burst"], ["body"], ["contact"], ["noise", "floor"]]
    return [[p for p in g if p in names] for g in groups if any(p in names for p in g)]


class Inputs:
    """Builds the generator inputs from a batch (or a long excerpt) of real audio and DDSP parts."""

    def __init__(self, a, h, names):
        self.a, self.h = a, h
        self.groups = stems_of(names) if a.stems else []
        need = set()
        if a.input == "strings" or a.inject == "strings":
            need.add("strings")
        for g in self.groups:
            need.update(g)
        self.parts = sorted(need)

    def __call__(self, y, x):
        a, h = self.a, self.h
        if a.input == "real":
            return dict(mel=mel(h, y[:, 0]))
        rows = {p: 1 + i for i, p in enumerate(self.parts)}
        main = x[:, 0] if a.input == "mix" else x[:, rows["strings"]]
        out = dict(mel=mel(h, main))
        if self.groups:
            out["stems"] = torch.cat([mel(h, sum(x[:, rows[p]] for p in g)) for g in self.groups], 1)
        if a.inject != "none":
            out["src"] = (x[:, 0] if a.inject == "mix" else x[:, rows["strings"]])[:, None]
        return out


def ddsp_rec_table(renders_name):
    """(recording ids, standardised embedding rows) of the DDSP whose renders these are (its e6 or e4 run)."""
    run = RES / "e6" / renders_name
    if not run.exists():
        run = RES / "e4" / renders_name
    base = json.loads((run / "meta.json").read_text())["base"] if (run / "meta.json").exists() else renders_name
    ids = EVAL_IDS + json.loads((RES / "e4" / base / "config.json").read_text()).get("extra_ids", [])
    sd = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    sd = sd if isinstance(sd, dict) else sd.state_dict()
    E = sd.get("model", sd)["rec_emb.weight"].float()
    assert E.shape[0] == len(ids), (E.shape, len(ids))
    return ids, (E - E.mean(0)) / E.std()


def refiner_from_config(cfg, h, inputs):
    """The generator a saved run used (its weights still to be loaded)."""
    return Refiner(h, n_stems=len(inputs.groups), inject=cfg.get("inject", "none") != "none",
                   rec=cfg.get("rec_emb", "none"), rec_ids=cfg.get("rec_ids"), rec_dim=cfg.get("rec_dim", 32))


@torch.no_grad()
def vocode_long(G, inputs, y, x, dev, win=SR * 6, xf=SR // 2, rec=None):
    """Vocode a long excerpt in crossfaded windows (segment-level peak gain as in training, undone afterwards).
    rec: the recording's embedding row (G.rec_of), for a run with --rec-emb."""
    n = y.shape[-1]
    out, wsum = np.zeros(n + win, np.float32), np.zeros(n + win, np.float32)
    ramp = np.ones(win, np.float32)
    ramp[:xf], ramp[-xf:] = np.linspace(0, 1, xf), np.linspace(1, 0, xf)
    for a in range(0, n, win - xf):
        yy = y[..., a:a + win]
        xx = x[..., a:a + win] if x is not None else None
        m = yy.shape[-1]
        if m < 2048:
            break
        pad = (-m) % G.h.hop_size
        ref = (xx[:, 0] if xx is not None else yy[:, 0])
        g = 0.95 / max(float(ref.abs().max()), 1e-4)
        yy = torch.nn.functional.pad(yy, (0, pad)) * g
        xx = torch.nn.functional.pad(xx, (0, pad)) * g if xx is not None else None
        kw = inputs(yy.to(dev), xx.to(dev) if xx is not None else None)
        if rec is not None:
            kw["rec"] = rec
        o = G(**kw)[0, 0, :m].float().cpu().numpy() / g
        out[a:a + m] += o * ramp[:m]
        wsum[a:a + m] += ramp[:m]
    k = wsum > 1e-3
    out[k] /= wsum[k]
    return out[:n]


def eval_windows(renders, out_sr=SR):
    """The 85 held-out 3 s windows of the 20 recordings (starts as in e4_train: 5 per recording after 85%).
    (rid, real 24 kHz, DDSP (names, parts)); with out_sr != 24 kHz the DDSP parts are upsampled and a fourth item is
    the real window at out_sr (the input of a real-input run; the score is still against the 24 kHz window)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import e2_one_piece as e2
    wins = []
    L = 3 * SR
    for rid in EVAL_IDS:
        if renders is not None and not (renders / f"{rid}.npz").exists():
            continue
        y = np.load(D / "audio" / f"{rid}.npy").astype(np.float32) / 32768
        on = pd.read_csv(D / "notes" / f"{rid}.csv").onset.values
        st = e2.valid_starts(y, L, 240)
        st = st[(st >= int((on.min() - 0.5) * SR)) & (st + L <= int((on.max() + 2.0) * SR))]
        test = st[st >= int(TEST_FROM * len(y))]
        test = test[np.linspace(0, len(test) - 1, min(5, len(test))).astype(int)] if len(test) else test
        z = np.load(renders / f"{rid}.npz") if renders is not None else None
        y2 = np.load(PROCESSED / f"sr{out_sr}" / "audio" / f"{rid}.npy", mmap_mode="r") if out_sr != SR else None
        for w0 in test:
            x = None
            if z is not None:
                rlo, names = int(z["lo"]), list(z["parts"])
                x = (names, z["x"][:, w0 - rlo:w0 - rlo + L].astype(np.float32))
                if y2 is not None:
                    x = (names, resample_poly(x[1], *UP, axis=1).astype(np.float32))
            if y2 is None:
                wins.append((rid, y[w0:w0 + L], x))
            else:
                s2 = int(round(w0 * UP[0] / UP[1]))
                wins.append((rid, y[w0:w0 + L], x, np.asarray(y2[s2:s2 + 3 * out_sr], np.float32) / 32768))
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init", default="pretrained", help="'pretrained' or an e12 run folder name (its last.pt)")
    ap.add_argument("--input", choices=["real", "mix", "strings"], default="real")
    ap.add_argument("--stems", action="store_true")
    ap.add_argument("--inject", choices=["none", "mix", "strings"], default="none")
    ap.add_argument("--renders", default=None, help="analysis/results/e12/renders/<name> (phase 3)")
    ap.add_argument("--ids", default="train", help="'train' (GAPS training split) or 'rendered' (all rendered recordings)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seg", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--no-ckpt", action="store_true", help="keep activations (needs far more GPU memory)")
    ap.add_argument("--sr", type=int, choices=sorted(CKPTS), default=SR, help="output rate (phase 5: 44100)")
    ap.add_argument("--rec-emb", choices=["none", "ddsp", "table"], default="none")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = RES / "e12" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rec_table = None
    if a.rec_emb == "ddsp":
        assert a.renders, "--rec-emb ddsp needs the DDSP's renders"
        a.rec_ids, rec_table = ddsp_rec_table(a.renders)
        a.rec_dim = rec_table.shape[1]
    elif a.rec_emb == "table":
        assert a.renders, "--rec-emb table: the recordings are the rendered ones"
        a.rec_ids, a.rec_dim = sorted(p.stem for p in (RES / "e12" / "renders" / a.renders).glob("*.npz")), 32
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    ck = CKPTS[a.sr]
    h = AttrDict(json.loads((ck / "config.json").read_text()))
    assert h.sampling_rate == a.sr

    renders = RES / "e12" / "renders" / a.renders if a.renders else None
    if renders is not None:
        ids = sorted(p.stem for p in renders.glob("*.npz"))
        names = list(np.load(renders / f"{ids[0]}.npz")["parts"])
    else:
        man = pd.read_csv(D / "manifest.csv")
        ids = man[man.split == "train"].id.tolist()
        names = []
    inputs = Inputs(a, h, names)
    data = Data(ids, renders, a.sr)

    G = Refiner(h, n_stems=len(inputs.groups), inject=a.inject != "none", ckpt=not a.no_ckpt, rec=a.rec_emb,
                rec_ids=getattr(a, "rec_ids", None), rec_dim=getattr(a, "rec_dim", 32), rec_table=rec_table).to(dev)
    mpd, cqtd = MultiPeriodDiscriminator(h).to(dev), MultiScaleSubbandCQTDiscriminator(h).to(dev)
    table = [p for n, p in G.named_parameters() if n == "rec_table"]  # a fresh embedding learns 10x faster
    groups = [dict(params=[p for n, p in G.named_parameters() if n != "rec_table"], mult=1.0)]
    if table:
        groups.append(dict(params=table, mult=10.0))
    opt_g = torch.optim.AdamW(groups, a.lr, betas=(h.adam_b1, h.adam_b2))
    # same parameter order as BigVGAN's train.py (CQT discriminator first), so its saved Adam moments fit
    opt_d = torch.optim.AdamW(list(cqtd.parameters()) + list(mpd.parameters()), a.lr, betas=(h.adam_b1, h.adam_b2))
    new_params = inputs.groups or a.inject != "none" or a.rec_emb != "none"
    if a.init == "pretrained":
        G.g.load_state_dict(torch.load(ck / "bigvgan_generator.pt", map_location="cpu")["generator"])
        sd = torch.load(ck / "bigvgan_discriminator_optimizer.pt", map_location="cpu", weights_only=False)
        mpd.load_state_dict(sd["mpd"])
        cqtd.load_state_dict(sd["mrd"])
        opt_state = dict(g=sd["optim_g"], d=sd["optim_d"])
    else:
        sd = torch.load(RES / "e12" / a.init / "last.pt", map_location="cpu", weights_only=False)
        missing, unexpected = G.load_state_dict(sd["G"], strict=False)
        assert not unexpected, unexpected
        print("new (zero-initialised) parameters:", sorted({k.split(".")[0] for k in missing}), flush=True)
        if a.rec_emb == "ddsp":  # the loaded vocoder has no table: keep the DDSP's
            assert torch.equal(G.rec_table.cpu(), rec_table.float())
        mpd.load_state_dict(sd["mpd"])
        cqtd.load_state_dict(sd["cqtd"])
        opt_state = dict(g=sd.get("opt_g"), d=sd.get("opt_d"))
    # Adam moments of the loaded model (fresh moments make the first updates sign-like and knock a pretrained model)
    if opt_state["d"] is not None:
        opt_d.load_state_dict(opt_state["d"])
    if opt_state["g"] is not None and not new_params:
        opt_g.load_state_dict(opt_state["g"])
    for o in (opt_g, opt_d):
        for pg in o.param_groups:
            pg["lr"] = pg["initial_lr"] = a.lr * pg.get("mult", 1.0)
    del sd, opt_state
    warm = 1000  # linear warm-up of both learning rates
    sched = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1.0, (s + 1) / warm)) for o in (opt_g, opt_d)]
    mel_loss = MultiScaleMelSpectrogramLoss(sampling_rate=a.sr)
    mss = MultiScaleSpectralLoss()
    wins = eval_windows(renders, a.sr)

    def evaluate():
        G.eval()
        vals = []
        for w in wins:
            y, x = w[1], w[2]
            yt = torch.tensor(w[3] if len(w) > 3 else y)[None, None]
            xt = None
            if x is not None:
                nm, arr = x
                rows = [0] + [1 + nm.index(p) for p in inputs.parts]
                xt = torch.tensor(arr[rows])[None]
            o = vocode_long(G, inputs, yt, xt, dev, win=a.sr * 6, xf=a.sr // 2, rec=G.rec_of([w[0]], dev))
            if a.sr != SR:
                o = resample_poly(o, UP[1], UP[0]).astype(np.float32)[:len(y)]
            vals.append(mss(torch.tensor(o)[None], torch.tensor(y)[None]).item())
        G.train()
        return float(np.mean(vals))

    log = open(out / "log.txt", "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say(f"{a.tag}: step 0 eval {evaluate():.3f}")
    rng = np.random.default_rng(0)
    t0 = time.time()
    hist = {}
    for step in range(1, a.steps + 1):
        y, x = data.batch(a.batch, a.seg, rng, inputs.parts)
        y = y.to(dev)
        x = x.to(dev) if x is not None else None
        kw = inputs(y, x)
        if a.rec_emb != "none":
            kw["rec"] = G.rec_of(data.last_rids, dev)
        y_hat = G(**kw)
        # discriminators
        r, g, _, _ = mpd(y, y_hat.detach())
        loss_d = discriminator_loss(r, g)[0]
        r, g, _, _ = cqtd(y, y_hat.detach())
        loss_d = loss_d + discriminator_loss(r, g)[0]
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(cqtd.parameters()), h.clip_grad_norm)
        opt_d.step()
        # generator
        lm = mel_loss(y, y_hat) * h.lambda_melloss
        _, g_f, f_r, f_g = mpd(y, y_hat)
        _, g_s, s_r, s_g = cqtd(y, y_hat)
        lg = generator_loss(g_f)[0] + generator_loss(g_s)[0]
        lf = feature_loss(f_r, f_g) + feature_loss(s_r, s_g)
        loss_g = lm + lg + lf
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), h.clip_grad_norm)
        opt_g.step()
        for sc in sched:
            sc.step()
        for k, v in dict(mel=lm.item() / h.lambda_melloss, adv=lg.item(), fm=lf.item(), d=loss_d.item()).items():
            hist.setdefault(k, []).append(v)
        if step % 200 == 0:
            say(f"  step {step:6d}  " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in hist.items())
                + f"  ({time.time() - t0:.0f} s)")
            hist = {}
        if step % a.eval_every == 0 or step == a.steps:
            say(f"  EVAL step {step}: {evaluate():.3f}")
            torch.save(dict(G=G.state_dict(), mpd=mpd.state_dict(), cqtd=cqtd.state_dict(), opt_g=opt_g.state_dict(),
                            opt_d=opt_d.state_dict(), step=step), out / "last.pt")
    torch.save(G.state_dict(), out / "generator.pt")
    say("done")


if __name__ == "__main__":
    main()
