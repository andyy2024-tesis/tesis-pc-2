"""Experiment 1: fit synthesizer variants directly to GAPS isolated notes (no neural network).

Each note window starts PRE s before the onset and lasts DUR s. Every variant has the same catch-all filtered noise
(framewise, N_BANDS bands) and the same model of the previous chord still ringing (decaying partial stacks).
What differs is how the new note itself is modelled:

  ddsp        harmonic partials (k * f0), free amplitude per partial per 5 ms frame   (plain DDSP capacity)
  ddsp_inh    same, with stiff-string frequencies k * f0 * sqrt(1 + B k^2)
  phys_nob    stiff-string partials with exponential decay: 2 numbers per partial
  phys        phys_nob + a pluck transient (filtered noise burst, sample-accurate start, learned decay)
  phys_body   phys + body resonances: damped modes on a fixed 60-800 Hz grid below 0.8 f0, excited at the pluck
  hybrid      phys_body with a slow learned correction of each partial's decay (25 Hz frames, small L2 penalty)
  ddsp_full   ddsp_inh + pluck transient + body resonances (flexible amplitudes with every structured part)

The catch-all noise uses log-spaced bands (DDSP's linear bands are ~185 Hz wide here, too coarse for the low end).

Reported per note: spectral loss, energy taken by the noise (whole note and first 50 ms), and parameter count.
Usage: uv run python scripts/e1_single_notes.py [--steps 1500] [--variants ddsp,ddsp_inh,phys_nob,phys]
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from tesis.data.gaps import PROCESSED, REPO
from tesis.data.onsets import group_onsets
from tesis.synth.dsp import (MultiScaleSpectralLoss, damped_modes, filter_noise, filtered_noise, frames_to_samples,
                             log_band_matrix, partial_bank)
from tesis.synth.inharmonicity import fit_partials

SR = 24000
PRE, DUR = 0.02, 1.0
T = int((PRE + DUR) * SR)
HOP, N_FFT_NOISE, N_BANDS = 120, 480, 64
N_FRAMES = T // HOP + 1
K = 80  # partials for the new note
K_PREV = 40  # partials per previous note
MAX_PREV = 6
NOISE_L1 = 1e-3  # small push towards explaining energy with the deterministic parts
B_DEFAULT = 5e-5  # median measured on GAPS isolated notes
BODY_MODES = np.geomspace(60, 800, 48)  # ~5.5% spacing
BODY_DECAY = (2.0, 50.0)  # 1/s: body modes ring for tens of ms, never a sub-ms click
MAX_CENTS_OFF, MAX_B = 50, 2e-3  # reject partial fits that are clearly wrong
MIN_PARTIAL_SHARE = 0.3  # found partials must carry this share of the note's energy
CORR_RATE, CORR_L2 = 25, 1e-2  # hybrid: correction frames per second, penalty on the log-amplitude correction
N_CORR = int((PRE + DUR) * CORR_RATE) + 1
HAS_BURST = ("phys", "phys_body", "hybrid", "ddsp_full")
HAS_BODY = ("phys_body", "hybrid", "ddsp_full")
OUT = REPO / "analysis" / "results" / "e1"


def midi_hz(p, cents):
    return 440 * 2 ** ((p - 69 + cents / 100) / 12)


def stiff(f0, B, k):
    return k * f0 * np.sqrt(1 + B * k * k)


def build_items(d, max_notes=None):
    iso = pd.read_csv(d / "isolated_notes.csv")
    man = pd.read_csv(d / "manifest.csv").set_index("id")
    items = []
    for rid, g in iso.groupby("id"):
        y = np.load(d / "audio" / f"{rid}.npy").astype(np.float32) / 32768
        notes = pd.read_csv(d / "notes" / f"{rid}.csv")
        gid = group_onsets(notes.onset.values)
        tc = man.loc[rid, "tuning_cents"]
        for _, r in g.iterrows():
            a = int(round((r.onset - PRE) * SR))
            seg = y[a:a + T]
            if a < SR or len(seg) < T:
                continue
            fit = fit_partials(y[int((r.onset + 0.03) * SR):int((r.onset + 0.63) * SR)], SR, midi_hz(r.pitch, tc))
            if len(fit.k) < 3 or fit.B > MAX_B or abs(1200 * np.log2(fit.f0 / midi_hz(r.pitch, tc))) > MAX_CENTS_OFF:
                continue
            fit_seg = y[int((r.onset + 0.03) * SR):int((r.onset + 0.63) * SR)]
            if np.sum(fit.amp ** 2) / 2 < MIN_PARTIAL_SHARE * np.mean(fit_seg ** 2):
                continue  # the MIDI pitch doesn't explain the audio (mislabelled note)
            f0h = float(np.sum(fit.freq * fit.k) / np.sum(fit.k ** 2))  # best purely harmonic f0
            amp0 = np.full(K, 1e-4)
            amp0[fit.k[fit.k <= K] - 1] = fit.amp[fit.k <= K]
            # previous chord (the latest onset group before this note) is still ringing
            i_note = int(np.argmin(np.abs(notes.onset.values - r.onset)))
            prev_g = gid[i_note] - 1
            prev = notes[gid == prev_g].head(MAX_PREV) if prev_g >= 0 else notes.iloc[:0]
            pre_seg = y[max(a - int(0.8 * SR), 0):a]
            prev_f = np.zeros((MAX_PREV, K_PREV))
            prev_a = np.full((MAX_PREV, K_PREV), 1e-5)
            for j, (_, p) in enumerate(prev.iterrows()):
                pf = fit_partials(pre_seg, SR, midi_hz(p.pitch, tc)) if len(pre_seg) > 0.3 * SR else None
                f0p, Bp = (pf.f0, pf.B) if pf is not None and len(pf.k) >= 3 else (midi_hz(p.pitch, tc), B_DEFAULT)
                prev_f[j] = stiff(f0p, Bp, np.arange(1, K_PREV + 1))
                if pf is not None and len(pf.k):
                    m = pf.k <= K_PREV
                    prev_a[j, pf.k[m] - 1] = pf.amp[m]
            items.append(dict(id=rid, onset=r.onset, pitch=int(r.pitch), pre_minus_post_db=r.pre_minus_post_db,
                              y=seg, f0=fit.f0, B=fit.B, f0h=f0h, amp0=amp0, n_prev=len(prev),
                              prev_f=prev_f, prev_a=prev_a))
            if max_notes and len(items) >= max_notes:
                return items
    return items


class NoteModel(torch.nn.Module):
    def __init__(self, variant, batch, dev):
        super().__init__()
        self.variant = variant
        n = len(batch)
        k = torch.arange(1, K + 1, device=dev, dtype=torch.float64)
        f0 = torch.tensor([b["f0"] for b in batch], device=dev, dtype=torch.float64)
        B = torch.tensor([b["B"] for b in batch], device=dev, dtype=torch.float64)
        f0h = torch.tensor([b["f0h"] for b in batch], device=dev, dtype=torch.float64)
        if variant == "ddsp":
            freqs = f0h[:, None] * k
        else:
            freqs = f0[:, None] * k * torch.sqrt(1 + B[:, None] * k ** 2)
        self.register_buffer("freqs", freqs.float())
        self.register_buffer("prev_f", torch.tensor(np.stack([b["prev_f"] for b in batch]), device=dev).float().flatten(1))
        self.register_buffer("start", torch.full((n,), int(PRE * SR), device=dev))
        log_a0 = torch.log(torch.tensor(np.stack([b["amp0"] for b in batch]), device=dev).float() * 1.5)
        t_frames = torch.arange(N_FRAMES, device=dev) * HOP / SR - PRE
        if variant.startswith("ddsp"):  # free log-amplitude per partial per frame, initialised as a 3/s decay
            self.log_amp = torch.nn.Parameter(log_a0[:, :, None] - 3.0 * t_frames.clamp(min=0))
        else:
            self.log_amp0 = torch.nn.Parameter(log_a0)
            self.decay = torch.nn.Parameter(torch.full((n, K), math.log(math.expm1(3.0)), device=dev))
        if variant == "hybrid":
            self.corr = torch.nn.Parameter(torch.zeros((n, K, N_CORR), device=dev))
        self.prev_log_amp = torch.nn.Parameter(torch.log(torch.tensor(np.stack([b["prev_a"] for b in batch]), device=dev).float().flatten(1)))
        self.prev_decay = torch.nn.Parameter(torch.full_like(self.prev_log_amp, math.log(math.expm1(2.0))))
        self.noise_logmag = torch.nn.Parameter(torch.full((n, N_FRAMES, N_BANDS), -9.0, device=dev))
        self.register_buffer("W", log_band_matrix(N_BANDS, N_FFT_NOISE // 2 + 1, SR, device=dev))
        g = torch.Generator(device=dev).manual_seed(0)
        self.register_buffer("phase", torch.rand((n, N_FFT_NOISE // 2 + 1, N_FRAMES), generator=g, device=dev) * 2 * math.pi)
        if variant in HAS_BURST:
            self.register_buffer("burst_noise", torch.randn((n, T), generator=g, device=dev))
            self.burst_logmag = torch.nn.Parameter(torch.full((n, 33), -4.0, device=dev))
            self.burst_logtau = torch.nn.Parameter(torch.full((n,), math.log(0.01), device=dev))
        if variant in HAS_BODY:
            self.register_buffer("body_f", torch.tensor(BODY_MODES, device=dev).float())
            self.body_logamp = torch.nn.Parameter(torch.full((n, len(BODY_MODES)), math.log(1e-4), device=dev))
            self.register_buffer("body_phase", torch.rand((n, len(BODY_MODES)), generator=g, device=dev) * 2 * math.pi)
            self.body_decay = torch.nn.Parameter(torch.zeros((n, len(BODY_MODES)), device=dev))  # -> 26/s
            # single-note test: body modes only below the note's fundamental, so they can't take over the partials
            self.register_buffer("body_mask", (self.body_f[None] < 0.8 * f0.float()[:, None]).float())

    def n_params(self):
        """Parameters that describe the new note (excluding the shared noise and previous-note models)."""
        amps = K * N_FRAMES if self.variant.startswith("ddsp") else 2 * K + (K * N_CORR if self.variant == "hybrid" else 0)
        return amps + (33 + 1 if self.variant in HAS_BURST else 0) + (2 * len(BODY_MODES) if self.variant in HAS_BODY else 0)

    def penalty(self):
        return CORR_L2 * (self.corr ** 2).mean() if self.variant == "hybrid" else 0.0

    def forward(self):
        n = self.freqs.shape[0]
        t = (torch.arange(T, device=self.freqs.device) / SR)[None, None, :]
        if self.variant.startswith("ddsp"):
            amps = frames_to_samples(torch.exp(self.log_amp), T)
        else:
            tn = (t - PRE).clamp(min=0)
            log_amps = self.log_amp0[..., None] - torch.nn.functional.softplus(self.decay)[..., None] * tn
            if self.variant == "hybrid":
                log_amps = log_amps + frames_to_samples(self.corr, T)
            amps = torch.exp(log_amps)
        note = partial_bank(self.freqs, amps, SR, self.start)
        prev_amps = torch.exp(self.prev_log_amp[..., None] - torch.nn.functional.softplus(self.prev_decay)[..., None] * t)
        prev = partial_bank(self.prev_f, prev_amps, SR)
        parts = dict(note=note, prev=prev)
        if self.variant in HAS_BURST:
            tn = (t[:, 0] - PRE)
            env = torch.exp(-tn.clamp(min=0) / torch.exp(self.burst_logtau)[:, None]) * (tn >= 0)
            parts["burst"] = filter_noise(self.burst_noise, torch.exp(self.burst_logmag), SR) * env
        if self.variant in HAS_BODY:
            decay = BODY_DECAY[0] + (BODY_DECAY[1] - BODY_DECAY[0]) * torch.sigmoid(self.body_decay)
            parts["body"] = damped_modes(self.body_f, torch.exp(self.body_logamp) * self.body_mask, decay, T, SR,
                                         self.start, self.body_phase)
        parts["noise"] = filtered_noise(torch.exp(self.noise_logmag), T, N_FFT_NOISE, HOP, self.phase, self.W)
        return sum(parts.values()), parts


def energy(x, a=None, b=None):
    return (x[..., a:b] ** 2).sum(-1)


def energy_below(x, f_hi, a, b):
    """Energy of x[a:b] below f_hi (per item) via FFT."""
    X = torch.fft.rfft(x[..., a:b] * torch.hann_window(b - a, device=x.device))
    f = torch.fft.rfftfreq(b - a, 1 / SR).to(x.device)
    return ((X.abs() ** 2) * (f[None] < f_hi[:, None])).sum(-1)


def run_variant(variant, items, steps, dev, batch_size):
    loss_fn = MultiScaleSpectralLoss()
    rows = []
    for s in range(0, len(items), batch_size):
        batch = items[s:s + batch_size]
        y = torch.tensor(np.stack([b["y"] for b in batch]), device=dev)
        model = NoteModel(variant, batch, dev)
        opt = torch.optim.Adam(model.parameters(), lr=0.03)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            opt.zero_grad()
            out, parts = model()
            loss = loss_fn(out, y) + NOISE_L1 * torch.exp(model.noise_logmag).mean() * N_BANDS + model.penalty()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            out, parts = model()
            per = loss_fn(out, y, reduce=False)
            a0, a50 = int(PRE * SR), int((PRE + 0.05) * SR)
            e_all = sum(energy(p, a0) for p in parts.values())
            e_early = sum(energy(p, a0, a50) for p in parts.values())
            f_lo = 0.8 * torch.tensor([b["f0"] for b in batch], device=dev).float()
            e_low = sum(energy_below(p, f_lo, a0, T) for p in parts.values())
            n_low = energy_below(parts["noise"], f_lo, a0, T)
            tgt_low = energy_below(y, f_lo, a0, T) / energy(y, a0)
            for i, b in enumerate(batch):
                rows.append(dict(id=b["id"], onset=b["onset"], pitch=b["pitch"], pre_minus_post_db=b["pre_minus_post_db"],
                                 variant=variant, spectral_loss=per[i].item(),
                                 noise_frac=(energy(parts["noise"], a0)[i] / e_all[i]).item(),
                                 noise_frac_first50ms=(energy(parts["noise"], a0, a50)[i] / e_early[i]).item(),
                                 noise_frac_below_f0=(n_low[i] / e_low[i]).item(),
                                 target_frac_below_f0=tgt_low[i].item(),
                                 body_frac=(energy(parts["body"], a0)[i] / e_all[i]).item() if "body" in parts else 0.0,
                                 burst_frac_first50ms=(energy(parts["burst"], a0, a50)[i] / e_early[i]).item() if "burst" in parts else 0.0,
                                 note_params=model.n_params()))
        print(f"  {variant}: {min(s + batch_size, len(items))}/{len(items)} notes", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--variants", default="ddsp,ddsp_inh,phys_nob,phys,phys_body")
    ap.add_argument("--max-notes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="", help="suffix for the results file, so runs don't overwrite each other")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items = build_items(PROCESSED / f"sr{SR}", a.max_notes)
    print(f"{len(items)} notes prepared in {time.time() - t0:.0f} s", flush=True)
    rows = []
    for v in a.variants.split(","):
        t0 = time.time()
        rows += run_variant(v, items, a.steps, dev, a.batch)
        print(f"{v} done in {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"e1_results{a.tag}.csv", index=False)
    summary = df.groupby("variant", sort=False).agg(
        spectral_loss=("spectral_loss", "median"), noise_frac=("noise_frac", "median"),
        noise_frac_first50ms=("noise_frac_first50ms", "median"), noise_frac_below_f0=("noise_frac_below_f0", "median"),
        body_frac=("body_frac", "median"), burst_frac_first50ms=("burst_frac_first50ms", "median"),
        note_params=("note_params", "first"))
    print(summary.to_string(float_format=lambda v: f"{v:.4g}"))
    (OUT / f"e1_summary{a.tag}.json").write_text(json.dumps(summary.reset_index().to_dict("records"), indent=1))


if __name__ == "__main__":
    main()
