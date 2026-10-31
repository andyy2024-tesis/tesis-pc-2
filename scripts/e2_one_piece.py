"""Experiment 2: fit the whole-recording synthesizer to one polyphonic GAPS piece, then test note-level recovery.

1. Real fit: optimize PieceModel on random windows of the recording.
2. Render the full piece and excerpts (original, resynthesis, and each component) to WAV for listening.
3. Synthetic recovery: render a version of the piece whose per-note gains are known (fitted gains + random
   +/-4 dB changes), refit a fresh model on it, and compare recovered note levels with the truth. A naive
   estimate read straight from the mix spectrogram is the baseline.

Usage: uv run python scripts/e2_one_piece.py [--id 088_Yf1wc] [--steps 3000]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from tesis.data.gaps import PROCESSED, REPO
from tesis.synth.dsp import MultiScaleSpectralLoss
from tesis.synth.piece import PieceConfig, PieceModel, note_table

SR = 24000
D = PROCESSED / f"sr{SR}"
OUT = REPO / "analysis" / "results" / "e2"
WINDOWS_PER_STEP = 4
EXCERPT = 30.0  # s rendered per component for listening


def valid_starts(y_np, L, hop):
    """Window starts (multiples of hop) whose window has no digitally silent 10 ms frame.

    GAPS files contain stretches of exact zeros; the log-magnitude loss explodes there and drags the noise floor down.
    """
    fr = 240
    silent = np.abs(y_np[: len(y_np) // fr * fr].reshape(-1, fr)).max(1) < 2 / 32768
    starts = np.arange(0, len(y_np) - L + 1, hop)
    cs = np.concatenate([[0], np.cumsum(silent)])
    a, b = starts // fr, np.minimum((starts + L) // fr + 1, len(silent))
    return starts[(cs[b] - cs[a]) == 0]


def fit(model, y, starts, steps, lr=0.02, log_every=500, tag="", seed=0):
    loss_fn = MultiScaleSpectralLoss()
    # each catch-all noise frame is only seen when its window is sampled, so it gets a larger step size
    noise = [model.noise_logmag]
    rest = [p for n, p in model.named_parameters() if n != "noise_logmag"]
    opt = torch.optim.Adam([dict(params=rest, lr=lr), dict(params=noise, lr=5 * lr)])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    L = model.window_len()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for s in range(steps):
        opt.zero_grad()
        tot = 0.0
        for _ in range(WINDOWS_PER_STEP):
            w0 = int(rng.choice(starts))
            out, _ = model(w0)
            loss = loss_fn(out[None], y[None, w0:w0 + L]) + model.regularizer(w0)
            (loss / WINDOWS_PER_STEP).backward()
            tot += loss.item() / WINDOWS_PER_STEP
        opt.step()
        sched.step()
        if s % log_every == 0 or s == steps - 1:
            print(f"  {tag} step {s:5d}  loss {tot:.3f}  ({time.time() - t0:.0f} s)", flush=True)


@torch.no_grad()
def render(model, n_samples, seed=0):
    """Render the whole piece window by window; returns dict of component signals (numpy)."""
    L = model.window_len()
    acc = {}
    for i, w0 in enumerate(range(0, n_samples - L + 1, L)):
        _, parts = model(w0, noise_seed=seed + i)
        for k, v in parts.items():
            acc.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


@torch.no_grad()
def evaluate(model, y, parts, starts):
    """Spectral loss and energy shares over the rendered windows that contain no digital silence."""
    loss_fn = MultiScaleSpectralLoss()
    L = model.window_len()
    total = sum(parts.values())
    ok = [w for w in range(0, len(total) - L + 1, L) if w in set(starts.tolist())]
    losses = [loss_fn(torch.tensor(total[w:w + L])[None], y[None, w:w + L].cpu()).item() for w in ok]
    sel = np.concatenate([np.arange(w, w + L) for w in ok])
    e = {k: float((v[sel] ** 2).sum()) for k, v in parts.items()}
    et = sum(e.values())
    return dict(spectral_loss=float(np.mean(losses)), n_windows=len(ok), **{f"share_{k}_db": 10 * np.log10(v / et + 1e-12) for k, v in e.items()})


def naive_levels(y, table, sr):
    """Baseline note level read from the mix: harmonic-bin energy 20-80 ms after onset, in dB."""
    n_fft = 2048
    win = np.hanning(n_fft)
    out = []
    for s, f in zip(table["onset"], table["freqs"]):
        a = s + int(0.02 * sr)
        seg = y[a:a + n_fft]
        if len(seg) < n_fft:
            out.append(np.nan)
            continue
        X = np.abs(np.fft.rfft(seg * win)) ** 2
        b = np.round(f[f < sr / 2] / sr * n_fft).astype(int)
        out.append(10 * np.log10(X[b].sum() + 1e-12))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="088_Yf1wc")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--perturb-db", type=float, default=4.0)
    ap.add_argument("--config", default="{}", help='PieceConfig overrides as JSON, e.g. {"attack_ms": 2}')
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    ap.add_argument("--seed", type=int, default=0, help="window sampling and noise seed")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the synthetic recovery test")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    out = OUT / f"{a.id}{a.tag}"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    cfg = PieceConfig(sr=SR, **json.loads(a.config))

    man = pd.read_csv(D / "manifest.csv").set_index("id")
    notes = pd.read_csv(D / "notes" / f"{a.id}.csv")
    y_np = np.load(D / "audio" / f"{a.id}.npy").astype(np.float32) / 32768
    L = int(cfg.win * SR)
    y_np = y_np[: (len(y_np) // L) * L]
    notes = notes[notes.onset * SR < len(y_np)].reset_index(drop=True)
    y = torch.tensor(y_np, device=dev)
    table = note_table(notes, cfg, man.loc[a.id, "tuning_cents"])
    starts = valid_starts(y_np, L, cfg.hop)
    print(f"{a.id}: {len(notes)} notes, {len(y_np) / SR:.0f} s, {len(starts)} valid window starts", flush=True)

    # 1. real fit
    model = PieceModel(table, len(y_np), cfg, dev)
    fit(model, y, starts, a.steps, tag="real", seed=a.seed)
    parts = render(model, len(y_np))
    metrics = {"real": evaluate(model, y, parts, starts)}
    print("real fit:", json.dumps(metrics["real"], indent=None), flush=True)

    # 2. audio for listening (excerpt + full resynthesis)
    ex = slice(int(starts[0]), int(starts[0]) + int(EXCERPT * SR))
    full = sum(parts.values())
    peak = max(np.abs(y_np).max(), np.abs(full).max()) * 1.05
    sf.write(out / "audio" / "original.wav", y_np[ex] / peak, SR)
    sf.write(out / "audio" / "resynth_full.wav", full[ex] / peak, SR)
    sf.write(out / "audio" / "resynth_without_noise.wav", (full - parts["noise"] - parts["floor"])[ex] / peak, SR)
    for k, v in parts.items():
        sf.write(out / "audio" / f"part_{k}.wav", v[ex] / (np.abs(v[ex]).max() * 1.05 + 1e-9), SR)
    sf.write(out / "audio" / "resynth_full_piece.wav", full / peak, SR)
    torch.save(model.state_dict(), out / "model_real.pt")
    if a.no_synthetic:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        return

    # 3. synthetic recovery test
    with torch.no_grad():
        rng = np.random.default_rng(1)
        delta = rng.normal(0, a.perturb_db / 20 * np.log(10), len(notes))  # dB -> natural-log amplitude
        model.gain += torch.tensor(delta, device=dev, dtype=torch.float32)
        true_levels = model.note_levels_db().cpu().numpy()
        syn = sum(render(model, len(y_np), seed=1000).values()).astype(np.float32)
    y_syn = torch.tensor(syn, device=dev)
    fresh = PieceModel(table, len(y_np), cfg, dev)
    fit(fresh, y_syn, starts, a.steps, tag="synthetic")
    with torch.no_grad():
        est_levels = fresh.note_levels_db().cpu().numpy()
    naive = naive_levels(syn, table, SR)

    def score(est):
        m = np.isfinite(est)
        d = est[m] - true_levels[m]
        d = d - np.median(d)  # only relative levels matter (overall gain is arbitrary)
        return dict(corr=float(np.corrcoef(est[m], true_levels[m])[0, 1]), median_abs_err_db=float(np.median(np.abs(d))),
                    p90_abs_err_db=float(np.percentile(np.abs(d), 90)))

    # notes that overlap others at their onset (in a chord or with ringing notes) vs alone
    midi_end = table["onset"] + table["midi_dur"]
    n_active = np.array([((table["onset"] <= s) & (midi_end > s)).sum() for s in table["onset"]])
    metrics["recovery_model"] = score(est_levels)
    metrics["recovery_naive"] = score(naive)
    for name, m in [("alone", n_active == 1), ("2-3 sounding", (n_active >= 2) & (n_active <= 3)), ("4+ sounding", n_active >= 4)]:
        if m.sum() > 5:
            d_model = est_levels[m] - true_levels[m]
            d_naive = naive[m] - true_levels[m]
            metrics[f"recovery_model_{name}"] = dict(n=int(m.sum()), median_abs_err_db=float(np.median(np.abs(d_model - np.median(est_levels - true_levels)))))
            metrics[f"recovery_naive_{name}"] = dict(n=int(m.sum()), median_abs_err_db=float(np.nanmedian(np.abs(d_naive - np.nanmedian(naive - true_levels)))))
    pd.DataFrame(dict(onset=notes.onset, pitch=notes.pitch, voice=notes.voice, n_sounding=n_active, true_db=true_levels,
                      model_db=est_levels, naive_db=naive, fitted_real_gain=model.gain.detach().cpu().numpy() - delta)
                 ).to_csv(out / "note_levels.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
