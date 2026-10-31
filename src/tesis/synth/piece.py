"""Whole-recording guitar synthesizer fitted directly to one recording (no neural network).

Per recording (shared by all notes): a log-amplitude template and decay rate per (pitch, partial), stiff-string
inharmonicity B(pitch), release damping, pluck-transient spectrum, body modes, a stationary noise floor and a
framewise catch-all noise. Per note: only a loudness (log gain) and a brightness (spectral tilt).

So every note is described by two numbers; whether those can be recovered from the polyphonic mix is the
question of experiment 2.

Rendering works on windows of WIN seconds; notes ring from their onset until their voice is reused (or MAX_RING).
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

from tesis.synth.dsp import damped_modes, filter_noise, filtered_noise, log_band_matrix

LOG_B = (-11.3465, 0.0304)  # log B = a + b * pitch, fitted on GAPS isolated notes (experiment 1)


@dataclass
class PieceConfig:
    sr: int = 24000
    win: float = 3.0
    k: int = 60  # partials per note
    max_ring: float = 4.0
    fade: float = 0.005  # s, fade-out when a voice is reused
    hop: int = 120  # catch-all noise frame hop (5 ms)
    n_fft_noise: int = 480
    n_bands: int = 64
    burst_bands: int = 33
    body_modes: int = 48
    body_range: tuple = (60.0, 800.0)
    body_decay: tuple = (2.0, 50.0)  # 1/s
    body_len: float = 1.0  # s
    attack_ms: float = 0.0  # >0: strings fade in (raised cosine) instead of starting with a sawtooth-like step
    pluck_per_note: bool = False  # True: each note has its own transient level (slurs can have none)
    seed: int = 0  # random phases of the body modes
    source: str = "free"  # "free": template per (pitch, partial); "smooth": quadratic in log k per pitch (3 numbers)
    body: str = "modes"  # "modes": free response excited at each onset; "filter": shared frequency response on partials
    body_bins: int = 96  # log-spaced points of the body frequency response (50 Hz - Nyquist)


def note_table(notes: pd.DataFrame, cfg: PieceConfig, tuning_cents: float, log_b=LOG_B):
    """Per-note sample positions, partial frequencies and ring end (voice reuse or max_ring).

    log_b: (a, b) of log B = a + b * pitch (default: nylon strings, measured on GAPS).
    """
    sr = cfg.sr
    onset = np.round(notes.onset.values * sr).astype(np.int64)
    end = onset + int(cfg.max_ring * sr)
    for v in np.unique(notes.voice.values):
        idx = np.where(notes.voice.values == v)[0]
        end[idx[:-1]] = np.minimum(end[idx[:-1]], onset[idx[1:]])
    pitch = notes.pitch.values.astype(np.int64)
    f0 = 440 * 2 ** ((pitch - 69 + tuning_cents / 100) / 12)
    B = np.exp(log_b[0] + log_b[1] * pitch)
    k = np.arange(1, cfg.k + 1)
    freqs = f0[:, None] * k * np.sqrt(1 + B[:, None] * k ** 2)
    dur = np.maximum(np.round(notes.offset.values * sr).astype(np.int64) - onset, 1)
    return dict(onset=onset, end=np.maximum(end, onset + 1), pitch=pitch, freqs=freqs, midi_dur=dur)


class PieceModel(torch.nn.Module):
    def __init__(self, table: dict, n_samples: int, cfg: PieceConfig, dev: str):
        super().__init__()
        self.cfg, self.dev = cfg, dev
        self.n_samples = n_samples
        self.onset_np, self.end_np = table["onset"], table["end"]
        n = len(table["onset"])
        self.p_lo = int(table["pitch"].min())
        P = int(table["pitch"].max()) - self.p_lo + 1
        K = cfg.k
        T = lambda x, dt=torch.float32: torch.tensor(x, device=dev, dtype=dt)  # noqa: E731
        self.register_buffer("onset", T(table["onset"], torch.int64))
        self.register_buffer("end", T(table["end"], torch.int64))
        self.register_buffer("pidx", T(table["pitch"] - self.p_lo, torch.int64))
        self.register_buffer("freqs64", T(table["freqs"], torch.float64))
        self.register_buffer("midi_dur", T(table["midi_dur"], torch.int64))
        self.register_buffer("logk", torch.log(torch.arange(1, K + 1, device=dev, dtype=torch.float32)))

        # per note
        self.gain = torch.nn.Parameter(torch.zeros(n, device=dev))
        self.bright = torch.nn.Parameter(torch.zeros(n, device=dev))
        # per (pitch, partial): template and decay. init: -6 dB/octave, decay 1.5/s + 2/s per kHz
        if cfg.source == "smooth":
            self.src = torch.nn.Parameter(torch.tensor([[math.log(0.03), -1.0, 0.0]], device=dev).repeat(P, 1))
        else:
            self.tmpl = torch.nn.Parameter((math.log(0.03) - self.logk)[None].repeat(P, 1).clone())
        if cfg.body == "filter":
            # log-gain of the body at log-spaced frequencies, read at each partial's absolute frequency
            centers = np.geomspace(50.0, cfg.sr / 2, cfg.body_bins)
            pos = np.interp(np.log(np.clip(table["freqs"], 50.0, cfg.sr / 2)), np.log(centers), np.arange(cfg.body_bins))
            lo = np.minimum(np.floor(pos).astype(np.int64), cfg.body_bins - 2)
            self.register_buffer("body_centers", T(centers))
            self.register_buffer("body_lo", T(lo, torch.int64))
            self.register_buffer("body_frac", T(pos - lo))
            self.body_resp = torch.nn.Parameter(torch.zeros(cfg.body_bins, device=dev))
        f_pk = 440 * 2 ** ((torch.arange(P, device=dev) + self.p_lo - 69) / 12)[:, None] * torch.arange(1, K + 1, device=dev)
        sigma0 = 1.5 + 2.0 * f_pk / 1000
        self.decay_raw = torch.nn.Parameter(sigma0 + torch.log(-torch.expm1(-sigma0)))  # stable inverse softplus
        self.release_raw = torch.nn.Parameter(torch.tensor(math.log(math.expm1(1.0)), device=dev))
        # pluck transient
        self.burst_logmag = torch.nn.Parameter(torch.full((cfg.burst_bands,), -3.0, device=dev))
        self.burst_logtau = torch.nn.Parameter(torch.tensor(math.log(0.005), device=dev))
        if cfg.pluck_per_note:
            self.pluck = torch.nn.Parameter(torch.zeros(n, device=dev))
        # body modes, excited by each pluck in proportion to the note's gain
        self.register_buffer("body_f", T(np.geomspace(*cfg.body_range, cfg.body_modes)))
        g = torch.Generator(device=dev).manual_seed(cfg.seed)
        self.register_buffer("body_phase", torch.rand(cfg.body_modes, generator=g, device=dev) * 2 * math.pi)
        self.body_logamp = torch.nn.Parameter(torch.full((cfg.body_modes,), math.log(1e-3), device=dev))
        self.body_decay_raw = torch.nn.Parameter(torch.zeros(cfg.body_modes, device=dev))
        # noise: stationary floor + framewise catch-all
        self.floor_logmag = torch.nn.Parameter(torch.full((cfg.n_bands,), -6.0, device=dev))
        n_frames = n_samples // cfg.hop + 2
        self.noise_logmag = torch.nn.Parameter(torch.full((n_frames, cfg.n_bands), -9.0, device=dev))
        self.register_buffer("W", log_band_matrix(cfg.n_bands, cfg.n_fft_noise // 2 + 1, cfg.sr, device=dev))

    def window_len(self):
        return int(self.cfg.win * self.cfg.sr)

    def active(self, w0: int, L: int):
        return np.where((self.onset_np < w0 + L) & (self.end_np > w0))[0]

    def template(self):
        """(P, K) log-amplitude of each partial for each pitch, before per-note gain/brightness."""
        if self.cfg.source == "smooth":
            lk = self.logk[None]
            return self.src[:, :1] + self.src[:, 1:2] * lk + self.src[:, 2:3] * lk ** 2 / 4
        return self.tmpl

    def body_gain(self, idx=None):
        """(Na, K) log body gain at each partial frequency (0 without a body filter)."""
        if self.cfg.body != "filter":
            return 0.0
        lo = self.body_lo if idx is None else self.body_lo[idx]
        fr = self.body_frac if idx is None else self.body_frac[idx]
        return self.body_resp[lo] * (1 - fr) + self.body_resp[lo + 1] * fr

    def note_levels_window_db(self, t0: float, t1: float):
        """Mean partial power of each note between t0 and t1 s after its onset (its own decay included), in dB.

        Matches level estimators that read a spectrum some ms after the onset.
        """
        la = self.gain[:, None] + self.template()[self.pidx] + self.bright[:, None] * self.logk + self.body_gain()
        sigma = Fn.softplus(self.decay_raw)[self.pidx]
        mean_decay = (torch.exp(-2 * sigma * t0) - torch.exp(-2 * sigma * t1)) / (2 * sigma * (t1 - t0))
        p = torch.exp(2 * la) * mean_decay * (self.freqs64 < self.cfg.sr / 2).float()
        return 10 * torch.log10(p.sum(1) + 1e-20)

    def note_levels_db(self):
        """Initial level of each note (sum of partial powers at its onset), in dB."""
        la = self.gain[:, None] + self.template()[self.pidx] + self.bright[:, None] * self.logk + self.body_gain()
        la = la + torch.log((self.freqs64 < self.cfg.sr / 2).float() + 1e-12)
        return (10 / math.log(10)) * torch.logsumexp(2 * la, dim=1)

    def forward(self, w0: int, noise_seed: int = None):
        cfg, sr = self.cfg, self.cfg.sr
        L = self.window_len()
        dev = self.dev
        gen = torch.Generator(device=dev)
        gen.manual_seed(noise_seed if noise_seed is not None else int(torch.randint(0, 2 ** 31, (1,))))
        idx = torch.as_tensor(self.active(w0, L), device=dev)
        n_abs = w0 + torch.arange(L, device=dev)
        parts = {}

        # strings
        if len(idx):
            t_rel = n_abs[None, :] - self.onset[idx, None]  # (Na, L) samples since onset
            on = (t_rel >= 0) & (n_abs[None, :] < self.end[idx, None])
            to_end = (self.end[idx, None] - n_abs[None, :]).float() / (cfg.fade * sr)
            gate = on.float() * to_end.clamp(max=1.0)
            if cfg.attack_ms > 0:
                ramp = (t_rel.float() / (cfg.attack_ms * 1e-3 * sr)).clamp(0, 1)
                gate = gate * 0.5 * (1 - torch.cos(math.pi * ramp))
            ts = t_rel.clamp(min=0).float() / sr
            sigma = Fn.softplus(self.decay_raw)[self.pidx[idx]]  # (Na, K)
            rel = Fn.softplus(self.release_raw) * (t_rel - self.midi_dur[idx, None]).clamp(min=0).float() / sr
            log_a = (self.gain[idx, None] + self.template()[self.pidx[idx]] + self.bright[idx, None] * self.logk
                     + self.body_gain(idx))[..., None] \
                - sigma[..., None] * ts[:, None, :] - rel[:, None, :]
            f = self.freqs64[idx]  # (Na, K)
            alive = (f < sr / 2).float()[..., None]
            cyc = torch.remainder(f[..., None] * t_rel.clamp(min=0).double()[:, None, :] / sr, 1.0).float()
            parts["strings"] = (torch.exp(log_a) * alive * torch.sin(2 * math.pi * cyc) * gate[:, None, :]).sum((0, 1))

            # pluck transient: shared spectrum, per-note level, exponential decay
            level = self.gain[idx] + (self.pluck[idx] if cfg.pluck_per_note else 0.0)
            env = (torch.exp(level[:, None] - ts / torch.exp(self.burst_logtau)) * (t_rel >= 0)).sum(0)
            wn = torch.randn(L, generator=gen, device=dev)
            parts["burst"] = filter_noise(wn[None], torch.exp(self.burst_logmag)[None], sr)[0] * env
        else:
            parts["strings"] = parts["burst"] = torch.zeros(L, device=dev)

        # body: impulse at each onset (scaled by the note gain) convolved with the modal impulse response
        if cfg.body == "filter":
            parts["floor"] = filter_noise(torch.randn(1, L, generator=gen, device=dev), torch.exp(self.floor_logmag)[None], sr)[0]
            return self._finish(parts, w0, L, gen)
        BL = int(cfg.body_len * sr)
        hist = torch.as_tensor(np.where((self.onset_np >= w0 - BL) & (self.onset_np < w0 + L))[0], device=dev)
        imp = torch.zeros(BL + L, device=dev)
        if len(hist):
            imp = imp.index_add(0, self.onset[hist] - (w0 - BL), torch.exp(self.gain[hist]))
        decay = cfg.body_decay[0] + (cfg.body_decay[1] - cfg.body_decay[0]) * torch.sigmoid(self.body_decay_raw)
        ir = damped_modes(self.body_f, torch.exp(self.body_logamp)[None], decay[None], BL, sr,
                          torch.zeros(1, device=dev, dtype=torch.long), self.body_phase[None])[0]
        nfft = 2 ** math.ceil(math.log2(2 * (BL + L)))
        body = torch.fft.irfft(torch.fft.rfft(imp, nfft) * torch.fft.rfft(ir, nfft), nfft)[BL:BL + L]
        parts["body"] = body
        parts["floor"] = filter_noise(torch.randn(1, L, generator=gen, device=dev), torch.exp(self.floor_logmag)[None], sr)[0]
        return self._finish(parts, w0, L, gen)

    def _finish(self, parts, w0, L, gen):
        """Add the framewise catch-all noise and sum."""
        cfg, dev = self.cfg, self.dev
        f0 = w0 // cfg.hop
        F = L // cfg.hop + 1
        ph = torch.rand((1, cfg.n_fft_noise // 2 + 1, F), generator=gen, device=dev) * 2 * math.pi
        parts["noise"] = filtered_noise(torch.exp(self.noise_logmag[f0:f0 + F])[None], L, cfg.n_fft_noise, cfg.hop, ph,
                                        self.W)[0]
        return sum(parts.values()), parts

    def regularizer(self, w0: int):
        """Noise sparsity (push energy to the structured parts) + smooth templates across pitch."""
        F = self.window_len() // self.cfg.hop + 1
        f0 = w0 // self.cfg.hop
        noise = torch.exp(self.noise_logmag[f0:f0 + F]).mean() * self.cfg.n_bands
        tmpl = self.template()
        smooth = (tmpl[1:] - tmpl[:-1]).pow(2).mean() + (self.decay_raw[1:] - self.decay_raw[:-1]).pow(2).mean()
        reg = 1e-3 * noise + 1e-3 * smooth + 1e-3 * self.gain.pow(2).mean()
        if self.cfg.body == "filter":
            reg = reg + 1e-3 * (self.body_resp[2:] - 2 * self.body_resp[1:-1] + self.body_resp[:-2]).pow(2).mean()
        return reg
