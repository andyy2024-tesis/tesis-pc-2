"""Note-conditioned polyphonic guitar synthesizer with a neural controller (experiment 4).

Notes are assigned to N_VOICES monophonic voices. A GRU runs per voice over 10 ms frames (shared weights, with
the mean of all voices' features and a per-recording embedding as context) and drives the synthesis.

structured=True (ours):
  partials at stiff-string frequencies; log amplitude of partial k of note n at time t after its onset =
      template_k(note) + body(f_k) - sigma_k(pitch) * t + delta_k(voice, t)
  i.e. a per-note template (from a note MLP: pitch, measured loudness, onset clarity, recording) with a physical
  decay line, plus a GRU correction; smooth 2 ms string attack; per-note pluck transient; smooth body response and
  stationary noise floor per recording; catch-all noise in log-spaced bands. Optional (body_ir): a short low-end
  body response shared by all recordings (per-recording 6-band EQ), excited by every pluck at a per-note level.
Optional phenomena (experiment 8, each a NetConfig switch):
  two_decay: each partial is two polarizations of the string, with their own decay rates and a small detune
    (double decay + slow beating); sympathetic: the six open strings ring along when played partials hit theirs;
  symp_pluck: every pluck also kicks all open strings (the bridge transmits the transient);
  pluck_click: two learned short waveforms (nail click, finger thud) around every onset (from 20 ms before), at
    per-note levels;
  squeak: filtered noise in the 60 ms before each note on a wound string (pitch below G3), at a per-note level
    from the interval to the previous wound-string note and the time since it;
  body_eq_max / body_ir_cutoff: how far up the shared body response reaches;
  note_timbre: the note MLP also reads the note's attack timbre from the audio (first timbre_k partials relative to
    their mean, see tesis.data.loudness.note_timbre_db): pluck position and nail vs flesh vary note by note.
  timbre_late: with note_timbre, also the same reading 150-235 ms after the onset (how the note's spectrum changed:
    per-pluck differences in partial decay).
noise=False removes the catch-all noise (ablation); noise_gate: the catch-all noise only sounds around the plucks (from
  gate_pre before each onset, then decaying with gate_tau), so it can't play a background between them: the steady
  room / mic background is left to the stationary floor. reverb: per-recording room response (8 bands, level and RT60
each, no direct sound) applied to the instrument; dry parts are rendered from reverb_len before the window.
structured=False (plain DDSP baseline):
  harmonic partials; the GRU outputs each voice's overall amplitude and harmonic distribution every frame;
  catch-all noise in 65 linear bands; nothing else.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from tesis.data.loudness import note_timbre_db
from tesis.synth.dsp import filter_noise, filtered_noise, frames_to_samples, log_band_matrix

N_FEAT = 7
OPEN_STRINGS = np.array([40, 45, 50, 55, 59, 64])  # E2 A2 D3 G3 B3 E4


@dataclass
class NetConfig:
    sr: int = 24000
    win: float = 3.0
    burn: float = 1.0  # s of context before the window (GRU warm-up, outputs discarded)
    hop: int = 240  # network frame hop (10 ms)
    k: int = 60
    n_voices: int = 6
    max_ring: float = 4.0
    fade: float = 0.005
    hidden: int = 256
    rec_dim: int = 16
    structured: bool = True
    attack_ms: float = 2.0
    body_bins: int = 24  # ~1/3 octave between 50 Hz and Nyquist: smooth by construction
    burst_bands: int = 33
    noise_hop: int = 120
    n_fft_noise: int = 480
    noise: bool = True  # False: no catch-all noise (ablation)
    noise_gate: bool = False  # catch-all noise only around the plucks (see gate_*)
    gate_pre: float = 0.02  # s before the onset (fades in)
    gate_hold: float = 0.01  # s after the onset at full level
    gate_tau: float = 0.025  # s, then decays exponentially
    body_ir: bool = False  # structured: body response excited by each pluck (the low "thump")
    body_ir_len: float = 0.3  # s
    body_ir_tau: float = 0.08  # s, fixed decay envelope: the thump can't take over the sustain
    body_ir_cutoff: float = 600.0  # Hz, low end only
    body_eq_bands: int = 6  # per-recording EQ of the shared response, log-spaced 40-800 Hz
    delta_tv: float = 0.0  # weight of a penalty on frame-to-frame changes of the GRU amplitude correction (anti-flapping)
    audio_feats: bool = False  # per-voice loudness envelope + pitch deviation read from the audio (resynthesis)
    reverb: bool = False  # per-recording room reverb applied to the instrument (strings, transient, body)
    reverb_len: float = 1.0  # s (must be <= burn)
    reverb_bands: int = 8
    body_eq_max: float = 800.0  # Hz, top of the per-recording body EQ bands
    two_decay: bool = False  # structured: two polarizations per partial (own decay rates, small detune)
    pluck_click: bool = False  # structured: learned nail-click and finger-thud waveforms at every onset
    pluck_click_len: float = 0.03  # s after the onset
    pluck_click_lead: float = 0.02  # s before the onset (finger landing on the string, previous note being stopped)
    sympathetic: bool = False  # structured: open strings resonate with played partials near theirs
    symp_k: int = 16  # partials per open string
    symp_pluck: bool = False  # sympathetic: every pluck also kicks all open strings through the bridge (broadband)
    squeak: bool = False  # structured: noise just before notes on the wound strings (below G3), per-note level
    squeak_len: float = 0.06  # s before the onset
    squeak_bands: int = 16
    note_timbre: bool = False  # structured: per-note attack timbre read from the audio, input of the note MLP
    timbre_k: int = 16
    timbre_late: bool = False

    @property
    def n_feat(self):
        return N_FEAT + 2 * int(self.audio_feats)

    @property
    def n_bands(self):
        return 64 if self.structured else 65


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class Recording:
    """Per-recording note arrays (numpy) needed to build windows."""

    def __init__(self, idx, y, notes, tuning_cents, cfg: NetConfig, log_b):
        sr = cfg.sr
        self.idx, self.y = idx, y
        on = np.round(notes.onset.values * sr).astype(np.int64)
        end = on + int(cfg.max_ring * sr)
        for v in range(cfg.n_voices):
            ii = np.where(notes.voice.values == v)[0]
            end[ii[:-1]] = np.minimum(end[ii[:-1]], on[ii[1:]])
        self.onset, self.end = on, np.maximum(end, on + 1)
        self.midi_end = np.maximum(np.round(notes.offset.values * sr).astype(np.int64), on + 1)
        self.pitch = notes.pitch.values.astype(np.int64)
        self.voice = notes.voice.values.astype(np.int64)
        self.loud = np.nan_to_num(notes.loud_rel_db.values / 10.0)
        self.conf = np.log1p(notes.onset_conf.values) / 3.0
        f0 = 440 * 2 ** ((self.pitch - 69 + tuning_cents / 100) / 12)
        k = np.arange(1, cfg.k + 1)
        B = np.exp(log_b[0] + log_b[1] * self.pitch) if cfg.structured else np.zeros(len(f0))
        self.freqs = f0[:, None] * k * np.sqrt(1 + B[:, None] * k ** 2)
        if cfg.sympathetic:
            self.open_pitch = OPEN_STRINGS
            fo = 440 * 2 ** ((OPEN_STRINGS - 69 + tuning_cents / 100) / 12)
            ks = np.arange(1, cfg.symp_k + 1)
            Bo = np.exp(log_b[0] + log_b[1] * OPEN_STRINGS)
            self.open_freqs = fo[:, None] * ks * np.sqrt(1 + Bo[:, None] * ks ** 2)  # (6, symp_k)
        if cfg.note_timbre:  # dB/10, clipped
            self.timbre = (note_timbre_db(y, sr, notes.onset.values, self.freqs, cfg.timbre_k) / 10).clip(-3, 3).astype(np.float32)
            if cfg.timbre_late:
                late = note_timbre_db(y, sr, notes.onset.values, self.freqs, cfg.timbre_k, start=0.15)
                self.timbre = np.concatenate([self.timbre, (late / 10).clip(-3, 3).astype(np.float32)], 1)
        self.by_voice = [np.where(self.voice == v)[0] for v in range(cfg.n_voices)]
        if cfg.squeak:  # notes below G3 can only be on a wound string; context = the previous such note
            self.wound = self.pitch < 55
            self.wound_interval = np.zeros(len(on))
            self.wound_gap = np.full(len(on), 5.0)
            w = np.where(self.wound)[0]
            w = w[np.argsort(on[w])]
            self.wound_interval[w[1:]] = np.abs(np.diff(self.pitch[w]))
            self.wound_gap[w[1:]] = np.minimum(np.diff(on[w]) / sr, 5.0)
        self.track_loud = self.track_dev = None  # (V, frames at hop), set by set_tracks

    def set_tracks(self, loud_db, dev_cents):
        """Per-voice frame tracks from the audio (see tesis.data.note_tracks), frames centered at i * hop."""
        med = np.nanmedian(loud_db)
        self.track_loud = np.nan_to_num((loud_db - med) / 10.0, nan=-3.0).astype(np.float32)
        self.track_dev = np.nan_to_num(dev_cents, nan=0.0).clip(-100, 100).astype(np.float32)

    def track_frames(self, tf, cfg):
        """Track values at absolute sample positions tf (multiples of hop): loudness (V, F), deviation cents (V, F)."""
        i = tf // cfg.hop
        ok = (i >= 0) & (i < self.track_loud.shape[1])
        ic = np.clip(i, 0, self.track_loud.shape[1] - 1)
        return np.where(ok, self.track_loud[:, ic], -3.0), np.where(ok, self.track_dev[:, ic], 0.0)

    def frame_features(self, w0, cfg: NetConfig):
        """(V, F, N_FEAT) features at frames w0 - burn + i*hop, and the note index each voice frame belongs to."""
        sr, H = cfg.sr, cfg.hop
        B, L = int(cfg.burn * sr), int(cfg.win * sr)
        tf = w0 - B + H * np.arange((B + L) // H + 1)
        feats = np.zeros((cfg.n_voices, len(tf), cfg.n_feat), np.float32)
        if cfg.audio_feats:
            tl, td = self.track_frames(tf, cfg)
        for v, ii in enumerate(self.by_voice):
            if len(ii) == 0:
                continue
            j = np.searchsorted(self.onset[ii], tf, side="right") - 1
            n = ii[np.clip(j, 0, None)]
            act = (j >= 0) & (self.end[n] > tf)
            since = np.clip(tf - self.onset[n], 0, None) / sr
            after = np.clip(tf - self.midi_end[n], 0, None) / sr
            f = np.stack([np.ones_like(since), (self.pitch[n] - 64) / 12.0, np.log1p(since / 0.05),
                          np.log1p(after / 0.05), self.loud[n], ((tf - self.onset[n]) >= 0) & ((tf - self.onset[n]) < H),
                          self.conf[n]], -1)
            if cfg.audio_feats:
                f = np.concatenate([f, tl[v][:, None], td[v][:, None] / 50.0], -1)
            feats[v] = f * act[:, None]
        return feats


class GuitarNet(nn.Module):
    def __init__(self, cfg: NetConfig, n_recordings: int):
        super().__init__()
        self.cfg = cfg
        K, Hd, R = cfg.k, cfg.hidden, cfg.rec_dim
        self.rec_emb = nn.Embedding(n_recordings, R)
        nn.init.normal_(self.rec_emb.weight, std=0.1)
        self.inp = nn.Sequential(nn.Linear(2 * cfg.n_feat + R, Hd), nn.LeakyReLU(), nn.Linear(Hd, Hd), nn.LeakyReLU())
        self.gru = nn.GRU(Hd, Hd, batch_first=True)
        self.register_buffer("logk", torch.log(torch.arange(1, K + 1).float()))
        out_dim = K if cfg.structured else K + 1
        self.voice_out = nn.Linear(Hd, out_dim)
        nn.init.zeros_(self.voice_out.weight)
        nn.init.zeros_(self.voice_out.bias)
        self.noise_out = nn.Sequential(nn.Linear(Hd + R, Hd), nn.LeakyReLU(), nn.Linear(Hd, cfg.n_bands))
        nn.init.constant_(self.noise_out[-1].bias, -8.0)
        if cfg.structured:
            self.note_mlp = nn.Sequential(nn.Linear(3 + R + cfg.timbre_k * int(cfg.note_timbre) * (1 + int(cfg.timbre_late)), Hd), nn.LeakyReLU(), nn.Linear(Hd, Hd), nn.LeakyReLU(),
                                          nn.Linear(Hd, K + 1 + int(cfg.body_ir) + 2 * int(cfg.pluck_click)))
            nn.init.zeros_(self.note_mlp[-1].weight)
            nn.init.zeros_(self.note_mlp[-1].bias)
            self.decay_mlp = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 1))
            nn.init.zeros_(self.decay_mlp[-1].weight)
            nn.init.zeros_(self.decay_mlp[-1].bias)
            self.body_base = nn.Parameter(torch.zeros(cfg.body_bins))
            self.rec_body_filter = nn.Embedding(n_recordings, cfg.body_bins)  # per-recording deviation
            nn.init.zeros_(self.rec_body_filter.weight)
            self.register_buffer("body_centers_log", torch.log(torch.tensor(np.geomspace(50.0, cfg.sr / 2, cfg.body_bins))).float())
            self.rec_floor = nn.Embedding(n_recordings, cfg.n_bands)
            nn.init.constant_(self.rec_floor.weight, -10.0)  # step 2 ended near -30 dB share
            self.rec_burst = nn.Embedding(n_recordings, cfg.burst_bands)
            nn.init.constant_(self.rec_burst.weight, -3.0)
            self.burst_logtau = nn.Parameter(torch.tensor(math.log(0.003)))
            if cfg.body_ir:
                n_ir = int(cfg.body_ir_len * cfg.sr)
                g = torch.Generator().manual_seed(0)
                self.body_ir_wave = nn.Parameter(1e-2 * torch.randn(n_ir, generator=g))  # shared by all recordings
                self.rec_body_eq = nn.Embedding(n_recordings, cfg.body_eq_bands)
                nn.init.zeros_(self.rec_body_eq.weight)
                t = torch.arange(n_ir) / cfg.sr
                self.register_buffer("body_ir_env", torch.exp(-t / cfg.body_ir_tau))
            if cfg.two_decay:  # per (pitch, partial frequency): share of the 2nd polarization, its decay ratio, detune
                self.pol_mlp = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 3))
                nn.init.zeros_(self.pol_mlp[-1].weight)
                with torch.no_grad():  # 30 % in the 2nd polarization, decaying 3x slower, 0.6 Hz apart
                    self.pol_mlp[-1].bias.copy_(torch.tensor([math.log(0.3 / 0.7), -0.916, 0.2027]))
            if cfg.pluck_click:
                n_c = int((cfg.pluck_click_lead + cfg.pluck_click_len) * cfg.sr)
                g = torch.Generator().manual_seed(2)
                self.click_wave = nn.Parameter(1e-2 * torch.randn(2, n_c, generator=g))  # shared by all recordings
                t = torch.arange(n_c) / cfg.sr - cfg.pluck_click_lead  # 0 = onset
                pre = torch.exp(-t.abs() / 0.008)  # before the onset: fades in over ~8 ms
                self.register_buffer("click_env", torch.stack([torch.where(t < 0, pre, torch.exp(-t / 0.004)),
                                                               torch.where(t < 0, pre, torch.exp(-t / 0.012))]))
            if cfg.squeak:
                self.squeak_mlp = nn.Sequential(nn.Linear(4 + R, 64), nn.LeakyReLU(), nn.Linear(64, 1))
                nn.init.zeros_(self.squeak_mlp[-1].weight)
                nn.init.constant_(self.squeak_mlp[-1].bias, -5.0)
                self.rec_squeak = nn.Embedding(n_recordings, cfg.squeak_bands)  # per-recording squeak spectrum
                with torch.no_grad():  # start in 1.5-6 kHz
                    c = torch.tensor(np.geomspace(40.0, cfg.sr / 2, cfg.squeak_bands)).float()
                    self.rec_squeak.weight[:] = torch.where((c > 1500) & (c < 6000), 0.0, -6.0)
            if cfg.sympathetic:
                self.symp_gain = nn.Parameter(torch.full((len(OPEN_STRINGS), cfg.symp_k), -3.0))  # log
                self.symp_logbw = nn.Parameter(torch.tensor(math.log(2.0)))  # Hz, how close a partial must be
                self.rec_symp = nn.Embedding(n_recordings, 1)  # per-recording level (players damp differently)
                nn.init.zeros_(self.rec_symp.weight)
                if cfg.symp_pluck:  # log gain of the pluck kick per open-string partial
                    self.symp_kick = nn.Parameter(torch.zeros(len(OPEN_STRINGS), cfg.symp_k))
        if cfg.reverb:
            assert cfg.reverb_len <= cfg.burn
            Rn = int(cfg.reverb_len * cfg.sr)
            self.rec_reverb = nn.Embedding(n_recordings, 2 * cfg.reverb_bands)  # per recording: band levels, RT60s
            with torch.no_grad():  # ~-20 dB wet at init; RT60 ~0.8 s
                self.rec_reverb.weight[:, :cfg.reverb_bands] = -6.0
                self.rec_reverb.weight[:, cfg.reverb_bands:] = -1.0
            g = torch.Generator().manual_seed(1)
            noise = torch.randn(Rn, generator=g)
            Xn = torch.fft.rfft(noise)
            fr = torch.fft.rfftfreq(Rn, 1 / cfg.sr)
            lc = np.log(np.geomspace(60.0, 10000.0, cfg.reverb_bands))
            lf = torch.log(fr.clamp(min=30.0)).numpy()
            bands = []
            for c_i in range(cfg.reverb_bands):  # triangular bands in log frequency, summing to ~1
                w = np.clip(1 - np.abs(lf - lc[c_i]) / (lc[1] - lc[0]), 0, 1)
                if c_i == 0:
                    w[lf < lc[0]] = 1
                if c_i == cfg.reverb_bands - 1:
                    w[lf > lc[-1]] = 1
                bands.append(torch.fft.irfft(Xn * torch.tensor(w, dtype=torch.float32), Rn))
            self.register_buffer("rev_noise", torch.stack(bands))
            t = torch.arange(Rn) / cfg.sr
            self.register_buffer("rev_t", t)
            self.register_buffer("rev_onset", ((t - 0.002) / 0.003).clamp(0, 1))  # no direct sound: 2-5 ms fade-in
        W = log_band_matrix(cfg.n_bands, cfg.n_fft_noise // 2 + 1, cfg.sr) if cfg.structured else None
        self.register_buffer("W", W if W is not None else torch.zeros(1))

    # ---- helpers
    def body_db(self, f, emb):
        """Log body gain at frequencies f (any shape), recording embedding emb (R,)."""
        resp = self.body_base + self.rec_body_filter.weight[self._rec]
        x = torch.log(f.clamp(50.0, self.cfg.sr / 2))
        pos = (x - self.body_centers_log[0]) / (self.body_centers_log[1] - self.body_centers_log[0])
        lo = pos.floor().long().clamp(0, self.cfg.body_bins - 2)
        fr = (pos - lo).clamp(0, 1)
        return resp[lo] * (1 - fr) + resp[lo + 1] * fr

    def note_input(self, rec, ii, emb):
        """(len(ii), 3) pitch / loudness / onset clarity of these notes, and the note MLP's full input."""
        dev = emb.device
        x = torch.stack([torch.as_tensor((rec.pitch[ii] - 64) / 12.0, device=dev).float(),
                         torch.as_tensor(rec.loud[ii], device=dev).float(),
                         torch.as_tensor(rec.conf[ii], device=dev).float()], -1)
        full = [x, emb[None].expand(len(ii), -1)]
        if self.cfg.note_timbre:
            full.append(torch.as_tensor(rec.timbre[ii], device=dev))
        return x, torch.cat(full, -1)

    def sigma(self, pitch, f):
        """Decay rate (1/s) for partial frequencies f (Na, K) of notes with these pitches."""
        x = torch.stack([((pitch.float() - 64) / 12)[:, None].expand_as(f), torch.log(f / 1000)], -1)
        sigma0 = 1.5 + 2.0 * f / 1000
        return Fn.softplus(self.decay_mlp(x)[..., 0] + inv_softplus(sigma0))

    def body_ir(self, emb, n_fft):
        """Spectrum (n_fft//2+1,) of the shared pluck response with this recording's EQ, low-passed."""
        cfg = self.cfg
        f = torch.fft.rfftfreq(n_fft, 1 / cfg.sr).to(emb.device)
        lp = 1 / (1 + (f / cfg.body_ir_cutoff) ** 8)
        centers = torch.log(torch.tensor(np.geomspace(40.0, cfg.body_eq_max, cfg.body_eq_bands), device=emb.device).float())
        x = torch.log(f.clamp(40.0, cfg.body_eq_max))
        pos = ((x - centers[0]) / (centers[1] - centers[0])).clamp(0, cfg.body_eq_bands - 1 - 1e-4)
        lo = pos.floor().long()
        eq_db = self.rec_body_eq.weight[self._rec]
        eq = eq_db[lo] * (1 - (pos - lo)) + eq_db[lo + 1] * (pos - lo)
        return torch.fft.rfft(self.body_ir_wave * self.body_ir_env, n_fft) * lp * torch.exp(eq)

    def sympathetic(self, rec, f, amp, n_abs, emb, s0):
        """Open strings driven by played partials close to theirs.

        The drive of open-string partial j is the summed amplitude of played partials near it (Gaussian in Hz);
        its own amplitude follows that drive through a one-pole resonator with the decay of a free string at that
        frequency (self.sigma), so it builds up while driven and keeps ringing after.
        f: (Na, K) played partial frequencies; amp: (Na, K, Ld) their amplitudes; returns (Ld,).
        """
        cfg, sr, dev = self.cfg, self.cfg.sr, amp.device
        fo = torch.as_tensor(rec.open_freqs, device=dev).float()  # (S, Ks)
        S, Ks = fo.shape
        bw = torch.exp(self.symp_logbw)
        c = torch.exp(-((f[None, None] - fo[..., None, None]) / bw) ** 2)  # (S, Ks, Na, K)
        drive = (c.reshape(S * Ks, -1) @ amp.reshape(-1, amp.shape[-1])).reshape(S, Ks, -1)
        if cfg.symp_pluck:  # each onset in the span kicks every open-string partial, scaled by the note's loudness
            Ld = amp.shape[-1]
            ex = np.where((rec.onset >= s0) & (rec.onset < s0 + Ld))[0]
            if len(ex):
                kick = torch.zeros(Ld, device=dev).index_add(
                    0, torch.as_tensor(rec.onset[ex] - s0, device=dev),
                    torch.exp(1.151 * torch.as_tensor(rec.loud[ex], device=dev).float()) * 0.03)
                drive = drive + torch.exp(self.symp_kick)[..., None] * kick * sr / 100  # ~10 ms worth of drive
        sig = self.sigma(torch.as_tensor(rec.open_pitch, device=dev), fo)  # decay of a free string
        Ld = amp.shape[-1]
        t = torch.arange(Ld, device=dev) / sr
        h = (1 - torch.exp(-sig / sr))[..., None] * torch.exp(-sig[..., None] * t)  # (S, Ks, Ld), unit DC gain
        nfft = 2 ** math.ceil(math.log2(2 * Ld))
        env = torch.fft.irfft(torch.fft.rfft(drive, nfft) * torch.fft.rfft(h, nfft), nfft)[..., :Ld]
        gain = torch.exp(self.symp_gain + self.rec_symp.weight[self._rec] + self.body_db(fo, emb))
        cyc = torch.remainder(fo.double()[..., None] * n_abs.double() / sr, 1.0).float()
        alive = (fo < sr / 2).float()[..., None]
        return (gain[..., None] * env * alive * torch.sin(2 * math.pi * cyc)).sum((0, 1))

    def squeaks(self, rec, s0, Ld, emb, gen):
        """Filtered noise in the squeak_len before each wound-string onset; level from the note context."""
        cfg, sr, dev = self.cfg, self.cfg.sr, emb.device
        n_s = int(cfg.squeak_len * sr)
        ex = np.where(rec.wound & (rec.onset > s0) & (rec.onset - n_s < s0 + Ld))[0]
        env = torch.zeros(Ld, device=dev)
        if len(ex):
            x = torch.as_tensor(np.stack([(rec.pitch[ex] - 45) / 12.0, rec.wound_interval[ex] / 12.0,
                                          np.log(rec.wound_gap[ex] + 0.01), rec.loud[ex]], -1), device=dev).float()
            lv = self.squeak_mlp(torch.cat([x, emb[None].expand(len(ex), -1)], -1))[:, 0]
            t = (torch.arange(Ld, device=dev)[None] + s0 - torch.as_tensor(rec.onset[ex], device=dev)[:, None]).float()
            shape = torch.sin(math.pi * ((t + n_s) / n_s).clamp(0, 1)) ** 2  # hump over [-squeak_len, 0]
            env = (torch.exp(lv)[:, None] * shape).sum(0)
        wn = torch.randn(1, Ld, generator=gen, device=dev)
        return filter_noise(wn, torch.exp(self.rec_squeak.weight[self._rec])[None], sr)[0] * env

    def pluck_gate(self, rec, w0, n, dev):
        """(n,) gain in [0, 1] at the noise frames w0 + i * noise_hop: 1 near an onset, 0 far from all of them."""
        cfg, sr = self.cfg, self.cfg.sr
        pre, hold, tau = cfg.gate_pre * sr, cfg.gate_hold * sr, cfg.gate_tau * sr
        tq = w0 + cfg.noise_hop * np.arange(n)
        ex = np.where((rec.onset > tq[0] - hold - 8 * tau) & (rec.onset < tq[-1] + pre))[0]
        if len(ex) == 0:
            return torch.zeros(n, device=dev)
        dt = torch.as_tensor(tq[None] - rec.onset[ex][:, None], device=dev).float()  # samples after the onset
        rise = torch.sin(0.5 * math.pi * ((dt + pre) / pre).clamp(0, 1)) ** 2
        fall = torch.exp(-(dt - hold).clamp(min=0) / tau)
        return torch.where(dt < 0, rise, fall).sum(0).clamp(max=1.0)

    def reverb_ir(self, emb):
        """Per-recording room response: 8 log-spaced bands of fixed noise, each with its own level and decay (RT60)."""
        out = self.rec_reverb.weight[self._rec]
        nb = self.cfg.reverb_bands
        gain = torch.exp(out[:nb])
        rt60 = 0.2 + 2.3 * torch.sigmoid(out[nb:])  # 0.2-2.5 s: a room, never an endless sustain
        t = self.rev_t
        env = torch.exp(-t[None] * 6.91 / rt60[:, None])  # (bands, Rn)
        return (gain[:, None] * env * self.rev_noise).sum(0) * self.rev_onset

    def forward(self, rec: Recording, w0: int, noise_seed=None):
        cfg, sr, dev = self.cfg, self.cfg.sr, self.logk.device
        K = cfg.k
        L, Bn, H = int(cfg.win * sr), int(cfg.burn * sr), cfg.hop
        R = int(cfg.reverb_len * sr) if cfg.reverb else 0  # dry parts start R samples early (reverb tail)
        s0, Ld = w0 - R, L + R
        gen = torch.Generator(device=dev)
        gen.manual_seed(noise_seed if noise_seed is not None else int(torch.randint(0, 2 ** 31, (1,))))
        emb = self.rec_emb.weight[rec.idx]
        self._rec = rec.idx

        feats = torch.tensor(rec.frame_features(w0, cfg), device=dev)  # (V, F, N_FEAT), frames from w0 - burn
        V, F, _ = feats.shape
        x = torch.cat([feats, feats.mean(0, keepdim=True).expand(V, -1, -1), emb[None, None].expand(V, F, -1)], -1)
        h_all, _ = self.gru(self.inp(x))  # (V, F, Hd)
        h = h_all[:, Bn // H:]  # window frames (noise head)
        vout = self.voice_out(h_all[:, (Bn - R) // H:])  # frames covering the dry span [s0, w0 + L]

        parts = {}
        idx = np.where((rec.onset < s0 + Ld) & (rec.end > s0))[0]
        n_abs = s0 + torch.arange(Ld, device=dev)
        if len(idx):
            pol_phase = None
            on = torch.as_tensor(rec.onset[idx], device=dev)
            end = torch.as_tensor(rec.end[idx], device=dev)
            voice = torch.as_tensor(rec.voice[idx], device=dev)
            f = torch.as_tensor(rec.freqs[idx], device=dev, dtype=torch.float64)
            t_rel = n_abs[None] - on[:, None]
            gate = ((t_rel >= 0) & (n_abs[None] < end[:, None])).float() * \
                ((end[:, None] - n_abs[None]).float() / (cfg.fade * sr)).clamp(max=1)
            ts = t_rel.clamp(min=0).float() / sr
            if cfg.structured:
                ramp = (t_rel.float() / (cfg.attack_ms * 1e-3 * sr)).clamp(0, 1)
                gate = gate * 0.5 * (1 - torch.cos(math.pi * ramp))
                note_in, mlp_in = self.note_input(rec, idx, emb)
                nout = self.note_mlp(mlp_in)
                tmpl = nout[:, :K] + math.log(0.03) - self.logk + 1.151 * note_in[:, 1:2]  # loud is power dB/10 -> log amplitude
                static = tmpl + self.body_db(f.float(), emb)
                sig = self.sigma(torch.as_tensor(rec.pitch[idx], device=dev), f.float())
                delta = frames_to_samples(vout.permute(0, 2, 1), Ld)  # (V, K, Ld)
                if cfg.two_decay:
                    x_p = torch.stack([note_in[:, :1].expand(-1, K), torch.log(f.float() / 1000)], -1)
                    po = self.pol_mlp(x_p)
                    w2 = torch.sigmoid(po[..., 0])
                    r2 = 0.02 + 0.98 * torch.sigmoid(po[..., 1])  # 2nd decay rate / 1st
                    det = 3.0 * torch.tanh(po[..., 2])  # Hz
                    # smooth in time (decays, beating < 3 Hz): computed every ~2 ms and interpolated, to save memory
                    tc = ts[:, torch.linspace(0, Ld - 1, Ld // 48 + 1, device=dev).round().long()][:, None, :]
                    la1 = torch.log1p(-w2 + 1e-6)[..., None] - sig[..., None] * tc
                    la2 = torch.log(w2 + 1e-6)[..., None] - (sig * r2)[..., None] * tc
                    mx = torch.maximum(la1, la2)
                    a1, a2 = torch.exp(la1 - mx), torch.exp(la2 - mx)
                    th = 2 * math.pi * det[..., None] * tc
                    re, im = a1 + a2 * torch.cos(th), a2 * torch.sin(th)
                    decay = frames_to_samples(mx + 0.5 * torch.log(re ** 2 + im ** 2 + 1e-8), Ld)
                    ph = torch.atan2(im, re + 1e-7) / (2 * math.pi)  # cycles, wrapped to [-0.5, 0.5]
                    dph = torch.diff(ph, dim=-1)
                    ph = torch.cat([ph[..., :1], ph[..., :1] + torch.cumsum(dph - torch.round(dph), -1)], -1)  # unwrapped
                    pol_phase = frames_to_samples(ph, Ld)
                else:
                    decay = -sig[..., None] * ts[:, None, :]
                log_a = static[..., None] + decay + delta[voice]
                pluck = nout[:, K] + 1.151 * note_in[:, 1]
                env = (torch.exp(pluck[:, None] - ts / torch.exp(self.burst_logtau)) * (t_rel >= 0)).sum(0)
                wn = torch.randn(1, Ld, generator=gen, device=dev)
                parts["burst"] = filter_noise(wn, torch.exp(self.rec_burst.weight[self._rec])[None], sr)[0] * env
                self._delta_reg = (vout ** 2).mean()
                self._delta_tv = ((vout[:, 1:] - vout[:, :-1]) ** 2).mean()
            else:
                amp = vout[..., :1] + math.log(0.1)
                dist = Fn.log_softmax(vout[..., 1:] - self.logk, -1)
                log_a_frames = (amp + dist).permute(0, 2, 1)  # (V, K, frames)
                log_a = frames_to_samples(log_a_frames, Ld)[voice]
                self._delta_reg = torch.zeros((), device=dev)
            alive = (f < sr / 2).float()[..., None]
            if cfg.audio_feats:
                # measured pitch deviation bends all partials of the note: warp the note's time axis
                tf_d = s0 + H * np.arange(Ld // H + 1)
                _, dev_f = rec.track_frames(tf_d, cfg)
                dev_s = frames_to_samples(torch.as_tensor(dev_f, device=dev).float(), Ld)[voice].double()  # (Na, Ld)
                ratio = torch.pow(2.0, dev_s / 1200) * (t_rel >= 0).double()
                tau = (s0 - on).clamp(min=0).double()[:, None] / sr + torch.cumsum(ratio, 1) / sr
                cyc = torch.remainder(f[..., None] * tau[:, None, :], 1.0).float()
            else:
                cyc = torch.remainder(f[..., None] * t_rel.clamp(min=0).double()[:, None, :] / sr, 1.0).float()
            if pol_phase is not None:
                cyc = cyc + pol_phase
            amp = torch.exp(log_a) * alive * gate[:, None]
            parts["strings"] = (amp * torch.sin(2 * math.pi * cyc)).sum((0, 1))
            if cfg.structured and cfg.sympathetic:
                parts["sympathetic"] = self.sympathetic(rec, f.float(), amp, n_abs, emb, s0)
        else:
            parts["strings"] = torch.zeros(Ld, device=dev)
            self._delta_reg = torch.zeros((), device=dev)

        if cfg.structured and cfg.body_ir:
            # every pluck (also those just before the span) excites the shared body response
            n_ir = self.body_ir_wave.shape[0]
            ex = np.where((rec.onset >= s0 - n_ir) & (rec.onset < s0 + Ld))[0]
            imp = torch.zeros(n_ir + Ld, device=dev)
            if len(ex):
                ex_in, mlp_in = self.note_input(rec, ex, emb)
                thump = self.note_mlp(mlp_in)[:, K + 1] + 1.151 * ex_in[:, 1]
                imp = imp.index_add(0, torch.as_tensor(rec.onset[ex] - (s0 - n_ir), device=dev), torch.exp(thump))
            nfft = 2 ** math.ceil(math.log2(2 * (n_ir + Ld)))
            parts["body"] = torch.fft.irfft(torch.fft.rfft(imp, nfft) * self.body_ir(emb, nfft), nfft)[n_ir:n_ir + Ld]

        if cfg.structured and cfg.pluck_click:
            # every pluck (also those just before the span) plays the two short click waveforms at its own levels
            n_c, lead = self.click_wave.shape[1], int(cfg.pluck_click_lead * sr)
            ex = np.where((rec.onset - lead >= s0 - n_c) & (rec.onset - lead < s0 + Ld))[0]
            imp = torch.zeros(2, n_c + Ld, device=dev)
            if len(ex):
                ex_in, mlp_in = self.note_input(rec, ex, emb)
                j0 = K + 1 + int(cfg.body_ir)
                lv = self.note_mlp(mlp_in)[:, j0:j0 + 2] + 1.151 * ex_in[:, 1:2]
                pos = torch.as_tensor(rec.onset[ex] - lead - (s0 - n_c), device=dev)
                imp = torch.stack([imp[c].index_add(0, pos, torch.exp(lv[:, c])) for c in range(2)])
            nfft = 2 ** math.ceil(math.log2(2 * (n_c + Ld)))
            kern = torch.fft.rfft(self.click_wave * self.click_env, nfft)
            parts["click"] = torch.fft.irfft((torch.fft.rfft(imp, nfft) * kern).sum(0), nfft)[n_c:n_c + Ld]

        if cfg.structured and cfg.squeak:
            parts["squeak"] = self.squeaks(rec, s0, Ld, emb, gen)

        if cfg.reverb:
            dry = sum(parts.values())  # (Ld,) instrument signal from s0
            nfft = 2 ** math.ceil(math.log2(Ld + R))
            wet = torch.fft.irfft(torch.fft.rfft(dry, nfft) * torch.fft.rfft(self.reverb_ir(emb), nfft), nfft)
            parts = {k: v[R:] for k, v in parts.items()}
            parts["reverb"] = wet[R:R + L]

        if cfg.structured:
            parts["floor"] = filter_noise(torch.randn(1, L, generator=gen, device=dev), torch.exp(self.rec_floor.weight[self._rec])[None], sr)[0]
        if not cfg.noise:
            self._noise_mag = torch.zeros((), device=dev)
            return sum(parts.values()), parts
        nb = self.noise_out(torch.cat([h.mean(0), emb[None].expand(h.shape[1], -1)], -1))  # (Fw, n_bands)
        Fn_ = L // cfg.noise_hop + 1
        nb = frames_to_samples(nb.T, Fn_).T  # (Fn_, n_bands)
        if cfg.noise_gate:
            nb = nb + torch.log(self.pluck_gate(rec, w0, Fn_, dev) + 1e-4)[:, None]
        self._noise_mag = torch.exp(nb).mean()
        ph = torch.rand((1, cfg.n_fft_noise // 2 + 1, Fn_), generator=gen, device=dev) * 2 * math.pi
        parts["noise"] = filtered_noise(torch.exp(nb)[None], L, cfg.n_fft_noise, cfg.noise_hop, ph,
                                        self.W if cfg.structured else None)[0]
        return sum(parts.values()), parts

    def regularizer(self):
        reg = 1e-3 * self._noise_mag * self.cfg.n_bands + 1e-4 * self._delta_reg
        if self.cfg.delta_tv > 0 and hasattr(self, "_delta_tv"):
            reg = reg + self.cfg.delta_tv * self._delta_tv
        return reg
