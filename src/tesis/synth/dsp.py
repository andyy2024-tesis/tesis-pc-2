"""Differentiable synthesis building blocks (PyTorch). Shapes: B = batch, K = partials, T = samples, F = frames."""
import math

import numpy as np
import torch
import torch.nn.functional as Fn


def frames_to_samples(x: torch.Tensor, n: int) -> torch.Tensor:
    """Linear interpolation of the last axis from F frames to n samples."""
    shape = x.shape
    y = Fn.interpolate(x.reshape(-1, 1, shape[-1]), size=n, mode="linear", align_corners=True)
    return y.reshape(*shape[:-1], n)


def partial_bank(freqs: torch.Tensor, amps: torch.Tensor, sr: int, start: torch.Tensor = None) -> torch.Tensor:
    """Sum of sinusoids with constant frequencies.

    freqs: (B, K) Hz; amps: (B, K, T) linear; start: (B,) sample index where the partials begin (silent before).
    Partials at or above Nyquist are muted.
    """
    T = amps.shape[-1]
    t = torch.arange(T, device=amps.device, dtype=amps.dtype)
    if start is not None:
        t = (t[None, :] - start[:, None]).clamp(min=0)[:, None, :]  # (B, 1, T)
        on = (torch.arange(T, device=amps.device)[None, :] >= start[:, None]).to(amps.dtype)[:, None, :]
    else:
        t, on = t[None, None, :], 1.0
    phase = 2 * math.pi * freqs[..., None] * t / sr
    alive = (freqs < sr / 2).to(amps.dtype)[..., None]
    return (amps * alive * on * torch.sin(phase)).sum(1)


def log_band_matrix(n_bands: int, n_bins: int, sr: int, f_min: float = 40.0, device=None) -> torch.Tensor:
    """(n_bands, n_bins) matrix that interpolates log-spaced band magnitudes onto linear FFT bins.

    Linear bands (as in DDSP) are ~185 Hz wide at 24 kHz with 65 bands, far too coarse for guitar low end.
    """
    centers = np.geomspace(f_min, sr / 2, n_bands)
    bins = np.linspace(0, sr / 2, n_bins)
    lc, lb = np.log(centers), np.log(np.maximum(bins, f_min))
    W = np.zeros((n_bands, n_bins))
    j = np.clip(np.searchsorted(lc, lb) - 1, 0, n_bands - 2)
    frac = np.clip((lb - lc[j]) / (lc[j + 1] - lc[j]), 0, 1)
    W[j, np.arange(n_bins)] = 1 - frac
    W[j + 1, np.arange(n_bins)] += frac
    return torch.tensor(W, dtype=torch.float32, device=device)


def bands_to_bins(band_mags: torch.Tensor, n_bins: int, W: torch.Tensor = None) -> torch.Tensor:
    """(..., n_bands) band magnitudes -> (..., n_bins). Linear band spacing unless a log_band_matrix W is given."""
    if W is not None:
        return band_mags @ W
    return frames_to_samples(band_mags, n_bins)


def filtered_noise(band_mags: torch.Tensor, n_samples: int, n_fft: int, hop: int, phase: torch.Tensor,
                   W: torch.Tensor = None) -> torch.Tensor:
    """Time-varying filtered noise built in the STFT domain.

    band_mags: (B, F, n_bands) >= 0 with F = n_samples // hop + 1; phase: (B, n_fft//2+1, F) fixed random phases.
    """
    mags = bands_to_bins(band_mags, n_fft // 2 + 1, W).transpose(1, 2)  # (B, bins, F)
    spec = torch.polar(mags, phase)
    win = torch.hann_window(n_fft, device=band_mags.device)
    return torch.istft(spec, n_fft, hop, window=win, length=n_samples, center=True)


def filter_noise(noise: torch.Tensor, band_mags: torch.Tensor, sr: int = None, f_min: float = 40.0) -> torch.Tensor:
    """Zero-phase filter a (B, T) noise signal with a (B, n_bands) magnitude response (log-spaced if sr is given)."""
    T = noise.shape[-1]
    n = 2 ** math.ceil(math.log2(T))
    X = torch.fft.rfft(noise, n)
    W = log_band_matrix(band_mags.shape[-1], X.shape[-1], sr, f_min, noise.device) if sr else None
    H = bands_to_bins(band_mags, X.shape[-1], W)
    return torch.fft.irfft(X * H, n)[..., :T]


def damped_modes(freqs: torch.Tensor, amps: torch.Tensor, decays: torch.Tensor, T: int, sr: int,
                 start: torch.Tensor, phases: torch.Tensor = None) -> torch.Tensor:
    """Sum of exponentially decaying sinusoids starting at `start` (impulse response of a modal resonator bank).

    freqs: (M,) or (B, M) Hz; amps, decays: (B, M) (decay in 1/s); start: (B,) samples; phases: (B, M) radians.
    """
    t = torch.arange(T, device=amps.device, dtype=amps.dtype)
    tn = (t[None, :] - start[:, None]) / sr  # (B, T)
    on = (tn >= 0).to(amps.dtype)[:, None, :]
    tn = tn.clamp(min=0)[:, None, :]
    f = freqs if freqs.dim() == 2 else freqs[None].expand_as(amps)
    ph = 0.0 if phases is None else phases[..., None]
    x = amps[..., None] * torch.exp(-decays[..., None] * tn) * torch.sin(2 * math.pi * f[..., None] * tn + ph) * on
    return x.sum(1)


class MultiScaleSpectralLoss(torch.nn.Module):
    """DDSP loss: sum over FFT sizes of L1(|S| - |S^|) + log_weight * L1(log|S| - log|S^|).

    log_weight=0 keeps only the linear-magnitude term, which barely cares about near-silent regions (the log term is
    what pays the model for filling them with noise).
    """

    def __init__(self, fft_sizes=(2048, 1024, 512, 256, 128, 64), eps=1e-7, log_weight=1.0):
        super().__init__()
        self.fft_sizes, self.eps, self.log_weight = fft_sizes, eps, log_weight

    def spec(self, x, n):
        return torch.stft(x, n, n // 4, window=torch.hann_window(n, device=x.device), return_complex=True).abs()

    def forward(self, x, y, reduce=True):
        total = 0.0
        for n in self.fft_sizes:
            sx, sy = self.spec(x, n), self.spec(y, n)
            d = (sx - sy).abs().mean((-1, -2))
            if self.log_weight:
                d = d + self.log_weight * (torch.log(sx + self.eps) - torch.log(sy + self.eps)).abs().mean((-1, -2))
            total = total + d
        return total.mean() if reduce else total


class MultiScaleMelLoss(torch.nn.Module):
    """Multi-resolution mel-spectrogram loss: L1 on mel magnitudes + L1 on log mel magnitudes.

    Compared with the linear-bin spectral loss: frequencies are weighted roughly like hearing (mel), and the log term
    has a floor (~25th percentile of real magnitudes at each resolution) so near-silent regions are not worth filling
    with noise (the log-spectral loss with a tiny floor rewarded exactly that: audible "wind").
    """

    def __init__(self, sr, fft_sizes=(4096, 2048, 1024, 512, 256), n_mels=(160, 128, 80, 40, 20), f_min=30.0,
                 eps_per_fft=3e-7):
        super().__init__()
        import torchaudio.functional as AF
        # floor ~ 25th percentile of real GAPS mel magnitudes at each resolution (magnitudes grow with n_fft)
        self.fft_sizes, self.eps_per_fft = fft_sizes, eps_per_fft
        for n, m in zip(fft_sizes, n_mels):
            fb = AF.melscale_fbanks(n // 2 + 1, f_min, sr / 2, m, sr, norm="slaney", mel_scale="slaney")
            self.register_buffer(f"fb{n}", fb)

    def forward(self, x, y, reduce=True):
        total = 0.0
        for n in self.fft_sizes:
            w = torch.hann_window(n, device=x.device)
            fb = getattr(self, f"fb{n}").to(x.device)
            sx = torch.stft(x, n, n // 4, window=w, return_complex=True).abs().transpose(-1, -2) @ fb
            sy = torch.stft(y, n, n // 4, window=w, return_complex=True).abs().transpose(-1, -2) @ fb
            eps = self.eps_per_fft * n
            d = (sx - sy).abs().mean((-1, -2)) + (torch.log(sx + eps) - torch.log(sy + eps)).abs().mean((-1, -2))
            total = total + d
        return total.mean() if reduce else total
