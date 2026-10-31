"""Multi-resolution STFT discriminator (as in neural audio codecs, e.g. EnCodec's MS-STFT discriminator).

Each sub-discriminator looks at the complex STFT (real and imaginary parts as channels) at one resolution with a
small 2D conv stack, so it can judge fine time (attacks) and frequency (partials, hiss between them) detail.
Hinge loss; feature-matching loss on the intermediate activations.

kind="logmag" sub-discriminators see the log magnitude instead: on the complex (linear) view, quiet passages are tiny
next to loud partials, so background noise ("wind") barely registers; on a log scale it is as visible as the notes.
kind="both" uses the two views at every resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class STFTDisc(nn.Module):
    def __init__(self, n_fft, ch=32, logmag=False):
        super().__init__()
        self.n_fft, self.logmag = n_fft, logmag
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1 if logmag else 2, ch, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 3), padding=(1, 1))),
        ])
        self.out = weight_norm(nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):  # x: (B, T)
        X = torch.stft(x, self.n_fft, self.n_fft // 4, window=torch.hann_window(self.n_fft, device=x.device),
                       return_complex=True)  # (B, F, T)
        if self.logmag:  # ~-11.5 (floor) to ~+5 -> roughly -2 to +2
            h = ((torch.log(X.abs() + 1e-5) + 4) / 4)[:, None].transpose(2, 3)  # (B, 1, T, F)
        else:
            h = torch.stack([X.real, X.imag], 1).transpose(2, 3)  # (B, 2, T, F)
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
            feats.append(h)
        return self.out(h), feats


class MultiResDiscriminator(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048), ch=32, kind="complex"):
        super().__init__()
        views = {"complex": [False], "logmag": [True], "both": [False, True]}[kind]
        self.discs = nn.ModuleList([STFTDisc(n, ch, lm) for n in fft_sizes for lm in views])

    def forward(self, x):
        return [d(x) for d in self.discs]


def d_hinge_loss(real_out, fake_out):
    return sum(F.relu(1 - r[0]).mean() + F.relu(1 + f[0]).mean() for r, f in zip(real_out, fake_out)) / len(real_out)


def g_adv_loss(fake_out):
    return sum(F.relu(1 - f[0]).mean() for f in fake_out) / len(fake_out)


def feature_matching(real_out, fake_out):
    """L1 between discriminator activations, each layer normalized by the real activations' mean magnitude."""
    loss, n = 0.0, 0
    for (_, rf), (_, ff) in zip(real_out, fake_out):
        for r, f in zip(rf, ff):
            loss = loss + (r.detach() - f).abs().mean() / (r.detach().abs().mean() + 1e-5)
            n += 1
    return loss / n
