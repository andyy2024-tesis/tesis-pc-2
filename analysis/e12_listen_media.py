"""Experiment 12: listening-page media for vocoder versions (and any excerpt at its own sample rate, e.g. 44.1 kHz).

For each version given (a file label in analysis/results/slides/audio/<recording>/<label>.wav):
- AAC at the file's own rate in listen/audio/<recording>/<label>.mp4, at the recording's shared page-1 gain (as
  make_listening_media.py --only), so levels compare with every other version;
- if it is a vocoder version (in build_listening_catalog.VOC with an input), three parts: "entrada" (what went into
  the vocoder, resampled to the output rate), "anadido" (what it added: the spectrogram magnitude where the output is
  louder than the input, with the output's phase) and "quitado" (what it removed: where the input is louder, with the
  input's phase). A vocoder keeps the timing but not the phase of the waveform (sample correlation output/input
  -0.2 to 0.25 on these excerpts, envelope lag under 1.2 ms, no consistent sign), so a plain waveform difference would sound like the whole
  piece again. Stems use render_stems.py's loudness rule; true levels go into the page's STEMS table.
Usage: uv run python analysis/e12_listen_media.py original44k voc-bigvgan24-original ...
"""
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, "analysis")
sys.path.insert(0, "scripts")
from build_listening_catalog import VOC  # noqa: E402
from make_listening_media import FFMPEG, LATER, VERSIONS  # noqa: E402
from render_stems import AUDIO, LISTEN, LISTEN_IDS, OFF_DB, rms, update_page  # noqa: E402

SRC = {e["key"]: e["src"] for e in VOC if e["src"]}


def to_aac(x, sr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav")
    sf.write(tmp, x, sr)
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", str(tmp), "-c:a", "aac", "-b:a", "160k", str(path)],
                   check=True)
    tmp.unlink()


def shared_gain(rid):
    """The gain the page-1 files of this recording were encoded with (make_listening_media.py --only)."""
    return 0.89 / max(np.abs(sf.read(AUDIO / rid / f"{v}.wav", dtype="float32")[0]).max()
                      for v in VERSIONS if v not in LATER and not v.endswith("-noise"))


def env_lag(a, b, max_lag=600):
    """Shift (samples) that best lines up the amplitude envelopes of a and b (phase-blind)."""
    n = min(len(a), len(b))
    k = np.ones(120) / 120
    ea, eb = np.convolve(np.abs(a[:n]), k, "same"), np.convolve(np.abs(b[:n]), k, "same")
    ea, eb = ea - ea.mean(), eb - eb.mean()
    c = [np.dot(ea[max(t, 0):n + min(t, 0)], eb[max(-t, 0):n - max(t, 0)]) for t in range(-max_lag, max_lag + 1, 3)]
    return int(np.argmax(c)) * 3 - max_lag


def spectral_diff(y, x, sr):
    """What the vocoder added (|Y| above |X|, output phase) and removed (|X| above |Y|, input phase)."""
    nfft = 2048 if sr <= 24000 else 4096
    win = torch.hann_window(nfft)
    st = lambda z: torch.stft(torch.as_tensor(z), nfft, nfft // 4, window=win, return_complex=True)
    ist = lambda Z: torch.istft(Z, nfft, nfft // 4, window=win, length=len(y)).numpy()
    Y, X = st(y), st(x)
    added = ist(torch.relu(Y.abs() - X.abs()) * torch.exp(1j * Y.angle()))
    removed = ist(torch.relu(X.abs() - Y.abs()) * torch.exp(1j * X.angle()))
    return added, removed


def main():
    for v in sys.argv[1:]:
        entry = {}
        for rid in LISTEN_IDS:
            g = shared_gain(rid)
            y, sr = sf.read(AUDIO / rid / f"{v}.wav", dtype="float32")
            peak = np.abs(y * g).max()
            if peak > 1.0:
                print(f"  {rid} {v}: peak {peak:.2f} at the shared gain, clipped", flush=True)
            to_aac(np.clip(y * g, -1, 1), sr, LISTEN / "audio" / rid / f"{v}.mp4")
            if v not in SRC:
                continue
            x, sx = sf.read(AUDIO / rid / f"{SRC[v]}.wav", dtype="float32")
            if sx != sr:
                x = torchaudio.functional.resample(torch.as_tensor(x)[None], sx, sr)[0].numpy()
            d = env_lag(y, x)
            assert abs(d) <= sr // 500, f"{rid} {v}: vocoder output is {d} samples off its input"  # within 2 ms
            n = min(len(y), len(x))
            mix, x = y[:n], x[:n]
            added, removed = spectral_diff(mix, x, sr)
            parts = {"entrada": x, "anadido": added, "quitado": removed}
            levels = {}
            for name, p in parts.items():
                levels[name] = round(20 * np.log10(rms(p) / rms(mix)), 1)
                if levels[name] < OFF_DB:
                    continue
                gg = min(rms(mix) / rms(p) * 0.89 / np.abs(mix).max(), 0.89 / np.abs(p).max())
                to_aac(np.clip(p * gg, -1, 1), sr, LISTEN / "audio" / rid / "stems" / v / f"{name}.mp4")
            entry[rid] = levels
        if entry:
            update_page({v: entry})
        print("done", v, entry, flush=True)


if __name__ == "__main__":
    main()
