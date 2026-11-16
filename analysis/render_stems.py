"""Render every part of a model separately (stems) for the listening page, on the same 9 s excerpts.

Same windows, crossfades and noise seeds as analysis/render_slide_audio.py, so the parts add up to the mix that the
page already plays. For each (recording, model, part):
- raw float wav in analysis/results/slides/audio/<recording>/stems/<label>/<part>.wav (true level);
- AAC in analysis/results/slides/listen/audio/<recording>/stems/<label>/<part>.mp4, brought to the loudness of the
  mix (RMS) so quiet parts can be heard, with the peak capped at 0.89;
- parts below OFF_DB get no file (the page shows them as off);
- its true level (RMS relative to the mix, dB) goes into the STEMS table of the listening page (between the
  /*STEMS*/ markers of index.html).
Usage: uv run python analysis/render_stems.py <run> [<run> ...]   (e.g. e6/structured_v9_base_t32_noise_strongD)
"""
import json
import re
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "analysis")
from compare_runs import load  # noqa: E402
from e6_adversarial import EXCERPT_S, L, LISTEN_IDS, SR, XFADE_S  # noqa: E402
from tesis.data.gaps import REPO  # noqa: E402

AUDIO = REPO / "analysis" / "results" / "slides" / "audio"
LISTEN = REPO / "analysis" / "results" / "slides" / "listen"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
OFF_DB = -80.0  # parts quieter than this (RMS relative to the mix) are reported as off
ORDER = ["strings", "sympathetic", "burst", "click", "contact", "body", "squeak", "reverb", "floor", "noise"]


@torch.no_grad()
def render_parts(model, rec, start, dur=EXCERPT_S, seed=0):
    """Like e6_adversarial.render_long, but returns every part: {part: (n,) array}."""
    hop = L - int(XFADE_S * SR)
    n = int(dur * SR)
    ramp = np.ones(L, np.float32)
    xf = int(XFADE_S * SR)
    ramp[:xf] = np.linspace(0, 1, xf)
    ramp[-xf:] = np.linspace(1, 0, xf)
    out, wsum = {}, np.zeros(n + L, np.float32)
    for i, a in enumerate(range(0, n, hop)):
        w0 = start + a
        if w0 + L > len(rec.y):
            break
        _, parts = model(rec, int(w0), noise_seed=seed + i)
        for k, v in parts.items():
            out.setdefault(k, np.zeros(n + L, np.float32))[a:a + L] += v.cpu().numpy() * ramp
        wsum[a:a + L] += ramp
    m = wsum > 1e-3
    k = int(min(n, len(rec.y) - start))
    res = {}
    for name, x in out.items():
        x[m] /= wsum[m]
        res[name] = x[:k]
    return res


def to_aac(x, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav")
    sf.write(tmp, x, SR)
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", str(tmp), "-c:a", "aac", "-b:a", "128k", str(path)],
                   check=True)
    tmp.unlink()


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def update_page(table):
    """Merge this run's entries into the STEMS table of index.html."""
    page = LISTEN / "index.html"
    html = page.read_text()
    m = re.search(r"/\*STEMS\*/(.*?)/\*END STEMS\*/", html, re.S)
    cur = json.loads(m.group(1).strip().removeprefix("const STEMS = ").rstrip(";")) if m else {}
    cur.update(table)
    block = "/*STEMS*/const STEMS = " + json.dumps(cur, ensure_ascii=False, separators=(",", ":")) + ";/*END STEMS*/"
    html = html[:m.start()] + block + html[m.end():] if m else html.replace("<script>\n", "<script>\n" + block + "\n", 1)
    page.write_text(html)


def main():
    ids = pd.read_csv(REPO / "analysis" / "results" / "e4_recordings.csv").id.tolist()
    for run in sys.argv[1:]:
        m, recs, splits = load(run, ids)
        label = run.replace("/", "-")
        entry = {}
        for rid in LISTEN_IDS:
            r = ids.index(rid)
            parts = render_parts(m, recs[r], int(splits[r][1][0]))
            mix = sum(parts.values())
            # the page's mix of this model: same render, at the recording's shared gain (make_listening_media.load)
            ref = AUDIO / rid / f"{label}.wav"
            if ref.exists():
                y = sf.read(ref, dtype="float32")[0][:len(mix)]
                assert np.abs(y - mix[:len(y)]).max() < 1e-3 * np.abs(y).max() + 1e-6, f"{rid}: parts do not add up"
            levels = {}
            for name in sorted(parts, key=lambda k: ORDER.index(k) if k in ORDER else 99):
                x = parts[name]
                if rms(x) < 1e-12:  # part absent in this excerpt (e.g. no notes near it): skip
                    continue
                levels[name] = round(20 * np.log10(rms(x) / rms(mix)), 1)
                if levels[name] < OFF_DB:  # switched off by training: nothing to hear, no file
                    continue
                (AUDIO / rid / "stems" / label).mkdir(parents=True, exist_ok=True)
                sf.write(AUDIO / rid / "stems" / label / f"{name}.wav", x.astype(np.float32), SR, subtype="FLOAT")
                # loudness of the mix played at full scale, unless that clips
                g = min(rms(mix) / rms(x) * 0.89 / np.abs(mix).max(), 0.89 / np.abs(x).max())
                to_aac(np.clip(x * g, -1, 1), LISTEN / "audio" / rid / "stems" / label / f"{name}.mp4")
            entry[rid] = levels
        update_page({label: entry})
        print("done", run, json.dumps(entry), flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
