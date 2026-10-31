"""Turn the slide excerpts (analysis/render_slide_audio.py) into short spectrogram videos with sound.

One video per (comparison, recording): the versions play one after another (6 s each), the spectrogram of the version
playing is shown with a moving playhead, and the header names it. Also writes a still (poster) per video.
All versions of one video share one gain and one color scale, so loudness and noise levels stay comparable.
Usage: uv run python analysis/make_slide_videos.py
"""

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import stft

from tesis.data.gaps import REPO

SR = 24000
SEG_S, GAP_S, FPS = 6.0, 0.5, 20
W, H = 1200, 480
HEAD, CHIPS, LEFT = 64, 44, 76  # header height, chip row height, frequency-label margin
F_MIN, F_MAX, DB_RANGE = 40.0, 12000.0, 80.0
AUDIO = REPO / "analysis" / "results" / "slides" / "audio"
OUT = REPO / "analysis" / "results" / "slides" / "video"
RECS = {"023_Ypswc": "023", "176_mM1wc": "176", "276_jw1wc": "276", "277_Pw1wc": "277", "306_7c1wc": "306"}
BOOST_DB = 20.0

ORIG = ("original", "Original", "Original (grabación real)")
# comparison -> list of (file label, short chip text, full header text)
SEQS = {
    "red": [ORIG, ("ddsp_v2", "DDSP", "DDSP (armónicos + ruido)"),
            ("structured_v2", "Estructurado", "Estructurado (sin cuerpo)"),
            ("structured_v2_bodyir", "+ cuerpo", "Estructurado + respuesta del cuerpo")],
    "sinruido": [ORIG, ("ddsp_v2_nonoise", "DDSP sin ruido", "DDSP sin ruido residual"),
                 ("structured_v2_bodyir_nonoise", "Estruct. sin ruido", "Estructurado + cuerpo, sin ruido residual")],
    "reverb": [ORIG, ("structured_v4_body", "Sin reverb", "Estructurado + cuerpo, sin reverb"),
               ("structured_v4_body_reverb", "Con reverb", "Estructurado + cuerpo + reverb de sala")],
    "audio": [ORIG, ("ddsp_v5_audio", "DDSP + audio", "DDSP + features de audio"),
              ("structured_v5_body_audio", "Estruct. + audio", "Estructurado + cuerpo + features de audio")],
    "viento": [ORIG, ("structured_v5_body_audio", "Modelo completo", "Estructurado + audio: todo junto"),
               ("structured_v5_body_audio-noise", "Solo ruido +20 dB", "Solo el ruido residual (amplificado +20 dB)")],
    "mel": [ORIG, ("structured_v5_body_audio", "Lineal + ruido", "Loss lineal, con ruido (antes)"),
            ("structured_v6_mel_noise", "Mel + ruido", "Loss mel, con ruido"),
            ("structured_v6_mel_nonoise", "Mel sin ruido", "Loss mel, sin ruido"),
            ("e6-structured_v5_body_audio_nonoise_strongD", "Adversarial", "Adversarial, sin ruido (3000 pasos)")],
    "adv9k": [ORIG, ("adv9k-step2000", "2000 pasos", "Adversarial: 2000 pasos"),
              ("e6-structured_v5_body_audio_nonoise_strongD_9k", "9000 pasos", "Adversarial: 9000 pasos (final)")],
}

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, INK, MUTED, ACCENT = (20, 24, 31), (240, 237, 230), (150, 156, 166), (233, 148, 58)


def spec_image(x, vmax, height, width):
    """Log-frequency spectrogram as an RGB array (height, width, 3)."""
    f, _, Z = stft(x, SR, nperseg=2048, noverlap=2048 - 240)
    db = 20 * np.log10(np.abs(Z) + 1e-9)
    rows = np.geomspace(F_MAX, F_MIN, height)  # top row = high frequency
    img = np.stack([np.interp(rows, f, db[:, j]) for j in range(db.shape[1])], 1)
    cols = np.linspace(0, img.shape[1] - 1, width)
    img = np.stack([np.interp(cols, np.arange(img.shape[1]), r) for r in img])
    v = np.clip((img - (vmax - DB_RANGE)) / DB_RANGE, 0, 1)
    return (cm.magma(v)[..., :3] * 255).astype(np.uint8)


def frame_base(rid, segs, i, spec, header):
    """Frame without playhead: header, segment chips, frequency labels, spectrogram."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    fb, f, fs = ImageFont.truetype(FONT_B, 34), ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT, 20)
    d.text((20, 12), RECS[rid], font=fb, fill=ACCENT)
    d.text((100, 12), header, font=f, fill=INK)
    if i is not None:
        t = f"{i + 1} / {len(segs)}"
        d.text((W - 20 - d.textlength(t, font=f), 12), t, font=f, fill=MUTED)
    x = 20
    for j, (_, short, _) in enumerate(segs):
        w = d.textlength(short, font=fs) + 24
        on = j == i
        d.rounded_rectangle((x, HEAD + 4, x + w, HEAD + CHIPS - 6), 8, fill=ACCENT if on else (44, 50, 60))
        d.text((x + 12, HEAD + 9), short, font=fs, fill=BG if on else INK)
        x += w + 10
    top = HEAD + CHIPS
    im.paste(Image.fromarray(spec), (LEFT, top))
    hgt = H - top
    for fr, lab in [(100, "100 Hz"), (1000, "1 kHz"), (10000, "10 kHz")]:
        y = top + int(np.log(F_MAX / fr) / np.log(F_MAX / F_MIN) * (hgt - 1))
        d.text((6, y - 11), lab, font=ImageFont.truetype(FONT, 16), fill=MUTED)
        d.line((LEFT - 6, y, LEFT, y), fill=MUTED, width=2)
    return np.asarray(im)


def make(seq, rid):
    segs = SEQS[seq]
    n = int(SEG_S * SR)
    xs = []
    for label, _, _ in segs:
        x = sf.read(AUDIO / rid / f"{label}.wav", dtype="float32")[0][:n]
        if label.endswith("-noise"):
            x = x * 10 ** (BOOST_DB / 20)
        xs.append(np.pad(x, (0, n - len(x))))
    gain = 0.89 / max(np.abs(x).max() for (lab, _, _), x in zip(segs, xs) if not lab.endswith("-noise"))
    xs = [np.clip(x * gain, -1, 1) for x in xs]
    _, _, Z = stft(xs[0], SR, nperseg=2048, noverlap=2048 - 240)
    vmax = 20 * np.log10(np.abs(Z).max())
    sw, sh = W - LEFT - 12, H - HEAD - CHIPS
    gap = np.zeros(int(GAP_S * SR), np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / f"_{seq}_{rid}.wav"
    sf.write(wav, np.concatenate([np.concatenate([x, gap]) for x in xs]), SR)
    path = OUT / f"{seq}_{rid}.mp4"
    w = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                    quality=None, bitrate="600k", audio_path=str(wav), audio_codec="aac",
                                    output_params=["-shortest", "-movflags", "+faststart", "-b:a", "128k"],
                                    macro_block_size=8, ffmpeg_log_level="error")
    w.send(None)
    seg_frames = int((SEG_S + GAP_S) * FPS)
    for i, x in enumerate(xs):
        base = frame_base(rid, segs, i, spec_image(x, vmax, sh, sw), segs[i][2])
        for k in range(seg_frames):
            fr = base.copy()
            px = LEFT + min(int(k / FPS / SEG_S * sw), sw - 1)
            fr[HEAD + CHIPS:, px:px + 3] = (255, 255, 255)
            w.send(np.ascontiguousarray(fr))
    w.close()
    wav.unlink()
    poster = frame_base(rid, segs, None, spec_image(xs[0], vmax, sh, sw), "▶  clic para escuchar")
    Image.fromarray(poster).save(OUT / f"{seq}_{rid}.png")
    return path


def main():
    for seq in SEQS:
        for rid in RECS:
            p = make(seq, rid)
            print(p.name, f"{p.stat().st_size / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
