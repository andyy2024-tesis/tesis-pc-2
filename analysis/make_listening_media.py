"""Media for the listening page and the deck, from the slide excerpts (analysis/render_slide_audio.py).

- Listening page: one AAC file per (recording, version), 9 s, in analysis/results/slides/listen/audio/<recording>/.
- Deck: one short spectrogram video (+ still) per (recording, version) in DECK_CLIPS, in analysis/results/slides/clips/.
All versions of one recording share one gain (so loudness and noise levels are comparable); the catch-all noise alone
("-noise") is boosted by BOOST_DB so it can be heard.
Usage: uv run python analysis/make_listening_media.py
"""
import subprocess

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image
from scipy.signal import stft

from make_slide_videos import AUDIO, BOOST_DB, FPS, H, HEAD, LEFT, RECS, SR, W, frame_base, spec_image
from tesis.data.gaps import REPO

OUT = REPO / "analysis" / "results" / "slides"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VERSIONS = {
    "original": "Original", "ddsp_v2": "DDSP", "structured_v2": "Estructurado",
    "structured_v2_bodyir": "Estructurado + cuerpo", "ddsp_v2_nonoise": "DDSP sin ruido",
    "structured_v2_bodyir_nonoise": "Estructurado + cuerpo, sin ruido", "structured_v4_body": "Sin reverb",
    "structured_v4_body_reverb": "Con reverb", "ddsp_v5_audio": "DDSP + audio",
    "structured_v5_body_audio": "Estructurado + audio (loss lineal, con ruido)",
    "structured_v5_body_audio-noise": "Solo el ruido residual (+20 dB)",
    "structured_v6_mel_noise": "Loss mel, con ruido", "structured_v6_mel_nonoise": "Loss mel, sin ruido",
    "e6-structured_v5_body_audio_nonoise_strongD": "Adversarial, 3000 pasos",
    "adv9k-step2000": "Adversarial, 2000 pasos", "e6-structured_v5_body_audio_nonoise_strongD_9k": "Adversarial, 9000 pasos",
    "e6-structured_v5_body_audio_noise_strongD": "Adversarial con ruido, 3000 pasos",
    "e6-structured_v5_body_audio_noise_strongD-noise": "Adversarial con ruido: solo el ruido (+20 dB)",
    "e6-structured_v5_body_audio_noise_strongD_9k": "Adversarial con ruido, 9000 pasos",
    "e6-structured_v5_body_audio_noise_strongD_9k-noise": "Adv. con ruido 9000: solo el ruido (+20 dB)",
    "e6-structured_v5_body_audio_noise_strongD_linonly": "Adversarial con ruido, loss solo lineal",
    "e6-structured_v5_body_audio_noise_strongD_linonly-noise": "Adv. loss solo lineal: solo el ruido (+20 dB)",
    "e6-structured_v5_body_audio_noise_strongD_both": "Adversarial con ruido, discriminator complejo + log",
    "e6-structured_v5_body_audio_noise_strongD_both-noise": "Discriminator complejo + log: solo el ruido (+20 dB)",
    "structured_v5_body_audio-nonoise": "Base con el ruido apagado",
    "structured_v8_bodyhi": "Cuerpo hasta 6 kHz", "structured_v8_bodyhi-nonoise": "Cuerpo hasta 6 kHz, ruido apagado",
    "structured_v8_twodecay": "Dos decaimientos", "structured_v8_twodecay-nonoise": "Dos decaimientos, ruido apagado",
    "structured_v8_click": "Clic de uña y dedo", "structured_v8_click-nonoise": "Clic, ruido apagado",
    "structured_v8_symp": "Resonancia simpática", "structured_v8_squeak": "Squeak",
    "structured_v8_combo": "Combinado", "structured_v8_combo-nonoise": "Combinado, ruido apagado",
    "e6-structured_v5_body_audio_noise_strongD_logmag": "Adversarial con ruido, discriminator solo log",
    "e6-structured_v5_body_audio_noise_strongD_logmag-noise": "Discriminator solo log: solo el ruido (+20 dB)",
    "e6-structured_v8_combo_noise_strongD_both": "Combinado + adversarial con ruido",
    "e6-structured_v8_combo_noise_strongD_both-noise": "Combinado + adversarial: solo el ruido (+20 dB)",
    "structured_v8_combo_nonoise": "Combinado entrenado sin ruido",
    "e6-structured_v5_body_audio_noise_strongD-nonoise": "Adv. con ruido 3000, ruido apagado",
    "e6-structured_v5_body_audio_noise_strongD_9k-nonoise": "Adv. con ruido 9000, ruido apagado",
    "e6-structured_v5_body_audio_noise_strongD_linonly-nonoise": "Adv. loss solo lineal, ruido apagado",
    "structured_v5_gated": "Base, ruido solo en el ataque", "structured_v5_gated-noise": "Base, ruido en el ataque (+20 dB)",
    "structured_v8_gated": "Combinado, ruido solo en el ataque",
    "structured_v8_gated-noise": "Combinado, ruido en el ataque (+20 dB)",
    "e6-structured_v5_gated_noise_strongD": "Base, ruido en el ataque + adversarial",
    "e6-structured_v5_gated_noise_strongD-noise": "Base, ruido en el ataque + adversarial: el ruido (+20 dB)",
    "e6-structured_v8_gated_noise_strongD": "Combinado, ruido en el ataque + adversarial",
    "e6-structured_v8_gated_noise_strongD-noise": "Combinado, ruido en el ataque + adversarial: el ruido (+20 dB)",
    "structured_v9_timbre": "Timbre de cada ataque", "e6-structured_v9_timbre_noise_strongD": "Timbre + adversarial",
    "e6-structured_v9_timbre32_noise_strongD": "Timbre 32 armónicos + adversarial",
    "structured_v9_timbre32": "Timbre 32 armónicos",
    "e6-structured_v9_timbre32_noise_strongD_9k": "Timbre 32 + adversarial, 9000 pasos",
    "structured_v9_timbre32late": "Timbre 32 + lectura tardía",
    "structured_v9_t32_simple": "Timbre 32 simplificado (sin dos decaimientos ni clic)",
}
DECK_CLIPS = {
    "023_Ypswc": ["original", "ddsp_v2", "ddsp_v2_nonoise", "structured_v2_bodyir_nonoise"],
    "176_mM1wc": ["original", "structured_v5_body_audio", "structured_v5_body_audio-noise", "adv9k-step2000",
                  "e6-structured_v5_body_audio_nonoise_strongD_9k"],
    "306_7c1wc": ["original", "structured_v5_body_audio", "structured_v6_mel_noise",
                  "e6-structured_v5_body_audio_nonoise_strongD"],
}


def load(rid):
    """All versions of one recording at one shared gain."""
    xs = {v: sf.read(AUDIO / rid / f"{v}.wav", dtype="float32")[0] for v in VERSIONS}
    gain = 0.89 / max(np.abs(x).max() for v, x in xs.items() if not v.endswith("-noise"))
    return {v: np.clip(x * gain * (10 ** (BOOST_DB / 20) if v.endswith("-noise") else 1), -1, 1) for v, x in xs.items()}


def to_aac(x, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav")
    sf.write(tmp, x, SR)
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", str(tmp), "-c:a", "aac", "-b:a", "128k", str(path)],
                   check=True)
    tmp.unlink()


def clip(rid, v, x, vmax):
    """Spectrogram video of one version with a moving playhead, plus its still."""
    sw, sh = W - LEFT - 12, H - HEAD - 44
    base = frame_base(rid, [], None, spec_image(x, vmax, sh, sw), VERSIONS[v])
    path = OUT / "clips" / f"{rid}_{v}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = path.with_suffix(".wav")
    sf.write(wav, x, SR)
    w = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                    quality=None, bitrate="600k", audio_path=str(wav), audio_codec="aac",
                                    output_params=["-shortest", "-movflags", "+faststart", "-b:a", "128k"],
                                    macro_block_size=8, ffmpeg_log_level="error")
    w.send(None)
    dur = len(x) / SR
    for k in range(int(dur * FPS)):
        fr = base.copy()
        px = LEFT + min(int(k / FPS / dur * sw), sw - 1)
        fr[HEAD + 44:, px:px + 3] = (255, 255, 255)
        w.send(np.ascontiguousarray(fr))
    w.close()
    wav.unlink()
    Image.fromarray(base).convert("RGB").save(path.with_suffix(".jpg"), quality=85)


def main():
    for rid in RECS:
        xs = load(rid)
        for v, x in xs.items():
            to_aac(x, OUT / "listen" / "audio" / rid / f"{v}.mp4")
        _, _, Z = stft(xs["original"], SR, nperseg=2048, noverlap=2048 - 240)
        vmax = 20 * np.log10(np.abs(Z).max())
        for v in DECK_CLIPS.get(rid, []):
            clip(rid, v, xs[v], vmax)
        print(rid, flush=True)


if __name__ == "__main__":
    main()
