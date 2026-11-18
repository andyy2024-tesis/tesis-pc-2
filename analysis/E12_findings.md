# Experiment 12: final DDSP model, then a neural vocoder that refines it

Plan (approved 2026-11-17 with V3b, stems and the 44.1 kHz phase): `analysis/E12_vocoder_plan.md`.
User's question on approval: why teach the vocoder guitar on real audio instead of DDSP renders of all 270 recordings?
Answer: phase 2 teaches the last step (mel -> waveform) that every mel route needs, and real mels are what V3's output
looks like; DDSP pairs are phase 3. More DDSP pairs need a DDSP trained on those recordings (per-recording parts) with
good note alignment and tab strings: 80 more training-split recordings pass (transcription f >= 0.93, tab matched >= 0.88
like the 20; `analysis/results/e12_more_ids.txt`), so a 100-recording DDSP gives ~5 h of pairs instead of 0.73 h. V1 is
trained on both to test whether pair count limits.

## Final DDSP model (`analysis/results/queue_final`)

Ear verdicts on experiment 11 picked T4adv (attention that looks ahead + contact + adv) and strings on M1; the
recording in every part with a 32-dim embedding; stiffness not measured from recordings. Two versions:

| run | stiffness per guitar | loss (non-adv, step 4000) |
|---|---|---|
| `structured_v12_final` | none (one table per string) | 5.69 |
| `structured_v12_final_learnedB` | learned from the embedding (per string, bounded x0.61-1.65) | 5.71 |
| T4 (`structured_v11_m1_attn_ahead_contact`), for reference | - | 5.73 |

Adversarial fine-tunes (same recipe as T4adv: complex D, 64 ch, D lr x2, 3000 steps, rise penalty 20):

| run | loss | wind (dB) | high clicks in the excerpts |
|---|---|---|---|
| `e6/structured_v12_final_noise_strongD_rise` | 5.81 | -20.6 | 0 |
| `e6/structured_v12_final_learnedB_noise_strongD_rise` | 5.82 | -23.9 | 0 |
| T4adv, for reference | 5.83 | -24.3 | 2 |

**Final DDSP: `e6/structured_v12_final_learnedB_noise_strongD_rise`** (learned stiffness, as the user wanted; loss tie,
less wind than the shared-stiffness version, no high clicks). Its renders of the 20 recordings (mix + parts, aligned
sample by sample with the real audio) are the vocoder pairs: `analysis/results/e12/renders/<run>/`.

A tie (seed spread ~0.05). What the learned version learned: shifts with a spread across guitars of 0.21 (log), as wide
as the measured per-guitar stiffness (0.22), and correlated with it (r = 0.28 over 120 guitar-string pairs): the
embedding found part of the real stiffness differences by itself, through the partial frequencies.

## Phase 1: ceilings, no training

Pretrained BigVGAN-v2 24 kHz on the 85 held-out windows, real mel -> audio vs the real audio: loss 4.24 (the ceiling of
a mel vocoder on this measure; no DDSP model has gone below 5.4). Listening excerpts (9 s), loss vs the real excerpt:

| excerpt | real -> BigVGAN | favourite | favourite -> BigVGAN | T4adv | T4adv -> BigVGAN |
|---|---|---|---|---|---|
| 023 | 4.22 | 5.37 | 5.73 | 5.36 | 5.81 |
| 176 | 4.11 | 7.16 | 7.29 | 5.93 | 6.10 |
| 276 | 4.29 | 6.31 | 6.52 | 6.38 | 6.58 |
| 277 | 4.40 | 6.23 | 6.47 | 6.24 | 6.47 |
| 306 | 4.52 | 6.67 | 6.91 | 6.66 | 6.85 |

Untrained on DDSP input, the vocoder only redraws the DDSP spectrum (+0.1-0.4): it cannot repair what it never saw.

On the 85 held-out windows (`analysis/e12_voc_metrics.py`; 44.1 kHz outputs resampled to 24 kHz and compared with the
real 24 kHz window; the 44.1 kHz input is the raw 48 kHz GAPS file at the same place):

| vocoder (pretrained) | real -> vocoder | final DDSP -> vocoder |
|---|---|---|
| none | 0 | 5.82 |
| BigVGAN-v2 24 kHz | 4.24 | 6.11 |
| BigVGAN-v2 44 kHz | 4.40 | 6.03 |
| DisCoder 44.1 kHz | 4.46 | 6.11 |

Listening page 6 (https://claude.ai/artifact/6du45ZyZdAEwrmTRSJWDrM): the final DDSP next to the favourite and T4adv;
the 24 kHz ceiling; the 44.1 kHz vocoders. The parts of a vocoder version are what went in, what it added and what it
removed, from spectrogram magnitudes: a vocoder keeps the timing (envelope lag < 1.2 ms) but not the waveform's phase
(sample correlation with its input -0.2 to 0.25), so subtracting waveforms would give the whole piece back.

## Phase 2: BigVGAN-v2 24 kHz on real GAPS audio

Memory: BigVGAN's anti-aliased activations (2x upsampled, at every layer) are huge; without help even batch 4 x 32768
samples ran out of 24 GB. With gradient checkpointing of each AMP block (recomputed in the backward pass;
`scripts/e12_vocoder.py`, on by default) batch 8 x 32768 fits (20 GB) at 1.3 s/step, so 40000 steps would take 14 h:
cut to 8000 steps (~3 h), evaluations every 2000. After 100 steps the held-out loss already went 4.24 -> 4.10.
(A first restart was stopped by a queue watcher that matched an old "end phase2" line; the night queue is now one
sequential script, `queue_final/night.sh`: phase 2, then V1 on the 20-recording pairs, then the 100-recording DDSP.)

The 100-recording DDSP needs ~20 GB (vs ~7 GB for 20 recordings: the peak comes from the densest windows of the extra
recordings, not from per-recording data), so it cannot share the GPU with a vocoder run; it goes last.

## V3: shallow diffusion on the mel

First run (`v3_shallow`, DiffSinger's noise prediction, 50% unpaired real batches): at 10000 steps the reverse process
from depth 20 gives 8.93 and from depth 40-80 27-29 (depth 0, the DDSP mel through the pretrained vocoder: 6.09).
Diagnosis: the noise error stays ~0.15 even at heavy noise, where the DDSP condition should make it easy, so the
clean-mel estimate at depth 20 (MSE 0.023 vs the real mel) is already worse than the DDSP mel itself (0.009), and the
error grows along the reverse chain (values up to +-4 at depth 60). Second run `v3_x0res`: the network predicts the
clean mel as the DDSP mel plus a correction (zero-initialised, so it starts as the DDSP mel), reverse step by the DDPM
posterior mean. Then V3b (flow from the DDSP mel; also starts as the identity).

Results (vocoded by the pretrained BigVGAN; identity = the DDSP mel itself, 6.09):

| run | step 10000 | 20000 | 30000 | 40000 |
|---|---|---|---|---|
| V3 noise prediction, depth 20 / 40 / 80 | 8.93 / 27.3 / 28.1 | 8.01 / 10.9 / 29.3 | 7.69 / 9.70 / 25.5 | - |
| V3 x0res (first run), depth 20 / 40 / 80 | 6.28 / 6.34 / 6.39 | (crashed saving: rerun) | | |
| V3b flow (no source noise) | 6.12 | 6.13 | 6.17 | 6.18 |

Why the flow gets slightly worse: its mels are closer to the real ones (mel MSE 0.0125 vs 0.0145 for the DDSP mel) but
smooth: their frame-to-frame variation is 27% of the real mel's (DDSP mel: 90%). Trained on MSE from a fixed start,
the flow learns the average correction, which washes out texture, and the vocoder turns a smooth mel into dull audio.
Fix being tried: start from the DDSP mel plus noise (`--src-noise 0.3`, bridge 0.3) so the flow has to create texture.
Every evaluation now also reports the texture ratio. Caveat on the measure: the spectral loss compares with one specific
recording, so a refiner that adds realistic but different detail is penalised; for these the ear decides.
Also rendered: BigVGAN-v2 44 kHz (from the full-band 48 kHz excerpt and from the DDSP upsampled) and DisCoder (44.1 kHz,
music-trained). `analysis/e12_vocode.py`, `analysis/e12_discoder.py` (own environment in external/discoder/.venv).
