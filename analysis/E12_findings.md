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
Also rendered: BigVGAN-v2 44 kHz (from the full-band 48 kHz excerpt and from the DDSP upsampled) and DisCoder (44.1 kHz,
music-trained). `analysis/e12_vocode.py`, `analysis/e12_discoder.py` (own environment in external/discoder/.venv).
