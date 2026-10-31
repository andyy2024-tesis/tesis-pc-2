# Experiment 6: adversarial fine-tuning (2026-10-27)

Motivation (user listening): with audio features all samples sound better, but 176 still sounds like a cheap MIDI
keyboard and there is a wind-like background noise. Confirmed by ear and measurement: the wind is the catch-all
filtered noise (about -24 dB relative to the original in the quietest passages); noise floor and body are inaudible.
Likely cause: the log-magnitude spectral loss rewards filling quiet spectral regions with noise.

Code: `src/tesis/model/discriminator.py` (multi-resolution complex-STFT discriminator, FFT 512/1024/2048, hinge loss,
feature matching), `scripts/e6_adversarial.py` (fine-tunes an experiment-4 model: spectral loss + 1.0 * adversarial
+ 2.0 * feature matching, Adam 1e-4, betas 0.5/0.9, 200 discriminator-only warm-up steps, 3000 steps x 8 windows).
Listening: ~9 s held-out excerpts (consecutive windows with 0.5 s crossfades), `analysis/results/e6/<run>/audio/`,
files `_1_original`, `_2_before` (structured + body + audio features, with noise), `_3_after_adversarial`.

| Run | Held-out spectral loss | Noise in quiet passages |
|---|---|---|
| before (structured + body + audio features) | 6.00 | -25.2 dB |
| adversarial, noise kept, discriminator 32 ch | 6.05 | -22.9 dB (louder) |
| adversarial, noise removed, discriminator 64 ch, 2x lr | 6.64 | none |

- First attempt: the discriminator stayed near chance (hinge loss ~1.9) and the wind got slightly louder.
- Second attempt removes the catch-all noise (round 2 showed the structured model doesn't need it) and strengthens
  the discriminator (loss ~1.6-1.7, discriminating). Removing the noise from the trained model costs 7.83 at first;
  it recovers to 6.64 in 3000 steps at the low fine-tuning learning rate (still improving). The spectral loss is
  expected to be worse; this run is judged by ear.
- The 3 s "023 restarts" excerpt: not corrupted (3.0 s, no repetition); short excerpts of a repetitive samba sound
  like they loop. Excerpts are ~9 s now.

# User listening of the adversarial no-noise run (2026-10-27)

- No wind in any sample; no "cheap MIDI" character anymore. 306 sounds realistic ("super good").
- Others (176 most, 023 some) have a "robotic flapping" sound, and the harmonics sound less polished than before.
  Measured (`analysis/flapping_check.py`, fast 4-50 Hz loudness wobble vs 0-4 Hz): 176 adversarial -1.7 dB vs
  original -3.4 and before -3.0 (more wobble); 277 -1.3 vs -2.2. The model had not recovered from losing the noise
  (loss 6.64 vs 6.00, still improving).

# Experiment 7: mel loss (2026-10-27)

`MultiScaleMelLoss` (src/tesis/synth/dsp.py): FFT 4096-256, 160-20 mel bands, L1 on mel magnitude + L1 on log mel with
a floor at ~the 25th percentile of real GAPS mel magnitudes per resolution (so near-silence earns little). Trained from
scratch (structured + body + audio features, 4000 steps), compared with `analysis/compare_runs.py`
(listening: analysis/results/compare/mel_vs_mss/, files numbered in table order, 0 = original).

| Model | Held-out loss (linear) | Held-out loss (mel) | Wind (noise in quiet passages) | Flapping vs original |
|---|---|---|---|---|
| 1 linear loss, with noise (best of round 4) | 6.00 | 2.28 | -25.2 dB | -0.2 dB |
| 2 mel loss, with noise | 6.07 | **2.20** | **-19.8 dB (louder)** | -0.1 dB |
| 3 mel loss, no noise | 6.18 | 2.24 | none | +0.3 dB |
| 4 adversarial, no noise (exp. 6) | 6.64 | 2.50 | none | +0.5 dB |

- The mel loss alone does not remove the wind: with the catch-all noise available it uses even more (-19.8 dB).
  Removing the noise is what removes the wind.
- Mel + no noise is a clean candidate without adversarial training: no wind, little flapping, losses close to the best.
- Next: adversarial fine-tuning starting from (3), with a penalty on frame-to-frame changes of the amplitude
  correction (delta_tv) against flapping, and longer.

## Experiment 6b: same adversarial run, restarted and trained 9000 steps

Same settings as `structured_v5_body_audio_nonoise_strongD` (no noise, 64-channel discriminator, 2x D lr), restarted
from scratch with full checkpoints (generator, discriminator, optimizers, RNG; `--resume` continues exactly) and
listening clips every 2000 steps (`analysis/results/e6/structured_v5_body_audio_nonoise_strongD_9k/audio/step*/`).

Listening (user): adversarial was the best-sounding model on all five recordings despite the worst spectral loss;
spectral losses do not track perceived realism, so listening decides. Mel without noise had bad harmonics; mel with
noise had cleaner harmonics than without, i.e. the catch-all noise seems to cover something real, not only "wind".

| step | held-out loss | flapping vs original (dB; 023 176 276 277 306 / mean) |
|---|---|---|
| 1000 | 6.91 | |
| 2000 | 6.79 | +0.25 +1.66 -0.48 +0.97 +0.73 / +0.63 |
| 3000 | 6.82 (first run: 6.64; GPU non-determinism) | |
| 4000 | 6.82 | +0.36 +1.63 -0.30 +1.19 +0.50 / +0.68 |
| 6000 | 6.97 | +0.49 +1.84 -0.29 +1.08 +0.42 / +0.71 |
| 8000 | 6.73 | +0.00 +1.32 -0.56 +0.64 +0.09 / +0.30 |
| 9000 | 6.61 | +0.09 +1.12 -0.61 +0.94 +0.39 / +0.38 |

Loss plateaus 2000-6000 and improves only slightly by 9000. Flapping drops after 6000 (mean +0.7 -> +0.3/+0.4 dB) but
176 stays clearly above the original (+1.1 dB).

Listening (user, 2000 vs 9000 steps): 176 sounds equally ugly after 9000 steps; the others improve a little, but
the artifacts are still present and very ugly. Consistent with the loss barely moving: more steps do not fix it.

## Diagnosis after 6b (2026-10-28)

- Data: 20 recordings = 51 min, 85 % for training (43.5 min). GAPS processed: 300 recordings, ~16.6 h (we use ~5 %).
  Training spectral loss (~6.1-7.0, noisy) is about the same as held-out (6.6-6.8): the generator is not memorizing,
  so more data is unlikely to change the synthesizer's sound by itself. The discriminator loss slowly falls
  (1.68 -> ~1.1-1.4), a hint it may be memorizing 44 min of real audio; more data may matter for it later.
- The generator objective is already hybrid: spectral loss + 1.0 adversarial + 2.0 feature matching.
- Hypothesis: the discriminator asks for the small random variation of a real guitar; without a noise source the only
  fast knob is the harmonic amplitudes, so the model wobbles them (flapping). Evidence: models with noise have no
  extra flapping (-0.2, -0.1 dB); every no-noise model has it (mel +0.3, adversarial +0.5 / +0.4 dB); mel with noise
  had cleaner harmonics than without.
- Test (Experiment 6c): adversarial fine-tuning from the with-noise model (noise kept), 64-channel discriminator, 2x D
  lr, 3000 steps, clips every 1000. The only earlier with-noise run used the weak 32-channel discriminator (near
  chance). Reading: flapping gone -> hypothesis confirmed (then deal with wind if it returns); flapping stays -> not
  the noise, go to an anti-flapping penalty or a periodicity-aware discriminator.

## Experiment 6c: adversarial with the noise kept (2026-10-28)
`analysis/results/e6/structured_v5_body_audio_noise_strongD/` (log: analysis/results/e6_structured_noise_strongD_log.txt).
Starts at 6.00 (no "noise removal" shock). Flapping vs original (dB; 023 176 276 277 306 / mean, from
`analysis/flapping_check.py` on the listening excerpts, rounded):

| model | flapping | held-out loss | wind (quiet passages) |
|---|---|---|---|
| before (with noise, no adversarial) | -0.1 +0.4 -1.2 +0.4 -0.2 / -0.1 | 6.00 | -25.2 dB |
| adversarial no noise, 2000 steps (6b) | +0.2 +1.6 -0.5 +1.0 +0.8 / +0.6 | 6.79 | none |
| adversarial with noise, 1000 steps | +0.4 +0.6 -0.9 +0.8 +0.1 / +0.2 | 6.12 | -20.9 dB |
| adversarial with noise, 2000 steps | +0.4 +0.8 -1.0 +0.9 +0.3 / +0.3 | 6.11 | -23.0 dB |
| adversarial with noise, 3000 steps | +0.1 +0.7 -1.0 +1.0 +0.3 / +0.2 | 6.12 | -23.5 dB |

- With noise, flapping is about half of the no-noise run on 176 and 306, but 277 is unchanged (+1.0) and all are
  still above "before". Partial support for the hypothesis, not a clean confirmation.
- Wind is back slightly louder than before (-25.2 -> -23.5 dB; noise share of the mix -24.7 -> -21.4 dB).
- Caveat: with noise the discriminator separates real from fake less (hinge loss ~1.7-1.8 vs ~1.1-1.4 without
  noise), so part of the lower flapping may be weaker adversarial pressure overall.
- Listening: pending (listening page section #adversarial-ruido).

## Experiment 6d: adversarial with noise, spectral loss without its log term (2026-10-29)
`--spec-log-weight 0` (training spectral loss = linear-magnitude term only; evaluation still the full loss).
`analysis/results/e6/structured_v5_body_audio_noise_strongD_linonly/`.

| step | held-out loss | noise share of mix | wind (quiet passages) | flapping (023 176 276 277 306 / mean) |
|---|---|---|---|---|
| with log term (6c), 3000 | 6.12 | -21.4 dB | -23.5 dB | +0.1 +0.7 -1.0 +1.0 +0.3 / +0.2 |
| without, 1000 | 6.21 | -14.8 dB | -23.0 dB | +0.1 +0.4 -1.0 +1.2 +0.1 / +0.2 |
| without, 2000 | 6.22 | -14.2 dB | -23.4 dB | -0.2 +0.6 -1.0 +1.5 +0.3 / +0.2 |
| without, 3000 | 6.30 | -12.7 dB | -20.1 dB | +0.1 +0.9 -0.8 +1.5 +0.2 / +0.4 |

The hypothesis "the log term pays for the wind" is rejected for the adversarial phase: without it the noise grows by
~9 dB and the wind gets louder. The log term is symmetric, so it also punishes too much noise; it pins the noise to
the original's level in quiet passages. The push toward more noise comes from the adversarial side (the
complex-STFT discriminator can't hear it), and the wind's problem is its kind (texture, placement), not mainly its
level. Also in 6c continued (9k, in progress): noise share -21.4 -> -16.6 dB and wind -23.5 -> -21.4 dB by step 4000,
i.e. longer adversarial training keeps adding noise.

## Experiment 6c continued to 9000 steps (2026-10-29)
`analysis/results/e6/structured_v5_body_audio_noise_strongD_9k/` (copy of 6c resumed exactly from its 3000-step
checkpoint; 6c's own folder keeps the 3000-step model).

| step | held-out loss | noise share | wind | flapping (023 176 276 277 306 / mean) |
|---|---|---|---|---|
| 3000 (6c) | 6.12 | -21.4 dB | -23.5 dB | +0.1 +0.7 -1.0 +1.0 +0.3 / +0.2 |
| 4000 | 6.15 | -16.6 dB | -21.4 dB | +0.1 +1.3 -0.9 +0.9 +0.4 / +0.4 |
| 6000 | 6.14 | -17.7 dB | -23.9 dB | +0.1 +1.2 -1.0 +1.2 +0.5 / +0.4 |
| 8000 | 6.14 | -15.9 dB | -21.3 dB | +0.0 +1.7 -0.8 +1.2 +0.5 / +0.5 |
| 9000 | 6.15 | -14.5 dB | -19.9 dB | +0.3 +1.6 -0.8 +1.1 +0.5 / +0.5 |

Longer training does not help: the loss is flat, the noise keeps growing (+7 dB) and the wind gets louder, and the
flapping comes back (176 back to +1.6 dB, the no-noise level). By the numbers the 3000-step checkpoint is the best;
listening pending. Consistent with 6d: the complex-STFT discriminator keeps asking for more noise.

## Experiment 6e: a discriminator that also sees the log magnitude (2026-10-29)
`--d-kind both`: at each of the three resolutions, a second sub-discriminator looks at the log magnitude (on the
complex view, quiet passages are tiny next to the partials, so background wind barely registers). Same as 6c
otherwise (base structured_v5_body_audio, with noise, 64 channels, D learning rate x2, 3000 steps).
`analysis/results/e6/structured_v5_body_audio_noise_strongD_both/`.

| step | held-out loss | noise share | wind | flapping (023 176 276 277 306 / mean) |
|---|---|---|---|---|
| 6c (complex only), 3000 | 6.12 | -21.4 dB | -23.5 dB | +0.1 +0.7 -1.0 +1.0 +0.3 / +0.2 |
| both, 1000 | 6.12 | -22.4 dB | -23.4 dB | +0.0 +0.6 -1.1 +0.7 +0.1 / +0.1 |
| both, 2000 | 6.13 | -18.5 dB | -20.6 dB | +0.4 +0.9 -1.0 +0.7 +0.2 / +0.2 |
| both, 3000 | 6.15 | -20.0 dB | -22.3 dB | +0.1 +1.1 -0.9 +0.8 +0.3 / +0.3 |

The discriminator separates real from generated better (its loss ~1.5 vs ~1.8), but the generator's numbers are the
same as 6c within run-to-run noise: no less wind. A discriminator that can see the wind does not make the generator
drop it, because nothing in the synthesizer can produce what the wind is covering (consistent with E8). 1000 steps
already gives the full effect. Listening decides between 6c and 6e. Next: log magnitude only (6f).

## Experiment 6f: log-magnitude discriminator only (2026-10-29)
`--d-kind logmag`, otherwise as 6c/6e. `analysis/results/e6/structured_v5_body_audio_noise_strongD_logmag/` (stopped by
a shutdown at step ~2100 and resumed from the 2000-step checkpoint).

| step | held-out loss | noise share | wind | flapping (023 176 276 277 306 / mean) |
|---|---|---|---|---|
| 1000 | 6.14 | -21.0 dB | -21.6 dB | +0.0 +0.7 -1.1 +0.4 +0.0 / +0.0 |
| 2000 | 6.14 | -21.0 dB | -21.3 dB | +0.0 +1.0 -1.0 +0.4 +0.1 / +0.1 |
| 3000 | 6.15 | -22.7 dB | -22.5 dB | -0.3 +1.0 -1.0 +0.3 +0.1 / +0.0 |

The lowest flapping of all adversarial runs (277: +0.3 vs +1.0 dB in 6c), the D separates best (loss ~1.2), but the
wind is the same as 6c/6e (-22 to -23 dB). Seeing the quiet passages on a log scale does not make the generator drop
the wind either.

## Experiment 6g: adversarial with noise on the combined model (2026-10-29)
Base `structured_v8_combo` (body up to 6 kHz + two decay rates + pluck click), `--d-kind both`, otherwise as 6c.
`analysis/results/e6/structured_v8_combo_noise_strongD_both/` (resumed from its 1000-step checkpoint after a shutdown).

| step | held-out loss | noise share | wind | flapping (023 176 276 277 306 / mean) |
|---|---|---|---|---|
| before (combo) | 6.00 | -18.9 dB | -21.8 dB | -0.6 +0.7 -1.0 -0.2 -0.2 / -0.3 |
| 1000 | 6.12 | -18.4 dB | -20.4 dB | -0.2 +0.8 -0.8 -0.1 +0.4 / +0.0 |
| 2000 | 6.17 | -16.5 dB | -17.8 dB | -0.3 +1.2 -0.6 +0.4 +0.4 / +0.2 |
| 3000 | 6.17 | -16.8 dB | -20.2 dB | -0.8 +1.0 -0.7 +0.4 +0.3 / +0.0 |

The extra parts do not reduce the wind under adversarial training either: the noise grows as in 6c. With the noise
available, every setup tried ends with the wind at -20 to -23 dB. Remaining test: the combined model with no
catch-all noise at all (`structured_v8_combo_nonoise`, E8), now that its parts cover ~77 % of what the noise covered.
