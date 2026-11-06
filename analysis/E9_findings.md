# Experiment 9: noise only around the plucks (2026-10-29, overnight)

Why: the room check (E8) says the wind is each recording's steady background (room tone, mic hiss) played with gusts by
the catch-all noise, which can move every frame. User listening: with the noise switched off, the adversarial model
(3000 steps) sounds the same minus the wind, so the noise only adds wind there; at 9000 steps it is less clear.

Change (`noise_gate` in NetConfig): the catch-all noise is multiplied by a gate that is 1 around each onset (fades in
over the 20 ms before it, full for 10 ms after, then decays with a 25 ms time constant) and 0 far from all onsets. The
steady background is left to the per-recording stationary floor, which can't gust. On the held-out windows the gate is
above 0.5 only 3-19 % of the time (per recording) and below 0.01 41-92 % of the time.

Runs (from scratch, 4000 steps, then the adversarial fine-tune of the user's favourite: complex discriminator, 64
channels, D lr x2, 3000 steps):
- `structured_v5_gated`: base (structured + body + audio features) + gate -> `e6/structured_v5_gated_noise_strongD`
- `structured_v8_gated`: combined (body to 6 kHz + two decays + click) + gate -> `e6/structured_v8_gated_noise_strongD`

## Per-note attack timbre (pluck position), item (d) of the plan

The note MLP only sees pitch, loudness, onset clarity and the recording, so every note of the same pitch and loudness
gets the same partial template ("cheap MIDI"?). Real plucks differ: pluck position (comb-shaped dips), nail vs flesh,
which string. Feature (`tesis.data.loudness.note_timbre_db`, NetConfig `note_timbre`): level of the first 16 partials
20-105 ms after the onset, minus the power that was already there just before it, relative to their mean, read from
the audio (like the per-note loudness and the audio tracks: resynthesis conditioning).
Check (notes with no other onset in the next 0.3 s, all 20 recordings): after removing the mean timbre of each pitch,
the per-note deviation of the first 8 partials is 4.6 dB (std), and it persists in a later window (150-235 ms):
correlation 0.47 (0.41-0.59 per recording). So there is real per-note timbre the model can't know today.

## Results: gated models before the adversarial fine-tune (held-out, 85 windows)

| model | loss | noise share | noise in the quietest 20 % of frames ("wind") | floor share |
|---|---|---|---|---|
| base (`structured_v5_body_audio`) | 6.00 | -24.7 | -25.2 | -46.5 |
| base + gate (`structured_v5_gated`) | 6.15 | -35.8 | -41.3 | -32.4 |
| combined (`structured_v8_combo`) | 6.00 | -18.9 | -21.8 | -40.2 |
| combined + gate (`structured_v8_gated`) | **6.00** | -53.6 | **-59.3** | -33.5 |

With the gate, the steady floor takes over the background (+7-8 dB) and the noise in quiet moments drops by 16-37 dB.
The combined model loses nothing at all (6.00 = 6.00): its parts already cover what the noise did at the attacks, so
the gated noise is almost unused. The base model loses 0.15. Flapping on the listening excerpts unchanged (within
+-0.5 dB of the ungated models). The two adversarial fine-tunes can't share the GPU (out of memory): run one after
the other.

## Results: per-note attack timbre (`structured_v9_timbre` = combined + gate + `note_timbre`, 4000 steps)

Held-out loss **5.82** (combined + gate: 6.00; steps 1000-4000: 6.08, 5.90, 5.83, 5.82). Noise share -52.6 dB (still
almost unused), floor -35.8 dB. The largest gain from one change since the audio features (round 5).
Does it use each note's own timbre? (`analysis/timbre_shuffle.py`): same model evaluated with every note's timbre
swapped for a random other note of the same pitch in the same recording: 6.17 (worse than no timbre input at all);
all timbres 0: 6.38. So yes: the gain is the individual pluck, not the pitch's average timbre.
Caveat: like the per-note loudness and the audio tracks, the timbre is read from the recording being resynthesized
(conditioning, not generation from the score alone). To generate from a score, it would need a model of how
timbres vary from note to note (e.g. sample them from the training notes of the same pitch and loudness).

## Results: adversarial fine-tunes of the gated models (complex D, 3000 steps, as the user's favourite)

| model | loss | noise share | wind (quiet 20 %) |
|---|---|---|---|
| favourite: base + adversarial with noise (`e6/structured_v5_body_audio_noise_strongD`) | 6.12 | -21.4 | -23.5 |
| base + gate + adversarial (`e6/structured_v5_gated_noise_strongD`) | 6.22 | -35.4 | -41.3 |
| combined + gate + adversarial (`e6/structured_v8_gated_noise_strongD`) | **6.09** | -48.5 | **-56.3** |

The gate holds under adversarial training: the noise no longer climbs (the ungated runs went from ~-22 to ~-20).
Combined + gate + adversarial has a lower loss than the favourite and ~33 dB less wind. Whether it sounds as real
as the favourite is the user's call (listening page section "ruido-en-ataque").

| model | loss | noise share | wind |
|---|---|---|---|
| timbre (16 partials) + adversarial (`e6/structured_v9_timbre_noise_strongD`) | 5.86 | -50.9 | -52.8 |
| timbre 32 partials (`structured_v9_timbre32`) | 5.77 | -48.2 | -51.3 |
| timbre 32 + adversarial (`e6/structured_v9_timbre32_noise_strongD`) | 5.84 | -46.8 | -51.6 |

32 partials instead of 16: a small further gain (5.82 -> 5.77). The adversarial fine-tune costs ~0.05-0.07 of loss
here (as before: 6.00 -> 6.09 without timbre); whether it still helps the sound is for the ear. Continued to 9000 steps
(`_9k`) because the user preferred the 9000-step favourite.

### Longer adversarial on timbre 32 (resumed 3000 -> 9000 steps, `e6/structured_v9_timbre32_noise_strongD_9k`)

| step | loss | noise share | wind (quiet) |
|---|---|---|---|
| 3000 | 5.84 | | -51.6 |
| 4000 | 5.88 | -45.1 | -51.7 |
| 6000 | 5.87 | -40.9 | -49.3 |
| 8000 | 5.86 | -37.7 | -45.5 |
| 9000 | 5.88 | -38.2 | -45.1 |

No gain from more adversarial steps: the loss stays flat and the gated noise slowly grows (+7 dB in the quiet 20%),
same direction as the ungated 9k run, but still ~20 dB below the favorite's wind. 3000 steps stays the recipe.

### Chord buzz of the new models (analysis/buzz_check.py, model minus original, dB)

| model | single full | chord full | chord strings |
|---|---|---|---|
| combo (no gate) | +0.98 | +0.72 | +0.33 |
| combo + gate (v8_gated) | +0.91 | +0.90 | +0.92 |
| + timbre 32 | +0.23 | +0.56 | +0.71 |
| + timbre 32 + adv. | +0.32 | +0.49 | +0.59 |
| favorite (adv. with noise) | +0.63 | +0.71 | +0.18 |

The gate moves some chord roughness into the strings (+0.6 dB: the ungated noise used to smooth over the partials'
wobble, as in the no-noise runs), but the full output is about as rough as before. The timbre input brings the full
output closest to the recording (+0.2 to +0.5 dB), the least buzzy models so far. All differences are within ~1 dB.

## More data: 80 recordings (`--extra-recordings 60`, timbre 32 config; evaluation still on the 85 windows of the 20)

60 more GAPS training-split recordings with f-measure >= 0.93 (random, fixed seed): 3.7x the training audio.
At the same 4000 steps: 5.87 (20 recordings: 5.77; steps 1000-4000: 6.17, 6.00, 5.88, 5.87). Each evaluation recording
gets a quarter of the updates, including its own per-recording parts (floor, body EQ, embedding), so equal steps is
not a fair comparison: re-run with 12000 steps (`structured_v9_timbre32_data80_12k`). Its adversarial fine-tune
(`e6/structured_v9_timbre32_data80_noise_strongD`) tests whether the discriminator gains from more data.

### 12000 steps on the 80 recordings (`structured_v9_timbre32_data80_12k`)

| step | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 |
|---|---|---|---|---|---|---|
| loss | 6.02 | 6.00 | 5.85 | 5.83 | 5.83 | 5.84 |

Plateaus at 5.83 from step 8000, still worse than the 20 recordings alone (5.77, flat from step 3000). More recordings
do not help on these held-out windows: the evaluation windows come from the 20 training recordings, so their own
per-recording parts and the shared parts tuned to them matter more than general guitar knowledge; extra recordings
compete for the shared capacity. (Would matter for unseen recordings, which this evaluation does not measure.)
Wind -53.1 dB, noise share -54.1 dB: the gate holds. The queued adversarial on the 4000-step data80 run was dropped
(the 12k one answers the question); the GPU slot went to timbre with all 60 modeled partials instead (`structured_v9_timbre60`).

Its adversarial fine-tune (`e6/structured_v9_timbre32_data80_12k_noise_strongD`, same recipe): 5.90 / 5.93 / 5.95 at
steps 1000 / 2000 / 3000 (wind -53.6 / -47.0 / -51.0). Worse than the 20-recording adversarial (5.84) and drifting up:
the discriminator does not gain from more data here either. Decision: stay with the 20 recordings.

### Timbre with all 60 modeled partials (`structured_v9_timbre60`, 4000 steps)

Loss 5.99 / 5.86 / 5.80 / 5.78 at steps 1000-4000: same as 32 partials (5.77). 32 is enough; above it the partials
are weak or past Nyquist for high notes.

### Late timbre reading (`structured_v9_timbre32late`, `timbre_late`, 4000 steps)

Adds the same per-partial reading 150-235 ms after the onset (per-pluck differences in decay) to the 32-partial timbre
(code in guitar_net.py / loudness.note_timbre_db `start`).

| step | loss | noise share | floor share |
|---|---|---|---|
| 1000 | 6.01 (timbre32: 5.97) | -70.0 | -37.6 |
| 2000 | 5.87 (5.87) | -61.9 | -33.5 |
| 3000 | 5.79 (5.78) | -51.2 | -30.1 |
| 4000 | 5.79 (5.77) | -51.1 | -30.2 |

Wind (quiet 20 %): -54.0 vs -51.3 for timbre32 (same script, which reproduces timbre32's -51.3).
Loss is a tie (0.02 worse, within run-to-run spread); the catch-all noise is ~3 dB quieter and the stationary floor
~4 dB louder, i.e. a bit of steady background moved from the gusty noise into the floor. The late reading adds no
fidelity by the loss; timbre32 stays the reference. No adversarial fine-tune run on it (not better on loss).

## Overnight 2026-10-31: seed spread, leave-one-out, bigger GRU, unseen recordings

Queue: `analysis/results/queue_overnight/` (run.sh, stage2.sh, status.txt, wind.txt). All 4000 steps, timbre 32 config
unless noted; held-out = the usual 85 windows; wind = noise in the quietest 20 % of frames.

### Seed spread (same config, different initialization and window sampling)

| seed | loss | wind |
|---|---|---|
| 0 (`structured_v9_timbre32`) | 5.77 | -51.3 |
| 1 (`_seed1`) | 5.80 | -51.5 |
| 2 (`_seed2`) | 5.80 | -53.5 |

Mean 5.79, range 0.03. Differences of a few hundredths between runs are noise: late timbre (5.79), 60 partials (5.78),
GRU 512 (5.77) and 16 partials (5.82) are all ties with timbre 32. Wind varies by ~2 dB between seeds.

### Leave-one-out: the final model minus one part (vs seed mean 5.79)

| removed | run | loss | change | wind | what the part buys |
|---|---|---|---|---|---|
| note timbre | `structured_v8_gated` (existing) | 6.00 | +0.21 | -59.3 | most of the fidelity |
| body response | `_v9_t32_no_bodyir` | 5.89 | +0.10 | -41.5 | fidelity; without it the noise covers the body (+10 dB wind) |
| two decays | `_v9_t32_no_twodecay` | 5.81 | +0.02 | -53.0 | nothing measurable |
| catch-all noise | `_v9_t32_no_noise` | 5.81 | +0.02 | (none) | nothing measurable |
| pluck click | `_v9_t32_no_click` | 5.78 | -0.01 | -51.2 | nothing measurable |
| noise gate | `_v9_t32_no_gate` | 5.79 | 0.00 | -21.3 | no loss change, but 30 dB less wind |

Bigger GRU (512, `_v9_t32_gru512`): 5.77, wind -55.7: a tie; network size is not the limit.

Reading: inside the final model only two parts carry the loss, note timbre and the body response. The gate is what
removes the wind. Two decays, click and the catch-all noise no longer move the loss once timbre is in (E8 measured them
added to the old base, where some helped); whether they are audible is for the ear.

### Unseen recordings (`analysis/unseen_adapt.py`)

10 recordings never used in any training (the first 10 of `e4_train.extra_ids`; 7 have held-out windows, 35 windows).
Every shared weight is frozen; only the per-recording tables (embedding, body EQ/filter, floor, transient...) are fitted,
1500 steps on the first 85 % of each new recording; scored on the windows after 85 %. Results in `analysis/results/unseen/`.

| model | before fitting (mean tables) | fitted, 1500 steps | wind |
|---|---|---|---|
| timbre 32 (final) | 7.98 | **5.34** | -45.9 |
| base structured (`structured_v5_body_audio`) | 8.07 | 5.59 | -22.6 |
| plain DDSP (`ddsp_v5_audio`) | 8.08 | 5.78 | -14.3 |
| reference: timbre 32 trained with the 10 included (`structured_v9_timbre32_data30`) | | 5.38 | -48.5 |

The final model's shared part generalizes: on recordings it never heard, fitting only the per-recording settings gives
5.34, as good as a model trained on those recordings (5.38). The gap to plain DDSP is larger on new recordings (0.44)
than on the training recordings (E4). Converges in ~750 steps (~6 min). Without fitting (7.98) the per-recording
settings matter a lot: room, mic and guitar differ between recordings. Still resynthesis: loudness, tracks and timbre
are read from the new audio.

### How much audio of a new recording is needed (`--train-seconds`, 1000 steps)

| audio used per new recording | loss | wind |
|---|---|---|
| none (mean tables) | 7.98 | -43.6 |
| 10 s | 5.48 | -45.3 |
| 30 s | 5.44 | -46.0 |
| 60 s | 5.41 | -46.8 |
| all of the first 85 % (minutes) | 5.34 | -45.9 |

10 seconds of a new recording already gets most of the way: the per-recording settings are few and smooth.

### Simplified final model (`structured_v9_t32_simple`: timbre 32 + body + gate, no two decays, no click)

Loss 6.02 / 5.87 / 5.80 / 5.79 at steps 1000-4000, wind -49.2: a tie with the full timbre 32 model (seed mean 5.79,
wind -51 to -54). By the numbers, two decays and the click can go; keep them only if they are audible.
