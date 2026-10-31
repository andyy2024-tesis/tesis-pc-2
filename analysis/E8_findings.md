# Experiment 8: what is the wind covering? New guitar phenomena (2026-10-28, overnight)

Context (user listening of experiment 6c): adversarial training with the catch-all noise kept removes the artifacts on
all five recordings and adds realism, but the wind comes back. Pattern: when the synthesizer cannot produce something,
the model fills the gap with whatever it has (harmonic wobble without noise, wind with noise). Plan agreed with the
user: (1) continue the with-noise adversarial run to 9000 steps; (2) loss test: spectral loss without its log term
(the log term pays for filling quiet regions); (3) wind map; (4) try every candidate phenomenon; (5) a better
discriminator if time allows.

## Wind map (`analysis/wind_map.py`, results in `analysis/results/wind_map.json`)

All 85 held-out windows. Y = original, D = model without its catch-all noise, N = catch-all noise.
Time classes use a 21 ms window (the 85 ms one smears the attack into "before the onset").

| frames | noise vs original | share of all noise | share of original energy | noise concentration |
|---|---|---|---|---|
| 12-40 ms before an onset | -29.2 dB | 8.5 % | 3.6 % | 2.4x |
| 0-10 ms after an onset | -27.3 dB | 15.3 % | 4.1 % | 3.7x |
| 10-30 ms | -32.3 dB | 14.3 % | 12.0 % | 1.2x |
| 30-60 ms | -35.3 dB | 9.6 % | 16.3 % | 0.6x |
| 60-150 ms | -36.1 dB | 15.7 % | 31.8 % | 0.5x |
| note held (later) | -32.6 dB | 34.8 % | 31.7 % | 1.1x |
| ringing after the written end | -27.2 dB | 1.8 % | 0.5 % | 3.6x |

(base model structured_v5_body_audio; the adversarial-with-noise model is the same within ~1-2 dB, slightly more noise
while notes are held.)

- The noise piles up right at the attack (first 10 ms) and just before onsets: finger/nail noise, and something
  before the pluck (finger landing on the string, the previous note being stopped). It is under-used in the body of
  the note (60-150 ms).
- By frequency: 69 % of the noise energy is at 0.8-3.2 kHz; relative to the original it is strongest in the highs
  (3.2-6.4 kHz -13 dB, 6.4-12.8 kHz -8 dB). The deterministic part lacks energy above 3.2 kHz (-4.4 / -5.6 dB) and
  at 50-100 Hz (-4.4 dB).
- Between partials (60-3000 Hz, notes held): the original has 1.4 % of its energy away from every sounding partial,
  the model 0.5 % (-5.5 dB); a quarter of the noise lands there.
- Open strings: in bins away from sounding partials, the original is 13.5 dB louder at the partials of the six open
  strings than elsewhere; the model (no such mechanism; leakage control) 9.4 dB. About +4 dB of real sympathetic
  resonance.

## Squeak check (`analysis/squeak_check.py`)

1.5-6 kHz energy in the 12-40 ms before each onset, model (no catch-all noise) vs original:

| onsets | n | model vs original |
|---|---|---|
| wound string (below G3), after another wound-string note at a different pitch | 125 | -3.8 dB |
| wound string, other | 165 | -4.7 dB |
| treble (G3 and up) | 564 | -1.5 dB |

Something noisy happens before plucks on the wound strings that the model lacks, but it does not depend on a shift
from the previous bass note (crude proxy: no string information yet). Modeled as `squeak` (noise before wound-string
onsets, level from the note context); real shifts need the tablature (strings and frets).

## Candidates (each a NetConfig switch in `src/tesis/model/guitar_net.py`; trained like the best base model:
structured + body + audio features, linear spectral loss, 4000 steps; runs `analysis/results/e4/structured_v8_*`)

- `two_decay`: each partial = two polarizations (share, decay ratio, detune in Hz from a small MLP over pitch and
  partial frequency): double decay and slow beating.
- `pluck_click`: two learned short waveforms (nail click ~4 ms, finger thud ~12 ms) from 20 ms before to 30 ms after
  every onset, per-note levels from the note MLP.
- `sympathetic`: six open strings x 16 partials, driven by played partials within a learned bandwidth (Hz), each a
  one-pole resonator with the decay of a free string at that frequency, through the body filter.
- `squeak`: filtered noise (per-recording spectrum, starting at 1.5-6 kHz) over the 60 ms before each wound-string
  onset; level from pitch, interval and time since the previous wound-string note, loudness.
- body above 600 Hz: shared pluck body response low-passed at 6 kHz instead of 600 Hz, EQ up to 8 kHz (10 bands).
- reverb: room reverb on (round 3 found no gain without audio features; re-checked).

- `symp_pluck` (added after the first sympathetic run): every pluck also kicks all open-string partials (the bridge
  passes the broadband transient to every string), not only played partials close to theirs.

## Notes on the candidates as they finish

- Run-to-run variation: the base model's noise share moves by ~4 dB between its own checkpoints (-28.7, -24.0,
  -24.7 dB at 2000-4000 steps), so single runs only show large effects. A repeat of the base model with another seed
  (`structured_v8_base_seed1`) replaces the reverb re-check to measure this.
- sympathetic (played partials only): used (-34.6 dB of the mix), held-out loss 6.03 vs 6.00, noise share -19.1 vs
  -24.7 dB. Wind map: the model's excess at open-string partials (between played partials) did not rise (8.9 vs 9.4
  dB; original 13.5), because a resonance driven by a coinciding played partial sits on that partial. Hence
  `symp_pluck`.
- `analysis/ring_check.py`: the original's open-string excess is not from open strings the synthesizer cut short
  (6.9 dB when the open string was played in the last 4 s and its note already ended, 10.1 dB when it was not played).
  Side finding: 33 % of notes are ended by the next note on their voice before ringing 1 s (6 voices are not strings);
  the real-string assignment from the tablature would fix ring lengths.

## Results so far (held-out, 85 windows; `analysis/compare_runs.py e8 ...`, listening excerpts in
`analysis/results/compare/e8/`)

"Without noise" = the same trained model rendered with its catch-all noise switched off: how much the deterministic
parts cover by themselves (the noise is pinned to the original's level in quiet passages by the log term whatever
else exists, so the with-noise loss and noise share barely move).

| run | loss | loss without noise | wind | flapping | noise share | new part share |
|---|---|---|---|---|---|---|
| base (structured_v5_body_audio) | 6.00 | 7.82 (+1.82) | -25.1 dB | -0.2 | -24.9 dB | |
| + sympathetic (played partials) | 6.02 | 7.90 (+1.88) | -22.6 dB | -0.1 | -19.1 dB | -34.5 dB |
| + pluck click | 6.00 | 7.55 (+1.55) | -22.7 dB | -0.2 | -24.0 dB | -52.6 dB (burst -95) |
| + body response up to 6 kHz | 6.02 | **6.95 (+0.94)** | -21.3 dB | -0.0 | -23.3 dB | body -17.6 dB |
| + two decay rates | 6.04 | **7.36 (+1.33)** | -23.0 dB | -0.3 | -23.7 dB | |
| + squeak | 6.02 | 7.78 (+1.75) | -23.6 dB | -0.2 | -25.5 dB | -60.4 dB |

- The full-band body response after each pluck covers about half of what the noise was covering: the clearest
  effect so far. The pluck click covers ~15 % (and replaces the old noise burst completely).
- Sympathetic resonance from played partials covers nothing measurable; the squeak is barely used (-60 dB) and covers
  little (+1.75). Two decay rates cover ~27 %.
- The wind (noise in quiet passages) does not drop with any added part (-21 to -24 dB vs -25 dB): with the noise
  available, the log term keeps it at the original's level in quiet passages whatever else exists. The parts cover
  real content when the noise is off, but do not push the noise out by themselves.
| combined: body up to 6 kHz + two decay rates + click | 6.00 | **6.42 (+0.42)** | -22.2 dB | -0.3 | -19.2 dB | click -59.3 dB |

- Combined model (`structured_v8_combo`): the three parts add up. Switching its noise off now costs +0.42 instead of
  +1.82: the deterministic parts cover ~77 % of what the noise was covering. Its noise share with noise on is still
  -19 dB (the log term again), and the stationary floor rose from -46 to -40 dB.
- Next: the combined model trained without the catch-all noise (`structured_v8_combo_nonoise`), and the adversarial
  fine-tune on the combined model with noise (experiment 6g, E6_findings.md).

## Combined model trained without the catch-all noise (`structured_v8_combo_nonoise`)

| run | loss | mel loss | flapping | floor share |
|---|---|---|---|---|
| combined, with noise | 6.00 | 2.29 | -0.3 | -40.2 dB |
| combined, trained without noise | **6.03** | 2.33 | -0.0 | -34.6 dB |
| old structured + body without noise (round 2, no audio features) | 6.36 | 2.53 | +0.9 | -37.7 dB |

Removing the noise now costs 0.03 (it cost ~0.3-0.4 before the new parts), and the flapping that every earlier
no-noise model had is gone (-0.0 dB vs the original). The stationary floor took a little more (-34.6 dB); it is
constant over time, so it cannot follow the attacks as the wind did, but worth a listen. Next (to discuss): the
adversarial fine-tune on this no-noise model, the setup that sounded most realistic before but had artifacts.

## Is the wind a room reverb tail? (`analysis/room_check.py`, results in `analysis/results/room_check.json`)

User listening (2026-10-29): adversarial with noise at 9000 steps sounds "more in the room" than 3000 despite more wind;
each original comes from a different recording setup. Test: correlate (log, 10 ms frames, 200 Hz-6.4 kHz) the noise,
and what the original has between partials that the deterministic part lacks, with the playing level and the attacks
smeared by a one-sided exponential of tau = 0-0.8 s (a reverb tail follows the smeared past).

- The noise is a nearly steady background: 10th percentile over median -4 to -5 dB. Its correlation with the playing
  is low (base: 0.17 at tau 0, 0.21 with attacks at 25-50 ms) and falls to ~0 after long adversarial training (9k).
  In the combined model it is anti-correlated with the playing level (-0.35): it rises when the playing gets quiet.
- What the original has between partials and the model lacks is anti-correlated with the playing level at every tau
  (-0.15 at tau 0): it shows up in quiet moments; no positive link with the smeared past, so not a reverb tail.
- The round-3 per-recording reverb (`structured_v4_body_reverb`) stayed near its initialization: unused.

Conclusion: the wind is not the model's attempt at a reverb tail. It behaves like each recording's steady background
(room tone, mic/preamp hiss), played with gusts by a noise that can move frame by frame. The per-recording stationary
floor could produce a steady background, but training prefers the movable noise (it can drop it under loud passages).

## Chord buzz in the combined model trained without noise (`analysis/buzz_check.py`, `analysis/results/buzz_check.json`)

User listening: `structured_v8_combo_nonoise` sounds buzzy, sometimes metallic when a chord is played; no steady hiss.
Test: per octave band (400 Hz - 6.4 kHz) the envelope in dB at 1 kHz, its fluctuation power at 30-150 Hz (roughness,
faster than the 4-50 Hz flapping) in 250 ms segments with 1 note sounding (212) or 3+ (577), model minus original.

| model | full, single | full, chord | strings alone, single | strings alone, chord |
|---|---|---|---|---|
| combo trained without noise | +0.8 | +0.9 | +0.2 | **+1.4** |
| combo (noise) | +1.0 | +0.7 | -0.5 | +0.3 |
| two decays only | +0.9 | +0.8 | -0.5 | +0.4 |
| body to 6 kHz only | +1.0 | +0.9 | -0.5 | +0.4 |
| base | +0.8 | +0.8 | -0.6 | +0.4 |
| base + adversarial with noise (favourite) | +0.6 | +0.7 | -0.5 | +0.2 |

All models are a bit rougher than the original (+0.6 to +1 dB). The strings of the model trained without noise are
~1 dB rougher in chords than in any model trained with noise; two decays alone add nothing. So the buzz is not the
two-polarization beating: without a noise source the training makes the strings themselves wobble fast in chords
(the same mechanism as the flapping of the no-noise adversarial models, at a faster rate). Small effect on this metric.
