# Experiment 11: the final model (2026-11-17, overnight)

## What the user's ear decided (experiment 10 listening, 2026-11-17 ~00:40)

- Windowed attention sounds best (cleaner dynamics and harmonics) on 306, 276 and 176, despite a worse loss; on 277
  (strummed chords) the bidirectional LSTM is better: attention struggles when many notes sound.
- Strings (tab voices, per-string stiffness) sound synthetic, "like a cheap MIDI keyboard", above all the low strings.
  Adversarial training improves them, but the favourite (M1 + timbre 32 + adv) is still preferred.
- Keep the per-note attack: it is the part that should play the attack, whatever the loss says.
- The rise penalty removed most harmonic ticks; a dedicated part for fast high events is reasonable.
- Room background: no audible difference; kept because it leaves only room tone in the noise stem.

## Candidates (all M1 + timbre 32 + per-note attack + room background, no strings; then the favourite's adversarial
recipe, 3000 steps, complex D, with the rise penalty 20)

| run | network | why |
|---|---|---|
| A0 `structured_v10_m1_burst_floor` | GRU | control |
| T1 `structured_v11_m1_attn` | windowed causal attention (2 layers, 64 past frames) | the ear's choice, tested only on M2 so far |
| T2 `structured_v11_m1_attn_ahead` | T1 + 16 future frames (160 ms; offline only) | strums: the biLSTM's edge was seeing ahead |
| T3 `structured_v11_m1_attn_ahead_xvoice` | T2 + attention across voices every frame | strums: voices see each other's state |
| T4 `structured_v11_m1_attn_ahead_contact` | T2 + contact part from scratch (starts audible, -8) + rise penalty in training | fast high events get their own part |
| R `structured_v11_m1_attn_ahead_rec` | T2 + the recording reaches every part that could depend on it (`rec_all`) | different guitars, microphones and rooms |

R, the user's request: audit of what read the recording before: the network input, per-frame correction, note levels,
per-note attack, noise, floor and body EQ did; the base decay of each partial (pitch and frequency only), the body
response (one waveform for all guitars, 6-band EQ below 800 Hz: can't move a resonance), the stiffness (one law) and
the attack time (2 ms) and end-of-note fade (5 ms) did not. Double decay/beating, click and sympathetic strings are not
in M1. `rec_all`: the decay MLP reads the embedding; a 24-band EQ of the body response (40-800 Hz) from the embedding,
on top of the table; attack and fade scaled per recording (x0.22-4.5, starting at 1); the stiffness law shifted by the
guitar's measured offset relative to the average (analysis/inharmonicity_recording.py: residuals against the pooled
per-string law, shrunk with 3 notes, centered; sd 0.16, range -0.38..+0.24), so the average stiffness stays.

Tail (no adversarial, after R's adversarial step starts), in order:
- M2rec `structured_v11_m2_rec`: M2 + per-note attack (`structured_v10_m2_burst`, 5.77) with `rec_all`, which in M2
  also reaches the polarization MLP (double decay, beating) and the click (a per-recording EQ from the embedding,
  ~4 bands per octave); the body's fine EQ goes to 8 kHz there (42 bands). The user's question: did M2 fail because
  those parts were the same for every guitar? (M2 has no sympathetic strings either.)
- S1, S5, S3, S2 (below).
- R32 `structured_v11_m1_attn_ahead_rec32`: R with a 32-dim recording embedding. With 20 recordings and 16 dims, the
  embeddings span at most 16 directions, so the linear per-recording readouts (and everything that reads the embedding
  linearly) can't give each recording independent values; 32 >= 20 can. Beyond ~20 nothing more to tell apart.

## Why strings sound synthetic

Stiffness per guitar: in the per-note fits (analysis/inharmonicity_strings.csv, tab-matched clean notes) a string's log
B at the open string is very consistent within a recording (median |deviation| 0.06-0.11) but varies between recordings
(sd of the recording medians 0.20-0.66; ranges up to -12.0..-10.6 on the 4th string): each GAPS recording is a
different guitar and string set, and one table for all of them is a poor average. The wound-string fits use 24-30
partials, so their low stiffness (3-4x below the old single law) is well measured, at least 30-330 ms after the attack.

Runs (from M1 + timbre 32 + per-note attack + room background + strings, GRU, 5.70; no adversarial):

| run | change | question |
|---|---|---|
| S1 `structured_v11_m1_strings_oldB` | the old single stiffness law over pitch | does stiffness give the basses their color? |
| S5 `structured_v11_m1_strings_guitarB` | stiffness per guitar and string (analysis/inharmonicity_guitar.py) | does each guitar need its own? |
| S3 `structured_v11_m1_strings_decay` | the decay MLP also reads string, fret and the recording embedding | the decay knew only pitch and frequency |
| S2 `structured_v11_m1_strings_oldvoices` | strings known, old voice assignment | is it one voice per string? |

## Results (no adversarial; loss = mean MSS over the 85 held-out windows, seed spread ~0.05)

Measures: wind (dB, lower = less hiss between notes), snare (noise lock to the attack, 1-4k / 4-12k dB), share (noise's
share of the attack energy), room (noise over the measured room background, 700-4k / 4k-11k median dB, and % of frames
>6 dB above it), ticks (harmonic clicks away from onsets, inside the excerpt).

Ticks: the detector also fires at the excerpt's cut edges (t = 0 and 9 s) on every model; counting only clicks inside
the excerpt, every M1 model so far has none, adversarial or not. Real ticks are an M2 thing (M2 + per-note attack: 10,
on 277 and 306; M2 + strings + adv + rise penalty: 4, all on 306).

| run | loss | wind | snare | share | room | ticks |
|---|---|---|---|---|---|---|
| A0 base, GRU `structured_v10_m1_burst_floor` | 5.81 | -25.7 | -1.9 / +2.0 | 2-7 % | -0.0 / +5.5 (48 %) | 0 |
| T1 attention `structured_v11_m1_attn` | 5.80 | -24.2 | +1.4 / +1.4 | 4-7 % | -2.3 / +3.5 (40 %) | 0 |
| T2 + look-ahead `structured_v11_m1_attn_ahead` | 5.76 | -26.8 | +3.9 / +3.7 | 6-12 % | -1.1 / +4.6 (44 %) | 0 |
| T3 + across voices `structured_v11_m1_attn_ahead_xvoice` | 5.71 | -23.7 | -1.0 / +3.8 | 2-12 % | -3.7 / +3.2 (40 %) | 0 |
| T4 T2 + contact `structured_v11_m1_attn_ahead_contact` | 5.73 | -26.9 | +0.4 / +2.3 | 0-0 % | -24.9 / -21.0 (6 %) | 0 |
| R T2 + recording everywhere `structured_v11_m1_attn_ahead_rec` | 5.72 | -23.6 | +1.7 / +2.4 | 6-10 % | -1.7 / +4.0 (42 %) | 0 |

Reading so far: each step of the attention line lowers the loss a little (5.80 -> 5.76 -> 5.71); T3 is the first
attention model below the GRU by more than the seed spread. Letting voices see each other is aimed at strums (277), so
that sample is the one to judge it on.

T4: the contact part stays on (-34 to -47 dB below the mix, loudest on 306 and 277, the samples with most fast
events), and the noise part almost switches off: 0 % of the attack, 21-25 dB below the room background. Between the
per-note attack, the contact and the room background there is nothing left for the noise to do. Loss 5.73, a tie with
T2's 5.76.

### Adversarial (the favourite's recipe + rise penalty 20)

| run | loss | wind | snare | share | room | ticks |
|---|---|---|---|---|---|---|
| favourite `e6/structured_v9_base_t32_noise_strongD` | 5.93 | -19.0 | +14.2 / +12.3 | 100-99 % | +8.0 / +8.5 (62 %) | 0 |
| A0 adv `e6/structured_v10_m1_burst_floor_noise_strongD_rise` | 5.92 | -23.4 | -0.5 / +1.1 | 3-8 % | -1.7 / +4.5 (43 %) | 0 |
| T1 adv `e6/structured_v11_m1_attn_noise_strongD_rise` | 5.88 | -25.3 | +1.3 / +1.8 | 4-10 % | -4.3 / +2.5 (37 %) | 0 |
| T2 adv `e6/structured_v11_m1_attn_ahead_noise_strongD_rise` | 5.87 | -24.5 | +2.1 / +3.6 | 5-16 % | -2.2 / +4.3 (44 %) | 0 |
| T3 adv `e6/structured_v11_m1_attn_ahead_xvoice_noise_strongD_rise` | 5.81 | -20.5 | -0.6 / +8.6 | 4-38 % | -5.9 / +2.2 (37 %) | 0 |
| T4 adv `e6/structured_v11_m1_attn_ahead_contact_noise_strongD_rise` | 5.83 | -24.3 | +0.2 / +2.6 | 0-2 % | -30.2 / -16.9 (8 %) | 2 |
| R adv `e6/structured_v11_m1_attn_ahead_rec_noise_strongD_rise` | 5.82 | -21.3 | -0.1 / +1.5 | 4-10 % | -4.1 / +3.0 (39 %) | 0 |

T1 adv: the attention keeps its small edge through the adversarial step (5.88 vs 5.92) and has the least hiss so far;
the noise stays near the room background. The favourite's noise plays the whole attack (99-100 % share, +14 dB snare);
in every M1 + per-note-attack model the noise's share is under 10 %: the attack part plays it.

T2 adv: a tie with T1 adv (5.87 vs 5.88); a little more noise at the attack (up to 16 % of it). Whether the look-ahead
helps is for the ear on 277.

T3 adv: the best adversarial loss (5.81), but the adversarial step put noise back into the attack above 4 kHz (+8.6 dB
after each pluck, up to 38 % of the attack there) and raised the wind to -20.5 dB: half-way back to the favourite's
snare. Across voices helps the loss; whether the ear accepts that extra attack noise is the question.

R (recording everywhere, no adversarial): 5.72 vs T2's 5.76, within the seed spread; less noise at the attack (+1.7 /
+2.4 dB vs +3.9 / +3.7), a little more wind (-23.6 vs -26.8). What it learned (analysis/rec_use.py):
- attack time: every recording made it shorter, 2 ms -> 0.5-1.5 ms (scale 0.27-0.76): the fixed 2 ms ramp was too slow
  for all guitars, not just some;
- end-of-note fade: 4-16 ms depending on the recording (scale 0.8-3.3 of the old 5 ms): this one really differs per
  recording;
- partial decay: per recording, T60 of the low partials varies about +-20 % (E2 at 200 Hz: 8.4-11.9 s), the high ones
  +-5 % (6 kHz: 0.67-0.72 s): bass sustain is where guitars differ;
- fine body EQ (24 bands, 40-800 Hz): a shared correction of about 10 dB peak to peak, the same for every recording
  below 500 Hz (sd 0.2-0.6 dB across recordings), 1-2 dB per-recording differences only at 600-800 Hz: the body
  response's resonances did not move per guitar. (The strings already had a per-recording colour before: a 24-bin
  gain table, 50 Hz-22 kHz, per recording, `rec_body_filter`; that may be where each guitar's body went.)

T4 adv (resumed from step 2000 after the 06:33 reboot): 5.83, the second-best adversarial loss, with no snare and low
wind. The contact part stays on (loudest on 277 and 306, -33 dB) and the adversarial step does not hand the attack back
to the noise as it did in T3: the contact does that job. Two real ticks, both on 306 (5.27 s, 6.69 s).

R adv (resumed from step 1000): 5.82, a tie with T3 adv and T4 adv; no snare, no ticks, but more wind (-21.3 dB).

### Strings runs (no adversarial)

| run | loss | wind | snare | share | room | ticks |
|---|---|---|---|---|---|---|
| strings reference `structured_v10_m1_burst_floor_strings` | 5.70 | -25.5 | -2.2 / +1.0 | 2-6 % | +0.6 / +5.0 (45 %) | 0 |
| S1 old stiffness law `structured_v11_m1_strings_oldB` | 5.75 | -24.3 | -2.1 / +1.5 | 2-6 % | +1.5 / +5.6 (48 %) | 2 |
| S5 stiffness per guitar `structured_v11_m1_strings_guitarB` | 5.69 | -25.8 | -1.4 / +1.9 | 2-7 % | +0.7 / +5.1 (46 %) | 0 |
| S3 decay knows string, fret, guitar `structured_v11_m1_strings_decay` | 5.72 | -25.0 | -1.4 / +1.2 | 2-5 % | +0.6 / +5.2 (46 %) | 0 |

S1: going back to the old single stiffness law costs 0.05 (the seed spread): the measured per-string stiffness is a
little better by loss; whether the old law gives the basses their colour is for the ear.

S5: each guitar's own measured stiffness: 5.69, a tie with the pooled table (5.70). By loss, the stiffness per guitar
barely matters; the ear decides whether the basses sound less synthetic.

S3: a decay that also knows the string, the fret and the guitar: 5.72, a tie (5.70).

### M2 with the recording in every part (no adversarial)

| run | loss | wind | snare | share | room | ticks |
|---|---|---|---|---|---|---|
| M2 + per-note attack `structured_v10_m2_burst` | 5.77 | -17.9 | -2.0 / -3.8 | 2-2 % | +1.5 / +3.2 (37 %) | 10 |
| M2rec `structured_v11_m2_rec` | 5.75 | -19.4 | -3.8 / -4.8 | 1-2 % | +0.9 / +2.9 (35 %) | 7 |

M2rec: a tie with M2 + per-note attack (5.75 vs 5.77); the ticks stay (7 vs 10, at the same moments on 277 and 306:
4.03 s, 4.47 s; 6.36, 6.49, 7.34, 7.70 s). Letting the double decay, beating and click depend on the guitar did not fix
M2: the ticks are not caused by those parts being the same for every recording. M2 is still behind M1 by ear and noise
(wind -19.4 vs -24 to -27 dB in the M1 attention models).
