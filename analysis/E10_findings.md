# Experiment 10: attack vs noise, strings from the tablature, network backbones (2026-11-16, overnight)

User listening of the section-18 stems (2026-11-15): (1) M1's residual noise has the "snare"; M2's sounds like very
violent wind, maybe still picking up attacks, much less obviously. (2) The short high pitches of M2 + timbre 32 + adv are
in the harmonics stem: a very light, high click, quiet but noticeable because it is high.

## The ticks in the harmonics (analysis/harmonic_clicks.py)

Detector on the strings part of the 5 listening excerpts: frames (256 samples, 2 ms hop) whose power above 4 kHz is more
than 10 dB above the median of the surrounding +-40 ms; counted only if more than 30 ms from an onset.

| model | ticks | GRU correction off (delta = 0) | no detune between polarizations | measured pitch deviation off |
|---|---|---|---|---|
| M2 + timbre 32 + adv | 42 | 5 | 42 | 42 |
| M2 + timbre 32 | 10 | 2 | 10 | 9 |
| M1 + timbre 32 + adv | 5 | 2 | 5 | 5 |

So the ticks come from the GRU's per-frame correction, and the adversarial fine-tune of M2 multiplied them by four. They
fall 50-80 ms before the next onset (median 62 ms), about 40 ms before the MIDI end of a note. Looking at the correction
around three of them: in the voice whose measured loudness is dropping (the note is being damped), partials 1-10 go down
by 1.5-2 (log amplitude, correct), while partials 30-60 go UP by 1-1.5 for 20-30 ms: a brief boost of only the high
harmonics, i.e. a tonal tick.

Does the real recording have something there? The original's power above 4 kHz, max within +-6 ms, relative to its own
local median: at the model's ticks median +8.2 dB, 60 % of them above +6 dB; at random times +0.9 dB, 10 %. So the
adversarial M2 is copying a real sound, most likely the finger or nail touching the string to damp it or to prepare the next
pluck, with the only tool it has at that moment: the harmonics. The rise statistic used below (mean squared frame-to-frame
rise of the correction in partials >= 16, more than 40 ms after an onset): M2 + timbre 32 0.0049, + adv 0.0070 (+45 %);
M1 + timbre 32 0.0035, + adv 0.0048.

Fix under test (stage 2): a per-voice contact noise driven by the GRU (`contact`), plus a penalty on those fast rises
(`delta_rise` = 20), in the same adversarial recipe.

## Strings from the tablature (src/tesis/data/tab.py, analysis/tab_strings.py)

GAPS MusicXML has a tab part with string/fret per note; syncpoints give the audio time of each performed bar (some points
also carry an offset inside the bar). Performers repeat sections the score doesn't mark (204: no repeat signs, 46 performed
bars for 30 written), so repeats are not unrolled from the score: a Viterbi alignment matches each performed bar to the
written bar whose pitches best match the MIDI notes sounding in it (Dice overlap; moving to the next written bar is free,
any other move costs 0.5). Tab notes get a time by their position inside the bar; each MIDI note is matched to the closest
same-pitch tab note (one-to-one, within 0.25 s, then 0.6 s). Remaining notes take the string of the nearest same-pitch
matched note, else the lowest fret.

On the 20 training recordings: 99 % of notes matched (median; lowest 89 %); the tab fret equals pitch minus the open
string in every piece in standard tuning; three pieces (204, 218, 345) have the 6th string lowered to D, read from the
matched notes. Only 18 of 11118 notes (0.16 %) land on a string within 30 ms of another note on the same string.

## Inharmonicity per string (analysis/inharmonicity_strings.py)

Physics: on one string B grows as 1/L^2, so fret n multiplies B by 2^(n/6) (log B +0.116 per fret). The model used one
line over all pitches, log B = -11.35 + 0.0304 * pitch. Fits of (f0, B) on 797 clean notes (no other onset from -50 to
+350 ms, >= 6 partials, residual < 3 cents), tab-matched:

| string | notes | slope of log B per fret | B of the open string | single-law B at the open pitch |
|---|---|---|---|---|
| 1 (E4, nylon) | 262 | 0.104 | 3.9e-5 | 8.3e-5 |
| 2 (B3, nylon) | 211 | 0.084 | 5.3e-5 | 7.1e-5 |
| 3 (G3, nylon) | 79 | 0.036 | 6.4e-5 | 6.3e-5 |
| 4 (D3, wound) | 110 | 0.092 | 1.5e-5 | 5.4e-5 |
| 5 (A2, wound) | 92 | 0.160 | 1.1e-5 | 4.6e-5 |
| 6 (E2, wound) | 43 | 0.063 | 2.1e-5 | 4.0e-5 |

The wound basses are 3-4x less stiff than the single law says (the winding adds mass, not stiffness). Per string with the
fret law, the median error of log B is 0.21 vs 0.51 for the single law. On an open D string that moves partial 40 (5.9 kHz)
by ~50 cents: today's model plays the high partials of the bass strings up to half a semitone sharp. `strings` in NetConfig
uses these open-string values with the fret law, one voice per string, and gives the GRU and the note MLP the string and
fret.

## Is the noise only the room? (analysis/noise_vs_room.py)

Background per FFT bin = 10th percentile of the recording's power away from all sounding partials while the piece plays
(minimum statistics; calibrated: noise synthesized at that level measures within 1-2 dB of it). Noise part vs background on
the 85 held-out windows, median dB (share of cells more than 6 dB above):

| model | 0-700 Hz | 700 Hz-4 kHz | 4-11 kHz |
|---|---|---|---|
| M2 + timbre 32 | -5.2 (18 %) | +0.8 (30 %) | +2.9 (35 %) |
| M2 + timbre 32 + adv | -6.3 (18 %) | -2.1 (22 %) | +1.8 (32 %) |
| M1 + timbre 32 | -3.4 (21 %) | +8.9 (63 %) | +8.5 (62 %) |
| M1 + timbre 32 + adv | -1.9 (27 %) | +8.0 (58 %) | +8.5 (62 %) |
| M1 + adv | -7.8 (13 %) | +7.2 (55 %) | +7.8 (58 %) |

M2's noise sits at the room's level most of the time, with gusts (a third of the cells 6 dB or more above): the "violent
wind". M1's noise plays 7-9 dB above the room, most of the time: the snare and more.
