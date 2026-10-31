# Experiment 4: first neural model, structured vs plain DDSP (2026-10-26)

Code: `src/tesis/model/guitar_net.py`, `scripts/e4_train.py`, `analysis/e4_targeted.py`.
Results: `analysis/results/e4/` (models, histories, `heldout_music_windows.csv`, `targeted_metrics.csv`, `listen/`).

Setup: 20 GAPS train-split recordings (20 performers, 51 min, f-measure >= 0.93), 24 kHz. Train on the first 85% of
each recording, test on 3 s windows after the 85% point. Inputs: notes on 6 voices + per-note loudness read from the
mix (experiment 3) + onset clarity + a learned per-recording embedding. A GRU (256 units) runs per voice on 10 ms
frames with 1 s of context. 6000 steps x 8 windows (~27 min each on the RTX 4090), same network size and budget:
- structured: stiff-string partials, per-note template + physical decay line + GRU correction, 2 ms attack, per-note
  pluck burst, smooth per-recording body response (24 bins), noise floor, log-band noise (0.66 M parameters)
- ddsp: harmonic partials, GRU predicts amplitude + harmonic distribution per frame, linear-band noise (0.57 M)

Held-out results (85 music windows, 17 recordings):

| | Spectral loss | Noise share (median) | Attack LSD | 2-8 kHz LSD | <200 Hz LSD |
|---|---|---|---|---|---|
| structured | 6.29 | -22.2 dB | 7.70 | 8.43 | 10.16 |
| ddsp | 6.26 | -21.9 dB | 7.73 | 8.37 | 8.98 |

(LSD = log-spectral distance in dB; structured better on 46% of windows overall, 52% attack, 45% high, 12% low.)

Findings
- No benefit from the structure yet in this first network setting; the low end is clearly worse.
- Below 200 Hz both models rely on the catch-all noise (noise alone matches the target better than the strings
  alone). The pluck-excited body response, which explained ~90% of the energy below f0 for single notes
  (experiment 1), was dropped after experiment 2 showed free onset modes are not identifiable per piece; without it
  the low "thump" is faked by noise in both models: the original thesis motivation, now measured.
- Data bug fixed: 25 of the original 90 test windows had no notes (talk/applause/room noise at the end of YouTube
  videos); windows are now restricted to the span of the notes. Those windows had inflated the noise share to ~18%.
- Training vs held-out: loss ~6.1 vs ~6.3 on music windows (little overfitting at this budget).
- The multi-scale spectral loss is dominated by what a score can't predict (vibrato, timing, dynamic shaping);
  effects of the structured parts in experiment 1 were only 1-2% of the loss. Listening and targeted metrics matter.

Next
- Bring back the body's pluck response in an identifiable form: shared across recordings with small per-recording
  variation, excited by the pluck transient, limited to the low end; check the <200 Hz metric and noise share.
- Investigate why inharmonicity shows no gain at 2-8 kHz (nylon B trend from 211 notes may be too crude per string;
  per-recording tuning drift; vibrato).
- More data/steps once the structure pays off; listening comparisons in `analysis/results/e4/listen/`.

# Round 2: body pluck response and no-noise ablation (2026-10-26)

All five models retrained on music-only windows (data bug fixed), 4000 steps x 8 windows each, same network.
`body_ir`: one short (0.3 s, fixed 80 ms decay envelope) low-end (<600 Hz) body response shared by all recordings,
per-recording 6-band EQ, excited by every pluck at a per-note level from the note MLP. `noise=false`: no catch-all
noise (the structured model keeps its stationary noise floor). Results: `analysis/results/e4b_targeted.txt`,
listening excerpts in `analysis/results/e4/listen_structured_v2-...-ddsp_v2_nonoise/` (files numbered in table order).

Held-out medians (85 music windows, 17 recordings; LSD = log-spectral distance, lower is better):

| Model | Spectral loss | Noise share | Body share | Attack LSD | 2-8 kHz LSD | <200 Hz LSD |
|---|---|---|---|---|---|---|
| structured | 6.29 | -22.7 dB | - | 7.73 | 8.34 | 10.34 |
| structured + body pluck response | 6.34 | **-30.2 dB** | -17.1 dB | 7.80 | 8.38 | **8.59** |
| structured + body, **no noise** | **6.31** | - | -17.6 dB | 7.89 | 8.58 | 10.38 |
| ddsp | 6.37 | -22.6 dB | - | 7.80 | 8.41 | 8.94 |
| ddsp, **no noise** | 6.52 | - | - | 8.19 | 9.29 | 11.15 |

Findings
- Body pluck response: low-end error 10.34 -> 8.59 (better on 94% of windows, now better than DDSP), and the noise
  share drops by 7.5 dB (-22.7 -> -30.2 dB): the body now explains what the noise was faking. Overall loss unchanged.
- No-noise ablation (user's idea): removing the catch-all noise costs plain DDSP 0.15 loss (6.37 -> 6.52; the
  structured model beats it on 92% of windows) but costs the structured model nothing (6.34 -> 6.31). Without noise,
  structured beats DDSP on every metric (attack 7.89 vs 8.19, 2-8 kHz 8.58 vs 9.29, <200 Hz 10.38 vs 11.15).
  This is the clearest evidence so far for the thesis claim: the structure explains what plain DDSP needs noise for.
- Without noise the low end gets worse again (10.38) despite the body: some low-end energy is genuinely noise-like
  (room rumble, handling), which a small residual noise should keep covering.
- Listening (round 1): all models sounded equally bad to the user; a likely large missing piece is room reverb
  (GAPS recordings are full of room sound; no model has any reverb yet).

# Round 3: room reverb + per-recording settings fix (2026-10-26/27)

Reverb: per recording, 8 log-spaced bands of fixed noise with a level and an RT60 (0.2-2.5 s) each, no direct sound,
applied to the instrument (strings + transient + body); dry parts rendered 1 s before the window for the tail.
First attempt computed per-recording settings as Linear(recording embedding) with zero init: they barely moved
(RT60 0.70-0.83 s for all, wet level went down). Fixed by direct per-recording tables (reverb, body EQ/filter,
noise floor, transient spectrum) with a 10x learning rate. Results: `analysis/results/e4d_targeted.txt`;
learned reverb per recording: `uv run python analysis/e4_reverb_report.py structured_v4_body_reverb`.

| Model (held-out medians) | Spectral loss | Noise share | Attack LSD | 2-8 kHz LSD | <200 Hz LSD |
|---|---|---|---|---|---|
| structured + body + reverb (fixed tables) | 6.28 | -29.8 dB | 7.75 | 8.43 | 9.49 |
| structured + body (fixed tables) | **6.22** | -30.4 dB | 7.83 | 8.39 | 9.11 |
| structured + body (round 2) | 6.34 | -30.2 dB | 7.80 | 8.38 | 8.59 |
| ddsp + reverb | 6.39 | -21.8 dB | 7.83 | 8.37 | 8.99 |
| ddsp (round 2) | 6.37 | -22.6 dB | 7.80 | 8.41 | 8.94 |

- The per-recording table fix helps (6.34 -> 6.22, best so far). The reverb doesn't help the metrics (6.28; worse
  than without on 60% of windows), for either model.
- Learned RT60s now differ per recording (0.4-1.3 s mid bands) but don't match listening: 276 (clear reverb to the
  user's ear) gets one of the weakest wet levels. Likely the strings' own learned sustain absorbs the room tail
  (the spectral loss can't tell a reverb tail from a longer string decay).
- Structured stays ahead of DDSP with reverb (6.28 vs 6.39, attack better on 71% of windows) and needs far less noise.
- User listening (round 2 models): structured sounds a little better than DDSP; 023 sounds very good, 176/276/277
  sound like a cheap MIDI keyboard, 306 in between. Per-recording loss tracks this (023 5.8 vs 6.7-7.7).
- Dynamics hypothesis (`analysis/dynamics_hypothesis.py`): dynamic range does not explain it; how brightness changes
  with loudness does correlate with error (rho -0.50) but the measure is likely confounded by reverb/overlap.

# Round 4: frame-by-frame features from the audio (resynthesis setting, 2026-10-27)

`src/tesis/data/note_tracks.py`: per voice, every 10 ms while a note rings, the note's loudness (energy in its partial
bins) and pitch deviation (partials 1-8 below 4 kHz, parabolic peak interpolation), cleaned with plucked-string
physics (loudness may ripple <= 6 dB but not rise again: bleed from other notes is clipped; pitch trusted only within
20 dB of the note's peak, otherwise held). Both models get them as two extra GRU inputs per voice, and the measured
pitch deviation bends each note's partials directly (time warp of the note's phase). Tracks visibly capture beating,
clean decays and the pitch drop of loud notes as they decay (tension modulation). Cached in data/processed/sr24000/tracks/.

| Model (held-out medians) | Spectral loss | Noise share | Attack LSD | 2-8 kHz LSD | <200 Hz LSD |
|---|---|---|---|---|---|
| structured + body + audio features | **5.87** | -32.3 dB | **7.58** | **8.00** | **8.64** |
| ddsp + audio features | 6.06 | -22.6 dB | 7.70 | 8.30 | 8.73 |
| structured + body (score only) | 6.22 | -30.4 dB | 7.83 | 8.39 | 9.11 |
| ddsp (score only) | 6.37 | -22.6 dB | 7.80 | 8.41 | 8.94 |

- Audio features help both models; structured + audio is best on every metric and beats ddsp + audio on 68% of windows.
- Per recording (structured, score-only -> audio): 023 5.79 -> 5.36, 176 7.43 -> 6.86 (largest gain), 276 6.60 -> 6.27,
  277 6.92 -> 6.71, 306 7.66 -> 7.38; median gain 0.23. The gap between recordings remains (176 still much worse than 023),
  so missing expression is not the whole story for the recordings that sound like MIDI.
- Listening excerpts: analysis/results/e4/listen_structured_v5_body_audio-ddsp_v5_audio-structured_v4_body-ddsp_v2/
