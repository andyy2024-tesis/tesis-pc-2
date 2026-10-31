# Experiment 3: per-note loudness from a mix vs per-string truth (GuitarSet, 2026-10-26)

`scripts/e3_guitarset.py`, report `analysis/e3_report.py` -> `analysis/results/e3_report.txt`,
per-note data `analysis/results/e3/note_levels.csv`.

Data: 6 GuitarSet tracks (players 01-05, all five styles, 3 comp / 3 solo, 825 notes), steel-string acoustic.
Audio from the jhartquist/guitarset Hugging Face mirror (Zenodo was ~1 MB/s; player 00's shard wouldn't download),
annotations from Zenodo. Truth: energy in each note's partial bins 20-105 ms after onset, read from its own
string channel (hex pickup, debleeded). The model (experiment 2's whole-recording synthesizer: smooth string
spectrum, body filter, 2 ms attack, per-note pluck; strings as voices; 1500 steps) and the naive estimator
(same bins, read from the mix) both only see a mix.

Steel-string inharmonicity measured on the hex channels: log B = -7.17 - 0.038 * pitch, median B 8.6e-5
(decreasing with pitch across strings, unlike nylon within GAPS).

| Mix | Model median err | Naive median err | Model p90 | Naive p90 | Model corr | Naive corr |
|---|---|---|---|---|---|---|
| hexsum (sum of the string channels) | 1.44 dB | 0.15 dB | 8.6 dB | 2.5 dB | 0.76 | 0.95 |
| mic | 3.35 dB | 2.88 dB | 10.9 dB | 9.4 dB | 0.66 | 0.71 |

With a fixed per-string offset removed (pickup sensitivity per string; the mic hears body/room): hexsum model
1.26 / naive 0.12 dB, mic model 2.48 / naive 1.97 dB. Notes with 4+ sounding on hexsum: model 1.55, naive 0.29 dB.

Conclusions
- On real audio the analysis-by-synthesis model does NOT beat reading the mix directly. The near-perfect recovery
  of experiment 2 (0.02 dB) was within the model family; with real notes, model mismatch dominates.
- Model errors are worse for short notes (<0.3 s: ~2 dB vs 1.1 dB for longer) and mostly too quiet (70% of the
  worst 10%): likely the shared per-pitch decay and smooth spectrum, and energy in shared bins (octaves/unisons)
  given to notes already ringing.
- Good news for GAPS: a simple direct measurement from the mix gives per-note levels within ~0.3 dB of the
  per-string truth even with 4+ notes sounding (hexsum condition; p90 4 dB). The hexsum condition favors the naive
  estimator (same bins as the truth; exact when a note sounds alone), so this is an upper bound; the mic numbers
  are pessimistic because the truth comes from a different transducer.
- Design decision: estimate note loudness directly from the mix in preprocessing and give it to the network as an
  input feature (like Jonason's loudness), instead of asking the synthesizer to infer it.
- Caveats: 6 tracks, steel strings (GAPS is nylon), one estimator family for truth and naive.
