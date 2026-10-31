# Experiment 2: one polyphonic piece (2026-10-23)

`scripts/e2_one_piece.py` with `src/tesis/synth/piece.py`; outputs in `analysis/results/e2/088_Yf1wc/`
(metrics.json, note_levels.csv, audio/*.wav).

Piece: 088_Yf1wc, Tárrega "Lágrima" (Rafael Elizondo), 108 s, 321 notes, 24 kHz. Direct optimization, no network,
3000 steps x 4 random 3 s windows (~6 min on the RTX 4090).

Model: per recording, a log-amplitude template and decay per (pitch, partial), stiff-string frequencies with
B(pitch) from experiment 1, release damping after the MIDI offset, pluck transient, 48 body modes (60-800 Hz)
excited at every onset, stationary noise floor, framewise catch-all noise. **Per note only 2 numbers: loudness and
brightness.** Notes ring until their voice is reused (max 4 s).

## Real fit

- Spectral loss 5.29 on 3 s polyphonic windows (single notes in experiment 1: ~3.5 on 1 s windows).
- Energy shares: strings -0.1 dB, body -15.9 dB, noise floor -33 dB, catch-all noise -39 dB, transient -71 dB (unused).
- Model note levels vs a naive level read from the audio: correlation 0.77; note levels span ~22 dB (p5-p95).

## Synthetic recovery (the GAPS concern: per-note loudness from the mix)

Re-rendered the piece with known per-note gains (fitted gains + N(0, 4 dB)), refit a fresh model from scratch:

| | Correlation | Median error | 90th pct error |
|---|---|---|---|
| Model (analysis-by-synthesis) | 0.999 | 0.02 dB | 0.27 dB |
| Naive (harmonic bins from the mix spectrogram) | 0.88 | 0.88 dB | 3.7 dB |

Same picture for notes alone (n=40), with 2-3 notes sounding (n=234) and 4+ (n=47): model ~0.02 dB, naive ~0.8-1.2 dB.

Caveat: the synthetic target comes from the same model family, so this shows the mix keeps the note-level
information *when the model structure is right*. It does not show accuracy on real audio; that is step 3
(GuitarSet, which has per-string recordings as ground truth).

## Problems found

- Digital silence (exact zeros, ~5% of GAPS audio, here the first/last 5.4 s) makes the log-spectral loss explode
  and drags the noise floor to zero. Windows with any silent 10 ms frame are now excluded. Should move into the data pipeline.
- Body modes may be stealing partials of frequent notes: strongest modes at 171, 487, 213 Hz sit within a few %
  of E3, B4, G#3, all common in this piece (they could also be real top-plate modes; can't tell from one piece).
  No mode near the ~100 Hz air resonance seen in experiment 1.
- The pluck transient collapsed (decay time 0.2 ms). In experiment 1 it helped. Likely reasons: the strings'
  abrupt start already makes a click, and about 44% of onsets are not sample-refined.
- The per-frame catch-all noise learns slowly in direct optimization (each frame is only updated when its window
  is sampled); not an issue once a network predicts it.

# Follow-up: body stealing and the transient (2026-10-23 to 10-26)

Scripts: `analysis/e2_diagnostics.py`, `analysis/e2_body_consistency.py`, `analysis/e2_body_filter_report.py`;
new PieceConfig options `attack_ms`, `pluck_per_note`, `seed`, `source`, `body`.

## Transient
- The strings started every partial at sin(0); with ~1/k amplitudes that sums to a sawtooth-like step, i.e. an
  automatic click at every onset. On clearly plucked notes it alone reproduced the real high-frequency onset rise
  (+14.7 vs +14.9 dB), so the burst component had nothing to do and collapsed.
- On the 93 notes without a clear attack (likely slurs: hammer-ons/pull-offs), real audio rises +2.0 dB but the
  model clicked anyway (+4.6 dB).
- With a 2 ms smooth string attack and a per-note pluck level: unclear-attack notes +2.5 dB (real +2.0), plucked
  notes +12.9 dB (real +14.9, ~2 dB short). Pluck levels are ~3 dB lower on unclear-attack notes (corr with onset
  clarity 0.41), consistent with slurs. Burst becomes a 0.7 ms click. Overall spectral loss unchanged (5.31): the
  multi-scale spectral loss barely weighs transients, so a transient-focused loss term will be needed for fidelity.

## Body
- Onset-triggered free body modes were NOT stealing note partials (no piece's mode profile follows its own notes),
  but they were an artifact: with different random mode phases the same piece learns unrelated modes
  (correlation 0.1 / -0.07; loss identical). The apparent agreement across pieces came from a shared seed.
- Cause: source-filter ambiguity; a free template per (pitch, partial) can explain any body resonance.
- Body as a shared frequency response on every partial (absolute Hz) + smooth string spectrum (3 numbers per pitch):
  stable across seeds (0.98), loss 5.42 vs 5.36 with far fewer parameters.
- Across pieces: the broad shape (1/2-octave smoothed: rise from ~100 Hz to a hump at 400-600 Hz, roll-off above)
  agrees (0.79 same performer, 0.67 other performer). Fine peaks (many of 15-25 dB, mostly above 600 Hz) don't agree
  at all (0.05 / 0.16): they absorb piece-specific detail (likely pluck-position comb or template limits), not the body.
- Design decision: body = shared, smooth (~1/2-octave resolution) frequency response; fine spectral detail belongs
  to the string model (e.g. pluck-position comb), not the body.
