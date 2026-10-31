# GAPS audit findings (2026-10-23)

Produced by `analysis/gaps_audit.py` (per-recording CSV in `analysis/results/gaps_audit.csv`) and
`analysis/tab_ambiguity.py`. Complements `GAPS_data_notes.md` (syncpoint/MIDI format notes).

## Contents of the download (`data/GAPS`, 16 GB)

- 404 WAVs, 404 MIDI, 405 MusicXML, 404 syncpoint files, 95 `.match` files. Metadata CSV has 401 rows.
- Splits: 270 train / 30 test (the 300 recordings with alignment f-measure >= 0.75, about 16.1 h).
  101 rows have no split (rejected alignments, 6.6 h); some train files still have very low f-measure (min 0.03).

## Audio

- 48 kHz PCM16, mostly stereo (median L/R correlation 0.88; we downmix to mono). 10 files are 44.1 kHz.
- Decoded from YouTube: codec low-pass cutoff around 20 kHz for most files, about 8% below 16.5 kHz (one ~11 kHz).
- No meaningful clipping. Many files contain digital silence (padding).

## Tuning and pitch

- Per-recording tuning offset vs MIDI pitch (measured from partials 1-3 of notes): within +/-42 cents; 28 files beyond +/-20 cents.
  Within-file spread is about 4 cents, so one offset per recording is enough.
- 9 files have MIDI transposed vs audio by 1-2 semitones (chroma test). Fix or exclude.

## MIDI vs audio onsets

- 95% of MIDI onset groups have an audio onset within 150 ms.
- Median residual +12 ms (likely detector lag, so mostly a constant bias); per-file IQR median 10 ms, 90th percentile 40 ms.
- Within 25 ms: median file 83%, worst 10% of files below 47%. Fine for transcription, too loose for attack synthesis: needs per-note refinement.
- MIDI velocity is a constant placeholder (see `GAPS_data_notes.md`).

## Polyphony

- Mean 2.3 notes sounding when anything sounds (MIDI durations, which underestimate guitar ring time).
- Only 29% of active time is monophonic by MIDI durations.
- Truly isolated notes (single onset, nothing else within -0.5/+1.0 s): only 435 in the whole dataset.

## Strings (tablature)

- MusicXML part P2 is tablature with `<string>`/`<fret>` per note.
- Pitch alone determines the string for only 22.5% of MIDI notes, so string assignment needs score-to-MIDI note alignment
  (syncpoints + pitch matching, repeats unrolled) or a fingering heuristic.
