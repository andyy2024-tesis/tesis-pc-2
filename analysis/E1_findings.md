# Experiment 1: single-note fits (2026-10-23, 7 variants)

`scripts/e1_single_notes.py`, report `analysis/e1_report.py`, per-note results `analysis/results/e1/e1_results.csv`.

Setup: 285 isolated GAPS notes at 24 kHz (426 candidates minus failed partial fits and mislabelled notes), 1.02 s
windows, each variant optimized directly (Adam, 1500 steps, multi-scale spectral loss), no neural network.
All variants share a catch-all filtered noise (64 log-spaced bands, 5 ms frames) and a model of the previous chord ringing.

| Variant | Spectral loss | Noise share | Noise share, first 50 ms | Noise share below f0 | Params for the note |
|---|---|---|---|---|---|
| ddsp (harmonic, free amplitudes) | 3.53 | -21.3 dB | -21.9 dB | 19% | 16,400 |
| ddsp_inh (+ inharmonicity) | 3.49 | -21.8 dB | -23.3 dB | 18% | 16,400 |
| phys_nob (inharmonic, exponential decay) | 3.94 | -17.7 dB | -18.1 dB | 100% | 160 |
| phys (+ pluck transient) | 3.91 | -19.0 dB | -20.1 dB | 100% | 194 |
| phys_body (+ body modes below f0) | 3.88 | -20.7 dB | -22.2 dB | 10% | 290 |
| hybrid (phys_body + slow 25 Hz decay correction) | 3.74 | -22.5 dB | -23.5 dB | 13% | 2,370 |
| ddsp_full (ddsp_inh + transient + body) | **3.47** | **-23.3 dB** | **-25.2 dB** | 11% | 16,530 |

Paired (same notes):
- Inharmonicity helps on 66% of notes, most in the bass register (E2-E3, wound strings).
- One exponential decay per partial is clearly not enough: worse on 99% of notes vs free amplitudes (+0.45 loss, +3.2 dB noise).
  Real partial envelopes have beating / two-stage decay that the noise then has to absorb.
- The pluck transient helps on 75% of notes (-1.0 dB noise in the first 50 ms).
- Body modes help on 66% of notes; noise share below f0 drops from 100% to 10% (the ~100 Hz air resonance).

- The slow correction fixes most of what plain exponential decay misses: better on 99% of notes vs phys_body
  (-0.15 loss, -1.5 dB noise). With 7x fewer numbers than DDSP it leaves less to the noise than plain DDSP
  (-22.5 vs -21.3 dB) but still fits worse (3.74 vs 3.53): faster envelope detail (attack, fast beating) is missing.
- All structured parts on flexible amplitudes (ddsp_full) vs plain DDSP: better fit on 86% of notes, noise share
  -1.5 dB overall and -3.1 dB in the first 50 ms. Best variant on every measure.

Lessons for the full model:
- Component stealing is real: unconstrained body modes (60-800 Hz, linear amplitudes, unbounded decay) took >90% of the energy.
  Fixed by log amplitudes, random fixed phases, decay bounded to 2-50 1/s and (for single notes) modes only below f0.
- DDSP's linear noise bands are ~185 Hz wide at 24 kHz; log-spaced bands are needed for the low end.
- Body modes + MIDI pitch expose mislabelled notes (energy far below the MIDI fundamental).
- Amplitudes: keep them flexible (network-predicted per frame) but on top of a physical decay baseline
  (log amplitude = decay line + learned correction). Per-note fits can't show the baseline's main benefit
  (making multi-second decays easy for a network to learn); that is tested in the network experiments.
- Caveats: single notes only, body fitted per note (not shared per recording), energy-based noise share.
