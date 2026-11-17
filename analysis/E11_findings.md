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
