# Experiment 12: final DDSP model, then a neural vocoder that refines it

Plan (approved 2026-11-17 with V3b, stems and the 44.1 kHz phase): `analysis/E12_vocoder_plan.md`.
User's question on approval: why teach the vocoder guitar on real audio instead of DDSP renders of all 270 recordings?
Answer: phase 2 teaches the last step (mel -> waveform) that every mel route needs, and real mels are what V3's output
looks like; DDSP pairs are phase 3. More DDSP pairs need a DDSP trained on those recordings (per-recording parts) with
good note alignment and tab strings: 80 more training-split recordings pass (transcription f >= 0.93, tab matched >= 0.88
like the 20; `analysis/results/e12_more_ids.txt`), so a 100-recording DDSP gives ~5 h of pairs instead of 0.73 h. V1 is
trained on both to test whether pair count limits.

## Final DDSP model (`analysis/results/queue_final`)

Ear verdicts on experiment 11 picked T4adv (attention that looks ahead + contact + adv) and strings on M1; the
recording in every part with a 32-dim embedding; stiffness not measured from recordings. Two versions:

| run | stiffness per guitar | loss (non-adv, step 4000) |
|---|---|---|
| `structured_v12_final` | none (one table per string) | 5.69 |
| `structured_v12_final_learnedB` | learned from the embedding (per string, bounded x0.61-1.65) | 5.71 |
| T4 (`structured_v11_m1_attn_ahead_contact`), for reference | - | 5.73 |

Adversarial fine-tunes (same recipe as T4adv: complex D, 64 ch, D lr x2, 3000 steps, rise penalty 20):

| run | loss | wind (dB) | high clicks in the excerpts |
|---|---|---|---|
| `e6/structured_v12_final_noise_strongD_rise` | 5.81 | -20.6 | 0 |
| `e6/structured_v12_final_learnedB_noise_strongD_rise` | 5.82 | -23.9 | 0 |
| T4adv, for reference | 5.83 | -24.3 | 2 |

**Final DDSP: `e6/structured_v12_final_learnedB_noise_strongD_rise`** (learned stiffness, as the user wanted; loss tie,
less wind than the shared-stiffness version, no high clicks). Its renders of the 20 recordings (mix + parts, aligned
sample by sample with the real audio) are the vocoder pairs: `analysis/results/e12/renders/<run>/`.

A tie (seed spread ~0.05). What the learned version learned: shifts with a spread across guitars of 0.21 (log), as wide
as the measured per-guitar stiffness (0.22), and correlated with it (r = 0.28 over 120 guitar-string pairs): the
embedding found part of the real stiffness differences by itself, through the partial frequencies.

## Phase 1: ceilings, no training

Pretrained BigVGAN-v2 24 kHz on the 85 held-out windows, real mel -> audio vs the real audio: loss 4.24 (the ceiling of
a mel vocoder on this measure; no DDSP model has gone below 5.4). Listening excerpts (9 s), loss vs the real excerpt:

| excerpt | real -> BigVGAN | favourite | favourite -> BigVGAN | T4adv | T4adv -> BigVGAN |
|---|---|---|---|---|---|
| 023 | 4.22 | 5.37 | 5.73 | 5.36 | 5.81 |
| 176 | 4.11 | 7.16 | 7.29 | 5.93 | 6.10 |
| 276 | 4.29 | 6.31 | 6.52 | 6.38 | 6.58 |
| 277 | 4.40 | 6.23 | 6.47 | 6.24 | 6.47 |
| 306 | 4.52 | 6.67 | 6.91 | 6.66 | 6.85 |

Untrained on DDSP input, the vocoder only redraws the DDSP spectrum (+0.1-0.4): it cannot repair what it never saw.

On the 85 held-out windows (`analysis/e12_voc_metrics.py`; 44.1 kHz outputs resampled to 24 kHz and compared with the
real 24 kHz window; the 44.1 kHz input is the raw 48 kHz GAPS file at the same place):

| vocoder (pretrained) | real -> vocoder | final DDSP -> vocoder |
|---|---|---|
| none | 0 | 5.82 |
| BigVGAN-v2 24 kHz | 4.24 | 6.11 |
| BigVGAN-v2 44 kHz | 4.40 | 6.03 |
| DisCoder 44.1 kHz | 4.46 | 6.11 |

Listening page 6 (https://claude.ai/artifact/6du45ZyZdAEwrmTRSJWDrM): the final DDSP next to the favourite and T4adv;
the 24 kHz ceiling; the 44.1 kHz vocoders. The parts of a vocoder version are what went in, what it added and what it
removed, from spectrogram magnitudes: a vocoder keeps the timing (envelope lag < 1.2 ms) but not the waveform's phase
(sample correlation with its input -0.2 to 0.25), so subtracting waveforms would give the whole piece back.

## Phase 2: BigVGAN-v2 24 kHz on real GAPS audio

Memory: BigVGAN's anti-aliased activations (2x upsampled, at every layer) are huge; without help even batch 4 x 32768
samples ran out of 24 GB. With gradient checkpointing of each AMP block (recomputed in the backward pass;
`scripts/e12_vocoder.py`, on by default) batch 8 x 32768 fits (20 GB) at 1.3 s/step, so 40000 steps would take 14 h:
cut to 8000 steps (~3 h), evaluations every 2000. After 100 steps the held-out loss already went 4.24 -> 4.10.
(A first restart was stopped by a queue watcher that matched an old "end phase2" line; the night queue is now one
sequential script, `queue_final/night.sh`: phase 2, then V1 on the 20-recording pairs, then the 100-recording DDSP.)

The 100-recording DDSP needs ~20 GB (vs ~7 GB for 20 recordings: the peak comes from the densest windows of the extra
recordings, not from per-recording data), so it cannot share the GPU with a vocoder run; it goes last.

Result (85 held-out windows; real audio -> mel -> fine-tuned vocoder): 4.238 (pretrained) -> 4.146 / 4.101 / 4.103 /
4.082 at 2000 / 4000 / 6000 / 8000 steps. The final DDSP through the fine-tuned vocoder: 5.891 (pretrained vocoder
6.105, raw DDSP 5.82). Fine-tuning on real guitar alone, without ever seeing a DDSP mel, already narrows the gap that
the vocoder adds on top of the DDSP from 0.29 to 0.07. Run `p2_real` is the starting point of every trained route.

## V3: shallow diffusion on the mel

First run (`v3_shallow`, DiffSinger's noise prediction, 50% unpaired real batches): at 10000 steps the reverse process
from depth 20 gives 8.93 and from depth 40-80 27-29 (depth 0, the DDSP mel through the pretrained vocoder: 6.09).
Diagnosis: the noise error stays ~0.15 even at heavy noise, where the DDSP condition should make it easy, so the
clean-mel estimate at depth 20 (MSE 0.023 vs the real mel) is already worse than the DDSP mel itself (0.009), and the
error grows along the reverse chain (values up to +-4 at depth 60). Second run `v3_x0res`: the network predicts the
clean mel as the DDSP mel plus a correction (zero-initialised, so it starts as the DDSP mel), reverse step by the DDPM
posterior mean. Then V3b (flow from the DDSP mel; also starts as the identity).

Results (vocoded by the pretrained BigVGAN; identity = the DDSP mel itself, 6.09):

| run | step 10000 | 20000 | 30000 | 40000 |
|---|---|---|---|---|
| V3 noise prediction, depth 20 / 40 / 80 | 8.93 / 27.3 / 28.1 | 8.01 / 10.9 / 29.3 | 7.69 / 9.70 / 25.5 | - |
| V3 x0res (first run), depth 20 / 40 / 80 | 6.28 / 6.34 / 6.39 | (crashed saving: rerun) | | |
| V3 x0res (rerun), depth 20 / 40 / 80 | 6.49 / 6.57 / 6.62 | 6.28 / 6.30 / 6.32 | 6.30 / 6.33 / 6.34 | 6.30 / 6.33 / 6.34 |
| V3b flow (no source noise) | 6.12 | 6.13 | 6.17 | 6.18 |
| V3b flow, source noise 0.3 + bridge 0.3 | 7.89 | 7.82 | 7.78 | 7.78 |

Texture kept (frame-to-frame variance of the mel / the real mel's; DDSP mel 0.90): x0res 0.46 / 0.48 / 0.49 at depth
10 / 20 / 40; flow 0.27; flow with source noise 10.5 (it never learns to remove the starting noise; also its target
ignores the bridge noise). Both mel refiners trained by squared error learn the average correction, and the average is
smooth: half the texture or less, so the vocoder makes duller audio. Mel-domain repair as set up here does not beat
the DDSP mel on this measure; the GAN routes (V1, V2, S), whose adversarial losses reward texture, come next.

Why the flow gets slightly worse: its mels are closer to the real ones (mel MSE 0.0125 vs 0.0145 for the DDSP mel) but
smooth: their frame-to-frame variation is 27% of the real mel's (DDSP mel: 90%). Trained on MSE from a fixed start,
the flow learns the average correction, which washes out texture, and the vocoder turns a smooth mel into dull audio.
Fix being tried: start from the DDSP mel plus noise (`--src-noise 0.3`, bridge 0.3) so the flow has to create texture.
Every evaluation now also reports the texture ratio.

Re-evaluated with the fine-tuned vocoder (`p2_real`; `--eval-only`): identity 5.880 (texture 0.90), x0res depth 20
6.078 (0.48), flow 5.874 (0.27). With a vocoder that has heard real guitar, the flow ties the identity in spite of
its smooth mel; x0res stays worse. Excerpts of both through `p2_real` are on page 6 for the ear (envelope lag after
mel repair up to 2.3 ms, mixed signs: smeared attacks, not a shift; the media script now allows 5 ms). Caveat on the measure: the spectral loss compares with one specific
recording, so a refiner that adds realistic but different detail is penalised; for these the ear decides.
Also rendered: BigVGAN-v2 44 kHz (from the full-band 48 kHz excerpt and from the DDSP upsampled) and DisCoder (44.1 kHz,
music-trained). `analysis/e12_vocode.py`, `analysis/e12_discoder.py` (own environment in external/discoder/.venv).

## V1: DDSP mel -> real audio (from phase 2), 20-recording pairs

`v1_mix20`: the fine-tuned vocoder trained 6000 more steps (batch 8 x 32768, ~2 h) on pairs: the final DDSP's mel of a
window in, the real recording of that window out (BigVGAN's mel + adversarial + feature-matching losses). Held-out
loss 5.891 (step 0 = `p2_real` on the DDSP) -> 5.814 / 5.817 / 5.796 at 2000 / 4000 / 6000. The first vocoder that
does not make the DDSP worse (raw DDSP 5.82), but a tie by this measure. The training mel loss kept falling
(1.56 -> 1.39) while the held-out loss stalled after step 2000: with 0.69 h of pairs the vocoder starts memorising
them, which is what the 100-recording DDSP (~5 h of pairs) tests next.

Timing: attack-level lag (cross-correlation of onset strengths, 32-sample resolution) V1 vs its DDSP input within
2.7 ms, and V1 vs the real recording 0 in 3 of 5 excerpts; in 277 the DDSP's attacks are 64 samples off the real ones
and V1's are not, so V1 moves attacks toward the real timing. Plain vocoders: 0 on all excerpts. (The earlier
whole-excerpt envelope lag wandered by up to 6 ms even between the DDSP and the real audio, because the decays
dominate it; the media script now checks attacks.)

Queue after V1 (night.sh -> ddsp100_run.sh): V2 (`v2_inject20`, the DDSP waveform injected NSF-style), S
(`s_stems20`, the parts as extra mel channels), both on the same pairs from `p2_real`; then the 100-recording DDSP,
its adversarial fine-tune, its pair renders, and V1 on those pairs (`v1_mix100`).

## V2: the DDSP waveform injected (NSF-style), 20-recording pairs

`v2_inject20`: as V1 (from `p2_real`, same pairs, 6000 steps) plus the DDSP mix waveform entering every upsampling
stage through zero-initialised strided convolutions. Held-out loss 5.891 (step 0) -> 5.794 / 5.790 / 5.762 at
2000 / 4000 / 6000, still falling. The first vocoder more than 0.05 better than the raw DDSP (5.82); against V1
(5.796) a tie. With the waveform the vocoder can copy the DDSP's phase and fine detail instead of inventing them.
Page 6 section "vocoder-v2". (Page 6 now has 443 files: S fits; the 100-recording results go on a page 7.)

## S: first run diverged

`s_stems20` (the parts as extra mel channels) was fine until step 400, then blew up as the learning rate warmed up
(mel loss 1.6 -> 12.9 by step 1600, discriminator loss -> 0.02; held-out 58.6 at step 2000) and was stopped. Cause:
log-mels floor at log(1e-5) = -11.5, so a silent part is a large constant over all its 100 channels; once the
zero-initialised input layer moves, that constant swamps the vocoder (V2's injected waveform stays within +-1). Fix
in `Refiner.forward`: the part mels enter as (log-mel + 11.513) x 0.1 (silence 0, loud parts ~1). Rerun
`s_stems20b` at the start of `phase45.sh`.

## Measuring realism: KAD instead of the spectral loss (decided with the user, 2026-11-18)

The spectral loss compares with one specific recording, so it rewards a smooth average and penalises realistic but
different detail; in the literature it also misranks systems against listeners (Jonason et al. 2023, polyphonic
guitar DDSP: a lower MSSL with a significantly lower MOS; Kandpal et al. 2022: spectral distances correlate 0.40-0.56
with MOS and miss added noise). FAD depends on the embedding and is biased at small N; KAD (Chung et al. 2025, MMD
with a Gaussian kernel) is unbiased at small N. Setup (`analysis/e12_kad_clips.py`, `analysis/e12_kad.py`, kadtk in
its own environment `external/kadtk/.venv`):
- clips: the same 85 held-out 3 s windows, every clip at RMS -23 dBFS;
- embeddings: PANNs Wavegram-LogMel (one 2048-vector per clip; the KAD authors' choice) and EnCodec 24 kHz (one
  128-vector per second; codec features, sensitive to distortion and low-pass). CLAP left out: it pads clips to 10 s;
- references: "real" (the same windows; paired) and "realtest" (600 clips of the 30 GAPS test-split recordings);
- kernel bandwidth from the reference for every version (kadtk takes it from the evaluated set);
- U-statistic over distinct clips (a clip is never compared with its own real window: the paired cross term made
  kadtk's estimate negative); 1000 paired bootstrap resamples, 95% intervals for each score and each difference.
- The bigger test set (all ~7 min of held-out audio instead of the 85 windows' 4.25 min) is not needed for KAD (no
  small-sample bias) and would narrow intervals only ~25%; kept as an option for close calls.

KAD x 100 (lower = closer to real), [95% interval]; spectral loss for comparison:

| version | MSS | PANNs vs same windows | PANNs vs unseen real | EnCodec vs same windows |
|---|---|---|---|---|
| final DDSP | 5.82 | 3.21 [2.26, 4.47] | 6.74 [5.14, 8.56] | 0.176 [0.080, 0.290] |
| pretrained BigVGAN on DDSP | 6.11 | 1.78 [1.29, 2.46] | 4.42 [2.64, 7.07] | 0.264 [0.153, 0.388] |
| fine-tuned BigVGAN (p2) on DDSP | 5.89 | 1.82 [1.30, 2.50] | 4.23 [2.46, 6.74] | 0.221 [0.120, 0.330] |
| mel repair, diffusion (x0res d20) | 6.08 | 6.96 [5.60, 8.57] | 10.16 [7.45, 13.41] | 0.595 [0.403, 0.862] |
| mel repair, flow | 5.87 | 7.33 [6.18, 8.63] | 13.14 [9.88, 17.07] | 0.400 [0.255, 0.605] |
| V1 | 5.80 | 2.33 [1.54, 3.37] | 3.69 [2.15, 5.86] | 0.123 [0.051, 0.204] |
| V2 | 5.76 | 1.35 [0.89, 2.00] | 3.12 [1.71, 5.22] | 0.183 [0.099, 0.295] |
| real -> pretrained BigVGAN | 4.24 | 1.75 [1.35, 2.26] | 3.43 [1.76, 5.73] | 0.020 [0.013, 0.028] |
| real -> fine-tuned BigVGAN | 4.08 | 1.46 [1.06, 1.97] | 3.04 [1.47, 5.45] | 0.029 [0.020, 0.039] |
| real (held-out windows) | - | - | 2.03 [0.89, 3.82] | - |

- The mel refiners, a tie by MSS, are the least real by every KAD (significant): the smooth mels sound unreal.
- The pretrained vocoder, worse than the DDSP by MSS, is much closer to real by PANNs (difference -1.44 [-2.94,
  -0.06]) but slightly further by EnCodec (+0.088 [+0.050, +0.126]): it makes the sound more guitar-like overall and
  changes fine acoustic detail away from the recording.
- V2 vs V1: PANNs, same windows -0.98 [-1.57, -0.53] (V2 better); unseen real -0.57 [-1.29, +0.07]; EnCodec +0.06
  [-0.05, +0.17] (tie). V2 reaches the vocoder's own ceiling on real audio (1.46).
- EnCodec against unseen real cannot tell real held-out audio (1.20) from the DDSP (1.35): recording and piece
  differences dominate it; not used for decisions.
- DDSP models (rendered window by window as their loss, `model-<key>` in e12_kad_clips.py; the final DDSP this way
  3.220 vs 3.215 from its pair renders, a check that both paths agree). PANNs vs same windows / EnCodec:
  favourite M1 + timbre 32 + adv 1.54 / 0.171 (loss 5.93), T4adv 1.99 / 0.212 (5.83), final DDSP learned + adv
  3.22 / 0.177 (5.82), final shared + adv 3.67 / 0.257 (5.81), final learned without adversarial 7.22 / 0.187 (5.71).
  The loss ranks the favourite last; PANNs ranks it most real (favourite - final -1.68 [-2.61, -0.92]); EnCodec
  sees no difference. The final DDSP (strings + recording everywhere) is less real by PANNs than the DDSP the user
  preferred by ear, and V2 on it (1.35) only reaches the favourite alone. The adversarial fine-tune is what makes
  any DDSP real by PANNs (7.22 -> 3.22).
- Listening page: every model table shows loss and KAD (PANNs · EnCodec); page 6 section "medir-realismo".
- Phase 4/5 winner rule now by KAD (`phase45.sh`): V2 or S replace V1 only if the PANNs (same windows) difference to
  V1 is below 0 with its whole interval and EnCodec does not say worse; same for the 100-recording pairs. On today's
  numbers: V2, 20-recording pairs.

## Phase 4 and 5 prepared (run after the night queue: `queue_final/phase45.sh`)

Winner rule (the ear has not judged yet; loss differences under 0.05 are ties): V2 or S only replace V1 if at least
0.05 better at their last evaluation; the 100-recording pairs only if V1 on them is at least 0.05 better than on the
20. Phase 4: the winner fed the harmonic part only (`--input strings`). Phase 5: `--sr 44100` in
`scripts/e12_vocoder.py` (BigVGAN-v2 44 kHz, 128 bands; real audio at 44.1 kHz from `scripts/e12_prep44k.py`, same
soxr recipe as the 24 kHz files, 270 recordings, 4.4 GB; positions and renders stay at 24 kHz, the DDSP is upsampled
per segment, the output is scored at 24 kHz). Smoke tests (pretrained, CPU): real 4.382 and DDSP 6.014 (phase 1 with
polyphase resampling of the raw file: 4.402 and 6.033); the 24 kHz path is unchanged (5.891).
Then `p5_real44` (as phase 2, 8000 steps; retried at batch 4 if 8 does not fit) and the winner at 44.1 kHz from it.

## S rerun, phase 4 and the 44.1 kHz vocoder (2026-11-18 afternoon; page 7)

Run order: `s_stems20b` -> KAD pick (V2, 20-recording pairs) -> `p4_v2_strings20` -> `p5_real44` -> `p5_v2_mix20_44k`
(killed at step ~2000 by a machine reboot at 21:24 and restarted from zero by `queue_final/restart.sh`, followed by the
re-queued data100 adversarial, renders and V1-100). KAD x 100 against the same 85 windows, [95% interval]:

| version | MSS | PANNs | EnCodec |
|---|---|---|---|
| V2 (DDSP mix + waveform) | 5.76 | 1.35 [0.89, 2.00] | 0.183 [0.099, 0.295] |
| S rerun (mix + each part's mel) | 5.76 | 2.28 [1.52, 3.25] | 0.450 [0.321, 0.590] |
| phase 4: V2 with the strings part only | 6.17 | 1.18 [0.79, 1.73] | 0.150 [0.067, 0.249] |
| DDSP -> BigVGAN 44k pretrained | 6.03 | 2.15 [1.55, 2.97] | 0.384 [0.254, 0.527] |
| DDSP -> BigVGAN 44k tuned on real (p5_real44) | 6.00 | 2.15 [1.49, 3.06] | 0.455 [0.319, 0.605] |
| real -> BigVGAN 44k pretrained | 4.40 | 2.38 [1.82, 3.13] | 0.068 [0.055, 0.083] |
| real -> BigVGAN 44k tuned (p5_real44) | 4.49 | 2.16 [1.61, 2.83] | 0.254 [0.225, 0.283] |
| real -> BigVGAN 24k tuned (p2, for reference) | 4.08 | 1.46 [1.06, 1.97] | 0.029 [0.020, 0.039] |

- S: the fixed input scaling trained normally (5.80 / 5.81 / 5.76) but S is less real than V2 by both embeddings
  (PANNs +0.93 [+0.48, +1.48], EnCodec +0.27 [+0.13, +0.40]); the pick stayed V2.
- Phase 4: worst by MSS, tied with full V2 by KAD (PANNs -0.18 [-0.66, +0.19], EnCodec -0.03 [-0.14, +0.07]; unseen
  real +0.18 [-0.52, +0.87]). The vocoder invents body, attacks and noise as convincingly (for these embeddings) as the
  DDSP's own parts; MSS penalises it because invented detail does not match the recording note by note. If the ear
  agrees, the DDSP in front of the vocoder could be much simpler.
- Phase 5: tuning BigVGAN-v2 44k on real GAPS audio (same recipe as phase 2) made it worse, unlike at 24 kHz: real
  audio 4.38 -> 4.50 / 4.46 / 4.45 / 4.49 (steps 0/2000/4000/6000/8000); EnCodec on real 0.068 -> 0.254 (+0.19
  [+0.16, +0.21]); PANNs a tie. Band energies of real excerpts through it stay within 1 dB of the original from 0 to
  22 kHz (no band lost), so the loss is fine detail. Both metrics are computed at 24 kHz, so neither hears the band
  above 12 kHz, the only thing the 44.1 kHz path adds; up to 12 kHz the 44.1 kHz vocoders are less real than the
  24 kHz ones. Not investigated further (learning rate, data resampling) until the ear says the top band matters.
- `analysis/e12_kad_clips.py` now handles 44.1 kHz runs (real window at 44.1 kHz or DDSP upsampled, output back to
  24 kHz) and `voc-pretrained44[-real]`; `analysis/e12_kad.py` takes extra paired baselines from `KAD_BASES`.

## The user's ear on page 6 (2026-11-19) and what followed

- By ear V1 and V2 are by far the best ("finally some realism", good room acoustics); favourite V2. Mel refiners
  (flow, diffusion) sound very ugly. The vocoders tuned only on real audio pass the DDSP's wind through and barely
  improve the timbre. Agrees with KAD on every point. V2's parts: it adds room/reverb ("through a thick wall") and
  removes wind and high harmonics.
- Favourite DDSP by ear: the final DDSP (learned stiffness + adv), not M1 + timbre 32 + adv that KAD ranks first; the
  user's guess is that the earlier favourite is too clean for the per-window loss. Decision: train V2 on the pairs of
  M1 + timbre 32 + adv anyway (`v2_fav20`, queued in `restart.sh` after V1-100).
- 44.1 kHz: hard to tell from 24 kHz (276_jw1wc has more detail at 44.1 kHz); to be discussed before more work.
  V2 at 44.1 kHz (`p5_v2_mix20_44k`, from p5_real44): loss 6.00 / 6.03 / 6.07 / 6.01 (steps 0-6000; no gain over its
  input, unlike 5.89 -> 5.76 at 24 kHz); KAD PANNs 1.68 vs 1.35 for V2 at 24 kHz (+0.33 [-0.02, +0.77]), EnCodec 0.43
  vs 0.18 (+0.25 [+0.11, +0.39]). Up to 12 kHz the whole 44.1 kHz route is worse.
- "contact sounds like noise, not contact; is it a catch-all and the culprit?" (`analysis/e12_contact_check.py`, pair
  renders of the 20 recordings, medians): contact is -41 dB of the mix (-29 dB in the quietest 20% of frames), centroid
  2.6 kHz, its energy within 50 ms of an onset/offset only 1.3x chance (burst 3.2x, body 2.6x), uncorrelated with the
  noise part (r = 0.03). So it is a quiet high hiss, not finger contact. But it is not the culprit: KAD without it is
  much worse by PANNs (3.22 -> 8.70, +5.48 [+4.12, +7.09]; unseen real 6.74 -> 14.02), EnCodec unchanged; without the
  noise part a tie (3.27), without the room floor slightly worse (3.61). The log-mel embedding needs something in the
  upper bands, which the strings leave nearly empty; real recordings have hiss there.

## Last night's runs and the DDSP chain by KAD (2026-11-19 morning; page 7)

KAD x 100 against the same 85 windows (PANNs / EnCodec), loss at the last evaluation:

| version | MSS | PANNs | EnCodec |
|---|---|---|---|
| V2 on the final DDSP (v2_inject20) | 5.76 | 1.35 | 0.183 |
| V2 on M1 + timbre 32 + adv pairs (v2_fav20) | 5.79 | 1.64 | 0.337 |
| V1 on 20 recordings (v1_mix20) | 5.80 | 2.33 | 0.123 |
| V1 on 100 recordings (v1_mix100, its own DDSP) | 5.75 | 3.38 | 0.515 |
| final DDSP, 100 recordings + adv | 5.72 | 3.26 | 0.116 |

- V2 on the KAD-preferred DDSP is not better: vs V2 on the final DDSP PANNs +0.29 [-0.11, +0.69], EnCodec +0.15
  [+0.01, +0.29]. The vocoder barely improves that DDSP (1.54 -> 1.64) but takes the final DDSP from 3.22 to 1.35:
  what matters is how DDSP and vocoder complement each other, not how real the DDSP sounds alone. The user's ear
  (final DDSP preferred) was the better guide; V2 on the final DDSP stays the best result.
- 100 recordings: the DDSP is as real by PANNs (3.26 vs 3.22) and slightly better by EnCodec (0.12 vs 0.18); loss
  5.72 vs 5.82. V1 on its pairs gains by loss (5.75 vs 5.80) but is less real by KAD (PANNs +1.04 [+0.24, +1.96],
  EnCodec +0.39 [+0.21, +0.57], unseen real +2.22 [+0.84, +4.06]). Possible reason (not tested): with 80 other guitars
  and rooms the vocoder learns an average sound instead of each recording's. The 20-recording pairs stay.
- DDSP chain (all adversarial, PANNs vs same windows): M1 + timbre 32 + adv 1.54; M1 (+ per-note attack) 2.17; M1 +
  strings 1.39 (strings help: -0.78 [-1.56, -0.16]); M1 + measured floor 2.18, with strings 1.45; attention 1.94;
  attention ahead 1.46; + cross-voice 1.45; ahead + recording everywhere 1.83 (+0.37 [-0.11, +0.84]); ahead + contact
  (T4adv) 1.99 (+0.54 [+0.11, +0.99]); final (T4adv + strings + recording everywhere 32 + learned stiffness) 3.22
  (+1.23 [+0.60, +2.00] over T4adv); final on 100 recordings 3.26. Strings alone help and contact alone hurts a
  little; the large drop appears only in the full combination, and no existing model isolates which piece causes
  it (candidates: ahead + strings, final without contact). EnCodec sees no clear differences (0.06-0.26).
