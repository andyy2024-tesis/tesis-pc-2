# Experiment 12 plan: a neural vocoder that refines the DDSP output (for approval)

Idea (user, 2026-11-17): the DDSP model already gets pitch, inharmonicity, timing, dynamics and the parts of the
sound roughly right; a neural network should not relearn that a note is a fundamental plus partials. It should take the
DDSP output and make it realistic. Inference speed no longer matters, only realism.

## What shapes the choice

1. **Little paired data.** The DDSP model trains on 20 recordings: 0.85 h, of which 0.73 h is training audio. A vocoder
   trained from scratch on that would memorise it, so every option starts from a pretrained vocoder. GAPS has 270
   training-split recordings (~13 h, 48 kHz stereo) of real audio, which a vocoder can use to learn "classical guitar"
   from real mel to real audio without any DDSP.
2. **Our pairs are aligned.** The DDSP model reads per-voice loudness and pitch tracks measured from the recording, so its
   output lines up in time with the real audio. Direct pair losses (mel L1) plus a GAN therefore work. GuitarFlow
   (below) did not have this.
3. **One RTX 4090 (24 GB).** Fine-tuning a ~110M-parameter GAN vocoder takes hours. Training a waveform diffusion model
   from scratch takes about a million steps and many hours of data, so it is out.
4. **Guitar is polyphonic.** The source-filter vocoders (NSF family) drive their network with one sine from one f0. Our
   DDSP output is itself a polyphonic, inharmonic source, so it can play that role.

## The options, one by one

| family | model | verdict | why |
|---|---|---|---|
| GAN | MelGAN (2019) | no | superseded; metallic artefacts |
| GAN | Parallel WaveGAN (2020) | no | weaker than HiFi-GAN, same niche |
| GAN | UnivNet (2021) | no, as a whole | its multi-resolution discriminator lives on inside BigVGAN |
| GAN | HiFi-GAN (2020) | reference only | strong but speech-pretrained; BigVGAN is its successor |
| GAN | **BigVGAN-v2** (2024) | **yes, main GAN** | Snake (periodic) activations suit harmonic sound; pretrained 5M steps on speech + instruments + environmental sound; a 24 kHz / 100-band checkpoint matches our sample rate (44 kHz one too); discriminator weights released, so it can really be fine-tuned; MIT licence |
| GAN (codec) | DisCoder (2025) | zero-shot check only | best published music vocoder (mel -> DAC latent -> DAC decoder, 44.1 kHz, trained on Jamendo music), but 430M parameters |
| diffusion | DiffWave, WaveGrad (2020), FastDiff (2022) | no | waveform diffusion from noise: ~1M steps, many hours, no music checkpoints |
| diffusion | PriorGrad (2022) | idea only | starts the noise around a prior taken from the conditioning; our DDSP is a much better prior, which is what shallow diffusion uses |
| diffusion | **shallow diffusion** (DiffSinger 2022; used on DDSP output in DDSP-SVC) | **yes** | the reverse process starts from the DDSP mel noised to depth k instead of from pure noise. k is a dial (small: trust the DDSP; large: freedom) and can be swept at inference without retraining. Needs a mel vocoder after it |
| flow matching | PeriodWave (ICLR 2025), RFWave, WaveFM | no, as vocoders | trained from scratch on speech; at best they match BigVGAN |
| flow matching | **flow from DDSP to real** (GuitarFlow 2025; bridge vocoders such as BridgeVoC 2025) | **yes, as a variant of shallow diffusion** | the flow starts at the DDSP rendition, not at noise, and moves it to the real one. GuitarFlow did our idea on electric guitar (Guitar Pro render -> realistic), 5.75 h of data, 12 min of training. Same network as the shallow diffusion, only the training target changes |
| NSF | NSF, hn-NSF (2019), SiFi-GAN (2023), NSF-HiFiGAN | not as-is | one-f0 sine source (monophonic voice). Their idea (give the network a source that already has the right pitch) is kept in V2 with the DDSP output as the source. SnakeGAN (2023) fed a coarse DDSP signal into a GAN vocoder and beat HiFi-GAN |
| Fourier | iSTFTNet (2022), APNet / APNet2 | no | their advantage is speed |
| Fourier | Vocos (2023) | reserve | natural for a residual in the STFT domain (correct the DDSP magnitudes and phases); small, fast to train; speech-pretrained only, below BigVGAN on music |
| codec latent | DDSP latent -> real latent in DAC / Music2Latent (GuitarFlow's route) | not first | quality capped by the codec; low frame rate smears plucks; GuitarFlow reports "obvious neural artifacts" on single notes (most of classical guitar) and barely more realism than the raw render without amplifier |

## How to connect them

1. **Mel of the DDSP output -> vocoder.** Plug-and-play with pretrained BigVGAN. The vocoder redraws the waveform and the
   DDSP's exact phases and partial positions are lost.
2. **Mel + the DDSP waveform injected at every resolution of the generator** (NSF style). The new branches start at zero,
   so at step 0 it is exactly the pretrained vocoder; training decides how much of the DDSP waveform (pitch,
   inharmonicity, plucks) to keep. Our "best of both worlds".
3. **Stems as separate channels** (a mel per part: harmonics, attack, body, contact, noise). Only possible because the
   model is structured: the vocoder knows what is pluck and what is noise. Extra channels start at zero.
4. **Waveform in, residual waveform out** (output = DDSP + correction). Starts exactly at the DDSP, but nothing pretrained
   exists for it; option 2 gets the same benefit with pretrained weights.
5. **STFT magnitude + phase residual** (Vocos-style). Reserve.
6. **Mel -> shallow diffusion / flow -> mel -> vocoder.** The diffusion fixes the spectrogram, the vocoder makes audio.

## Full model or simple harmonics?

Inputs to compare: **A** the full mix of the final DDSP; **H** only its harmonic stem (the "simple DDSP that sends
harmonics", no new DDSP training); **S** all stems as channels (connection 3). H makes the vocoder invent plucks, body
and room; A gives them but also the wind; S lets it choose.

## Phases

1. **Ceilings, no training (half a day).** Real recording -> mel -> pretrained BigVGAN-v2 24 kHz (and DisCoder 44 kHz):
   how close can a mel vocoder get to the real thing? Final DDSP -> mel -> pretrained BigVGAN: what we get for free.
   Listening page 6: real / real through the vocoder / DDSP / DDSP through the vocoder.
2. **Teach BigVGAN guitar (one night).** Fine-tune BigVGAN-v2 24 kHz on the real audio of the 270 training recordings
   (~13 h), real mel -> real audio. It becomes the vocoder of every mel route.
3. **The contenders (2-3 nights)**, all evaluated on the same held-out windows as every DDSP model:
   - **V1 GAN on mel:** fine-tune the phase-2 vocoder on DDSP mel -> real audio; inputs A and H (S if time).
   - **V2 GAN + DDSP waveform injection** (NSF style): input A (H if time).
   - **V3 shallow diffusion on mel** (+ **V3b flow from DDSP**, same network): a DiffSinger-style denoiser (~20M
     parameters) trained on real mels of all of GAPS; at inference start from the DDSP mel at depths k = 0.2 ... 0.8;
     audio from the phase-2 BigVGAN.
4. **Simple vs full on the winner (one night):** rerun the winner with input H.
5. **Optional, 44.1 kHz output:** BigVGAN-v2 44 kHz with targets from the original 48 kHz files, so the vocoder adds the
   top octave the DDSP never had (nails, string noise).

## Measures and listening

Same multi-scale spectral loss as before (comparable with every DDSP number), mel distance, FAD/KAD with a music
embedding on the held-out windows, and a "notes kept" check (onsets and pitches of output vs input, so the vocoder
cannot change what is played). The ear decides. Per-model parts on the page: DDSP input / vocoder output / what the
vocoder added (output - input).

## Risks

- 0.73 h of pairs: mitigated by pretraining + phase 2 on 13 h of real audio. If still data-bound, retrain the DDSP on 80
  recordings (E9: 5.83 vs 5.77 on the 20, but 4x the pairs).
- The DDSP is better on its training windows than on held-out ones, so the vocoder may trust it too much. Check on
  held-out windows; mitigate with input noise or renders from an earlier checkpoint.
- GAPS is YouTube audio (lossy): "real" includes YouTube artefacts.
- A vocoder can hallucinate notes: the notes-kept check.

## Sources

- BigVGAN-v2: https://github.com/NVIDIA/BigVGAN, https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x
- DisCoder: https://arxiv.org/abs/2502.12759
- DDSP-SVC (DDSP + shallow diffusion / rectified flow + NSF-HiFiGAN): https://github.com/yxlllc/DDSP-SVC
- GuitarFlow: https://arxiv.org/abs/2510.21872
- Kim et al. 2024, acoustic guitar diffusion with outpainting: https://arxiv.org/abs/2401.13498
- SnakeGAN (DDSP prior in a GAN vocoder): https://arxiv.org/abs/2309.07803
- BridgeVoC: https://arxiv.org/abs/2511.07116
- PeriodWave: https://arxiv.org/abs/2408.07547
- Vocos: https://arxiv.org/abs/2306.00814
- Hayes et al., DDSP review (hybrids as a direction): https://arxiv.org/abs/2308.15422
