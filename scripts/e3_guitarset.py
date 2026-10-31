"""Experiment 3: is per-note loudness learned from a mix right on real audio? (GuitarSet, steel-string acoustic)

Audio for these tracks was extracted from the jhartquist/guitarset Hugging Face mirror (Zenodo was ~1 MB/s);
annotations are the original JAMS from Zenodo. GuitarSet records every string separately (hexaphonic pickup, "debleeded"), so each note's true level can be read
from its own string channel. The model only sees a mix and the note list (with strings as voices) and learns each
note's level by fitting the whole excerpt, as in experiment 2. Two mixes:

  hexsum  sum of the six string channels (same signals as the truth, mixed)
  mic     the microphone recording (realistic: body radiation, room, different transducer)

Truth and the naive baseline use the same estimator: energy in the note's partial bins 20-105 ms after onset,
read from the string's own channel (truth) or from the mix (naive).

Usage: uv run python scripts/e3_guitarset.py [--ids a,b,...] [--steps 1500]
"""
import argparse
import json

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from tesis.data.gaps import REPO
from tesis.synth.inharmonicity import fit_partials
from tesis.synth.piece import PieceConfig, PieceModel, note_table

import e2_one_piece as e2  # noqa: E402  (scripts/ is on sys.path when run as a script)

SR = 24000
GS = REPO / "data" / "GuitarSet"
OUT = REPO / "analysis" / "results" / "e3"
CONFIG = dict(source="smooth", body="filter", attack_ms=2.0, pluck_per_note=True)
# one track per player 01-05 (player 00's mirror shard would not download), all five styles, comp and solo
DEFAULT_IDS = ["01_Funk2-108-Eb_solo", "02_Jazz1-130-D_comp", "02_Rock1-90-C#_solo",
               "03_BN2-166-Ab_comp", "04_SS2-88-F_comp", "05_BN1-147-Gb_solo"]


def load_notes(rid):
    jam = json.loads((GS / "annotation" / f"{rid}.jams").read_text())
    rows = []
    for ann in jam["annotations"]:
        if ann["namespace"] != "note_midi":
            continue
        string = int(ann["annotation_metadata"]["data_source"])
        for d in ann["data"]:
            rows.append(dict(onset=d["time"], offset=d["time"] + d["duration"], midi=d["value"], voice=string))
    n = pd.DataFrame(rows).sort_values(["onset", "voice"]).reset_index(drop=True)
    n["pitch"] = np.round(n.midi).astype(int)
    return n


def load_audio(rid):
    hex_, sr = sf.read(GS / "audio_hex-pickup_debleeded" / f"{rid}_hex_cln.wav", dtype="float32", always_2d=True)
    mic, sr2 = sf.read(GS / "audio_mono-mic" / f"{rid}_mic.wav", dtype="float32", always_2d=True)
    hex_ = librosa.resample(hex_.T, orig_sr=sr, target_sr=SR)  # (6, T)
    mic = librosa.resample(mic.mean(1), orig_sr=sr2, target_sr=SR)
    n = min(hex_.shape[1], len(mic))
    return hex_[:, :n], mic[:n]


def bin_levels(y, onsets, freqs, sr=SR):
    """Energy (dB) in each note's partial bins, 20-105 ms after its onset."""
    n_fft = 2048
    win = np.hanning(n_fft)
    out = np.full(len(onsets), np.nan)
    for i, (s, f) in enumerate(zip(onsets, freqs)):
        a = s + int(0.02 * sr)
        seg = y[a:a + n_fft]
        if len(seg) < n_fft:
            continue
        X = np.abs(np.fft.rfft(seg * win)) ** 2
        b = np.round(f[f < sr / 2] / sr * n_fft).astype(int)
        out[i] = 10 * np.log10(X[b].sum() + 1e-12)
    return out


def steel_log_b(ids, all_notes, all_hex):
    """Fit log B = a + b * pitch on long notes, read from their own string channel."""
    ps, bs = [], []
    for rid in ids:
        n, hx = all_notes[rid], all_hex[rid]
        for _, r in n[(n.offset - n.onset) > 0.4].iterrows():
            a = int((r.onset + 0.03) * SR)
            seg = hx[int(r.voice), a:a + int(0.35 * SR)]
            if len(seg) < int(0.3 * SR):
                continue
            fit = fit_partials(seg, SR, 440 * 2 ** ((r.midi - 69) / 12))
            if len(fit.k) >= 8 and 1e-7 < fit.B < 5e-3:
                ps.append(int(r.pitch)), bs.append(fit.B)
    slope, icpt = np.polyfit(ps, np.log(bs), 1)
    return (float(icpt), float(slope)), len(ps), float(np.median(bs))


def score(est, truth):
    m = np.isfinite(est) & np.isfinite(truth)
    d = est[m] - truth[m]
    d = d - np.median(d)  # overall level is arbitrary
    return dict(n=int(m.sum()), corr=float(np.corrcoef(est[m], truth[m])[0, 1]),
                median_abs_err_db=float(np.median(np.abs(d))), p90_abs_err_db=float(np.percentile(np.abs(d), 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS))
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    ids = a.ids.split(",")
    dev = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PieceConfig(sr=SR, **CONFIG)
    L = int(cfg.win * SR)

    notes = {rid: load_notes(rid) for rid in ids}
    audio = {rid: load_audio(rid) for rid in ids}
    log_b, n_b, b_med = steel_log_b(ids, notes, {r: audio[r][0] for r in ids})
    print(f"steel-string inharmonicity: log B = {log_b[0]:.2f} + {log_b[1]:.4f} * pitch "
          f"(from {n_b} notes, median B {b_med:.2e})", flush=True)

    rows, summary = [], {}
    for rid in ids:
        n = notes[rid]
        hx, mic = audio[rid]
        cut = (hx.shape[1] // L) * L
        hx, mic = hx[:, :cut], mic[:cut]
        n = n[n.onset * SR < cut - int(0.2 * SR)].reset_index(drop=True)
        tuning = float(np.median((n.midi - n.pitch) * 100))
        table = note_table(n, cfg, tuning, log_b)
        truth = np.array([bin_levels(hx[v], [s], [f])[0] for v, s, f in zip(n.voice, table["onset"], table["freqs"])])
        midi_end = table["onset"] + table["midi_dur"]
        sounding = np.array([((table["onset"] <= s) & (midi_end > s)).sum() for s in table["onset"]])
        for mix_name, y_np in [("hexsum", hx.sum(0)), ("mic", mic)]:
            y_np = (y_np / (np.abs(y_np).max() + 1e-9) * 0.5).astype(np.float32)
            starts = e2.valid_starts(y_np, L, cfg.hop)
            torch.manual_seed(0)
            model = PieceModel(table, len(y_np), cfg, dev)
            e2.fit(model, torch.tensor(y_np, device=dev), starts, a.steps, tag=f"{rid} {mix_name}", log_every=a.steps)
            with torch.no_grad():
                est = model.note_levels_window_db(0.02, 0.02 + 2048 / SR).cpu().numpy()  # same window as truth
            naive = bin_levels(y_np, table["onset"], table["freqs"])
            res = {"model": score(est, truth), "naive": score(naive, truth)}
            for lab, m in [("1 sounding", sounding == 1), ("2-3 sounding", (sounding >= 2) & (sounding <= 3)),
                           ("4+ sounding", sounding >= 4)]:
                if m.sum() >= 10:
                    res[f"model {lab}"] = score(est[m], truth[m])
                    res[f"naive {lab}"] = score(naive[m], truth[m])
            summary[f"{rid} {mix_name}"] = res
            print(f"{rid} {mix_name}: model corr {res['model']['corr']:.2f} err {res['model']['median_abs_err_db']:.2f} dB | "
                  f"naive corr {res['naive']['corr']:.2f} err {res['naive']['median_abs_err_db']:.2f} dB", flush=True)
            rows += [dict(id=rid, mix=mix_name, onset=o, pitch=p, string=v, sounding=c, truth_db=t, model_db=m_, naive_db=nv)
                     for o, p, v, c, t, m_, nv in zip(n.onset, n.pitch, n.voice, sounding, truth, est, naive)]
    pd.DataFrame(rows).to_csv(OUT / "note_levels.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(dict(log_b=log_b, config=CONFIG, steps=a.steps, results=summary), indent=1))


if __name__ == "__main__":
    main()
