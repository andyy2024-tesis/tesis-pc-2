"""Loading GAPS recordings, notes and metadata."""
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
GAPS_ROOT = REPO / "data" / "GAPS"
PROCESSED = REPO / "data" / "processed"


@dataclass
class Notes:
    """Note list sorted by onset. Times in seconds, pitch as MIDI number."""

    onset: np.ndarray
    offset: np.ndarray
    pitch: np.ndarray

    def __len__(self):
        return len(self.onset)


def load_metadata(root: Path = GAPS_ROOT) -> pd.DataFrame:
    return pd.read_csv(root / "gaps_metadata_with_splits.csv")


def load_audio(rid: str, sr: int, root: Path = GAPS_ROOT) -> np.ndarray:
    """Mono float32 audio resampled to `sr`."""
    x, file_sr = sf.read(root / "audio" / f"{rid}.wav", dtype="float32", always_2d=True)
    y = x.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")
    return y.astype(np.float32)


def load_notes(rid: str, root: Path = GAPS_ROOT) -> Notes:
    m = pretty_midi.PrettyMIDI(str(root / "midi" / f"{rid}.mid"))
    notes = sorted((n for inst in m.instruments for n in inst.notes), key=lambda n: (n.start, n.pitch))
    return Notes(
        onset=np.array([n.start for n in notes]),
        offset=np.array([n.end for n in notes]),
        pitch=np.array([n.pitch for n in notes], dtype=np.int16),
    )
