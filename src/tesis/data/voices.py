"""Assign notes to monophonic voices.

A guitar string plays one note at a time, so the synthesizer renders a fixed number of monophonic voices.
Until notes are matched to tablature strings, voices are allocated greedily:

1. a re-plucked pitch goes back to the voice already playing it (most likely the same string);
2. otherwise the voice that has been free the longest (lets earlier notes ring as long as possible);
3. if every voice is still inside its MIDI note, the one whose note ends first is cut ("steal").
"""
import numpy as np

N_VOICES = 6


def allocate_voices(onset: np.ndarray, offset: np.ndarray, pitch: np.ndarray, n_voices: int = N_VOICES):
    """Return (voice per note, number of steals). Notes must be sorted by onset."""
    voice = np.full(len(onset), -1, dtype=np.int16)
    v_pitch = np.full(n_voices, -1)
    v_end = np.full(n_voices, -np.inf)  # MIDI offset of each voice's current note
    steals = 0
    for i in range(len(onset)):
        t = onset[i]
        same = np.where(v_pitch == pitch[i])[0]
        free = np.where(v_end <= t)[0]
        taken_now = voice[:i][onset[:i] == t] if i else []
        if len(same) and same[0] not in taken_now:
            v = same[0]
        elif len(free):
            v = free[np.argmin(v_end[free])]
        else:
            v = int(np.argmin(v_end))
            steals += 1
        voice[i] = v
        v_pitch[v] = pitch[i]
        v_end[v] = offset[i]
    return voice, steals
