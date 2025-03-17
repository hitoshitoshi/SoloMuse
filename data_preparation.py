# data_preparation.py

import os
import numpy as np
import pretty_midi
from itertools import combinations
from tqdm import tqdm  # progress bar

from config import (
    MIDI_FOLDER,
    NUM_FILES,
    SEQUENCE_LENGTH,
    REST_TOKEN,
    CHORD_VECTOR_SIZE,
    HIGHEST_PITCH,
    STEPS_PER_QUARTER,
    DEFAULT_BPM,
    LOWEST_PITCH
)

###############################################################################
# CHORD DETECTION (IN MEMORY)
###############################################################################

def notes_overlap(note_a, note_b, min_overlap=0.03):
    """
    Returns True if note_a and note_b overlap in time by at least `min_overlap` seconds.
    """
    overlap = min(note_a.end, note_b.end) - max(note_a.start, note_b.start)
    return overlap >= min_overlap

def separate_guitar_tracks_improved_in_memory(
    midi_path,
    start_time_threshold=0.05,
    overlap_threshold=0.03
):
    """
    Reads `midi_path` with pretty_midi and separates chord vs. solo notes.
    Notes are grouped if they start within `start_time_threshold`.
    If at least one overlapping pair (using overlap_threshold) exists in a group,
    the group is treated as a chord; otherwise, as solo.
    Returns two pretty_midi.Instrument objects: chord_instrument, solo_instrument.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    chord_instrument = pretty_midi.Instrument(program=28)  # e.g., Electric Guitar (Jazz)
    solo_instrument  = pretty_midi.Instrument(program=26)  # e.g., Electric Guitar (Clean)

    for instrument in midi_data.instruments:
        if not (25 <= instrument.program <= 32):
            continue

        sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
        cluster = []
        prev_start = None

        for note in sorted_notes:
            if prev_start is None:
                cluster = [note]
                prev_start = note.start
                continue

            if abs(note.start - prev_start) <= start_time_threshold:
                cluster.append(note)
            else:
                if len(cluster) > 1:
                    overlapping_pairs = sum(1 for n1, n2 in combinations(cluster, 2)
                                             if notes_overlap(n1, n2, min_overlap=overlap_threshold))
                    if overlapping_pairs > 0:
                        chord_instrument.notes.extend(cluster)
                    else:
                        solo_instrument.notes.extend(cluster)
                else:
                    solo_instrument.notes.extend(cluster)
                cluster = [note]
            prev_start = note.start

        if cluster:
            if len(cluster) > 1:
                overlapping_pairs = sum(1 for n1, n2 in combinations(cluster, 2)
                                         if notes_overlap(n1, n2, min_overlap=overlap_threshold))
                if overlapping_pairs > 0:
                    chord_instrument.notes.extend(cluster)
                else:
                    solo_instrument.notes.extend(cluster)
            else:
                solo_instrument.notes.extend(cluster)

    return chord_instrument, solo_instrument

###############################################################################
# QUANTIZATION UTILITIES
###############################################################################

def quantize_instrument_to_monophonic_array(instrument: pretty_midi.Instrument, steps_per_quarter=4, default_bpm=120):
    """
    Converts a (mostly) monophonic instrument (e.g., solo) to a 1D array of note tokens.
    Each time step is determined by the fixed step duration. If no note is active, REST_TOKEN is used.
    """
    if not instrument.notes:
        return np.array([])

    max_end = max(note.end for note in instrument.notes)
    spb = 120.0 / default_bpm  # seconds per beat
    step_duration = spb / steps_per_quarter
    total_steps = int(np.ceil(max_end / step_duration))

    events = []
    for note in instrument.notes:
        events.append((note.start, note.pitch, True))
        events.append((note.end, note.pitch, False))
    events.sort(key=lambda x: x[0])

    array_out = np.full((total_steps,), REST_TOKEN, dtype=np.int32)
    current_note = REST_TOKEN
    e_idx = 0
    n_events = len(events)

    for step in range(total_steps):
        t = step * step_duration
        while e_idx < n_events and events[e_idx][0] <= t:
            _, pitch_val, is_on = events[e_idx]
            if is_on:
                current_note = pitch_val
            else:
                if pitch_val == current_note:
                    current_note = REST_TOKEN
            e_idx += 1

        if current_note != REST_TOKEN and LOWEST_PITCH <= current_note <= HIGHEST_PITCH:
            array_out[step] = current_note - LOWEST_PITCH
        else:
            array_out[step] = REST_TOKEN

    return array_out

def quantize_instrument_to_chord_array(instrument: pretty_midi.Instrument, steps_per_quarter=4, default_bpm=120):
    """
    Converts a chord instrument to a multi-hot array of shape (time_steps, CHORD_VECTOR_SIZE).
    For each time step, active notes (within the defined pitch range) are marked as 1.
    """
    if not instrument.notes:
        return np.zeros((0, CHORD_VECTOR_SIZE), dtype=np.float32)

    max_end = max(note.end for note in instrument.notes)
    spb = 120.0 / default_bpm
    step_duration = spb / steps_per_quarter
    total_steps = int(np.ceil(max_end / step_duration))

    events = []
    for note in instrument.notes:
        events.append((note.start, note.pitch, True))
        events.append((note.end, note.pitch, False))
    events.sort(key=lambda x: x[0])

    array_out = np.zeros((total_steps, CHORD_VECTOR_SIZE), dtype=np.float32)
    active = set()
    e_idx = 0
    n_events = len(events)

    for step in range(total_steps):
        t = step * step_duration
        while e_idx < n_events and events[e_idx][0] <= t:
            _, pitch_val, is_on = events[e_idx]
            if is_on:
                active.add(pitch_val)
            else:
                active.discard(pitch_val)
            e_idx += 1

        for p in active:
            if LOWEST_PITCH <= p <= HIGHEST_PITCH:
                idx = p - LOWEST_PITCH
                if idx < CHORD_VECTOR_SIZE:
                    array_out[step, idx] = 1.0

    return array_out

###############################################################################
# MIDI PARSING WITH KEY TRANSPOSITION
###############################################################################
def parse_midi_file(midi_path, transpose_semitones=0):
    """
    Reads a MIDI file and returns:
      - notes_array: 1D array (total_steps,) of note tokens.
      - chord_array: 2D array (total_steps, CHORD_VECTOR_SIZE) as multi-hot vectors.
    Applies key transposition if specified and quantizes to 1/16-note steps.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    if transpose_semitones != 0:
        for inst in pm.instruments:
            for note in inst.notes:
                note.pitch += transpose_semitones

    chord_inst, solo_inst = separate_guitar_tracks_improved_in_memory(midi_path)
    if transpose_semitones != 0:
        for inst in [chord_inst, solo_inst]:
            for note in inst.notes:
                note.pitch += transpose_semitones

    chord_array = quantize_instrument_to_chord_array(chord_inst, steps_per_quarter=STEPS_PER_QUARTER, default_bpm=DEFAULT_BPM)
    notes_array = quantize_instrument_to_monophonic_array(solo_inst, steps_per_quarter=STEPS_PER_QUARTER, default_bpm=DEFAULT_BPM)

    L = max(len(chord_array), len(notes_array))
    if len(chord_array) < L:
        chord_array = np.pad(chord_array, ((0, L - len(chord_array)), (0, 0)), mode='constant')
    if len(notes_array) < L:
        notes_array = np.pad(notes_array, (0, L - len(notes_array)), mode='constant', constant_values=REST_TOKEN)

    return notes_array, chord_array

###############################################################################
# SLICING INTO TRAINING WINDOWS
###############################################################################
def slice_into_training_windows(notes_array, chord_array, T=32):
    """
    Slices the note and chord arrays into non-overlapping windows of length T.
    Returns:
      - X_notes: shape (#windows, T-1)
      - X_chords: shape (#windows, T-1, CHORD_VECTOR_SIZE)
      - y_notes: shape (#windows, T-1)
    """
    L = len(notes_array)
    if L < T:
        return None, None, None

    X_notes_list = []
    X_chords_list = []
    y_notes_list = []
    for i in range(0, L - T, T):
        X_notes_list.append(notes_array[i:i+T-1])
        X_chords_list.append(chord_array[i:i+T-1, :])
        y_notes_list.append(notes_array[i+1:i+T])
    
    if not X_notes_list:
        return None, None, None

    X_notes_out = np.array(X_notes_list, dtype=np.int32)
    X_chords_out = np.array(X_chords_list, dtype=np.float32)
    y_notes_out = np.array(y_notes_list, dtype=np.int32)

    return X_notes_out, X_chords_out, y_notes_out

###############################################################################
# BUILD DATASET FROM A FOLDER WITH KEY TRANSPOSITION AUGMENTATION
###############################################################################
def build_training_dataset(midi_folder=MIDI_FOLDER, T=SEQUENCE_LENGTH, transpose_range=(-6,7)):
    """
    Iterates over all MIDI files in midi_folder (regardless of file name) and, for each file,
    applies key transposition for each semitone shift in the given range,
    then slices the result into training windows.
    Returns X_notes, X_chords, y_notes.
    """
    all_X_notes = []
    all_X_chords = []
    all_y_notes = []

    midi_files = [os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.lower().endswith('.mid')]

    # Progress bar over files
    for filepath in tqdm(midi_files, desc="Processing MIDI files"):
        if not os.path.exists(filepath):
            print(f"Skipping missing file: {filepath}")
            continue

        for semitones in range(transpose_range[0], transpose_range[1]):
            try:
                notes_array, chord_array = parse_midi_file(filepath, transpose_semitones=semitones)
                if len(notes_array) == 0:
                    continue

                Xn, Xc, Yn = slice_into_training_windows(notes_array, chord_array, T)
                if Xn is None:
                    continue

                all_X_notes.append(Xn)
                all_X_chords.append(Xc)
                all_y_notes.append(Yn)
            except Exception as e:
                print(f"Error parsing {filepath} with transposition {semitones}: {e}")
                continue

    if not all_X_notes:
        print("No valid data found.")
        return None, None, None

    X_notes_full = np.concatenate(all_X_notes, axis=0)
    X_chords_full = np.concatenate(all_X_chords, axis=0)
    y_notes_full = np.concatenate(all_y_notes, axis=0)

    return X_notes_full, X_chords_full, y_notes_full

###############################################################################
# EXAMPLE USAGE
###############################################################################
if __name__ == "__main__":
    X_notes, X_chords, y_notes = build_training_dataset()
    if X_notes is not None:
        print("Dataset shapes:")
        print("X_notes:",  X_notes.shape)
        print("X_chords:", X_chords.shape)
        print("y_notes:",  y_notes.shape)
    else:
        print("No data built.")
