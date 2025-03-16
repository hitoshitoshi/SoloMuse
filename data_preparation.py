# data_preparation.py

import os
import numpy as np
import pretty_midi

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
    overlap_start = max(note_a.start, note_b.start)
    overlap_end   = min(note_a.end,   note_b.end)
    return (overlap_end - overlap_start) >= min_overlap

def separate_guitar_tracks_improved_in_memory(
    midi_path,
    start_time_threshold=0.05,
    overlap_threshold=0.03
):
    """
    Reads `midi_path` with pretty_midi, separates chord vs. solo notes in memory.

    Logic:
    - Sort notes by start time.
    - Group them if they start within `start_time_threshold`.
    - If at least two notes in that group truly overlap for `overlap_threshold`, 
      => chord group. Otherwise => solo group.

    Returns two `pretty_midi.Instrument` objects: chord_instrument, solo_instrument
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    chord_instrument = pretty_midi.Instrument(program=28)  # e.g. Electric Guitar (Jazz)
    solo_instrument  = pretty_midi.Instrument(program=26)  # e.g. Electric Guitar (Clean)

    for instrument in midi_data.instruments:
        # (Optional) only process if instrument is in a certain range
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

            # If this note's start is close to the previous note's start => same cluster
            if abs(note.start - prev_start) <= start_time_threshold:
                cluster.append(note)
            else:
                # Evaluate the cluster we just ended
                if len(cluster) > 1:
                    # Check if at least two truly overlap
                    overlapping_pairs = 0
                    for i in range(len(cluster)):
                        for j in range(i + 1, len(cluster)):
                            if notes_overlap(cluster[i], cluster[j], min_overlap=overlap_threshold):
                                overlapping_pairs += 1
                    if overlapping_pairs > 0:
                        chord_instrument.notes.extend(cluster)
                    else:
                        solo_instrument.notes.extend(cluster)
                else:
                    # single note => solo
                    solo_instrument.notes.extend(cluster)

                # Start new cluster
                cluster = [note]

            prev_start = note.start

        # Handle last cluster
        if cluster:
            if len(cluster) > 1:
                overlapping_pairs = 0
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        if notes_overlap(cluster[i], cluster[j], min_overlap=overlap_threshold):
                            overlapping_pairs += 1
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

def quantize_instrument_to_monophonic_array(
    instrument: pretty_midi.Instrument,
    steps_per_quarter=4,
    default_bpm=120
):
    """
    For a (mostly) monophonic instrument (like the solo), returns
    an array of length = # time steps, each = pitch token or REST_TOKEN.

    If multiple notes overlap, last note read might override the previous, 
    but presumably solo is monophonic.
    """
    if not instrument.notes:
        return np.array([])

    # 1) find total duration
    max_end = max(note.end for note in instrument.notes)
    # fallback tempo -> seconds per beat
    spb = 60.0 / default_bpm

    # We won't do advanced tempo changes, just approximate
    total_steps = int(np.ceil(max_end / (spb / steps_per_quarter)))

    # 2) Build events
    events = []
    for note in instrument.notes:
        # note_on
        events.append((note.start, note.pitch, True))
        # note_off
        events.append((note.end,   note.pitch, False))
    events.sort(key=lambda x: x[0])  # sort by time

    step_duration_s = spb / steps_per_quarter
    array_out = np.full((total_steps,), REST_TOKEN, dtype=np.int32)
    current_note = REST_TOKEN
    e_idx = 0
    n_events = len(events)

    for step_idx in range(total_steps):
        step_time_s = step_idx * step_duration_s
        while e_idx < n_events and events[e_idx][0] <= step_time_s:
            _, pitch_val, is_on = events[e_idx]
            if is_on:
                current_note = pitch_val
            else:
                # note_off => if it's the same pitch, revert to rest
                if pitch_val == current_note:
                    current_note = REST_TOKEN
            e_idx += 1

        # map pitch to [0..range] or rest
        if current_note == REST_TOKEN:
            array_out[step_idx] = REST_TOKEN
        else:
            if LOWEST_PITCH <= current_note <= HIGHEST_PITCH:
                array_out[step_idx] = current_note - LOWEST_PITCH
            else:
                array_out[step_idx] = REST_TOKEN
    
    return array_out

def quantize_instrument_to_chord_array(
    instrument: pretty_midi.Instrument,
    steps_per_quarter=4,
    default_bpm=120
):
    """
    For a chord instrument, build a multi-hot array of shape (time_steps, CHORD_VECTOR_SIZE).
    If multiple notes overlap in the same time step, set those indices to 1.
    """
    if not instrument.notes:
        return np.zeros((0, CHORD_VECTOR_SIZE), dtype=np.float32)

    max_end = max(note.end for note in instrument.notes)
    spb = 60.0 / default_bpm
    total_steps = int(np.ceil(max_end / (spb / steps_per_quarter)))

    # Gather note_on/note_off events
    events = []
    for note in instrument.notes:
        events.append((note.start, note.pitch, True))
        events.append((note.end,   note.pitch, False))
    events.sort(key=lambda x: x[0])

    step_duration_s = spb / steps_per_quarter
    array_out = np.zeros((total_steps, CHORD_VECTOR_SIZE), dtype=np.float32)

    # We'll maintain a set of currently active pitches
    active_pitches = set()
    e_idx = 0
    n_events = len(events)

    for step_idx in range(total_steps):
        step_time_s = step_idx * step_duration_s

        # consume all events up to step_time
        while e_idx < n_events and events[e_idx][0] <= step_time_s:
            _, pitch_val, is_on = events[e_idx]
            if is_on:
                active_pitches.add(pitch_val)
            else:
                if pitch_val in active_pitches:
                    active_pitches.remove(pitch_val)
            e_idx += 1

        # mark 1 for each pitch in active_pitches that's in [LOWEST_PITCH..HIGHEST_PITCH]
        for p in active_pitches:
            if LOWEST_PITCH <= p <= HIGHEST_PITCH:
                idx = p - LOWEST_PITCH
                if idx < CHORD_VECTOR_SIZE:
                    array_out[step_idx, idx] = 1.0

    return array_out


###############################################################################
# PARSE A MIDI FILE --> (notes_array, chord_array) in memory
###############################################################################
def parse_midi_file(midi_path):
    """
    1) Calls improved chord detection to separate chord vs. solo notes in memory.
    2) Quantizes chord_instrument -> chord_array (multi-hot).
    3) Quantizes solo_instrument -> monophonic notes_array.
    4) Return them, trimming/padding so they're same length if needed.
    """
    chord_inst, solo_inst = separate_guitar_tracks_improved_in_memory(midi_path)

    # quantize chord => multi-hot
    chord_array = quantize_instrument_to_chord_array(
        chord_inst, steps_per_quarter=STEPS_PER_QUARTER, default_bpm=DEFAULT_BPM
    )
    # quantize solo => monophonic tokens
    notes_array = quantize_instrument_to_monophonic_array(
        solo_inst, steps_per_quarter=STEPS_PER_QUARTER, default_bpm=DEFAULT_BPM
    )

    # unify lengths
    L = max(len(chord_array), len(notes_array))
    # expand or pad
    if len(chord_array) < L:
        pad_shape = (L - len(chord_array), CHORD_VECTOR_SIZE)
        chord_array = np.concatenate([chord_array, np.zeros(pad_shape, dtype=np.float32)], axis=0)
    if len(notes_array) < L:
        pad_array = np.full((L - len(notes_array),), REST_TOKEN, dtype=np.int32)
        notes_array = np.concatenate([notes_array, pad_array], axis=0)

    return notes_array, chord_array


###############################################################################
# SLICING INTO TRAINING WINDOWS
###############################################################################
def slice_into_training_windows(notes_array, chord_array, T=32):
    """
    Slices each piece into windows of length T. 
    Returns X_notes, X_chords, y_notes in shape (#windows, T-1), (#windows, T-1, chord_vec), (#windows, T-1).
    """
    L = len(notes_array)
    if L < T:
        return None, None, None

    sequences_x_notes  = []
    sequences_x_chords = []
    sequences_y_notes  = []

    i = 0
    while i + T < L:
        window_notes  = notes_array[i : i + T]
        window_chords = chord_array[i : i + T]

        # (T-1,) for X, (T-1,) for y
        X_notes_  = window_notes[:-1]
        X_chords_ = window_chords[:-1, :]
        y_notes_  = window_notes[1:]

        sequences_x_notes.append(X_notes_)
        sequences_x_chords.append(X_chords_)
        sequences_y_notes.append(y_notes_)

        i += T  # non-overlapping

    if not sequences_x_notes:
        return None, None, None

    X_notes_out  = np.array(sequences_x_notes,  dtype=np.int32)
    X_chords_out = np.array(sequences_x_chords, dtype=np.float32)
    y_notes_out  = np.array(sequences_y_notes,  dtype=np.int32)

    return X_notes_out, X_chords_out, y_notes_out


###############################################################################
# BUILD DATASET FROM A FOLDER
###############################################################################
def build_training_dataset(midi_folder=MIDI_FOLDER, num_files=NUM_FILES, T=SEQUENCE_LENGTH):
    all_X_notes  = []
    all_X_chords = []
    all_y_notes  = []

    for idx in range(1, num_files+1):
        filepath = os.path.join(midi_folder, f"{idx}.mid")
        if not os.path.exists(filepath):
            print(f"Skipping missing file: {filepath}")
            continue

        try:
            # 1) parse with chord detection in memory
            notes_array, chord_array = parse_midi_file(filepath)
            if len(notes_array) == 0:
                continue

            # 2) slice
            Xn, Xc, Yn = slice_into_training_windows(notes_array, chord_array, T)
            if Xn is None:
                continue

            all_X_notes.append(Xn)
            all_X_chords.append(Xc)
            all_y_notes.append(Yn)

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            continue

    if not all_X_notes:
        print("No valid data found.")
        return None, None, None

    X_notes_full  = np.concatenate(all_X_notes,  axis=0)
    X_chords_full = np.concatenate(all_X_chords, axis=0)
    y_notes_full  = np.concatenate(all_y_notes,  axis=0)

    return X_notes_full, X_chords_full, y_notes_full