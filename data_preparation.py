# data_preparation.py

import os
import mido
import numpy as np

from config import (
    MIDI_FOLDER,
    NUM_FILES,
    DEFAULT_BPM,
    STEPS_PER_QUARTER,
    LOWEST_PITCH,
    HIGHEST_PITCH,
    REST_TOKEN,
    CHORD_VECTOR_SIZE,
    SEQUENCE_LENGTH
)

def parse_midi_file(filepath, steps_per_quarter=4, default_bpm=120):
    """
    Reads a MIDI file and returns:
      notes_array: shape (total_steps,) of integer tokens
      chord_array: shape (total_steps, CHORD_VECTOR_SIZE)  (dummy version here)
    Quantizes to 1/16-note steps. Monophonic logic. 
    Tempo changes are simplified (only first set_tempo).
    """
    mid = mido.MidiFile(filepath)
    tempo_us_per_beat = None
    if mid.tracks:
        for msg in mid.tracks[0]:
            if msg.type == 'set_tempo':
                tempo_us_per_beat = msg.tempo
                break
    # fallback if no tempo found
    if tempo_us_per_beat is None:
        tempo_us_per_beat = 60000000 // default_bpm

    # collect note on/off events
    events = []
    for track in mid.tracks:
        current_time_s = 0.0
        for msg in track:
            delta_s = mido.tick2second(
                msg.time,
                mid.ticks_per_beat,
                tempo_us_per_beat
            )
            current_time_s += delta_s

            if msg.type in ['note_on', 'note_off']:
                velocity = msg.velocity if (msg.type == 'note_on') else 0
                events.append((current_time_s, msg.note, velocity))

    events.sort(key=lambda e: e[0])
    if not events:
        return np.array([]), np.zeros((0, CHORD_VECTOR_SIZE), dtype=np.float32)

    total_duration_s = events[-1][0]
    beat_duration_s = tempo_us_per_beat / 1_000_000.0
    step_duration_s = beat_duration_s / steps_per_quarter
    total_steps = int(np.ceil(total_duration_s / step_duration_s))

    notes_array = np.full((total_steps,), REST_TOKEN, dtype=np.int32)
    current_note = REST_TOKEN
    event_idx = 0
    n_events = len(events)

    for step_idx in range(total_steps):
        step_time_s = step_idx * step_duration_s
        while event_idx < n_events and events[event_idx][0] <= step_time_s:
            _, note, vel = events[event_idx]
            if vel > 0:
                current_note = note
            else:
                if note == current_note:
                    current_note = REST_TOKEN
            event_idx += 1

        # map to pitch range or rest
        if current_note == REST_TOKEN:
            notes_array[step_idx] = REST_TOKEN
        else:
            if LOWEST_PITCH <= current_note <= HIGHEST_PITCH:
                notes_array[step_idx] = current_note - LOWEST_PITCH
            else:
                notes_array[step_idx] = REST_TOKEN

    # Dummy chord logic: random 45-dim vector
    chord_array = np.random.rand(total_steps, CHORD_VECTOR_SIZE).astype(np.float32)

    return notes_array, chord_array

def slice_into_training_windows(notes_array, chord_array, T=32):
    """
    Slices a single notes_array (L,) and chord_array (L, CHORD_VECTOR_SIZE)
    into windows of length T. 
    Returns X_notes, X_chords, y_notes in shape (#windows, T-1) or (#windows, T-1, 45).
    """
    L = len(notes_array)
    if L < T:
        return None, None, None

    sequences_x_notes = []
    sequences_x_chords = []
    sequences_y_notes = []

    i = 0
    # Non-overlapping windows
    while i + T < L:
        window_notes = notes_array[i : i + T]
        window_chords = chord_array[i : i + T]

        X_notes_  = window_notes[:-1]           # (T-1,)
        X_chords_ = window_chords[:-1, :]       # (T-1, 45)
        y_notes_  = window_notes[1:]            # (T-1,)

        sequences_x_notes.append(X_notes_)
        sequences_x_chords.append(X_chords_)
        sequences_y_notes.append(y_notes_)

        i += T

    if len(sequences_x_notes) == 0:
        return None, None, None

    X_notes_out  = np.array(sequences_x_notes)   # (#windows, T-1)
    X_chords_out = np.array(sequences_x_chords)  # (#windows, T-1, 45)
    y_notes_out  = np.array(sequences_y_notes)   # (#windows, T-1)

    return X_notes_out, X_chords_out, y_notes_out

def build_training_dataset(midi_folder, num_files=5000, T=32):
    """
    Loops over 1.mid..num_files.mid, parse each, slice, gather into large arrays.
    """
    all_X_notes  = []
    all_X_chords = []
    all_y_notes  = []

    for idx in range(1, num_files+1):
        filepath = os.path.join(midi_folder, f"{idx}.mid")
        if not os.path.exists(filepath):
            print(f"Skipping missing file: {filepath}")
            continue

        try:
            notes_array, chord_array = parse_midi_file(filepath, steps_per_quarter=STEPS_PER_QUARTER, default_bpm=DEFAULT_BPM)
            if len(notes_array) == 0:
                continue
            Xn, Xc, Yn = slice_into_training_windows(notes_array, chord_array, T)
            if Xn is None:
                continue

            all_X_notes.append(Xn)
            all_X_chords.append(Xc)
            all_y_notes.append(Yn)

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            continue

    if len(all_X_notes) == 0:
        print("No valid data found.")
        return None, None, None

    X_notes_full  = np.concatenate(all_X_notes, axis=0)
    X_chords_full = np.concatenate(all_X_chords, axis=0)
    y_notes_full  = np.concatenate(all_y_notes, axis=0)
    return X_notes_full, X_chords_full, y_notes_full
