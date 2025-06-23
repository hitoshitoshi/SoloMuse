# add_generated_notes.py

import os
import numpy as np
import pretty_midi
import tensorflow as tf

import solomuse.data_preparation as data_preparation
import solomuse.models as models

# Configuration constants (could also be imported from config.py)
WEIGHTS_PATH       = "unrolled_lstm.weights.h5"
REST_TOKEN         = 45           # rest token index
LOWEST_PITCH       = 40           # token 0 -> MIDI note 40
STEPS_PER_QUARTER  = 4            # e.g., 1/16 note resolution
###############################################################################

def sample_note(prob_dist, temperature=1.0):
    """Randomly sample a note token from a probability distribution."""
    log_dist = np.log(prob_dist + 1e-9) / temperature
    exp_dist = np.exp(log_dist)
    softmax_dist = exp_dist / np.sum(exp_dist)
    return np.random.choice(range(len(prob_dist)), p=softmax_dist)

###############################################################################
# Convert chord array to a chord track by merging consecutive identical chords
###############################################################################
def chord_array_to_midi_instrument(chord_array, tempo_us_per_beat=674157, program=32):
    """
    Converts a chord array of shape (time_steps, CHORD_VECTOR_SIZE) into a pretty_midi.Instrument.
    For each time step, the chord vector is assumed to be multi-hot (values > 0.5 indicate active pitches).
    Consecutive time steps with the same active indices are merged into one sustained chord event.
    """
    chord_instrument = pretty_midi.Instrument(program=program, name="Chord Track")

    beat_duration_s = tempo_us_per_beat / 1_000_000.0
    step_duration_s = beat_duration_s / STEPS_PER_QUARTER

    merged_events = []  # Each element: (active_indices, start_time, end_time)
    current_event = None

    total_steps = chord_array.shape[0]
    for t in range(total_steps):
        # Determine active indices for this time step
        # Here we use a threshold of 0.5; adjust if needed.
        active_indices = tuple(np.where(chord_array[t] > 0.5)[0])
        current_time = t * step_duration_s

        if current_event is None:
            # Start new event (even if active_indices is empty, we'll record it)
            current_event = (active_indices, current_time, current_time + step_duration_s)
        else:
            # If current active_indices match previous, extend the event.
            if active_indices == current_event[0]:
                current_event = (current_event[0], current_event[1], current_time + step_duration_s)
            else:
                # Only save event if it has non-empty active indices.
                if current_event[0]:
                    merged_events.append(current_event)
                # Start a new event with current indices.
                current_event = (active_indices, current_time, current_time + step_duration_s)
    # Append the final event if non-empty.
    if current_event is not None and current_event[0]:
        merged_events.append(current_event)

    # Now, for each merged event, create note objects for each active pitch.
    for event in merged_events:
        active_indices, start_time, end_time = event
        for idx in active_indices:
            midi_pitch = idx + LOWEST_PITCH
            note_obj = pretty_midi.Note(
                velocity=100,
                pitch=midi_pitch,
                start=start_time,
                end=end_time
            )
            chord_instrument.notes.append(note_obj)
    return chord_instrument

###############################################################################
# Convert generated tokens to a lead instrument, merging consecutive ties
###############################################################################
def generated_tokens_to_midi_instrument(generated_tokens, tempo_us_per_beat=402685, program=26):
    """
    Converts a list of generated note tokens into a PrettyMIDI Instrument.
    Consecutive identical tokens (non-rest) are merged into one sustained note.
    """
    gen_instrument = pretty_midi.Instrument(program=program, name="Generated Lead")

    beat_duration_s = tempo_us_per_beat / 1_000_000.0
    step_duration_s = beat_duration_s / STEPS_PER_QUARTER

    time_cursor = 0.0
    i = 0
    n = len(generated_tokens)
    while i < n:
        token = generated_tokens[i]
        if token == REST_TOKEN:
            time_cursor += step_duration_s
            i += 1
            continue

        start_time = time_cursor
        count = 1
        i += 1
        time_cursor += step_duration_s
        while i < n and generated_tokens[i] == token:
            count += 1
            time_cursor += step_duration_s
            i += 1
        end_time = start_time + count * step_duration_s

        midi_pitch = token + LOWEST_PITCH
        note_obj = pretty_midi.Note(
            velocity=100,
            pitch=midi_pitch,
            start=start_time,
            end=end_time
        )
        gen_instrument.notes.append(note_obj)
    return gen_instrument

###############################################################################
# Main function: generate notes for an entire song and add them to a new MIDI file
###############################################################################
def add_generated_notes_to_midi(input_midi_path, output_midi_path, temperature=1.0):
    """
    1. Parse chord data from input_midi_path using data_preparation.parse_midi_file.
       (Assumes parse_midi_file returns (notes_array, chord_array).)
    2. Build single-step model, load weights from the unrolled model, copy weights.
    3. Generate a note token for every time step of the chord array.
    4. Convert the chord array into a merged chord track and the generated tokens into a lead track.
    5. Create a new PrettyMIDI object containing only these two tracks and save it.
    """
    if not os.path.exists(input_midi_path):
        raise FileNotFoundError(f"Input MIDI file not found: {input_midi_path}")

    # Parse the input MIDI file to obtain chord array (and a solo array, though we ignore it here)
    notes_arr, chords_arr = data_preparation.parse_midi_file(input_midi_path)
    total_steps = len(chords_arr)
    if total_steps == 0:
        print(f"No chord data found in {input_midi_path}. Aborting.")
        return

    # Build and load models:
    unrolled_model = models.build_unrolled_model()
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Trained weights not found: {WEIGHTS_PATH}")
    unrolled_model.load_weights(WEIGHTS_PATH)

    rt_model = models.build_single_step_model()
    for rt_layer in rt_model.layers:
        lname = rt_layer.name
        try:
            src_layer = unrolled_model.get_layer(lname)
            rt_layer.set_weights(src_layer.get_weights())
        except:
            pass

    # Generate note tokens for each time step.
    rt_model.get_layer("lstm").reset_states()
    generated_tokens = []
    current_token = REST_TOKEN  # seed token
    for t in range(total_steps):
        chord_vec = chords_arr[t]
        chord_vec_input = chord_vec.reshape((1, 1, chord_vec.shape[-1])).astype(np.float32)
        note_input = np.array([[current_token]], dtype=np.int32)

        preds, _, _ = rt_model.predict([note_input, chord_vec_input], verbose=0)
        preds = preds[0]  # shape: (NOTE_VOCAB_SIZE,)
        next_token = sample_note(preds, temperature=temperature)
        generated_tokens.append(next_token)
        current_token = next_token

    # Use a fixed tempo (default 120 BPM, 500000 us/beat)
    tempo_us_per_beat = 674157

    # Convert chord array into a chord track (merging consecutive identical chords)
    chord_instrument = chord_array_to_midi_instrument(chords_arr, tempo_us_per_beat=tempo_us_per_beat, program=26)
    # Convert generated tokens into a lead track (merging consecutive ties)
    lead_instrument  = generated_tokens_to_midi_instrument(generated_tokens, tempo_us_per_beat=tempo_us_per_beat, program=26)

    # Create a new PrettyMIDI object with only the chord track and generated lead track
    pm_out = pretty_midi.PrettyMIDI()
    pm_out.instruments.append(chord_instrument)
    pm_out.instruments.append(lead_instrument)

    pm_out.write(output_midi_path)
    print(f"Created new MIDI with chords and generated notes: {output_midi_path}")

###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    input_mid  = "TP.mid"            # Input MIDI file containing the original chord information.
    output_mid = "output/TP_with_generated.mid"  # Output file to be created.
    add_generated_notes_to_midi(input_mid, output_mid, temperature=1.0)
