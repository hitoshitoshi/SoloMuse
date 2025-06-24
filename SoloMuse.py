import mido
import time
import fluidsynth
import numpy as np
import tensorflow as tf
import argparse
import sys
import os

from solomuse.config import (
    NOTE_VOCAB_SIZE, CHORD_VECTOR_SIZE,
    LOWEST_PITCH, REST_TOKEN, STEPS_PER_QUARTER,
    SOUNDFONT
)

from solomuse.models import build_single_step_model, build_unrolled_model

def sample_note(prob_dist, temperature=1.0):
    """Randomly sample a note token from a probability distribution."""
    log_dist = np.log(prob_dist + 1e-9) / temperature
    exp_dist = np.exp(log_dist)
    softmax_dist = exp_dist / np.sum(exp_dist)
    return np.random.choice(range(len(prob_dist)), p=softmax_dist)

def rt_generate_and_play(port_name, model, fs, temperature=1.0):
    """
    Main real-time generation loop.
    Listens for MIDI input, updates a chord vector, and generates/plays notes.
    """
    print(f"\nOpening MIDI port: {port_name}")
    try:
        midi_input_port = mido.open_input(port_name)
    except (IOError, OSError) as e:
        print(f"Error opening MIDI port: {e}")
        sys.exit(1)

    print("Ready to play. Hold down a chord on your MIDI device...")

    # Initialize state
    model.get_layer("lstm").reset_states()
    current_chord_vector = np.zeros(CHORD_VECTOR_SIZE)
    last_note_token = REST_TOKEN
    held_notes = set()

    # Timing
    BEAT_DURATION_S = 0.5  # Corresponds to 120 BPM
    STEP_DURATION_S = BEAT_DURATION_S / STEPS_PER_QUARTER

    try:
        while True:
            # Consume all pending MIDI messages
            for msg in midi_input_port.iter_pending():
                if msg.type == 'note_on' and msg.velocity > 0:
                    held_notes.add(msg.note)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    held_notes.discard(msg.note)

            # Update chord vector
            current_chord_vector.fill(0)
            for note in held_notes:
                if LOWEST_PITCH <= note < LOWEST_PITCH + CHORD_VECTOR_SIZE:
                    idx = note - LOWEST_PITCH
                    current_chord_vector[idx] = 1.0

            # Prepare model inputs
            chord_input = current_chord_vector.reshape((1, 1, CHORD_VECTOR_SIZE)).astype(np.float32)
            note_input = np.array([[last_note_token]], dtype=np.int32)

            # Predict next note
            preds, _, _ = model.predict([note_input, chord_input], verbose=0)
            next_note_token = sample_note(preds[0], temperature=temperature)

            # Play the note
            if next_note_token != REST_TOKEN:
                midi_pitch = next_note_token + LOWEST_PITCH
                fs.noteon(0, midi_pitch, 100)
                time.sleep(STEP_DURATION_S * 0.9) # Play for most of the step duration
                fs.noteoff(0, midi_pitch)
            else:
                time.sleep(STEP_DURATION_S)

            # Update last note for next iteration
            last_note_token = next_note_token

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        midi_input_port.close()
        fs.delete()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time chord-conditioned music generation with SoloMuse."
    )
    parser.add_argument(
        '--soundfont',
        type=str,
        default=SOUNDFONT,
        help=f"Path to the SoundFont file (.sf2). Default: {SOUNDFONT}"
    )
    parser.add_argument(
        '--midi-device',
        type=str,
        help="Specify the name of the MIDI input device to use."
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.1,
        help="Controls the randomness of the generated notes (e.g., 0.8=less random, 1.2=more random)."
    )
    args = parser.parse_args()

    available_ports = mido.get_input_names()

    # Check for available ports
    if not available_ports:
        print("Error: No MIDI input devices found. Please connect a MIDI device.")
        sys.exit(1)

    # Determine which MIDI port to use
    port_name = None
    if args.midi_device:
        if args.midi_device in available_ports:
            port_name = args.midi_device
        else:
            print(f"Error: Specified MIDI device '{args.midi_device}' not found.")
            print("Please choose from the available devices:")
            for port in available_ports:
                print(f"  - {port}")
            sys.exit(1)
    else:
        # If no device is specified, prompt the user
        print("Please select a MIDI input device:")
        for i, port in enumerate(available_ports):
            print(f"  [{i}]: {port}")
        try:
            selection = int(input(f"Enter number (0-{len(available_ports)-1}): "))
            if 0 <= selection < len(available_ports):
                port_name = available_ports[selection]
            else:
                raise ValueError
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            sys.exit(1)

    # Load model weights
    WEIGHTS_PATH = "./saved_models/unrolled_lstm.weights.h5"
    unrolled_model = build_unrolled_model()
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Trained weights not found: {WEIGHTS_PATH}")
    unrolled_model.load_weights(WEIGHTS_PATH)

    rt_model = build_single_step_model()
    for rt_layer in rt_model.layers:
        lname = rt_layer.name
        try:
            src_layer = unrolled_model.get_layer(lname)
            rt_layer.set_weights(src_layer.get_weights())
        except:
            pass

    # Set up FluidSynth
    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload(args.soundfont)
    fs.program_select(0, sfid, 0, 0)

    # Start the main loop
    rt_generate_and_play(port_name, rt_model, fs, temperature=args.temperature)