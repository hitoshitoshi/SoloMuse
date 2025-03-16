# main.py

import os
import numpy as np
import tensorflow as tf
import fluidsynth
import time
import random

from config import (
    MIDI_FOLDER,
    NUM_FILES,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    NOTE_VOCAB_SIZE,
    REST_TOKEN,
    LOWEST_PITCH
)
from data_preparation import build_training_dataset
from models import build_unrolled_model, build_single_step_model

def sample_note(prob_dist, temperature=1.0):
    """Randomly sample a note token from a probability distribution."""
    log_dist = np.log(prob_dist + 1e-9) / temperature
    exp_dist = np.exp(log_dist)
    softmax_dist = exp_dist / np.sum(exp_dist)
    return np.random.choice(range(len(prob_dist)), p=softmax_dist)

def main():
    ############################################################################
    # 1) Load or Build Dataset
    ############################################################################
    DATASET_CACHE = "cached_dataset.npz"
    
    if not os.path.exists(DATASET_CACHE):
        print("No cached dataset found. Building dataset from MIDI folder...")
        X_notes, X_chords, y_notes = build_training_dataset(
            MIDI_FOLDER,
            num_files=NUM_FILES,
            T=SEQUENCE_LENGTH
        )
        if X_notes is None:
            print("No data found. Exiting.")
            return
        np.savez(DATASET_CACHE, X_notes=X_notes, X_chords=X_chords, y_notes=y_notes)
        print(f"Dataset saved to {DATASET_CACHE}")
    else:
        print(f"Loading dataset from cached file: {DATASET_CACHE}")
        data = np.load(DATASET_CACHE)
        X_notes  = data["X_notes"]
        X_chords = data["X_chords"]
        y_notes  = data["y_notes"]
        print("Dataset loaded successfully.")

    print("Dataset shapes:")
    print("X_notes:", X_notes.shape)   
    print("X_chords:", X_chords.shape) 
    print("y_notes:", y_notes.shape)

    ############################################################################
    # 2) Train or Load the Unrolled Model Weights
    ############################################################################
    WEIGHTS_PATH = "unrolled_lstm.weights.h5"
    
    # Build the unrolled model (same architecture either way)
    unrolled_model = build_unrolled_model()
    unrolled_model.summary()

    if os.path.exists(WEIGHTS_PATH):
        # If weights file already exists, skip training
        print(f"Found existing weights at {WEIGHTS_PATH}, skipping training.")
        unrolled_model.load_weights(WEIGHTS_PATH)
    else:
        # If not, train offline once
        print("No trained weights found. Training unrolled LSTM model...")
        y_notes_expanded = np.expand_dims(y_notes, axis=-1)
        
        unrolled_model.fit(
            [X_notes, X_chords],
            y_notes_expanded,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.1
        )
        # Save weights
        unrolled_model.save_weights(WEIGHTS_PATH)
        print(f"Saved trained weights to {WEIGHTS_PATH}")

    ############################################################################
    # 3) Build Single-Step Model (Stateful) & Copy Weights
    ############################################################################
    rt_model = build_single_step_model()
    rt_model.summary()

    # Load or reload unrolled weights, then copy
    unrolled_model.load_weights(WEIGHTS_PATH)
    for rt_layer in rt_model.layers:
        lname = rt_layer.name
        try:
            source_layer = unrolled_model.get_layer(lname)
            rt_layer.set_weights(source_layer.get_weights())
            print(f"Copied weights for layer '{lname}'")
        except:
            print(f"Skipping layer '{lname}'")

    ############################################################################
    # 4) Demonstrate Generation
    ############################################################################
    print("Generating a short sequence with the real-time model...")

    # Reset states
    rt_model.get_layer("lstm").reset_states()

    generated_notes = []
    current_note = 0  # e.g. start token
    chord_vec_size = rt_model.input_shape[1][2]  # the 3rd dimension in (1, 1, CHORD_VECTOR_SIZE)
    dummy_chords = np.random.rand(16, chord_vec_size).astype(np.float32)

    for t in range(16):
        note_input = np.array([[current_note]], dtype=np.int32)
        chord_input = dummy_chords[t].reshape(1,1,-1)

        preds, h, c = rt_model.predict([note_input, chord_input], verbose=0)
        preds = preds[0]

        next_note = sample_note(preds, temperature=1.0)
        generated_notes.append(next_note)
        current_note = next_note

    print("Generated notes:", generated_notes)



if __name__ == "__main__":
    main()
