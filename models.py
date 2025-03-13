# models.py

import tensorflow as tf
from tensorflow.keras import layers, Model

from config import (
    NOTE_VOCAB_SIZE,
    NOTE_EMBED_DIM,
    CHORD_HIDDEN_DIM,
    LSTM_UNITS,
    CHORD_VECTOR_SIZE
)

def build_unrolled_model():
    """
    Unrolled chord-conditional LSTM for offline training on sequences.
    Input: (batch, time), (batch, time, chord_vec)
    Output: (batch, time, note_vocab) with softmax over next note.
    """
    notes_in = tf.keras.Input(shape=(None,), name="notes_in")                # (batch, T-1)
    chords_in = tf.keras.Input(shape=(None, CHORD_VECTOR_SIZE), name="chords_in")  # (batch, T-1, 45)

    # Note embedding
    note_embed = layers.Embedding(
        input_dim=NOTE_VOCAB_SIZE,
        output_dim=NOTE_EMBED_DIM,
        name="note_embedding"
    )(notes_in)  # (batch, time, NOTE_EMBED_DIM)

    # Chord projection
    chord_dense = layers.TimeDistributed(
        layers.Dense(CHORD_HIDDEN_DIM, activation='relu'),
        name="chord_dense"
    )(chords_in)  # (batch, time, CHORD_HIDDEN_DIM)

    # Concatenate
    combined = layers.Concatenate(axis=-1)([note_embed, chord_dense])
    # shape: (batch, time, NOTE_EMBED_DIM + CHORD_HIDDEN_DIM)

    # LSTM unrolled
    lstm_out = layers.LSTM(LSTM_UNITS, return_sequences=True, name="lstm")(combined)
    # shape: (batch, time, LSTM_UNITS)

    note_logits = layers.TimeDistributed(layers.Dense(NOTE_VOCAB_SIZE), name="note_logits")(lstm_out)
    note_probs = layers.Activation("softmax")(note_logits)

    model = Model([notes_in, chords_in], note_probs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_single_step_model():
    """
    Real-time model: processes exactly 1 time step each call, stateful LSTM, batch=1.
    """
    notes_in_rt = tf.keras.Input(
        batch_shape=(1, 1),
        dtype=tf.int32,
        name="notes_in_rt"
    )
    chords_in_rt = tf.keras.Input(
        batch_shape=(1, 1, CHORD_VECTOR_SIZE),
        dtype=tf.float32,
        name="chords_in_rt"
    )

    note_embed_rt = layers.Embedding(
        input_dim=NOTE_VOCAB_SIZE,
        output_dim=NOTE_EMBED_DIM,
        name="note_embedding"
    )(notes_in_rt)  # (1, 1, NOTE_EMBED_DIM)

    chord_dense_rt = layers.TimeDistributed(
        layers.Dense(CHORD_HIDDEN_DIM, activation='relu'),
        name="chord_dense"
    )(chords_in_rt)  # (1, 1, CHORD_HIDDEN_DIM)

    combined_rt = layers.Concatenate(axis=-1)([note_embed_rt, chord_dense_rt])

    lstm_layer_rt = layers.LSTM(
        LSTM_UNITS,
        stateful=True,
        return_sequences=False,
        return_state=True,
        name="lstm"
    )
    lstm_out_rt, state_h_rt, state_c_rt = lstm_layer_rt(combined_rt)

    note_logits_rt = layers.Dense(NOTE_VOCAB_SIZE, name="note_logits")(lstm_out_rt)
    note_probs_rt = layers.Activation("softmax")(note_logits_rt)

    rt_model = Model(
        inputs=[notes_in_rt, chords_in_rt],
        outputs=[note_probs_rt, state_h_rt, state_c_rt]
    )
    rt_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    )
    return rt_model
