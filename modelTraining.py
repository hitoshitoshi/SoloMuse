import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

NOTE_EMBED_DIM = 16
CHORD_HIDDEN_DIM = 16
LSTM_UNITS = 64

notes_in = tf.keras.Input(shape=(None,), name="notes_in")  # (batch, time)
chords_in = tf.keras.Input(shape=(None, CHORD_VECTOR_SIZE), name="chords_in")  # (batch, time, 32)

# Note embedding
note_embed = layers.Embedding(
    input_dim=NOTE_VOCAB_SIZE,
    output_dim=NOTE_EMBED_DIM,
    name="note_embedding"
)(notes_in)  # shape: (batch, time, NOTE_EMBED_DIM)

# Process chord vectors
chord_dense = layers.TimeDistributed(
    layers.Dense(CHORD_HIDDEN_DIM, activation='relu'),
    name="chord_dense"
)(chords_in)  # (batch, time, CHORD_HIDDEN_DIM)

# Concatenate note+chord features
combined = layers.Concatenate(axis=-1)([note_embed, chord_dense])
# shape: (batch, time, NOTE_EMBED_DIM + CHORD_HIDDEN_DIM)

# LSTM (unrolled for 'time' steps)
lstm_out = layers.LSTM(LSTM_UNITS, return_sequences=True)(combined)
# shape: (batch, time, LSTM_UNITS)

# Predict next note
note_logits = layers.TimeDistributed(
    layers.Dense(NOTE_VOCAB_SIZE),
    name="note_logits"
)(lstm_out)
note_probs = layers.Activation("softmax")(note_logits)

# Build training model
train_model = Model([notes_in, chords_in], note_probs)
train_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # integer targets
    metrics=['accuracy']
)

train_model.summary()
