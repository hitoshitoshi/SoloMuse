import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# 1) HYPERPARAMETERS

# Suppose we have 1/16 note resolution for each time step
# We'll define:
T = 32                   # sequence length (e.g., 2 measures of 16th notes in 4/4)
NOTE_VOCAB_SIZE = 46     # 32 pitch tokens + 1 'rest' or special token
CHORD_VECTOR_SIZE = 45   # 32-dim multi-hot chord vector

# 2) BUILD & TRAIN THE UNROLLED MODEL

NOTE_EMBED_DIM = 16
CHORD_HIDDEN_DIM = 16
LSTM_UNITS = 64

# Inputs for unrolled training
notes_in = tf.keras.Input(shape=(None,), name="notes_in")  # (batch, time)
chords_in = tf.keras.Input(shape=(None, CHORD_VECTOR_SIZE), name="chords_in")  # (batch, time, 32)

# Note embedding
note_embed = layers.Embedding(
    input_dim=NOTE_VOCAB_SIZE,
    output_dim=NOTE_EMBED_DIM,
    name="note_embedding"
)(notes_in)  # shape: (batch, time, NOTE_EMBED_DIM)

# Chord projection
chord_dense = layers.TimeDistributed(
    layers.Dense(CHORD_HIDDEN_DIM, activation='relu'),
    name="chord_dense"
)(chords_in)  # shape: (batch, time, CHORD_HIDDEN_DIM)

# Concatenate (batch, time, NOTE_EMBED_DIM + CHORD_HIDDEN_DIM)
combined = layers.Concatenate(axis=-1)([note_embed, chord_dense])

# Unrolled LSTM over 'time'
lstm_out = layers.LSTM(LSTM_UNITS, return_sequences=True, name="lstm")(combined)

# TimeDistributed Dense -> predict next note
note_logits = layers.TimeDistributed(layers.Dense(NOTE_VOCAB_SIZE), name="note_logits")(lstm_out)
note_probs = layers.Activation("softmax")(note_logits)

# Build training model
train_model = Model([notes_in, chords_in], note_probs)
train_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

train_model.summary()

# 4) COPY WEIGHTS FROM UNROLLED -> RT

train_model.load_weights("unrolled_lstm.weights.h5")

# We'll copy layer by layer using matching names
for rt_layer in rt_model.layers:
    layer_name = rt_layer.name
    try:
        # Get corresponding layer in train_model
        source_layer = train_model.get_layer(layer_name)
        rt_layer.set_weights(source_layer.get_weights())
        print(f"Copied weights for layer '{layer_name}'")
    except:
        print(f"Skipping layer '{layer_name}'")


# # Create random "previous notes" sequences: shape (N, T-1)
# X_notes = np.random.randint(0, NOTE_VOCAB_SIZE, size=(NUM_SAMPLES, T-1))
# # Create random chord vectors: shape (N, T-1, 32)
# X_chords = np.random.randint(0, 2, size=(NUM_SAMPLES, T-1, CHORD_VECTOR_SIZE))
# # Create random "next notes" targets: shape (N, T-1)
# y_notes = np.random.randint(0, NOTE_VOCAB_SIZE, size=(NUM_SAMPLES, T-1))

# # Expand y_notes for time-distributed (N, T-1, 1)
# y_notes_expanded = np.expand_dims(y_notes, axis=-1)

# print("X_notes shape:   ", X_notes.shape)
# print("X_chords shape:  ", X_chords.shape)
# print("y_notes shape:   ", y_notes_expanded.shape)

# # Train offline (with dummy data here)
# history = train_model.fit(
#     [X_notes, X_chords],
#     y_notes_expanded,
#     batch_size=16,
#     epochs=5,          # Increase for real data
#     validation_split=0.2
# )

# Save weights
train_model.save_weights("unrolled_lstm.weights.h5")