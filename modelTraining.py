import tensorflow as tf
from tensorflow.keras import layers, Model

# Suppose your note vocab = 32 distinct pitches (0..31).
NOTE_VOCAB_SIZE = 32

# We'll feed the chord as a 32-dim vector (multi-hot).
CHORD_VECTOR_SIZE = 32

# Model hyperparameters
NOTE_EMBED_DIM = 16   # embedding dimension for previous note
CHORD_HIDDEN_DIM = 16 # dimension to process chord vector
LSTM_UNITS = 64

# 1. Define Inputs
notes_in = tf.keras.Input(shape=(None,), name="notes_in")           # shape: (batch, time)
chords_in = tf.keras.Input(shape=(None, CHORD_VECTOR_SIZE), 
                           name="chords_in")                         # shape: (batch, time, 32)

# 2. Embedding for note tokens
note_embed = layers.Embedding(
    input_dim=NOTE_VOCAB_SIZE,  # size of note vocab
    output_dim=NOTE_EMBED_DIM,  # embedding dimension
    name="note_embedding"
)(notes_in)
# Output shape: (batch, time, NOTE_EMBED_DIM)

# 3. Process chord vectors (optional Dense)
# We can pass the chord vectors through a small Dense or remain as is.
chord_dense = layers.TimeDistributed(
    layers.Dense(CHORD_HIDDEN_DIM, activation='relu'),
    name="chord_dense"
)(chords_in)
# Output shape: (batch, time, CHORD_HIDDEN_DIM)

# 4. Concatenate note embedding + chord embedding
combined = layers.Concatenate(axis=-1)(
    [note_embed, chord_dense]
)
# Shape: (batch, time, NOTE_EMBED_DIM + CHORD_HIDDEN_DIM)

# 5. LSTM
lstm_out = layers.LSTM(
    units=LSTM_UNITS,
    return_sequences=True    # we want an output at every time step
)(combined)
# Output shape: (batch, time, LSTM_UNITS)

# 6. Predict next note
# We apply a Dense layer over each time step
note_logits = layers.TimeDistributed(
    layers.Dense(NOTE_VOCAB_SIZE),
    name="note_logits"
)(lstm_out)
note_probs = layers.Activation("softmax")(note_logits)  # note probabilities

# Build final model
model = Model(inputs=[notes_in, chords_in], outputs=note_probs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # typical for integer targets
    metrics=['accuracy']
)

model.summary()
