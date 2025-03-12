import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

#####################################
# 1) HYPERPARAMETERS & DUMMY DATA
#####################################

# Suppose we have 1/16 note resolution for each time step
# We'll define:
T = 32                   # sequence length (e.g., 2 measures of 16th notes in 4/4)
NOTE_VOCAB_SIZE = 33     # 32 pitch tokens + 1 'rest' or special token
CHORD_VECTOR_SIZE = 32   # 32-dim multi-hot chord vector
NUM_SAMPLES = 200        # how many training sequences (dummy)

# Create random "previous notes" sequences: shape (N, T-1)
X_notes = np.random.randint(0, NOTE_VOCAB_SIZE, size=(NUM_SAMPLES, T-1))
# Create random chord vectors: shape (N, T-1, 32)
X_chords = np.random.randint(0, 2, size=(NUM_SAMPLES, T-1, CHORD_VECTOR_SIZE))
# Create random "next notes" targets: shape (N, T-1)
y_notes = np.random.randint(0, NOTE_VOCAB_SIZE, size=(NUM_SAMPLES, T-1))

# Expand y_notes for time-distributed (N, T-1, 1)
y_notes_expanded = np.expand_dims(y_notes, axis=-1)

print("X_notes shape:   ", X_notes.shape)
print("X_chords shape:  ", X_chords.shape)
print("y_notes shape:   ", y_notes_expanded.shape)

#####################################
# 2) BUILD & TRAIN THE UNROLLED MODEL
#####################################

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

# Train offline (with dummy data here)
history = train_model.fit(
    [X_notes, X_chords],
    y_notes_expanded,
    batch_size=16,
    epochs=5,          # Increase for real data
    validation_split=0.2
)

# Save weights
train_model.save_weights("unrolled_lstm.weights.h5")

#####################################
# 3) BUILD THE REAL-TIME, SINGLE-STEP MODEL
#####################################

# This model processes exactly 1 time step each call.
notes_in_rt = tf.keras.Input(
    batch_shape=(1, 1),  # batch=1, time=1
    dtype=tf.int32,
    name="notes_in_rt"
)
chords_in_rt = tf.keras.Input(
    batch_shape=(1, 1, CHORD_VECTOR_SIZE),  # batch=1, time=1, chord vec=32
    dtype=tf.float32,
    name="chords_in_rt"
)

# Reuse same layer names/dims so we can copy weights
note_embed_rt = layers.Embedding(
    input_dim=NOTE_VOCAB_SIZE,
    output_dim=NOTE_EMBED_DIM,
    name="note_embedding"
)(notes_in_rt)  # shape: (1, 1, NOTE_EMBED_DIM)

chord_dense_rt = layers.TimeDistributed(
    layers.Dense(CHORD_HIDDEN_DIM, activation='relu'),
    name="chord_dense"
)(chords_in_rt)  # shape: (1, 1, CHORD_HIDDEN_DIM)

combined_rt = layers.Concatenate(axis=-1)([note_embed_rt, chord_dense_rt])
# shape: (1, 1, NOTE_EMBED_DIM + CHORD_HIDDEN_DIM)

# Stateful LSTM: one step at a time
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
    # In practice, we won't be training this single-step model
)

rt_model.summary()

#####################################
# 4) COPY WEIGHTS FROM UNROLLED -> RT
#####################################

# Reload the unrolled model weights
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

#####################################
# 5) RUN REAL-TIME GENERATION (DEMO)
#####################################

def sample_note(prob_dist, temperature=1.0):
    """Sample an integer note token from a probability distribution with temperature."""
    log_dist = np.log(prob_dist + 1e-9) / temperature
    exp_dist = np.exp(log_dist)
    softmax_dist = exp_dist / np.sum(exp_dist)
    return np.random.choice(len(prob_dist), p=softmax_dist)

# Reset the LSTM states at the start of generation
rt_model.get_layer("lstm").reset_states()

# We'll do a short generation of 16 steps
generated_notes = []
current_note = 0  # e.g., "rest" or "start" token

# Make a dummy chord progression of length 16
chord_prog = np.random.randint(0, 2, size=(16, CHORD_VECTOR_SIZE)).astype(np.float32)

for t in range(16):
    # Prepare model inputs: shape (1,1) for note, (1,1,32) for chord
    note_input = np.array([[current_note]], dtype=np.int32)
    chord_vec = chord_prog[t].reshape(1,1,CHORD_VECTOR_SIZE)  # shape (1,1,32)

    # Predict
    preds, h, c = rt_model.predict([note_input, chord_vec], verbose=0)
    preds = preds[0]  # shape (NOTE_VOCAB_SIZE,)

    # Sample next note
    next_note = sample_note(preds, temperature=1.0)

    generated_notes.append(next_note)
    current_note = next_note

print("Generated notes (tokens):", generated_notes)
