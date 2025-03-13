# config.py

# Global hyperparameters and settings

# MIDI folder containing 1.mid..5000.mid
MIDI_FOLDER = "./midi_files/"
NUM_FILES = 5050

# Timing / Tempo
DEFAULT_BPM = 120
STEPS_PER_QUARTER = 4    # 1 quarter note -> 4 x 16th notes

# Monophonic pitch range
LOWEST_PITCH = 40  # C3
HIGHEST_PITCH = 84 # G5
REST_TOKEN = (HIGHEST_PITCH - LOWEST_PITCH + 1)  # index for "no note"

# Derived
NOTE_VOCAB_SIZE = (HIGHEST_PITCH - LOWEST_PITCH + 1) + 1  # +1 for rest

# Chord vector size (as requested, 45)
CHORD_VECTOR_SIZE = 45

# Sequence slicing length for training
SEQUENCE_LENGTH = 32

# Model architecture
NOTE_EMBED_DIM   = 16
CHORD_HIDDEN_DIM = 16
LSTM_UNITS       = 64

# Training
BATCH_SIZE = 32
EPOCHS = 5
