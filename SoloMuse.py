import sounddevice as sd
import numpy as np
import librosa

# 🎤 Audio Settings
SAMPLERATE = 44100  # Standard sample rate for real-time audio
BUFFER_SIZE = 2048  # Buffer size for audio chunks (lower = faster response)
DURATION = 2  # Time window for chord detection

# 🎼 Chord Mapping Dictionary
chord_map = {
    0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"
}

# 🎼 Chord Detection Function
def detect_chord(audio, sr):
    # Extract harmonic content using chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    # Compute average chroma vector (aggregates harmonic presence over time)
    avg_chroma = np.mean(chroma, axis=1)

    # Find the strongest frequency component (most prominent note)
    detected_note = np.argmax(avg_chroma)
    detected_chord = chord_map[detected_note]

    return detected_chord

# 🎤 Audio Input Callback Function
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    # Convert stereo to mono
    audio_mono = np.mean(indata, axis=1)
    # Detect chord
    detected_chord = detect_chord(audio_mono, SAMPLERATE)
    print("B")
    # Print the detected chord in real-time
    print(f"🎼 Detected Chord: {detected_chord}")

# 🎧 Start Live Guitar Capture with Real-Time Chord Detection
with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLERATE, blocksize=BUFFER_SIZE):
    print("🎸 Listening for Chords... (Press Ctrl+C to stop)")
    while True:
        pass
