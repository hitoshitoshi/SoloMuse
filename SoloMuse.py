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

notes_to_frequencies = {
    "C": 261.63,
    "C#": 277.18,
    "D": 293.66,
    "D#": 311.13,
    "E": 329.63,
    "F": 349.23,
    "F#": 369.99,
    "G": 392.00,
    "G#": 415.30,
    "A": 440.00,
    "A#": 466.16,
    "B": 493.88
}


def play_note(frequency, duration=1.0, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    sd.play(wave, sample_rate)
    sd.wait()

# 🎼 Chord Detection Function
def detect_chord(audio, sr):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    avg_chroma = np.mean(chroma, axis=1)
    detected_note = np.argmax(avg_chroma)
    detected_chord = chord_map[detected_note]
    return detected_chord

def audio_callback(indata, frames, time, status):
    audio_mono = np.mean(indata, axis=1)
    detected_chord = detect_chord(audio_mono, SAMPLERATE)
    play_note(notes_to_frequencies[detected_chord])
    print(f"🎼 Detected Chord: {detected_chord}")

with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLERATE, blocksize=BUFFER_SIZE):
    print("🎸 Listening for Chords... (Press Ctrl+C to stop)")
    while True:
        pass
