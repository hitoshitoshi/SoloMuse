import sounddevice as sd
import numpy as np
import librosa
import time
import mido
from mido import Message, MidiFile, MidiTrack
import fluidsynth

# 🎤 Audio Settings
SAMPLERATE = 44100  # Standard sample rate for real-time audio
BUFFER_SIZE = 2048  # Buffer size for audio chunks (lower = faster response)
DURATION = 2  # Time window for chord detection

SOUNDFONT_PATH = "acoustic.sf2"

fs = fluidsynth.Synth()
fs.start()
sfid = fs.sfload(SOUNDFONT_PATH)
fs.program_select(0, sfid, 0, 0)

MIDI_FILE = "generated_midi.mid"

def create_midi(note):
    """Creates a MIDI file with vibrato and pitch bends."""
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    velocity = 64
    note += 50
    duration = 1000
    track.append(Message('note_on', note=note, velocity=velocity, time=0))
    for i in range(0, 8192, 512):
        pitch_value = min(8191, 8191 + i)
        track.append(Message('pitchwheel', pitch=pitch_value, time=20))
    track.append(Message('pitchwheel', pitch=8191, time=100))
    for i in range(10):
        pitch_offset = 600 if i % 2 == 0 else -600
        pitch_value = max(-8192, min(8191, 8191 + pitch_offset))
        track.append(Message('pitchwheel', pitch=pitch_value, time=20))
    track.append(Message('pitchwheel', pitch=0, time=0))
    track.append(Message('note_off', note=note, velocity=velocity, time=duration))
    mid.save(MIDI_FILE)

def play_midi(midi_path):
    mid = mido.MidiFile(midi_path)
    for msg in mid.play():
        if msg.type == 'note_on':
            fs.noteon(0, msg.note, msg.velocity)
        elif msg.type == 'note_off':
            fs.noteoff(0, msg.note)
        elif msg.type == 'pitchwheel':
            fs.pitch_bend(0, msg.pitch)

def detect_chord(audio, sr):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    avg_chroma = np.mean(chroma, axis=1)
    detected_note = np.argmax(avg_chroma)
    return detected_note

def audio_callback(indata, frames, time, status):
    audio_mono = np.mean(indata, axis=1)
    detected_chord = detect_chord(audio_mono, SAMPLERATE)
    create_midi(detected_chord)
    play_midi(MIDI_FILE)
    print(f"🎼 Detected Chord: {detected_chord}")

with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLERATE, blocksize=BUFFER_SIZE):
    print("🎸 Listening for Chords... (Press Ctrl+C to stop)")
    while True:
        time.sleep(0.01)
        pass
