import os
import requests

# Define base URLs
BASE_URL = "https://www.midiworld.com"
DOWNLOAD_URL = BASE_URL + "/download/"

# Directory to save MIDI files
SAVE_DIR = "midi_files"
os.makedirs(SAVE_DIR, exist_ok=True)

# Fetch MIDI file listing
response = requests.get(BASE_URL)
if response.status_code != 200:
    print("Failed to fetch MIDIWorld homepage.")
    exit()

# Download all MIDI files
for midi_id in range(0,20000):
    midi_url = DOWNLOAD_URL + str(midi_id)
    midi_path = os.path.join(SAVE_DIR, f"{midi_id}.mid")

    try:
        midi_response = requests.get(midi_url, stream=True)
        if midi_response.status_code == 200:
            with open(midi_path, "wb") as f:
                for chunk in midi_response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {midi_path}")
        else:
            print(f"Failed to download {midi_id}")
    except Exception as e:
        print(f"Error downloading {midi_id}: {e}")

print("All downloads completed.")
