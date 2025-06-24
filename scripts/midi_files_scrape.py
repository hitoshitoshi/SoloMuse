import os
from datasets import load_dataset

def download_full_gigamidi(subset="v1.0.0", output_dir="data/midi_files"):
    """
    Downloads the full GigaMIDI dataset (all splits combined) for the specified subset.

    Args:
        subset (str): The dataset subset to download (e.g., "all-instruments-with-drums", "drums-only", "no-drums").
        output_dir (str): Directory where the MIDI files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Loading the full dataset (train+validation+test)...")
    dataset = load_dataset("Metacreation/GigaMIDI", subset, trust_remote_code=True,
                           split="train+validation+test")
    
    print("Starting download of MIDI files...")
    for i, sample in enumerate(dataset):
        if sample["instrument_category"] != "all-instruments-with-drums":
            continue
        file_hash = sample["md5"]
        # Check if "music" is a dict containing MIDI bytes or if it's already bytes
        if isinstance(sample["music"], dict):
            midi_bytes = sample["music"]["bytes"]
        else:
            midi_bytes = sample["music"]
        
        file_path = os.path.join(output_dir, f"{file_hash}.mid")
        with open(file_path, "wb") as f:
            f.write(midi_bytes)
        
        if (i + 1) % 1000 == 0:
            print(f"{i + 1} files downloaded...")
    
    print("Download completed.")

if __name__ == "__main__":
    download_full_gigamidi()
