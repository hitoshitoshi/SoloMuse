# SoloMuse: Real-Time AI Melody Generator

SoloMuse is a deep learning system that generates monophonic melodies in real-time, conditioned on live chord input from a MIDI controller. It uses a Long Short-Term Memory (LSTM) network built in TensorFlow to learn the relationship between chord progressions and melodic lines from the GigaMIDI dataset.

## Key Features

  * **Real-Time Melody Generation**: Listens to chords played on a MIDI device and generates a harmonically-aware solo voice on the fly.
  * **Offline Solo Composition**: Can take an existing MIDI file, analyze its chord progression, and write a brand new solo track for it.
  * **Adaptive AI Model**: The generated melody is not random; it is guided by the underlying chord progressions you play.
  * **Data-Driven**: Trained on the extensive GigaMIDI dataset to learn patterns from a vast corpus of music.
  * **Customizable**: All hyperparameters, from model architecture to pitch ranges, are centralized in a configuration file for easy modification.

## How It Works

The project is a complete pipeline from data acquisition to real-time inference:

1.  **Data Acquisition (`scripts/midi_files_scrape.py`)**: A script is provided to automatically download the GigaMIDI dataset, which forms the training corpus for the model.

2.  **Data Preparation (`solomuse/data_preparation.py`)**: This is the core of the data pipeline.

      * MIDI files are parsed to intelligently separate chordal parts from monophonic solo lines by analyzing note timing and overlap.
      * The solo and chord parts are quantized into a sequence of 16th-note time steps.
      * Solo melodies are converted into a sequence of integer tokens (representing specific pitches and rests).
      * Chords are transformed into a multi-hot vector for each time step, creating a rich harmonic context.
      * The dataset is augmented by transposing each piece into multiple keys, expanding the training data significantly.

3.  **Model Architecture (`solomuse/models.py`)**: The system uses a dual-model architecture for efficiency:

      * **Unrolled Model**: Used for offline training. It processes entire sequences of notes and chords at once, allowing for fast, parallelized training on a GPU.
      * **Single-Step Model**: A stateful, real-time version of the LSTM. It predicts one note at a time while preserving its internal state (memory), making it perfect for interactive generation.

4.  **Training (`train.py`)**: This script orchestrates the training process. It builds the dataset from the MIDI files (caching the result for future runs), trains the unrolled LSTM model, and saves the learned weights.

5.  **Inference (Real-Time & Offline)**:

      * **`SoloMuse.py`**: The main real-time application. It loads the pre-trained weights into the stateful model, listens for MIDI input to determine the current chord, and uses the model to predict and play the next note in the melody via a software synthesizer.
      * **`testModel.py`**: The offline script. It reads chords from a specified MIDI file, generates a full-length solo from start to finish, and saves the result as a new MIDI file.

## Getting Started

Follow these steps to set up and run SoloMuse.

### Prerequisites

  * Python 3.11
  * A SoundFont file (`.sf2`) for audio synthesis.

### 1\. Clone the Repository

```bash
git clone https://github.com/hitoshitoshi/SoloMuse.git
cd SoloMuse
```

### 2\. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3\. Download MIDI Data

The model is trained on the GigaMIDI dataset. Run the included script to download the necessary files into the `data/midi_files/` directory.

```bash
python scripts/midi_files_scrape.py
```

### 4\. Train the Model

A pre-trained model is provided at `saved_models/unrolled_lstm.weights.h5`. You can use it to run the application immediately.

To train the model from scratch on the data you just downloaded, run:

```bash
python train.py
```

This script will:

1.  Process all MIDI files in the data folder.
2.  Create a cached, pre-processed dataset in `data/cache/`.
3.  Train the unrolled LSTM model.
4.  Save the new weights to `saved_models/unrolled_lstm.weights.h5`.

To force retraining even if a weights file exists, use the `--force-retrain` flag:

```bash
python train.py --force-retrain
```

### 5\. Run the Application

You can now use the trained model in one of two ways:

#### A) Real-Time Generation

Connect a MIDI keyboard or controller to your computer. Run the main script:

```bash
python SoloMuse.py
```

The script will prompt you to select your MIDI device. Once selected, play some chords (3+ notes) and SoloMuse will begin generating a melody in response.

  * **Options**:
      * `--midi-device "Your Device Name"`: Specify the MIDI device directly to skip the prompt.
      * `--temperature 1.2`: Control the randomness of the output. Higher values are more random, lower values are more predictable.

#### B) Offline Generation for an Existing MIDI File

To generate a new solo track for a MIDI file that already contains chords:

```bash
python testModel.py --input-midi /path/to/your/input.mid --output-midi /path/to/your/output.mid
```

The script will analyze the input file, generate a new melody, and save the result as a new MIDI file containing both the original chords and the AI-generated solo.

## Project Structure

```
.
├── SoloMuse.py             # Main script for real-time MIDI generation
├── train.py                # Script for training the LSTM model
├── testModel.py            # Script for offline generation over a MIDI file
├── requirements.txt        # Project dependencies
├── README.md               # This README file
├── scripts/
│   └── midi_files_scrape.py # Script to download the GigaMIDI dataset
└── solomuse/
    ├── __init__.py
    ├── config.py           # Configuration and hyperparameters
    ├── data_preparation.py # MIDI data processing and quantization
    └── models.py           # Keras model definitions (unrolled and single-step)
```

## License

This project is licensed under the MIT License.
