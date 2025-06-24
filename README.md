# SoloMuse

SoloMuse is a real-time, chord-conditioned music generation system using a Long Short-Term Memory (LSTM) network built with TensorFlow/Keras. It learns the relationship between chord progressions and melodic lines from a corpus of MIDI files and can generate new monophonic melodies in real-time based on live MIDI chord input.

## Features

  * **Real-time Music Generation**: Plays a continuous stream of music that adapts to the chords you play on a MIDI controller.
  * **Chord-Conditioned Model**: The melody generated is guided by the underlying chord progressions.
  * **Data-Driven**: Trained on the GigaMIDI dataset to learn patterns from a vast collection of music.
  * **Offline Mode**: Includes a script to process an existing MIDI file and overlay a newly generated solo melody.

## How It Works

The project is composed of several key components:

1.  **Data Preparation (`solomuse/data_preparation.py`)**:

      * MIDI files are parsed to separate chord tracks from solo (monophonic) tracks. This is achieved by analyzing the timing and overlap of notes.
      * Both the solo and chord parts are quantized into a sequence of 16th-note time steps.
      * Solo melodies are represented as a sequence of note tokens, including a special token for rests.
      * Chords are represented as a multi-hot vector for each time step, indicating which notes are active.

2.  **Model Architecture (`solomuse/models.py`)**:

      * The core of the project is a chord-conditional LSTM network.
      * The model takes two inputs at each time step: the previous note played (as an embedding) and the current chord vector.
      * It's designed in two forms:
          * **Unrolled Model**: Used for efficient, offline training on sequences of music data.
          * **Single-Step Model**: A stateful version of the LSTM used for real-time generation, where it predicts one note at a time while maintaining its internal state.

3.  **Training (`train.py`)**:

      * The `train.py` script builds a dataset from a folder of MIDI files, creating sequences of a fixed length.
      * It trains the unrolled LSTM model on this dataset and saves the learned weights.

4.  **Real-Time Generation (`SoloMuse.py`)**:

      * This script loads the pre-trained weights into the single-step, stateful model.
      * It listens for MIDI input to continuously update a chord vector representing the currently held notes.
      * At each time step, it feeds the last generated note and the current chord vector to the model to predict the next note.
      * The generated notes are played back using the `fluidsynth` library.

5.  **Offline Generation (`testModel.py`)**:

      * The `testModel.py` script provides a way to use the trained model without a MIDI controller.
      * It reads the chord progression from an existing MIDI file, generates a new solo melody for it from start to finish, and saves the result as a new MIDI file containing both the original chords and the generated melody.

## Usage

### 1\. Prerequisites

First, you need to have Python and the required dependencies installed. You will also need a SoundFont file for audio output.

**Dependencies**

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:

```
tensorflow
numpy
mido
pretty_midi
tqdm
python-fluidsynth
datasets
```

**SoundFont**

The real-time generation script `SoloMuse.py` requires a SoundFont file (`.sf2`) to synthesize the audio. This project is configured to use `acoustic.sf2`. You can find many free SoundFont files online.

### 2\. Data Preparation

The model is trained on the GigaMIDI dataset. You can download the necessary MIDI files using the provided script:

```bash
python scripts/midi_files_scrape.py
```

### 3\. Training the Model

A pre-trained model is already given in `saved_models/unrolled_lstm.weights.h5`.

To train the model on the downloaded data, delete `saved_models/unrolled_lstm.weights.h5`, and run the `train.py` script:

```bash
python train.py
```

This will:

1.  Process the MIDI files in the `data/midi/` folder.
2.  Cache the processed dataset in `data/cache/`.
3.  Train the unrolled LSTM model.
4.  Save the trained weights to `unrolled_lstm.weights.h5`.

If a weights file already exists, the script will skip training and load the existing weights.

### 4\. Running Real-Time Generation

To start the real-time music generation, connect a MIDI keyboard and run `SoloMuse.py`:

```bash
python SoloMuse.py
```

The script will detect your MIDI device and begin generating a melody in response to the chords you play. The generated music will be played back through FluidSynth.

### 5\. Generating a Solo for an Existing MIDI File

To generate a new melody for an existing MIDI file that contains chords:

1.  Place your input MIDI file in the root directory (e.g., `Test.mid`).
2.  Run the `testModel.py` script:

<!-- end list -->

```bash
python testModel.py
```

The script will process `Test.mid`, generate a new solo track, and save the output to `output/Test_with_generated.mid`.

## Project Structure

```
.
├── SoloMuse.py             # Main script for real-time MIDI generation
├── train.py                # Script for training the LSTM model
├── testModel.py            # Script for offline generation over a MIDI file
├── requirements.txt        # Project dependencies
├── .gitignore              # Files and directories to ignore
├── scripts/
│   └── midi_files_scrape.py # Script to download the GigaMIDI dataset
└── solomuse/
    ├── __init__.py
    ├── config.py           # Configuration and hyperparameters
    ├── data_preparation.py # MIDI data processing and quantization
    └── models.py           # Keras model definitions (unrolled and single-step)
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
