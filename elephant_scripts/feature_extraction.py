"""
This script carries out all of the audio pre-processing steps before extracting
the acoustic features (embeddings) using the pre-trained CNN vggish and adding
in the missing duration information.
"""
# Import libraries

import warnings
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tensorflow_hub as hub
from tqdm import tqdm

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

# Set path

P_DIR = Path.cwd()

# Here we define the default values for the parameters that will be used in the
# audio pre-processing steps.
DEFAULT_SAMPLERATE = 4000  # Hz

# This is the duration of audio that will be extracted from the original audio
# and passed to the vggish model. The vggish model takes 0.96s
# of audio at 16kHz as input, which is 15360 samples. Since the samplerate of
# the audio files is 4000Hz, we need to change the window duration to 4s
# in order to extract the correct number of samples.
DEFAULT_WINDOW_SIZE = 4  # seconds

# Audio of annotations will be extracted from the original audio file and
# inserted into a 4s window. This parameter defines the position of the
# annotation within the 4s window.
DEFAULT_CLIP_POSITION = "middle"

# Define functions needed to read audio files, apply frequency bandpass filter,
# extract the annotation based on timestamp, normalise amplitude, zero-pad and
# centre the annotation with the vggish-compatible 0.96s input window.


def read_audio(path):
    wav, sr = librosa.load(path, sr=None)
    return wav, sr


def apply_bandpass_filter(
    wav, low_freq, high_freq, samplerate=DEFAULT_SAMPLERATE, order=5
):
    # Define the bandpass filter
    sos = scipy.signal.butter(
        order,
        [low_freq, high_freq],
        fr=samplerate,
        btype="band",
        output="sos",
    )

    # Apply the bandpass filter to the signal
    filtered_signal = scipy.signal.sosfilt(sos, wav)
    return filtered_signal


def extract_audio(wav, annotation, samplerate=DEFAULT_SAMPLERATE):
    start_index = int(annotation.start_time * samplerate)
    end_index = int(annotation.end_time * samplerate)
    extracted_audio = wav[start_index:end_index]
    return extracted_audio


def normalise_sound_file(data):
    # Calculate the peak amplitude
    peak_amplitude = np.max(np.abs(data))

    # Set the whole sound file to the peak amplitude
    normalised_data = data * (1 / peak_amplitude)

    return normalised_data


def wav_cookiecutter(
    annotation,
    window_size=DEFAULT_WINDOW_SIZE,
    position=DEFAULT_CLIP_POSITION,
    samplerate=DEFAULT_SAMPLERATE,
):
    """Extract the acoustic features of a single annotation."""
    # Get path of the audio file from the annotation info
    path = str(annotation.audio_filepaths)

    # Read the audio
    wav, sr = read_audio(path)

    # If the samplerate of the audio file does not match the samplerate
    # then the filtering and annotation extraction will produce
    # incorrect results, so we need to check this.
    assert sr == samplerate

    # Apply the bandpass filter to the signal
    wav = apply_bandpass_filter(
        wav, annotation.low_freq, annotation.high_freq, samplerate=samplerate
    )

    annotation_duration = annotation.end_time - annotation.start_time
    num_windows = np.ceil(annotation_duration / window_size)
    return_duration = num_windows * window_size
    return_size = int(return_duration * sr)

    # Pad the wav array to match return_size
    if len(wav) < return_size:
        wav = np.pad(wav, (0, return_size - len(wav)))
    elif len(wav) > return_size:
        wav = wav[:return_size]

    return_clip = np.zeros(return_size)

    if position == "start":
        return_clip[: len(wav)] = wav
    elif position == "middle":
        annotation_size = len(wav)
        size_difference = return_size - annotation_size
        start = int(size_difference // 2)
        end = int(start + len(wav))

        if start >= 0 and end <= return_size:
            return_clip[start:end] = wav
        else:
            start = max(0, -start)
            end = min(return_size, return_size - size_difference)
            return_clip[start:end] = wav[: end - start]

    # Normalise the sound file
    normalised_clip = normalise_sound_file(return_clip)

    return normalised_clip


# Pass the pre-processed data to the vggish model to extract the automated
# acoustic features


def feature_extraction(
    df, samplerate=DEFAULT_SAMPLERATE, window_size=DEFAULT_WINDOW_SIZE
):
    """Extracts all features from annotation data in dataframe."""
    # Load the vggish model
    model = hub.load("https://tfhub.dev/google/vggish/1")

    # Extract VGGish features
    results = []
    for _, annotation in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Apply the wav_cookiecutter function combining all of the audio
        # pre-processing steps
        wav = wav_cookiecutter(
            annotation,
            window_size=window_size,
            position="middle",
            samplerate=samplerate,
        )

        embeddings = model(wav)
        assert embeddings.shape[1] == 128  # Check the number of features per frame

        # Store info of the embeddings of each frame
        for embedding in embeddings:
            results.append(
                {
                    "recording_id": annotation.recording_id,
                    **{f"feature_{n}": feat for n, feat in enumerate(embedding)},
                }
            )

    print("Features successfully extracted")

    results = pd.DataFrame(results)
    # average vggish annotation feature vectors back into original # of annotations
    results = results.groupby("recording_id").mean()

    print("Features successfully averaged per vocalisation")

    # Add in the missing duration information as the 129th feature
    duration = df[["recording_id", "duration"]].copy()
    results = results.join(duration.set_index("recording_id"))

    # Store the embeddings in the results dataframe
    print("Duration successfully added as the 129th feature")
    return pd.DataFrame(results)  # Return the processed results


# Define function to visualise the extracted features as a heat map


def feature_heatmap(results):
    # Normalise each column
    def normalise_column(col):
        return (col - col.min()) / (col.max() - col.min())

    # Apply the function to each column
    normalised_results = results.apply(normalise_column, axis=0)

    # Visualise the embeddings in a heatmap
    plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
    sns.heatmap(normalised_results, cmap="magma", xticklabels=False, yticklabels=False)
    plt.title("Heatmap of VGGish features")
    plt.xlabel("VGGish features")
    plt.ylabel("Recordings")
    plt.show()


print("Functions for Audio Pre-processing and Feature Extraction successfully loaded")
