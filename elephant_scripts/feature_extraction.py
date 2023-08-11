"""
This script carries out all of the audio pre-processing steps before extracting the acoustic features (embeddings) using the pre-trained CNN vggish and adding in the missing duration information.
"""
# Import libraries

import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from collections import namedtuple
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

# Set path

P_DIR = Path.cwd()

# Define functions needed to read audio files, apply frequency bandpass filter, extract the annotation based on timestamp, normalise amplitude, zero-pad and centre the annotation with the vggish-compatible 0.96s input window.


def read_audio(path):
    wav, sr = librosa.load(path, sr=None)
    return wav, sr


def apply_bandpass_filter(wav, low_freq, high_freq):
    # Define the bandpass filter
    sr = 4000
    nyquist_freq = 0.5 * sr
    low_normalized = low_freq / sr
    high_normalized = high_freq / sr
    sos = scipy.signal.butter(
        5, [low_normalized, high_normalized], btype="band", output="sos"
    )

    # Apply the bandpass filter to the signal
    filtered_signal = scipy.signal.sosfilt(sos, wav)
    return filtered_signal


def extract_audio(wav, annotation):
    sr = 4000
    start_index = int(annotation.start_time * sr)
    end_index = int(annotation.end_time * sr)
    extracted_audio = wav[start_index:end_index]
    return extracted_audio


def normalise_sound_file(data):
    # Calculate the peak amplitude
    peak_amplitude = np.max(np.abs(data))

    # Set the whole sound file to the peak amplitude
    normalised_data = data * (1 / peak_amplitude)

    return normalised_data


def wav_cookiecutter(
    path, annotation, window_size, position, low_freq, high_freq
):
    # Read the audio
    wav, sr = read_audio(path)

    # Apply the bandpass filter to the signal
    wav = apply_bandpass_filter(wav, low_freq, high_freq)

    Annotation = namedtuple("Annotation", "start_time end_time")
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


# Pass the pre-processed data to the vggish model to extract the automated acoustic features


def feature_extraction(df):
    # Load the vggish model
    model = hub.load("https://tfhub.dev/google/vggish/1")

    # Extract VGGish features
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        path = str(row.audio_filepaths)
        Annotation = namedtuple("Annotation", "start_time end_time")
        annotation = Annotation(row.start_time, row.end_time)
        sr = 4000
        window_size = 4
        low_freq = (
            row.low_freq
        )  # Extract low frequency value from the DataFrame
        high_freq = (
            row.high_freq
        )  # Extract high frequency value from the DataFrame

        # Apply the wav_cookiecutter function combining all of the audio pre-processing steps
        wav = wav_cookiecutter(
            path, annotation, window_size, "middle", low_freq, high_freq
        )

        embeddings = model(wav)
        assert (
            embeddings.shape[1] == 128
        )  # Check the number of features per frame

        # Store info of the embeddings of each frame
        for index, embedding in enumerate(embeddings):
            results.append(
                {
                    "recording_id": row.recording_id,
                    **{
                        f"feature_{n}": feat
                        for n, feat in enumerate(embedding)
                    },
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
    sns.heatmap(
        normalised_results, cmap="magma", xticklabels=False, yticklabels=False
    )
    plt.title("Heatmap of VGGish features")
    plt.xlabel("VGGish features")
    plt.ylabel("Recordings")
    plt.show()


print(
    "Functions for Audio Pre-processing and Feature Extraction successfully loaded"
)
