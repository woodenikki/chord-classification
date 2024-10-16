import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks
import librosa
import os
import pickle
import itertools
import pandas as pd

# Load the pre-trained model
MODEL_FILENAME = "chord_classifier_model.pkl"
with open(MODEL_FILENAME, 'rb') as f:
    model = pickle.load(f)

# Directory containing the piano samples
SAMPLES_DIR = "piano_samples"

# Map of note filenames for each key
note_filenames = {
    'C3': 'c3.mp3', 'C#3': 'c-3.mp3', 'D3': 'd3.mp3', 'D#3': 'd-3.mp3', 'E3': 'e3.mp3',
    'F3': 'f3.mp3', 'F#3': 'f-3.mp3', 'G3': 'g3.mp3', 'G#3': 'g-3.mp3', 'A3': 'a3.mp3',
    'A#3': 'a-3.mp3', 'B3': 'b3.mp3', 'C4': 'c4.mp3', 'C#4': 'c-4.mp3', 'D4': 'd4.mp3',
    'D#4': 'd-4.mp3', 'E4': 'e4.mp3', 'F4': 'f4.mp3', 'F#4': 'f-4.mp3', 'G4': 'g4.mp3',
    'G#4': 'g-4.mp3', 'A4': 'a4.mp3', 'A#4': 'a-4.mp3', 'B4': 'b4.mp3', 'C5': 'c5.mp3',
    'C#5': 'c-5.mp3', 'D5': 'd5.mp3', 'D#5': 'd-5.mp3', 'E5': 'e5.mp3', 'F5': 'f5.mp3',
    'F#5': 'f-5.mp3', 'G5': 'g5.mp3', 'G#5': 'g-5.mp3', 'A5': 'a5.mp3', 'A#5': 'a-5.mp3', 'B5': 'b5.mp3'
}

# Function to load piano samples
def load_note_sample(note):
    file_path = os.path.join(SAMPLES_DIR, note_filenames[note])
    try:
        audio, sr = librosa.load(file_path, sr=44100)
        return audio, sr
    except FileNotFoundError:
        st.error(f"Note {note} not found. Please check the piano samples.")
        return None, None

# Generate a chord from the selected notes
def generate_chord(notes, duration=1.0):
    chord = np.zeros(int(44100 * duration))  # Initialize the array for the chord
    for note in notes:
        audio, sr = load_note_sample(note)
        if audio is not None:
            if len(audio) < len(chord):  # Pad with zeros if necessary
                audio = np.pad(audio, (0, len(chord) - len(audio)))
            chord += audio[:len(chord)]
        else:
            st.write(f"Failed to load {note}")
    return chord


# Find harmonics and intervals
def find_harmonics(signal, sr=22050, n_fft=2048):
    if len(signal) == 0:
        st.write("No valid signal found for harmonic extraction")
        return [], []

    S = np.abs(librosa.stft(signal, n_fft=n_fft))
    magnitude = np.mean(S, axis=1)
    frequency = np.fft.fftfreq(len(magnitude), 1/sr)
    positive_freq_idxs = np.where(frequency >= 0)
    positive_freqs = frequency[positive_freq_idxs]
    positive_magnitude = magnitude[positive_freq_idxs]

    peaks, _ = find_peaks(positive_magnitude, height=np.max(positive_magnitude) * 0.1)
    harmonic_frequencies = positive_freqs[peaks]
    harmonic_intervals = np.diff(harmonic_frequencies) if len(harmonic_frequencies) > 1 else []

    return harmonic_frequencies, harmonic_intervals


# Extract harmonic ratios for the model
def extract_harmonic_ratios(harmonics):
    harmonic_ratios = {}
    if len(harmonics) > 1:
        for i, j in itertools.combinations(range(len(harmonics)), 2):
            if harmonics[i] > 1e-6:
                ratio_key = f'ratio_{i+1}_to_{j+1}'
                harmonic_ratios[ratio_key] = harmonics[j] / harmonics[i]
            else:
                harmonic_ratios[ratio_key] = np.nan
    return harmonic_ratios


# Prepare features for classification
def extract_harmonic_features(chord):
    harmonics, _ = find_harmonics(chord)
    if harmonics is None or len(harmonics) == 0:
        st.write("No harmonics found")
        return [0] * 18  # If no harmonics found, return default features
    
    harmonic_ratios = extract_harmonic_ratios(harmonics)
    
    selected_features = [
        'ratio_5_to_12', 'ratio_5_to_10', 'ratio_10_to_14', 'ratio_10_to_12', 
        'ratio_5_to_15', 'ratio_7_to_13', 'ratio_7_to_15', 'ratio_10_to_13', 
        'ratio_1_to_9', 'ratio_5_to_16', 'ratio_10_to_15', 'ratio_1_to_4', 
        'ratio_5_to_9', 'ratio_7_to_9', 'ratio_2_to_3', 'ratio_3_to_12', 
        'ratio_6_to_15', 'ratio_5_to_13'
    ]
    
    # Fill missing ratios with 0
    features = [harmonic_ratios.get(ratio, 0) for ratio in selected_features]
    return features


# Classify the chord using the pre-trained model
def classify_chord(notes):
    chord = generate_chord(notes)
    features = extract_harmonic_features(chord)
    
    # Predict using the model
    result = model.predict([features])
    return 'Major' if result == 1 else 'Minor'

# Streamlit app interface
st.title('Chord Classification App')

# Virtual keyboard
st.write("Press the keys to select notes for the chord:")
selected_notes = st.multiselect("Select notes", options=list(note_filenames.keys()))

if selected_notes:
    st.write(f"Selected notes: {', '.join(selected_notes)}")

    if st.button('Play Chord'):
        chord = generate_chord(selected_notes)
        sd.play(chord, samplerate=44100)
        sd.wait()  # Wait until the chord finishes playing

    if st.button('Classify Chord'):
        chord_type = classify_chord(selected_notes)
        st.write(f"The chord is classified as: {chord_type}")