# capstone

## Business Understanding

This project addresses the challenge of accurately identifying musical chords as either major or minor using audio input. Manual chord transcription can be time-consuming and requires a solid grasp of music theory. By leveraging machine learning and audio processing techniques, this tool automates chord classification, providing real-time results that are valuable to musicians, music producers, and educators.

### Stakeholders:

- **Musicians** can use the tool to quickly identify chords during practice or live performances, enhancing learning and improvisation.
- **Music Producers** benefit from real-time chord analysis, streamlining the composition and arrangement process.
- **Educators** gain an interactive way to teach chord progressions and musical theory, helping students grasp harmonic relationships more effectively.

### Conclusion:

The model achieved 98% accuracy by utilizing data augmentation to balance the dataset, originally skewed towards major chords (502 major, 357 minor). Augmenting 143 minor chord samples created a more even distribution, improving classification performance. This tool has the potential to save time and improve the workflow for musicians, producers, and educators by automating chord recognition, making it more accessible and efficient.
---

## Tools/Methodologies

To handle the workflow, I'll use several Python libraries:

- **Audio feature extraction**: `librosa`, `scipy`
- **Data manipulation**: `numpy`, `pandas`
- **Visualization**: `matplotlib`, `IPython.display`
- **Machine Learning**: `scikit-learn`
- **Dataset management**: `Kaggle API`
- **Warning suppression**: `warnings` module

---

## Data Understanding

The dataset used for this project is the [Musical Instrument Chord Classification (Audio)](https://www.kaggle.com/datasets/deepcontractor/musical-instrument-chord-classification) dataset from Kaggle. It consists of .wav audio files, containing major and minor chords played on guitar and piano. Since the dataset is readily available, it eliminates the need for manual data collection. The clear distinction between major and minor chords makes it highly suitable for this classification task.

### Dataset Size and Features:

#### Size: 
> The dataset contains 859 audio samples—502 major chords and 357 minor chords. To balance the dataset, we applied augmentation, adding 143 minor chord samples, resulting in 500 samples for each class.

#### Features: 
> While earlier iterations explored features such as Mel-frequency cepstral coefficients (MFCCs), Chroma, Spectral centroid, zero-crossing rate, and Mel-spectrograms, the final model focuses exclusively on harmonic content. Specifically, harmonic ratios were extracted and analyzed to capture the major/minor third distinction. These selected harmonic ratios were identified both visually and through statistical analysis, with a p-value < 0.05, indicating their statistical significance in distinguishing between major and minor chords.

The experimental exploration of other features and iterations can be found in the accompanying `workbook.ipynb`, but the final model is built solely on harmonic extraction and analysis.

### Harmonic Ratio Features:

Harmonic ratios directly reflect the intervals between notes in a chord, which is the key distinction between major and minor chords. The visually and statistically significant harmonic ratios make this approach highly relevant for the classification task, focusing precisely on the differences between the two chord types.

#### Data Limitations:

One limitation of the dataset is its restriction to two instruments: guitar and piano. This limits the model's applicability to other instruments, as the harmonic content of chords can vary with timbre. Additionally, the dataset lacks variation in dynamics and playing styles, which might affect chord recognition in diverse musical contexts. Therefore, while the model performs well on this dataset, its generalization to other instruments or more complex real-world music may be limited.

---

## Data Preparation

The distinction between major and minor chords lies in the intervals between their constituent notes, particularly the major and minor third intervals. By analyzing the harmonics of audio signals, we can accurately capture these relationships, as the harmonics are directly tied to the frequencies that define the chord's structure.

The extraction functions were designed to isolate and analyze these harmonics using Fourier transforms, which break down the audio signal into its individual frequency components. This allows us to identify not just the fundamental frequencies, but also their harmonic overtones, giving a clear picture of the harmonic intervals that characterize each chord. The harmonic intervals and ratios extracted through these functions provide precise and relevant data for distinguishing between major and minor chords, making this approach both targeted and effective for the problem at hand.

By using only the harmonic content for the final model, we focused on the core features essential to chord classification, ensuring both the relevance and efficiency of the feature extraction process.

---

## Modeling

This is a classification problem, with the goal of predicting whether a chord is major or minor. I went through multiple iterations while developing this model. First, I'll define a baseline using a dummy classifier, followed by a summary of key highlights from the other iterations, before presenting the final model: a Random Forest Classifier.

### Evaluation

Our final confusion matrix shows that the model made only 4 misclassifications (2 for each class), confirming that it has a high degree of accuracy in distinguishing between major and minor chords.

### Confusion Matrix

|               | Predicted Major | Predicted Minor |
|---------------|:---------------:|:---------------:|
| **Actual Major** |      98         |       2         |
| **Actual Minor** |      2          |      70         |

Several misclassifications were initially identified in the model's predictions, but upon closer analysis, it was found that some of these chords were mislabeled in the dataset. The model correctly identified complex harmonic structures, such as augmented and extended chords, which were likely misinterpreted due to the data labeling. This highlights the need for careful data validation when dealing with nuanced musical elements.

---

## Deployment

The results will be delivered through a simple Streamlit web app, where users can upload or record audio files to classify as major or minor chords. The app will provide real-time feedback, displaying the classification result along with confidence scores and visualizations such as waveforms or spectrograms. The app will be hosted on Streamlit Cloud, making it easily accessible and user-friendly for quick chord analysis.
