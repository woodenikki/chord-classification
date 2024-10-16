# Chord Classification
My capstone project for the Flatiron School's Data Science Bootcamp Capstone.

## Overview

This project automates the identification of musical chords (major or minor) from audio recordings, simplifying a traditionally manual and time-consuming task. By applying machine learning and audio processing techniques, the tool delivers real-time chord classification, valuable to musicians, producers, and educators.

### Who Can Benefit?

- **Musicians**: Quickly identify chords while practicing or performing.
- **Music Producers**: Streamline the composition process with real-time chord analysis.
- **Educators**: Use this tool to teach music theory interactively.

---

## Tools and Methods

We used a variety of Python libraries to build and refine this model:

- **Audio analysis**: `librosa`, `scipy`
- **Data processing**: `numpy`, `pandas`
- **Visualization**: `matplotlib`, `IPython.display`
- **Machine learning**: `scikit-learn`
- **Dataset handling**: `Kaggle API`

---

## Dataset

The dataset contains audio recordings of major and minor chords played on guitar and piano. Initially, it was imbalanced with 502 major and 357 minor chords, so we applied data augmentation to even out the number of samples (500 per class).

### Data Insights

- **Size**: 859 audio samples
- **Focus**: We extracted harmonic ratios, which capture the core differences between major and minor chords. Harmonic ratios were chosen as they best represent the distinction between the two chord types.
  
While the dataset is effective for this task, it’s limited to guitar and piano, so the model may need additional training for other instruments or playing styles.

---

## Approach

By analyzing the harmonics of the audio signals, we isolated the core features that define each chord. The process involved Fourier transforms to break down the audio signals into their frequency components, helping us capture the relevant harmonic intervals.

For the final model, we focused solely on harmonic content, discarding other features like MFCCs and chroma, as harmonics gave us the most accurate results.

---

## Model

We tested several models, starting with a baseline classifier, and ultimately selected a **Random Forest Classifier**. The model achieved an impressive 98% accuracy.

### Confusion Matrix

|               | Predicted Major | Predicted Minor |
|---------------|:---------------:|:---------------:|
| **Actual Major** |      98         |       2         |
| **Actual Minor** |      2          |      70         |

Although some initial misclassifications were made, closer inspection revealed labeling issues in the dataset, not in the model itself. This underlines the importance of thorough data validation.

---

## Repository

- **[final_notebook.ipynb](final_notebook.ipynb)**: Contains all steps for developing the final model, including data preparation, feature extraction, model training, and evaluation.
- **[workbook.ipynb](workbook.ipynb)**: This notebook covers exploratory data analysis, experimental models, unofficial feature extraction (not used in the final model), and other workflow-related activities.
- **[kaggle.json](kaggle.json)**: Template file for accessing the Kaggle API to download the dataset. You will need to update this file with your own Kaggle credentials.
- **[slides.pdf](slides.pdf)**: Final presentation slides summarizing the project and results.

---

### Steps to Reproduce

To reproduce the project, follow these steps:

1. **Clone the Repository**:
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/woodenikki/chord-classification.git
   cd chord-classification
   ```

1. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Kaggle API Credentials**:

   To access the dataset from Kaggle, you'll need to set up the Kaggle API credentials:

   a. **Obtain your Kaggle API credentials**:
      - Go to your Kaggle account: https://www.kaggle.com/
      - Click on your profile picture (top right) and select `Account`.
      - Scroll down to the `API` section and click `Create New API Token`. This will download a file called `kaggle.json`.

   b. **Save your `kaggle.json` file**:
      - Move the `kaggle.json` file to the appropriate directory on your system:
        - **Linux/Mac**: Place the file in `~/.kaggle/kaggle.json`
        - **Windows**: Place the file in `C:\Users\<your-username>\.kaggle\kaggle.json`

   c. **Ensure your `kaggle.json` file has the correct format**:
      In case you need to manually edit or replace the `kaggle.json` file, ensure it looks like this:
      ```json
      {
        "username": "your_kaggle_username",
        "key": "your_kaggle_api_key"
      }
      ```
      Replace `"your_kaggle_username"` and `"your_kaggle_api_key"` with the actual values from your Kaggle account.
      Additionally, if you are running this on Google Colab - and not cloning the entire repo - there are cells in the final_notebook to hold your access key and username.

5. **Run final_notebook.ipynb**
    This should download the dataset for you (using the Kaggle API you just set up).
