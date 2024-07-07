# Deepfake Detection

This project aims to detect deepfake images using machine learning techniques. It includes feature extraction from images, model training, and real-time detection using a webcam.

## Table of Contents

- [Deepfake Detection](#deepfake-detection)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Prepare Dataset](#prepare-dataset)
    - [Train Classifier](#train-classifier)
    - [Test on a Single Image](#test-on-a-single-image)
    - [Real-time Detection](#real-time-detection)
  - [Feature Extraction](#feature-extraction)

## Project Structure

```plaintext
.
├── data
│   ├── fake               # Folder containing fake images
│   ├── real               # Folder containing real images
│   └── test               # Folder containing test images
├── scripts
│   ├── extract_features.py   # Feature extraction script
│   ├── prepare_dataset.py    # Script to prepare dataset
│   ├── train_classifier.py   # Script to train classifier
│   ├── test_image.py         # Script to test on a single image
│   └── real_time_detection.py# Script for real-time detection using webcam
├── features.npy           # Numpy array of extracted features
├── labels.npy             # Numpy array of labels corresponding to features
└── README.md              # This README file
```

# Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/deepfake-detection.git
    cd deepfake-detection
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

# Usage

## Prepare Dataset

To prepare the dataset, run the `prepare_dataset.py` script. This script extracts features from real and fake images and saves them as numpy arrays.

    ```sh
    python scripts/prepare_dataset.py
    ```

## Train Classifier

To train the classifier, run the `train_classifier.py` script. This script trains a `RandomForestClassifier` using the extracted features and saves the trained model.

    ```sh
    python scripts/train_classifier.py
    ```

## Test on a Single Image

To test the trained model on a single image, run the `test_image.py` script. This script extracts features from the input image and uses the trained model to classify it as real or fake.

    ```sh
    python scripts/test_image.py
    ```

## Real-time Detection

To perform real-time detection using a webcam, run the `real_time_detection.py` script. This script captures frames from the webcam, extracts features, and uses the trained model to classify each frame.

    ```sh
    python scripts/real_time_detection.py
    ```

## Feature Extraction

The feature extraction process involves several techniques to analyze various aspects of the images:

- **Local Binary Pattern (LBP)**: Captures texture information.
- **Gray-Level Co-occurrence Matrix (GLCM)**: Analyzes texture based on pixel pairs.
- **Fourier Transform**: Analyzes frequency components.
- **Wavelet Transform**: Analyzes different frequency components and spatial resolutions.
- **Color Histogram Analysis**: Analyzes color distribution.
- **Edge Detection**: Detects edges in the image.

