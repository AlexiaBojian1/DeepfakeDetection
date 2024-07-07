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
  - [Contributing](#contributing)
  - [License](#license)

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
