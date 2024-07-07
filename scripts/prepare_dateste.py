import os
import numpy as np
from extract_features import extract_features

def prepare_dataset(real_folder, fake_folder):
    features = []
    labels = []
    
    for filename in os.listdir(real_folder):
        file_path = os.path.join(real_folder, filename)
        if os.path.isfile(file_path):
            feature_vector = extract_features(file_path)
            if len(feature_vector) > 0:  # Check if feature_vector is not empty
                features.append(feature_vector)
                labels.append(0)  # Real label
    
    for filename in os.listdir(fake_folder):
        file_path = os.path.join(fake_folder, filename)
        if os.path.isfile(file_path):
            feature_vector = extract_features(file_path)
            if len(feature_vector) > 0:  # Check if feature_vector is not empty
                features.append(feature_vector)
                labels.append(1)  # Fake label

    # Convert lists to arrays and pad/truncate feature vectors to the same length
    max_length = max(len(f) for f in features)
    padded_features = np.array([np.pad(f, (0, max_length - len(f))) if len(f) < max_length else f[:max_length] for f in features])
    
    return padded_features, np.array(labels)

if __name__ == "__main__":
    real_folder = '../data/real'
    fake_folder = '../data/fake'
    X, y = prepare_dataset(real_folder, fake_folder)
    np.save('features.npy', X)
    np.save('labels.npy', y)
