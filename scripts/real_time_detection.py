import cv2
import numpy as np
from joblib import load
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import pywt

def extract_features_from_frame(frame):
    features = []
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    features.extend(lbp_hist)
    
    # Gray-Level Co-occurrence Matrix (GLCM)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    features.extend([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ])
    
    # Fourier Transform
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    features.extend(np.mean(magnitude_spectrum, axis=1))
    
    # Wavelet Transform
    coeffs2 = pywt.dwt2(gray_image, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    features.extend([
        np.mean(LL), np.mean(LH),
        np.mean(HL), np.mean(HH)
    ])
    
    # Color Histogram Analysis
    color_hist = cv2.calcHist([frame], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    features.extend(color_hist.flatten())
    
    # Edge Detection
    edges = cv2.Canny(gray_image, 100, 200)
    features.extend(edges.flatten())
    
    return features

# Load the trained model
clf = load('deepfake_detector.pkl')

def process_frame(frame, classifier):
    # Convert frame to features
    features = extract_features_from_frame(frame)
    features = np.array(features).reshape(1, -1)
    
    # Predict using the classifier
    prediction = classifier.predict(features)
    
    return prediction

# Example of processing a video stream
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    prediction = process_frame(frame, clf)
    
    label = 'Real' if prediction == 0 else 'Fake'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Deepfake Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
