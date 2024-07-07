import cv2
import numpy as np
from joblib import load
from skimage.feature import local_binary_pattern,  graycomatrix, graycoprops
import pywt

MAX_FEATURE_LENGTH = 17241336

def extract_features(image_path):
    features = []
    
    # Load image in color mode
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Check if the image was successfully loaded
    if image is None:
        print(f"Error: Could not load image at path {image_path}")
        return features
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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
    
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    features.extend(color_hist.flatten())

    edges = cv2.Canny(gray_image, 100, 200)
    features.extend(edges.flatten())
    
    if len(features) > MAX_FEATURE_LENGTH:
        features = features[:MAX_FEATURE_LENGTH]  # Truncate if longer
    elif len(features) < MAX_FEATURE_LENGTH:
        features = np.pad(features, (0, MAX_FEATURE_LENGTH - len(features)), 'constant')  # Pad if shorter
    

    return features

def test_image(image_path, model_path):
    features = extract_features(image_path)
    
    if not features:
        print("Failed to extract features from the image.")
        return
    
    features = np.array(features).reshape(1, -1)
    
    model = load(model_path)
    
    prediction = model.predict(features)
    
    if prediction == 0:
        print("The image is classified as Real.")
    else:
        print("The image is classified as Fake.")
        print(prediction)

if __name__ == "__main__":
    image_path = "../data/test/image.png"  
    model_path = "deepfake_detector.pkl"  
    test_image(image_path, model_path)
