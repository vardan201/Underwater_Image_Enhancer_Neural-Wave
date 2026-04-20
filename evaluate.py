import cv2
import numpy as np

def uciqe(img):
    """
    Underwater Color Image Quality Evaluation (UCIQE)
    A linear combination of chroma, saturation and contrast.
    """
    # Convert BGR to Lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]

    # Calculate chroma
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    mu_c = np.mean(chroma)

    # Calculate contrast (using bottom 1% and top 1% to be robust)
    l_norm = l / 255.0
    contrast = np.percentile(l_norm, 99) - np.percentile(l_norm, 1)

    # Calculate saturation
    # Added small epsilon to avoid division by zero
    saturation = mu_c / np.mean(l + 1e-6)

    # UCIQE formula weights (commonly used in literature)
    return 0.4680 * sigma_c + 0.2745 * contrast + 0.2576 * saturation

def image_entropy(img):
    """Calculate the entropy of an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    logs = np.log2(hist + 1e-10)
    entropy = -np.sum(hist * logs)
    return entropy

def evaluate_image(name, filepath):
    img = cv2.imread(filepath)
    if img is None:
        print(f"Could not read {filepath}")
        return

    print(f"=== {name} ===")
    
    # 1. Color Channels (BGR)
    print(f"Mean Red:   {np.mean(img[:,:,2]):.2f}")
    print(f"Mean Green: {np.mean(img[:,:,1]):.2f}")
    print(f"Mean Blue:  {np.mean(img[:,:,0]):.2f}")
    
    # 2. Saturation (from HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(f"Mean Saturation: {np.mean(hsv[:,:,1]):.2f}/255")
    
    # 3. Contrast (Std Dev of Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Contrast (StdDev): {np.std(gray):.2f}")

    # 4. Metrics from the paper
    print(f"Entropy: {image_entropy(img):.3f}")
    print(f"UCIQE Score: {uciqe(img):.3f}")
    print()

if __name__ == "__main__":
    evaluate_image("INPUT IMAGE", "input.png")
    evaluate_image("OUTPUT IMAGE", "output.png")
