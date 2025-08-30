import os
import cv2
import numpy as np


def evaluate_contrast(image_dir, image_ids):
    """
    Evaluate contrast of images based on grayscale histogram spread.

    Args:
        image_dir (str): Path to image folder
        image_ids (list): List of image filenames

    Returns:
        list: Contrast labels ("low", "medium", "high")
    """
    def histogram_contrast(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        bins = np.arange(256)
        mean_intensity = np.sum(bins * hist_norm)
        return float(np.sqrt(np.sum(((bins - mean_intensity) ** 2) * hist_norm)))

    scores = []
    for img_id in image_ids:
        img_path = os.path.join(image_dir, img_id)
        img = cv2.imread(img_path)
        scores.append(histogram_contrast(img) if img is not None else None)

    valid_scores = [s for s in scores if s is not None]
    max_score = max(valid_scores) if valid_scores else 1.0
    norm_scores = [s / max_score if s is not None else 0.0 for s in scores]

    return ["low" if s < 0.33 else "medium" if s < 0.66 else "high" for s in norm_scores]
