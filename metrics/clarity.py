import os
import cv2
import numpy as np


def evaluate_clarity(image_dir, image_ids):
    """
    Evaluate clarity (sharpness) of images based on edge strength.

    Args:
        image_dir (str): Path to image folder
        image_ids (list): List of image filenames

    Returns:
        list: Clarity labels ("low", "medium", "high")
    """
    def edge_profile_clarity(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        edges = cv2.Canny(img, 100, 200)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_gradients = gradient_magnitude[edges > 0]
        return float(np.mean(edge_gradients)) if len(edge_gradients) > 0 else 0.0

    raw_scores = [edge_profile_clarity(os.path.join(image_dir, img_id)) for img_id in image_ids]
    valid_scores = [s for s in raw_scores if s is not None]
    max_score = max(valid_scores) if valid_scores else 1.0
    norm_scores = [s / max_score if s is not None else 0.0 for s in raw_scores]

    return ["low" if ns < 0.33 else "medium" if ns < 0.66 else "high" for ns in norm_scores]
