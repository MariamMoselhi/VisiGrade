import os
import numpy as np
from PIL import Image


def evaluate_brightness(image_dir, image_ids):
    """
    Evaluate brightness of images using perceptual luminance.

    Args:
        image_dir (str): Path to image folder
        image_ids (list): List of image filenames

    Returns:
        list: Brightness labels ("low", "medium", "high")
    """
    def brightness_luminance(img):
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        return float(np.mean(0.299 * r + 0.587 * g + 0.114 * b))

    scores = []
    for img_id in image_ids:
        try:
            img = Image.open(os.path.join(image_dir, img_id))
            scores.append(brightness_luminance(img))
        except:
            scores.append(None)

    valid_scores = [s for s in scores if s is not None]
    max_score = max(valid_scores) if valid_scores else 1.0
    norm_scores = [s / max_score if s is not None else 0.0 for s in scores]

    return ["low" if s < 0.3 else "medium" if s < 0.7 else "high" for s in norm_scores]
