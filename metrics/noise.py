import os
import torch
import piq
import torchvision.transforms as T
from PIL import Image


def evaluate_noise(image_dir, image_ids):
    """
    Evaluate noise using BRISQUE score.
    Low score = low noise, High score = high noise.

    Args:
        image_dir (str): Path to image folder
        image_ids (list): List of image filenames

    Returns:
        list: Noise labels ("low", "medium", "high")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = T.ToTensor()

    scores = []
    for img_id in image_ids:
        img_path = os.path.join(image_dir, img_id)
        if not os.path.isfile(img_path):
            scores.append(None)
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            x = to_tensor(img).unsqueeze(0).to(device)
            scores.append(piq.brisque(x, data_range=1.0).item())
        except:
            scores.append(None)

    valid_scores = [s for s in scores if s is not None]
    min_score, max_score = (min(valid_scores), max(valid_scores)) if valid_scores else (0, 1)
    score_range = max_score - min_score + 1e-8

    def normalize(score):
        return (score - min_score) / score_range if score is not None else 0.0

    norm_scores = [normalize(s) for s in scores]
    return ["low" if ns < 0.4 else "medium" if ns < 0.6 else "high" for ns in norm_scores]
