import pandas as pd
from metrics.clarity import evaluate_clarity
from metrics.contrast import evaluate_contrast
from metrics.brightness import evaluate_brightness
from metrics.noise import evaluate_noise


def run_image_quality_pipeline(csv_path, image_dir, output_csv="Image_Quality_Classification.csv"):
    """
    Run the full image quality assessment pipeline:
    - Clarity
    - Contrast
    - Brightness
    - Noise
    Saves results into a single CSV.

    Args:
        csv_path (str): Path to input CSV with 'image_id' column
        image_dir (str): Directory containing images
        output_csv (str): Output CSV path

    Returns:
        pd.DataFrame: DataFrame with quality metrics
    """
    df = pd.read_csv(csv_path)
    if 'image_id' not in df.columns:
        raise ValueError("CSV must contain a column named 'image_id'.")

    image_ids = df['image_id'].tolist()

    # Run metrics
    df["image_clarity"] = evaluate_clarity(image_dir, image_ids)
    df["image_contrast"] = evaluate_contrast(image_dir, image_ids)
    df["image_brightness"] = evaluate_brightness(image_dir, image_ids)
    df["image_noise"] = evaluate_noise(image_dir, image_ids)

    # Save
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Image Quality Classification saved to {output_csv}")

    return df
