README.md
# VisiGrade - Image Quality Evaluation Tool

VisiGrade is a Python tool that evaluates image quality based on:
- **Clarity** (sharpness via edge detection)
- **Contrast** (grayscale histogram spread)
- **Brightness** (perceptual luminance)
- **Noise** (BRISQUE score)

## Installation
```bash
git clone https://github.com/your-username/visigrade.git
cd visigrade
pip install -r requirements.txt
```
## Usage

Your CSV must have a column image_id containing image filenames.
```
from visigrade import run_image_quality_pipeline

df = run_image_quality_pipeline(
    csv_path="data.csv",
    image_dir="path/to/images/",
    output_csv="Image_Quality_Classification.csv"
)
```

The tool will generate a new CSV with the following columns:

image_clarity

image_contrast

image_brightness

image_noise

## Repository Structure
```
metrics/
 ├── clarity.py
 ├── contrast.py
 ├── brightness.py
 ├── noise.py
visigrade.py
requirements.txt
README.md
```
