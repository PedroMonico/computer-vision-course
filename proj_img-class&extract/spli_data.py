## from chatgpt
import os
import shutil
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Define paths
base_dir = Path("data/weather_dataset")
categories = ['cloudy', 'rain', 'shine', 'sunrise']
train_dir = base_dir / "train"
test_dir = base_dir / "test"

# Create target directories
for category in categories:
    (train_dir / category).mkdir(parents=True, exist_ok=True)
    (test_dir / category).mkdir(parents=True, exist_ok=True)

# Split and copy files
for category in categories:
    source_dir = base_dir / category
    images = list(source_dir.glob("*"))  # or use `*.jpg` if specific
    random.shuffle(images)

    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Copy files to train and test directories
    for img in train_images:
        shutil.copy(img, train_dir / category / img.name)
    for img in test_images:
        shutil.copy(img, test_dir / category / img.name)

print("Dataset split complete.")
