import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Directories for the dataset
DATASET_DIR = "input/Katarak2"
OUTPUT_DIR = "input/Katarak2_Split"
TRAIN_RATIO = 0.8  # 80% training, 20% validation

# Paths for images and annotations
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
ANNOT_DIR = os.path.join(DATASET_DIR, "annotations")

# Output directories for train/val split
TRAIN_IMG_DIR = os.path.join(OUTPUT_DIR, "train", "images")
TRAIN_ANNOT_DIR = os.path.join(OUTPUT_DIR, "train", "annotations")
VAL_IMG_DIR = os.path.join(OUTPUT_DIR, "val", "images")
VAL_ANNOT_DIR = os.path.join(OUTPUT_DIR, "val", "annotations")

# Ensure output directories exist
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_ANNOT_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_ANNOT_DIR, exist_ok=True)

# Collect all image filenames
image_filenames = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

# Split dataset into train and validation
train_images, val_images = train_test_split(image_filenames, train_size=TRAIN_RATIO, random_state=42)

# Helper function to copy files
def copy_files(file_list, src_dir, dst_dir):
    for filename in file_list:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy(src_path, dst_path)

# Copy train images and annotations
copy_files(train_images, IMAGE_DIR, TRAIN_IMG_DIR)
copy_files([f.replace(".jpg", ".xml") for f in train_images], ANNOT_DIR, TRAIN_ANNOT_DIR)

# Copy validation images and annotations
copy_files(val_images, IMAGE_DIR, VAL_IMG_DIR)
copy_files([f.replace(".jpg", ".xml") for f in val_images], ANNOT_DIR, VAL_ANNOT_DIR)

print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")
print(f"Dataset split completed and saved in {OUTPUT_DIR}")
