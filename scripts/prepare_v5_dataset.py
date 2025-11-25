"""
Prepare Half Sword v5 Dataset for Training
This dataset is already in YOLOv8 format, just needs path fixes
"""
import os
import shutil
from pathlib import Path

# Paths
SOURCE_DATASET = Path("Half Sword.v5i.yolov8")
TARGET_DATASET = Path("data/yolo_dataset_v5")

print("=" * 80)
print("Preparing Half Sword v5 Dataset for Training")
print("=" * 80)

# Copy dataset structure
print("\n1. Copying dataset structure...")
if TARGET_DATASET.exists():
    print(f"   Removing existing dataset at {TARGET_DATASET}")
    shutil.rmtree(TARGET_DATASET)

# Create directory structure
TARGET_DATASET.mkdir(parents=True, exist_ok=True)
for split in ['train', 'valid', 'test']:
    (TARGET_DATASET / split / 'images').mkdir(parents=True, exist_ok=True)
    (TARGET_DATASET / split / 'labels').mkdir(parents=True, exist_ok=True)

# Copy files
for split in ['train', 'valid', 'test']:
    source_split = SOURCE_DATASET / split
    target_split = TARGET_DATASET / split
    
    if source_split.exists():
        # Copy images
        source_images = source_split / 'images'
        target_images = target_split / 'images'
        if source_images.exists():
            for img_file in source_images.glob('*.jpg'):
                shutil.copy2(img_file, target_images / img_file.name)
            print(f"   Copied {len(list(source_images.glob('*.jpg')))} images from {split}")
        
        # Copy labels
        source_labels = source_split / 'labels'
        target_labels = target_split / 'labels'
        if source_labels.exists():
            for label_file in source_labels.glob('*.txt'):
                shutil.copy2(label_file, target_labels / label_file.name)
            print(f"   Copied {len(list(source_labels.glob('*.txt')))} labels from {split}")

# Create data.yaml with correct paths
print("\n2. Creating data.yaml...")
yaml_content = f"""# Half Sword Dataset v5 - YOLOv8 Format
# Dataset from Roboflow (v5)

path: {TARGET_DATASET.absolute()}
train: train/images
val: valid/images
test: test/images

# Number of classes
nc: 4

# Class names
names:
  0: Blood
  1: Enemy
  2: Player
  3: 'You Won'
"""
yaml_path = TARGET_DATASET / "data.yaml"
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"   Created data.yaml at {yaml_path}")

# Count annotations
print("\n3. Dataset Statistics:")
for split in ['train', 'valid', 'test']:
    split_path = TARGET_DATASET / split
    images_count = len(list((split_path / 'images').glob('*.jpg'))) if (split_path / 'images').exists() else 0
    labels_count = len(list((split_path / 'labels').glob('*.txt'))) if (split_path / 'labels').exists() else 0
    
    # Count total annotations
    total_annotations = 0
    if (split_path / 'labels').exists():
        for label_file in (split_path / 'labels').glob('*.txt'):
            with open(label_file, 'r') as f:
                total_annotations += len([line for line in f if line.strip()])
    
    print(f"   {split.upper()}: {images_count} images, {labels_count} labels, {total_annotations} annotations")

print("\n" + "=" * 80)
print("Dataset Preparation Complete!")
print("=" * 80)
print(f"\nDataset ready at: {TARGET_DATASET.absolute()}")
print(f"YAML config: {yaml_path.absolute()}")
print("\nTo train the model:")
print(f"  python scripts/train_yolo_model_v5.py")

