"""
Convert Roboflow TensorFlow dataset to YOLO format
For Half Sword AI project
"""
import os
import csv
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
ROBOFLOW_DATASET = Path("Half Sword.v1i.tensorflow")
OUTPUT_DATASET = Path("data/yolo_dataset")
# Classes will be auto-detected from CSV
CLASSES = []  # Will be populated from annotations

def parse_csv_annotations(csv_path):
    """Parse Roboflow CSV annotations"""
    annotations = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '')
            if not filename:
                continue
            
            # Parse bounding box (format: xmin,ymin,xmax,ymax)
            xmin = float(row.get('xmin', 0))
            ymin = float(row.get('ymin', 0))
            xmax = float(row.get('xmax', 0))
            ymax = float(row.get('ymax', 0))
            class_name = row.get('class', 'Fighting')
            
            if filename not in annotations:
                annotations[filename] = []
            
            annotations[filename].append({
                'class': class_name,
                'bbox': [xmin, ymin, xmax, ymax]
            })
    
    return annotations

def convert_to_yolo_format(bbox, img_width, img_height):
    """Convert bounding box from absolute to YOLO format (normalized center x, y, width, height)"""
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center and dimensions
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Normalize
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return center_x_norm, center_y_norm, width_norm, height_norm

def get_image_size(image_path):
    """Get image dimensions"""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    height, width = img.shape[:2]
    return width, height

def convert_dataset():
    """Convert Roboflow TensorFlow dataset to YOLO format"""
    logger.info("=" * 80)
    logger.info("Converting Roboflow TensorFlow Dataset to YOLO Format")
    logger.info("=" * 80)
    
    # Check if dataset exists
    if not ROBOFLOW_DATASET.exists():
        logger.error(f"Dataset not found: {ROBOFLOW_DATASET}")
        return False
    
    csv_path = ROBOFLOW_DATASET / "train" / "_annotations.csv"
    if not csv_path.exists():
        logger.error(f"Annotations CSV not found: {csv_path}")
        return False
    
    # Parse annotations
    logger.info("Parsing annotations...")
    annotations = parse_csv_annotations(csv_path)
    logger.info(f"Found annotations for {len(annotations)} images")
    
    # Get unique classes
    all_classes = set()
    for anns in annotations.values():
        for ann in anns:
            all_classes.add(ann['class'])
    
    classes = sorted(list(all_classes))
    logger.info(f"Classes found: {classes}")
    
    # Create YOLO dataset structure
    output_train = OUTPUT_DATASET / "train"
    output_train_images = output_train / "images"
    output_train_labels = output_train / "labels"
    
    output_train_images.mkdir(parents=True, exist_ok=True)
    output_train_labels.mkdir(parents=True, exist_ok=True)
    
    # Copy images and create label files
    train_dir = ROBOFLOW_DATASET / "train"
    converted_count = 0
    skipped_count = 0
    
    import cv2
    
    for image_file in train_dir.glob("*.jpg"):
        if image_file.name == "_annotations.csv":
            continue
        
        # Get image dimensions
        img = cv2.imread(str(image_file))
        if img is None:
            logger.warning(f"Could not read image: {image_file}")
            skipped_count += 1
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Copy image
        dest_image = output_train_images / image_file.name
        shutil.copy2(image_file, dest_image)
        
        # Create label file
        label_file = output_train_labels / (image_file.stem + ".txt")
        
        # Get annotations for this image
        image_annotations = annotations.get(image_file.name, [])
        
        if not image_annotations:
            logger.warning(f"No annotations for {image_file.name}")
            # Create empty label file
            label_file.write_text("")
            skipped_count += 1
            continue
        
        # Write YOLO format labels
        with open(label_file, 'w') as f:
            for ann in image_annotations:
                class_name = ann['class']
                class_id = classes.index(class_name) if class_name in classes else 0
                
                bbox = ann['bbox']
                x_center, y_center, width, height = convert_to_yolo_format(
                    bbox, img_width, img_height
                )
                
                # YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        converted_count += 1
    
    # Create dataset.yaml for YOLO
    dataset_yaml = OUTPUT_DATASET / "dataset.yaml"
    yaml_content = f"""# Half Sword Dataset - YOLO Format
# Converted from Roboflow TensorFlow format

path: {OUTPUT_DATASET.absolute()}
train: train/images
val: train/images  # Using same for validation (small dataset)

# Number of classes
nc: {len(classes)}

# Class names
names:
"""
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"
    
    with open(dataset_yaml, 'w') as f:
        f.write(yaml_content)
    
    logger.info("=" * 80)
    logger.info("Conversion Complete!")
    logger.info("=" * 80)
    logger.info(f"Converted: {converted_count} images")
    logger.info(f"Skipped: {skipped_count} images")
    logger.info(f"Classes: {classes}")
    logger.info(f"Output directory: {OUTPUT_DATASET.absolute()}")
    logger.info(f"Dataset YAML: {dataset_yaml.absolute()}")
    logger.info("=" * 80)
    
    return True

if __name__ == "__main__":
    success = convert_dataset()
    if success:
        print("\n✓ Dataset converted successfully!")
        print(f"\nTo train YOLO model, use:")
        print(f"  python -c \"from half_sword_ai.perception.yolo_detector import YOLODetector; detector = YOLODetector(); detector.train_custom_model('{OUTPUT_DATASET}/dataset.yaml', epochs=100)\"")
    else:
        print("\n✗ Conversion failed. Check errors above.")

