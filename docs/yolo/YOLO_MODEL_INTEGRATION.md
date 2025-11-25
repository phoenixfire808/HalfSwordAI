# YOLO Model Integration - Half Sword v5 Dataset

## Overview
The Half Sword AI agent now uses a custom-trained YOLOv8 model specifically trained on Half Sword gameplay data.

## Model Details

### Dataset
- **Version:** v5 (Roboflow)
- **Total Images:** 15
- **Split:**
  - Train: 9 images (70 annotations)
  - Validation: 4 images (11 annotations)
  - Test: 2 images (28 annotations)

### Classes
The model detects 4 classes:
1. **Blood** - Blood splatter effects
2. **Enemy** - Enemy characters
3. **Player** - Player character
4. **You Won** - Victory screen UI text

### Performance Metrics
- **Precision:** 91.4%
- **Recall:** 64.3%
- **mAP50:** 66.3%
- **mAP50-95:** 51.1%

## Integration Points

### Configuration (`half_sword_ai/config/__init__.py`)
```python
YOLO_MODEL_PATH = "yolo_training/half_sword_detector2/weights/best.pt"
YOLO_USE_CUSTOM_MODEL = True
YOLO_CONFIDENCE_THRESHOLD = 0.5
```

### YOLO Detector (`half_sword_ai/perception/yolo_detector.py`)
- Automatically loads custom model from config
- Updated class names to match dataset
- Handles path resolution (relative/absolute)

### Vision Processor (`half_sword_ai/perception/vision.py`)
- Initializes YOLODetector with config settings
- Uses custom model when `YOLO_USE_CUSTOM_MODEL = True`

## Usage

The model is automatically loaded when the agent starts:

```python
from half_sword_ai.perception.yolo_detector import YOLODetector

detector = YOLODetector()  # Automatically loads custom model from config
detections = detector.detect(frame)
```

## Model Files

- **Model Path:** `yolo_training/half_sword_detector2/weights/best.pt`
- **Training Logs:** `yolo_training/half_sword_detector2/results.csv`
- **Dataset Config:** `data/yolo_dataset_v5/data.yaml`

## Next Steps

1. **Collect More Data:** Add more annotated images to improve recall
2. **Fine-tune:** Continue training on new gameplay data
3. **Monitor Performance:** Track detection accuracy in real gameplay
4. **Self-Learning:** Enable YOLO self-learning to improve over time

## Training New Models

To train a new model:
```bash
python scripts/train_yolo_model_v5.py
```

To prepare a new dataset:
```bash
python scripts/prepare_v5_dataset.py
```

