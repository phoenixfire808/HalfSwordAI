# Roboflow Setup Guide for Half Sword Dataset

Roboflow is a great choice for computer vision datasets! Here's how to get started:

## Roboflow Advantages:
- ✅ Better UI for object detection
- ✅ Built-in dataset versioning
- ✅ Easy export formats (YOLO, COCO, etc.)
- ✅ Preprocessing and augmentation tools
- ✅ Model training integration

## Setup Steps:

### 1. Create Roboflow Account
- Go to: https://roboflow.com
- Sign up for free account
- Verify your email

### 2. Create New Project
- Click "Create New Project"
- Project Name: **Half Sword Dataset**
- Project Type: **Object Detection**
- License: Choose appropriate (usually "MIT" or "CC BY 4.0")

### 3. Upload Your Images
- Go to "Upload" tab
- Drag and drop images from: `C:\Users\Drew\Pictures\Screenshots\Half sword data set\2025-11`
- Or use the upload folder option
- Wait for images to process

### 4. Start Labeling
- Click on an image
- Use the bounding box tool to draw rectangles
- Assign labels:
  - Enemy
  - Weapon
  - Player
  - UI_Element
  - Health_Bar
  - Stamina_Bar
  - Damage_Indicator

### 5. Export Dataset
- Once labeled, go to "Generate" tab
- Choose export format (YOLO is recommended for most ML frameworks)
- Click "Generate" to create dataset version
- Download or use API to access

## Roboflow Python SDK

If you want to integrate with your Half Sword AI project:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="your-api-key")
project = rf.workspace("your-workspace").project("half-sword-dataset")
dataset = project.version(1).download("yolov8")
```

## Next Steps:
1. Set up your Roboflow account
2. Upload the 12 images
3. Start labeling
4. Export when ready for training

Need help with any specific Roboflow setup or integration?

