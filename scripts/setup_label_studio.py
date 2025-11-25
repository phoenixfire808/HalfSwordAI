"""
Setup Label Studio for Half Sword Dataset
Prepares images and creates Label Studio project configuration
"""
import os
import json
from pathlib import Path

# Dataset path
DATASET_PATH = r"C:\Users\Drew\Pictures\Screenshots\Half sword data set\2025-11"
PROJECT_NAME = "Half Sword Dataset"
OUTPUT_DIR = Path("data/label_studio")

def create_label_studio_config():
    """Create Label Studio configuration for game screenshot labeling"""
    
    # Configuration for object detection (enemies, weapons, UI elements)
    config = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Enemy" background="red"/>
    <Label value="Weapon" background="blue"/>
    <Label value="Player" background="green"/>
    <Label value="UI_Element" background="yellow"/>
    <Label value="Health_Bar" background="orange"/>
    <Label value="Stamina_Bar" background="purple"/>
    <Label value="Damage_Indicator" background="pink"/>
  </RectangleLabels>
</View>
"""
    return config

def prepare_images_for_import():
    """Prepare image list for Label Studio import"""
    dataset_path = Path(DATASET_PATH)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {DATASET_PATH}")
        return None
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    
    for ext in image_extensions:
        images.extend(list(dataset_path.glob(f"*{ext}")))
        images.extend(list(dataset_path.glob(f"*{ext.upper()}")))
    
    if not images:
        print(f"No images found in {DATASET_PATH}")
        return None
    
    print(f"Found {len(images)} images")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare JSON for Label Studio import
    # Label Studio expects either:
    # 1. Local file paths (if running locally)
    # 2. URLs (if hosting images)
    # 3. Base64 encoded images
    
    # For local files, we'll create a JSON file with file paths
    tasks = []
    for img_path in sorted(images):
        # Use absolute path for Label Studio
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d={img_path.parent}/{img_path.name}"
            }
        })
    
    # Save tasks JSON
    tasks_file = OUTPUT_DIR / "tasks.json"
    with open(tasks_file, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Created tasks file: {tasks_file}")
    print(f"Total images: {len(tasks)}")
    
    return tasks_file, len(tasks)

def create_import_instructions():
    """Create instructions for importing into Label Studio"""
    instructions = f"""
# Label Studio Setup Instructions for Half Sword Dataset

## Dataset Location
{DATASET_PATH}

## Steps to Import:

1. **Start Label Studio:**
   ```bash
   label-studio
   ```
   Or if that doesn't work:
   ```bash
   python -m label_studio
   ```

2. **Create New Project:**
   - Open http://localhost:8080
   - Click "Create Project"
   - Project Name: "{PROJECT_NAME}"

3. **Configure Labeling Interface:**
   - Go to "Settings" → "Labeling Interface"
   - Paste the following configuration:
   
{create_label_studio_config()}

4. **Import Data:**
   - Go to "Import" tab
   - Choose "Upload Files" or "Upload Directory"
   - Select directory: {DATASET_PATH}
   - Or use the prepared tasks.json file: {OUTPUT_DIR / "tasks.json"}

5. **Start Labeling:**
   - Click on an image to start labeling
   - Draw rectangles around objects (enemies, weapons, UI elements)
   - Assign appropriate labels

## Alternative: Use Label Studio SDK

You can also use Python SDK to create the project programmatically:
```python
from label_studio_sdk import Client

LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = 'your-api-key-here'  # Get from Label Studio settings

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
project = ls.create_project(
    title='{PROJECT_NAME}',
    label_config=create_label_studio_config()
)

# Import tasks
project.import_tasks_from_file('{OUTPUT_DIR / "tasks.json"}')
```
"""
    
    instructions_file = OUTPUT_DIR / "SETUP_INSTRUCTIONS.md"
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"\nSetup instructions saved to: {instructions_file}")
    return instructions_file

if __name__ == "__main__":
    print("=" * 80)
    print("Half Sword Dataset - Label Studio Setup")
    print("=" * 80)
    
    # Prepare images
    result = prepare_images_for_import()
    if result:
        tasks_file, count = result
        print(f"\n✓ Prepared {count} images for import")
        print(f"✓ Tasks file: {tasks_file}")
    
    # Create instructions
    create_import_instructions()
    
    # Save config
    config_file = OUTPUT_DIR / "labeling_config.xml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(create_label_studio_config())
    print(f"✓ Labeling config saved to: {config_file}")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("1. Start Label Studio: label-studio")
    print("2. Open http://localhost:8080")
    print("3. Create a new project")
    print("4. Import images from:", DATASET_PATH)
    print("5. Copy the labeling config from:", config_file)
    print("=" * 80)

