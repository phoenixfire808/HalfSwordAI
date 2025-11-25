"""
Import Half Sword Dataset to Label Studio using API
"""
import os
import json
from pathlib import Path
from label_studio_sdk import Client

# Label Studio configuration
LABEL_STUDIO_URL = 'http://localhost:8080'
# NOTE: This should be an ACCESS TOKEN, not a refresh token
# Get it from: Label Studio → Account Settings → Access Token
API_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MTIzMjM0OSwiaWF0IjoxNzY0MDMyMzQ5LCJqdGkiOiIzMzc2NWU0MTlkZDg0YTEyYTA3MTk5MTRhNmM5ZjQzZSIsInVzZXJfaWQiOiIxIn0.2iAYTBl1B8ay2a82dvfxs3lbxLJzyaqnAsOwhP8AYgs'

# Dataset path
DATASET_PATH = Path(r"C:\Users\Drew\Pictures\Screenshots\Half sword data set\2025-11")
PROJECT_NAME = "Half Sword Dataset"

# Labeling configuration
LABELING_CONFIG = """<View>
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
</View>"""

def get_label_studio_client():
    """Initialize Label Studio client"""
    try:
        client = Client(url=LABEL_STUDIO_URL, api_key=API_TOKEN)
        return client
    except Exception as e:
        print(f"Error connecting to Label Studio: {e}")
        return None

def create_or_get_project(client):
    """Create project or get existing one"""
    # Check if project already exists
    projects = client.get_projects()
    for project in projects:
        if project.title == PROJECT_NAME:
            print(f"Found existing project: {PROJECT_NAME} (ID: {project.id})")
            return project
    
    # Create new project
    print(f"Creating new project: {PROJECT_NAME}")
    project = client.create_project(
        title=PROJECT_NAME,
        label_config=LABELING_CONFIG,
        description="Half Sword game screenshot dataset for object detection"
    )
    print(f"Created project: {PROJECT_NAME} (ID: {project.id})")
    return project

def prepare_tasks_from_directory(dataset_path):
    """Prepare tasks from image directory"""
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    
    for ext in image_extensions:
        images.extend(list(dataset_path.glob(f"*{ext}")))
        images.extend(list(dataset_path.glob(f"*{ext.upper()}")))
    
    if not images:
        print(f"No images found in {dataset_path}")
        return []
    
    print(f"Found {len(images)} images")
    
    # Prepare tasks
    # For local files, we need to use the file:// protocol or upload them
    tasks = []
    for img_path in sorted(images):
        # Use absolute Windows path
        abs_path = img_path.resolve()
        # Label Studio local file format
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d={abs_path.parent}/{abs_path.name}"
            }
        })
    
    return tasks

def import_tasks(project, tasks):
    """Import tasks to project"""
    if not tasks:
        print("No tasks to import")
        return
    
    print(f"Importing {len(tasks)} tasks to project...")
    
    # Import tasks in batches
    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        try:
            project.import_tasks(batch)
            print(f"Imported batch {i//batch_size + 1} ({len(batch)} tasks)")
        except Exception as e:
            print(f"Error importing batch {i//batch_size + 1}: {e}")
            # Try individual imports
            for task in batch:
                try:
                    project.import_tasks([task])
                except Exception as task_error:
                    print(f"  Failed to import {task['data']['image']}: {task_error}")

def main():
    """Main function"""
    print("=" * 80)
    print("Half Sword Dataset - Label Studio Import")
    print("=" * 80)
    
    # Connect to Label Studio
    print("\n1. Connecting to Label Studio...")
    client = get_label_studio_client()
    if not client:
        print("Failed to connect to Label Studio. Make sure it's running at http://localhost:8080")
        return
    
    print("✓ Connected to Label Studio")
    
    # Create or get project
    print("\n2. Setting up project...")
    project = create_or_get_project(client)
    if not project:
        print("Failed to create/get project")
        return
    
    print(f"✓ Project ready: {project.title}")
    print(f"  Project URL: {LABEL_STUDIO_URL}/projects/{project.id}/")
    
    # Prepare tasks
    print("\n3. Preparing tasks from dataset...")
    tasks = prepare_tasks_from_directory(DATASET_PATH)
    if not tasks:
        print("No tasks to import")
        return
    
    print(f"✓ Prepared {len(tasks)} tasks")
    
    # Import tasks
    print("\n4. Importing tasks...")
    import_tasks(project, tasks)
    
    print("\n" + "=" * 80)
    print("Import Complete!")
    print("=" * 80)
    print(f"Project URL: {LABEL_STUDIO_URL}/projects/{project.id}/")
    print(f"Total tasks: {len(tasks)}")
    print("\nYou can now start labeling in Label Studio!")
    print("=" * 80)

if __name__ == "__main__":
    main()

