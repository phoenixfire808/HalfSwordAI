"""
Import Half Sword Dataset to Label Studio using direct API calls
Works with both access tokens and refresh tokens
"""
import os
import json
import requests
from pathlib import Path

# Label Studio configuration
LABEL_STUDIO_URL = 'http://localhost:8080'
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

def make_request(method, endpoint, data=None, headers=None):
    """Make API request to Label Studio"""
    url = f"{LABEL_STUDIO_URL}{endpoint}"
    default_headers = {
        'Authorization': f'Token {API_TOKEN}',
        'Content-Type': 'application/json'
    }
    if headers:
        default_headers.update(headers)
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=default_headers)
        elif method == 'POST':
            response = requests.post(url, headers=default_headers, json=data)
        elif method == 'PATCH':
            response = requests.patch(url, headers=default_headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_or_create_project():
    """Get existing project or create new one"""
    # Try to get existing projects
    projects = make_request('GET', '/api/projects')
    
    if projects and 'results' in projects:
        for project in projects['results']:
            if project.get('title') == PROJECT_NAME:
                print(f"Found existing project: {PROJECT_NAME} (ID: {project['id']})")
                return project['id']
    
    # Create new project
    print(f"Creating new project: {PROJECT_NAME}")
    project_data = {
        'title': PROJECT_NAME,
        'label_config': LABELING_CONFIG,
        'description': 'Half Sword game screenshot dataset for object detection'
    }
    
    project = make_request('POST', '/api/projects', data=project_data)
    if project:
        print(f"Created project: {PROJECT_NAME} (ID: {project['id']})")
        return project['id']
    
    return None

def prepare_tasks():
    """Prepare tasks from image directory"""
    if not DATASET_PATH.exists():
        print(f"Error: Dataset path does not exist: {DATASET_PATH}")
        return []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    
    for ext in image_extensions:
        images.extend(list(DATASET_PATH.glob(f"*{ext}")))
        images.extend(list(DATASET_PATH.glob(f"*{ext.upper()}")))
    
    if not images:
        print(f"No images found in {DATASET_PATH}")
        return []
    
    print(f"Found {len(images)} images")
    
    # Prepare tasks - for local files, we'll use file paths
    tasks = []
    for img_path in sorted(images):
        abs_path = img_path.resolve()
        # Use Windows path format
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d={abs_path.parent}/{abs_path.name}"
            }
        })
    
    return tasks

def import_tasks(project_id, tasks):
    """Import tasks to project"""
    if not tasks:
        print("No tasks to import")
        return
    
    print(f"Importing {len(tasks)} tasks to project {project_id}...")
    
    # Import tasks
    result = make_request('POST', f'/api/projects/{project_id}/import', data=tasks)
    
    if result:
        print(f"✓ Successfully imported tasks")
        return True
    else:
        print("Failed to import tasks. Trying individual imports...")
        # Try importing one by one
        success_count = 0
        for task in tasks:
            result = make_request('POST', f'/api/projects/{project_id}/import', data=[task])
            if result:
                success_count += 1
        
        print(f"Imported {success_count}/{len(tasks)} tasks")
        return success_count > 0

def main():
    """Main function"""
    print("=" * 80)
    print("Half Sword Dataset - Label Studio Import (Direct API)")
    print("=" * 80)
    
    # Test connection
    print("\n1. Testing connection to Label Studio...")
    test = make_request('GET', '/api/version')
    if test:
        print(f"✓ Connected to Label Studio (version: {test.get('version', 'unknown')})")
    else:
        print("✗ Failed to connect. Check if Label Studio is running and token is correct.")
        print("\nTo get your access token:")
        print("1. Go to http://localhost:8080")
        print("2. Account Settings → Access Token")
        print("3. Copy the token and update API_TOKEN in this script")
        return
    
    # Get or create project
    print("\n2. Setting up project...")
    project_id = get_or_create_project()
    if not project_id:
        print("Failed to create/get project")
        return
    
    print(f"✓ Project ready: {PROJECT_NAME}")
    print(f"  Project URL: {LABEL_STUDIO_URL}/projects/{project_id}/")
    
    # Prepare tasks
    print("\n3. Preparing tasks from dataset...")
    tasks = prepare_tasks()
    if not tasks:
        print("No tasks to import")
        return
    
    print(f"✓ Prepared {len(tasks)} tasks")
    
    # Import tasks
    print("\n4. Importing tasks...")
    if import_tasks(project_id, tasks):
        print("\n" + "=" * 80)
        print("Import Complete!")
        print("=" * 80)
        print(f"Project URL: {LABEL_STUDIO_URL}/projects/{project_id}/")
        print(f"Total tasks: {len(tasks)}")
        print("\nYou can now start labeling in Label Studio!")
        print("=" * 80)
    else:
        print("\nImport had some issues. Check the errors above.")

if __name__ == "__main__":
    main()

