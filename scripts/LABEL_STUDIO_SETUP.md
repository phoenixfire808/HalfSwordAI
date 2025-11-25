# Label Studio Setup Guide

## Getting Your Access Token

The token you provided is a **refresh token**, but Label Studio API needs an **access token**. Here's how to get it:

### Steps:
1. **Open Label Studio**: http://localhost:8080
2. **Log in** to your account
3. **Click your profile icon** (top right corner)
4. **Go to**: "Account & Settings" → "Access Token" tab
5. **Copy the token** (it will be a long alphanumeric string, not a JWT)

### Alternative Method:
If you don't see "Access Token" in settings:
1. Go to: http://localhost:8080/user/account
2. Look for "Access Token" section
3. Click "Create Token" if needed
4. Copy the token

## Manual Import (Easiest Method)

Since we're having token issues, here's the easiest way to import your dataset:

### Step 1: Create Project
1. Open http://localhost:8080
2. Click "Create Project"
3. Name: **Half Sword Dataset**
4. Click "Create"

### Step 2: Configure Labeling Interface
1. Go to **Settings** → **Labeling Interface**
2. Copy the contents from: `data/label_studio/labeling_config.xml`
3. Paste into the editor
4. Click **Save**

### Step 3: Import Images
1. Go to **Import** tab
2. Choose **"Upload Directory"**
3. Select: `C:\Users\Drew\Pictures\Screenshots\Half sword data set\2025-11`
4. Click **Import**

### Step 4: Start Labeling!
- Click on any image
- Draw rectangles around objects
- Assign labels (Enemy, Weapon, Player, etc.)

## Using the Script (Once You Have Access Token)

Once you have the correct access token:

1. Update `scripts/import_to_label_studio_direct.py`:
   ```python
   API_TOKEN = 'your-access-token-here'
   ```

2. Run the script:
   ```bash
   python scripts/import_to_label_studio_direct.py
   ```

This will automatically:
- Create the project
- Configure the labeling interface
- Import all 12 images

## Current Status

- ✅ Label Studio is installed
- ✅ Dataset found (12 images)
- ✅ Labeling configuration ready
- ⏳ Waiting for correct access token OR manual import

