# Manual Import Guide - Half Sword Dataset

Since we're having token authentication issues, here's the fastest way to import your dataset manually:

## Step 1: Create Project
1. Open http://localhost:8080
2. Click **"Create Project"**
3. Project Name: **Half Sword Dataset**
4. Click **"Create"**

## Step 2: Configure Labeling Interface
1. Go to **Settings** → **Labeling Interface**
2. Delete any existing content
3. Copy and paste this configuration:

```xml
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
```

4. Click **"Save"**

## Step 3: Import Images
1. Go to **Import** tab
2. Click **"Upload Directory"** or **"Upload Files"**
3. Navigate to: `C:\Users\Drew\Pictures\Screenshots\Half sword data set\2025-11`
4. Select all images (or the folder)
5. Click **"Import"**

## Step 4: Start Labeling!
- Click on any image to start labeling
- Draw rectangles around objects (enemies, weapons, UI elements)
- Assign labels from the dropdown
- Use keyboard shortcuts for faster labeling

## Labeling Tips:
- **W** - Create rectangle
- **A/D** - Previous/Next image
- **Delete** - Remove selected annotation
- **Ctrl+Z** - Undo

Your 12 images should now be ready for labeling!

