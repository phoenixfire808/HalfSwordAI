# Quick Start Guide - Half Sword AI Agent

## Starting the Agent

### Option 1: Double-Click Batch File (Easiest)
1. Double-click `start.bat` in the project root directory
2. The agent will start automatically

### Option 2: Use Scripts Folder
1. Navigate to `scripts/` folder
2. Double-click `start_agent.bat`

### Option 3: Command Line
```bash
python main.py
```

## First-Time Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install OCR (Optional but Recommended)
For score tracking in Abyss mode:
```bash
# Option A: Tesseract OCR
pip install pytesseract
# Also install Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki

# Option B: EasyOCR (easier, no binary needed)
pip install easyocr
```

### 3. Configure Game Path
Edit `half_sword_ai/config/__init__.py`:
```python
GAME_EXECUTABLE_PATH: str = r"D:\Steam\steamapps\common\Half Sword Demo\HalfSwordUE5.exe"
```

## Controls

- **Ctrl+C**: Stop the agent
- **F8**: Emergency kill switch (immediate stop)
- **Mouse Movement**: Take manual control (bot pauses)
- **Stop Moving Mouse**: Return to bot control after 0.5s

## ScrimBrain Features

The agent includes ScrimBrain integration with:

- ✅ **DirectInput**: Low-level Windows API input (ctypes)
- ✅ **Discrete Actions**: 9 combat macro-actions (DQN-style)
- ✅ **Gesture Engine**: Smooth physics-compatible movements
- ✅ **OCR Score Tracking**: Automatic reward detection
- ✅ **Terminal State Detection**: Death screen detection

## Dashboard

Once the agent is running:
- Open http://localhost:5000 in your browser
- View real-time metrics, training progress, and system status

## Troubleshooting

### Python Not Found
- Install Python 3.8+ from https://www.python.org/
- Check "Add Python to PATH" during installation

### Dependencies Missing
```bash
pip install -r requirements.txt
```

### Game Not Detected
- Make sure Half Sword is running
- Check `GAME_PROCESS_NAME` in config matches your game executable
- Enable `AUTO_LAUNCH_GAME` in config to auto-start the game

### OCR Not Working
- Install at least one OCR library (pytesseract or easyocr)
- For pytesseract: Install Tesseract binary separately
- OCR is optional - agent will work without it (no score tracking)

## Safety Notes

⚠️ **Important**: 
- Use in offline/demo mode only
- Easy Anti-Cheat (EAC) may detect synthetic input
- Do not use in online multiplayer

## Next Steps

1. Start the agent with `start.bat`
2. Let it run and collect training data
3. Monitor progress in the dashboard
4. Model checkpoints saved to `models/` folder
5. Performance reports saved to `logs/` folder

For detailed information, see `SCRIMBRAIN_INTEGRATION.md`

