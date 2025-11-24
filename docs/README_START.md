# Starting the Half Sword AI Agent

## Quick Start Options

### Option 1: Double-Click Batch File (Recommended for Windows)
**File**: `start_agent.bat`
- Simply double-click `start_agent.bat`
- No terminal/command line needed
- Automatically checks for Python
- Shows clear error messages if something is wrong

### Option 2: Double-Click Python Script
**File**: `start_agent.py`
- Double-click `start_agent.py` (if Python file associations are set up)
- Or run: `python start_agent.py`

### Option 3: Double-Click VBS Launcher
**File**: `start_agent.vbs`
- Double-click `start_agent.vbs`
- Runs Python script in background
- No console window (for cleaner startup)

### Option 4: Command Line (Traditional)
```bash
python main.py
```

## What Each Launcher Does

### `start_agent.bat` (Windows Batch File)
- ✅ Checks if Python is installed
- ✅ Verifies main.py exists
- ✅ Changes to correct directory
- ✅ Shows clear error messages
- ✅ Pauses on exit so you can see any errors

### `start_agent.py` (Python Launcher)
- ✅ Handles imports gracefully
- ✅ Shows full error tracebacks
- ✅ Changes to correct directory automatically
- ✅ Cross-platform (works on Windows, Linux, Mac)

### `start_agent.vbs` (Visual Basic Script)
- ✅ Runs Python script silently
- ✅ No console window
- ✅ Good for background operation

## Troubleshooting

### "Python is not installed"
- Install Python 3.8+ from python.org
- Make sure to check "Add Python to PATH" during installation

### "main.py not found"
- Make sure you're in the project directory
- The launcher should handle this automatically

### "Module not found" errors
- Install dependencies: `pip install -r requirements.txt`
- Or install manually: `pip install torch numpy opencv-python flask flask-cors`

### Bash/Shell Errors
- Use `start_agent.bat` instead of command line
- Avoids all bash/terminal issues
- Works directly from Windows Explorer

## Recommended Method

**For Windows users**: Double-click `start_agent.bat`
- Easiest method
- No command line needed
- Clear error messages
- Handles everything automatically

---

**Note**: The launcher scripts automatically handle directory changes and error checking, so you can run them from anywhere.

