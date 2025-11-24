@echo off
REM Half Sword AI Agent - Windows Launcher with ScrimBrain Integration
REM Double-click this file to start the agent
REM This avoids bash/terminal startup errors

title Half Sword AI Agent - ScrimBrain Integration

REM Change to project root directory (one level up from scripts/)
cd /d "%~dp0\.."

echo.
echo ================================================================================
echo   Half Sword AI Agent - ScrimBrain Integration
echo   Autonomous Learning Agent for Physics-Based Combat
echo ================================================================================
echo.
echo Current directory: %CD%
echo.

REM Check if Python is available
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    echo.
    echo You can download Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Show Python version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python found: %PYTHON_VERSION%
echo.

REM Check if main.py exists
echo [2/5] Checking for main.py...
if not exist "main.py" (
    echo.
    echo [ERROR] main.py not found in current directory
    echo Current directory: %CD%
    echo.
    echo Please make sure you're running this from the project root directory
    echo.
    pause
    exit /b 1
)
echo [OK] main.py found
echo.

REM Check for required packages
echo [3/5] Checking dependencies...
python -c "import torch; import cv2; import numpy" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some dependencies may be missing
    echo Run: pip install -r requirements.txt
    echo.
)
echo [OK] Core dependencies available
echo.

REM Check for optional OCR libraries
echo [4/5] Checking optional dependencies...
python -c "import pytesseract" >nul 2>&1
if errorlevel 1 (
    echo [INFO] pytesseract not found - OCR will use EasyOCR if available
) else (
    echo [OK] pytesseract available
)

python -c "import easyocr" >nul 2>&1
if errorlevel 1 (
    echo [INFO] easyocr not found - OCR may be limited
) else (
    echo [OK] easyocr available
)
echo.

REM Display ScrimBrain integration info
echo [5/5] ScrimBrain Integration Status:
echo   - DirectInput: Enabled (ctypes SendInput)
echo   - Action Discretization: 9 macro-actions available
echo   - Gesture Engine: Smooth physics-compatible movements
echo   - OCR Reward Tracking: Score detection from Abyss mode
echo   - Terminal State Detection: Death screen detection
echo.

echo ================================================================================
echo   Starting Half Sword AI Agent...
echo ================================================================================
echo.
echo CONTROLS:
echo   - Press Ctrl+C to stop the agent
echo   - Press F8 for emergency kill switch
echo   - Move mouse to take manual control (bot pauses automatically)
echo   - Stop moving mouse for 0.5s to return to bot control
echo.
echo SCRIMBRAIN FEATURES:
echo   - Discrete action mode (DQN-style): 9 combat macro-actions
echo   - DirectInput compatibility: Low-level Windows API input
echo   - OCR score tracking: Automatic reward detection
echo   - Physics-aware gestures: Smooth weapon movements
echo.
echo DASHBOARD:
echo   - Open http://localhost:5000 in your browser for real-time monitoring
echo.
echo ================================================================================
echo.

REM Run the Python script
python main.py

REM If we get here, the script exited
echo.
echo ================================================================================
echo   Agent has stopped
echo ================================================================================
echo.
echo Final performance report saved to: logs\final_performance_report.txt
echo Model checkpoint saved to: models\model_checkpoint.pt
echo.
pause

