@echo off
REM Half Sword Dataset Builder - Windows Launcher
REM Run this script to collect training data

title Half Sword Dataset Builder

cd /d "%~dp0\.."

echo.
echo ================================================================================
echo   Half Sword Dataset Builder
echo   Collect training data by playing the game
echo ================================================================================
echo.
echo Instructions:
echo   1. Make sure Half Sword is running
echo   2. Play the game normally
echo   3. All frames, actions, and game state will be recorded
echo   4. Press Ctrl+C to stop and save dataset
echo.
echo ================================================================================
echo.

python scripts/build_dataset.py

pause

