# Half Sword AI Agent - PowerShell Launcher
# Right-click and "Run with PowerShell" or run: powershell -ExecutionPolicy Bypass -File start_agent.ps1

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  Half Sword AI Agent - Starting..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and add it to your PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if main.py exists
if (-not (Test-Path "main.py")) {
    Write-Host "ERROR: main.py not found in current directory" -ForegroundColor Red
    Write-Host "Current directory: $scriptDir" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Starting Half Sword AI Agent..." -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the agent" -ForegroundColor Yellow
Write-Host "Press F8 for emergency kill switch" -ForegroundColor Yellow
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Run the Python script
try {
    python main.py
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to start agent" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  Agent has stopped" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"

