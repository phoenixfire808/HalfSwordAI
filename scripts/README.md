# Launcher Scripts

This directory contains launcher scripts for starting the Half Sword AI Agent.

## Available Launchers

- **start_agent.bat** - Windows batch file (recommended for Windows)
- **start_agent.py** - Python launcher (cross-platform)
- **start_agent.ps1** - PowerShell script
- **start_agent.vbs** - Visual Basic script (runs silently)

## Usage

### Windows (Recommended)
Double-click `start_agent.bat`

### Command Line
```bash
python scripts/start_agent.py
```

### PowerShell
```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_agent.ps1
```

All launchers automatically:
- Check for Python installation
- Verify main.py exists
- Change to correct directory
- Show clear error messages

See `../docs/guides/QUICK_START.md` for detailed documentation.

