# Interception Driver Installation Guide

## Overview

The Interception driver provides kernel-level input control for Windows, allowing the Half Sword AI agent to:
- Block physical mouse input when bot is controlling
- Inject bot movements without interference
- Seamlessly switch between bot and human control

**Note**: The agent works without interception using DirectInput fallback, but interception provides better control.

## Installation Steps

### Step 1: Install Python Library

```bash
pip install interception-python
```

Or use the helper script:
```bash
python scripts/install_interception.py --install-python
```

### Step 2: Download Interception Driver

1. Visit: https://github.com/oblitum/Interception/releases
2. Download the latest `interception.zip`
3. Extract to a folder (e.g., `C:\interception\`)

### Step 3: Install Driver (Requires Admin)

**Option A: Using the helper script**
```bash
# Run as Administrator
python scripts/install_interception.py --driver-path C:\interception\command_line_installer\install-interception.exe
```

**Option B: Manual installation**
1. Open Command Prompt as Administrator:
   - Press `Win + X`
   - Select "Windows PowerShell (Admin)" or "Command Prompt (Admin)"
2. Navigate to the command line installer folder:
   ```cmd
   cd C:\interception\command_line_installer
   ```
3. Run the installer:
   ```cmd
   install-interception.exe /install
   ```
4. Restart your computer

### Step 4: Verify Installation

```bash
python scripts/install_interception.py --check-only
```

Or test in Python:
```python
from interception import Interception
interception = Interception()
devices = interception.get_devices()
print(f"Found {len(devices)} devices")
```

## Troubleshooting

### "Interception driver not available"

**Cause**: Driver not installed or Python library missing

**Solution**:
1. Check Python library: `pip list | findstr interception`
2. If missing: `pip install interception-python`
3. Check driver: Run `install-interception.exe /status` (in driver folder)
4. If not installed: Follow Step 3 above

### "Access Denied" or "Permission Error"

**Cause**: Driver installation requires Administrator privileges

**Solution**:
- Right-click Command Prompt → "Run as administrator"
- Or use the helper script with admin privileges

### "No mouse device found"

**Cause**: Driver installed but no mouse detected

**Solution**:
1. Check device manager for mouse
2. Try unplugging and replugging mouse
3. Restart computer after driver installation

### "Driver signature error" (Windows 11)

**Cause**: Secure Boot blocking unsigned driver

**Solution**:
1. Disable Secure Boot in BIOS/UEFI settings
2. Or enable "Test Mode" in Windows:
   ```cmd
   bcdedit /set testsigning on
   ```
   (Requires restart)

### Anti-Cheat Detection

**Warning**: Some games detect interception driver and may ban accounts.

**Solution**:
- Use DirectInput fallback (already implemented)
- The agent automatically falls back if interception is not available
- No need to uninstall interception - just don't use it for that game

## Uninstallation

To uninstall the interception driver:

```cmd
cd C:\interception\command_line_installer
install-interception.exe /uninstall
```

Then restart your computer.

## Current Status

The Half Sword AI agent currently uses **DirectInput fallback** which works well without interception. The interception driver provides:
- Better input isolation (no interference)
- True kernel-level control
- Seamless human override detection

But it's **optional** - the agent works fine without it!

## Alternative: DirectInput (Current Default)

The agent uses DirectInput (ctypes SendInput) by default, which:
- ✅ Works without driver installation
- ✅ No admin privileges needed
- ✅ No anti-cheat issues
- ✅ Good performance
- ⚠️ May detect bot movements as human input (keep mouse still)

This is the recommended approach unless you specifically need interception features.

## Verification

After installation, run:
```bash
python scripts/install_interception.py --check-only
```

You should see:
```
✅ Interception driver is installed and working!
   Found X devices
```

## Support

If you encounter issues:
1. Check Windows Event Viewer for driver errors
2. Verify driver is loaded: `sc query interception`
3. Check Python library: `python -c "from interception import Interception; print('OK')"`
4. Review logs in `logs/` directory

