"""
Quick check if interception driver is available
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_interception():
    """Check interception driver status"""
    print("Checking Interception Driver Status...")
    print("=" * 60)
    
    # Check Python library
    try:
        from interception import Interception
        print("✅ Python library: INSTALLED")
        python_lib_ok = True
    except ImportError:
        print("❌ Python library: NOT INSTALLED")
        print("   Install with: pip install interception-python")
        python_lib_ok = False
        return False
    
    # Check driver
    try:
        interception = Interception()
        if not interception.valid:
            print("❌ Driver: NOT INSTALLED or INVALID")
            print("   The driver may not be properly installed")
            return False
        
        devices = interception.devices
        mouse_devices = [d for d in devices if not d.is_keyboard]
        keyboard_devices = [d for d in devices if d.is_keyboard]
        
        if mouse_devices:
            print(f"✅ Driver: INSTALLED and WORKING")
            print(f"   Found {len(mouse_devices)} mouse device(s)")
            print(f"   Found {len(keyboard_devices)} keyboard device(s)")
            print(f"   Mouse available: {interception.mouse is not None}")
            print()
            print("✅ Interception driver is READY!")
            print("   The agent can use kernel-level input control")
            return True
        else:
            print("⚠️ Driver: INSTALLED but no mouse found")
            print("   Try restarting or replugging mouse")
            return False
    except Exception as e:
        print(f"❌ Driver: NOT INSTALLED or ERROR")
        print(f"   Error: {e}")
        print()
        print("To install:")
        print("  1. Download from: https://github.com/oblitum/Interception/releases")
        print("  2. Extract and run: install-interception.exe /install (as Admin)")
        print("  3. Restart computer")
        print()
        print("See INTERCEPTION_INSTALL.md for detailed instructions")
        return False

if __name__ == "__main__":
    status = check_interception()
    print()
    if not status:
        print("Note: The agent will work fine without interception")
        print("      It uses DirectInput fallback automatically")
    sys.exit(0 if status else 1)

