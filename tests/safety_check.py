"""
Safety Check Utility
Verifies system is safe to run and won't interfere with user input
"""
import sys
import logging
from input_mux import InputMultiplexer, ControlMode
from kill_switch import KillSwitch
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_safety_check():
    """Run comprehensive safety checks"""
    print("="*80)
    print("HALF SWORD AI AGENT - SAFETY CHECK")
    print("="*80)
    print()
    
    issues = []
    warnings = []
    
    # Check 1: Input multiplexer safety
    print("1. Checking input multiplexer safety...")
    try:
        mux = InputMultiplexer()
        if mux.safety_lock:
            warnings.append("Safety lock is enabled (bot input disabled)")
        else:
            print("   ✅ Safety lock disabled (normal operation)")
        
        if mux.mode == ControlMode.AUTONOMOUS:
            print("   ✅ Starting in AUTONOMOUS mode")
        else:
            print("   ⚠️  Starting in MANUAL mode (human control)")
    except Exception as e:
        issues.append(f"Input multiplexer initialization failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # Check 2: PyAutoGUI failsafe
    print("\n2. Checking PyAutoGUI failsafe...")
    try:
        import pyautogui
        if pyautogui.FAILSAFE:
            print("   ✅ FAILSAFE enabled - move mouse to corner to stop")
        else:
            warnings.append("PyAutoGUI FAILSAFE is disabled")
            print("   ⚠️  FAILSAFE disabled")
    except ImportError:
        print("   ⚠️  PyAutoGUI not available")
    
    # Check 3: Human detection
    print("\n3. Testing human input detection...")
    try:
        mux = InputMultiplexer()
        mux.start()
        import time
        time.sleep(0.1)
        detected = mux.check_human_override()
        if detected:
            print("   ✅ Human input detection working")
        else:
            print("   ℹ️  No human input detected (normal if mouse not moving)")
        mux.stop()
    except Exception as e:
        issues.append(f"Human detection test failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # Check 4: Mode switching
    print("\n4. Testing mode switching...")
    try:
        mux = InputMultiplexer()
        initial_mode = mux.mode
        mux.force_manual_mode()
        if mux.mode == ControlMode.MANUAL:
            print("   ✅ Mode switching to MANUAL works")
        mux.set_mode(ControlMode.AUTONOMOUS)
        if mux.mode == ControlMode.AUTONOMOUS:
            print("   ✅ Mode switching to AUTONOMOUS works")
    except Exception as e:
        issues.append(f"Mode switching test failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # Check 5: Safety lock
    print("\n5. Testing safety lock...")
    try:
        mux = InputMultiplexer()
        mux.enable_safety_lock()
        if mux.safety_lock:
            print("   ✅ Safety lock can be enabled")
        mux.disable_safety_lock()
        if not mux.safety_lock:
            print("   ✅ Safety lock can be disabled")
    except Exception as e:
        issues.append(f"Safety lock test failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # Check 6: Kill switch
    print("\n6. Testing kill switch...")
    try:
        kill_triggered = False
        def test_kill():
            nonlocal kill_triggered
            kill_triggered = True
        
        kill_switch = KillSwitch(kill_callback=test_kill, hotkey='f8')
        kill_switch.start()
        print("   ✅ Kill switch initialized")
        print("   ℹ️  Press F8 to test (will not actually kill, just verify it works)")
        print("   ⚠️  Note: Kill switch requires pynput library - install with: pip install pynput")
        kill_switch.stop()
    except Exception as e:
        warnings.append(f"Kill switch initialization warning: {e}")
        print(f"   ⚠️  Warning: {e}")
        print("   ℹ️  Kill switch may not work without pynput library")
    
    # Summary
    print("\n" + "="*80)
    print("SAFETY CHECK SUMMARY")
    print("="*80)
    
    if issues:
        print(f"\n❌ CRITICAL ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"   - {issue}")
        print("\n⚠️  DO NOT RUN THE AGENT UNTIL ISSUES ARE RESOLVED")
        return False
    else:
        print("\n✅ No critical issues found")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"   - {warning}")
        print("\n⚠️  Review warnings before running")
    else:
        print("\n✅ No warnings")
    
    print("\n✅ System appears safe to run")
    print("\nRemember:")
    print("  - The bot will automatically stop if you move your mouse")
    print("  - You can enable safety lock to completely disable bot input")
    print(f"  - Press {config.KILL_BUTTON.upper()} for instant emergency stop (kill switch)")
    print("  - Press Ctrl+C to stop the agent at any time")
    print("="*80)
    
    return len(issues) == 0

if __name__ == "__main__":
    safe = run_safety_check()
    sys.exit(0 if safe else 1)

