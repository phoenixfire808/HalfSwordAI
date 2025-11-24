"""
Test input injection - verify bot can actually send commands
"""
import time
import logging
from input_mux import InputMultiplexer
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("="*60)
    print("INPUT INJECTION TEST")
    print("="*60)
    print("\nThis will test if the bot can inject mouse movements")
    print("Keep your mouse STILL during the test!")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    input_mux = InputMultiplexer()
    input_mux.start()
    
    # Force autonomous mode
    input_mux.force_autonomous_mode()
    
    print("[OK] Input multiplexer started")
    print(f"   Mode: {input_mux.mode.value}")
    print(f"   Safety lock: {input_mux.safety_lock}")
    print("\nTesting mouse movement injection...")
    print("You should see the mouse move slightly\n")
    
    # Test small movements
    for i in range(5):
        print(f"Test {i+1}/5: Injecting small movement...")
        input_mux.inject_action(0.1, 0.0, {})  # Small right movement
        time.sleep(0.5)
        
        stats = input_mux.get_stats()
        print(f"   Injections: {stats['bot_injections']}")
        print(f"   Mode: {stats['mode']}")
        print(f"   Safety locked: {stats['safety_locked']}")
        print()
    
    print("="*60)
    print("TEST COMPLETE")
    print("="*60)
    stats = input_mux.get_stats()
    print(f"Total injections: {stats['bot_injections']}")
    print(f"Human overrides: {stats['human_overrides']}")
    print(f"Mode switches: {stats['mode_switches']}")
    
    if stats['bot_injections'] > 0:
        print("\n[SUCCESS] Bot injected actions!")
    else:
        print("\n[FAILED] No actions were injected")
        print("   Check if:")
        print("   - Mode is AUTONOMOUS")
        print("   - Safety lock is disabled")
        print("   - Mouse is still (not moving)")
    
    input_mux.stop()

if __name__ == "__main__":
    main()

