"""
Test Kill Switch
Quick test to verify kill switch functionality
"""
import time
import logging
from kill_switch import KillSwitch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_callback():
    """Test callback for kill switch"""
    logger.info("Kill switch callback triggered!")

def main():
    print("="*80)
    print("KILL SWITCH TEST")
    print("="*80)
    print()
    print("This will test the kill switch functionality.")
    print("Press F8 to trigger the kill switch.")
    print("Press Ctrl+C to exit.")
    print()
    print("Starting test in 2 seconds...")
    time.sleep(2)
    
    kill_switch = KillSwitch(kill_callback=test_callback, hotkey='f8')
    kill_switch.start()
    
    print("\n✅ Kill switch active - Press F8 now!")
    print("Waiting for kill switch activation...\n")
    
    try:
        while not kill_switch.is_killed():
            time.sleep(0.1)
        
        print("\n" + "="*80)
        print("✅ KILL SWITCH TEST PASSED!")
        print("="*80)
        print(f"Kill switch was triggered {kill_switch.kill_count} time(s)")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        kill_switch.stop()
        print("Kill switch stopped")

if __name__ == "__main__":
    main()

