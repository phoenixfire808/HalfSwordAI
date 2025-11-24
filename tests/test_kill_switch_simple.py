"""
Simple kill switch test - verify it works
"""
import time
import logging
from kill_switch import KillSwitch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_callback():
    logger.critical("="*60)
    logger.critical("KILL SWITCH TRIGGERED!")
    logger.critical("="*60)

def main():
    print("\n" + "="*60)
    print("KILL SWITCH TEST")
    print("="*60)
    print("\nPress F8 to trigger the kill switch")
    print("Press Ctrl+C to exit")
    print("\nStarting in 2 seconds...\n")
    time.sleep(2)
    
    kill_switch = KillSwitch(kill_callback=test_callback, hotkey='f8')
    kill_switch.start()
    
    print("✅ Kill switch is active!")
    print(f"   Listening for: {kill_switch.hotkey.upper()}")
    print("   Press F8 now to test...\n")
    
    try:
        start_time = time.time()
        while not kill_switch.is_killed() and (time.time() - start_time) < 30:
            time.sleep(0.1)
            if int(time.time() - start_time) % 5 == 0 and int(time.time() - start_time) > 0:
                print(f"   Still waiting... ({int(time.time() - start_time)}s)")
        
        if kill_switch.is_killed():
            print("\n" + "="*60)
            print("✅ SUCCESS! Kill switch was triggered!")
            print(f"   Kill count: {kill_switch.kill_count}")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("⚠️  Timeout - Kill switch was not triggered")
            print("   Make sure you pressed F8")
            print("="*60)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        kill_switch.stop()
        print("\nKill switch stopped")

if __name__ == "__main__":
    main()

