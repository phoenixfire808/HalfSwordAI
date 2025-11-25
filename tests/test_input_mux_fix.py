
import time
import logging
import threading
from unittest.mock import MagicMock, patch
from half_sword_ai.input.input_mux import InputMultiplexer, ControlMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_input_mux_fix():
    print("Testing InputMultiplexer noise threshold fix...")
    
    # Mock dependencies to avoid actual hardware interaction
    with patch('pyautogui.position') as mock_position, \
         patch('half_sword_ai.input.input_mux.InputMultiplexer._get_current_button_states') as mock_buttons:
        
        # Setup mocks
        mock_buttons.return_value = {'left': False, 'w': False}  # No buttons pressed
        
        # Initialize mux
        mux = InputMultiplexer()
        mux.noise_threshold = 2.0  # Set a known threshold
        
        # Simulate initial position
        mock_position.return_value = MagicMock(x=100, y=100)
        mux.get_current_human_input() # Initialize last_mouse_pos
        
        print(f"Initial state: Mode={mux.mode}, Last Input Time={mux.last_human_input_time}")
        
        # Test 1: Significant movement (should update time)
        print("\nTest 1: Significant movement (10 pixels)")
        time.sleep(0.1)
        start_time = mux.last_human_input_time
        mock_position.return_value = MagicMock(x=110, y=100) # +10 pixels
        mux.get_current_human_input()
        
        if mux.last_human_input_time > start_time:
            print("SUCCESS: Significant movement updated input time")
        else:
            print("FAILURE: Significant movement did NOT update input time")
            
        # Test 2: Jitter (should NOT update time)
        print("\nTest 2: Jitter (1 pixel)")
        time.sleep(0.1)
        start_time = mux.last_human_input_time
        mock_position.return_value = MagicMock(x=111, y=100) # +1 pixel (from 110)
        mux.get_current_human_input()
        
        if mux.last_human_input_time == start_time:
            print("SUCCESS: Jitter ignored (input time unchanged)")
        else:
            print(f"FAILURE: Jitter updated input time! (Diff: {mux.last_human_input_time - start_time})")
            
        # Test 3: Button press (should update time)
        print("\nTest 3: Button press")
        time.sleep(0.1)
        start_time = mux.last_human_input_time
        mock_buttons.return_value = {'left': True, 'w': False}
        mux.get_current_human_input()
        
        if mux.last_human_input_time > start_time:
            print("SUCCESS: Button press updated input time")
        else:
            print("FAILURE: Button press did NOT update input time")

if __name__ == "__main__":
    test_input_mux_fix()
