
import numpy as np
import logging
from half_sword_ai.learning.human_recorder import HumanActionRecorder

# Configure logging to avoid errors
logging.basicConfig(level=logging.INFO)

def test_fix():
    print("Testing HumanActionRecorder fix...")
    recorder = HumanActionRecorder(save_path=".")
    
    val = np.bool_(True)
    print(f"Testing value: {val} (type: {type(val)})")
    
    try:
        serialized = recorder._make_json_serializable(val)
        print(f"Serialized: {serialized} (type: {type(serialized)})")
        
        if isinstance(serialized, bool) and serialized is True:
            print("SUCCESS: np.bool_ converted to bool")
        else:
            print("FAILURE: Conversion result unexpected")
            
    except Exception as e:
        print(f"FAILURE: Exception during conversion: {e}")

if __name__ == "__main__":
    test_fix()
