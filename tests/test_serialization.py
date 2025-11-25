
import json
import numpy as np

def test_serialization():
    print(f"Numpy version: {np.__version__}")
    
    val = np.bool_(True)
    print(f"Value: {val}, Type: {type(val)}")
    print(f"Is instance of np.bool_: {isinstance(val, np.bool_)}")
    print(f"Is instance of np.generic: {isinstance(val, np.generic)}")
    print(f"Has item(): {hasattr(val, 'item')}")
    
    if hasattr(val, 'item'):
        py_val = val.item()
        print(f"item() result: {py_val}, Type: {type(py_val)}")
        print(f"Is standard bool: {isinstance(py_val, bool)}")

if __name__ == "__main__":
    test_serialization()
