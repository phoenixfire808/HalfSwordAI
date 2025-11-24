"""
Inspect a Half Sword dataset
Usage: python scripts/inspect_dataset.py <dataset_path>
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from half_sword_ai.tools.dataset_utils import inspect_dataset

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_dataset.py <dataset_path>")
        print("\nExample:")
        print("  python scripts/inspect_dataset.py data/datasets/half_sword_dataset_1234567890.npz")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    inspect_dataset(dataset_path)

