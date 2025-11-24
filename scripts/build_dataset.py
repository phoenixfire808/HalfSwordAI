"""
Quick script to build Half Sword dataset
Run this while playing the game to collect training data
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from half_sword_ai.tools.dataset_builder import DatasetBuilder

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     Half Sword Dataset Builder                             ║
    ║     Collect training data by playing the game              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    print("Starting dataset collection...")
    print("  - Play Half Sword normally")
    print("  - All frames, actions, and game state will be recorded")
    print("  - Press Ctrl+C to stop and save dataset")
    print()
    
    builder = DatasetBuilder()
    builder.run_recording_loop(target_fps=60)

